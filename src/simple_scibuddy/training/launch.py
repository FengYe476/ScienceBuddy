"""Prepare a model stage and launch its SkyRL training or evaluation entrypoint."""

import json
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

from simple_scibuddy.artifacts import ROOT, digest, file_digest, tree_identity, write_json


def build_overrides(cfg, mode, run, training_data, data):
    """Translate effective settings without importing SkyRL or touching devices."""
    overrides = json.loads((ROOT / "configs/skyrl.json").read_text())
    # skyrl.json 按 8 卡写死；按实际可见卡数覆盖，让同一份配置能在 4 卡节点上跑。
    # colocate_all=true，所以策略、参考模型和推理引擎共享这批卡；GRPO 无 critic，
    # 但 SkyRL 仍读取该字段，一并对齐以免放置阶段报不一致。
    if cfg.get("devices"):
        overrides.update(
            {
                "trainer.placement.policy_num_gpus_per_node": cfg["devices"],
                "trainer.placement.ref_num_gpus_per_node": cfg["devices"],
                "trainer.placement.critic_num_gpus_per_node": cfg["devices"],
                "generator.inference_engine.num_engines": cfg["devices"],
            }
        )
        if cfg["devices"] < 8:
            # 卡数减半，每张卡的 FSDP 分片与优化器状态翻倍。上游 0.65 给 vLLM
            # 预留 80 GB 中的 52 GB，而 max_num_seqs=1 只需要一条 24K 序列的
            # KV cache（4B 模型约 3.6 GB），预留量本就过剩 14 倍。降到 0.45
            # 仍有 10 倍余量，腾出的约 16 GB 留给训练状态，避免 colocate_all
            # 下的 OOM。不影响生成确定性：温度 0、单序列，与缓存大小无关。
            overrides["generator.inference_engine.gpu_memory_utilization"] = 0.45
    overrides.update(
        {
            "trainer.policy.model.path": cfg["model"],
            "trainer.ref.model.path": cfg["model"],
            "trainer.project_name": "simple_scibuddy-model-evolution",
            "trainer.run_name": run.name,
            "trainer.log_path": str(run / "infra"),
            "trainer.ckpt_path": str(run / "checkpoints"),
            "trainer.export_path": str(run / "exports"),
            "trainer.hf_save_interval": 0,
            "trainer.train_batch_size": 8,
            "trainer.policy_mini_batch_size": 8,
            "trainer.eval_batch_size": 8,
            "trainer.epochs": 100,
            "trainer.max_training_steps": cfg.get("steps", {"smoke": 2, "overfit": 20}.get(mode, 50)),
            "trainer.ckpt_interval": 0,
            "trainer.max_prompt_length": cfg["context_tokens"] - cfg["max_tokens"],
            "generator.max_input_length": cfg["context_tokens"] - cfg["max_tokens"],
            "generator.sampling_params.max_generate_length": cfg["max_tokens"],
            "generator.inference_engine.engine_init_kwargs.max_model_len": cfg["context_tokens"],
            "generator.inference_engine.max_num_batched_tokens": cfg["context_tokens"],
            "generator.inference_engine.max_num_seqs": 1,  # Match repeatable Qwen3.5 harness inference.
            "generator.step_wise_trajectories": True,
            "generator.n_samples_per_prompt": cfg["samples_per_prompt"],
            "generator.max_turns": cfg["actions"],
            "trainer.eval_before_train": not cfg.get("no_eval", False),
            "trainer.eval_interval": 0
            if cfg.get("no_eval")
            else cfg.get("eval_interval", cfg["checkpoint_interval"]),
            "environment.env_class": "sciencebuddy",
            "data.train_data": [
                str(training_data / (f"{mode}.parquet" if mode in ("smoke", "overfit") else "train.parquet"))
            ],
            "data.val_data": [str(data / "test.parquet")],
        }
    )
    if cfg.get("coevolve"):
        identity = cfg["coevolve"]
        overrides["trainer.run_name"] = f"{identity['round_id']}-{identity['base_model_id']}-{mode}"
    if cfg.get("eval_samples") is not None:
        overrides["generator.eval_n_samples_per_prompt"] = cfg["eval_samples"]
    overrides["generator.eval_sampling_params.temperature"] = 0.0
    overrides["generator.eval_sampling_params.top_p"] = 1.0
    if cfg.get("weight_audit"):
        overrides["generator.inference_engine.engine_init_kwargs.worker_extension_cls"] = (
            "simple_scibuddy.inference.weight_audit.AuditedInferenceWorker"
        )
    return overrides


def launch(cfg, mode, resume=None, validate_only=False):
    if cfg.get("eval_temperature", 0.0) != 0.0:
        raise ValueError("Evaluation uses temperature 0")
    import psutil
    from skyrl.train.config.config import overrides_dict_to_dotlist

    # 上游硬性要求恰好 8 卡。Anvil 全集群每节点只有 4 张 GPU（ai/gpu/gpu-debug
    # 分区的 Gres 都是 gpu:4），该约束在那里永远无法满足。放宽为 2 的幂，并把
    # 实际卡数传给 build_overrides 覆盖 skyrl.json 里写死的 8。
    # 算法不受影响：有效 batch 由 trainer.train_batch_size / policy_mini_batch_size
    # 决定（见 build_overrides），与卡数无关；卡少只是梯度累积步数翻倍、吞吐下降。
    selected = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    devices = len(set(selected))
    if devices != len(selected) or devices not in (1, 2, 4, 8):
        raise RuntimeError(f"Select 1, 2, 4 or 8 distinct GPUs; got {selected!r}")
    cfg = dict(cfg, devices=devices)
    if resume:
        manifest = Path(resume) / "latest-checkpoint.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("resumable") is False:
            raise ValueError(
                "This run saved model weights only. Start a new training stage with its exported model; optimizer resume is unavailable."
            )
        previous = json.loads((Path(resume) / "settings.json").read_text())
        mode = previous["mode"]
    training_data = Path(cfg.get("data_dir", ROOT / "data/sciencebuddy"))
    data = training_data
    if not (data / "manifest.json").exists():
        raise RuntimeError("Run preflight first")
    if cfg.get("planned_run_dir"):
        run = Path(cfg["planned_run_dir"]).resolve()
        if not run.is_relative_to(ROOT / "runs"):
            raise ValueError("Planned run directory must be inside this repository runs/")
        run.mkdir(parents=True, exist_ok=False)
    else:
        run = Path(tempfile.mkdtemp(prefix=f"sciencebuddy-{mode}-", dir=ROOT / "runs"))
    cfg["run_dir"] = str(run)
    if cfg.get("eval_index"):
        from simple_scibuddy.evaluation.dataset import prepare_index

        data = prepare_index(cfg, run / "evaluation-data")
    shutil.copyfile(cfg["harness"], run / "harness.py")
    cfg["harness"] = str(run / "harness.py")
    cfg["mode"] = mode
    cfg["harness_hash"] = digest(Path(cfg["harness"]).read_bytes())
    cfg["model_identity"] = tree_identity(cfg["model"])
    cfg["verifier_identity"] = file_digest(ROOT / "src/simple_scibuddy/data/verifier.py")
    cfg["runtime_identity"] = digest((Path(cfg["dataset"]) / "environment.lock.json").read_bytes())
    cfg["dataset_identity"] = digest((data / "manifest.json").read_bytes())
    cfg["weight_audit"] = os.environ.get("SKYRL_SIMPLE_SCIBUDDY_WEIGHT_AUDIT") == "1"
    shutil.copyfile(data / "manifest.json", run / "split.json")
    cfg["split_manifest"] = str(run / "split.json")
    write_json(run / "settings.json", cfg)
    os.environ["SKYRL_SIMPLE_SCIBUDDY_SETTINGS"] = str(run / "settings.json")
    os.environ["RAY_ADDRESS"] = "local"
    os.environ["RAY_TMPDIR"] = tempfile.mkdtemp(prefix="sb-ray-", dir="/tmp")
    addresses = [
        a.address
        for aa in psutil.net_if_addrs().values()
        for a in aa
        if a.family in (socket.AF_INET, socket.AF_INET6)
    ]
    bypass = ",".join(["localhost", "127.0.0.1", "::1"] + addresses + [os.environ.get("NO_PROXY", "")])
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = bypass
    os.environ["WANDB_DIR"] = str(run)
    overrides = build_overrides(cfg, mode, run, training_data, data)
    if resume:
        previous = json.loads((Path(resume) / "settings.json").read_text())
        for key in (
            "harness_hash",
            "observation_chars",
            "tool_response_tokens",
            "tool_history_tokens",
            "verifier_identity",
            "runtime_identity",
            "model_identity",
            "seconds",
            "tool_seconds",
            "dataset_identity",
            "model",
            "samples_per_prompt",
            "actions",
            "max_tokens",
            "context_tokens",
            "runtime_image",
            "baked_lake",
        ):
            if cfg.get(key) != previous.get(key):
                raise RuntimeError(f"Resume identity changed: {key}")
        overrides["trainer.ckpt_path"] = str(Path(resume) / "checkpoints")
        overrides["trainer.resume_mode"] = "latest"
    args = overrides_dict_to_dotlist(overrides)
    if mode == "evaluate":
        overrides["data.val_data"] = [str(data / "test.parquet")]
        args = overrides_dict_to_dotlist(overrides)
    write_json(run / "overrides.json", overrides)
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.utils.utils import validate_cfg

    validate_cfg(SkyRLTrainConfig.from_cli_overrides(args))
    print(f"Run directory: {run}", flush=True)
    if validate_only:
        print("CONFIG VALIDATION PASS", flush=True)
        return
    sys.argv = ["simple_scibuddy.training.entrypoint", *args]
    if mode == "evaluate":
        overrides["data.val_data"] = [str(data / "test.parquet")]
        sys.argv = ["simple_scibuddy.evaluation.entrypoint", *overrides_dict_to_dotlist(overrides)]
        from simple_scibuddy.evaluation.entrypoint import main
    else:
        from simple_scibuddy.training.entrypoint import main
    try:
        main()
    except BaseException:
        write_json(run / "status.json", {"status": "failed"})
        raise
    write_json(run / "status.json", {"status": "completed"})
    return run
