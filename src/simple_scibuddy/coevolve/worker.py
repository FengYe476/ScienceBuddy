"""One durable RL handoff round, using the existing training/evaluation launcher."""

import copy
import json
import shutil
import uuid
from pathlib import Path

from simple_scibuddy.artifacts import ROOT, file_digest, tree_identity
from simple_scibuddy.coevolve import protocol
from simple_scibuddy.coevolve.program import validate_program
from simple_scibuddy.data.validation import validate_payloads

TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt",
                   "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")


def plan_round(cfg, directory, steps=50):
    directory = Path(directory).resolve()
    if (directory / "cancelled.json").exists():
        raise ValueError("Round has been cancelled")
    request = protocol.read_request(directory)
    contract = request["runtime_contract"]
    expected = {"thinking_enabled": False, "execution_network": "disabled",
                "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                "entrypoint": "run(task, api)"}
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Unsupported runtime contract: {key}={contract.get(key)!r}")
    if contract["max_model_calls"] != contract["max_tool_calls"]:
        raise ValueError("This broker currently requires equal model/tool call budgets")
    if steps < 1:
        raise ValueError("RL steps must be positive")
    result = copy.deepcopy(cfg)
    for key in ("eval_index", "expected_count", "planned_run_dir", "data_dir", "run_dir"):
        result.pop(key, None)
    result.update(model=str(Path(request["base_model_path"]).resolve()),
                  harness=str(directory / request["harness_file"]), steps=steps,
                  no_eval=False, no_checkpoint=False, eval_samples=1, eval_temperature=0.0)
    mapping = {"context_tokens": "context_tokens", "max_output_tokens": "max_tokens",
               "max_model_calls": "actions", "rollout_timeout_seconds": "seconds",
               "tool_response_tokens": "tool_response_tokens", "tool_history_tokens": "tool_history_tokens"}
    for source, destination in mapping.items():
        value = contract[source]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Invalid budget: {source}")
        result[destination] = value
    if result["max_tokens"] >= result["context_tokens"] or result["tool_response_tokens"] < 256:
        raise ValueError("Invalid context or observation budget")
    if not (Path(result["model"]) / "config.json").is_file():
        raise ValueError("Requested base model is not available")
    validate_program(result["harness"])
    result["harness_program_sha256"] = file_digest(Path(result["harness"]))
    frozen = Path(result["dataset"])
    manifest = json.loads((frozen / "manifest.json").read_text())
    index_hashes = {}
    counts = manifest['split_counts']
    for split, key in (("train", "train_index"), ("test", "eval_index")):
        count = counts[split]
        index = Path(contract[key]).resolve()
        rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
        ids = [row["id"] for row in rows]
        expected_ids = {r["id"] for r in manifest["tasks"] if r["split"] == split}
        if len(ids) != count or len(set(ids)) != count or set(ids) != expected_ids:
            raise ValueError(f"Request {split} index differs from frozen {count}-task split")
        if any(r["split"] != split for r in rows):
            raise ValueError(f"Incorrect {split} assignments")
        validate_payloads(rows, index, frozen)
        index_hashes[key] = file_digest(index)
    for name, sha in manifest.get("payload_hashes", {}).items():
        if file_digest(frozen / name) != sha:
            raise ValueError(f"Frozen payload changed: {name}")
    result["coevolve"] = {"round_id": request["round_id"], "base_model_id": request["base_model_id"],
                          "harness_id": request["harness_id"],
                          "request_sha256": file_digest(directory / "request.json"),
                          "protocol_sha256": file_digest(Path(protocol.__file__)),
                          "index_hashes": index_hashes,
                          "reward_contract": "Fresh on-policy attempts, first-answer verifier reward; no user replies"}
    return request, result


def prepare_data(cfg, directory):
    from transformers import AutoTokenizer

    from simple_scibuddy.training.data import prepare

    target = directory / "data"
    target.mkdir(exist_ok=False)
    prepare(cfg, target, AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True))
    cfg["data_dir"] = str(target)


def preserve_tokenizer(base, exported):
    base, exported = Path(base), Path(exported)
    a = json.loads((base / "config.json").read_text())
    b = json.loads((exported / "config.json").read_text())
    # SkyRL trains/exports Qwen3.5's language model without its vision wrapper.
    text_export = (a.get("model_type") == "qwen3_5"
                   and a.get("architectures") == ["Qwen3_5ForConditionalGeneration"]
                   and a.get("text_config", {}).get("model_type") == "qwen3_5_text"
                   and b.get("model_type") == "qwen3_5_text"
                   and b.get("architectures") == ["Qwen3_5ForCausalLM"])
    for key in ("model_type", "architectures"):
        if not text_export and a.get(key) != b.get(key):
            raise ValueError(f"Export changes architecture: {key}")
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "vocab_size", "intermediate_size"):
        if a.get("text_config", a).get(key) != b.get("text_config", b).get(key):
            raise ValueError(f"Export changes model structure: {key}")
    for name in TOKENIZER_FILES:
        if (base / name).exists():
            shutil.copyfile(base / name, exported / name)
        elif (exported / name).exists():
            raise ValueError(f"Export introduces incompatible tokenizer file: {name}")


def run_round(cfg, directory, worker_id, steps=50, validate_only=False,
              *, launcher=None, preparer=prepare_data):
    directory = Path(directory).resolve()
    request, configured = plan_round(cfg, directory, steps)
    if validate_only:
        print(json.dumps({"status": "validated", "request": request,
                          "rl_settings": configured}, indent=2))
        return
    # The other project must configure its shared handoff directory here. Never
    # mutate the external ScienceBuddy-RSI tree, even for acknowledgments.
    if not directory.is_relative_to(ROOT / "runs"):
        raise ValueError("Executable handoff rounds must live under this repository's runs/ directory")
    if not worker_id:
        raise ValueError("A stable worker ID is required")
    if (directory / "result.json").exists():
        print(json.dumps(protocol.validate_result(directory), indent=2))
        return
    if (directory / "consumed.json").exists():
        raise ValueError("Consumed round cannot start training again")
    accepted = directory / "accepted.json"
    if accepted.exists() and json.loads(accepted.read_text())["worker_id"] != worker_id:
        raise ValueError("Another worker owns this round")
    job = directory / "rl-job.json"
    if job.exists():
        raise RuntimeError("This round already has an RL job. Inspect/recover it; do not launch duplicate training.")
    protocol.claim(directory, worker_id)
    configured["base_model_identity"] = tree_identity(configured["model"])
    configured["planned_run_dir"] = str(directory / "rl-train")
    state = {"job_id": uuid.uuid4().hex, "status": "preparing", "worker_id": worker_id, "request_sha256": file_digest(directory / "request.json"),
             "training_run": configured["planned_run_dir"], "steps": steps}
    protocol.atomic_json(job, state, immutable=True)
    if launcher is None:
        from simple_scibuddy.training.launch import launch

        launcher = launch
    try:
        preparer(configured, directory)
        state["status"] = "training"
        protocol.atomic_json(job, state)
        run = Path(launcher(configured, "train"))
        if json.loads((run / "status.json").read_text())["status"] != "completed":
            raise RuntimeError("Training did not complete")
        checkpoint = json.loads((run / "latest-checkpoint.json").read_text())
        if checkpoint["step"] != steps:
            raise RuntimeError("Final checkpoint has not been saved")
        exported = Path(checkpoint["export"]).resolve()
        if not exported.is_relative_to(run.resolve()):
            raise ValueError("Checkpoint must be owned by this training run")
        preserve_tokenizer(configured["model"], exported)
        # Actual inference from the exported checkpoint, not the training-side
        # in-memory policy, is the producer's load check required by protocol v1.
        state.update(status="checking_export", export=str(exported))
        protocol.atomic_json(job, state)
        evaluation = copy.deepcopy(configured)
        evaluation.update(model=str(exported), planned_run_dir=str(directory / "rl-export-eval"))
        check = Path(launcher(evaluation, "evaluate"))
        if json.loads((check / "status.json").read_text())["status"] != "completed":
            raise RuntimeError("Export load evaluation failed")
        episodes = [json.loads(p.read_text()) for p in (check / "episodes").glob("*/episode.json")]
        expected = set(json.loads((Path(configured["data_dir"]) / "manifest.json").read_text())["task_ids"]["test"])
        if len(episodes) != len(expected) or {e["task_id"] for e in episodes} != expected:
            raise RuntimeError(f"Export evaluation did not cover all {len(expected)} tasks")
        if any(not e["calls"] for e in episodes) or all(
            len(set(c["text"].strip())) <= 1 for e in episodes for c in e["calls"]
        ):
            raise RuntimeError("Export generated empty or degenerate outputs")
        if tree_identity(configured["model"]) != configured["base_model_identity"]:
            raise RuntimeError("Base checkpoint changed during this round")
        if file_digest(directory / "request.json") != state["request_sha256"]:
            raise RuntimeError("Request changed during this round")
        result = protocol.publish_result(directory, worker_id, f"{request['base_model_id']}-{request['round_id']}",
                                         str(exported), load_checked=True)
        state.update(status="completed", result=result)
        protocol.atomic_json(job, state)
        print(json.dumps(result, indent=2))
    except BaseException as error:
        state.update(status="failed", error=str(error))
        protocol.atomic_json(job, state)
        if not (directory / "result.json").exists():
            protocol.publish_result(directory, worker_id, failed=str(error) or type(error).__name__)
        raise
