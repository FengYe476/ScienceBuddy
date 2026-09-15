"""Alternate harness and RL stages and record experiment state."""

import json
import os
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

from simple_scibuddy.artifacts import ROOT, digest, file_digest, tree_identity, write_json
from simple_scibuddy.coevolve.continuation import completed_boundary
from simple_scibuddy.coevolve.phase import harness_evolve
from simple_scibuddy.coevolve.program import validate_program
from simple_scibuddy.coevolve.protocol import atomic_json, publish_request, validate_result
from simple_scibuddy.coevolve.worker import run_round
from simple_scibuddy.data.dataset import TaskDataset
from simple_scibuddy.training.tracking import MetricRelay, publish_config


def launch_model(cfg, mode, relay=None):
    """Isolate Ray lifetimes and CUDA state between training and export evaluation."""
    path = Path(cfg["planned_run_dir"])
    settings_path = path.parent / f"{path.name}-input.json"
    write_json(settings_path, cfg)
    command = [
        sys.executable,
        "-c",
        "import json,sys; from simple_scibuddy.training.launch import launch; "
        "launch(json.load(open(sys.argv[1])), sys.argv[2])",
        str(settings_path),
        mode,
    ]
    with (path.parent / f"{path.name}.log").open("w") as log:
        with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as process:
            while True:
                try:
                    status = process.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    if relay:
                        relay.drain(path)
            if relay:
                relay.drain(path)
            if status:
                raise subprocess.CalledProcessError(status, command)
    return path


def run(config, cfg):
    import wandb

    root = Path(config["experiment"])
    if not root.resolve().is_relative_to(ROOT / "runs"):
        raise ValueError("Experiment must be inside runs/")
    root.mkdir(parents=True, exist_ok=False)
    hcfg = {
        **cfg,
        "workers": 8,
        "user_turns": 1,
        **{k: v for k, v in config["harness_evolve"].items() if k != "improver"},
    }
    dataset = TaskDataset(cfg["dataset"])
    counts = dataset.manifest["split_counts"]
    for split, expected in counts.items():
        rows = dataset.tasks(split)
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} {split} tasks")
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(dict(r, task_dir=os.path.relpath(dataset.root / r["id"], root))) + "\n" for r in rows)
        )
    for name, sha in dataset.manifest.get("payload_hashes", {}).items():
        if file_digest(dataset.root / name) != sha:
            raise ValueError(f"Frozen payload changed: {name}")
    atomic_json(root / "config.json", config, immutable=True)
    identities = {
        "code": tree_identity(ROOT / "src/simple_scibuddy", "*.py"),
        "model": tree_identity(cfg["model"]),
        "improver": config["harness_evolve"]["improver"],
        "dataset": digest(dataset.manifest),
        "runtime": cfg["runtime_image"],
        "verifier": file_digest(ROOT / "src/simple_scibuddy/data/verifier.py"),
    }
    boundary = (
        completed_boundary(config["continue_from"], config, identities)
        if config.get("continue_from")
        else None
    )
    if boundary:
        identities["starting_model"] = tree_identity(boundary["model"])
        atomic_json(root / "continuation.json", boundary, immutable=True)
    atomic_json(root / "identities.json", identities, immutable=True)
    harness = root / "h0.py"
    shutil.copyfile(boundary["harness"] if boundary else cfg["harness"], harness)
    validate_program(harness)
    train = sorted(dataset.tasks("train"), key=lambda r: digest([hcfg["seed"], 0, r["id"]]))
    validation_rows = dataset.tasks("val")
    cfg["validation_task_ids"] = [r["id"] for r in validation_rows]
    write_json(
        root / "validation-split.json",
        {
            "task_ids": cfg["validation_task_ids"],
            "source": "explicit val",
            "excluded_from": ["harness interactions", "RL training"],
            "seed": hcfg["seed"],
        },
    )
    model, round_number = (boundary["model"], boundary["rounds"]) if boundary else (cfg["model"], 0)
    os.environ["WANDB_RUN_GROUP"] = root.name
    tracker = wandb.init(
        project="simple_scibuddy-model-evolution",
        name=root.name,
        group=root.name,
        job_type=config["mode"],
        dir=str(root),
        config=config,
    )
    publish_config(tracker, root, cfg.get("launch_config_toml"))
    cfg["single_wandb_run"] = True
    relay = MetricRelay(tracker)
    cursor = boundary["train_cursor"] if boundary else 0
    history = boundary["history"] if boundary else []
    if boundary:
        write_json(root / "harness_evolve-history.json", history)
    co = config["coevolve"]
    max_rounds = co.get("max_rounds", 1) if config["mode"] == "coevolve" else 0
    stop_reason = "round_budget"
    completed = False
    phase_number = round_number
    try:
        for phase_number in range(round_number + 1, max(1, max_rounds) + 1):
            phase_id = f"h{phase_number:04d}"
            if (
                boundary
                and boundary.get("boundary") == "completed_harness"
                and phase_number == boundary["rounds"] + 1
            ):
                phase_summary = dict(
                    boundary["phase_summary"],
                    selected_harness=str(harness),
                    continued_from=boundary["source"],
                    original_summary_sha256=boundary["phase_summary_sha256"],
                )
                write_json(root / "harness_evolve" / phase_id / "summary.json", phase_summary)
                tracker.log(
                    {
                        "continuation/reused_harness_phase": phase_id,
                        "continuation/restarted_rl_round": phase_number,
                    }
                )
                print(
                    f"Reusing completed {phase_id} from {boundary['source']}; restarting RL from its base model",
                    flush=True,
                )
            else:
                harness, cursor, phase_summary = harness_evolve(
                    root,
                    phase_id,
                    model,
                    harness,
                    train,
                    cursor,
                    dataset,
                    hcfg,
                    config["harness_evolve"],
                    tracker,
                    round_number,
                    history,
                )
            # Each configured cycle consists of harness learning followed by RL.
            if round_number == max_rounds:
                break
            if not (phase_summary["improved"] or co.get("trigger") == "always_debug"):
                stop_reason = "no_harness_improvement"
                break
            round_number += 1
            contract = {
                "thinking_enabled": False,
                "execution_network": "disabled",
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "entrypoint": "run(task, api)",
                "max_model_calls": hcfg["actions"],
                "max_tool_calls": hcfg["actions"],
                "max_output_tokens": hcfg["max_tokens"],
                "context_tokens": hcfg["context_tokens"],
                "rollout_timeout_seconds": hcfg["seconds"],
                "tool_response_tokens": hcfg["tool_response_tokens"],
                "tool_history_tokens": hcfg["tool_history_tokens"],
                "train_index": str(root / "train.jsonl"),
                "eval_index": str(root / "test.jsonl"),
            }
            directory = publish_request(
                root / "shared",
                f"r{round_number:04d}",
                harness,
                base_model_id=f"M{round_number - 1}",
                base_model_path=model,
                harness_id=phase_id,
                evaluation={
                    "trigger": co.get("trigger", "improvement"),
                    "metrics": phase_summary["selected"],
                    "harness_phase": phase_id,
                },
                reference_correct=phase_summary["initial"]["correct"],
                contract=contract,
                source_run=root,
            )
            write_json(
                root / "status.json", {"status": "rl", "round": round_number, "harness_phase": phase_id}
            )
            print(f"Starting RL round {round_number}, {config['model'].get('steps', 20)} updates", flush=True)
            run_round(
                dict(cfg),
                directory,
                co.get("worker_id", "rl-01"),
                config["model"].get("steps", 20),
                launcher=partial(launch_model, relay=relay),
            )
            result = validate_result(directory)
            model = result["model_path"]
            atomic_json(directory / "consumed.json", {"model": model})
        write_json(
            root / "status.json",
            {
                "status": "completed",
                "rounds": round_number,
                "model": model,
                "harness": str(harness),
                "harness_phases": phase_number,
                "train_cursor": cursor,
                "stop_reason": stop_reason,
            },
        )
        completed = True
    except BaseException:
        write_json(root / "status.json", {"status": "failed", "rounds": round_number})
        raise
    finally:
        tracker.finish(exit_code=0 if completed else 1)
