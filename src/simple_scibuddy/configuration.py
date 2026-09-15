"""Load runtime defaults, local overrides and experiment TOML without launching services."""

import json
import os
import tomllib
from pathlib import Path

from simple_scibuddy.artifacts import ROOT
from simple_scibuddy.coevolve import improver
from simple_scibuddy.paths import local_path, local_tree


def settings():
    cfg = json.loads(local_path("configs/defaults.json", base=ROOT).read_text())
    local = local_path("configs/local.json", base=ROOT)
    if local.exists():
        cfg.update(json.loads(local.read_text()))
    if os.environ.get("SIMPLE_SCIBUDDY_CONFIG_TOML"):
        cfg["launch_config_toml"] = os.environ["SIMPLE_SCIBUDDY_CONFIG_TOML"]
    for key in {
        "harness",
        "dataset",
        "model",
        "eval_index",
        "data_dir",
        "run_dir",
        "planned_run_dir",
        "split_manifest",
    } & cfg.keys():
        cfg[key] = str(local_path(cfg[key], base=ROOT, field=key))
        local_tree(cfg[key], root=ROOT)
    cfg["samples_per_prompt"] = int(os.environ.get("N_SAMPLES_PER_PROMPT", cfg["samples_per_prompt"]))
    if cfg["samples_per_prompt"] not in (8, 16):
        raise ValueError("N_SAMPLES_PER_PROMPT must be 8 or 16")
    return cfg


MODEL_KEYS = {
    "mode",
    "model",
    "harness",
    "run",
    "eval_index",
    "eval_samples",
    "eval_temperature",
    "expected_count",
    "concurrency",
    "context_tokens",
    "max_tokens",
    "tool_seconds",
    "steps",
    "eval_interval",
    "checkpoint_interval",
    "no_eval",
    "no_checkpoint",
}
PATH_KEYS = {"model", "harness", "run", "eval_index"}


def load_config(path, *, root=None):
    root = Path(root if root is not None else ROOT).resolve()
    path = local_path(path, base=root, relative=False, field="configuration file")
    config = tomllib.loads(path.read_text())
    unknown = set(config) - {
        "mode",
        "experiment",
        "resume",
        "continue_from",
        "model",
        "data",
        "harness_evolve",
        "coevolve",
    }
    if unknown:
        raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
    if config.get("mode") not in {"model", "harness_evolve", "coevolve"}:
        raise ValueError("mode must be model, harness_evolve, or coevolve")
    if "data" in config:
        data = config["data"]
        if (
            not isinstance(data, dict)
            or set(data) != {"dataset"}
            or not isinstance(data["dataset"], str)
            or not data["dataset"].strip()
        ):
            raise ValueError("[data] requires only a dataset path")
        data["dataset"] = str(local_path(data["dataset"], base=path.parent, root=root, field="data.dataset"))
    harness = config.get("harness_evolve", {})
    if not isinstance(harness, dict):
        raise ValueError("harness must be a TOML table")
    if "improver" in harness:
        improver.validate(
            dict(harness["improver"], base_url=harness["improver"].get("base_url", "http://localhost/v1"))
        )
        if any(key.startswith("improver_") for key in harness):
            raise ValueError("Use [harness_evolve.improver] without legacy harness_evolve.improver_* fields")
    coevolve = config.setdefault("coevolve", {})
    if not isinstance(coevolve, dict) or set(coevolve) - {"worker_id", "max_rounds", "trigger"}:
        raise ValueError("[coevolve] accepts worker_id, max_rounds and trigger")
    if (
        not isinstance(coevolve.get("worker_id", "rl-01"), str)
        or not coevolve.get("worker_id", "rl-01").strip()
    ):
        raise ValueError("coevolve.worker_id must be a nonempty string")
    if type(coevolve.get("max_rounds", 1)) is not int or coevolve.get("max_rounds", 1) < 0:
        raise ValueError("coevolve.max_rounds must be nonnegative")
    if coevolve.get("trigger", "improvement") not in {"improvement", "always_debug"}:
        raise ValueError("coevolve.trigger must be improvement or always_debug")
    model = config.setdefault("model", {})
    if not isinstance(model, dict) or set(model) - MODEL_KEYS:
        raise ValueError("Unknown model settings; see the README configuration example")
    if model.get("mode", "train") not in {"train", "smoke", "overfit", "evaluate", "resume"}:
        raise ValueError("model.mode must be train, smoke, overfit, evaluate, or resume")
    if config["mode"] == "coevolve" and model.get("mode", "train") != "train":
        raise ValueError("Co-evolution requires model.mode = train")
    if config["mode"] == "coevolve" and set(model) - {
        "mode",
        "steps",
        "model",
        "concurrency",
        "tool_seconds",
        "eval_interval",
        "checkpoint_interval",
    }:
        raise ValueError("Co-evolution episode/evaluation settings come from the handoff contract")
    if type(config.get("resume", False)) is not bool:
        raise ValueError("resume must be boolean")
    if model.get("mode") == "resume" and not model.get("run"):
        raise ValueError("Model resume requires model.run")
    if "eval_index" in model and model.get("mode") != "evaluate":
        raise ValueError("model.eval_index requires model.mode = evaluate")
    for key in (
        "steps",
        "eval_interval",
        "concurrency",
        "context_tokens",
        "max_tokens",
        "tool_seconds",
        "eval_samples",
        "expected_count",
    ):
        if key in model and (type(model[key]) is not int or model[key] < 1):
            raise ValueError(f"model.{key} must be a positive integer")
    if "checkpoint_interval" in model and (
        type(model["checkpoint_interval"]) is not int or model["checkpoint_interval"] < 0
    ):
        raise ValueError("model.checkpoint_interval must be nonnegative (0 means final only)")
    for key in ("no_eval", "no_checkpoint"):
        if key in model and type(model[key]) is not bool:
            raise ValueError(f"model.{key} must be boolean")
    if model.get("eval_temperature", 0.0) != 0.0:
        raise ValueError("Evaluation uses temperature 0")
    for key in PATH_KEYS & model.keys():
        model[key] = str(local_path(model[key], base=path.parent, root=root, field="model." + key))
    if config["mode"] != "model":
        if not config.get("experiment") or not isinstance(config.get("harness_evolve"), dict):
            raise ValueError("Harness learning requires experiment and [harness_evolve] settings")
    if "experiment" in config:
        config["experiment"] = str(
            local_path(config["experiment"], base=path.parent, root=root, field="experiment")
        )
    if config["mode"] != "model":
        if config.get("resume"):
            raise ValueError("Harness resume is not implemented; use a fresh experiment")
        if "max_steps" in harness:
            raise ValueError(
                "Replace harness_evolve.max_steps with harness_evolve.steps_per_phase (steps within each stage)"
            )
        harness.setdefault("steps_per_phase", 5)
        if "preflight_task_id" in harness and (
            not isinstance(harness["preflight_task_id"], str) or not harness["preflight_task_id"].strip()
        ):
            raise ValueError("harness_evolve.preflight_task_id must be a nonempty training task ID")
        allowed = {
            "workers",
            "feedback_every",
            "candidates",
            "selection_tasks",
            "preflight_task_id",
            "steps_per_phase",
            "user_turns",
            "actions",
            "max_tokens",
            "context_tokens",
            "seed",
            "improver",
            "tool_response_tokens",
            "tool_history_tokens",
        }
        if set(harness) - allowed or "improver" not in harness:
            raise ValueError("Unknown harness settings or missing [harness_evolve.improver]")
        for key in allowed - {"improver", "preflight_task_id"}:
            if key in harness and (
                type(harness[key]) is not int or harness[key] < (0 if key in {"user_turns"} else 1)
            ):
                raise ValueError(f"Invalid harness_evolve.{key}")
        if harness.get("feedback_every", 8) > 715:
            raise ValueError("harness_evolve.feedback_every cannot exceed the 715-task training set")
        if not harness["improver"].get("base_url"):
            if harness["improver"]["backend"] != "vllm":
                raise ValueError("Remote improver requires base_url")
            harness["improver"]["model"] = str(
                local_path(harness["improver"]["model"], base=path.parent, root=root, field="improver.model")
            )
    if "continue_from" in config:
        if config["mode"] != "coevolve" or not isinstance(config["continue_from"], str):
            raise ValueError("continue_from requires a coevolution run directory")
        config["continue_from"] = str(
            local_path(config["continue_from"], base=path.parent, root=root, field="continue_from")
        )
    for key in ("model", "harness"):
        if key in model:
            local_tree(model[key], root=root)
    if harness.get("improver") and not harness["improver"].get("base_url"):
        local_tree(harness["improver"]["model"], root=root)
    return config
