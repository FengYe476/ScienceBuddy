"""Command-line arguments for model stages and runtime preparation."""

import argparse
from pathlib import Path

from simple_scibuddy.artifacts import ROOT
from simple_scibuddy.configuration import settings
from simple_scibuddy.environments.preflight import preflight
from simple_scibuddy.paths import local_path, local_tree
from simple_scibuddy.training.launch import launch


def main(argv=None, *, planned_run_dir=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "smoke", "overfit", "train", "resume", "evaluate"])
    parser.add_argument("--run")
    parser.add_argument("--dataset", type=Path, help="Optional frozen dataset; defaults remain unchanged")
    parser.add_argument("--harness", type=Path, help="Frozen harness for this run")
    parser.add_argument("--eval-index", type=Path, help="Explicit evaluation-only JSONL task index")
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--eval-temperature", type=float)
    parser.add_argument("--expected-count", type=int, help="Required evaluation task count")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--model", help="Model or exported HF checkpoint for evaluation")
    parser.add_argument("--concurrency", type=int, help="Maximum simultaneous rollout episodes")
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--tool-seconds", type=int, help="Timeout for each Python tool call")
    parser.add_argument(
        "--checkpoint-interval", type=int, help="Save every N updates; 0 saves only at the end"
    )
    parser.add_argument("--eval-interval", type=int, help="Evaluate every N RL updates")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--no-eval", action="store_true", help="Skip baseline and checkpoint evaluation")
    parser.add_argument(
        "--no-checkpoint", action="store_true", help="Skip saves for disposable performance tests"
    )
    args = parser.parse_args(argv)
    if args.run:
        args.run = str(local_path(args.run, base=ROOT, relative=False, field="run"))
    cfg = settings()
    if args.dataset:
        cfg["dataset"] = str(local_path(args.dataset, base=ROOT, relative=False, field="dataset"))
    if planned_run_dir is not None:
        cfg["planned_run_dir"] = str(planned_run_dir)
    if args.checkpoint_interval is not None:
        if args.checkpoint_interval < 0:
            parser.error("--checkpoint-interval must be nonnegative")
        cfg["checkpoint_interval"] = args.checkpoint_interval
    if args.eval_samples is not None:
        if args.eval_samples < 1:
            parser.error("--eval-samples must be positive")
        cfg["eval_samples"] = args.eval_samples
    if args.eval_temperature is not None:
        if args.eval_temperature != 0:
            parser.error("Evaluation uses temperature 0")
        cfg["eval_temperature"] = args.eval_temperature
    if args.harness:
        cfg["harness"] = str(local_path(args.harness, base=ROOT, relative=False, field="harness"))
    if args.eval_index:
        if args.mode != "evaluate":
            parser.error("--eval-index is only supported for evaluate")
        cfg.update(
            eval_index=str(local_path(args.eval_index, base=ROOT, relative=False, field="eval_index")),
            expected_count=args.expected_count,
        )
    for key in ("concurrency", "context_tokens", "max_tokens", "tool_seconds", "steps", "eval_interval"):
        value = getattr(args, key)
        if value is not None:
            if value < 1:
                parser.error(f"{key} must be positive")
            cfg[key] = value
    cfg.update(no_eval=args.no_eval, no_checkpoint=args.no_checkpoint)
    if args.model:
        cfg["model"] = str(local_path(args.model, base=ROOT, relative=False, field="model"))
    for key in ("dataset", "harness", "model"):
        if getattr(args, key):
            local_tree(cfg[key], root=ROOT)
    if args.mode == "preflight":
        preflight(cfg)
    else:
        if args.mode == "resume" and not args.run:
            parser.error("resume requires --run")
        launch(cfg, args.mode, args.run, args.validate_only)


if __name__ == "__main__":
    main()
