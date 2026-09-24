#!/usr/bin/env python3
"""Re-measure the intermediate harnesses of an evolution run on the test split, to
locate which step the improvement happened at.

    bash deploy/rerun_on_test.sh configs/h-only-small.toml runs/h-only-small-01

The original experiment only measured H0 (baseline) and the finally selected one
(step-NNNN/evaluation) on test; the intermediate generations only have val scores. But
the subtask composition of val and test barely overlaps, so the step-by-step curve on
val cannot answer "which step did that jump on test come from".

This script calls coevolve.phase.batch directly -- the same function the original
experiment used, so the task seed (digest([cfg.seed, task_id])), temperature 0, and the
actions / context / tool budgets are all byte-for-byte identical, and the results can be
compared side by side with baseline and evaluation.

It runs neither the improver nor the reviewer (user_turns=0); it only performs a single
answering pass and grades it. Output is written to
<run>/harness_evolve/<phase>/attribution/H<N>/, leaving the existing directories untouched.
"""

import argparse
import asyncio
import collections
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

from simple_scibuddy.artifacts import ROOT, write_json
from simple_scibuddy.configuration import load_config, settings
from simple_scibuddy.coevolve.phase import batch, summarize
from simple_scibuddy.data.dataset import TaskDataset
from simple_scibuddy.inference.server import serve_pool

BAR = "=" * 74


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def stages(root):
    """[(generation name, harness path)]; the program selected at the last step already
    has an evaluation, so it is skipped."""
    steps = sorted(root.glob("step-*"))
    out = []
    for i, step in enumerate(steps, start=1):
        update = step / "update.json"
        if not update.is_file():
            continue
        picked = json.loads(update.read_text()).get("selected_candidate")
        if picked is None:
            continue
        harness = step / f"candidate-{picked:02d}" / "harness.py"
        measured = (step / "evaluation").is_dir()
        out.append((f"H{i}", harness, measured, step.name, picked))
    return out


def load_folder(folder):
    out = []
    if folder.is_dir():
        for p in sorted(folder.rglob("episode.json")):
            try:
                out.append(json.loads(p.read_text()))
            except (OSError, ValueError):
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="the toml that produced this experiment, so that "
                                   "budgets and seed stay identical")
    ap.add_argument("run")
    ap.add_argument("--phase", default="h0001")
    ap.add_argument("--only", default=None,
                    help="comma-separated generation names, e.g. H1,H2")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        sys.exit("CUDA_VISIBLE_DEVICES is required; launch via deploy/rerun_on_test.sh")

    root = resolve(args.run) / "harness_evolve" / args.phase
    if not root.is_dir():
        sys.exit(f"cannot find {root}")

    config = load_config(resolve(args.config))
    base = settings()
    # Reproduce the assembly order of loop.py:57-61 verbatim (the toml overrides the
    # first two entries), so that seed, actions, and the context/tool budgets are exactly
    # the same as in the original experiment -- if any one of them differed, the
    # re-measured results could no longer be compared side by side with baseline and
    # evaluation.
    cfg = {
        **base,
        "workers": 8,
        "user_turns": 1,
        **{k: v for k, v in config["harness_evolve"].items() if k != "improver"},
    }
    # Override after the expansion: the re-measurement has no reviewer attached.
    # phase.py:104-105 already zeroes user_turns when reviewer=None; this is spelled out
    # only to make the intent readable.
    cfg = dict(cfg, model=base["model"], user_turns=0)
    if cfg.get("seed") is None:
        sys.exit("no seed in the config; if the task seed does not match, comparability is lost")

    dataset = TaskDataset(cfg["dataset"])
    rows = dataset.tasks("test")

    todo = [(name, h, step, pick) for name, h, measured, step, pick in stages(root)
            if not measured and h.is_file()]
    if args.only:
        wanted = {w.strip() for w in args.only.split(",")}
        todo = [t for t in todo if t[0] in wanted]
    if not todo:
        sys.exit("no generation needs re-measuring (the last step already has an evaluation)")

    print(BAR)
    print(f"re-measuring {len(todo)} generations x {len(rows)} test tasks = "
          f"{len(todo) * len(rows)} episodes")
    for name, harness, step, pick in todo:
        print(f"  {name}  <- {step}/candidate-{pick:02d}/harness.py")
    print(BAR, flush=True)

    from transformers import AutoTokenizer

    model = str(resolve(cfg["model"]))
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    out_root = root / "attribution"
    out_root.mkdir(parents=True, exist_ok=True)

    with ExitStack() as services:
        url = services.enter_context(
            serve_pool(model, devices, out_root / "solver",
                       max(cfg["context_tokens"], 32768), share_last=False)
        )
        for name, harness, _, _ in todo:
            folder = out_root / name
            if folder.is_dir() and load_folder(folder):
                print(f"\n{name} already exists, skipping", flush=True)
                continue
            print(f"\n--- {name} ---", flush=True)
            episodes = asyncio.run(batch(dataset, rows, harness, url, tokenizer, cfg, folder))
            stats = summarize(episodes)
            write_json(folder / "summary.json", {"harness": str(harness), **stats})
            print(f"  {name}: {stats['correct']}/{stats['tasks']} = {stats['accuracy']:.1%}", flush=True)

    # ---- Summary: H0 / intermediate generations / H* side by side, broken down by subtask
    series = [("H0", load_folder(root / "baseline"))]
    for name, _, _, _ in todo:
        series.append((name, load_folder(out_root / name)))
    for step in sorted(root.glob("step-*"), reverse=True):
        final = load_folder(step / "evaluation")
        if final:
            series.append(("H* (final)", final))
            break
    series = [(n, e) for n, e in series if e]

    subtask = {}
    for row in dataset.manifest["tasks"]:
        if row["split"] != "test":
            continue
        try:
            subtask[row["id"]] = json.loads(
                (dataset.root / row["id"] / "public/task.json").read_text()).get("subtask")
        except (OSError, ValueError):
            subtask[row["id"]] = None

    print("\n" + BAR)
    print("per-generation results on test")
    print(BAR)
    for name, eps in series:
        correct = sum(e["reward"] == 1 for e in eps)
        print(f"  {name:12} {correct:3}/{len(eps)} = {correct / len(eps):5.1%}")

    print("\n" + BAR)
    print("broken down by subtask -- which generation and which kind of task the jump happened in")
    print(BAR)
    keys = sorted({subtask.get(e["task_id"]) or "?" for _, eps in series for e in eps})
    print("  " + f"{'subtask':40}" + "".join(f"{n:>12}" for n, _ in series))
    for key in keys:
        cells = ""
        for _, eps in series:
            hit = sum(e["reward"] == 1 for e in eps if (subtask.get(e["task_id"]) or "?") == key)
            n = sum(1 for e in eps if (subtask.get(e["task_id"]) or "?") == key)
            cells += f"{(f'{hit}/{n}' if n else '-'):>12}"
        print(f"  {key[:40]:40}{cells}")
    print("\n  Same batch of tasks, same seed and temperature 0, so the columns can be "
          "compared directly.\n")


if __name__ == "__main__":
    main()
