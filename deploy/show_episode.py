#!/usr/bin/env python3
"""Lay out the full trajectory of a single episode, to judge where the answer came from.

    python3 deploy/show_episode.py runs/h-only-small-01 --stage step-0003/evaluation
    python3 deploy/show_episode.py runs/h-only-small-01 --stage step-0003/evaluation --task gwas-0255fd96bde0
    python3 deploy/show_episode.py runs/h-only-small-01 --harness

When accuracy is suspiciously high, the question to answer is "on what grounds did the
model get it right". Three cases can be told apart:
  * the answer was in the prompt    -> the reference answer appears in public_task.prompt
  * a tool dug it up                -> the answer appears in some tool observation,
                                       earlier in time than the submission
  * the harness handed it over      -> the answer appears in the system prompt or in the
                                       harness source
Read-only, pure standard library.
"""

import argparse
import json
import pathlib
import re
import sys

BAR = "=" * 78


def load(path):
    return json.loads(path.read_text())


def clip(text, n):
    text = str(text)
    return text if len(text) <= n else text[:n] + f" …[{len(text)} characters in total]"


def show_harness(root):
    """List the harness accepted at each step, with the line-count delta against the parent."""
    for step in sorted(root.glob("step-*")):
        update = step / "update.json"
        if not update.is_file():
            continue
        data = load(update)
        picked = data.get("selected_candidate")
        print(f"\n{BAR}\n{step.name}  status={data.get('status')}  selected=c{picked}\n{BAR}")
        if data.get("hypothesis"):
            print("  hypothesis:", clip(data["hypothesis"], 900))
        if data.get("reason"):
            print("  reason:", clip(data["reason"], 400))
        for cand in sorted(step.glob("candidate-*")):
            src = cand / "harness.py"
            if src.is_file():
                lines = src.read_text().splitlines()
                mark = " <- accepted" if picked is not None and cand.name.endswith(f"{picked:02d}") else ""
                print(f"    {cand.name}  {len(lines)} lines{mark}")


def find_answer_in(text, answer):
    """Does the answer appear as a standalone word in the text (so that 'A' does not
    match an arbitrary letter)?"""
    if not answer or not text:
        return False
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(answer))}(?![A-Za-z0-9_])", str(text)) is not None


def show(episode, reference=None):
    print(BAR)
    print(f"{episode['task_id']}   family={episode.get('family')}   reward={episode['reward']}")
    print(f"stop={episode['stop_reason']}   harness_sha256={str(episode.get('harness_sha256'))[:16]}")
    print(BAR)

    public = episode.get("public_task", {})
    prompt = public.get("prompt", "")
    print("\n[task prompt]")
    print(clip(prompt, 1500))
    if public.get("options"):
        print("\n[options]")
        for i, o in enumerate(public["options"]):
            print(f"  {chr(65 + i)}. {clip(o, 160)}")

    submitted = episode["submissions"][-1]["text"] if episode.get("submissions") else ""
    outcome = episode["submissions"][-1]["outcome"] if episode.get("submissions") else {}
    graded = outcome.get("answer", "")

    print(f"\n[submission] {clip(submitted, 400)}")
    print(f"[grading] answer={graded!r}  score={outcome.get('score')}  "
          f"format_valid={outcome.get('answer_format_valid')}")

    # Where the answer shows up -- this is the core of deciding whether there is leakage
    print("\n[answer provenance]")
    if not graded:
        print("  (no answer to trace)")
    else:
        print(f"  appears in the task prompt: {find_answer_in(prompt, graded)}")
        for i, t in enumerate(episode.get("tool_calls", [])):
            obs = t["observation"]
            hit_out = find_answer_in(obs.get("stdout", ""), graded)
            hit_code = find_answer_in(t.get("code", ""), graded)
            if hit_out or hit_code:
                where = "tool output" if hit_out else "code written by the model"
                print(f"  tool[{i}] appears in {where} (call_index={t.get('call_index')})")

    print("\n[turn by turn]")
    tools = episode.get("tool_calls", [])
    for i, call in enumerate(episode.get("calls", [])):
        print(f"\n  --- model call {i} ---")
        if i == 0:
            for m in call.get("messages", [])[:2]:
                print(f"  [{m['role']}] {clip(m['content'], 1200)}")
        print(f"  [assistant] {clip(call.get('text', ''), 1200)}")
        for t in tools:
            if t.get("call_index") == i:
                print(f"  [tool code] {clip(t.get('code', ''), 700)}")
                obs = t["observation"]
                if obs.get("error"):
                    print(f"  [tool error] {clip(obs['error'], 300)}")
                else:
                    print(f"  [tool output] {clip(obs.get('stdout', ''), 700)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--phase", default="h0001")
    ap.add_argument("--stage", default="step-0003/evaluation")
    ap.add_argument("--task", default=None,
                    help="if omitted, list every task in the stage and pick the first correct one")
    ap.add_argument("--harness", action="store_true",
                    help="only show the harness accepted at each step and the improver hypothesis")
    args = ap.parse_args()

    root = pathlib.Path(args.run) / "harness_evolve" / args.phase
    if not root.is_dir():
        sys.exit(f"cannot find {root}")
    if args.harness:
        show_harness(root)
        return

    folder = root / args.stage
    if not folder.is_dir():
        sys.exit(f"cannot find {folder}")
    paths = sorted(folder.glob("*/episode.json"))
    if not paths:
        sys.exit(f"no episodes under {folder}")

    if args.task:
        picked = [p for p in paths if p.parent.name == args.task]
        if not picked:
            sys.exit(f"no {args.task}; available: " + ", ".join(p.parent.name for p in paths[:8]))
    else:
        correct = [p for p in paths if load(p)["reward"] == 1]
        picked = correct[:1] or paths[:1]
        print(f"(no --task given, picked {picked[0].parent.name}; {len(paths)} tasks in this "
              f"stage, {len(correct)} answered correctly)\n")
    show(load(picked[0]))


if __name__ == "__main__":
    main()
