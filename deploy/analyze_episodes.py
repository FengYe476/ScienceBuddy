#!/usr/bin/env python3
"""Summarize the episode.json files of one harness-evolution run into "what got in
the task model's way".

    python3 deploy/analyze_episodes.py runs/h-only-01

Read-only, does not depend on .venv (pure standard library), so it also runs on a
login node.

The directory layout is decided by coevolve/phase.py:
    <run>/harness_evolve/h0001/
        baseline/<task>/episode.json              H0 on test (phase.py:200)
        step-000N/interaction/<task>/…            training interaction, with reviewer feedback
        step-000N/validation-parent/<task>/…      the parent program on the selection set
        step-000N/validation-0M/<task>/…          candidate M on the selection set (selection.py:34)
        step-000N/evaluation/<task>/…             the last step's H* on test (phase.py:291)

Obstacles are split into four layers, matching four different sources of evidence
inside episode.json:
    1. stop_reason        why the episode ended (broker.py:174-182)
    2. submission format  anything other than exactly 1 <answer> tag scores 0
                          (data/verifier.py:9-10)
    3. tool errors        tool_calls[].observation.error, grouped by exception type
    4. remaining science errors   submitted, correctly formatted, but the answer is wrong
"""

import argparse
import collections
import json
import pathlib
import re
import sys
import unicodedata

BAR = "=" * 72
SHOW_ERRORS = 0


def pad(text, width):
    """CJK takes two terminal columns; str.format measures width in code points,
    which would skew the tables."""
    used = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - used)

# In broker.py everything except first_answer_evaluation means the task was not finished
INCOMPLETE = {
    "no_submission": "no answer submitted",
    "output_truncated": "single output truncated",
    "context_budget": "context exhausted",
    "model_call_budget": "model-call budget exhausted",
    "tool_call_budget": "tool-call budget exhausted",
    "tool_timeout": "tool timed out",
    "time_budget": "episode timed out",
    "candidate_error": "harness itself crashed",
    "execution_memory_budget": "execution container out of memory",
    "first_answer_evaluation": "submitted normally",
}


def load(folder):
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*/episode.json")):
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            print(f"  [skipping {path}: {exc}]", file=sys.stderr)
    return out


def error_kind(text):
    """Pull the exception type out of the last traceback line; network errors get their
    own bucket, because those come from the container's network isolation."""
    tail = text.strip().splitlines()[-1] if text.strip() else "(empty)"
    kind = re.split(r"[:(]", tail)[0].strip() or tail[:40]
    low = text.lower()
    if "errno 101" in low or "unreachable" in low or "temporary failure in name resolution" in low:
        return "network unreachable (container has no network)"
    if "no module named" in low:
        missing = re.search(r"[Nn]o module named ['\"]?([\w.]+)", text)
        return f"missing module {missing[1]}" if missing else "missing module"
    return kind[:44]


def answer_tags(episode):
    """Count the <answer> tags in the last submission, reproducing verifier.py's check."""
    subs = episode.get("submissions") or []
    if not subs:
        return None
    return len(re.findall(r"<answer>\s*(.*?)</answer>", subs[-1]["text"], re.S | re.I))


def report(name, episodes):
    print(BAR)
    print(f"{name}    {len(episodes)} tasks")
    print(BAR)
    if not episodes:
        print("  (no episodes)\n")
        return

    n = len(episodes)
    correct = sum(e["reward"] == 1 for e in episodes)
    print(f"  correct {correct}/{n} = {correct / n:.0%}\n")

    # ---- 1. why the episode ended ------------------------------------------
    print("  [1] reason the episode ended")
    for reason, count in collections.Counter(e["stop_reason"] for e in episodes).most_common():
        label = INCOMPLETE.get(reason, reason)
        print(f"      {pad(label, 34)}{count:3}  ({count / n:.0%})")

    # ---- 2. submission and format ------------------------------------------
    nosub = [e for e in episodes if not e.get("submissions")]
    bad_format, good_format = [], []
    for e in episodes:
        tags = answer_tags(e)
        if tags is None:
            continue
        (good_format if tags == 1 else bad_format).append((e, tags))
    print("\n  [2] submission and answer format")
    print(f"      {pad('never submitted', 30)}{len(nosub):3}   <- more than no_submission "
          "in [1]; truncation/exhaustion also never reach a submission")
    print(f"      {pad('submitted but tag count != 1', 30)}{len(bad_format):3}"
          "   <- verifier.py:9 scores it 0 outright")
    print(f"      {pad('submitted with valid format', 30)}{len(good_format):3}")
    for e, tags in bad_format[:3]:
        print(f"        {e['task_id']}  {tags} tags")

    # ---- 3. tool errors ----------------------------------------------------
    errs = [
        (e, t)
        for e in episodes
        for t in e.get("tool_calls", [])
        if t["observation"].get("error") and not t["observation"].get("infrastructure_error")
    ]
    total_tools = sum(len(e.get("tool_calls", [])) for e in episodes)
    print(f"\n  [3] {total_tools} tool calls, {len(errs)} errors"
          f" ({len(errs) / max(1, total_tools):.0%})")
    kinds = collections.Counter(error_kind(t["observation"]["error"]) for _, t in errs)
    for kind, count in kinds.most_common(10):
        print(f"      {count:3}x  {kind}")
    if SHOW_ERRORS:
        for e, t in errs[:SHOW_ERRORS]:
            code = t.get("code", "").strip()
            tail = t["observation"]["error"].strip().splitlines()[-1]
            print(f"\n      --- {e['task_id']} ---")
            for line in code.splitlines()[:12]:
                print(f"      | {line[:100]}")
            if len(code.splitlines()) > 12:
                print(f"      | … {len(code.splitlines())} lines in total")
            print(f"      => {tail[:160]}")

    # Hitting the same error over and over within one task = the "repeated failure"
    # of improver hypothesis 2(c)
    repeats = 0
    for e in episodes:
        seen = collections.Counter(
            error_kind(t["observation"]["error"])
            for t in e.get("tool_calls", [])
            if t["observation"].get("error")
        )
        repeats += sum(v - 1 for v in seen.values() if v > 1)
    if repeats:
        print(f"      of which {repeats} are repeats of the same error class within the same task")

    # ---- 4. submitted, correctly formatted, still wrong --------------------
    science = [e for e, _ in good_format if e["reward"] != 1]
    print(f"\n  [4] valid submission but wrong answer   {len(science):3}"
          "   <- the real science-capability gap")
    # Overall accuracy mixes two things: whether an answer was submitted at all, and
    # whether a submitted answer was right. Only the latter is science capability.
    attempted = len(good_format)
    if attempted:
        print(f"      post-submission hit rate {correct}/{attempted} = {correct / attempted:.0%}"
              "   <- read apart from overall accuracy; only this one tracks capability")

    # ---- failing samples ---------------------------------------------------
    stuck = nosub or [e for e in episodes if e["stop_reason"] in INCOMPLETE and e["stop_reason"] != "first_answer_evaluation"]
    if stuck:
        print(f"\n  Unfinished tasks (first 3, to see where they got stuck):")
        for e in stuck[:3]:
            calls = e.get("calls") or []
            tail = calls[-1]["text"][-200:].replace("\n", " ⏎ ") if calls else "(zero model calls)"
            print(f"      {e['task_id']}  stop={e['stop_reason']}  "
                  f"model calls {len(calls)}  tool calls {len(e.get('tool_calls', []))}")
            print(f"        last output …{tail}")
    print()


def transitions(before, after, label_a, label_b):
    a = {e["task_id"]: e for e in before}
    b = {e["task_id"]: e for e in after}
    shared = sorted(set(a) & set(b))
    if not shared:
        return
    print(BAR)
    print(f"per-task change   {label_a} -> {label_b}")
    print(BAR)
    tally = collections.Counter()
    for tid in shared:
        x, y = int(a[tid]["reward"] == 1), int(b[tid]["reward"] == 1)
        mark = {(0, 1): "fixed ++", (1, 0): "broke --",
                (1, 1): "still right", (0, 0): "still wrong"}[(x, y)]
        tally[mark] += 1
        print(f"  {tid:26} {pad(mark, 13)}{a[tid]['stop_reason']:24} -> {b[tid]['stop_reason']}")
    print(f"\n  fixed {tally['fixed ++']} · broke {tally['broke --']} · "
          f"still right {tally['still right']} · still wrong {tally['still wrong']}")
    net = tally["fixed ++"] - tally["broke --"]
    print(f"  net {net:+d} tasks -- the aggregate accuracy difference hides "
          f"{tally['broke --']} tasks that regressed\n")


def retest(root, steps):
    """Drift when the same harness is measured twice -- the yardstick for whether
    accepting on Δ>0 means anything.

    phase.py:264 turns the candidate selected at each step into the next step's parent,
    and selection.py:30 re-evaluates that parent completely at the next step
    (algorithm.md explicitly forbids reusing cached scores). So step-N's
    validation-parent and the candidate that was accepted in step-(N-1) run the same
    program, the same batch of tasks, the same temperature 0 and the same task seed.
    The difference between the two scores is measurement noise.

    harness_sha256 from the episodes is used to confirm they really do come from the
    same source, so that a "the program changed" case is not mistaken for drift.
    """
    rows = []
    for i in range(1, len(steps)):
        update = steps[i - 1] / "update.json"
        if not update.is_file():
            continue
        try:
            picked = json.loads(update.read_text()).get("selected_candidate")
        except (OSError, ValueError):
            continue
        if picked is None:
            continue
        before = load(steps[i - 1] / f"validation-{picked:02d}")
        after = load(steps[i] / "validation-parent")
        if not before or not after:
            continue
        shas = {e.get("harness_sha256") for e in before} | {e.get("harness_sha256") for e in after}
        same = len(shas) == 1 and None not in shas
        a = {e["task_id"]: int(e["reward"] == 1) for e in before}
        b = {e["task_id"]: int(e["reward"] == 1) for e in after}
        shared = sorted(set(a) & set(b))
        flips = [t for t in shared if a[t] != b[t]]
        rows.append((steps[i - 1].name, steps[i].name, picked, sum(a[t] for t in shared),
                     sum(b[t] for t in shared), len(shared), flips, same, before, after))
    if not rows:
        return
    print(BAR)
    print("same-program re-measurement drift (the yardstick for measurement noise)")
    print(BAR)
    worst = 0
    for s0, s1, picked, x, y, n, flips, same, before, after in rows:
        tag = "same program ✓" if same else "!! harness_sha256 differs, not the same program"
        print(f"  {s0}/validation-{picked:02d} -> {s1}/validation-parent   {tag}")
        print(f"      {x}/{n}  ->  {y}/{n}   diff {y - x:+d} tasks; per-task flips {len(flips)}")
        worst = max(worst, len(flips))
        for t in flips[:4]:
            ea = next(e for e in before if e["task_id"] == t)
            eb = next(e for e in after if e["task_id"] == t)
            print(f"        {t:26} {ea['stop_reason']:22} -> {eb['stop_reason']}")
    print(f"\n  Re-measuring the same program flips at most {worst} tasks.")
    print("  selection.py:45 accepts on Δ>0 -- a Δ smaller than that drift cannot be "
          "told apart from noise.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="runs/h-only-01", help="experiment directory")
    ap.add_argument("--phase", default="h0001")
    ap.add_argument("--show-errors", type=int, default=0, metavar="N",
                    help="additionally print the source and exception of N failing tool "
                         "calls per section, to judge what the model is actually looking for")
    args = ap.parse_args()
    global SHOW_ERRORS
    SHOW_ERRORS = args.show_errors

    root = pathlib.Path(args.run) / "harness_evolve" / args.phase
    if not root.is_dir():
        found = sorted(str(p) for p in pathlib.Path(args.run).glob("*/*")) \
            if pathlib.Path(args.run).is_dir() else []
        sys.exit(f"cannot find {root}"
                 + ("\nactually present: " + ", ".join(found) if found else ""))

    steps = sorted(root.glob("step-*"))
    print(f"experiment {root}, {len(steps)} steps in total\n")

    # Progress: while the experiment is only half done, the denominators of the sections
    # below are "so far", not the final values
    stages = [("initial-preflight", root / "initial-preflight"), ("baseline", root / "baseline")]
    for step in steps:
        stages.append((f"{step.name}/interaction", step / "interaction"))
        for cand in sorted(step.glob("candidate-*")):
            stages.append((f"{step.name}/{cand.name}/preflight", cand / "preflight"))
        stages.append((f"{step.name}/validation-parent", step / "validation-parent"))
        for cand in sorted(step.glob("validation-[0-9][0-9]")):
            stages.append((f"{step.name}/{cand.name}", cand))
        stages.append((f"{step.name}/evaluation", step / "evaluation"))

    print(BAR)
    print("progress")
    print(BAR)
    done = 0
    active = None
    for label, folder in stages:
        eps = load(folder) if folder.is_dir() else []
        if not eps:
            continue
        done += len(eps)
        correct = sum(e["reward"] == 1 for e in eps)
        active = label
        print(f"  {pad(label, 34)}{len(eps):4} tasks   correct {correct}")
    print(f"\n  {done} episodes done (478 expected)   current stage {active or 'not started yet'}")
    print("  Note: while the experiment is unfinished, each section's denominator below "
          "is the number of tasks so far\n")

    baseline = load(root / "baseline")
    report("H0 baseline (test set)", baseline)

    for step in steps:
        inter = load(step / "interaction")
        if inter:
            report(f"{step.name} training interaction (train set, this is the evidence "
                   "the improver sees)", inter)

    final = None
    for step in reversed(steps):
        final = load(step / "evaluation")
        if final:
            report(f"H* final (test set, {step.name})", final)
            break

    if baseline and final:
        transitions(baseline, final, "H0", "H*")

    retest(root, steps)

    # Performance on the selection set at every step of the evolution
    print(BAR)
    print("parent vs candidates on the selection set (val)")
    print(BAR)
    for step in steps:
        row = []
        parent = load(step / "validation-parent")
        if parent:
            row.append(f"parent {sum(e['reward'] == 1 for e in parent)}/{len(parent)}")
        for cand in sorted(step.glob("validation-[0-9][0-9]")):
            eps = load(cand)
            if eps:
                row.append(f"{cand.name[-2:]} {sum(e['reward'] == 1 for e in eps)}/{len(eps)}")
        update = step / "update.json"
        status = ""
        if update.is_file():
            try:
                data = json.loads(update.read_text())
                status = f"  -> {data.get('status')} (c{data.get('selected_candidate')})"
            except (OSError, ValueError):
                pass
        print(f"  {step.name}  " + (" · ".join(row) if row else "(no validation data)") + status)
    print()


if __name__ == "__main__":
    main()
