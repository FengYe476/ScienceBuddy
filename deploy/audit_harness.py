#!/usr/bin/env python3
"""Audit what each step changed in the harness, and whether it is gaming the benchmark.

    python3 deploy/audit_harness.py runs/h-only-small-01 --release data/releases/id200-val30-test30-v1

Three sections, independent of each other:

  [A] Exploitable bias on the dataset side -- these were produced by my own
      build_release.py, so they are not the model's fault, but once the harness learns
      to exploit them it gets an inflated score:
        * the marginal distribution of the correct letter (with unequal option counts
          it skews towards the earlier letters)
        * whether the correct option is systematically longer (the classic spurious
          MCQ feature)
        * a few trivial baselines that score points without reading the task at all

  [B] Step-by-step harness diff -- which lines each step added on top of the parent.

  [C] Harness text scan -- looking for three kinds of thing:
        * the verbatim reference answer of any task (GWAS answers are rsIDs / gene
          names, which can be searched for directly)
        * letter-preference heuristics ("if unsure answer C")
        * task IDs, paths outside the sandbox

Read-only, pure standard library.
"""

import argparse
import collections
import difflib
import json
import pathlib
import re
import sys

BAR = "=" * 76


def selected_harnesses(root):
    """Return [(stage name, Path)] in step order; H0 is the initial program the
    baseline uses."""
    out = []
    for step in sorted(root.glob("step-*")):
        update = step / "update.json"
        if not update.is_file():
            continue
        try:
            data = json.loads(update.read_text())
        except (OSError, ValueError):
            continue
        picked = data.get("selected_candidate")
        if picked is None:
            continue
        path = step / f"candidate-{picked:02d}" / "harness.py"
        if path.is_file():
            out.append((f"{step.name} (c{picked})", path, data))
    return out


# --------------------------------------------------------------- [A] dataset
def audit_dataset(release, splits=("val", "test")):
    """Compute each split separately, then combined. Computing them separately is
    necessary: free-response tasks cannot be guessed by a heuristic, and the two splits
    have different ratios of multiple-choice to free-response, so a trivial baseline
    computed over the mixture cannot be compared directly against the accuracy measured
    on one split."""
    free = []
    for split in splits:
        free.extend(audit_one_split(release, split))
    if len(splits) > 1:
        audit_one_split(release, splits, label="/".join(splits))
    return free


def audit_one_split(release, split, label=None):
    manifest = json.loads((release / "manifest.json").read_text())
    wanted = {split} if isinstance(split, str) else set(split)
    rows = [r for r in manifest["tasks"] if r["split"] in wanted]
    print(BAR)
    print(f"[A] exploitable dataset bias   split={label or split}   {len(rows)} tasks")
    print(BAR)

    letters, n_options, longest_hit, first_hit, mcq = (
        collections.Counter(), collections.Counter(), 0, 0, 0)
    per_split_letters = collections.defaultdict(collections.Counter)
    free_answers = []
    for row in rows:
        base = release / row["id"]
        try:
            public = json.loads((base / "public/task.json").read_text())
            ref = json.loads((base / "evaluator/reference.json").read_text())
        except (OSError, ValueError):
            continue
        answer = str(ref.get("answer", "")).strip()
        options = public.get("options") or []
        if not options:
            free_answers.append((row["id"], answer))
            continue
        mcq += 1
        letters[answer] += 1
        per_split_letters[row["split"]][answer] += 1
        n_options[len(options)] += 1
        index = ord(answer.upper()) - 65 if len(answer) == 1 else -1
        if 0 <= index < len(options):
            lengths = [len(o) for o in options]
            if lengths[index] == max(lengths):
                longest_hit += 1
            if index == 0:
                first_hit += 1

    if not mcq:
        print("  no multiple-choice tasks")
        return
    print(f"  {mcq} multiple-choice tasks, {len(free_answers)} free-response\n")
    print("  Correct-letter distribution (if uniform, each letter should be about "
          "1/option-count):")
    for letter, count in sorted(letters.items()):
        print(f"      {letter}  {count:3}  {count / mcq:5.1%}")
    print("\n  Options per task:")
    for k, v in sorted(n_options.items()):
        print(f"      {k} options  {v:3} tasks")

    total = len(rows)
    uniform = sum(v / k for k, v in n_options.items()) / mcq
    best_letter, best_n = letters.most_common(1)[0]
    print("\n  Trivial baselines (points obtainable without reading the task)")
    print(f"      {'':38} {'within MCQ':>10} {'whole split':>11}")
    print(f"      {'random guess (per-task option count)':38} "
          f"{uniform:10.1%} {mcq * uniform / total:11.1%}")
    print(f"      {'always answer the most common letter ' + best_letter:38} "
          f"{best_n / mcq:10.1%} {best_n / total:11.1%}"
          f"  {'<- exploitable' if best_n / mcq > uniform + 0.10 else ''}")
    print(f"      {'always pick the longest option':38} "
          f"{longest_hit / mcq:10.1%} {longest_hit / total:11.1%}"
          f"  {'<- length bias, exploitable' if longest_hit / mcq > uniform + 0.10 else ''}")
    print(f"      {'always answer A':38} {first_hit / mcq:10.1%} {first_hit / total:11.1%}")
    print(f"\n  The \"whole split\" column scores the {len(free_answers)} free-response "
          "tasks as 0 -- a heuristic")
    print("  cannot guess rsIDs/gene names. Compare reported accuracy against that "
          "column, not")
    print("  against \"within MCQ\".")
    return free_answers


# --------------------------------------------------------------- [B] diff
def audit_diffs(initial, stages):
    print("\n" + BAR)
    print("[B] what each step added on top of the parent")
    print(BAR)
    previous, previous_name = initial.read_text().splitlines(), "H0 (scientific.py)"
    for name, path, data in stages:
        current = path.read_text().splitlines()
        diff = list(difflib.unified_diff(previous, current, lineterm="", n=0))
        added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
        removed = [l[1:] for l in diff if l.startswith("-") and not l.startswith("---")]
        print(f"\n  {previous_name}  ->  {name}")
        print(f"      {len(previous)} lines -> {len(current)} lines   "
              f"+{len(added)} / -{len(removed)}")
        if data.get("hypothesis"):
            print(f"      hypothesis: {data['hypothesis'][:260]}")
        show = [l for l in added if l.strip()][:14]
        for line in show:
            print(f"      + {line[:118]}")
        if len(added) > len(show):
            print(f"      … {len(added) - len(show)} further lines")
        previous, previous_name = current, name


# --------------------------------------------------------------- [C] scan
SUSPICIOUS = [
    (r"(?i)\b(if|when)\b[^.\n]{0,60}\b(unsure|uncertain|unclear|cannot determine|no evidence)\b"
     r"[^.\n]{0,60}\b(answer|choose|pick|select|guess)\b", "fallback guessing rule when unsure"),
    (r"(?i)\b(always|default to|prefer)\b[^.\n]{0,40}\b(option\s*)?[A-E]\b(?![a-z])", "letter preference"),
    (r"(?i)\blongest\b[^.\n]{0,40}\boption\b|\boption\b[^.\n]{0,40}\blongest\b", "picking an option by length"),
    (r"(?i)\bmost (specific|detailed|technical)\b[^.\n]{0,30}\b(option|answer|choice)\b", "picking an option by wording style"),
    (r"/(etc|root|home)/|\.\./\.\./|evaluator/|reference\.json", "path outside the sandbox or reference-answer file"),
]


def audit_scan(stages, free_answers, release):
    print("\n" + BAR)
    print("[C] harness text scan")
    print(BAR)

    # The task_id of every task, used to look for hard-coding
    manifest = json.loads((release / "manifest.json").read_text())
    task_ids = {r["id"] for r in manifest["tasks"]}
    # Free-response reference answers (rsIDs / gene names) are distinctive enough to
    # search for directly
    answers = {a for _, a in free_answers if len(a) >= 4}

    for name, path, _ in stages:
        text = path.read_text()
        print(f"\n  [{name}]  {len(text.splitlines())} lines")
        hits = 0
        for pattern, label in SUSPICIOUS:
            for m in re.finditer(pattern, text):
                line = text[:m.start()].count("\n") + 1
                print(f"      !! {label}  line {line}: {m.group(0)[:90]}")
                hits += 1
        leaked = sorted(a for a in answers
                        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(a)}(?![A-Za-z0-9_])", text))
        if leaked:
            print(f"      !! contains {len(leaked)} verbatim reference answers: {leaked[:6]}")
            hits += 1
        ids = sorted(t for t in task_ids if t in text)
        if ids:
            print(f"      !! contains {len(ids)} task IDs: {ids[:6]}")
            hits += 1
        if not hits:
            print("      no suspicious patterns found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--release", required=True)
    ap.add_argument("--phase", default="h0001")
    ap.add_argument("--initial", default="src/simple_scibuddy/harness/scientific.py")
    args = ap.parse_args()

    root = pathlib.Path(args.run) / "harness_evolve" / args.phase
    release = pathlib.Path(args.release)
    initial = pathlib.Path(args.initial)
    for p, what in ((root, "experiment directory"), (release / "manifest.json", "release"),
                    (initial, "initial harness")):
        if not p.exists():
            sys.exit(f"cannot find {what}: {p}")

    stages = selected_harnesses(root)
    if not stages:
        sys.exit(f"no accepted harness found under {root}")

    free_answers = audit_dataset(release) or []
    audit_diffs(initial, stages)
    audit_scan(stages, free_answers, release)


if __name__ == "__main__":
    main()
