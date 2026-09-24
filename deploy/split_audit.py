#!/usr/bin/env python3
"""Check whether the val / test splits are interchangeable, and which subtasks each is
made of.

    python3 deploy/split_audit.py data/releases/id200-val30-test30-v1
    python3 deploy/split_audit.py data/releases/id200-val30-test30-v1 --run runs/h-only-small-01

Why this needs checking: split_by_group in build_release.py:184 assigns whole
source_group blocks at a time, and DbQA's source_group is f"dbqa:{subtask}" -- one
subtask is one group, and DbQA has only 10 groups. So test receives one or two whole
subtasks and val receives one or two different ones, and the two splits are no longer
interchangeable in difficulty. In h-only-small-01 the very same harness scores 15/30
on val and 26/30 on test, and the ratio is likewise about 0.5 at H0 (4/30 vs 8/30) --
which looks like a property of the split rather than a difference between programs.
This script is how that gets confirmed.

Given --run, it additionally breaks the baseline and final-evaluation accuracies down
by subtask, to see which subtasks the gain is concentrated in.
Read-only, pure standard library.
"""

import argparse
import collections
import json
import pathlib
import sys

BAR = "=" * 74


def read_tasks(release):
    manifest = json.loads((release / "manifest.json").read_text())
    out = []
    for row in manifest["tasks"]:
        public = release / row["id"] / "public/task.json"
        subtask = None
        if public.is_file():
            try:
                subtask = json.loads(public.read_text()).get("subtask")
            except (OSError, ValueError):
                pass
        out.append({**row, "subtask": subtask})
    return out


def episodes(folder):
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
    ap.add_argument("release")
    ap.add_argument("--run", default=None)
    ap.add_argument("--phase", default="h0001")
    args = ap.parse_args()

    release = pathlib.Path(args.release)
    if not (release / "manifest.json").is_file():
        sys.exit(f"cannot find {release}/manifest.json")
    tasks = read_tasks(release)

    print(BAR)
    print("family composition of each split")
    print(BAR)
    fam = collections.defaultdict(collections.Counter)
    for t in tasks:
        fam[t["split"]][t["family"]] += 1
    for split in ("train", "val", "test"):
        total = sum(fam[split].values())
        share = "  ".join(f"{k}={v}" for k, v in sorted(fam[split].items()))
        print(f"  {split:6} {total:4}   {share}")

    print("\n" + BAR)
    print("subtask composition of each split -- an empty intersection means the two "
          "splits are not interchangeable")
    print(BAR)
    sub = collections.defaultdict(collections.Counter)
    for t in tasks:
        sub[t["split"]][t["subtask"] or "(none)"] += 1
    for split in ("val", "test"):
        print(f"\n  [{split}]")
        for name, n in sub[split].most_common():
            print(f"      {n:3}  {name}")
    shared = set(sub["val"]) & set(sub["test"])
    only_val = set(sub["val"]) - set(sub["test"])
    only_test = set(sub["test"]) - set(sub["val"])
    print(f"\n  subtasks in common {len(shared)}: {sorted(shared) or 'none'}")
    print(f"  val only  {len(only_val)}: {sorted(only_val) or 'none'}")
    print(f"  test only {len(only_test)}: {sorted(only_test) or 'none'}")
    if not shared:
        print("\n  !! val and test share no subtask at all -- they are not measuring the "
              "same thing;")
        print("     selecting on val and reporting on test amounts to swapping the exam paper.")

    print("\n" + BAR)
    print("source_group overlap (experiments.md:127 records the paper as 20 Train-Val groups)")
    print(BAR)
    groups = collections.defaultdict(set)
    for t in tasks:
        groups[t["split"]].add(t["source_group"])
    print(f"  Train-Val  {len(groups['train'] & groups['val']):3} groups")
    print(f"  Train-Test {len(groups['train'] & groups['test']):3} groups")
    print(f"  Val-Test   {len(groups['val'] & groups['test']):3} groups")

    if not args.run:
        return

    root = pathlib.Path(args.run) / "harness_evolve" / args.phase
    base = episodes(root / "baseline")
    final = []
    for step in sorted(root.glob("step-*"), reverse=True):
        final = episodes(step / "evaluation")
        if final:
            break
    if not (base and final):
        print("\n(no complete baseline / evaluation, skipping the per-subtask comparison)")
        return

    print("\n" + BAR)
    print("H0 -> H* by subtask on test  -- where the gain is concentrated")
    print(BAR)
    by_id = {t["id"]: t for t in tasks}
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for e in base:
        key = by_id.get(e["task_id"], {}).get("subtask") or e.get("family", "?")
        agg[key][0] += int(e["reward"] == 1)
        agg[key][2] += 1
    for e in final:
        key = by_id.get(e["task_id"], {}).get("subtask") or e.get("family", "?")
        agg[key][1] += int(e["reward"] == 1)
    print(f"  {'subtask':46} {'H0':>6} {'H*':>6} {'tasks':>5}")
    for key, (h0, hs, n) in sorted(agg.items(), key=lambda kv: -(kv[1][1] - kv[1][0])):
        print(f"  {key[:46]:46} {h0:6} {hs:6} {n:5}")

    # ---- Free measurements from the intermediate steps ------------------
    # phase.py begins each step with the harness selected at the previous step, so
    # step-N/interaction runs H(N-1). The tasks differ from step to step (the cursor
    # rolls forward), so the accuracies are not comparable, but it does show which
    # generation each subtask appears in and how it does there -- locating which step
    # an improvement happened at, without rerunning anything.
    print("\n" + BAR)
    print("train interaction by subtask (step-N/interaction runs H(N-1))")
    print(BAR)
    steps = sorted(root.glob("step-*"))
    seen = set()
    table = {}
    for i, step in enumerate(steps):
        eps = episodes(step / "interaction")
        if not eps:
            continue
        counter = collections.defaultdict(lambda: [0, 0])
        for e in eps:
            key = by_id.get(e["task_id"], {}).get("subtask") or e.get("family", "?")
            counter[key][0] += int(e["reward"] == 1)
            counter[key][1] += 1
            seen.add(key)
        table[f"H{i}"] = counter
    if not table:
        print("  (no interaction data)")
        return
    names = sorted(seen)
    header = "  " + f"{'subtask':44}" + "".join(f"{k:>10}" for k in table)
    print(header)
    for name in names:
        cells = ""
        for counter in table.values():
            got, n = counter.get(name, [0, 0])
            cells += f"{(f'{got}/{n}' if n else '-'):>10}"
        print(f"  {name[:44]:44}{cells}")
    target = [n for n in names if n in {t.get('subtask') for t in tasks if t['split'] == 'test'}]
    print(f"\n  subtasks shared with test: {target or 'none'}")
    print("  The tasks differ at every step, so the accuracies cannot be compared "
          "directly; what this")
    print("  shows is which generation a class of task first appears in and whether it "
          "was answered correctly.")


if __name__ == "__main__":
    main()
