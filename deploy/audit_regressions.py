#!/usr/bin/env python3
"""Test whether a growing harness breaks more of what already worked.

    python3 deploy/audit_regressions.py runs/h-only-v3

The claim under test is that each accepted edit enlarges the instruction set and the
control flow, and that the additions interfere with what is already there -- rules that
conflict, or emphasis that gets diluted. That claim makes a prediction the data can
check: as the harness grows, a larger share of the tasks the previous generation
answered correctly should break.

Two competing explanations predict the opposite or nothing:
  * headroom exhaustion -- the easy failures are fixed, so later edits simply add
    nothing; they should not actively break working tasks
  * draw variance -- regressions should stay flat, not trend with size

So the discriminating measurement is the regression rate per generation pair, on the
same 90 held-out tasks, plotted against harness size. Absolute counts will not do: a
generation that answers more tasks correctly has more tasks that can break, so the rate
is normalised by the number correct before the edit.

Section 2 lists the tasks that broke and never recovered, which is where a genuine rule
conflict would show up as a specific, repeatable casualty.

Read-only, pure standard library. Requires deploy/rerun_on_test.sh to have filled in the
intermediate generations.
"""

import argparse
import collections
import json
import pathlib
import sys

BAR = "=" * 76


def load(folder):
    out = {}
    if folder.is_dir():
        for path in sorted(folder.rglob("episode.json")):
            try:
                episode = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            out[episode["task_id"]] = episode["reward"] == 1
    return out


def harness_size(root, step, picked):
    path = root / step / f"candidate-{picked:02d}" / "harness.py"
    return len(path.read_text().splitlines()) if path.is_file() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--phase", default="h0001")
    ap.add_argument("--initial", default="src/simple_scibuddy/harness/scientific.py")
    ap.add_argument("--release", default=None,
                    help="release directory; enables the reachable / no-lookup split")
    args = ap.parse_args()

    root = pathlib.Path(args.run) / "harness_evolve" / args.phase
    if not root.is_dir():
        sys.exit(f"cannot find {root}")

    # Build the chain of distinct harnesses. A rejected step leaves the harness unchanged,
    # so it contributes no new generation.
    chain = [("H0", load(root / "baseline"),
              len(pathlib.Path(args.initial).read_text().splitlines()))]
    for step in sorted(root.glob("step-*")):
        update = step / "update.json"
        if not update.is_file():
            continue
        data = json.loads(update.read_text())
        if data.get("status") != "applied":
            continue
        index = step.name.split("-")[1].lstrip("0") or "0"
        name = f"H{index}"
        scores = load(root / "attribution" / name)
        if not scores:
            scores = load(step / "evaluation")
        if not scores:
            print(f"  [{name} has no test measurement; run deploy/rerun_on_test.sh]",
                  file=sys.stderr)
            continue
        chain.append((name, scores, harness_size(root, step.name, data["selected_candidate"])))

    if len(chain) < 3:
        sys.exit("need at least two measured generations beyond H0")

    print(BAR)
    print("[1] does a bigger harness break more of what already worked")
    print(BAR)
    print(f"\n  {'pair':14}{'lines':>13}{'right before':>14}{'fixed':>7}{'broke':>7}"
          f"{'broke rate':>12}{'net':>6}")
    rows = []
    for (a_name, a, a_size), (b_name, b, b_size) in zip(chain, chain[1:]):
        shared = sorted(set(a) & set(b))
        right = sum(a[t] for t in shared)
        fixed = sum(1 for t in shared if not a[t] and b[t])
        broke = sum(1 for t in shared if a[t] and not b[t])
        rate = broke / right if right else 0.0
        rows.append((f"{a_name}->{b_name}", a_size, b_size, right, fixed, broke, rate))
        print(f"  {a_name + ' -> ' + b_name:14}{str(a_size) + '->' + str(b_size):>13}"
              f"{right:14}{fixed:7}{broke:7}{rate:11.0%}{fixed - broke:+6}")

    sizes = [b for _, _, b, _, _, _, _ in rows]
    rates = [r for *_, r in rows]
    if len(rows) >= 3:
        n = len(rows)
        mx, my = sum(sizes) / n, sum(rates) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(sizes, rates))
        vx = sum((x - mx) ** 2 for x in sizes) ** 0.5
        vy = sum((y - my) ** 2 for y in rates) ** 0.5
        r = cov / (vx * vy) if vx and vy else 0.0
        print(f"\n  correlation between harness size and regression rate: r = {r:+.2f}"
              f"  (n = {n} pairs)")
        print("  A positive r is what the interference claim predicts; with this few pairs it "
              "is\n  suggestive at best, never conclusive.")

    if args.release:
        # Splitting by reachability separates two effects that both lower the rate.
        # Where nothing can be looked up, a correct answer is always a guess, so the
        # grounding of answers cannot improve there and the rate measures only how much
        # each edit perturbs the model. Where lookups are possible, a falling rate can
        # equally mean answers became evidence-backed rather than lucky.
        DEAD = {"scientific_literature_reading", "variant_from_sequence_task",
                "variant_multi_sequence_task", "experimental_protocol_troubleshooting"}
        release = pathlib.Path(args.release)
        manifest = json.loads((release / "manifest.json").read_text())
        subtask = {}
        for row in manifest["tasks"]:
            try:
                subtask[row["id"]] = json.loads(
                    (release / row["id"] / "public/task.json").read_text()).get("subtask")
            except (OSError, ValueError):
                subtask[row["id"]] = None

        print("\n" + BAR)
        print("[1b] the same rate, split by whether a harness can reach the task")
        print(BAR)
        print(f"\n  {'pair':14}" + f"{'reachable':>26}" + f"{'no lookup possible':>26}")
        print(f"  {'':14}{'right':>8}{'broke':>8}{'rate':>10}{'right':>8}{'broke':>8}{'rate':>10}")
        series = {True: [], False: []}
        for (a_name, a, _), (b_name, b, b_size) in zip(chain, chain[1:]):
            cells = ""
            for reachable in (True, False):
                shared = [t for t in set(a) & set(b)
                          if (subtask.get(t) not in DEAD) == reachable]
                right = sum(a[t] for t in shared)
                broke = sum(1 for t in shared if a[t] and not b[t])
                rate = broke / right if right else 0.0
                series[reachable].append((b_size, rate))
                cells += f"{right:8}{broke:8}{rate:9.0%} "
            print(f"  {a_name + ' -> ' + b_name:14}{cells}")

        print()
        for reachable, label in ((True, "reachable"), (False, "no lookup possible")):
            pts = series[reachable]
            n = len(pts)
            mx = sum(x for x, _ in pts) / n
            my = sum(y for _, y in pts) / n
            cov = sum((x - mx) * (y - my) for x, y in pts)
            vx = sum((x - mx) ** 2 for x, _ in pts) ** 0.5
            vy = sum((y - my) ** 2 for _, y in pts) ** 0.5
            r = cov / (vx * vy) if vx and vy else 0.0
            print(f"  {label:22} size vs rate: r = {r:+.2f}   "
                  f"rates {[f'{y:.0%}' for _, y in pts]}")
        print("\n  If the no-lookup rate stays flat while the reachable rate falls, the fall is")
        print("  answers becoming evidence-backed. If the no-lookup rate rises with size, larger")
        print("  edits perturb the model more, which is what the interference claim needs.")

    print("\n" + BAR)
    print("[2] tasks that broke and never came back")
    print(BAR)
    names = [name for name, _, _ in chain]
    everything = set.intersection(*(set(scores) for _, scores, _ in chain))
    casualties = []
    for task in sorted(everything):
        series = [scores[task] for _, scores, _ in chain]
        if series[0] and not series[-1]:
            first_loss = next(i for i, ok in enumerate(series) if not ok)
            casualties.append((task, series, names[first_loss]))
    print(f"\n  {len(casualties)} task(s) correct at H0 and wrong at the end\n")
    print(f"  {'task':26}" + "".join(f"{n:>6}" for n in names) + "   broke at")
    for task, series, where in casualties:
        marks = "".join(f"{('ok' if ok else '--'):>6}" for ok in series)
        print(f"  {task:26}{marks}   {where}")

    flapping = []
    for task in sorted(everything):
        series = [scores[task] for _, scores, _ in chain]
        changes = sum(1 for x, y in zip(series, series[1:]) if x != y)
        if changes >= 2:
            flapping.append((task, series, changes))
    print(f"\n  {len(flapping)} task(s) changed answer correctness more than once "
          "(unstable across edits)")
    for task, series, changes in flapping[:10]:
        marks = "".join(f"{('ok' if ok else '--'):>6}" for ok in series)
        print(f"  {task:26}{marks}   {changes} flips")
    print()


if __name__ == "__main__":
    main()
