#!/usr/bin/env python3
"""Use the measured wall time of the round that already finished to extrapolate how
long a scaled-up run would take.

    python3 deploy/estimate_runtime.py runs/h-only-01

Why this cannot simply be scaled in proportion to the number of tasks:
  * server.py:27 pins --max-num-seqs 1 (paired evaluation demands token-by-token
    reproducibility), so vLLM handles exactly one sequence at a time and every model
    call on the GPU is strictly serial. workers=16 only queues them up; it does not
    speed up model inference.
  * An evolved harness makes more calls per task (4.0 -> 7.4 in this round), and a
    scaled-up experiment spends most of its time on the "already evolved" harness, so
    using the average over the whole run underestimates it.
  * Candidate proposals go through the improver API (over the network), which is
    unrelated to the GPU and can overlap with inference.

So the model is "total model calls x per-call latency", and two estimates are given
separately: optimistic (using the baseline's calls per task) and conservative (using
the final harness's calls per task).

Episode count per harness phase (phase.py:198-300, selection.py:28-35):
    1                  initial preflight
  + T                  baseline on Test
  + S x [ F            training interaction per step
        + C            one preflight per candidate
        + (1+C) x V ]  parent and candidates each run a full Val
  + T                  evaluation of the last step on Test
"""

import argparse
import json
import pathlib
import sys

BAR = "=" * 74


def episodes_under(folder):
    out = []
    for path in folder.rglob("episode.json"):
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            pass
    return out


def count_episodes(steps, feedback, candidates, val, test):
    """Return (total episode count, breakdown)."""
    per_step = feedback + candidates + (1 + candidates) * val
    parts = {
        "initial preflight": 1,
        "baseline (Test)": test,
        f"{steps} steps x [interaction {feedback} + candidate preflight {candidates} + "
        f"({1}+{candidates})x{val} validation]": steps * per_step,
        "final evaluation (Test)": test,
    }
    return 1 + test + steps * per_step + test, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="runs/h-only-01")
    ap.add_argument("--phase", default="h0001")
    args = ap.parse_args()

    root = pathlib.Path(args.run) / "harness_evolve" / args.phase
    if not root.is_dir():
        sys.exit(f"cannot find {root}")

    eps = episodes_under(root)
    if not eps:
        sys.exit(f"no episode.json under {root}")

    total_calls = sum(len(e.get("calls", [])) for e in eps)
    total_seconds = sum(e.get("elapsed_seconds", 0.0) for e in eps)
    if not total_calls:
        sys.exit("no model calls in the episodes, cannot estimate")

    per_call = total_seconds / total_calls
    baseline = episodes_under(root / "baseline")
    finals = []
    for step in sorted(root.glob("step-*"), reverse=True):
        finals = episodes_under(step / "evaluation")
        if finals:
            break

    def mean_calls(group):
        return sum(len(e.get("calls", [])) for e in group) / len(group) if group else 0.0

    calls_h0 = mean_calls(baseline)
    calls_hstar = mean_calls(finals)

    print(BAR)
    print(f"measured ({root})")
    print(BAR)
    print(f"  episodes                     {len(eps)}")
    print(f"  total model calls            {total_calls}")
    print(f"  cumulative episode time      {total_seconds / 3600:.2f} hours")
    print(f"  average per model call       {per_call:.1f} seconds")
    print(f"  model calls per task  H0 {calls_h0:.2f}  ->  H* {calls_hstar:.2f}")
    print("  Note: --max-num-seqs 1, serial on the GPU; the extrapolation below "
          "assumes serial execution\n")

    # The structure of the round that finished, used to check the formula against the
    # actual directory counts
    steps = len(list(root.glob("step-*")))
    inter = episodes_under(root / "step-0001" / "interaction")
    val = len(episodes_under(root / "step-0001" / "validation-parent"))
    cands = len(list((root / "step-0001").glob("validation-[0-9][0-9]")))
    predicted, _ = count_episodes(steps, len(inter), cands, val, len(baseline))
    print(f"  formula check: S={steps} F={len(inter)} C={cands} V={val} T={len(baseline)}"
          f" -> predicted {predicted} episodes, actual {len(eps)}"
          f" ({'match' if predicted == len(eps) else 'mismatch, discount the estimates below'})\n")

    scenarios = [
        ("current (already finished)", steps, len(inter), cands, val, len(baseline)),
        ("small profile: V=30 T=30, 3 steps", 3, 8, 3, 30, 30),
        ("small profile: V=30 T=30, 5 steps", 5, 8, 3, 30, 30),
        ("scale Test only: V=10 T=90, 3 steps", 3, 8, 3, 10, 90),
        ("full profile: V=90 T=90, 3 steps", 3, 8, 3, 90, 90),
        ("paper protocol: 10 steps x 16-task interaction, V=90 T=90", 10, 16, 3, 90, 90),
    ]

    print(BAR)
    print("extrapolation")
    print(BAR)
    print(f"  {'scenario':58} {'episode':>8} {'optimistic':>12} {'conservative':>12}")
    print(f"  {'':58} {'':>8} {'(H0 calls)':>12} {'(H* calls)':>12}")
    for name, s, f, c, v, t in scenarios:
        n, _ = count_episodes(s, f, c, v, t)
        low = n * calls_h0 * per_call / 3600
        high = n * calls_hstar * per_call / 3600
        print(f"  {name:58} {n:8} {low:11.1f}h {high:11.1f}h")

    print(f"\n  Optimistic and conservative differ only in model calls per task "
          f"({calls_h0:.1f} vs {calls_hstar:.1f}).")
    print("  The true value leans conservative: the evolved harness stays in use until "
          "the experiment ends.")
    print("  Not counted: the improver's proposal API round trips (over the network, can")
    print("          overlap with GPU inference), vLLM startup, and Slurm queueing time.\n")


if __name__ == "__main__":
    main()
