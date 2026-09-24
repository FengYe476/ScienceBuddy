#!/usr/bin/env python3
"""核对 val / test 两个 split 是否可交换，以及各自由哪些 subtask 构成。

    python3 deploy/split_audit.py data/releases/id200-val30-test30-v1
    python3 deploy/split_audit.py data/releases/id200-val30-test30-v1 --run runs/h-only-small-01

为什么要查：build_release.py:184 的 split_by_group 按 source_group 整组分配，
而 DbQA 的 source_group 是 f"dbqa:{subtask}" —— 一个子任务就是一个组，DbQA
只有 10 个组。于是 test 拿到整块的一两个子任务、val 拿到另外一两个，两个
split 的难度不再可交换。h-only-small-01 里同一份 harness 在 val 上 15/30、
在 test 上 26/30，比值在 H0（4/30 vs 8/30）时同样约 0.5 —— 像是 split 属性
而非程序差异，本脚本用来确认。

给 --run 时，额外按 subtask 拆开 baseline 与最终评估的正确率，看涨幅集中
在哪些子任务上。只读，纯标准库。
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
        sys.exit(f"找不到 {release}/manifest.json")
    tasks = read_tasks(release)

    print(BAR)
    print("各 split 的 family 构成")
    print(BAR)
    fam = collections.defaultdict(collections.Counter)
    for t in tasks:
        fam[t["split"]][t["family"]] += 1
    for split in ("train", "val", "test"):
        total = sum(fam[split].values())
        share = "  ".join(f"{k}={v}" for k, v in sorted(fam[split].items()))
        print(f"  {split:6} {total:4}   {share}")

    print("\n" + BAR)
    print("各 split 的 subtask 构成 —— 交集为空就说明两个 split 不可交换")
    print(BAR)
    sub = collections.defaultdict(collections.Counter)
    for t in tasks:
        sub[t["split"]][t["subtask"] or "(无)"] += 1
    for split in ("val", "test"):
        print(f"\n  [{split}]")
        for name, n in sub[split].most_common():
            print(f"      {n:3}  {name}")
    shared = set(sub["val"]) & set(sub["test"])
    only_val = set(sub["val"]) - set(sub["test"])
    only_test = set(sub["test"]) - set(sub["val"])
    print(f"\n  共有 subtask {len(shared)} 个: {sorted(shared) or '无'}")
    print(f"  仅 val  {len(only_val)} 个: {sorted(only_val) or '无'}")
    print(f"  仅 test {len(only_test)} 个: {sorted(only_test) or '无'}")
    if not shared:
        print("\n  !! val 与 test 没有任何共同 subtask —— 两者测的不是同一件事，")
        print("     在 val 上做选择、在 test 上报结果，等于换了一份考卷。")

    print("\n" + BAR)
    print("source_group 重叠（experiments.md:127 记录论文为 Train-Val 20 组）")
    print(BAR)
    groups = collections.defaultdict(set)
    for t in tasks:
        groups[t["split"]].add(t["source_group"])
    print(f"  Train-Val  {len(groups['train'] & groups['val']):3} 组")
    print(f"  Train-Test {len(groups['train'] & groups['test']):3} 组")
    print(f"  Val-Test   {len(groups['val'] & groups['test']):3} 组")

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
        print("\n（没有完整的 baseline / evaluation，跳过分 subtask 的对照）")
        return

    print("\n" + BAR)
    print("test 上按 subtask 的 H0 -> H*  —— 涨幅集中在哪")
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
    print(f"  {'subtask':46} {'H0':>6} {'H*':>6} {'题数':>5}")
    for key, (h0, hs, n) in sorted(agg.items(), key=lambda kv: -(kv[1][1] - kv[1][0])):
        print(f"  {key[:46]:46} {h0:6} {hs:6} {n:5}")

    # ---- 中间步骤的免费测量 --------------------------------------------
    # phase.py 每步开头用的是上一步选中的 harness，所以 step-N/interaction
    # 跑的是 H(N-1)。题目每步不同（cursor 往后滚），准确率不可比，但能看出
    # 各子任务在哪一代出现、表现如何 —— 不用重跑就能定位改进发生在哪一步。
    print("\n" + BAR)
    print("train 交互按 subtask（step-N/interaction 跑的是 H(N-1)）")
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
        print("  （没有 interaction 数据）")
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
    print(f"\n  与 test 共有的 subtask: {target or '无'}")
    print("  题目每步不同，准确率不可直接比；看的是某类题在哪一代开始出现、做对没有。")


if __name__ == "__main__":
    main()
