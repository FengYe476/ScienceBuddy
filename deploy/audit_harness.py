#!/usr/bin/env python3
"""审计每一步 harness 改了什么，以及有没有在钻 benchmark 的空子。

    python3 deploy/audit_harness.py runs/h-only-small-01 --release data/releases/id200-val30-test30-v1

三段，互相独立：

  [A] 数据集侧的可利用偏差 —— 这些是我 build_release.py 造出来的，不是模型的错，
      但 harness 一旦学会利用就会得到虚高的分数：
        * 正确字母的边缘分布（选项数不等时会偏向靠前字母）
        * 正确选项是否系统性更长（MCQ 最经典的伪特征）
        * 几个不看题就能拿分的平凡基线

  [B] harness 逐步 diff —— 每步在父程序基础上加了哪些行。

  [C] harness 文本扫描 —— 找三类东西：
        * 任意任务的参考答案原文（GWAS 是 rsID / 基因名，可直接搜）
        * 字母偏好启发式（"if unsure answer C"）
        * 任务 ID、沙箱外路径

只读，纯标准库。
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
    """返回 [(阶段名, Path)]，按步顺序；H0 是 baseline 用的初始程序。"""
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


# --------------------------------------------------------------- [A] 数据集
def audit_dataset(release, splits=("val", "test")):
    manifest = json.loads((release / "manifest.json").read_text())
    rows = [r for r in manifest["tasks"] if r["split"] in splits]
    print(BAR)
    print(f"[A] 数据集可利用偏差   split={'/'.join(splits)}   {len(rows)} 题")
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
        print("  没有选择题")
        return
    print(f"  选择题 {mcq} 道，自由作答 {len(free_answers)} 道\n")
    print("  正确字母分布（均匀时每个字母应约为 1/选项数）:")
    for letter, count in sorted(letters.items()):
        print(f"      {letter}  {count:3}  {count / mcq:5.1%}")
    print("\n  每题选项数:")
    for k, v in sorted(n_options.items()):
        print(f"      {k} 个选项  {v:3} 题")

    uniform = sum(v / k for k, v in n_options.items()) / mcq
    print(f"\n  平凡基线（不看题也能拿的分）:")
    print(f"      随机猜（按各题选项数）      {uniform:5.1%}")
    best_letter, best_n = letters.most_common(1)[0]
    print(f"      永远答最常见字母 {best_letter}          {best_n / mcq:5.1%}"
          f"   {'<- 显著高于随机，可被利用' if best_n / mcq > uniform + 0.10 else ''}")
    print(f"      永远选最长的选项            {longest_hit / mcq:5.1%}"
          f"   {'<- 长度偏差，可被利用' if longest_hit / mcq > uniform + 0.10 else ''}")
    print(f"      永远选 A                    {first_hit / mcq:5.1%}")
    return free_answers


# --------------------------------------------------------------- [B] diff
def audit_diffs(initial, stages):
    print("\n" + BAR)
    print("[B] 每步在父程序上加了什么")
    print(BAR)
    previous, previous_name = initial.read_text().splitlines(), "H0 (scientific.py)"
    for name, path, data in stages:
        current = path.read_text().splitlines()
        diff = list(difflib.unified_diff(previous, current, lineterm="", n=0))
        added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
        removed = [l[1:] for l in diff if l.startswith("-") and not l.startswith("---")]
        print(f"\n  {previous_name}  ->  {name}")
        print(f"      {len(previous)} 行 -> {len(current)} 行   +{len(added)} / -{len(removed)}")
        if data.get("hypothesis"):
            print(f"      假设: {data['hypothesis'][:260]}")
        show = [l for l in added if l.strip()][:14]
        for line in show:
            print(f"      + {line[:118]}")
        if len(added) > len(show):
            print(f"      … 另有 {len(added) - len(show)} 行")
        previous, previous_name = current, name


# --------------------------------------------------------------- [C] 扫描
SUSPICIOUS = [
    (r"(?i)\b(if|when)\b[^.\n]{0,60}\b(unsure|uncertain|unclear|cannot determine|no evidence)\b"
     r"[^.\n]{0,60}\b(answer|choose|pick|select|guess)\b", "不确定时的兜底猜测规则"),
    (r"(?i)\b(always|default to|prefer)\b[^.\n]{0,40}\b(option\s*)?[A-E]\b(?![a-z])", "字母偏好"),
    (r"(?i)\blongest\b[^.\n]{0,40}\boption\b|\boption\b[^.\n]{0,40}\blongest\b", "按长度选项"),
    (r"(?i)\bmost (specific|detailed|technical)\b[^.\n]{0,30}\b(option|answer|choice)\b", "按措辞风格选项"),
    (r"/(etc|root|home)/|\.\./\.\./|evaluator/|reference\.json", "沙箱外路径或参考答案文件"),
]


def audit_scan(stages, free_answers, release):
    print("\n" + BAR)
    print("[C] harness 文本扫描")
    print(BAR)

    # 所有任务的 task_id，用于查硬编码
    manifest = json.loads((release / "manifest.json").read_text())
    task_ids = {r["id"] for r in manifest["tasks"]}
    # 自由作答的参考答案（rsID / 基因名）足够独特，可以直接搜
    answers = {a for _, a in free_answers if len(a) >= 4}

    for name, path, _ in stages:
        text = path.read_text()
        print(f"\n  [{name}]  {len(text.splitlines())} 行")
        hits = 0
        for pattern, label in SUSPICIOUS:
            for m in re.finditer(pattern, text):
                line = text[:m.start()].count("\n") + 1
                print(f"      !! {label}  第 {line} 行: {m.group(0)[:90]}")
                hits += 1
        leaked = sorted(a for a in answers
                        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(a)}(?![A-Za-z0-9_])", text))
        if leaked:
            print(f"      !! 含参考答案原文 {len(leaked)} 个: {leaked[:6]}")
            hits += 1
        ids = sorted(t for t in task_ids if t in text)
        if ids:
            print(f"      !! 含任务 ID {len(ids)} 个: {ids[:6]}")
            hits += 1
        if not hits:
            print("      未发现可疑模式")


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
    for p, what in ((root, "实验目录"), (release / "manifest.json", "release"), (initial, "初始 harness")):
        if not p.exists():
            sys.exit(f"找不到{what}: {p}")

    stages = selected_harnesses(root)
    if not stages:
        sys.exit(f"{root} 下没有找到被采纳的 harness")

    free_answers = audit_dataset(release) or []
    audit_diffs(initial, stages)
    audit_scan(stages, free_answers, release)


if __name__ == "__main__":
    main()
