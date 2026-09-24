#!/usr/bin/env python3
"""把一条 episode 的完整轨迹摊开，用来判断答案是从哪来的。

    python3 deploy/show_episode.py runs/h-only-small-01 --stage step-0003/evaluation
    python3 deploy/show_episode.py runs/h-only-small-01 --stage step-0003/evaluation --task gwas-0255fd96bde0
    python3 deploy/show_episode.py runs/h-only-small-01 --harness

准确率异常高时，要回答的是"模型凭什么答对"。能区分三种情况：
  * 题面里就有答案      -> public_task.prompt 里出现参考答案
  * 工具查出来的        -> 某次 tool 观测里出现答案，且时间早于提交
  * harness 直接给的    -> 系统提示或 harness 源码里出现答案
只读，纯标准库。
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
    return text if len(text) <= n else text[:n] + f" …[共 {len(text)} 字符]"


def show_harness(root):
    """列出每一步被采纳的 harness，并给出与父程序的行数差。"""
    for step in sorted(root.glob("step-*")):
        update = step / "update.json"
        if not update.is_file():
            continue
        data = load(update)
        picked = data.get("selected_candidate")
        print(f"\n{BAR}\n{step.name}  status={data.get('status')}  selected=c{picked}\n{BAR}")
        if data.get("hypothesis"):
            print("  假设:", clip(data["hypothesis"], 900))
        if data.get("reason"):
            print("  理由:", clip(data["reason"], 400))
        for cand in sorted(step.glob("candidate-*")):
            src = cand / "harness.py"
            if src.is_file():
                lines = src.read_text().splitlines()
                mark = " <- 被采纳" if picked is not None and cand.name.endswith(f"{picked:02d}") else ""
                print(f"    {cand.name}  {len(lines)} 行{mark}")


def find_answer_in(text, answer):
    """答案是否以独立词的形式出现在文本里（避免 'A' 命中任意字母）。"""
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
    print("\n[题面]")
    print(clip(prompt, 1500))
    if public.get("options"):
        print("\n[选项]")
        for i, o in enumerate(public["options"]):
            print(f"  {chr(65 + i)}. {clip(o, 160)}")

    submitted = episode["submissions"][-1]["text"] if episode.get("submissions") else ""
    outcome = episode["submissions"][-1]["outcome"] if episode.get("submissions") else {}
    graded = outcome.get("answer", "")

    print(f"\n[提交] {clip(submitted, 400)}")
    print(f"[评分] answer={graded!r}  score={outcome.get('score')}  "
          f"format_valid={outcome.get('answer_format_valid')}")

    # 答案出现在哪里 —— 这是判断泄漏的核心
    print("\n[答案溯源]")
    if not graded:
        print("  （无可溯源的答案）")
    else:
        print(f"  题面里出现: {find_answer_in(prompt, graded)}")
        for i, t in enumerate(episode.get("tool_calls", [])):
            obs = t["observation"]
            hit_out = find_answer_in(obs.get("stdout", ""), graded)
            hit_code = find_answer_in(t.get("code", ""), graded)
            if hit_out or hit_code:
                where = "工具输出" if hit_out else "模型写的代码"
                print(f"  tool[{i}] {where} 中出现（call_index={t.get('call_index')}）")

    print("\n[逐轮]")
    tools = episode.get("tool_calls", [])
    for i, call in enumerate(episode.get("calls", [])):
        print(f"\n  --- 模型调用 {i} ---")
        if i == 0:
            for m in call.get("messages", [])[:2]:
                print(f"  [{m['role']}] {clip(m['content'], 1200)}")
        print(f"  [assistant] {clip(call.get('text', ''), 1200)}")
        for t in tools:
            if t.get("call_index") == i:
                print(f"  [tool 代码] {clip(t.get('code', ''), 700)}")
                obs = t["observation"]
                if obs.get("error"):
                    print(f"  [tool 报错] {clip(obs['error'], 300)}")
                else:
                    print(f"  [tool 输出] {clip(obs.get('stdout', ''), 700)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--phase", default="h0001")
    ap.add_argument("--stage", default="step-0003/evaluation")
    ap.add_argument("--task", default=None, help="不给就列出该阶段所有题并挑第一道答对的")
    ap.add_argument("--harness", action="store_true", help="只看各步被采纳的 harness 与 improver 假设")
    args = ap.parse_args()

    root = pathlib.Path(args.run) / "harness_evolve" / args.phase
    if not root.is_dir():
        sys.exit(f"找不到 {root}")
    if args.harness:
        show_harness(root)
        return

    folder = root / args.stage
    if not folder.is_dir():
        sys.exit(f"找不到 {folder}")
    paths = sorted(folder.glob("*/episode.json"))
    if not paths:
        sys.exit(f"{folder} 下没有 episode")

    if args.task:
        picked = [p for p in paths if p.parent.name == args.task]
        if not picked:
            sys.exit(f"没有 {args.task}；可选: " + ", ".join(p.parent.name for p in paths[:8]))
    else:
        correct = [p for p in paths if load(p)["reward"] == 1]
        picked = correct[:1] or paths[:1]
        print(f"（未指定 --task，挑了 {picked[0].parent.name}；该阶段共 {len(paths)} 题，"
              f"答对 {len(correct)} 题）\n")
    show(load(picked[0]))


if __name__ == "__main__":
    main()
