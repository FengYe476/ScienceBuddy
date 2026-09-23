#!/usr/bin/env python3
"""把一次 harness 进化跑出来的 episode.json 汇总成"任务模型遇到了什么阻碍"。

    python3 deploy/analyze_episodes.py runs/h-only-01

只读，不依赖 .venv（纯标准库），因此在登录节点上也能跑。

目录结构由 coevolve/phase.py 决定：
    <run>/harness_evolve/h0001/
        baseline/<task>/episode.json              H0 在 test 上（phase.py:200）
        step-000N/interaction/<task>/…            训练交互，带 reviewer 反馈
        step-000N/validation-parent/<task>/…      父程序在选择集上
        step-000N/validation-0M/<task>/…          候选 M 在选择集上（selection.py:34）
        step-000N/evaluation/<task>/…             最后一步的 H* 在 test 上（phase.py:291）

阻碍分成四层，对应 episode.json 里四个不同的证据来源：
    1. stop_reason        回合为什么结束（broker.py:174-182）
    2. 提交格式           <answer> 标签数 != 1 就判 0（data/verifier.py:9-10）
    3. 工具报错           tool_calls[].observation.error，按异常类型归并
    4. 剩余的科学错误     交了、格式对、但答案不对
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
    """中文在终端占两列，str.format 按码点算宽度，会把表格对歪。"""
    used = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - used)

# broker.py 里除 first_answer_evaluation 外都是没做完题
INCOMPLETE = {
    "no_submission": "没提交答案",
    "output_truncated": "单次输出被截断",
    "context_budget": "上下文耗尽",
    "model_call_budget": "模型调用耗尽",
    "tool_call_budget": "工具调用耗尽",
    "tool_timeout": "工具超时",
    "time_budget": "回合超时",
    "candidate_error": "harness 自身崩溃",
    "execution_memory_budget": "执行容器内存耗尽",
    "first_answer_evaluation": "正常提交",
}


def load(folder):
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*/episode.json")):
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            print(f"  [跳过 {path}: {exc}]", file=sys.stderr)
    return out


def error_kind(text):
    """从 traceback 末行抽出异常类型；网络错误单独归类，因为那是环境隔离导致的。"""
    tail = text.strip().splitlines()[-1] if text.strip() else "(空)"
    kind = re.split(r"[:(]", tail)[0].strip() or tail[:40]
    low = text.lower()
    if "errno 101" in low or "unreachable" in low or "temporary failure in name resolution" in low:
        return "网络不可达（容器已断网）"
    if "no module named" in low:
        missing = re.search(r"[Nn]o module named ['\"]?([\w.]+)", text)
        return f"缺少模块 {missing[1]}" if missing else "缺少模块"
    return kind[:44]


def answer_tags(episode):
    """数最后一次提交里的 <answer> 标签，复现 verifier.py 的判定。"""
    subs = episode.get("submissions") or []
    if not subs:
        return None
    return len(re.findall(r"<answer>\s*(.*?)</answer>", subs[-1]["text"], re.S | re.I))


def report(name, episodes):
    print(BAR)
    print(f"{name}    {len(episodes)} 题")
    print(BAR)
    if not episodes:
        print("  （没有 episode）\n")
        return

    n = len(episodes)
    correct = sum(e["reward"] == 1 for e in episodes)
    print(f"  正确 {correct}/{n} = {correct / n:.0%}\n")

    # ---- 1. 回合为什么结束 -------------------------------------------------
    print("  [1] 回合结束原因")
    for reason, count in collections.Counter(e["stop_reason"] for e in episodes).most_common():
        label = INCOMPLETE.get(reason, reason)
        print(f"      {pad(label, 24)}{count:3}  ({count / n:.0%})")

    # ---- 2. 提交与格式 -----------------------------------------------------
    nosub = [e for e in episodes if not e.get("submissions")]
    bad_format, good_format = [], []
    for e in episodes:
        tags = answer_tags(e)
        if tags is None:
            continue
        (good_format if tags == 1 else bad_format).append((e, tags))
    print("\n  [2] 提交与答案格式")
    print(f"      {pad('从未提交', 24)}{len(nosub):3}   <- 比 [1] 的 no_submission 多，"
          "截断/耗尽同样没走到提交")
    print(f"      {pad('提交了但标签数 != 1', 24)}{len(bad_format):3}   <- verifier.py:9 直接判 0")
    print(f"      {pad('提交且格式合法', 24)}{len(good_format):3}")
    for e, tags in bad_format[:3]:
        print(f"        {e['task_id']}  标签 {tags} 个")

    # ---- 3. 工具报错 -------------------------------------------------------
    errs = [
        (e, t)
        for e in episodes
        for t in e.get("tool_calls", [])
        if t["observation"].get("error") and not t["observation"].get("infrastructure_error")
    ]
    total_tools = sum(len(e.get("tool_calls", [])) for e in episodes)
    print(f"\n  [3] 工具调用 {total_tools} 次，报错 {len(errs)} 次"
          f"（{len(errs) / max(1, total_tools):.0%}）")
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
                print(f"      | … 共 {len(code.splitlines())} 行")
            print(f"      => {tail[:160]}")

    # 同一题里反复撞同一个错 = improver 假设 2(c) 的"重复失败"
    repeats = 0
    for e in episodes:
        seen = collections.Counter(
            error_kind(t["observation"]["error"])
            for t in e.get("tool_calls", [])
            if t["observation"].get("error")
        )
        repeats += sum(v - 1 for v in seen.values() if v > 1)
    if repeats:
        print(f"      其中 {repeats} 次是在同一题里重复撞同一类错误")

    # ---- 4. 交了、格式对、仍然错 -------------------------------------------
    science = [e for e, _ in good_format if e["reward"] != 1]
    print(f"\n  [4] 提交合法但答案错误   {len(science):3}   <- 真正的科学能力差距")
    # 总准确率混了两件事：交没交，和交了之后对不对。只有后者是科学能力。
    attempted = len(good_format)
    if attempted:
        print(f"      提交后命中率 {correct}/{attempted} = {correct / attempted:.0%}"
              "   <- 与总准确率分开看，前者才随能力变化")

    # ---- 失败样本 ---------------------------------------------------------
    stuck = nosub or [e for e in episodes if e["stop_reason"] in INCOMPLETE and e["stop_reason"] != "first_answer_evaluation"]
    if stuck:
        print(f"\n  没答完的题（前 3 个，看它卡在哪）:")
        for e in stuck[:3]:
            calls = e.get("calls") or []
            tail = calls[-1]["text"][-200:].replace("\n", " ⏎ ") if calls else "(零次模型调用)"
            print(f"      {e['task_id']}  stop={e['stop_reason']}  "
                  f"模型调用 {len(calls)}  工具调用 {len(e.get('tool_calls', []))}")
            print(f"        最后输出 …{tail}")
    print()


def transitions(before, after, label_a, label_b):
    a = {e["task_id"]: e for e in before}
    b = {e["task_id"]: e for e in after}
    shared = sorted(set(a) & set(b))
    if not shared:
        return
    print(BAR)
    print(f"逐题变化   {label_a} -> {label_b}")
    print(BAR)
    tally = collections.Counter()
    for tid in shared:
        x, y = int(a[tid]["reward"] == 1), int(b[tid]["reward"] == 1)
        mark = {(0, 1): "变对 ++", (1, 0): "变错 --", (1, 1): "保持对", (0, 0): "仍然错"}[(x, y)]
        tally[mark] += 1
        print(f"  {tid:26} {pad(mark, 10)}{a[tid]['stop_reason']:24} -> {b[tid]['stop_reason']}")
    print(f"\n  变对 {tally['变对 ++']} · 变错 {tally['变错 --']} · "
          f"保持对 {tally['保持对']} · 仍然错 {tally['仍然错']}")
    net = tally["变对 ++"] - tally["变错 --"]
    print(f"  净增 {net:+d} 题 —— 汇总的准确率差值掩盖了 {tally['变错 --']} 道退步的题\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="runs/h-only-01", help="实验目录")
    ap.add_argument("--phase", default="h0001")
    ap.add_argument("--show-errors", type=int, default=0, metavar="N",
                    help="每段额外打印 N 个报错工具调用的源码与异常，用来判断模型到底在找什么")
    args = ap.parse_args()
    global SHOW_ERRORS
    SHOW_ERRORS = args.show_errors

    root = pathlib.Path(args.run) / "harness_evolve" / args.phase
    if not root.is_dir():
        found = sorted(str(p) for p in pathlib.Path(args.run).glob("*/*")) \
            if pathlib.Path(args.run).is_dir() else []
        sys.exit(f"找不到 {root}" + ("\n实际存在: " + ", ".join(found) if found else ""))

    steps = sorted(root.glob("step-*"))
    print(f"实验 {root}，共 {len(steps)} 步\n")

    baseline = load(root / "baseline")
    report("H0 基线（test 集）", baseline)

    for step in steps:
        inter = load(step / "interaction")
        if inter:
            report(f"{step.name} 训练交互（train 集，这是 improver 看到的证据）", inter)

    final = None
    for step in reversed(steps):
        final = load(step / "evaluation")
        if final:
            report(f"H* 最终（test 集，{step.name}）", final)
            break

    if baseline and final:
        transitions(baseline, final, "H0", "H*")

    # 进化过程中每一步选择集上的表现
    print(BAR)
    print("选择集（val）上父程序 vs 候选")
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
        print(f"  {step.name}  " + (" · ".join(row) if row else "（无验证数据）") + status)
    print()


if __name__ == "__main__":
    main()
