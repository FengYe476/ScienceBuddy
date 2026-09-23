#!/usr/bin/env python3
"""用已完成那轮的实测耗时，外推放大规模后要跑多久。

    python3 deploy/estimate_runtime.py runs/h-only-01

为什么不能简单按题数等比放大：
  * server.py:27 固定 --max-num-seqs 1（配对评估要求逐 token 可复现），
    所以 vLLM 一次只处理一条序列，GPU 上所有模型调用严格串行。
    workers=16 只是在排队，并不给模型推理提速。
  * 进化后的 harness 每题调用次数会涨（本轮 4.0 -> 7.4），放大后的实验
    大部分时间花在"已进化"的 harness 上，用全程平均会低估。
  * 候选提案走 improver API（网络），与 GPU 无关，可与推理重叠。

因此按"总模型调用次数 x 单次调用延迟"建模，并分别给出乐观（用 baseline
的每题调用数）和保守（用最终 harness 的每题调用数）两个估计。

每个 harness phase 的 episode 数（phase.py:198-300、selection.py:28-35）：
    1                  初始 preflight
  + T                  baseline 在 Test 上
  + S x [ F            每步的训练交互
        + C            每个候选一次 preflight
        + (1+C) x V ]  父程序与候选各跑一遍完整 Val
  + T                  最后一步在 Test 上的评估
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
    """返回 (总 episode 数, 分项)。"""
    per_step = feedback + candidates + (1 + candidates) * val
    parts = {
        "初始 preflight": 1,
        "baseline (Test)": test,
        f"{steps} 步 x [交互 {feedback} + 候选 preflight {candidates} + "
        f"({1}+{candidates})x{val} 验证]": steps * per_step,
        "最终评估 (Test)": test,
    }
    return 1 + test + steps * per_step + test, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="runs/h-only-01")
    ap.add_argument("--phase", default="h0001")
    args = ap.parse_args()

    root = pathlib.Path(args.run) / "harness_evolve" / args.phase
    if not root.is_dir():
        sys.exit(f"找不到 {root}")

    eps = episodes_under(root)
    if not eps:
        sys.exit(f"{root} 下没有 episode.json")

    total_calls = sum(len(e.get("calls", [])) for e in eps)
    total_seconds = sum(e.get("elapsed_seconds", 0.0) for e in eps)
    if not total_calls:
        sys.exit("episode 里没有模型调用，无法估算")

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
    print(f"实测（{root}）")
    print(BAR)
    print(f"  episode 数              {len(eps)}")
    print(f"  模型调用总数            {total_calls}")
    print(f"  episode 耗时累计        {total_seconds / 3600:.2f} 小时")
    print(f"  单次模型调用平均        {per_call:.1f} 秒")
    print(f"  每题模型调用  H0 {calls_h0:.2f}  ->  H* {calls_hstar:.2f}")
    print("  注：--max-num-seqs 1，GPU 上串行，下面按串行外推\n")

    # 已完成这轮的结构，用来核对公式是否与目录数一致
    steps = len(list(root.glob("step-*")))
    inter = episodes_under(root / "step-0001" / "interaction")
    val = len(episodes_under(root / "step-0001" / "validation-parent"))
    cands = len(list((root / "step-0001").glob("validation-[0-9][0-9]")))
    predicted, _ = count_episodes(steps, len(inter), cands, val, len(baseline))
    print(f"  公式核对：S={steps} F={len(inter)} C={cands} V={val} T={len(baseline)}"
          f" -> 预测 {predicted} 个 episode，实际 {len(eps)} 个"
          f"（{'一致' if predicted == len(eps) else '不一致，下面的估计要打折扣'}）\n")

    scenarios = [
        ("当前（已跑完）", steps, len(inter), cands, val, len(baseline)),
        ("small profile：V=30 T=30，3 步", 3, 8, 3, 30, 30),
        ("small profile：V=30 T=30，5 步", 5, 8, 3, 30, 30),
        ("只放大 Test：V=10 T=90，3 步", 3, 8, 3, 10, 90),
        ("full profile：V=90 T=90，3 步", 3, 8, 3, 90, 90),
        ("论文协议：10 步 x 16 题交互，V=90 T=90", 10, 16, 3, 90, 90),
    ]

    print(BAR)
    print("外推")
    print(BAR)
    print(f"  {'方案':38} {'episode':>8} {'乐观':>9} {'保守':>9}")
    print(f"  {'':38} {'':>8} {'(H0 调用数)':>9} {'(H* 调用数)':>9}")
    for name, s, f, c, v, t in scenarios:
        n, _ = count_episodes(s, f, c, v, t)
        low = n * calls_h0 * per_call / 3600
        high = n * calls_hstar * per_call / 3600
        print(f"  {name:38} {n:8} {low:8.1f}h {high:8.1f}h")

    print(f"\n  乐观/保守的差别只在每题模型调用数（{calls_h0:.1f} vs {calls_hstar:.1f}）。")
    print("  真实值偏保守：进化后的 harness 会一直用到实验结束。")
    print("  未计入：improver 提案的 API 往返（走网络，与 GPU 推理可重叠）、")
    print("          vLLM 启动、以及 Slurm 排队时间。\n")


if __name__ == "__main__":
    main()
