#!/usr/bin/env python3
"""把进化过程中的中间 harness 补测到 test split 上，用来定位改进发生在哪一步。

    bash deploy/rerun_on_test.sh configs/h-only-small.toml runs/h-only-small-01

原实验只在 test 上测了 H0（baseline）与最终选中的那个（step-NNNN/evaluation），
中间世代只有 val 分数。而 val 与 test 的子任务构成几乎不重叠，所以 val 上的
逐步曲线无法回答"test 那次跃升来自哪一步"。

本脚本直接调用 coevolve.phase.batch —— 与原实验同一个函数，因此 task seed
（digest([cfg.seed, task_id])）、temperature 0、actions / context / tool 预算
全部逐字节一致，结果可以和 baseline、evaluation 并排比较。

不跑 improver、不跑 reviewer（user_turns=0），只做单次作答评测。
输出写进 <run>/harness_evolve/<phase>/attribution/H<N>/，不触碰原有目录。
"""

import argparse
import asyncio
import collections
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

from simple_scibuddy.artifacts import ROOT, write_json
from simple_scibuddy.configuration import load_config, settings
from simple_scibuddy.coevolve.phase import batch, summarize
from simple_scibuddy.data.dataset import TaskDataset
from simple_scibuddy.inference.server import serve_pool

BAR = "=" * 74


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


def stages(root):
    """[(世代名, harness 路径)]；最后一步的选中程序已有 evaluation，跳过。"""
    steps = sorted(root.glob("step-*"))
    out = []
    for i, step in enumerate(steps, start=1):
        update = step / "update.json"
        if not update.is_file():
            continue
        picked = json.loads(update.read_text()).get("selected_candidate")
        if picked is None:
            continue
        harness = step / f"candidate-{picked:02d}" / "harness.py"
        measured = (step / "evaluation").is_dir()
        out.append((f"H{i}", harness, measured, step.name, picked))
    return out


def load_folder(folder):
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
    ap.add_argument("config", help="产生该实验的 toml，保证预算与 seed 一致")
    ap.add_argument("run")
    ap.add_argument("--phase", default="h0001")
    ap.add_argument("--only", default=None, help="逗号分隔的世代名，如 H1,H2")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        sys.exit("需要 CUDA_VISIBLE_DEVICES；请用 deploy/rerun_on_test.sh 启动")

    root = resolve(args.run) / "harness_evolve" / args.phase
    if not root.is_dir():
        sys.exit(f"找不到 {root}")

    config = load_config(resolve(args.config))
    base = settings()
    # 逐字复制 loop.py:57-61 的组装顺序（toml 覆盖前两项），这样 seed、actions、
    # context/tool 预算与原实验完全一致 —— 任一不同，补测结果就不能和 baseline
    # 及 evaluation 并排比较。
    cfg = {
        **base,
        "workers": 8,
        "user_turns": 1,
        **{k: v for k, v in config["harness_evolve"].items() if k != "improver"},
    }
    # 展开之后再覆盖：补测不接 reviewer。phase.py:104-105 在 reviewer=None 时
    # 本就会把 user_turns 归零，这里写明只为让意图可读。
    cfg = dict(cfg, model=base["model"], user_turns=0)
    if cfg.get("seed") is None:
        sys.exit("配置里没有 seed；task seed 对不上就失去了可比性")

    dataset = TaskDataset(cfg["dataset"])
    rows = dataset.tasks("test")

    todo = [(name, h, step, pick) for name, h, measured, step, pick in stages(root)
            if not measured and h.is_file()]
    if args.only:
        wanted = {w.strip() for w in args.only.split(",")}
        todo = [t for t in todo if t[0] in wanted]
    if not todo:
        sys.exit("没有需要补测的世代（最后一步已有 evaluation）")

    print(BAR)
    print(f"补测 {len(todo)} 个世代 x {len(rows)} 道 test 题 = {len(todo) * len(rows)} 个 episode")
    for name, harness, step, pick in todo:
        print(f"  {name}  <- {step}/candidate-{pick:02d}/harness.py")
    print(BAR, flush=True)

    from transformers import AutoTokenizer

    model = str(resolve(cfg["model"]))
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    out_root = root / "attribution"
    out_root.mkdir(parents=True, exist_ok=True)

    with ExitStack() as services:
        url = services.enter_context(
            serve_pool(model, devices, out_root / "solver",
                       max(cfg["context_tokens"], 32768), share_last=False)
        )
        for name, harness, _, _ in todo:
            folder = out_root / name
            if folder.is_dir() and load_folder(folder):
                print(f"\n{name} 已存在，跳过", flush=True)
                continue
            print(f"\n--- {name} ---", flush=True)
            episodes = asyncio.run(batch(dataset, rows, harness, url, tokenizer, cfg, folder))
            stats = summarize(episodes)
            write_json(folder / "summary.json", {"harness": str(harness), **stats})
            print(f"  {name}: {stats['correct']}/{stats['tasks']} = {stats['accuracy']:.1%}", flush=True)

    # ---- 汇总：把 H0 / 中间世代 / H* 并排，并按 subtask 拆开 ----------------
    series = [("H0", load_folder(root / "baseline"))]
    for name, _, _, _ in todo:
        series.append((name, load_folder(out_root / name)))
    for step in sorted(root.glob("step-*"), reverse=True):
        final = load_folder(step / "evaluation")
        if final:
            series.append(("H* (最终)", final))
            break
    series = [(n, e) for n, e in series if e]

    subtask = {}
    for row in dataset.manifest["tasks"]:
        if row["split"] != "test":
            continue
        try:
            subtask[row["id"]] = json.loads(
                (dataset.root / row["id"] / "public/task.json").read_text()).get("subtask")
        except (OSError, ValueError):
            subtask[row["id"]] = None

    print("\n" + BAR)
    print("test 上的逐世代结果")
    print(BAR)
    for name, eps in series:
        correct = sum(e["reward"] == 1 for e in eps)
        print(f"  {name:12} {correct:3}/{len(eps)} = {correct / len(eps):5.1%}")

    print("\n" + BAR)
    print("按 subtask 拆开 —— 跃升发生在哪一世代、哪类题")
    print(BAR)
    keys = sorted({subtask.get(e["task_id"]) or "?" for _, eps in series for e in eps})
    print("  " + f"{'subtask':40}" + "".join(f"{n:>12}" for n, _ in series))
    for key in keys:
        cells = ""
        for _, eps in series:
            hit = sum(e["reward"] == 1 for e in eps if (subtask.get(e["task_id"]) or "?") == key)
            n = sum(1 for e in eps if (subtask.get(e["task_id"]) or "?") == key)
            cells += f"{(f'{hit}/{n}' if n else '-'):>12}"
        print(f"  {key[:40]:40}{cells}")
    print("\n  同一批题、同样的 seed 与温度 0，各列可直接比较。\n")


if __name__ == "__main__":
    main()
