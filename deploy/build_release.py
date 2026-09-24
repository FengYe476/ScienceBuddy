"""在任意机器上确定性重建 frozen task release（不依赖 adapter/）。

源数据与论文 Table S1 的对应（已核对）：

    任务族        论文   源                                     差异
    DbQA          511    futurehouse/lab-bench DbQA       520    -9
    GWAS          180    biomni/Eval1 四个 gwas_* 任务     193   -13
    ProtocolQA    108    futurehouse/lab-bench ProtocolQA 108   一致
    LitQA2         96    futurehouse/lab-bench LitQA2     199  -103

DbQA 的 10 个 subtask 与论文 Table S1 的 10 个子主题逐一对应。我们按同样的族
比例缩放，不追求数量完全相同。

所有随机性都由 seed 决定，且 rng 的消耗顺序固定（DbQA -> LitQA2 -> ProtocolQA），
所以同一 seed 在任何机器上产出逐字节相同的 release。

    python deploy/build_release.py --out data/releases/id40-val10-test10-v1
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

LAB_BENCH = "hf://datasets/futurehouse/lab-bench/{sub}/train-00000-of-00001.parquet"
EVAL1 = "hf://datasets/biomni/Eval1/biomni_eval1_dataset.parquet"

SPLIT_SEED = 20260911
FAMILY_WEIGHTS = {"DbQA": 511, "GWAS": 180, "ProtocolQA": 108, "LitQA2": 96}
PROFILES = {"debug": (40, 10, 10), "small": (200, 30, 30), "full": (715, 90, 90)}
LAKE_DIRNAME = "resources"
# data/verifier.py:25 对这个 subtask 走大小写敏感的精确匹配，名字必须一字不差
CASE_SENSITIVE_SUBTASK = "gwas_variant_prioritization"
PROMPT_TOKEN_LIMIT = 24576 - 4096

GWAS_TASKS = {
    "gwas_causal_gene_gwas_catalog": "gwas_causal_gene_gwas_catalog",
    "gwas_causal_gene_opentargets": "gwas_causal_gene_opentargets",
    "gwas_causal_gene_pharmaprojects": "gwas_causal_gene_pharmaprojects",
    "gwas_variant_prioritization": CASE_SENSITIVE_SUBTASK,
}

ANSWER_HINT = "\n\nSubmit your final choice inside a single <answer>...</answer> tag."


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_key(*parts):
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


def load_sources(cache):
    import pandas as pd

    cache.mkdir(parents=True, exist_ok=True)
    frames = {}
    for sub in ("DbQA", "LitQA2", "ProtocolQA"):
        local = cache / f"labbench-{sub}.parquet"
        if local.exists():
            frames[sub] = pd.read_parquet(local)
        else:
            print(f"  下载 LAB-Bench/{sub} ...", flush=True)
            frame = pd.read_parquet(LAB_BENCH.format(sub=sub))
            frame.to_parquet(local)
            frames[sub] = frame
        print(f"    {sub:12} {len(frames[sub]):5} 条")
    local = cache / "biomni-eval1.parquet"
    if local.exists():
        frames["Eval1"] = pd.read_parquet(local)
    else:
        print("  下载 biomni/Eval1 ...", flush=True)
        frame = pd.read_parquet(EVAL1)
        frame.to_parquet(local)
        frames["Eval1"] = frame
    print(f"    {'Eval1':12} {len(frames['Eval1']):5} 条")
    return frames


def make_choice_task(row, family, subtask, source_group, rng, *, extra_prompt="", assets=None):
    """ideal + distractors -> 打乱后的选项；参考答案必须是字母。

    data/verifier.py 会把模型答案归一化成字母（"B" / "B. text" / 选项原文都认），
    参考答案写选项原文会全判错。
    """
    ideal = str(row["ideal"]).strip()
    options = [str(d).strip() for d in list(row["distractors"])] + [ideal]
    rng.shuffle(options)
    index = options.index(ideal)
    listing = "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(options))
    prompt = str(row["question"]).strip()
    if extra_prompt:
        prompt = extra_prompt + "\n\n" + prompt
    prompt = f"{prompt}\n\n{listing}{ANSWER_HINT}"
    public = {
        "id": None,
        "prompt": prompt,
        "family": family,
        "subtask": subtask,
        "verifier": f"lab-bench-{family.lower()}-mcq",
        "options": options,
    }
    return public, {"answer": chr(65 + index)}, source_group, (assets or {})


def convert_dbqa(frame, rng):
    out = []
    for _, row in frame.iterrows():
        subtask = str(row["subtask"]).replace("-v1-public", "")
        # Per-row groups. LAB-Bench DbQA ships an empty `source` column and each row is an
        # independent database question, so there is no shared material to keep together.
        # Grouping by subtask (v1) gave DbQA only 10 groups, which let split_by_group hand
        # whole subtasks to one split: val ended up 17/17 variant_from_sequence while test
        # was 15/17 vax_response, and the two splits stopped measuring the same thing.
        public, reference, group, assets = make_choice_task(
            row, "DbQA", subtask, f"dbqa:{row['id']}", rng)
        out.append((str(row["id"]), public, reference, group, assets))
    return out


def convert_litqa2(frame, rng):
    out = []
    for _, row in frame.iterrows():
        sources = list(row["sources"]) if row["sources"] is not None else []
        group = f"litqa2:{sources[0]}" if sources else f"litqa2:{row['id']}"
        # key-passage 是答案线索，绝不进公开 prompt
        public, reference, group, assets = make_choice_task(
            row, "LitQA2", "scientific_literature_reading", group, rng)
        out.append((str(row["id"]), public, reference, group, assets))
    return out


def convert_protocolqa(frame, rng):
    out = []
    for _, row in frame.iterrows():
        protocol = str(row["protocol"]).strip()
        assets = {"protocol.txt": protocol + "\n"}
        header = ("The following experimental protocol is also available at "
                  "/workspace/assets/protocol.txt\n\n--- PROTOCOL ---\n"
                  + protocol + "\n--- END PROTOCOL ---")
        public, reference, group, assets = make_choice_task(
            row, "ProtocolQA", "experimental_protocol_troubleshooting",
            f"protocolqa:{row['id']}", rng, extra_prompt=header, assets=assets)
        out.append((str(row["id"]), public, reference, group, assets))
    return out


def convert_gwas(frame):
    """Eval1 的 GWAS 是自由作答（基因名 / rsID），没有选项。"""
    out = []
    subset = frame[frame["task_name"].isin(GWAS_TASKS)]
    for _, row in subset.iterrows():
        public = {
            "id": None,
            "prompt": str(row["prompt"]).strip() + ANSWER_HINT,
            "family": "GWAS",
            "subtask": GWAS_TASKS[row["task_name"]],
            "verifier": f"biomni-eval1-{row['task_name']}",
        }
        reference = {"answer": str(row["answer"]).strip()}
        source_id = f"{row['task_name']}-{row['task_instance_id']}"
        # Per-row groups for the same reason as DbQA: grouping by task_name gave GWAS only
        # four groups, so val received six gwas_causal_gene_opentargets while test received
        # five gwas_variant_prioritization and one causal-gene task.
        out.append((source_id, public, reference, f"gwas:{source_id}", {}))
    return out


def allocate(total, weights):
    scale = sum(weights.values())
    counts = {f: int(total * w / scale) for f, w in weights.items()}
    for family in sorted(weights, key=lambda f: -weights[f]):
        if sum(counts.values()) >= total:
            break
        counts[family] += 1
    while sum(counts.values()) > total:
        counts[max(counts, key=lambda f: counts[f])] -= 1
    return counts


def stratum_of(item):
    """Stratification key: (subtask, is the correct option the longest?).

    Two properties have to match across splits or val stops being a usable proxy for test.

    Subtask, because the ten DbQA subtasks differ enormously in difficulty and in which
    data-lake file answers them.

    Longest-is-correct, because LAB-Bench writes `ideal` in more detail than its
    `distractors`, so "always pick the longest option" scores well above chance. That
    artifact cannot be removed without rewriting benchmark content, but it can be spread
    evenly, which keeps the trivial baseline identical on every split and therefore
    reportable as one reference line.
    """
    _, public, reference, _, _ = item
    key = public.get("subtask") or public["family"]
    options = public.get("options") or []
    if not options:
        return (key, "free-form")
    answer = str(reference["answer"]).strip().upper()
    index = ord(answer) - 65 if len(answer) == 1 else -1
    if not 0 <= index < len(options):
        return (key, "unknown")
    lengths = [len(o) for o in options]
    return (key, "longest" if lengths[index] == max(lengths) else "not-longest")


def split_by_group(tasks, sizes, rng):
    """Deal whole source_groups across splits, stratified by stratum_of.

    v1 shuffled groups and filled test, then val, then train. With coarse groups that
    handed whole subtasks to a single split. Now groups are interleaved stratum by stratum
    and each one goes to whichever split is furthest below its share, so every split ends
    up with the same subtask mix and the same longest-is-correct rate.

    Groups still stay intact: LitQA2 rows sharing a source paper are one group and land
    together. Only the final group of a split may be trimmed to hit the exact count.
    """
    groups = defaultdict(list)
    for item in tasks:
        groups[item[3]].append(item)

    strata = defaultdict(list)
    for name, members in groups.items():
        strata[stratum_of(members[0])].append(name)
    for names in strata.values():
        names.sort(key=stable_key)
        rng.shuffle(names)

    # Round-robin across strata so no split can absorb one stratum wholesale.
    order = []
    for depth in range(max(len(n) for n in strata.values())):
        for key in sorted(strata, key=lambda k: stable_key(*k)):
            if depth < len(strata[key]):
                order.append(strata[key][depth])

    result = {"train": [], "val": [], "test": []}
    for group in order:
        members = groups[group]
        room = {s: sizes[s] - len(result[s]) for s in result}
        if not any(v > 0 for v in room.values()):
            break
        # Largest remaining share first; stable_key breaks ties without positional bias.
        split = max(sorted(result), key=lambda s: (room[s] / sizes[s], room[s]))
        result[split].extend(members[: room[split]])

    for split, need in sizes.items():
        if len(result[split]) != need:
            raise SystemExit(
                f"{split} needs {need} tasks but only {len(result[split])} could be allocated"
            )
    return result


def build(out_root, profile, seed, cache):
    train_n, val_n, test_n = PROFILES[profile]
    sizes = {"train": train_n, "val": val_n, "test": test_n}
    total = sum(sizes.values())
    print(f"目标 {out_root}\n规格 train={train_n} val={val_n} test={test_n} 共 {total}\n种子 {seed}\n")

    print("读取源数据")
    frames = load_sources(cache)

    # rng 的消耗顺序必须固定，否则跨机器结果不一致
    rng = random.Random(seed)
    print("\n转换")
    pools = {
        "DbQA": convert_dbqa(frames["DbQA"], rng),
        "LitQA2": convert_litqa2(frames["LitQA2"], rng),
        "ProtocolQA": convert_protocolqa(frames["ProtocolQA"], rng),
        "GWAS": convert_gwas(frames["Eval1"]),
    }
    for family, items in pools.items():
        print(f"  {family:12} 可用 {len(items):5} 条")

    quota = allocate(total, FAMILY_WEIGHTS)
    print("\n按论文族比例分配")
    for family, n in quota.items():
        if n > len(pools[family]):
            raise SystemExit(f"{family} 需要 {n} 条但只有 {len(pools[family])} 条")
        print(f"  {family:12} {n:5} 条")

    selected = {f: sorted(pools[f], key=lambda x: stable_key(seed, x[0]))[:n]
                for f, n in quota.items()}

    assignments = {}
    for family, items in selected.items():
        share = {s: max(1, round(len(items) * sizes[s] / total)) if sizes[s] else 0
                 for s in ("test", "val", "train")}
        while sum(share.values()) > len(items):
            share[max(share, key=lambda s: share[s])] -= 1
        while sum(share.values()) < len(items):
            share["train"] += 1
        parts = split_by_group(items, share, random.Random(stable_key(seed, family)[:8]))
        for split, members in parts.items():
            for item in members:
                assignments[item[0]] = (split, family, item)

    if out_root.exists():
        print(f"\n清除已有目录 {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    print("\n写入任务")
    rows, long_prompts = [], []
    for source_id, (split, family, item) in sorted(assignments.items()):
        _, public, reference, group, assets = item
        task_id = f"{family.lower()}-{stable_key(seed, source_id)[:12]}"
        public = dict(public, id=task_id)
        directory = out_root / task_id
        write_json(directory / "public/task.json", public)
        write_json(directory / "evaluator/reference.json", reference)
        asset_dir = directory / "public/assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        # 上游 task policy 约定：完整任务文本在 /workspace/assets/task_prompt.txt
        (asset_dir / "task_prompt.txt").write_text(public["prompt"] + "\n")
        for name, content in assets.items():
            (asset_dir / name).write_text(content)
        rows.append({"id": task_id, "split": split, "family": family, "source_group": group})
        if len(public["prompt"]) > 4 * PROMPT_TOKEN_LIMIT:
            long_prompts.append((task_id, len(public["prompt"])))

    lake = out_root / LAKE_DIRNAME
    lake.mkdir()
    (lake / "README.md").write_text(
        "# 公开数据湖\n\ndeploy/bootstrap.sh 会用 Biomni 的 data_lake 填充本目录，"
        "并重写 environment.lock.json 的 data_lake.files。\n")
    files = {"README.md": {"sha256": sha256_file(lake / "README.md")}}
    write_json(out_root / "environment.lock.json", {
        "image": "apptainer:runtime.sif",
        "network": "none",
        "data_lake": {
            "path": LAKE_DIRNAME,
            "digest": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "files": files,
        },
    })

    counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")}
    payload_hashes = {f"{r['id']}/evaluator/reference.json":
                      sha256_file(out_root / r["id"] / "evaluator/reference.json") for r in rows}
    write_json(out_root / "manifest.json", {
        "tasks": rows, "split_counts": counts, "payload_hashes": payload_hashes,
        "provenance": {
            "sources": {"DbQA": "futurehouse/lab-bench DbQA",
                        "LitQA2": "futurehouse/lab-bench LitQA2",
                        "ProtocolQA": "futurehouse/lab-bench ProtocolQA",
                        "GWAS": "biomni/Eval1 gwas_* tasks"},
            "seed": seed, "profile": profile, "family_weights": FAMILY_WEIGHTS,
            "note": "族比例按论文 Table S1 缩放；数量不与论文逐项相同。"
                    "划分按 source_group 整组进行以减少材料泄漏。",
        },
    })

    print(f"\n完成：{len(rows)} 个任务")
    for split in ("train", "val", "test"):
        members = [r for r in rows if r["split"] == split]
        per = defaultdict(int)
        for r in members:
            per[r["family"]] += 1
        print(f"  {split:6} {len(members):4}  " + "  ".join(f"{f}={per[f]}" for f in sorted(per)))

    by_split = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add(r["source_group"])
    print("\n材料组重叠（越少越好）")
    print(f"  Train-Val   {len(by_split['train'] & by_split['val'])} 组   (论文 20)")
    print(f"  Train-Test  {len(by_split['train'] & by_split['test'])} 组   (论文 18)")

    identity = hashlib.sha256(
        json.dumps({r["id"]: r for r in rows}, sort_keys=True).encode()).hexdigest()
    print(f"\n任务集指纹 {identity[:16]}   （用于核对不同机器上重建是否一致）")
    if long_prompts:
        print(f"\n注意 {len(long_prompts)} 个任务 prompt 可能超预算，用 --check-prompt-budget 精确核对")
    return out_root


def main():
    ap = argparse.ArgumentParser(description="确定性重建 frozen task release")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="debug", choices=sorted(PROFILES))
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--cache", default=None, help="源 parquet 缓存目录")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    cache = Path(args.cache) if args.cache else out.parent / ".sources"
    build(out, args.profile, args.seed, cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
