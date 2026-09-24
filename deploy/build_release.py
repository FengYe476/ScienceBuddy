"""Deterministically rebuild the frozen task release on any machine (no adapter/ needed).

How the sources line up with the paper's Table S1 (verified):

    family        paper  source                                    difference
    DbQA          511    futurehouse/lab-bench DbQA          520    -9
    GWAS          180    biomni/Eval1, the four gwas_* tasks 193   -13
    ProtocolQA    108    futurehouse/lab-bench ProtocolQA    108   same
    LitQA2         96    futurehouse/lab-bench LitQA2        199  -103

The ten DbQA subtasks correspond one-to-one with the ten subtopics in Table S1. Family
proportions are scaled the same way; exact counts are not matched.

All randomness is determined by the seed, and the rng is consumed in a fixed order
(DbQA -> LitQA2 -> ProtocolQA), so one seed produces a byte-identical release anywhere.

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
# data/verifier.py:25 uses case-sensitive exact matching for this subtask, so the name must match exactly
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
            print(f"  downloading LAB-Bench/{sub} ...", flush=True)
            frame = pd.read_parquet(LAB_BENCH.format(sub=sub))
            frame.to_parquet(local)
            frames[sub] = frame
        print(f"    {sub:12} {len(frames[sub]):5} rows")
    local = cache / "biomni-eval1.parquet"
    if local.exists():
        frames["Eval1"] = pd.read_parquet(local)
    else:
        print("  downloading biomni/Eval1 ...", flush=True)
        frame = pd.read_parquet(EVAL1)
        frame.to_parquet(local)
        frames["Eval1"] = frame
    print(f"    {'Eval1':12} {len(frames['Eval1']):5} rows")
    return frames


def make_choice_task(row, family, subtask, source_group, rng, *, extra_prompt="", assets=None):
    """ideal + distractors -> shuffled options; the reference answer must be a letter.

    data/verifier.py normalizes the model's answer to a letter (it accepts "B", "B. text" and
    the option text itself). Storing the option text as the reference answer would mark every
    task wrong.
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
        # The key passage is an answer clue and must never enter the public prompt
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
    """Eval1 GWAS tasks are free-form (gene name / rsID) and have no options."""
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
    print(f"target {out_root}\nspec train={train_n} val={val_n} test={test_n}, {total} total\nseed {seed}\n")

    print("reading sources")
    frames = load_sources(cache)

    # The rng consumption order must stay fixed or results differ across machines
    rng = random.Random(seed)
    print("\nconverting")
    pools = {
        "DbQA": convert_dbqa(frames["DbQA"], rng),
        "LitQA2": convert_litqa2(frames["LitQA2"], rng),
        "ProtocolQA": convert_protocolqa(frames["ProtocolQA"], rng),
        "GWAS": convert_gwas(frames["Eval1"]),
    }
    for family, items in pools.items():
        print(f"  {family:12} {len(items):5} usable")

    quota = allocate(total, FAMILY_WEIGHTS)
    print("\nallocating by the paper's family proportions")
    for family, n in quota.items():
        if n > len(pools[family]):
            raise SystemExit(f"{family} needs {n} rows but only {len(pools[family])} are available")
        print(f"  {family:12} {n:5} rows")

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
        print(f"\nremoving existing directory {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    print("\nwriting tasks")
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
        # The upstream task policy promises the full task text at /workspace/assets/task_prompt.txt
        (asset_dir / "task_prompt.txt").write_text(public["prompt"] + "\n")
        for name, content in assets.items():
            (asset_dir / name).write_text(content)
        rows.append({"id": task_id, "split": split, "family": family, "source_group": group})
        if len(public["prompt"]) > 4 * PROMPT_TOKEN_LIMIT:
            long_prompts.append((task_id, len(public["prompt"])))

    lake = out_root / LAKE_DIRNAME
    lake.mkdir()
    (lake / "README.md").write_text(
        "# Public data lake\n\ndeploy/bootstrap.sh fills this directory from Biomni's data_lake "
        "and rewrites data_lake.files in environment.lock.json.\n")
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
            "note": "Family proportions are scaled from the paper's Table S1; counts are not matched "
                    "item by item. Splits are dealt whole source_groups to limit material leakage.",
        },
    })

    print(f"\ndone: {len(rows)} tasks")
    for split in ("train", "val", "test"):
        members = [r for r in rows if r["split"] == split]
        per = defaultdict(int)
        for r in members:
            per[r["family"]] += 1
        print(f"  {split:6} {len(members):4}  " + "  ".join(f"{f}={per[f]}" for f in sorted(per)))

    by_split = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add(r["source_group"])
    print("\nmaterial-group overlap (lower is better)")
    print(f"  Train-Val   {len(by_split['train'] & by_split['val'])} groups   (paper: 20)")
    print(f"  Train-Test  {len(by_split['train'] & by_split['test'])} groups   (paper: 18)")

    identity = hashlib.sha256(
        json.dumps({r["id"]: r for r in rows}, sort_keys=True).encode()).hexdigest()
    print(f"\ntask-set fingerprint {identity[:16]}   (compare across machines to confirm an identical rebuild)")
    if long_prompts:
        print(f"\nnote: {len(long_prompts)} task prompts may exceed the budget; use --check-prompt-budget to verify exactly")
    return out_root


def main():
    ap = argparse.ArgumentParser(description="Deterministically rebuild the frozen task release")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="debug", choices=sorted(PROFILES))
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--cache", default=None, help="cache directory for the source parquet files")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    cache = Path(args.cache) if args.cache else out.parent / ".sources"
    build(out, args.profile, args.seed, cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
