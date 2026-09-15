"""Select an explicit evaluation index from locally frozen task payloads."""

import json
import shutil
from pathlib import Path

from simple_scibuddy.artifacts import ROOT, digest, file_digest, write_json
from simple_scibuddy.data.validation import validate_payloads


def read_index(path, expected_count=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Evaluation index must contain nonempty, unique task IDs")
    if any(row.get("split") != "test" for row in rows):
        raise ValueError("Expected an explicit test split")
    if expected_count is not None and len(ids) != expected_count:
        raise ValueError(f"Expected {expected_count} test problems, found {len(ids)} in {path}")
    return rows


def prepare_index(cfg, destination):
    import pyarrow as pa
    import pyarrow.parquet as pq

    index = Path(cfg["eval_index"])
    rows = read_index(index, cfg.get("expected_count"))
    frozen = Path(cfg["dataset"])
    manifest = json.loads((frozen / "manifest.json").read_text())
    known = {r["id"] for r in manifest["tasks"]}
    missing = {r["id"] for r in rows} - known
    if missing:
        raise ValueError(f"Freeze missing task payloads before evaluation: {sorted(missing)}")
    validate_payloads(rows, index, frozen)
    available = {}
    for split in ("train", "test"):
        for row in pq.read_table(ROOT / f"data/sciencebuddy/{split}.parquet").to_pylist():
            available[row["extra_info"]["task_id"]] = row
    ids = [r["id"] for r in rows]
    destination.mkdir(parents=True)
    shutil.copyfile(index, destination / "source-index.jsonl")
    pq.write_table(pa.Table.from_pylist([available[i] for i in ids]), destination / "test.parquet")
    write_json(destination / "manifest.json", {
        "counts": {"train": 0, "test": len(ids)}, "task_ids": {"train": [], "test": ids},
        "source_index": str(index), "source_index_sha256": file_digest(index),
        "frozen_dataset_digest": digest(manifest),
    })
    return destination
