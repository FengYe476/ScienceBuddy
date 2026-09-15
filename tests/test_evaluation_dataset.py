import json

import pytest

from simple_scibuddy.data.validation import validate_payloads
from simple_scibuddy.evaluation.dataset import read_index


def test_eval_index_rejects_wrong_count_duplicate_and_training_split(tmp_path):
    path = tmp_path / "test.jsonl"
    row = {"id": "example", "split": "test"}
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Expected 180"):
        read_index(path, 180)
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="unique"):
        read_index(path)
    path.write_text(json.dumps(dict(row, split="train")))
    with pytest.raises(ValueError, match="explicit test"):
        read_index(path)


def test_eval_payloads_allow_split_metadata_changes_but_reject_asset_drift(tmp_path):
    for name, split in (("source", "dev"), ("frozen", "train")):
        root = tmp_path / name / "example"
        (root / "public/assets").mkdir(parents=True)
        (root / "evaluator").mkdir()
        (root / "public/task.json").write_text(json.dumps({"prompt": "Read evidence", "split": split}))
        (root / "evaluator/reference.json").write_text('{"answer": "A"}')
        (root / "public/assets/evidence.txt").write_text("original evidence")
    rows = [{"id": "example", "task_dir": "source/example"}]
    validate_payloads(rows, tmp_path / "index.jsonl", tmp_path / "frozen", root=tmp_path)
    (tmp_path / "source/example/public/assets/evidence.txt").write_text("changed evidence")
    with pytest.raises(ValueError, match="Frozen payload differs"):
        validate_payloads(rows, tmp_path / "index.jsonl", tmp_path / "frozen", root=tmp_path)
