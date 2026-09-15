"""Adapt frozen train/test assignments to SkyRL parquet; never create or resplit tasks."""

import random

from simple_scibuddy.artifacts import digest, write_json
from simple_scibuddy.harness.scientific import SYSTEM, build_messages


def prepare(settings, destination, tokenizer):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from simple_scibuddy.data.dataset import TaskDataset

    dataset = TaskDataset(settings["dataset"])
    validation_ids = {r['id'] for r in dataset.tasks('val')}
    if set(settings.get('validation_task_ids', validation_ids)) != validation_ids:
        raise ValueError('Validation IDs must match the explicit validation split')
    rows = []
    assignments = {}
    for record in dataset.manifest["tasks"]:
        if record["id"] in validation_ids:
            continue
        task = dataset.load(record["id"])
        reference = task.reference()
        if "answer" not in reference:
            raise ValueError(f"Missing private reference: {record['id']}")
        assignments[record["id"]] = record["split"]
        files = [p.name for p in sorted(task.assets.iterdir())]
        system = SYSTEM + "\nPublic files: " + ", ".join("/workspace/assets/" + f for f in files)
        prompt = build_messages(dict(task.public, system=system, budgets={
            k: settings[k] for k in ("tool_response_tokens", "tool_history_tokens") if k in settings
        }))
        tokens = tokenizer.apply_chat_template(
            prompt, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
        )
        if len(tokens) + settings["max_tokens"] > settings["context_tokens"]:
            raise RuntimeError(f"Task {record['id']} exceeds context budget; no tasks silently excluded")
        rows.append({
            "prompt": prompt, "env_class": "sciencebuddy", "data_source": record["family"],
            "extra_info": {"task_id": record["id"], "family": record["family"]},
        })
    result = {split: [r for r in rows if assignments[r['extra_info']['task_id']] == split]
              for split in ('train', 'test')}
    smoke = []
    shuffled = list(result["train"])
    random.Random(settings["seed"]).shuffle(shuffled)
    families = sorted({r["data_source"] for r in shuffled})
    for i in range(8):
        family = families[i % len(families)]
        choice = next(r for r in shuffled if r["data_source"] == family)
        smoke.append(choice)
        shuffled.remove(choice)
    for split, entries in result.items():
        pq.write_table(pa.Table.from_pylist(entries), destination / f"{split}.parquet")
    for preset in ("smoke", "overfit"):
        pq.write_table(pa.Table.from_pylist(smoke), destination / f"{preset}.parquet")
    write_json(destination / "manifest.json", {
        "validation_task_ids": sorted(validation_ids),
        "counts": {k: len(v) for k, v in result.items()},
        "task_ids": {k: [r["extra_info"]["task_id"] for r in v] for k, v in result.items()},
        "seed": settings["seed"],
        "split_policy": "frozen_train_val_test",
        "smoke_task_ids": [r["extra_info"]["task_id"] for r in smoke],
        "source_digest": digest(dataset.manifest),
        # Parquet prompts schedule tasks; the broker supplies actual program-generated inputs to RL.
        "prompt_digest": digest({"system": SYSTEM,
                                 "program": settings.get("harness_program_sha256")}),
        "observation_chars": settings["observation_chars"],
        "tool_response_tokens": settings.get("tool_response_tokens"),
        "tool_history_tokens": settings.get("tool_history_tokens"),
    })
    for obsolete in ("val.parquet", "panel.parquet"):
        (destination / obsolete).unlink(missing_ok=True)


def assigned_split(task_id, task_ids, phase):
    """Enforce our frozen split independently of the original release's labels."""
    if phase not in ("train", "eval"):
        raise ValueError(f"Unknown training phase: {phase}")
    split = "train" if phase == "train" else "test"
    if task_id not in task_ids[split]:
        raise RuntimeError(f"Task {task_id} is not assigned to {split}")
    return split
