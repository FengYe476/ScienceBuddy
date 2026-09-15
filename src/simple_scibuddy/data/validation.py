"""Validate frozen payloads against an explicit task index."""

import json
from pathlib import Path

from simple_scibuddy.artifacts import ROOT, file_digest
from simple_scibuddy.paths import local_path, local_tree


def validate_payloads(rows, index, frozen, *, root=None):
    root = Path(root if root is not None else ROOT).resolve()
    index = local_path(index, base=root, relative=False, field="task index")
    frozen = local_path(frozen, base=root, relative=False, field="task release")
    for row in rows:
        task = local_path(row["id"], base=frozen, field="task ID")
        source = local_path(row["task_dir"], base=index.parent, root=root, field="task_dir")
        local_tree(task, root=frozen)
        local_tree(source, root=root)
        # Split/group metadata may differ, but scientific inputs and answers must match.
        local_public = json.loads((task / "public/task.json").read_text())
        source_public = json.loads((source / "public/task.json").read_text())
        for key in ("prompt", "options", "verifier", "subtask"):
            if local_public.get(key) != source_public.get(key):
                raise ValueError(f"Frozen task differs from requested index: {row['id']} / {key}")
        names = {str(p.relative_to(source)) for p in (source / "public/assets").rglob("*") if p.is_file()}
        local_names = {str(p.relative_to(task)) for p in (task / "public/assets").rglob("*") if p.is_file()}
        if names != local_names:
            raise ValueError(f"Public asset set differs: {row['id']}")
        for name in names | {"evaluator/reference.json"}:
            if file_digest(source / name) != file_digest(task / name):
                raise ValueError(f"Frozen payload differs: {row['id']} / {name}")
