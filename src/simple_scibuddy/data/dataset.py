"""Read frozen local task payloads; private references stay in the host verifier."""

import json
from dataclasses import dataclass
from pathlib import Path

from simple_scibuddy.data.verifier import grade_answer
from simple_scibuddy.paths import local_path, local_tree


def load_environment(root):
    root = Path(root).resolve()
    lock = local_path("environment.lock.json", base=root)
    environment = json.loads(lock.read_text())
    environment["data_lake"]["path"] = str(
        local_path(environment["data_lake"]["path"], base=root, field="data_lake.path")
    )
    return environment


@dataclass(frozen=True)
class Task:
    directory: Path
    public: dict
    environment: dict

    @property
    def assets(self):
        return self.directory / "public/assets"

    @property
    def lake_path(self):
        return Path(self.environment["data_lake"]["path"])

    def reference(self):
        return json.loads((self.directory / "evaluator/reference.json").read_text())

    def verify(self, response):
        return grade_answer(self.public, self.reference(), response)


class TaskDataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        local_tree(self.root)
        self.manifest = json.loads(local_path("manifest.json", base=self.root).read_text())
        for name in self.manifest.get("payload_hashes", {}):
            local_path(name, base=self.root, field="payload hash path")
        if self.manifest.get("format") == "task-index-v1":
            raise ValueError("Freeze task payloads locally before running")
        self.rows = {row["id"]: row for row in self.manifest["tasks"]}
        if len(self.rows) != len(self.manifest["tasks"]):
            raise ValueError("Duplicate task IDs")
        counts = self.manifest.get("split_counts")
        if counts is not None:
            actual = {s: len(self.tasks(s)) for s in ("train", "val", "test")}
            if counts != actual or sum(actual.values()) != len(self.rows):
                raise ValueError("Dataset split counts do not match its task assignments")
        self.environment = load_environment(self.root)

    def tasks(self, split):
        return [row for row in self.rows.values() if row["split"] == split]

    def load(self, task_id):
        row = self.rows[task_id]
        directory = (self.root / task_id).resolve()
        if directory.parent != self.root:
            raise ValueError("Task payload must be inside the local release")
        public = json.loads((directory / "public/task.json").read_text())
        if public["id"] != task_id:
            raise ValueError("Task identity mismatch")
        public.update(split=row["split"], source_group=row["source_group"])
        return Task(directory, public, self.environment)
