import json
import shutil
import sys
from types import SimpleNamespace

import pytest

from simple_scibuddy import configuration
from simple_scibuddy.artifacts import write_json
from simple_scibuddy.data.dataset import TaskDataset
from simple_scibuddy.data.validation import validate_payloads
from simple_scibuddy.paths import local_path


@pytest.mark.parametrize("value", ["/external/model", "../../external/model", "~/model", "C:\\models\\base"])
def test_settings_reject_absolute_or_external_model_paths(tmp_path, monkeypatch, value):
    write_json(
        tmp_path / "configs/defaults.json",
        {
            "model": value,
            "dataset": "data/release",
            "harness": "seed.py",
            "samples_per_prompt": 8,
        },
    )
    monkeypatch.setattr(configuration, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="relative|inside"):
        configuration.settings()


@pytest.mark.parametrize(
    "target", ["model", "dataset", "harness", "run", "eval_index", "experiment", "continue_from", "improver"]
)
def test_toml_paths_cannot_borrow_another_project(tmp_path, target):
    base = 'mode="model"\n'
    if target == "dataset":
        text = base + '[data]\ndataset="../other/release"\n'
    elif target == "experiment":
        text = base + 'experiment="../other/run"\n'
    elif target == "continue_from":
        text = (
            'mode="coevolve"\nexperiment="runs/new"\ncontinue_from="../other/run"\n'
            '[harness_evolve.improver]\nbackend="vllm"\nmodel="models/base"\n'
        )
    elif target == "improver":
        text = (
            'mode="harness_evolve"\nexperiment="runs/new"\n'
            '[harness_evolve.improver]\nbackend="vllm"\nmodel="../other/model"\n'
        )
    else:
        text = base + "[model]\n" + ('mode="evaluate"\n' if target == "eval_index" else "")
        text += f'{target}="../other/resource"\n'
    path = tmp_path / "config.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="inside"):
        configuration.load_config(path, root=tmp_path)


def test_local_symlinks_cannot_reintroduce_external_model_files(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    write_json(
        root / "configs/defaults.json",
        {
            "model": "models/base",
            "dataset": "data/release",
            "harness": "seed.py",
            "samples_per_prompt": 8,
        },
    )
    (root / "models/base").mkdir(parents=True)
    outside = tmp_path / "other-weights.bin"
    outside.write_bytes(b"external")
    (root / "models/base/weights.bin").symlink_to(outside)
    monkeypatch.setattr(configuration, "ROOT", root)
    with pytest.raises(ValueError, match="resource symlink"):
        configuration.settings()
    path = root / "configs/train.toml"
    path.write_text('mode="model"\n[model]\nmodel="../models/base"\n')
    with pytest.raises(ValueError, match="resource symlink"):
        configuration.load_config(path)


def make_release(root):
    release = root / "data/release"
    row = {"id": "task", "split": "train", "family": "science", "source_group": "group"}
    write_json(release / "manifest.json", {"tasks": [row], "split_counts": {"train": 1, "val": 0, "test": 0}})
    write_json(
        release / "environment.lock.json",
        {"image": "fixture", "data_lake": {"path": "resources", "files": {}}},
    )
    write_json(release / "task/public/task.json", {"id": "task", "prompt": "Read the local material"})
    write_json(release / "task/evaluator/reference.json", {"answer": "A"})
    (release / "task/public/assets").mkdir()
    (release / "task/public/assets/evidence.txt").write_text("fixture evidence")
    (release / "resources").mkdir()
    return release


def test_dataset_and_indexes_survive_moving_the_whole_repository(tmp_path, monkeypatch):
    original = tmp_path / "original"
    make_release(original)
    write_json(
        original / "configs/defaults.json",
        {
            "model": "models/base",
            "dataset": "data/release",
            "harness": "seed.py",
            "samples_per_prompt": 8,
        },
    )
    (original / "runs").mkdir()
    row = {"id": "task", "task_dir": "../data/release/task"}
    (original / "runs/index.jsonl").write_text(json.dumps(row) + "\n")
    moved = tmp_path / "relocated"
    shutil.copytree(original, moved)
    shutil.rmtree(original)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(configuration, "ROOT", moved)
    cfg = configuration.settings()
    dataset = TaskDataset(cfg["dataset"])
    assert dataset.load("task").reference() == {"answer": "A"}
    assert dataset.load("task").lake_path == moved / "data/release/resources"
    validate_payloads([row], moved / "runs/index.jsonl", moved / "data/release", root=moved)
    assert not original.exists()


@pytest.mark.parametrize("path", ["../other-project/lake", "/external/lake"])
def test_release_environment_requires_a_local_relative_data_lake(tmp_path, path):
    release = make_release(tmp_path)
    write_json(release / "environment.lock.json", {"data_lake": {"path": path}})
    with pytest.raises(ValueError, match="data_lake.path"):
        TaskDataset(release)


def test_relative_index_rejects_external_task_payloads(tmp_path):
    release = make_release(tmp_path)
    with pytest.raises(ValueError, match="task_dir"):
        validate_payloads(
            [{"id": "task", "task_dir": "../../other/task"}],
            tmp_path / "runs/index.jsonl",
            release,
            root=tmp_path,
        )


def test_preflight_builds_the_explicit_dataset_instead_of_reloading_defaults(tmp_path, monkeypatch):
    from simple_scibuddy.environments import preflight
    from simple_scibuddy.training import data

    release = make_release(tmp_path)
    dockerfile = release / "runtime/env/Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text("FROM fixture\n")
    seed = tmp_path / "seed.py"
    seed.write_text("def run(task, api): pass\n")
    cfg = {"dataset": str(release), "model": str(tmp_path / "models/base"), "harness": str(seed)}
    seen = []
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: None)),
    )
    monkeypatch.setattr(data, "prepare", lambda settings, *a: seen.append(settings["dataset"]))

    def build(settings):
        seen.append(settings["dataset"])
        write_json(tmp_path / "configs/local.json", {"runtime_image": "fixture", "baked_lake": True})

    monkeypatch.setattr(preflight, "build", build)
    preflight.preflight(cfg)
    assert seen == [str(release), str(release)]


def test_in_repository_relative_parent_references_are_allowed(tmp_path):
    assert local_path("../models/base", base=tmp_path / "configs", root=tmp_path) == tmp_path / "models/base"
