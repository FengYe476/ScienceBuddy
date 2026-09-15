import json
from pathlib import Path

import pytest

from simple_scibuddy import configuration
from simple_scibuddy.training.launch import build_overrides


def test_portable_defaults_local_overrides_and_sampling_precedence(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/defaults.json").write_bytes((root / "configs/defaults.json").read_bytes())
    (tmp_path / "configs/local.json").write_text(
        json.dumps({"model": "custom-model", "samples_per_prompt": 8})
    )
    monkeypatch.setattr(configuration, "ROOT", tmp_path)
    monkeypatch.setenv("N_SAMPLES_PER_PROMPT", "16")
    cfg = configuration.settings()
    assert cfg["model"] == str(tmp_path / "custom-model")
    assert cfg["dataset"] == str(tmp_path / "data/releases/id715-val90-test90-v1")
    assert cfg["samples_per_prompt"] == 16


def test_release_recipe_keeps_experimental_schedule_and_rl_contract(monkeypatch):
    monkeypatch.delenv("N_SAMPLES_PER_PROMPT", raising=False)
    root = Path(__file__).parents[1]
    config = configuration.load_config(root / "configs/train.toml")
    cfg = configuration.settings() | {k: v for k, v in config["model"].items() if k != "mode"}
    assert "continue_from" not in config
    assert config["coevolve"] == {"worker_id": "rl-01", "max_rounds": 3, "trigger": "always_debug"}
    assert config["harness_evolve"]["feedback_every"] == 16
    assert config["harness_evolve"]["steps_per_phase"] == 3
    assert config["harness_evolve"]["selection_tasks"] == 90
    assert config["harness_evolve"]["candidates"] == 3
    overrides = build_overrides(
        cfg, "train", root / "runs/example", root / "data/prepared", root / "data/prepared"
    )
    assert overrides["trainer.max_training_steps"] == 30
    assert overrides["trainer.eval_interval"] == 30
    assert overrides["trainer.train_batch_size"] == 8
    assert overrides["generator.n_samples_per_prompt"] == 8
    assert overrides["trainer.algorithm.advantage_estimator"] == "grpo"
    assert overrides["trainer.algorithm.use_kl_loss"] is False
    assert overrides["trainer.algorithm.use_kl_in_reward"] is False
    assert overrides["generator.eval_sampling_params.temperature"] == 0


@pytest.mark.parametrize("field", ["eval_every", "task_scoped", "replay_tasks"])
def test_removed_settings_are_rejected_instead_of_silently_ignored(tmp_path, field):
    original = (Path(__file__).parents[1] / "configs/train.toml").read_text()
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs/train.toml"
    path.write_text(original.replace("[harness_evolve]", f"[harness_evolve]\n{field} = 1"))
    with pytest.raises(ValueError, match="Unknown harness settings"):
        configuration.load_config(path, root=tmp_path)
