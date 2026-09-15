import json
from pathlib import Path

import pytest

from simple_scibuddy.artifacts import write_json
from simple_scibuddy.coevolve import protocol, worker
from simple_scibuddy.harness import scientific


def make_round(tmp_path):
    model = tmp_path / "base"
    model.mkdir()
    write_json(model / "config.json", {"model_type": "fixture"})
    write_json(model / "tokenizer_config.json", {})
    write_json(model / "tokenizer.json", {})
    (model / "model.safetensors").write_bytes(b"mock model weights; no real inference in unit test")
    harness = tmp_path / "harness.py"
    harness.write_text(Path(scientific.__file__).read_text())
    directory = protocol.publish_request(tmp_path / "runs/shared", "r0001", harness,
                                         base_model_id="M0", base_model_path=model, harness_id="H1",
                                         evaluation={"correct": 2}, reference_correct=1,
                                         contract={}, source_run=tmp_path / "harness-run")
    return directory, model


def test_claim_conflict_and_result_requires_load_check(tmp_path):
    directory, model = make_round(tmp_path)
    protocol.claim(directory, "worker-a")
    protocol.claim(directory, "worker-a")
    with pytest.raises(ValueError, match="Conflicting"):
        protocol.claim(directory, "worker-b")
    with pytest.raises(ValueError, match="successful local checkpoint load"):
        protocol.publish_result(directory, "worker-a", "M1", model)


@pytest.mark.parametrize("fail_load", [False, True])
def test_round_runs_train_then_export_eval_and_never_duplicates(tmp_path, monkeypatch, fail_load):
    directory, model = make_round(tmp_path)
    request = protocol.read_request(directory)
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "plan_round", lambda cfg, path, steps: (request, dict(model=str(model))))
    task_ids = [f"task-{i}" for i in range(45)]

    def prepare(cfg, path):
        cfg["data_dir"] = str(path / "data")
        write_json(path / "data/manifest.json", {"task_ids": {"test": task_ids}})

    calls = []

    def launch(cfg, mode):
        calls.append(mode)
        run = Path(cfg["planned_run_dir"])
        run.mkdir()
        write_json(run / "status.json", {"status": "completed"})
        if mode == "train":
            import shutil

            shutil.copytree(model, run / "export")
            write_json(run / "latest-checkpoint.json", {"step": 2, "export": str(run / "export")})
        else:
            assert cfg["model"].endswith("rl-train/export")
            assert not (directory / "result.json").exists()
            if fail_load:
                raise RuntimeError("load check failed")
            for task_id in task_ids:
                write_json(run / f"episodes/{task_id}/episode.json", {
                    "task_id": task_id, "calls": [{"text": "<answer>A</answer>"}],
                })
        return run

    if fail_load:
        with pytest.raises(RuntimeError, match="load check failed"):
            worker.run_round({}, directory, "fixture", 2, launcher=launch, preparer=prepare)
    else:
        worker.run_round({}, directory, "fixture", 2, launcher=launch, preparer=prepare)
    result = protocol.validate_result(directory)
    assert result["status"] == ("failed" if fail_load else "completed")
    assert calls == ["train", "evaluate"]
    worker.run_round({}, directory, "fixture", 2, launcher=launch, preparer=prepare)
    assert calls == ["train", "evaluate"]
    if not fail_load:
        assert result["producer_load_checked"]
        assert json.loads((directory / "rl-job.json").read_text())["status"] == "completed"


def test_incomplete_job_cannot_be_silently_relaunched(tmp_path, monkeypatch):
    directory, _ = make_round(tmp_path)
    request = protocol.read_request(directory)
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "plan_round", lambda *args: (request, {}))
    write_json(directory / "rl-job.json", {"status": "training"})
    with pytest.raises(RuntimeError, match="already has an RL job"):
        worker.run_round({}, directory, "fixture")


def test_experiment_config_paths_and_model_dispatch(tmp_path, monkeypatch):
    from simple_scibuddy import cli
    from simple_scibuddy.coevolve import experiment

    config = tmp_path / 'train.toml'
    config.write_text('mode = "model"\n[model]\nmode = "smoke"\nsteps = 2\n'
                      'model = "weights"\nno_eval = true\nno_checkpoint = true\n')
    from simple_scibuddy import configuration
    monkeypatch.setattr(configuration, "ROOT", tmp_path)
    parsed = experiment.load_config(config)
    assert parsed['model']['model'] == str(tmp_path / 'weights')
    calls = []
    monkeypatch.setattr(cli, 'main', lambda argv: calls.append(argv))
    monkeypatch.setattr('sys.argv', ['experiment', str(config)])
    experiment.main()
    assert calls[0][0] == 'smoke'
    assert '--no-eval' in calls[0] and '--no-checkpoint' in calls[0]
    monkeypatch.setattr('sys.argv', ['experiment', str(config), '--dry-run'])
    experiment.main()
    assert len(calls) == 1


@pytest.mark.parametrize('content', [
    'mode = "typo"',
    'mode = "model"\n[model]\nstep = 2',
    'mode = "model"\n[model]\nsteps = 0',
    'mode = "coevolve"\n[model]\nno_checkpoint = true',
    'mode = "model"\n[model]\nmode = "resume"',
])
def test_experiment_config_rejects_invalid_settings(tmp_path, content):
    from simple_scibuddy.configuration import load_config

    config = tmp_path / 'invalid.toml'
    config.write_text(content)
    with pytest.raises(ValueError):
        load_config(config, root=tmp_path)


def test_failed_round_recovery_preserves_failure_and_advances_coordinator(tmp_path):
    directory, _ = make_round(tmp_path)
    protocol.claim(directory, 'fixture')
    protocol.publish_result(directory, 'fixture', failed='fixture failure')
    request = protocol.read_request(directory)
    state_path = Path(request['source_run']) / 'coevolve-state.json'
    write_json(state_path, {'awaiting': str(directory), 'loading': None, 'round_number': 1})
    result = protocol.retry_failed_round(directory)
    retry = Path(result['retry_round'])
    assert retry.name == 'r0002'
    assert protocol.read_request(retry)['base_model_id'] == request['base_model_id']
    assert protocol.validate_result(directory)['status'] == 'failed'
    assert json.loads(state_path.read_text())['awaiting'] == str(retry)
    assert not (retry / 'accepted.json').exists()


def test_qwen_text_export_retains_structure_guard(tmp_path):
    base, exported = tmp_path / 'base', tmp_path / 'export'
    base.mkdir()
    exported.mkdir()
    text = {'model_type': 'qwen3_5_text', 'hidden_size': 2560, 'num_hidden_layers': 32, 'vocab_size': 248320}
    write_json(base / 'config.json', {'model_type': 'qwen3_5',
               'architectures': ['Qwen3_5ForConditionalGeneration'], 'text_config': text})
    write_json(exported / 'config.json', dict(text, architectures=['Qwen3_5ForCausalLM']))
    worker.preserve_tokenizer(base, exported)
    write_json(exported / 'config.json', dict(text, architectures=['Qwen3_5ForCausalLM'], hidden_size=1))
    with pytest.raises(ValueError, match='model structure'):
        worker.preserve_tokenizer(base, exported)
    write_json(exported / 'config.json', dict(text, architectures=['UnrelatedModel']))
    with pytest.raises(ValueError, match='architecture'):
        worker.preserve_tokenizer(base, exported)
