"""The launcher owns only its environment, lock, and child process."""

import importlib.util
from pathlib import Path

import pytest


def launcher():
    path = Path(__file__).resolve().parents[1] / 'scripts/train.py'
    spec = importlib.util.spec_from_file_location('train_launcher', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_secrets_are_literal_and_caches_local(tmp_path, monkeypatch):
    module = launcher()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    monkeypatch.delenv('SCIENCEBUDDY_IMPROVER_API_KEY', raising=False)
    (tmp_path / '.secrets').mkdir()
    (tmp_path / '.secrets/wandb.env').write_text("export WANDB_API_KEY='$(never_execute_this)'\n")
    (tmp_path / '.secrets/improver.env').write_text('SCIENCEBUDDY_IMPROVER_API_KEY=test-only-key\n')
    env = module.environment()
    assert env['SCIENCEBUDDY_IMPROVER_API_KEY'] == 'test-only-key'
    assert env['WANDB_API_KEY'] == '$(never_execute_this)'
    assert env['UV_CACHE_DIR'] == str(tmp_path / '.cache/uv')


def test_launcher_does_not_inherit_other_project_python_paths_or_force_machine_libraries(tmp_path, monkeypatch):
    module = launcher()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setenv('PYTHONPATH', str(tmp_path / 'other-project'))
    monkeypatch.setenv('PYTHONHOME', str(tmp_path / 'other-python'))
    for name in ['CUDA_HOME', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'SIMPLE_SCIBUDDY_CUDA_HOME', 'SIMPLE_SCIBUDDY_CUDA_COMPAT']:
        monkeypatch.delenv(name, raising=False)
    env = module.environment()
    assert all(name not in env for name in ['PYTHONPATH', 'PYTHONHOME', 'CUDA_HOME', 'LD_PRELOAD', 'LD_LIBRARY_PATH'])


@pytest.mark.parametrize('relative', [False, True])
def test_launcher_passes_config_and_propagates_failure(tmp_path, monkeypatch, relative):
    module = launcher()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'environment', lambda: {'PATH': '/project/bin'})
    monkeypatch.setattr(module.shutil, 'which', lambda *a, **kw: '/project/bin/uv')
    monkeypatch.setattr(module.signal, 'signal', lambda *a: None)
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setattr('sys.argv', ['train.py', 'config.toml' if relative else str(tmp_path / 'config.toml'), '--dry-run'])
    calls = []

    class Child:
        stdout = ['configured\n']

        def wait(self):
            return 7

    def start(command, **kwargs):
        calls.append(command)
        return Child()

    monkeypatch.setattr(module.subprocess, 'Popen', start)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 7
    assert calls[0][-2:] == [str(tmp_path / 'config.toml'), '--dry-run']
    assert next((tmp_path / 'runs/logs').glob('*.log')).read_text() == 'configured\n'
