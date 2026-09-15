import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from simple_scibuddy.coevolve import phase
from simple_scibuddy.training.tracking import EventTracker, ExperimentTracking, MetricRelay, publish_config


def test_single_writer_keeps_round_axes_tables_and_partial_events(tmp_path, monkeypatch):
    logs, definitions = [], []
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(Table=lambda **kw: kw))
    owner = SimpleNamespace(log=logs.append, define_metric=lambda *a, **kw: definitions.append((a, kw)))
    relay = MetricRelay(owner)
    cfg = {'run_dir': str(tmp_path), 'mode': 'train', 'coevolve': {'round_id': 'r0001'}}
    tracker = EventTracker(cfg)
    tracker.log({'loss': 0.5}, step=20)
    tracker.log_samples_to_table('samples', ['text'], [('answer',)], step=20)
    tracker.finish()  # Does not finish the parent or create another W&B run.
    relay.drain(tmp_path)
    relay.drain(tmp_path)
    assert len(logs) == 2
    assert logs[0] == {'rl/r0001/loss': 0.5, 'rl/r0001/step': 20}
    assert logs[1]['rl/r0001/samples']['data'] == [['answer']]
    event = json.dumps({'kind': 'metrics', 'prefix': 'evaluation/r0002', 'step': 0, 'data': {'accuracy': 0.8}})
    with tracker.path.open('a') as stream:
        stream.write(event[:20])
    relay.drain(tmp_path)
    assert len(logs) == 2
    with tracker.path.open('a') as stream:
        stream.write(event[20:] + '\n')
    relay.drain(tmp_path)
    assert logs[-1] == {'evaluation/r0002/accuracy': 0.8, 'evaluation/r0002/step': 0}
    assert len(definitions) == 2


def test_child_tracker_never_initializes_wandb(tmp_path, monkeypatch):
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'single_wandb_run': True, 'run_dir': str(tmp_path),
                                'mode': 'evaluate', 'coevolve': {'round_id': 'r0001'}}))
    monkeypatch.setenv('SKYRL_SIMPLE_SCIBUDDY_SETTINGS', str(path))

    class Base:
        def get_tracker(self):
            raise AssertionError('Child must not initialize a W&B run')

    class Child(ExperimentTracking, Base):
        pass

    child = Child()
    assert child.get_tracker() is child.get_tracker()
    assert child.get_tracker().prefix == 'evaluation/r0001'


def test_checkpoint_eval_has_its_own_section(tmp_path):
    tracker = EventTracker({'run_dir': str(tmp_path), 'mode': 'train', 'coevolve': {'round_id': 'r0001'}})
    tracker.log({'eval/accuracy': 0.5, 'loss': 0.1}, step=20)
    records = [json.loads(line) for line in tracker.path.read_text().splitlines()]
    assert records[0]['prefix'] == 'evaluation/r0001/checkpoint'
    assert records[0]['data'] == {'eval/accuracy': 0.5}
    assert records[1]['prefix'] == 'rl/r0001'


def test_exact_toml_uploaded_and_parsed_config_visible(tmp_path):
    text = '# Preserve comments and formatting\r\nmode = "coevolve"\r\n[model]\r\nsteps = 20\r\n'
    config, uploads = {}, []
    run = SimpleNamespace(config=config, save=lambda *a, **kw: uploads.append((a, kw)))
    publish_config(run, tmp_path, text)
    assert (tmp_path / 'config.toml').read_bytes() == text.encode()
    assert config['launch']['model']['steps'] == 20
    assert uploads[0][1]['policy'] == 'now'


@pytest.mark.parametrize('interactive,temperature', [(False, 0.0), (True, 1.0)])
def test_harness_evaluation_is_greedy_but_interaction_samples(tmp_path, monkeypatch, interactive, temperature):
    observed = []

    class Client:
        model_name = 'fixture'

        def __init__(self, *args):
            pass

        async def close(self):
            pass

    async def episode(task, harness, policy, *args, **kwargs):
        observed.append(policy.sampling)
        return {'reward': 0, 'stop_reason': 'stop'}

    monkeypatch.setattr(phase, 'file_digest', lambda path: 'fixture')
    monkeypatch.setattr(phase, 'CompletionClient', Client)
    monkeypatch.setattr(phase, 'run_episode', episode)
    cfg = dict(workers=1, seed=42, model='fixture', context_tokens=100, max_tokens=10, actions=9,
               seconds=10, tool_seconds=10, runtime_image='fixture', tool_response_tokens=256, tool_history_tokens=512)
    asyncio.run(phase.batch(SimpleNamespace(load=lambda task: SimpleNamespace(public={})), [{'id': 'task'}], 'harness', 'url', None,
                           cfg, tmp_path, object() if interactive else None))
    assert observed[0]['temperature'] == temperature
    assert observed[0]['top_p'] == 1.0


def test_nonzero_evaluation_override_rejected_before_launch(tmp_path):
    from simple_scibuddy.configuration import load_config
    from simple_scibuddy.training.launch import launch

    with pytest.raises(ValueError, match='temperature 0'):
        launch({'eval_temperature': 1.0}, 'evaluate')
    config = tmp_path / 'train.toml'
    config.write_text('mode="model"\n[model]\nmode="evaluate"\neval_temperature=1.0\n')
    with pytest.raises(ValueError, match='temperature 0'):
        load_config(config, root=tmp_path)
