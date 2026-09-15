import asyncio
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_scibuddy.coevolve import continuation, loop, phase
from simple_scibuddy.coevolve.evidence import public_evidence
from simple_scibuddy.harness import scientific


@pytest.mark.parametrize('shared', [False, True])
def test_batch_bounds_concurrent_episodes_across_candidates(tmp_path, monkeypatch, shared):
    active = peak = closed = 0

    class Client:
        model_name = 'fixture'

        def __init__(self, url):
            pass

        async def close(self):
            nonlocal closed
            closed += 1

    async def episode(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.001)
            return {'reward': 0, 'stop_reason': 'fixture'}
        finally:
            active -= 1

    monkeypatch.setattr(phase, 'CompletionClient', Client)
    monkeypatch.setattr(phase, 'RecordingPolicy', lambda *args: None)
    monkeypatch.setattr(phase, 'run_episode', episode)
    dataset = SimpleNamespace(load=lambda task_id: SimpleNamespace(public={}))
    cfg = dict(workers=2, seed=42, context_tokens=100, max_tokens=10, actions=1,
               seconds=10, tool_seconds=1, runtime_image='fixture',
               tool_response_tokens=10, tool_history_tokens=10, model='fixture')
    rows = [{'id': str(i)} for i in range(5)]

    async def evaluate():
        gate = asyncio.Semaphore(cfg['workers']) if shared else None
        return await asyncio.gather(*(phase.batch(
            dataset, rows, Path(scientific.__file__), 'fixture', None, cfg,
            tmp_path / str(i), limit=gate) for i in range(3)))

    results = asyncio.run(evaluate())
    assert peak == (2 if shared else 6)
    assert active == 0 and closed == 3
    assert all(len(result) == len(rows) for result in results)


@pytest.mark.parametrize("mode,trigger,gain,rounds", [
    ("coevolve", "always_debug", False, 1),
    ("coevolve", "always_debug", False, 3),
    ("coevolve", "improvement", True, 2),
    ("coevolve", "improvement", False, 0),
    ("harness_evolve", "always_debug", True, 0),
])
@pytest.mark.parametrize("selection_tasks", [8, 16])
@pytest.mark.parametrize("replay_regression", [False, True])
def test_local_cycle_stops_services_for_rl_and_returns_updated_model(tmp_path, monkeypatch, mode, trigger, gain, rounds, replay_regression, selection_tasks):
    gain = gain and not replay_regression
    if trigger == "improvement" and not gain:
        rounds = 0
    models, training, logs, active, collected, definitions = [], [], [], [], [], []
    monkeypatch.setattr(loop, 'ROOT', tmp_path)
    from simple_scibuddy.artifacts import file_digest
    monkeypatch.setattr(loop, 'file_digest', lambda path: file_digest(path) if Path(path).is_file() else 'fixture')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    tracker = SimpleNamespace(log=lambda metrics: logs.append(metrics), finish=lambda **kwargs: None,
                              define_metric=lambda *args, **kwargs: definitions.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kwargs: tracker))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)))

    class Dataset:
        manifest = {'split_counts': {'train': 715, 'val': selection_tasks, 'test': 90}}
        root = tmp_path

        def __init__(self, root):
            pass

        def load(self, task_id):
            return SimpleNamespace(public={'subtask': 'test'})

        def tasks(self, split):
            return [{'id': f'{split}-{i}', 'split': split} for i in range({'train': 715, 'val': selection_tasks, 'test': 90}[split])]

    monkeypatch.setattr(loop, 'TaskDataset', Dataset)
    monkeypatch.setattr(phase, 'dataset_capabilities', lambda dataset: {'scope': 'public fixture'})

    @contextmanager
    def serve(model, gpu, folder, *args, **kwargs):
        models.append(str(model))
        active.append(gpu)
        yield 'http://localhost/v1'
        active.remove(gpu)

    monkeypatch.setattr(phase, 'serve', serve)
    monkeypatch.setattr('simple_scibuddy.inference.server.serve', serve)
    episode = {'reward': 0.0, 'stop_reason': 'first_answer_evaluation', 'calls': [], 'submissions': [{'outcome': {'answer_format_valid': True}}], 'task_id': 'train-1',
               'tool_calls': [], 'user_feedback': []}

    validation_limits = {}

    async def batch(*args, **kwargs):
        folder = args[6]
        if folder.name == "interaction":
            collected.extend(row['id'] for row in args[1])
        reward = int(gain and (folder.name == 'evaluation' or folder.name in {'validation-01', 'validation-02', 'validation-03'}))
        if folder.name.startswith('validation-'):
            gate = kwargs['limit']
            assert gate is validation_limits.setdefault(folder.parent, gate)
            assert len(args) == 7, 'Validation must disable researcher feedback'
            assert all(r['split'] == 'val' for r in args[1])
            assert not set(collected).intersection(r['id'] for r in args[1])
        return [dict(episode, reward=reward, task_id=r['id']) for r in args[1]]

    monkeypatch.setattr(phase, 'batch', batch)
    def propose(*args, **kwargs):
        Path(args[4]).write_text(Path(scientific.__file__).read_text() + '\n# candidate ' + str(args[4]))
        return {'status': 'validated'}

    monkeypatch.setattr(phase, 'propose', propose)

    def train(cfg, directory, worker, steps, launcher):
        assert not active, 'Harness services must release GPUs before RL'
        assert steps == 20
        assert cfg['single_wandb_run']
        phase_id = f'h{len(training) + 1:04d}'
        phase_path = tmp_path / 'runs/test/harness_evolve' / phase_id
        assert len(list(phase_path.glob('step-*/metrics.json'))) == 5
        request = json.loads((directory / 'request.json').read_text())
        assert request['harness_id'] == phase_id
        assert len(cfg['validation_task_ids']) == selection_tasks
        assert not set(cfg['validation_task_ids']).intersection(collected)
        if gain:
            assert request['eval']['metrics']['correct'] > request['reference_correct']
            assert 'step-0005' in json.loads((phase_path / 'summary.json').read_text())['selected_harness']
        training.append(directory)

    monkeypatch.setattr(loop, 'run_round', train)
    monkeypatch.setattr(loop, 'validate_result', lambda directory: {'model_path': str(tmp_path / f'updated-model-{len(training)}')})
    config = {'mode': mode, 'experiment': str(tmp_path / 'runs/test'), 'model': {'steps': 20},
              'harness_evolve': {'steps_per_phase': 5, 'feedback_every': 8, 'candidates': 3, 'selection_tasks': selection_tasks,
                                'preflight_task_id': sorted(Dataset(tmp_path).tasks('train'),
                                    key=lambda r: loop.digest([42, 0, r['id']]))[-1]['id'], 'improver': {
                  'backend': 'vllm', 'model': str(tmp_path / 'base-model')}},
              'coevolve': {'max_rounds': max(1, rounds), 'trigger': trigger}}
    cfg = {'dataset': str(tmp_path), 'harness': scientific.__file__, 'model': str(tmp_path / 'base-model'),
           'runtime_image': 'fixture', 'seed': 42, 'actions': 9, 'seconds': 900, 'max_tokens': 4096,
           'context_tokens': 24576, 'tool_response_tokens': 2048, 'tool_history_tokens': 8192}
    loop.run(config, cfg)
    expected_rounds = 0 if trigger == "improvement" and selection_tasks and replay_regression else rounds
    assert len(training) == expected_rounds
    expected_phases = max(1, expected_rounds)
    assert len(logs) == expected_phases * 6  # Baseline plus five harness_evolve steps.
    expected_models = []
    for i in range(expected_phases):
        expected_models.extend([str(tmp_path / ('base-model' if i == 0 else f'updated-model-{i}'))] * 8)
        if i:
            expected_models.append(str(tmp_path / 'base-model'))
    assert models == expected_models
    assert len(collected) == 5 * 8 * expected_phases and len(set(collected)) == len(collected)
    for i in range(1, expected_phases + 1):
        prefix = f'harness_evolve/h{i:04d}'
        steps = [row[prefix + '/step'] for row in logs if prefix + '/step' in row]
        assert steps == [0, 1, 2, 3, 4, 5]
        evaluation_steps = [row[prefix + "/step"] for row in logs if prefix + "/evaluation/accuracy" in row]
        assert evaluation_steps == [0, 5]
        assert any(kwargs['step_metric'] == prefix + '/step' for _, kwargs in definitions)
        phase_path = tmp_path / 'runs/test/harness_evolve' / f'h{i:04d}'
        reserved = json.loads((phase_path / 'selection-tasks.json').read_text())['task_ids']
        assert config["harness_evolve"]["preflight_task_id"] not in reserved
        assert len(reserved) == selection_tasks
        assert not set(reserved).intersection(collected)
        updates = [row for row in logs if prefix + '/update_applied' in row]
        assert all(row[prefix + '/update_applied'] == int(gain) for row in updates)
        for step in range(1, 6):
            result = json.loads((phase_path / f'step-{step:04d}/selection.json').read_text())
            assert len(result['candidates']) == 3
            assert result['selected_candidate'] == (1 if gain else 0)
    status = json.loads((tmp_path / 'runs/test/status.json').read_text())
    assert status['status'] == 'completed' and status['harness_phases'] == expected_phases
    if mode == 'coevolve' and not gain and trigger == 'improvement':
        assert status['stop_reason'] == 'no_harness_improvement'


def test_public_evidence_excludes_scores_and_private_outcomes():
    record = public_evidence({'task_id': 'x', 'stop_reason': 'stop', 'user_feedback': [{'reply': 'Check units'}],
                                  'calls': [], 'reward': 1, 'submissions': [{'outcome': 'secret answer'}]})
    assert 'secret answer' not in json.dumps(record)
    assert 'reward' not in record


def test_continuation_restores_only_a_completed_stage(tmp_path, monkeypatch):
    from simple_scibuddy.artifacts import file_digest, write_json
    monkeypatch.setattr(continuation, 'ROOT', tmp_path)
    source = tmp_path/'runs/previous'
    harness = tmp_path/'selected.py'
    harness.write_text(Path(scientific.__file__).read_text())
    config = {'mode': 'coevolve', 'model': {'steps': 30}, 'harness_evolve': {'steps_per_phase': 3},
              'coevolve': {'max_rounds': 2}}
    identities = {'dataset': 'd', 'runtime': 'r', 'verifier': 'v'}
    write_json(source/'config.json', config)
    write_json(source/'identities.json', identities)
    write_json(source/'shared/rounds/r0001/result.json', {'status': 'completed'})
    write_json(source/'harness_evolve/h0001/summary.json', {'selected_harness': str(harness), 'train_cursor': 48})
    write_json(source/'harness_evolve-history.json', [{'phase': 'h0001'}, {'phase': 'h0002'}])
    expected_hash = file_digest(harness)
    monkeypatch.setattr(continuation, 'validate_result', lambda directory: {
        'model_path': '/verified/export', 'trained_with_harness_sha256': expected_hash})
    state = continuation.completed_boundary(source, config, identities)
    assert state['rounds'] == 1 and state['train_cursor'] == 48
    assert state['model'] == '/verified/export' and state['history'] == [{'phase': 'h0001'}]
    with pytest.raises(ValueError, match='identity changed'):
        continuation.completed_boundary(source, config, dict(identities, dataset='changed'))
    harness.write_text('changed')
    with pytest.raises(ValueError, match='harness differs'):
        continuation.completed_boundary(source, config, identities)


def test_failed_rl_restarts_from_verified_harness_and_original_base(tmp_path, monkeypatch):
    from simple_scibuddy.artifacts import file_digest, tree_identity, write_json
    from simple_scibuddy.coevolve.protocol import claim, publish_request, publish_result
    monkeypatch.setattr(continuation, 'ROOT', tmp_path)
    source = tmp_path/'runs/previous'
    model = tmp_path/'M0'
    model.mkdir()
    (model/'weights.bin').write_bytes(b'unchanged M0')
    harness = tmp_path/'selected.py'
    harness.write_text('def run(task, api): pass\n')
    config = {'mode': 'coevolve', 'model': {'model': str(model), 'steps': 30},
              'harness_evolve': {'steps_per_phase': 3}, 'coevolve': {'max_rounds': 3}}
    identities = {'dataset': 'd', 'runtime': 'r', 'verifier': 'v', 'model': tree_identity(model)}
    summary = {'model': str(model), 'steps': 3, 'train_cursor': 48, 'selected_harness': str(harness),
               'selected_harness_sha256': file_digest(harness), 'initial': {'correct': 33},
               'selected': {'correct': 53, 'tasks': 90}}
    write_json(source/'config.json', config)
    write_json(source/'identities.json', identities)
    write_json(source/'status.json', {'status': 'failed'})
    write_json(source/'harness_evolve/h0001/summary.json', summary)
    write_json(source/'harness_evolve-history.json', [{'phase': 'h0001'}, {'phase': 'h0002'}])
    directory = publish_request(source/'shared', 'r0001', harness, base_model_id='M0', base_model_path=model,
                                harness_id='h0001', evaluation={'metrics': summary['selected']},
                                reference_correct=33, contract={}, source_run=source)
    claim(directory, 'rl-01')
    publish_result(directory, 'rl-01', failed='execution container stream closed')
    state = continuation.completed_boundary(source, config, identities)
    assert state['boundary'] == 'completed_harness' and state['rounds'] == 0
    assert state['model'] == str(model) and state['train_cursor'] == 48
    assert state['history'] == [{'phase': 'h0001'}]
    write_json(source/'status.json', {'status': 'rl'})
    with pytest.raises(ValueError, match='live source'):
        continuation.completed_boundary(source, config, identities)
    write_json(source/'status.json', {'status': 'failed'})
    (model/'weights.bin').write_bytes(b'changed model')
    with pytest.raises(ValueError, match='M0 identity changed'):
        continuation.completed_boundary(source, config, identities)


def test_harness_continuation_skips_h1_but_runs_all_three_rl_stages(tmp_path, monkeypatch):
    from simple_scibuddy.artifacts import file_digest
    monkeypatch.setattr(loop, 'ROOT', tmp_path)
    monkeypatch.setattr(loop, 'file_digest', lambda p: file_digest(p) if Path(p).is_file() else 'fixture')
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kw: SimpleNamespace(
        log=lambda *a, **kw: None, finish=lambda **kw: None)))
    harness = tmp_path/'selected.py'
    harness.write_text('def run(task, api): pass\n')
    summary = {'initial': {'correct': 0}, 'selected': {'correct': 1}, 'improved': True}
    boundary = {'source': str(tmp_path/'runs/previous'), 'boundary': 'completed_harness', 'rounds': 0,
                'model': str(tmp_path/'M0'), 'harness': str(harness), 'train_cursor': 48,
                'history': [{'phase': 'h0001'}], 'phase_summary': summary, 'phase_summary_sha256': 'fixture'}
    monkeypatch.setattr(loop, 'completed_boundary', lambda *args: boundary)
    class Dataset:
        manifest = {'split_counts': {'train': 12, 'val': 2, 'test': 2}}
        root = tmp_path
        def __init__(self, *args):
            pass
        def tasks(self, split):
            return [{'id': f'{split}-{i}', 'split': split} for i in range(self.manifest['split_counts'][split])]
    monkeypatch.setattr(loop, 'TaskDataset', Dataset)
    phases, rounds = [], []
    def evolve(root, phase, model, harness, train, cursor, *args):
        phases.append(phase)
        return harness, cursor + 48, summary
    monkeypatch.setattr(loop, 'harness_evolve', evolve)
    monkeypatch.setattr(loop, 'run_round', lambda cfg, directory, *a, **kw: rounds.append(directory.name))
    monkeypatch.setattr(loop, 'validate_result', lambda directory: {'model_path': str(tmp_path/directory.name)})
    config = {'mode': 'coevolve', 'continue_from': boundary['source'], 'experiment': str(tmp_path/'runs/new'),
              'model': {'steps': 30}, 'harness_evolve': {'steps_per_phase': 3, 'improver': {}},
              'coevolve': {'max_rounds': 3, 'trigger': 'always_debug'}}
    cfg = {'dataset': str(tmp_path), 'model': boundary['model'], 'harness': str(harness), 'runtime_image': 'fixture',
           'seed': 42, 'actions': 9, 'max_tokens': 4096, 'context_tokens': 24576, 'seconds': 900,
           'tool_response_tokens': 2048, 'tool_history_tokens': 8192}
    loop.run(config, cfg)
    assert phases == ['h0002', 'h0003']
    assert rounds == ['r0001', 'r0002', 'r0003']
    assert json.loads((tmp_path/'runs/new/status.json').read_text())['train_cursor'] == 144
    assert json.loads((tmp_path/'runs/new/harness_evolve/h0001/summary.json').read_text())['continued_from'] == boundary['source']
