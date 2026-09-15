import json
from types import SimpleNamespace

import pytest

from simple_scibuddy.artifacts import write_json
from simple_scibuddy.data.dataset import TaskDataset


def make_dataset(tmp_path):
    rows = []
    for split, count in [('train', 8), ('val', 2), ('test', 3)]:
        for i in range(count):
            identity = f'{split}-{i}'
            rows.append(dict(id=identity, split=split, family='science', source_group=identity))
            write_json(tmp_path / identity / 'public/task.json', dict(id=identity, prompt='Question'))
            write_json(tmp_path / identity / 'evaluator/reference.json', {'answer': 'private'})
            (tmp_path / identity / 'public/assets').mkdir()
    write_json(tmp_path / 'manifest.json', {'tasks': rows, 'split_policy': 'release_train_test',
                                           'split_counts': {'train': 8, 'val': 2, 'test': 3}})
    write_json(tmp_path / 'environment.lock.json', {'data_lake': {'path': 'resources'}})
    return rows


def test_native_val_never_enters_rl_or_test(tmp_path, monkeypatch):
    import sys

    from simple_scibuddy.training.data import prepare
    make_dataset(tmp_path)
    dataset = TaskDataset(tmp_path)
    assert len(dataset.tasks('val')) == 2
    assert dataset.load('val-0').public['split'] == 'val'
    written = {}
    pq = SimpleNamespace(write_table=lambda data, path: written.update({path.name: data}))
    monkeypatch.setitem(sys.modules, 'pyarrow', SimpleNamespace(
        Table=SimpleNamespace(from_pylist=lambda rows: rows), parquet=pq))
    monkeypatch.setitem(sys.modules, 'pyarrow.parquet', pq)
    cfg = {'dataset': str(tmp_path), 'seed': 42, 'max_tokens': 10, 'context_tokens': 100,
           'observation_chars': 100}
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [])
    output = tmp_path / 'prepared'
    prepare(cfg, output, tokenizer)
    assert len(written['train.parquet']) == 8 and len(written['test.parquet']) == 3
    assert all(row['extra_info']['task_id'].startswith('train-') for row in written['train.parquet'])
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['validation_task_ids'] == ['val-0', 'val-1']
    with pytest.raises(ValueError, match='explicit validation split'):
        prepare(dict(cfg, validation_task_ids=['test-0']), output, tokenizer)
    invalid = json.loads((tmp_path / 'manifest.json').read_text())
    invalid['split_counts']['train'] += 1
    write_json(tmp_path / 'manifest.json', invalid)
    with pytest.raises(ValueError, match='split counts'):
        TaskDataset(tmp_path)


def test_explicit_dataset_config_is_opt_in_and_resolves_paths(tmp_path):
    from simple_scibuddy.configuration import load_config
    config = tmp_path / 'train.toml'
    config.write_text('mode="model"\n[data]\ndataset="release"\n[model]\nmode="train"\n')
    assert load_config(config, root=tmp_path)['data']['dataset'] == str(tmp_path / 'release')
    config.write_text('mode="model"\n[model]\nmode="train"\n')
    assert 'data' not in load_config(config, root=tmp_path)




def test_rl_handoff_accepts_only_declared_train_test_ids(tmp_path, monkeypatch):
    from simple_scibuddy.coevolve import protocol, worker
    from simple_scibuddy.harness import scientific
    rows = make_dataset(tmp_path / 'dataset')
    base = tmp_path / 'model'
    write_json(base / 'config.json', {})
    harness = tmp_path / 'harness.py'
    harness.write_text(__import__('pathlib').Path(scientific.__file__).read_text())
    contract = dict(thinking_enabled=False, execution_network='disabled', temperature=1.0, top_p=1.0,
                    top_k=-1, entrypoint='run(task, api)', max_model_calls=9, max_tool_calls=9,
                    context_tokens=24576, max_output_tokens=4096, rollout_timeout_seconds=900,
                    tool_response_tokens=2048, tool_history_tokens=8192)
    for split, key in [('train', 'train_index'), ('test', 'eval_index')]:
        path = tmp_path / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows if r['split'] == split))
        contract[key] = str(path)
    request = {'runtime_contract': contract, 'base_model_path': str(base), 'harness_file': 'harness.py',
               'round_id': 'r0001', 'base_model_id': 'M0', 'harness_id': 'h0001'}
    write_json(tmp_path / 'request.json', request)
    monkeypatch.setattr(protocol, 'read_request', lambda _: request)
    monkeypatch.setattr(worker, 'validate_payloads', lambda *args: None)
    worker.plan_round({'dataset': str(tmp_path / 'dataset')}, tmp_path, 30)
    path = tmp_path / 'train.jsonl'
    path.write_text(path.read_text().replace('train-0', 'val-0'))
    with pytest.raises(ValueError, match='differs from frozen'):
        worker.plan_round({'dataset': str(tmp_path / 'dataset')}, tmp_path, 30)


def test_training_rejects_validation_and_test_tasks():
    from simple_scibuddy.training.data import assigned_split
    ids = {'train': ['train-0'], 'test': ['test-0']}
    assert assigned_split('train-0', ids, 'train') == 'train'
    assert assigned_split('test-0', ids, 'eval') == 'test'
    for task in ['val-0', 'test-0']:
        with pytest.raises(RuntimeError):
            assigned_split(task, ids, 'train')
