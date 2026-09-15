from contextlib import contextmanager

import pytest

from simple_scibuddy.inference import server


def test_pool_routes_all_devices_and_cleans_up(monkeypatch, tmp_path):
    active = set()
    memory = {}

    @contextmanager
    def fake(model, gpu, folder, context, memory=0.75):
        active.add(gpu)
        allocations[gpu] = memory
        try:
            yield f'url-{gpu}'
        finally:
            active.remove(gpu)

    allocations = memory
    monkeypatch.setattr(server, 'serve', fake)
    with server.serve_pool('model', list(map(str, range(8))), tmp_path, 32768, share_last=True) as urls:
        assert urls == [f'url-{i}' for i in range(8)]
        assert len(active) == 8
        assert allocations['7'] == 0.4
        assert allocations['0'] == 0.75
    assert not active


def test_pool_cleans_successful_servers_on_startup_failure(monkeypatch, tmp_path):
    active = set()

    @contextmanager
    def fake(model, gpu, folder, context, memory=0.75):
        if gpu == '1':
            raise RuntimeError('startup failed')
        active.add(gpu)
        try:
            yield gpu
        finally:
            active.remove(gpu)

    monkeypatch.setattr(server, 'serve', fake)
    with pytest.raises(RuntimeError, match='startup failed'):
        with server.serve_pool('model', ['0', '1', '2'], tmp_path, 32768):
            pytest.fail('failed startup must not yield')
    assert not active
