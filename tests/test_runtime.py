import asyncio
import json
from types import SimpleNamespace

import pytest

from simple_scibuddy.environments import runtime
from simple_scibuddy.environments.runtime import Container


def test_oversized_tool_record_is_drained_before_next_response(tmp_path):
    async def run():
        container = Container("unused", tmp_path, "execution")
        stream = asyncio.StreamReader(limit=64)
        container.reader = SimpleNamespace(stdout=stream)
        stream.feed_data(b'{"stdout":"' + b'x' * 200 + b'","error":null}\n')
        stream.feed_data(b'{"stdout":"42","error":null}\n')
        stream.feed_eof()
        assert "exceeded" in (await container.receive())["error"]
        assert await container.receive() == {"stdout": "42", "error": None}
    asyncio.run(run())


@pytest.mark.parametrize('oom,running', [(True, False), (False, False), (False, True), (True, True)])
def test_closed_stream_records_state_and_only_classifies_confirmed_memory_budget(tmp_path, monkeypatch, oom, running):
    state = {'OOMKilled': oom, 'Running': running, 'ExitCode': 137 if oom else 1}
    async def command(*args):
        assert args[:2] == ('docker', 'inspect')
        return json.dumps(state)
    monkeypatch.setattr(runtime, 'command', command)
    async def check():
        container = Container('unused', tmp_path, 'execution')
        stream = asyncio.StreamReader()
        stream.feed_eof()
        container.reader = SimpleNamespace(stdout=stream)
        with pytest.raises(RuntimeError) as failure:
            await container.receive()
        assert isinstance(failure.value, runtime.ExecutionMemoryBudget) == (oom and not running)
        assert json.loads((tmp_path/'exit-state.json').read_text()) == state
    asyncio.run(check())
