import asyncio
import sys
from types import SimpleNamespace

import pytest

from simple_scibuddy.inference.server import CompletionClient


@pytest.mark.parametrize('failures', [1, 3])
def test_completion_transport_retry_preserves_request(monkeypatch, failures):
    class TransportError(Exception):
        pass

    monkeypatch.setitem(sys.modules, 'httpx', SimpleNamespace(TransportError=TransportError))
    calls = []
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)

    async def post(endpoint, json):
        calls.append((endpoint, json))
        if len(calls) <= failures:
            raise TransportError('connection interrupted')
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'choices': [{
            'token_ids': [42], 'text': 'answer', 'logprobs': {'token_logprobs': [-0.2]},
            'finish_reason': 'stop'}]})

    monkeypatch.setattr(asyncio, 'sleep', sleep)
    client = CompletionClient.__new__(CompletionClient)
    client.client = SimpleNamespace(post=post)
    request = {'sampling_params': {'temperature': 0, 'seed': 42}, 'prompt_token_ids': [[1, 2]]}
    if failures == 3:
        with pytest.raises(TransportError):
            asyncio.run(client.generate(request, 'sciencebuddy'))
    else:
        result = asyncio.run(client.generate(request, 'sciencebuddy'))
        assert result['response_ids'] == [[42]]
    assert len(calls) == min(failures + 1, 3)
    assert all(call == calls[0] for call in calls)
    assert sleeps == ([1, 2] if failures == 3 else [1])
