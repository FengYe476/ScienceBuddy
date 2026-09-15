import asyncio

import pytest

from simple_scibuddy.inference.client import ContextBudget, RecordingPolicy


class Tokenizer:
    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert ids == [7, 8]
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return "text"

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]


class Client:
    async def generate(self, request, model):
        self.request = request
        return {
            "response_ids": [[7, 8]],
            "response_logprobs": [[-0.1, -0.2]],
            "responses": ["text"],
            "stop_reasons": ["stop"],
        }


def test_tokens_are_preserved_without_retokenizing_completion():
    client = Client()
    policy = RecordingPolicy(client, Tokenizer(), {}, "model", "session", max_tokens=4)
    call = asyncio.run(policy.generate([{"role": "user", "content": "x"}]))
    assert call["text"] == "text"
    assert call["completion_ids"] == [7, 8]
    assert call["old_logprobs"] == [-0.1, -0.2]
    assert client.request["prompt_token_ids"] == [[1, 2, 3]]


def test_context_exhaustion_makes_no_generation_call():
    policy = RecordingPolicy(Client(), Tokenizer(), {}, "model", "session", context=5, max_tokens=4)
    with pytest.raises(ContextBudget):
        asyncio.run(policy.generate([]))
    assert policy.calls == []


def test_server_eos_text_does_not_enter_harness_history():
    class EOSClient(Client):
        async def generate(self, request, model):
            result = await super().generate(request, model)
            result["responses"] = ["text<|im_end|>"]
            return result

    async def collect(client):
        policy = RecordingPolicy(client, Tokenizer(), {}, "model", "session", max_tokens=4)
        first = await policy.generate([{"role": "user", "content": "x"}])
        second = await policy.generate([
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": first["text"]},
            {"role": "user", "content": "tool observation"},
        ])
        return first, second

    plain = asyncio.run(collect(Client()))
    with_eos = asyncio.run(collect(EOSClient()))
    assert with_eos[1]["messages"] == plain[1]["messages"]
    assert with_eos[1]["messages"][1]["content"] == "text"
    assert with_eos[0]["completion_ids"] == [7, 8]
    assert with_eos[0]["old_logprobs"] == [-0.1, -0.2]
