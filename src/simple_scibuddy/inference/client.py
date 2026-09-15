"""Exact per-call token recording with an async SkyRL client."""

import copy
import time


class ContextBudget(Exception):
    pass


class RecordingPolicy:
    def __init__(self, client, tokenizer, sampling, model, session, context=32768, max_tokens=4096):
        self.client, self.tokenizer, self.model = client, tokenizer, model
        self.sampling, self.session = dict(sampling or {}), session
        self.context, self.max_tokens = context, max_tokens
        self.calls = []

    async def generate(self, messages):
        started = time.monotonic()
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False
        )
        if len(prompt) + self.max_tokens > self.context:
            raise ContextBudget("context_budget")
        tokenized = time.monotonic()
        sampling = dict(self.sampling, max_tokens=self.max_tokens, logprobs=0)
        result = await self.client.generate(
            {"prompt_token_ids": [prompt], "session_ids": [self.session], "sampling_params": sampling},
            model=self.model,
        )
        ids = result["response_ids"][0]
        probs = result.get("response_logprobs")
        if not ids or probs is None or len(ids) != len(probs[0]):
            raise RuntimeError("Inference must return aligned completion IDs and log probabilities")
        call = {
            "messages": copy.deepcopy(messages),
            "prompt_ids": prompt,
            "completion_ids": ids,
            "old_logprobs": probs[0],
            # Server APIs differ in whether they include EOS text. Only decoded
            # content belongs in harness history; retain IDs/logprobs unchanged.
            "text": self.tokenizer.decode(ids, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False),
            "finish_reason": result["stop_reasons"][0],
            "call_index": len(self.calls),
            "sampling": sampling,
            "timings": {"tokenize_seconds": tokenized - started,
                        "generate_seconds": time.monotonic() - tokenized},
        }
        self.calls.append(call)
        return call
