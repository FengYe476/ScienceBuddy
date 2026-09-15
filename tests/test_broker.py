import asyncio
import json
from types import SimpleNamespace

import pytest

from simple_scibuddy.environments.runtime import ExecutionMemoryBudget
from simple_scibuddy.harness.broker import bounded_observation, run_episode


class Task:
    environment = {"image": "fixture"}
    lake_path = "/fixture"
    public = {
        "id": "test-fixture",
        "family": "SeqQA",
        "source_group": "fixture",
        "split": "train",
        "prompt": "Compute 6*7",
    }

    def verify(self, text):
        return {"score": float(text == "<answer>42</answer>"), "private": "SECRET_REFERENCE"}


class FakeContainer:
    instances = []
    events = []
    fail_start = False

    def __init__(self, image, folder, kind, **kwargs):
        self.kind, self.closed, self.sent = kind, False, []
        self.instances.append(self)

    async def start(self, **kwargs):
        if self.fail_start:
            raise RuntimeError("Docker unavailable")

    async def send(self, value):
        self.sent.append(value)

    async def receive(self):
        return self.events.pop(0)

    async def execute(self, code, timeout=120):
        return {"stdout": "42", "error": None}

    async def collect(self):
        pass

    async def close(self):
        self.closed = True


@pytest.fixture
def setup(tmp_path):
    FakeContainer.instances, FakeContainer.events, FakeContainer.fail_start = [], [], False
    task = Task()
    task.assets = tmp_path / "assets"
    task.assets.mkdir()
    harness = tmp_path / "harness.py"
    harness.write_text("def run(task, api): pass")
    policy = SimpleNamespace(calls=[], max_tokens=4096, context=32768)

    async def generate(messages):
        c = {"text": "<answer>42</answer>", "finish_reason": "stop"}
        policy.calls.append(c)
        return c

    policy.generate = generate
    return task, harness, policy, tmp_path / "run"


def test_first_submission_ends_without_feedback_and_cleans(setup):
    FakeContainer.events = [
        {"op": "generate", "messages": [{"role": "user", "content": "Compute 6*7"}]},
        {"op": "execute", "code": "print(42)"},
        {"op": "submit", "text": "<answer>42</answer>"},
    ]
    result = asyncio.run(run_episode(*setup, "SYSTEM", container_factory=FakeContainer))
    assert result["reward"] == 1
    assert len(result["tool_calls"]) == 1
    assert all(c.closed for c in FakeContainer.instances)
    assert "SECRET_REFERENCE" not in str([c.sent for c in FakeContainer.instances])


def test_partial_start_failure_cleans_both_owned_resources(setup):
    FakeContainer.fail_start = True
    with pytest.raises(RuntimeError, match="Docker unavailable"):
        asyncio.run(run_episode(*setup, "SYSTEM", container_factory=FakeContainer))
    assert all(c.closed for c in FakeContainer.instances)


def test_interactive_feedback_preserves_first_reward_and_hides_reference(setup):
    FakeContainer.events = [
        {"op": "generate", "messages": [{"role": "user", "content": "Compute 6*7"}]}, {"op": "submit", "text": "<answer>0</answer>"},
        {"op": "generate", "messages": [{"role": "user", "content": "Compute 6*7"}]}, {"op": "submit", "text": "<answer>42</answer>"},
    ]

    async def feedback(public, calls, tools, answer):
        assert "SECRET_REFERENCE" not in str((public, calls, tools, answer))
        return {"accept": False, "reply": "Please recompute."}

    result = asyncio.run(run_episode(*setup, "SYSTEM", container_factory=FakeContainer,
                                    feedback=feedback, user_turns=1))
    assert result['reward'] == 0 and result['final_reward'] == 1
    assert len(result['user_feedback']) == 1 and len(result['submissions']) == 2
    assert "SECRET_REFERENCE" not in str([c.sent for c in FakeContainer.instances])


def test_empty_harness_is_a_failed_candidate(setup):
    FakeContainer.events = [{"op": "end"}]
    result = asyncio.run(run_episode(*setup, "SYSTEM", container_factory=FakeContainer))
    assert result["stop_reason"] == "candidate_error" and result["reward"] == 0
    assert "no trainable" in result["error"]
    assert all(c.closed for c in FakeContainer.instances)


def test_tool_provenance_is_archived_but_not_sent_to_model(setup):
    seen = []
    task, harness, policy, folder = setup
    async def generate(messages):
        seen.append(messages)
        value = {'text': '<answer>42</answer>', 'finish_reason': 'stop'}
        policy.calls.append(value)
        return value
    policy.generate = generate
    FakeContainer.events = [
        {'op': 'generate', 'messages': [{'role': 'user', 'content': task.public['prompt']}]},
        {'op': 'execute', 'code': 'print(42)'},
        {'op': 'generate', 'messages': [{'role': 'user', 'content': 'Arbitrary tool-result wording: 42', 'source': 'tool'}]},
        {'op': 'submit', 'text': '<answer>42</answer>'},
    ]
    result = asyncio.run(run_episode(*setup, 'SYSTEM', container_factory=FakeContainer))
    assert all(set(m) == {'role', 'content'} for messages in seen for m in messages)
    assert result['calls'][1]['input_provenance'][0]['source'] == 'tool'
    assert result['reward'] == 1


def test_early_end_without_answer_is_explicit_diagnostic(setup):
    FakeContainer.events = [
        {'op': 'generate', 'messages': [{'role': 'user', 'content': 'Compute 6*7'}]}, {'op': 'end'}]
    result = asyncio.run(run_episode(*setup, 'SYSTEM', container_factory=FakeContainer))
    assert result['stop_reason'] == 'no_submission' and result['reward'] == 0


def test_configured_tool_timeout_ends_attempt_and_cleans(setup, monkeypatch):
    async def execute(self, code, timeout=120):
        assert timeout == 60
        raise TimeoutError

    monkeypatch.setattr(FakeContainer, "execute", execute)
    FakeContainer.events = [{"op": "generate", "messages": [{"role": "user", "content": "Compute 6*7"}]},
                            {"op": "execute", "code": "while True: pass"}]
    result = asyncio.run(run_episode(*setup, "SYSTEM", tool_seconds=60, container_factory=FakeContainer))
    assert result["stop_reason"] == "tool_timeout"
    assert result["reward"] == 0
    assert all(c.closed for c in FakeContainer.instances)


def test_broker_archives_raw_output_but_sends_bounded_observation(setup, monkeypatch):
    async def execute(self, code, timeout=120):
        return {"stdout": "x" * 20000, "error": None}
    monkeypatch.setattr(FakeContainer, "execute", execute)
    FakeContainer.events = [
        {"op": "generate", "messages": [{"role": "user", "content": "Compute 6*7"}]},
        {"op": "execute", "code": "print(data)"},
        {"op": "submit", "text": "<answer>42</answer>"},
    ]
    result = asyncio.run(run_episode(*setup, "SYSTEM", container_factory=FakeContainer))
    tool = result["tool_calls"][0]
    assert len(tool["observation"]["stdout"]) == 6000
    raw = json.loads((setup[3] / tool["raw_observation"]).read_text())
    assert len(raw["stdout"]) == 20000
    assert "x" * 6001 not in str(FakeContainer.instances[0].sent)


def test_generated_exception_after_model_call_is_a_failed_candidate(setup):
    FakeContainer.events = [{'op': 'generate', 'messages': [{'role': 'user', 'content': 'Question', 'source': 'harness'}]},
                            {'op': 'error', 'error': 'NameError: undefined helper'}]
    result = asyncio.run(run_episode(*setup, 'SYSTEM', container_factory=FakeContainer))
    assert result['stop_reason'] == 'candidate_error' and result['reward'] == 0
    assert len(result['calls']) == 1 and 'NameError' in result['error']
    assert all(c.closed for c in FakeContainer.instances)


def test_observation_limit_is_explicit_and_preserves_original():
    raw = {"stdout": "x" * 20000, "error": "y" * 10000}
    bounded = bounded_observation(raw)
    assert len(bounded["stdout"]) + len(bounded["error"]) <= 8000
    assert "TRUNCATED" in bounded["stdout"] and "TRUNCATED" in bounded["error"]
    assert len(raw["stdout"]) == 20000


def test_one_tool_oom_does_not_cancel_other_samples_in_task_group(setup):
    task, harness, _, folder = setup
    class ScopedContainer(FakeContainer):
        def __init__(self, image, path, kind, **kwargs):
            super().__init__(image, path, kind, **kwargs)
            from pathlib import Path
            self.oom = Path(path).parent.name == 'oom'
            self.script = iter([
                {'op': 'generate', 'messages': [{'role': 'user', 'content': task.public['prompt']}]},
                {'op': 'execute', 'code': 'bad_dataframe_broadcast' if self.oom else 'print(42)'},
                {'op': 'submit', 'text': '<answer>42</answer>'}])
        async def receive(self):
            return next(self.script)
        async def execute(self, code, timeout=120):
            await asyncio.sleep(0)
            if self.oom:
                raise ExecutionMemoryBudget({'OOMKilled': True, 'Running': False, 'ExitCode': 137})
            return {'stdout': '42', 'error': None}
    def policy():
        value = SimpleNamespace(calls=[], max_tokens=4096, context=32768)
        async def generate(messages):
            call = {'text': 'solver output', 'finish_reason': 'stop'}
            value.calls.append(call)
            return call
        value.generate = generate
        return value
    async def check():
        async with asyncio.TaskGroup() as group:
            failed = group.create_task(run_episode(task, harness, policy(), folder/'oom', 'SYSTEM', container_factory=ScopedContainer))
            healthy = group.create_task(run_episode(task, harness, policy(), folder/'healthy', 'SYSTEM', container_factory=ScopedContainer))
        return failed.result(), healthy.result()
    failed, healthy = asyncio.run(check())
    assert failed['stop_reason'] == 'execution_memory_budget' and failed['reward'] == 0
    assert failed['execution_failure']['OOMKilled'] is True and len(failed['calls']) == 1
    assert failed['tool_calls'][0]['code'] == 'bad_dataframe_broadcast'
    assert healthy['reward'] == 1 and healthy['stop_reason'] == 'first_answer_evaluation'
