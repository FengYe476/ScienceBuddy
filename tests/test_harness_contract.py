import copy
from types import SimpleNamespace

import pytest

from simple_scibuddy.coevolve.program import validate_program
from simple_scibuddy.environments.runtime import ControllerFailure
from simple_scibuddy.harness import context, scientific
from simple_scibuddy.harness.messages import classify_messages, model_messages


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        return ''.join(map(chr, tokens))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(''.join(m['content'] for m in messages))


def test_new_instruction_is_visible_even_with_host_system(monkeypatch):
    monkeypatch.setattr(scientific, 'HARNESS_INSTRUCTIONS', 'ACTUAL_NEW_LOOKUP_INSTRUCTION')
    messages = scientific.build_messages({'prompt': 'Question', 'system': 'HOST_ENVIRONMENT'})
    assert messages[0]['content'] == 'HOST_ENVIRONMENT\nACTUAL_NEW_LOOKUP_INSTRUCTION'
    assert model_messages(messages)[1] == {'role': 'user', 'content': 'Question'}


def test_shadowed_prompt_is_rejected_without_executing_generated_code(tmp_path):
    marker = tmp_path/'should-not-exist'
    p = tmp_path/'candidate.py'
    p.write_text(f"open({str(marker)!r}, 'w').write('bad')\nNEW_SYSTEM = 'new instructions'\n"
                 "def run(task, api):\n system = task.get('system') or NEW_SYSTEM\n")
    with pytest.raises(ValueError, match='explicitly append'):
        validate_program(p)
    assert not marker.exists()
    p.write_text("NEW_SYSTEM = 'new instructions'\ndef run(task, api):\n system = task['system'] + '\\n' + NEW_SYSTEM\n")
    assert validate_program(p)
    p.write_text("NEW_SYSTEM = 'new instructions'\ndef run(task, api):\n system = (task.get('system') or '') + NEW_SYSTEM\n")
    assert validate_program(p)


@pytest.mark.parametrize('prefix,explicit', [('Unfamiliar wording: ', True), ('Tool observation (new words): ', False)])
def test_tool_history_budget_is_independent_of_display_wording(prefix, explicit):
    messages = [{'role': 'system', 'content': 'Keep instructions'}, {'role': 'user', 'content': 'Question'}]
    for i in range(5):
        m = {'role': 'user', 'content': prefix + str(i) * 1900}
        if explicit:
            m['source'] = 'tool'
        messages.append(m)
    tagged, audit = classify_messages(messages, 'Question', [], 5)
    bounded, changes = context.bound_tool_history(tagged, Tokenizer(), per_message=2048, total=8192, input_limit=20000)
    assert sum(len(m['content']) for m in bounded if m['source'] == 'tool') <= 8192
    assert changes and all(a['source'] == 'tool' for a in audit[2:])
    assert bounded[0]['content'] == 'Keep instructions'
    assert all(set(m) == {'role', 'content'} for m in model_messages(bounded))
    assert messages[-1]['content'] == prefix + '4' * 1900


@pytest.mark.parametrize('message,tools', [
    ({'role': 'user', 'content': 'Some arbitrary result'}, 1),
    ({'role': 'user', 'content': 'fabricated reply', 'source': 'feedback'}, 1),
    ({'role': 'user', 'content': 'not the question', 'source': 'task'}, 1),
    ({'role': 'assistant', 'content': 'tool data', 'source': 'tool'}, 1),
    ({'role': 'user', 'content': 'tool data', 'source': 'tool'}, 0),
])
def test_provenance_is_checked_not_just_an_optional_hint(message, tools):
    with pytest.raises(ControllerFailure):
        classify_messages([message], 'Question', ['Actual feedback'], tools)


def scripted_api(responses):
    replies = iter(responses)
    state = SimpleNamespace(inputs=[], executions=[], submissions=[])
    def generate(messages):
        state.inputs.append(copy.deepcopy(messages))
        return next(replies)
    def execute(code):
        state.executions.append(code)
        return {'stdout': 'tool result', 'error': None}
    def submit(answer):
        state.submissions.append(answer)
        return {'done': True}
    return SimpleNamespace(generate=generate, execute=execute, submit=submit), state


def test_three_step_lookup_is_not_blocked_by_a_two_tool_cap():
    api, state = scripted_api(['<execute>print(1)</execute>', '<execute>print(2)</execute>',
                               '<execute>print(3)</execute>', '<answer>A</answer>'])
    scientific.run({'prompt': 'Question', 'budgets': {'model_calls': 9, 'tool_calls': 9}}, api)
    assert len(state.executions) == 3 and state.submissions == ['<answer>A</answer>']
    assert all(m.get('source') == 'tool' for m in state.inputs[-1] if m['content'].startswith('Tool observation'))


def test_tool_exhaustion_sends_a_budget_notice_and_terminates_if_ignored():
    api, state = scripted_api(['<execute>print(1)</execute>'] * 9)
    scientific.run({'prompt': 'Question', 'budgets': {'model_calls': 9, 'tool_calls': 2}}, api)
    assert len(state.executions) == 2 and len(state.inputs) == 3
    assert 'budget notice' in state.inputs[-1][-1]['content']
    assert 'format notice' not in state.inputs[-1][-1]['content']
    assert not state.submissions, 'Never manufacture an answer on behalf of the solver'


def test_final_model_call_is_reserved_for_submission():
    api, state = scripted_api(['<execute>print(1)</execute>', '<execute>print(2)</execute>', '<answer>A</answer>'])
    scientific.run({'prompt': 'Question', 'budgets': {'model_calls': 3, 'tool_calls': 9}}, api)
    assert len(state.inputs) == 3 and len(state.executions) == 2
    assert 'last model call is reserved' in state.inputs[-1][-1]['content']
    assert state.submissions == ['<answer>A</answer>']
