import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from simple_scibuddy.coevolve.context import context_bundle, execution_fields, prm_schema, validate_assessment
from simple_scibuddy.coevolve.evidence import public_evidence
from simple_scibuddy.coevolve.feedback import review_submission
from simple_scibuddy.coevolve.program import propose


def assessment(reply):
    return {'feedback_type': 'correction', 'reward': -1, 'evidence': reply,
            'user_request': {'interpretation': 'Recheck after a tool error', 'quote': reply,
                             'specific_issue_explicit': False},
            'observations': [{'field': 'tools/0/observation', 'quote': 'ModuleNotFoundError',
                              'interpretation': 'The tool failed; its causal role is uncertain.'}],
            'related_fields': ['outputs/0/text', 'tools/0/code', 'tools/0/observation', 'answer'],
            'uncertainty': 'A failed tool does not prove why the answer is wrong.'}


def response(value):
    return {'choices': [{'message': {'content': json.dumps(value)}, 'finish_reason': 'stop'}]}


def test_full_public_inputs_and_grounded_prm_annotations_reach_improver(tmp_path):
    question = 'QUESTION_START' + 'x' * 7000 + 'QUESTION_END'
    answer = 'ANSWER_START' + 'y' * 7000 + 'ANSWER_END'
    code = 'import missing_package'
    observation = {'stdout': 'z' * 9500 + 'TAIL_EVIDENCE', 'error': 'ModuleNotFoundError'}
    calls = [{'text': 'Inspect the file', 'private_reference': 'NEVER_EXPOSE', 'completion_ids': []}]
    tools = [{'code': code, 'observation': observation, 'call_index': 0,
              'private_reference': 'NEVER_EXPOSE'}]
    messages_seen = []

    def generate(messages, schema, **kwargs):
        messages_seen.append(copy.deepcopy(messages))
        payload = json.loads(messages[1]['content'])
        if 'allowed_replies' in payload:
            assert payload['task'] == question and payload['answer'] == answer
            return response({'reply': payload['allowed_replies'][0]})
        fields = payload['fields']
        assert fields['question'] == question and fields['answer'] == answer
        assert json.loads(fields['tools/0/observation']) == observation
        assert 'NEVER_EXPOSE' not in json.dumps(payload)
        from jsonschema import Draft202012Validator
        result = assessment(fields['reply'])
        Draft202012Validator(schema).validate(result)
        return response(result)

    task = SimpleNamespace(verify=lambda text: {'passed': False, 'answer_format_valid': True,
                                               'private_reference': 'NEVER_EXPOSE'})
    feedback = asyncio.run(review_submission(task, SimpleNamespace(generate_command=generate),
                                             {'prompt': question}, calls, tools, answer))
    assert feedback['valid'] and not feedback['accept']
    assert len(messages_seen) == 2
    episode = {'task_id': 'a', 'public_task': {'prompt': question}, 'calls': calls,
               'tool_calls': tools, 'submissions': [{'text': answer, 'call_index': 0,
                                                    'outcome': {'answer': 'NEVER_EXPOSE'}}],
               'user_feedback': [feedback], 'stop_reason': 'first_answer_evaluation'}
    record = public_evidence(episode)
    assert 'NEVER_EXPOSE' not in json.dumps(record)
    annotation = record['prm_annotations'][0]
    assert annotation['assessment']['observations'][0]['quote'] == 'ModuleNotFoundError'
    assert 'submissions/0/text' in annotation['assessment']['related_fields']
    other = copy.deepcopy(record)
    other['task_id'] = 'b'
    bundle, readable = context_bundle([record, other], [], 'parent.py')
    assert readable[0]['submissions/0/text'] == answer
    for field in annotation['assessment']['related_fields']:
        assert field in readable[0]
    assert 'tool_error' in bundle['shared_observed_patterns']

    parent = tmp_path / 'parent.py'
    parent.write_text('def run(task, api): pass\n')
    improver_inputs = []

    def improve(messages, schema):
        improver_inputs.append(copy.deepcopy(messages))
        if len(improver_inputs) == 1:
            payload = json.loads(messages[1]['content'])
            assert payload['cross_task_overview'][0]['prm_annotations']
            assert payload['public_environment']['files'] == ['public.parquet']
            return response({'command': 'read', 'record': 0, 'field': 'tools/0/observation', 'offset': 0})
        page = json.loads(messages[-1]['content'])
        assert page['content'].endswith('"error": "ModuleNotFoundError"}')
        assert 'TAIL_EVIDENCE' in page['content'] and len(page['content']) > 9500
        assert page['truncated'] is False
        return response({'command': 'propose', 'code': 'def run(task, api):\n api.submit(api.generate([]))\n',
                         'hypothesis': 'Check imports before the lookup', 'reason_short': 'Shared tool errors',
                         'evidence_ids': ['e0:0', 'e1:0']})

    result = propose(SimpleNamespace(generate_command=improve), parent, [record, other], [],
                     tmp_path/'candidate.py', environment={'files': ['public.parquet']})
    assert result['status'] == 'validated'


@pytest.mark.parametrize('corrupt', ['field', 'classification', 'extra', 'boolean_reward', 'reply_as_execution'])
def test_prm_still_checks_structure_field_addresses_and_classification(corrupt):
    fields = execution_fields('question', [{'text': 'output'}],
                              [{'code': 'code', 'observation': {'error': 'ModuleNotFoundError'}}])
    fields.update(answer='answer', reply='Check again')
    value = assessment(fields['reply'])
    if corrupt == 'field':
        value['observations'][0]['field'] = 'private_reference'
    elif corrupt == 'classification':
        value['feedback_type'] = 'acceptance'
    elif corrupt == 'boolean_reward':
        value['reward'] = True
    elif corrupt == 'reply_as_execution':
        value['observations'][0].update(field='reply', quote=fields['reply'])
    else:
        value['private_reference'] = 'not allowed'
    assert validate_assessment(value, fields, fields['reply'], 'correction')[0] is False


def test_quote_format_differences_no_longer_drop_prm_annotation():
    fields = execution_fields('question', [{'text': 'output'}],
                              [{'code': 'code', 'observation': {'error': '**ModuleNotFoundError**'}}])
    fields.update(answer='answer', reply='Check again')
    value = assessment(fields['reply'])
    value['evidence'] = 'The user says: Check again'
    value['user_request']['quote'] = 'Check again.'
    value['observations'][0]['quote'] = 'ModuleNotFoundError occurred'
    assert validate_assessment(value, fields, fields['reply'], 'correction') == (True, None)


def test_improver_history_is_complete_and_read_addresses_are_record_specific():
    from jsonschema import Draft202012Validator

    from simple_scibuddy.coevolve.command_schema import command_schema
    records = [{'task_id': 'a', 'trace': [{'response': 'a'}]}, {'task_id': 'b', 'trace': []}]
    history = [{'step': i, 'status': 'applied' if i == 2 else 'skipped', 'reason': str(i)} for i in range(25)]
    bundle, readable = context_bundle(records, history, 'parent.py')
    assert len(bundle['modification_history']['recent_attempts']) == 20
    assert bundle['modification_history']['active_harness_reason']['step'] == 2
    assert json.loads(readable[2]['history']) == history
    schema = command_schema(2, ['e0:0', 'e1:0'], allow_read=True, readable=readable)
    Draft202012Validator(schema).validate({'command': 'read', 'record': 0, 'field': 'outputs/0/text', 'offset': 0})
    Draft202012Validator(schema).validate({'command': 'read', 'record': 2, 'field': 'history', 'offset': 0})
    assert not Draft202012Validator(schema).is_valid({'command': 'read', 'record': 1, 'field': 'outputs/0/text', 'offset': 0})
    assert not Draft202012Validator(prm_schema({'question': 'public'})).is_valid({})


def test_valid_acceptance_is_a_control_and_not_a_correction():
    reply = 'That answers my question, thank you.'
    value = assessment(reply)
    value.update(feedback_type='acceptance', reward=1, observations=[], related_fields=['answer'])
    feedback = {'valid': True, 'feedback_type': 'acceptance', 'reply': reply, 'assessment': value}
    record = public_evidence({'task_id': 'accepted', 'calls': [{'text': '<answer>A</answer>', 'completion_ids': []}],
                              'public_task': {'prompt': 'Question'}, 'tool_calls': [],
                              'submissions': [{'text': '<answer>A</answer>', 'call_index': 0}],
                              'user_feedback': [feedback], 'stop_reason': 'first_answer_evaluation'})
    assert record['feedback'] == [] and record['acceptance'] == [{'reply': reply}]
    assert record['prm_annotations'][0]['assessment']['feedback_type'] == 'acceptance'
