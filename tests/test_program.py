import json
from types import SimpleNamespace

import pytest

from simple_scibuddy.coevolve.program import propose, validate_program


def fixtures(tmp_path):
    parent = tmp_path / 'parent.py'
    parent.write_text('def run(task, api): pass\n')
    records = [{'task_id': str(i), 'feedback': [{'reply': 'Please check units.'}],
                'diagnostics': [], 'trace': [{'response': 'ACTUAL EXECUTION EVIDENCE'}]} for i in range(2)]
    valid = {'command': 'propose', 'reason_short': 'Check units', 'hypothesis': 'Use explicit units',
             'evidence_ids': ['e0:0', 'e1:0'], 'code': 'def run(task, api): pass\n'}
    return parent, records, valid


def client(responses, observed):
    responses = iter(responses)
    def generate(messages, schema):
        observed.append((json.loads(json.dumps(messages)), schema))
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(next(responses))}}]}
    return SimpleNamespace(generate_command=generate)


def test_program_validation_never_executes_generated_top_level_code(tmp_path):
    marker = tmp_path/'host-marker'
    program = tmp_path/'harness.py'
    program.write_text(f"open({str(marker)!r}, 'w').write('bad')\ndef run(task, api): pass\n")
    assert validate_program(program)
    assert not marker.exists()
    for code in ['def wrong(task, api): pass', 'async def run(task, api): pass', 'def run(task): pass']:
        program.write_text(code)
        with pytest.raises(ValueError):
            validate_program(program)


@pytest.mark.parametrize("discriminator", ["command", "wrapper"])
def test_full_file_proposal_repairs_interface_and_preserves_evidence(tmp_path, discriminator):
    parent = tmp_path/'parent.py'
    parent.write_text('def run(task, api): pass\n')
    records = [{'task_id': str(i), 'feedback': [{'reply': 'Please check units.'}],
                'diagnostics': [], 'trace': []} for i in range(2)]
    base = {'command': 'propose', 'reason_short': 'Check repeated unit errors',
            'hypothesis': 'Use a reusable unit conversion helper', 'evidence_ids': ['e0:0', 'e1:0']}
    source = "def convert(x): return x * 1000\ndef run(task, api):\n text = api.generate([{'role':'user','content':task['prompt']}])\n api.submit(text)\n"
    payloads = [dict(base, code='def wrong(task, api): pass'), dict(base, code=source)]
    if discriminator == 'type':
        payloads = [dict(type=p.pop('command'), **p) for p in payloads]
    elif discriminator == 'wrapper':
        payloads = [{p.pop('command'): p} for p in payloads]
    responses = iter(payloads)
    client = SimpleNamespace(generate_command=lambda *a: {'choices': [{'message': {'content': json.dumps(next(responses))}}]})
    candidate = tmp_path/'candidate.py'
    result = propose(client, parent, records, [], candidate)
    assert result['status'] == 'validated' and candidate.read_text() == source
    assert result['evidence'][0]['quote'] == 'Please check units.'
    assert json.loads((tmp_path/'proposal.json').read_text())['repairs'] == 1


def test_recorded_wrapper_pattern_can_read_then_propose_without_repairs(tmp_path):
    parent, records, valid = fixtures(tmp_path)
    wrapped = {k: v for k, v in valid.items() if k != 'command'}
    observed = []
    responses = [{'read': {'record': 0, 'field': 'trace', 'offset': 0}}, {'propose': wrapped}]
    path = tmp_path / 'candidate.py'
    result = propose(client(responses, observed), parent, records, [], path)
    assert result['status'] == 'validated'
    assert 'ACTUAL EXECUTION EVIDENCE' in observed[1][0][-1]['content']
    assert all('command' in variant['required'] for variant in observed[0][1]['anyOf'])
    saved = json.loads((tmp_path / 'proposal.json').read_text())
    assert saved['reads'] == 1 and saved['repairs'] == 0
    assert len(saved['normalizations']) == 2


def test_envelope_repair_does_not_block_a_later_read(tmp_path):
    parent, records, valid = fixtures(tmp_path)
    observed = []
    responses = [{'read': {'field': 'trace', 'offset': 0}},
                 {'command': 'read', 'record': 1, 'field': 'trace', 'offset': 0}, valid]
    result = propose(client(responses, observed), parent, records, [], tmp_path / 'candidate.py')
    assert result['status'] == 'validated'
    assert 'record' in observed[1][0][-1]['content']
    assert any(v['properties']['command']['const'] == 'read' for v in observed[1][1]['anyOf'])
    assert 'ACTUAL EXECUTION EVIDENCE' in observed[2][0][-1]['content']
    saved = json.loads((tmp_path / 'proposal.json').read_text())
    assert saved['reads'] == 1 and saved['repairs'] == 1


def test_candidate_repair_still_disallows_further_reads(tmp_path):
    parent, records, valid = fixtures(tmp_path)
    observed = []
    responses = [dict(valid, code='def wrong(task, api): pass'),
                 {'command': 'read', 'record': 0, 'field': 'trace', 'offset': 0}, valid]
    result = propose(client(responses, observed), parent, records, [], tmp_path / 'candidate.py')
    assert result['status'] == 'validated'
    assert all(v['properties']['command']['const'] != 'read' for v in observed[1][1]['anyOf'])
    saved = json.loads((tmp_path / 'proposal.json').read_text())
    assert saved['reads'] == 0 and saved['repairs'] == 2


@pytest.mark.parametrize('value', [
    {'propose': {}, 'skip': {'reason': 'ambiguous'}},
    {'read': {'command': 'skip', 'record': 0, 'field': 'trace', 'offset': 0}},
    {'command': 'read', 'record': True, 'field': 'trace', 'offset': 0},
    {'command': 'read', 'record': 2, 'field': 'trace', 'offset': 0},
    {'command': 'read', 'record': 0, 'field': 'private_reference', 'offset': 0},
    {'command': 'skip', 'reason': 'done', 'extra': 'unexpected'},
])
def test_ambiguous_or_invalid_commands_are_not_silently_accepted(value):
    from simple_scibuddy.coevolve.command_schema import CommandFormatError, command_schema, parse_command
    with pytest.raises(CommandFormatError):
        parse_command(value, command_schema(2, ['e0:0', 'e1:0'], allow_read=True))


def test_wrapped_skip_is_respected(tmp_path):
    parent, records, _ = fixtures(tmp_path)
    result = propose(client([{'skip': {'reason': 'No supported change'}}], []), parent, records,
                     [], tmp_path / 'candidate.py')
    assert result == {'status': 'skipped', 'reason': 'No supported change'}
    assert not (tmp_path / 'candidate.py').exists()


def test_concatenated_commands_receive_repair(tmp_path):
    parent, records, valid = fixtures(tmp_path)
    responses = iter(['{"read": {}}{"propose": {}}', json.dumps(valid)])
    model = SimpleNamespace(generate_command=lambda *args: {'choices': [{'message': {'content': next(responses)}}]})
    result = propose(model, parent, records, [], tmp_path/'candidate.py')
    assert result['status'] == 'validated'
    assert json.loads((tmp_path/'proposal.json').read_text())['repairs'] == 1
