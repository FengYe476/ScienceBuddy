import json

from simple_scibuddy.coevolve.evidence import preflight_passed, public_evidence


def test_preflight_uses_execution_not_accuracy():
    episode = {'stop_reason': 'first_answer_evaluation', 'reward': 0,
               'submissions': [{'outcome': {'answer_format_valid': True, 'passed': False}}]}
    assert preflight_passed(episode)
    assert not preflight_passed(dict(episode, error='infrastructure failure'))
    assert not preflight_passed(dict(episode, stop_reason='output_truncated'))


def test_feedback_quarantine_and_private_reference_exclusion():
    episode = {'task_id': 'x', 'stop_reason': 'context_budget', 'calls': [], 'reward': 0,
               'submissions': [{'outcome': {'answer': 'SECRET'}}], 'user_feedback': [
                   {'valid': False, 'feedback_type': 'correction', 'reply': 'BAD'},
                   {'valid': True, 'feedback_type': 'correction', 'reply': 'Check units', 'private': 'SECRET'}]}
    evidence = public_evidence(episode)
    assert evidence['feedback'] == [{'reply': 'Check units'}]
    assert evidence['diagnostics'] == ['context_budget']
    assert 'SECRET' not in json.dumps(evidence) and 'BAD' not in json.dumps(evidence)


def test_acceptance_controls_require_valid_public_feedback():
    episode = {'task_id': 'x', 'stop_reason': 'stop', 'calls': [], 'user_feedback': [
        {'valid': False, 'feedback_type': 'acceptance', 'reply': 'UNVERIFIED'},
        {'valid': True, 'feedback_type': 'acceptance', 'reply': 'Thank you.', 'private': 'SECRET'}]}
    evidence = public_evidence(episode)
    assert evidence['acceptance'] == [{'reply': 'Thank you.'}]
    assert not evidence['feedback']
    assert 'SECRET' not in json.dumps(evidence) and 'UNVERIFIED' not in json.dumps(evidence)
