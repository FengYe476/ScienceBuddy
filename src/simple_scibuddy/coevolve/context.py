"""Public, addressable execution evidence for the feedback reader and improver."""

import json
from collections import Counter


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def execution_fields(question, calls, tools):
    """Project only public outputs; never serialize entire host episode/call objects."""
    fields = {'question': question}
    for i, call in enumerate(calls):
        fields[f'outputs/{i}/text'] = call.get('text', call.get('response', ''))
    for i, tool in enumerate(tools):
        fields[f'tools/{i}/code'] = tool.get('code', '')
        fields[f'tools/{i}/observation'] = text(tool.get('observation', {}))
    return fields


def trajectory_index(calls, tools):
    return [{'field': f'outputs/{i}/text', 'call_index': i} for i in range(len(calls))] + [
        {'code_field': f'tools/{i}/code', 'observation_field': f'tools/{i}/observation',
         'call_index': tool.get('call_index')} for i, tool in enumerate(tools)]


def prm_schema(fields):
    string = {'type': 'string'}
    nonempty = {'type': 'string', 'minLength': 1}
    def obj(properties):
        return {'type': 'object', 'properties': properties, 'required': list(properties),
                'additionalProperties': False}
    reference = {'type': 'string', 'enum': sorted(fields)}
    execution_fields = sorted(field for field in fields if field == 'answer' or field.startswith(('outputs/', 'tools/')))
    execution_reference = {'type': 'string', 'enum': execution_fields} if execution_fields else False
    return obj({
        'feedback_type': {'enum': ['acceptance', 'correction', 'ambiguous']},
        'reward': {'type': 'integer', 'enum': [1, -1, 0]},
        'evidence': nonempty,
        'user_request': obj({'interpretation': string, 'quote': nonempty,
                             'specific_issue_explicit': {'type': 'boolean'}}),
        'observations': {'type': 'array', 'items': obj({
            'field': execution_reference, 'quote': nonempty, 'interpretation': nonempty})},
        'related_fields': {'type': 'array', 'uniqueItems': True, 'items': reference},
        'uncertainty': nonempty,
    })


def validate_assessment(value, fields, reply, kind):
    """Check structure, field addresses and feedback labels; quote text is not matched."""
    from jsonschema import Draft202012Validator
    violation = next(Draft202012Validator(prm_schema(fields)).iter_errors(value), None)
    if violation:
        return False, 'schema: ' + violation.message
    if value['feedback_type'] != kind or value['reward'] != (1 if kind == 'acceptance' else -1):
        return False, 'Feedback classification conflicts with the allowed reply semantics'
    return True, None


def record_fields(record):
    fields = execution_fields(record.get('question', ''), record.get('trace', []), record.get('tools', []))
    for i, call in enumerate(record.get('trace', [])):
        fields[f'inputs/{i}/messages'] = text(call.get('input_messages', []))
        fields[f'inputs/{i}/provenance'] = text(call.get('input_provenance', []))
        fields[f'inputs/{i}/tool_history_clipping'] = text(call.get('tool_history_clipping', []))
    # Preserve old command compatibility, while directing new requests to single-step fields.
    for name in ('trace', 'tools', 'feedback', 'diagnostics'):
        fields[name] = text(record.get(name, []))
    for i, feedback in enumerate(record.get('feedback', [])):
        fields[f'feedback/{i}/reply'] = feedback['reply']
    for i, annotation in enumerate(record.get('prm_annotations', [])):
        fields[f'prm/{i}'] = text(annotation)
        index = annotation['feedback_index']
        fields[f'submissions/{index}/text'] = annotation['submission_text']
        fields[f'user_feedback/{index}/reply'] = annotation['reply']
    return fields


def context_bundle(records, history, parent):
    """RSI-style overview, precise field addresses, and full-history access."""
    readable = {i: record_fields(record) for i, record in enumerate(records)}
    patterns = {}
    overview = []
    for i, record in enumerate(records):
        flags = []
        tools = record.get('tools', [])
        if not tools:
            flags.append('no_tool_execution')
        if any(t.get('observation', {}).get('error') and not t.get('observation', {}).get('infrastructure_error')
               for t in tools):
            flags.append('tool_error')
        stop = record.get('execution', {}).get('stop_reason')
        if stop:
            flags.append('stop:' + stop)
        for flag in flags:
            patterns.setdefault(flag, []).append({'record': i, 'task_id': record['task_id']})
        overview.append({'record': i, 'task_id': record['task_id'], 'observed_patterns': flags,
                         'prm_annotations': [{k: v for k, v in annotation.items() if k != 'submission_text'}
                                             for annotation in record.get('prm_annotations', [])],
                         'trajectory_index': trajectory_index(record.get('trace', []), tools),
                         'fields': [{'field': field, 'characters': len(value)}
                                    for field, value in readable[i].items()
                                    if field not in {'trace', 'tools'}]})
    history_index = len(records)
    readable[history_index] = {'history': text(history)}
    applied = [h for h in history if h.get('status') == 'applied']
    summary = {'recent_attempts': history[-20:],
               'active_harness': str(parent),
               'active_harness_reason': applied[-1] if applied else {'source': 'initial harness'},
               'older_status_counts': dict(Counter(h.get('status') for h in history[:-20])),
               'full_history': {'record': history_index, 'field': 'history', 'offset': 0},
               'interpretation': 'Applied means selected on Val, not proof of a causal mechanism or Test gain.'}
    shared = {flag: rows for flag, rows in patterns.items() if len({r['task_id'] for r in rows}) >= 2}
    return {'cross_task_overview': overview, 'shared_observed_patterns': shared,
            'modification_history': summary,
            'readable_fields': {i: list(fields) for i, fields in readable.items()}}, readable
