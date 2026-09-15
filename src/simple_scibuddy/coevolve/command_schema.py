"""Explicit read/propose/skip protocol for full-program harness proposals."""

from jsonschema import Draft202012Validator

FIELDS = ('question', 'trace', 'tools', 'feedback', 'diagnostics')
KINDS = ('read', 'propose', 'skip')


class CommandFormatError(ValueError):
    """The command envelope is invalid; no candidate has been evaluated."""


def object_schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def command_schema(record_count, evidence_ids, *, allow_read, readable=None):
    text = {'type': 'string', 'minLength': 1}
    variants = [object_schema({'command': {'const': 'propose'}, 'code': text,
        'hypothesis': text, 'reason_short': text,
        'evidence_ids': {'type': 'array', 'minItems': 2, 'maxItems': 4, 'uniqueItems': True,
                         'items': {'enum': list(evidence_ids)}}}),
        object_schema({'command': {'const': 'skip'}, 'reason': text})]
    if allow_read:
        read = object_schema({'command': {'const': 'read'},
            'record': {'type': 'integer', 'minimum': 0, 'maximum': record_count - 1},
            'field': {'enum': list(FIELDS)}, 'offset': {'type': 'integer', 'minimum': 0}})
        if readable is not None:
            read['properties']['record'] = {'type': 'integer', 'enum': list(readable)}
            read['properties']['field'] = {'enum': sorted({f for fields in readable.values() for f in fields})}
            read['allOf'] = [{'if': {'properties': {'record': {'const': i}}},
                             'then': {'properties': {'field': {'enum': list(fields)}}}}
                            for i, fields in readable.items()]
        variants.append(read)
    return {'anyOf': variants}


def parse_command(value, schema):
    """Accept flat commands and one unambiguous legacy wrapper; validate both identically."""
    if not isinstance(value, dict):
        raise CommandFormatError('Return a JSON object with a top-level command field.')
    normalized = False
    if 'command' not in value and len(value) == 1:
        kind, body = next(iter(value.items()))
        if kind in KINDS and isinstance(body, dict):
            if 'command' in body:
                raise CommandFormatError('Do not put another command inside a command wrapper.')
            value = {'command': kind, **body}
            normalized = True
    kind = value.get('command')
    if not isinstance(kind, str) or kind not in KINDS:
        raise CommandFormatError('Missing or invalid top-level command; use read, propose, or skip. '
                                 'Example: {"command":"read","record":0,"field":"trace","offset":0}.')
    branch = next((item for item in schema['anyOf'] if item['properties']['command']['const'] == kind), None)
    if branch is None:
        raise CommandFormatError('No further evidence reads allowed; propose or skip using the supplied evidence.')
    violation = next(Draft202012Validator(branch).iter_errors(value), None)
    if violation is not None:
        path = '.'.join(map(str, violation.absolute_path)) or 'command object'
        raise CommandFormatError(f'{kind}: {path}: {violation.message[:300]}')
    return value, normalized
