"""Full Python harness proposals; parsing on the host, execution only in containers."""

import ast
import copy
import json
from pathlib import Path

from simple_scibuddy.artifacts import write_json
from simple_scibuddy.coevolve.command_schema import CommandFormatError, command_schema, parse_command
from simple_scibuddy.coevolve.context import context_bundle
from simple_scibuddy.coevolve.evidence import evidence_preview

INTERFACE = """Write one self-contained Python file exporting synchronous run(task, api).
The controller has Python's standard library; do not import our repository or use network/credentials.
The task dictionary has prompt, family, subtask, files (basenames), system (environment guidance),
and budgets: model_calls, tool_calls, completion_tokens, context_tokens, seconds, tool_seconds,
user_turns, tool_response_tokens, tool_history_tokens.
api.generate(messages) returns a model response string. messages is a list of role/content dictionaries.
The host ALWAYS provides nonempty task.system as environment guidance. Explicitly append your new instructions:
system = task['system'] + '\\n' + YOUR_INSTRUCTIONS. A fallback such as task.get('system') or NEW_SYSTEM
silently discards NEW_SYSTEM and is rejected. Inspect actual inputs, not merely unused prompt constants.
Messages may carry source=harness/task/model/tool/feedback. Tool evidence MUST use source='tool', even when
summarized or renamed. User-role instructions and format/budget notices MUST use source='harness'.
source='task' is only the original task.prompt; source='feedback' is only the actual reply from api.submit.
Metadata is validated and stripped by the host before model tokenization; the model and RL receive role/content only.
api.execute(code) runs Python in a separate persistent execution container and returns a JSON observation
with stdout and error. Task files exist there under /workspace/assets, not in the controller.
api.submit(text) submits an answer to the host grader. Preserve the task's required answer format,
including <answer>...</answer>. When interaction continues it returns {done: false, reply: string};
append that reply and continue. Evaluation stops at the first submission. Return when done.
Call api.generate at least once for every task: only those model tokens can receive RL gradients.
You may redesign instructions, parsing, control flow, memory and tool orchestration within these interfaces.
Read budgets from task.budgets; do not impose an arbitrary two-tool limit on all tasks. Reserve a model call
for final submission. When tools are exhausted, say so rather than report a format error. If the model keeps
requesting tools, terminate within the original budget; never fabricate an answer or loop on the same notice.
Do not print protocol records or use stdin/stdout directly. Do not add dependencies or change host budgets,
grading, rewards, loss masks, isolation, or sampling settings. Generated Python is not a training target.
The host records the actual model context and enforces all budgets, regardless of your implementation.
"""


def validate_program(path):
    source = Path(path).read_text()
    tree = ast.parse(source)
    entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run']
    if len(entries) != 1:
        raise ValueError('Provide one synchronous run(task, api) function')
    args = entries[0].args
    if len(args.posonlyargs + args.args) != 2 or args.vararg or args.kwarg or args.kwonlyargs:
        raise ValueError('run must accept exactly task and api')
    for node in ast.walk(tree):
        if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
            continue
        for position, value in enumerate(node.values[:-1]):
            if all(isinstance(fallback, ast.Constant) and fallback.value in ('', None)
                   for fallback in node.values[position + 1:]):
                continue
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name) and value.func.value.id == 'task'
                    and value.func.attr == 'get' and value.args
                    and isinstance(value.args[0], ast.Constant) and value.args[0].value == 'system'):
                raise ValueError('task.system is always supplied: explicitly append new instructions; '
                                 'task.get("system") or NEW_SYSTEM hides the new prompt')
    return source


def propose(improver, parent, records, history, path, profile=None, environment=None):
    bundle, readable = context_bundle(records, history, parent)
    catalog = {}
    for i, record in enumerate(records):
        quotes = [f['reply'] for f in record['feedback']] + record['diagnostics']
        for j, quote in enumerate(dict.fromkeys(quotes)):
            catalog[f'e{i}:{j}'] = {'task_id': record['task_id'], 'quote': quote}
    if len({e['task_id'] for e in catalog.values()}) < 2:
        return {'status': 'skipped', 'reason': 'Need evidence from two distinct training tasks'}
    messages = [{'role': 'system', 'content':
        'Analyze the target model\'s trajectories and rewrite its harness to address a recurring problem. '
        'Propose a testable hypothesis, not a guaranteed fix. Compare failures and successful controls. '
        'Generic dissatisfaction does not establish a scientific cause. Do not invent correct answers. '
        'Use public evidence from at least two distinct tasks, not validation/test tasks or private labels. '
        'Generate one complete replacement file, not a diff. You may change the harness substantially. '
        'The parent remains available in validation selection. Return a JSON command: read, propose, or skip. '
        'Use a flat object with a top-level command field, not a wrapper named read/propose/skip. '
        'Read example: {"command":"read","record":0,"field":"trace","offset":0}. '
        'Skip example: {"command":"skip","reason":"Explain why"}. '
        'Propose shape: {"command":"propose","code":"...","hypothesis":"...",'
        '"reason_short":"...","evidence_ids":["catalog-id-1","catalog-id-2"]}. '
        'Use actual evidence IDs from the catalog, not these example IDs. '
        'For read provide record (integer index), an exact field from readable_fields, and offset (normally 0). '
        'Read individual outputs/i/text, tools/i/code or tools/i/observation to inspect the relevant step in full. '
        'Use inputs/i/messages to inspect what the solver ACTUALLY received, with inputs/i/provenance and '
        'inputs/i/tool_history_clipping for host processing. Prompt constants alone do not prove an input change. '
        'The host returns the complete selected field from offset onward, without a 6000-character clip. '
        'Use the public_environment inventory; do not infer missing scientific data from a missing package or failed guess. '
        'Inspect cross_task_overview, shared_observed_patterns and modification_history before editing. '
        'PRM annotations locate user intent and quoted execution evidence. Their interpretations are hypotheses, '
        'not verified causes. Quote text is not matched by the host; use related_fields to inspect originals and compare distinct tasks. '
        'Generic corrections do not establish a specific scientific error; acceptance does not validate every step. '
        'Original fields remain accessible even when PRM annotations are absent or invalid. '
        'For propose provide code, hypothesis, reason_short, evidence_ids (2-4 catalog IDs). '
        'For skip provide reason. At most three reads, two repairs, six calls.\n' + INTERFACE},
        {'role': 'user', 'content': json.dumps({'parent_code': validate_program(parent), 'solver': profile,
            **bundle, 'public_environment': environment or {}, 'evidence': catalog,
            'records': [{'record': i, **evidence_preview(r)} for i, r in enumerate(records)]})}]
    initial = copy.deepcopy(messages)
    transcript, reads, repairs = [], 0, 0
    schemas, normalizations, errors = [], [], []
    repairing_proposal = False
    result = {'status': 'skipped', 'reason': 'Proposal call budget exhausted'}
    for call in range(6):
        schema = command_schema(len(records), catalog, allow_read=reads < 3 and not repairing_proposal,
                                readable=readable)
        schemas.append(schema)
        try:
            response = improver.generate_command(messages, schema)
            transcript.append(response)
            choice = response['choices'][0]
            if choice.get('finish_reason') == 'length':
                raise CommandFormatError('Truncated response; return a shorter complete JSON command.')
            content = choice['message']['content']
            messages.append({'role': 'assistant', 'content': content})
            try:
                value = json.loads(content)
            except (ValueError, TypeError) as exc:
                raise CommandFormatError('Invalid JSON; return one complete JSON object without markdown fences.') from exc
            command, normalized = parse_command(value, schema)
            if normalized:
                normalizations.append({'call': call, 'command': command['command']})
            if command['command'] == 'skip':
                result = {'status': 'skipped', 'reason': str(command['reason'])}
                break
            if command['command'] == 'read':
                i, field, offset = command['record'], command['field'], command['offset']
                reads += 1
                page = readable[i][field]
                messages.append({'role': 'user', 'content': json.dumps({'record': i, 'field': field,
                    'offset': offset, 'total_characters': len(page), 'content': page[offset:],
                    'truncated': False}, ensure_ascii=False)})
                continue
            if command['command'] != 'propose':
                raise ValueError('Unknown command')
            repairing_proposal = True
            ids = command['evidence_ids']
            if (not isinstance(ids, list) or not 2 <= len(ids) <= 4
                    or any(not isinstance(i, str) or i not in catalog for i in ids)
                    or len({catalog[i]['task_id'] for i in ids}) < 2):
                raise ValueError('Cite 2-4 evidence IDs from two distinct tasks')
            for key in ('code', 'hypothesis', 'reason_short'):
                if not isinstance(command[key], str) or not command[key].strip():
                    raise ValueError('Missing ' + key)
            Path(path).write_text(command['code'])
            validate_program(path)
            result = {'status': 'validated', 'hypothesis': command['hypothesis'],
                      'reason': command['reason_short'], 'evidence': [catalog[i] for i in ids]}
            break
        except (ValueError, SyntaxError, KeyError, TypeError, IndexError) as exc:
            kind = 'command_format' if isinstance(exc, CommandFormatError) else 'candidate_validation'
            errors.append({'call': call, 'kind': kind, 'reason': str(exc)})
            if repairs == 2:
                result = {'status': 'skipped', 'reason': str(exc)}
                break
            repairs += 1
            instruction = ('Repair only the JSON command format; retain its intended content. '
                           'Use the top-level command field and the supplied schema. '
                           if kind == 'command_format' else 'Repair this same unevaluated candidate. ')
            instruction += ('No further reads; propose or skip.' if repairing_proposal or reads >= 3 else
                            'Evidence reads remain available within the original three-read limit.')
            messages.append({'role': 'user', 'content': json.dumps({'validation_error': str(exc),
                'repair_attempt': repairs, 'max_repairs': 2, 'instruction': instruction})})
        except RuntimeError as exc:
            result = {'status': 'skipped', 'reason': str(exc)}
            break
    if result['status'] != 'validated':
        Path(path).unlink(missing_ok=True)
    write_json(Path(path).parent/'proposal.json', {'result': result, 'input_messages': initial,
               'transcript': transcript, 'reads': reads, 'repairs': repairs, 'command_schemas': schemas,
               'normalizations': normalizations, 'validation_errors': errors})
    return result
