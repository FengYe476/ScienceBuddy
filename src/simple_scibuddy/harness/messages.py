"""Host-side message provenance; internal metadata never reaches the model API."""

from simple_scibuddy.environments.runtime import ControllerFailure


def classify_messages(messages, task_prompt, replies, tool_count):
    if not isinstance(messages, list) or not messages:
        raise ControllerFailure('generate requires nonempty role/content text messages')
    result, audit = [], []
    allowed = {'system': {'harness'}, 'assistant': {'model'},
               'user': {'task', 'feedback', 'harness', 'tool'}, 'tool': {'tool'}}
    for index, message in enumerate(messages):
        if (not isinstance(message, dict) or message.get('role') not in allowed
                or not isinstance(message.get('content'), str)):
            raise ControllerFailure('generate requires nonempty role/content text messages')
        role, content = message['role'], message['content']
        source = message.get('source')
        legacy = False
        if source is None:
            if role in {'system', 'assistant'}:
                source = 'harness' if role == 'system' else 'model'
            elif role == 'tool':
                source = 'tool'
            elif content == task_prompt:
                source = 'task'
            elif content in replies:
                source = 'feedback'
            elif content.startswith('Tool observation'):
                source, legacy = 'tool', True
            else:
                raise ControllerFailure('Non-task user messages must declare source="tool" for tool evidence '
                                        'or source="harness" for instructions/notices; source="feedback" is host feedback.')
        if not isinstance(source, str) or source not in allowed[role]:
            raise ControllerFailure('Invalid message source for role ' + role)
        if source == 'tool' and not tool_count:
            raise ControllerFailure('Tool evidence cannot precede an actual execute call')
        if source == 'task' and content != task_prompt:
            raise ControllerFailure('source="task" must contain the original public question')
        if source == 'feedback' and content not in replies:
            raise ControllerFailure('source="feedback" must contain an actual host-returned user reply')
        result.append({'role': role, 'content': content, 'source': source})
        audit.append({'message_index': index, 'source': source, 'legacy_tool_prefix': legacy})
    return result, audit


def model_messages(messages):
    return [{'role': message['role'], 'content': message['content']} for message in messages]
