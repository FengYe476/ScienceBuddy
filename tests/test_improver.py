import io
import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from simple_scibuddy.coevolve import improver
from simple_scibuddy.configuration import load_config


@pytest.mark.parametrize("thinking,temperature", [(False, 0.7), (True, 0.7), (True, 0.0)])
def test_local_vllm_chat_protocol(thinking, temperature):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, self.headers, json.load(io.BytesIO(
                self.rfile.read(int(self.headers['Content-Length']))))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({'choices': [{'message': {'content': '{"ops": []}'}}]}).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = improver.Improver({'backend': 'vllm', 'base_url': f'http://127.0.0.1:{server.server_port}/v1',
                                    'model': 'test-model', 'max_tokens': 512, 'enable_thinking': thinking})
        messages = [{'role': 'user', 'content': 'Evolve the harness.'}]
        result = client.generate_command(messages, {'type': 'object'}, temperature=temperature)
        path, headers, payload = requests[0]
        assert path == '/v1/chat/completions' and 'Authorization' not in headers
        assert payload['chat_template_kwargs']['enable_thinking'] == (thinking and temperature > 0)
        assert payload['model'] == 'test-model' and payload['max_tokens'] == 512
        assert payload['response_format'] == {'type': 'json_schema', 'json_schema': {
            'name': 'simple_scibuddy_command', 'schema': {'type': 'object'}, 'strict': True}}
        assert len(messages) == 1 and len(payload['messages']) == 2
        assert json.loads(result['choices'][0]['message']['content']) == {'ops': []}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_remote_auth_errors_and_redaction(monkeypatch):
    config = {'backend': 'remote', 'base_url': 'https://example.invalid/v1', 'model': 'test',
              'api_key_env': 'SIMPLE_SCIBUDDY_TEST_KEY'}
    monkeypatch.delenv('SIMPLE_SCIBUDDY_TEST_KEY', raising=False)
    with pytest.raises(RuntimeError, match='Missing improver credential'):
        improver.Improver(config)
    monkeypatch.setenv('SIMPLE_SCIBUDDY_TEST_KEY', 'fake-secret-for-test')

    class Opener:
        def open(self, request, timeout):
            assert request.get_header('Authorization') == 'Bearer fake-secret-for-test'
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'fake-secret-for-test'}}]}).encode())

    monkeypatch.setattr(improver.urllib.request, 'build_opener', lambda *args: Opener())
    client = improver.Improver(config)
    assert 'fake-secret-for-test' not in json.dumps(client.generate_command([], {}))

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError('https://example.invalid', 401, 'fake-secret-for-test', {}, None)

    monkeypatch.setattr(Opener, 'open', fail)
    with pytest.raises(RuntimeError, match='^Improver HTTP 401$'):
        client.generate_command([], {})


def test_nested_config_and_rejection_of_mixed_or_secret_settings(tmp_path):
    path = tmp_path / 'train.toml'
    config = ('mode="harness_evolve"\nexperiment="run"\n[harness_evolve]\n'
              '[harness_evolve.improver]\nbackend="vllm"\nbase_url="http://localhost:8000/v1"\nmodel="test"\n')
    path.write_text(config)
    assert load_config(path, root=tmp_path)['harness_evolve']['improver']['model'] == 'test'
    for invalid in (config + 'api_key="secret"\n', config.replace('backend="vllm"', 'backend="remote"'),
                    config.replace('[harness_evolve]\n', '[harness_evolve]\nimprover_model="legacy"\n')):
        path.write_text(invalid)
        with pytest.raises(ValueError):
            load_config(path, root=tmp_path)



def test_thinking_budget_leaves_room_for_command():
    config = {'backend': 'vllm', 'base_url': 'http://localhost:8000/v1', 'model': 'test',
              'enable_thinking': True, 'max_tokens': 4096, 'thinking_token_budget': 4096}
    with pytest.raises(ValueError, match='leave room'):
        improver.validate(config)
    assert improver.validate(dict(config, thinking_token_budget=2048))['thinking_token_budget'] == 2048


def test_reasoning_only_truncation_is_preserved(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            payload = json.loads(request.data)
            assert payload['thinking_token_budget'] == 2048
            return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'length',
                'message': {'content': None, 'reasoning': 'budget exhausted'}}]}).encode())
    monkeypatch.setattr(improver.urllib.request, 'build_opener', lambda *args: Opener())
    client = improver.Improver({'backend': 'vllm', 'base_url': 'http://localhost:8000/v1', 'model': 'test',
                               'enable_thinking': True, 'max_tokens': 4096, 'thinking_token_budget': 2048})
    result = client.generate_command([], {})
    assert result['choices'][0]['finish_reason'] == 'length'
    assert result['choices'][0]['message']['content'] == ''


def test_request_budget_preserves_ids_schema_and_output_room():
    from types import SimpleNamespace

    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: json.dumps(messages),
                                encode=lambda text, **kwargs: list(text))
    client = improver.Improver({'backend': 'vllm', 'base_url': 'http://localhost/v1', 'model': 'test'},
                               tokenizer=tokenizer, context_tokens=2500)
    evidence = {'evidence': {'e0:0': {'record': 0}}, 'history': [],
                'records': [{'record': i, 'first_response': 'x' * 500} for i in range(6)]}
    payload = {'messages': [{'role': 'user', 'content': json.dumps(evidence)},
                            {'role': 'user', 'content': 'SCHEMA'}], 'max_tokens': 512}
    budget = client._fit_request(payload)
    remaining = json.loads(payload['messages'][0]['content'])
    assert budget['input_tokens'] + budget['reserved_tokens'] <= 2500
    assert budget['omitted_preview_records']
    assert remaining['records'][-1]['record'] == 5
    assert remaining['evidence'] == evidence['evidence']
    assert payload['messages'][-1]['content'] == 'SCHEMA'
    assert payload['max_tokens'] == 512
    client.context_tokens = 100
    with pytest.raises(RuntimeError, match='context budget exceeded'):
        client._fit_request(payload)


@pytest.mark.parametrize("status", [408, 429, 500, 552, 599])
def test_remote_timeout_retries_are_bounded(monkeypatch, status):
    monkeypatch.setenv('SCIENCEBUDDY_IMPROVER_API_KEY', 'test-key')
    monkeypatch.setattr(improver.time, 'sleep', lambda _: None)
    calls = []
    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            raise urllib.error.HTTPError('https://example.invalid', status, 'timeout', {}, None)
    monkeypatch.setattr(improver.urllib.request, 'build_opener', lambda *a: Opener())
    client = improver.Improver({'backend': 'remote', 'base_url': 'https://example.invalid/v1', 'model': 'test'})
    with pytest.raises(improver.TransientImproverError, match=str(status)):
        client.generate_command([], {})
    assert len(calls) == 3


def test_bad_request_diagnostics_are_bounded_redacted_and_not_retried(monkeypatch):
    key = 'fake-secret-for-http-test'
    monkeypatch.setenv('SCIENCEBUDDY_IMPROVER_API_KEY', key)
    calls = []
    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            body = {'error': {'type': 'invalid_request_error', 'code': 'unsupported_parameter',
                'message': 'Unsupported parameter. Credential=' + key + ' Bearer secondary-token ' + 'x' * 500},
                'request_dump': 'DO_NOT_RECORD_THIS'}
            raise urllib.error.HTTPError('https://example.invalid', 400, 'bad request', {},
                                         io.BytesIO(json.dumps(body).encode()))
    monkeypatch.setattr(improver.urllib.request, 'build_opener', lambda *args: Opener())
    api = improver.Improver({'backend': 'remote', 'base_url': 'https://example.invalid/v1', 'model': 'test'})
    with pytest.raises(RuntimeError) as error:
        api.generate_command([], {'type': 'object'})
    text = str(error.value)
    assert 'unsupported_parameter' in text and 'request_bytes=' in text
    assert key not in text and 'secondary-token' not in text and 'DO_NOT_RECORD_THIS' not in text
    assert len(text) < 550 and len(calls) == 1
