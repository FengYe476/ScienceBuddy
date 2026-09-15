"""Chat Completions client for harness proposals and feedback interpretation."""

import copy
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request


class TransientImproverError(RuntimeError):
    """Remote service remains unavailable after bounded retries."""


def bad_request_detail(error, credential):
    """Keep bounded diagnostic fields, never headers or a raw response body."""
    try:
        data = json.loads(error.read(8192))
    except (OSError, ValueError, TypeError, AttributeError):
        return ''
    details = data.get('error') if isinstance(data, dict) else None
    if not isinstance(details, dict):
        return ''
    fields = []
    for key in ('type', 'code', 'message'):
        value = details.get(key)
        if not isinstance(value, (str, int)):
            continue
        value = str(value)
        if credential:
            value = value.replace(credential, '[REDACTED]')
        value = re.sub(r'(?i)Bearer\s+[^\s"\x27,;]+', 'Bearer [REDACTED]', value)
        value = ' '.join(value.split())[:300]
        fields.append(f'{key}={value}')
    return '; '.join(fields)


def validate(config):
    keys = {"backend", "base_url", "model", "api_key_env", "reasoning_effort", "max_tokens", "timeout_seconds", "enable_thinking", "thinking_token_budget"}
    if not isinstance(config, dict) or set(config) - keys:
        raise ValueError("Unknown [harness_evolve.improver] settings; store credentials in environment variables")
    if config.get("backend") not in {"vllm", "remote"}:
        raise ValueError("harness_evolve.improver.backend must be vllm or remote")
    for key in ("base_url", "model"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"harness_evolve.improver.{key} is required")
    url = urllib.parse.urlsplit(config["base_url"])
    schemes = {"https"} if config["backend"] == "remote" else {"http", "https"}
    if url.scheme not in schemes or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Invalid improver base_url; remote requires HTTPS and credentials belong in the environment")
    for key in ("api_key_env", "reasoning_effort"):
        if key in config and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValueError(f"harness_evolve.improver.{key} must be a nonempty string")
    for key in ("max_tokens", "timeout_seconds", "thinking_token_budget"):
        if key in config and (type(config[key]) is not int or config[key] < 1):
            raise ValueError(f"harness_evolve.improver.{key} must be a positive integer")
    if "enable_thinking" in config and (type(config["enable_thinking"]) is not bool or config["backend"] != "vllm"):
        raise ValueError("enable_thinking requires a boolean and the vllm backend")
    if "thinking_token_budget" in config and (not config.get("enable_thinking") or config["thinking_token_budget"] >= config.get("max_tokens", 4096)):
        raise ValueError("thinking_token_budget requires thinking and must leave room within max_tokens for a command")
    return dict(config)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Improver:
    requires_schema_validation = True

    def __init__(self, config, error_type=RuntimeError, *, tokenizer=None, context_tokens=None):
        self.config = validate(config)
        self.error_type = error_type
        self.tokenizer = tokenizer
        self.context_tokens = context_tokens
        self._credential()

    def _fit_request(self, payload):
        """Keep output room; omitted previews remain addressable by record ID."""
        if self.tokenizer is None or self.context_tokens is None:
            return None
        messages = payload['messages']
        reserve = payload.get('max_tokens', 4096) + 128
        evidence = None
        for message in messages:
            try:
                value = json.loads(message['content'])
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict) and isinstance(value.get('records'), list):
                evidence = (message, value)
                break
        omitted = []
        while True:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **payload.get('chat_template_kwargs', {}))
            tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            if tokens + reserve <= self.context_tokens:
                return {'input_tokens': tokens, 'reserved_tokens': reserve,
                        'context_tokens': self.context_tokens, 'omitted_preview_records': omitted}
            if evidence is None:
                raise self.error_type('Improver context budget exceeded without removable evidence previews')
            message, value = evidence
            if value.get('history'):
                value['history'].pop(0)
            elif len(value['records']) > 2:
                omitted.append(value['records'].pop(0)['record'])
                value['omitted_previews'] = 'Use read commands to inspect omitted records listed in evidence.'
            else:
                raise self.error_type('Improver context budget exceeded after compacting evidence previews')
            message['content'] = json.dumps(value)

    def _credential(self):
        name = self.config.get("api_key_env")
        if not name and self.config["backend"] == "remote":
            name = "SCIENCEBUDDY_IMPROVER_API_KEY"
        key = os.environ.get(name, "").strip() if name else ""
        if name and not key:
            raise self.error_type(f"Missing improver credential: set {name}")
        return key

    def public_configuration(self):
        return dict(self.config)

    def generate_command(self, messages, schema, *, temperature=0.7):
        messages = copy.deepcopy(messages)
        messages.append({"role": "user", "content":
                         "Return exactly one JSON command conforming to this JSON schema. No markdown fences.\n"
                         + json.dumps(schema, ensure_ascii=False)})
        payload = {"model": self.config["model"], "messages": messages,
                   "response_format": {"type": "json_object"}, "temperature": temperature}
        if self.config["backend"] == "vllm":
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "simple_scibuddy_command", "schema": schema, "strict": True}}
            payload["chat_template_kwargs"] = {"enable_thinking": self.config.get("enable_thinking", False) and temperature > 0}
        if self.config.get("enable_thinking") and temperature > 0 and "thinking_token_budget" in self.config:
            payload["thinking_token_budget"] = self.config["thinking_token_budget"]
        for key in ("max_tokens", "reasoning_effort"):
            if key in self.config:
                payload[key] = self.config[key]
        request_budget = self._fit_request(payload)
        credential = self._credential()
        headers = {"Content-Type": "application/json"}
        if credential:
            headers["Authorization"] = "Bearer " + credential
        request = urllib.request.Request(self.config["base_url"].rstrip("/") + "/chat/completions",
                                         json.dumps(payload).encode(), headers)
        handlers = [NoRedirect()]
        if self.config["backend"] == "vllm":
            # A machine-wide proxy must not intercept local inference requests.
            handlers.append(urllib.request.ProxyHandler({}))
        attempts = 3 if self.config["backend"] == "remote" else 1
        for attempt in range(attempts):
            try:
                with urllib.request.build_opener(*handlers).open(
                    request, timeout=self.config.get("timeout_seconds", 600)
                ) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                transient = exc.code in {408, 429} or 500 <= exc.code <= 599
                reason = f"Improver HTTP {exc.code}"
                if exc.code == 400:
                    detail = bad_request_detail(exc, credential)
                    if detail:
                        reason += ': ' + detail
                    reason += f' (request_bytes={len(request.data)})'
                exc.close()
                if not transient:
                    raise self.error_type(reason) from None
            except OSError as exc:
                reason = "Improver transport failure: " + type(exc).__name__
            except (ValueError, TypeError) as exc:
                raise self.error_type("Improver response failure: " + type(exc).__name__) from None
            if attempt + 1 == attempts:
                raise TransientImproverError(reason) from None
            time.sleep(2 ** attempt)
        try:
            message = result["choices"][0]["message"]
            if result["choices"][0].get("finish_reason") == "length":
                message["content"] = message.get("content") or ""
            elif not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError("Missing completion content")
        except urllib.error.HTTPError as exc:
            raise self.error_type(f"Improver HTTP {exc.code}") from None
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise self.error_type("Improver transport/response failure: " + type(exc).__name__) from None
        safe = {key: result[key] for key in ("id", "model", "choices", "usage") if key in result}
        if credential:
            safe = json.loads(json.dumps(safe).replace(credential, "[REDACTED]"))
        safe["request_configuration"] = self.public_configuration()
        safe["request_attempts"] = attempt + 1
        if request_budget is not None:
            safe['request_budget'] = request_budget
        return safe
