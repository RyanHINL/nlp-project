"""One stateless, non-streaming request to an explicitly configured local vLLM server."""
import http.client
import ipaddress
import json
import math
import re
import time
from urllib.parse import urlsplit

from evaluation.contract import parse_json


def load_config(path):
    config = parse_json(path.read_text(encoding='utf-8'))
    shared = {'endpoint', 'runtime_build', 'context_tokens', 'context_policy', 'timeout_seconds',
              'temperature', 'top_p', 'seed'}
    model_keys = {'model', 'revision', 'tokenizer_revision', 'chat_template_sha256', 'max_tokens'}
    if set(config) != shared | {'qa', 'judge'}:
        raise ValueError(f'Expected config keys {sorted(shared | {"qa", "judge"})}')
    endpoint_address(config['endpoint'])
    if config['context_policy'] != 'reject_overflow':
        raise ValueError('Only reject_overflow preserves the complete ranked passages')
    for name in ('context_tokens', 'timeout_seconds'):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    for name, low, high in [('temperature', 0, 2), ('top_p', 0, 1)]:
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} outside [{low},{high}]')
    if config['top_p'] == 0 or type(config['seed']) is not int:
        raise ValueError('top_p must be positive and seed an integer')
    for stage in ('qa', 'judge'):
        model = config[stage]
        if set(model) != model_keys:
            raise ValueError(f'Expected {stage} config keys {sorted(model_keys)}')
        for value in [config['runtime_build'], model['model'], model['revision'], model['tokenizer_revision']]:
            if not isinstance(value, str) or not value.strip() or value.startswith('<') or value in {'main', 'latest'}:
                raise ValueError('Fill in the agreed model, immutable revisions and runtime build')
        if not re.fullmatch('[0-9a-f]{64}', model['chat_template_sha256']):
            raise ValueError('Record the SHA256 of the deployed chat template')
        if type(model['max_tokens']) is not int or not 0 < model['max_tokens'] < config['context_tokens']:
            raise ValueError('max_tokens must be positive and below context_tokens')
    return config


def settings_for(config, stage):
    return {**{k: v for k, v in config.items() if k not in {'qa', 'judge'}}, **config[stage]}


def endpoint_address(endpoint):
    url = urlsplit(endpoint)
    if (url.scheme != 'http' or url.path not in {'', '/'} or url.query or url.fragment or
            url.username is not None or url.password is not None or not url.hostname):
        raise ValueError('endpoint must be an HTTP loopback server origin')
    if not ipaddress.ip_address(url.hostname).is_loopback:
        raise ValueError('Use a loopback IP, for example http://127.0.0.1:8000')
    return url.hostname, url.port or 80


class ServerError(ValueError):
    def __init__(self, message, body):
        super().__init__(message)
        self.body = body


def post(endpoint, path, payload, timeout):
    host, port = endpoint_address(endpoint)
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        connection.request('POST', path, body=body, headers={'Content-Type': 'application/json'})
        response = connection.getresponse()
        raw = response.read().decode('utf-8', errors='replace')
        if response.status != 200:
            raise ServerError(f'HTTP {response.status} at {path}', raw)
        try:
            result = parse_json(raw)
            # Reject lone surrogate escapes before saving a UTF-8 artifact.
            json.dumps(result, ensure_ascii=False).encode('utf-8')
        except ValueError as exc:
            raise ServerError(f'Invalid JSON at {path}', raw) from exc
        if not isinstance(result, dict):
            raise ServerError(f'Expected a JSON object at {path}', result)
        return result
    finally:
        connection.close()


def chat_request(messages, settings):
    return {key: settings[key] for key in ('model', 'temperature', 'top_p', 'seed', 'max_tokens')} | {
        'messages': messages, 'n': 1, 'stream': False, 'add_generation_prompt': True}


def call(audit, transport=post):
    """Record tokenization, one completion and failures in the row's audit trail."""
    started = time.monotonic()
    settings = audit['provenance']['settings']
    request = audit['request']
    tokenizer_request = {name: request[name] for name in ('model', 'messages', 'add_generation_prompt')}
    path = '/tokenize'
    try:
        tokenized = transport(settings['endpoint'], path, tokenizer_request, settings['timeout_seconds'])
        count = tokenized['count']
        limit = tokenized['max_model_len']
        if type(count) is not int or count <= 0 or type(limit) is not int or limit <= 0:
            raise ValueError('Invalid tokenizer counts')
        audit['prompt_tokens'] = count
        audit['server_context_tokens'] = limit
        if limit < settings['context_tokens']:
            raise ValueError('Server context window is smaller than the agreed budget')
        if count + request['max_tokens'] > settings['context_tokens']:
            audit.update(status='over_budget', error='Full prompt plus completion reserve exceeds budget')
            return
        path = '/v1/chat/completions'
        result = transport(settings['endpoint'], path, request, settings['timeout_seconds'])
        audit['response'] = result
        if result['model'] != request['model']:
            raise ValueError('Unexpected served model alias in response')
        choice, = result['choices']
        content = choice['message']['content']
        if content is not None and not isinstance(content, str):
            raise ValueError('Expected a text completion')
        audit['raw_output'] = content or ''
        usage = result['usage']
        audit['usage'] = usage
        if choice['finish_reason'] not in ('stop', 'length'):
            raise ValueError(f"Unexpected finish_reason {choice['finish_reason']!r}")
        if (not isinstance(usage, dict) or type(usage.get('prompt_tokens')) is not int or
                usage['prompt_tokens'] != count or type(usage.get('completion_tokens')) is not int or
                not 0 <= usage['completion_tokens'] <= request['max_tokens']):
            raise ValueError('Server usage differs from chat tokenizer count or completion reserve')
        if choice['finish_reason'] == 'length':
            audit.update(status='truncated', error='Server stopped at the generation limit')
        else:
            audit['status'] = 'ok' if audit['raw_output'].strip() else 'empty_output'
    except TimeoutError as exc:
        audit.update(status='timeout', error=f'{path}: {exc}')
    except (OSError, http.client.HTTPException, KeyError, ValueError, TypeError) as exc:
        audit.update(status='call_error', error=f'{path}: {type(exc).__name__}: {exc}')
        if isinstance(exc, ServerError):
            audit['error_response'] = exc.body
    finally:
        audit['seconds'] = time.monotonic() - started
