import json
import logging
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_discussion_sensenova as runner


class _Completions:
    def __init__(self, message):
        self.message = message

    def create(self, **_kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=self.message,
                )
            ]
        )


class _Client:
    def __init__(self, message):
        self.chat = types.SimpleNamespace(
            completions=_Completions(message),
        )


def test_call_mimo_uses_reasoning_content_fallback(monkeypatch):
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    message = types.SimpleNamespace(content=None, reasoning=None, reasoning_content='{"posts": []}')

    text = runner._call_mimo(
        _Client(message),
        "glm-5.2",
        [{"role": "user", "content": "generate"}],
    )

    assert text == '{"posts": []}'


def test_get_client_uses_sensenova_api_key_and_default_base_url(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(runner, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("SENSENOVA_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    runner._get_client()

    assert captured == {
        "api_key": "sk-test",
        "base_url": "https://token.sensenova.cn/v1",
    }


def test_json_formatter_emits_required_fields(monkeypatch):
    monkeypatch.setenv("ENV", "test")
    formatter = runner.JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.trace_id = "trace"
    record.request_id = "request"
    record.user_id = "user"
    record.status_code = "200"
    record.duration_ms = "12"

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["service"] == runner.SERVICE
    assert payload["env"] == "test"
    assert payload["trace_id"] == "trace"
    assert payload["request_id"] == "request"
    assert payload["user_id"] == "user"
    assert payload["status_code"] == "200"
    assert payload["duration_ms"] == "12"
    assert payload["timestamp"]
