from __future__ import annotations

from types import SimpleNamespace

from product_reddit_sim.llm_utils import create_json_object_completion


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )


class _FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_gpt5_completion_omits_none_reasoning_effort(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    client = _FakeClient()

    raw = create_json_object_completion(
        client=client,
        model="gpt-5-mini",
        prompt="Return JSON",
        temperature=0.3,
    )

    assert raw == '{"ok": true}'
    kwargs = client.chat.completions.kwargs
    assert "reasoning_effort" not in kwargs
    assert "temperature" not in kwargs


def test_non_gpt5_completion_does_not_use_reasoning_effort_env(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    client = _FakeClient()

    create_json_object_completion(
        client=client,
        model="gpt-4o-mini",
        prompt="Return JSON",
        temperature=0.3,
    )

    kwargs = client.chat.completions.kwargs
    assert kwargs["temperature"] == 0.3
    assert "reasoning_effort" not in kwargs


def test_gemini_25_flash_disables_thinking() -> None:
    client = _FakeClient()

    create_json_object_completion(
        client=client,
        model="gemini-2.5-flash",
        prompt="Return JSON",
        temperature=0.3,
    )

    kwargs = client.chat.completions.kwargs
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["temperature"] == 0.3
