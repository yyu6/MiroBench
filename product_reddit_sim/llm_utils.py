"""Shared LLM utilities for GEO generation stages."""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    from token_usage_tracker import record_openai_usage
except Exception:  # pragma: no cover - tracking must never block generation.
    def record_openai_usage(*args: Any, **kwargs: Any) -> None:
        return


DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_ATTEMPTS = 4


def _supports_custom_temperature(model: str) -> bool:
    """Return whether *model* accepts non-default temperature overrides."""

    return not model.strip().lower().startswith("gpt-5")


def _resolve_reasoning_effort(model: str) -> str | None:
    """Return the configured reasoning effort for simulation-side LLM calls."""

    normalized_model = model.strip().lower()
    if normalized_model == "gemini-2.5-flash":
        return "none"
    if not normalized_model.startswith("gpt-5"):
        return None
    effort = os.environ.get("LLM_REASONING_EFFORT", "").strip().lower()
    if effort == "none":
        return None
    return effort or None


def _sanitize_json_response(raw: str) -> str:
    """Extract a single valid JSON object from an LLM response.

    Some providers (e.g. Gemini via OpenAI-compatible API) occasionally return
    multiple JSON objects concatenated, or wrap JSON in markdown fences.
    This extracts only the first complete JSON object.
    """
    text = raw.strip()
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()

    # Fast path: valid single JSON object
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Extract just the first complete JSON object using raw_decode
    decoder = json.JSONDecoder()
    # Find the first '{' or '['
    for i, ch in enumerate(text):
        if ch in ('{', '['):
            try:
                obj, end = decoder.raw_decode(text, i)
                return json.dumps(obj, ensure_ascii=False)
            except json.JSONDecodeError:
                continue

    # Nothing worked, return original and let caller handle the error
    return raw


def create_json_object_completion(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> str:
    """Run one JSON-object completion with bounded retries.

    This protects long-running simulation batches from hanging forever on a
    single HTTP read. The caller is still responsible for validating the JSON
    schema it receives.
    """

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "timeout": timeout,
            }
            if _supports_custom_temperature(model):
                kwargs["temperature"] = temperature
            reasoning_effort = _resolve_reasoning_effort(model)
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(
                **kwargs,
            )
            record_openai_usage(response, model=model, component="product_reddit_sim_llm")
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise ValueError("LLM returned empty content")
            return _sanitize_json_response(raw)
        except (
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
            RateLimitError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_seconds = min(30.0, 2.5 * attempt + random.random())
            print(
                f"LLM request failed on attempt {attempt}/{max_attempts}: "
                f"{type(exc).__name__}: {exc}. Retrying in {sleep_seconds:.1f}s..."
            )
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error
