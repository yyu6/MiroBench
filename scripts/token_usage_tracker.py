"""Small OpenAI token/cost tracker shared by generation scripts.

Set TOKEN_USAGE_LOG_JSONL to enable logging. Each API response with usage
metadata appends one JSON line. The tracker is intentionally best-effort: it
never raises from caller code if usage extraction or logging fails.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


_SEEN_RESPONSE_KEYS: set[str] = set()


DEFAULT_PRICES_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
}


def record_openai_usage(
    response: Any,
    *,
    model: str,
    component: str,
    meta: dict[str, Any] | None = None,
) -> None:
    """Append usage from an OpenAI response to TOKEN_USAGE_LOG_JSONL."""

    log_path = os.environ.get("TOKEN_USAGE_LOG_JSONL", "").strip()
    if not log_path:
        return
    try:
        usage = _get(response, "usage")
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return
        if _already_recorded(response):
            return

        prompt_tokens = _int_value(
            _first_present(usage, "prompt_tokens", "input_tokens")
        )
        completion_tokens = _int_value(
            _first_present(usage, "completion_tokens", "output_tokens")
        )
        total_tokens = _int_value(_first_present(usage, "total_tokens"))
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        cached_tokens = _cached_prompt_tokens(usage)
        if cached_tokens > prompt_tokens:
            cached_tokens = 0

        pricing = price_for_model(model)
        estimated_cost = None
        if pricing is not None:
            input_tokens = max(0, prompt_tokens - cached_tokens)
            estimated_cost = (
                input_tokens * pricing["input"]
                + cached_tokens * pricing["cached_input"]
                + completion_tokens * pricing["output"]
            ) / 1_000_000

        record = {
            "ts": time.time(),
            "run_tag": os.environ.get("TOKEN_USAGE_RUN_TAG", ""),
            "pid": os.getpid(),
            "model": model,
            "component": component,
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
        }
        if meta:
            record["meta"] = meta

        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        return


def _already_recorded(response: Any) -> bool:
    external_id = _first_present(response, "id", "response_id")
    if external_id is not None:
        response_key = f"response:{external_id}"
    else:
        response_key = f"object:{os.getpid()}:{id(response)}"
    if response_key in _SEEN_RESPONSE_KEYS:
        return True
    _SEEN_RESPONSE_KEYS.add(response_key)
    return False


def price_for_model(model: str) -> dict[str, float] | None:
    override_input = _float_env("TOKEN_PRICE_INPUT_PER_1M")
    override_output = _float_env("TOKEN_PRICE_OUTPUT_PER_1M")
    override_cached = _float_env("TOKEN_PRICE_CACHED_INPUT_PER_1M")
    if override_input is not None and override_output is not None:
        return {
            "input": override_input,
            "cached_input": override_cached
            if override_cached is not None
            else override_input,
            "output": override_output,
        }

    normalized = model.strip().lower()
    if normalized in DEFAULT_PRICES_PER_1M:
        return DEFAULT_PRICES_PER_1M[normalized]
    for prefix, pricing in DEFAULT_PRICES_PER_1M.items():
        if normalized.startswith(prefix + "-"):
            return pricing
    return None


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _first_present(obj: Any, *keys: str) -> Any:
    for key in keys:
        value = _get(obj, key)
        if value is not None:
            return value
    return None


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _cached_prompt_tokens(usage: Any) -> int:
    details = _first_present(usage, "prompt_tokens_details", "input_tokens_details")
    if details is None:
        return 0
    return _int_value(_first_present(details, "cached_tokens", "cached_prompt_tokens"))
