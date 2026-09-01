"""Shared, dependency-light helpers for the multi-domain baseline workflow."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
MODEL_CONFIG_PATH = EXPERIMENT_ROOT / "config" / "models.json"


def load_model_specs(path: Path = MODEL_CONFIG_PATH) -> dict[str, dict[str, Any]]:
    """Load model endpoint, credential-variable, and pricing settings."""

    payload = read_json(path)
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"models mapping missing from {path}")
    return {str(name): dict(spec) for name, spec in models.items()}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def count_comments(comments: Any) -> int:
    """Count comments recursively in the exported GEO discussion schema."""

    if isinstance(comments, dict):
        children = comments.get("replies") or comments.get("children") or comments.get("comments") or []
        return 1 + count_comments(children)
    if not isinstance(comments, list):
        return 0
    return sum(count_comments(comment) for comment in comments)


def count_generated_artifact(root: Path) -> dict[str, int]:
    """Return produced run, thread, and comment counts below one job directory."""

    run_dirs = sorted(path for path in root.glob("run_*_sampled_reddit") if path.is_dir())
    run_count = 0
    thread_count = 0
    comment_count = 0
    for run_dir in run_dirs:
        discussion_path = run_dir / "discussion.json"
        if not discussion_path.exists():
            continue
        try:
            discussion = read_json(discussion_path)
        except (OSError, json.JSONDecodeError):
            continue
        posts = discussion.get("posts") or []
        if not isinstance(posts, list):
            continue
        run_count += 1
        thread_count += len(posts)
        comment_count += sum(count_comments(post.get("comments") or post.get("replies") or []) for post in posts)
    return {
        "run_count": run_count,
        "thread_count": thread_count,
        "comment_count": comment_count,
    }


def load_usage_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def summarize_usage(path: Path, model_spec: dict[str, Any]) -> dict[str, Any]:
    """Summarize usage and cost from provider-reported response token counts.

    Costs are recomputed here rather than trusting a global tracker, because
    DeepSeek has peak/off-peak rates and these jobs can span either window.
    """

    records = load_usage_records(path)
    totals: dict[str, Any] = {
        "requests": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "billable_output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "unknown_cost_requests": 0,
        "pricing_basis": str(model_spec.get("pricing_basis") or ""),
    }
    by_component: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "requests": 0,
            "prompt_tokens": 0,
            "cached_prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "billable_output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "unknown_cost_requests": 0,
        }
    )
    for record in records:
        component = str(record.get("component") or "unknown")
        bucket = by_component[component]
        reasoning_tokens, billable_output_tokens = _output_token_counts(record)
        for target in (totals, bucket):
            target["requests"] += 1
            for key in (
                "prompt_tokens",
                "cached_prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                target[key] += _safe_int(record.get(key))
            target["reasoning_tokens"] += reasoning_tokens
            target["billable_output_tokens"] += billable_output_tokens
        cost = estimate_record_cost(record, model_spec)
        for target in (totals, bucket):
            if cost is None:
                target["unknown_cost_requests"] += 1
            else:
                target["estimated_cost_usd"] += cost
    totals["by_component"] = dict(sorted(by_component.items()))
    return totals


def estimate_record_cost(record: dict[str, Any], model_spec: dict[str, Any]) -> float | None:
    pricing = _pricing_for_timestamp(record.get("ts"), model_spec)
    if pricing is None:
        return None
    prompt_tokens = _safe_int(record.get("prompt_tokens"))
    cached_tokens = min(prompt_tokens, _safe_int(record.get("cached_prompt_tokens")))
    _reasoning_tokens, output_tokens = _output_token_counts(record)
    try:
        input_price = float(pricing["input_per_million_usd"])
        cached_price = float(pricing.get("cached_input_per_million_usd", input_price))
        output_price = float(pricing["output_per_million_usd"])
    except (KeyError, TypeError, ValueError):
        return None
    return (
        (prompt_tokens - cached_tokens) * input_price
        + cached_tokens * cached_price
        + output_tokens * output_price
    ) / 1_000_000


def _output_token_counts(record: dict[str, Any]) -> tuple[int, int]:
    """Return informative reasoning tokens and provider-billed output tokens.

    Gemini's OpenAI-compatible API currently excludes hidden thinking tokens
    from completion_tokens but includes them in total_tokens. OpenAI and
    DeepSeek include billable reasoning in their completion/output count, so
    only Gemini needs the total-minus-prompt fallback.
    """

    prompt_tokens = _safe_int(record.get("prompt_tokens"))
    completion_tokens = _safe_int(record.get("completion_tokens"))
    total_tokens = _safe_int(record.get("total_tokens"))
    explicit_reasoning_tokens = _safe_int(record.get("reasoning_tokens"))
    explicit_billable_output = _safe_int(record.get("billable_output_tokens"))
    if explicit_billable_output > 0:
        return explicit_reasoning_tokens, explicit_billable_output

    model = str(record.get("model") or "").lower()
    if "gemini" not in model:
        return explicit_reasoning_tokens, completion_tokens

    inferred_reasoning_tokens = max(
        0, total_tokens - prompt_tokens - completion_tokens
    )
    reasoning_tokens = max(explicit_reasoning_tokens, inferred_reasoning_tokens)
    billable_output_tokens = max(completion_tokens, total_tokens - prompt_tokens)
    return reasoning_tokens, billable_output_tokens


def _pricing_for_timestamp(timestamp: Any, model_spec: dict[str, Any]) -> dict[str, Any] | None:
    pricing = model_spec.get("pricing")
    if not isinstance(pricing, dict):
        return None
    tiers = pricing.get("tiers")
    if not isinstance(tiers, dict):
        return pricing
    try:
        ts = float(timestamp)
        moment = datetime.fromtimestamp(ts, tz=UTC)
    except (TypeError, ValueError, OSError, OverflowError):
        return dict(tiers.get("default") or tiers.get("peak") or {})
    peak_hours = set(int(hour) for hour in pricing.get("peak_hours_utc", []))
    is_weekday = moment.weekday() < 5
    tier = "peak" if is_weekday and moment.hour in peak_hours else "off_peak"
    return dict(tiers.get(tier) or tiers.get("default") or {})


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
