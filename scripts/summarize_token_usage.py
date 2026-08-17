#!/usr/bin/env python3
"""Summarize TOKEN_USAGE_LOG_JSONL records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from token_usage_tracker import price_for_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OpenAI token usage JSONL.")
    parser.add_argument("usage_jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--elapsed-seconds", type=float)
    args = parser.parse_args()

    records = load_records(args.usage_jsonl)
    summary = summarize(records, elapsed_seconds=args.elapsed_seconds)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print_summary(summary, args.output)


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize(
    records: list[dict[str, Any]], *, elapsed_seconds: float | None
) -> dict[str, Any]:
    total = empty_bucket()
    by_component: dict[str, dict[str, Any]] = defaultdict(empty_bucket)
    by_model: dict[str, dict[str, Any]] = defaultdict(empty_bucket)

    for row in records:
        add(total, row)
        add(by_component[str(row.get("component") or "unknown")], row)
        add(by_model[str(row.get("model") or "unknown")], row)

    return {
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": None if elapsed_seconds is None else elapsed_seconds / 60.0,
        "total": total,
        "by_component": dict(sorted(by_component.items())),
        "by_model": dict(sorted(by_model.items())),
    }


def empty_bucket() -> dict[str, Any]:
    return {
        "requests": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "unknown_cost_requests": 0,
    }


def add(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    bucket["requests"] += 1
    for key in (
        "prompt_tokens",
        "cached_prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        bucket[key] += int(row.get(key) or 0)
    cost = row.get("estimated_cost_usd")
    if cost is None:
        cost = estimate_cost_from_row(row)
    if cost is None:
        bucket["unknown_cost_requests"] += 1
    else:
        bucket["estimated_cost_usd"] += float(cost or 0.0)


def estimate_cost_from_row(row: dict[str, Any]) -> float | None:
    pricing = price_for_model(str(row.get("model") or ""))
    if pricing is None:
        return None
    prompt_tokens = int(row.get("prompt_tokens") or 0)
    cached_tokens = int(row.get("cached_prompt_tokens") or 0)
    completion_tokens = int(row.get("completion_tokens") or 0)
    if cached_tokens > prompt_tokens:
        cached_tokens = 0
    input_tokens = max(0, prompt_tokens - cached_tokens)
    return (
        input_tokens * pricing["input"]
        + cached_tokens * pricing["cached_input"]
        + completion_tokens * pricing["output"]
    ) / 1_000_000


def print_summary(summary: dict[str, Any], output: Path | None) -> None:
    total = summary["total"]
    elapsed = summary.get("elapsed_seconds")
    elapsed_text = (
        "unknown" if elapsed is None else f"{elapsed:.0f}s ({elapsed / 60.0:.1f} min)"
    )
    print(
        "[token-summary] "
        f"requests={total['requests']} "
        f"input={total['prompt_tokens']} "
        f"cached_input={total['cached_prompt_tokens']} "
        f"output={total['completion_tokens']} "
        f"total={total['total_tokens']} "
        f"estimated_cost_usd=${total['estimated_cost_usd']:.4f} "
        f"unknown_cost_requests={total['unknown_cost_requests']} "
        f"elapsed={elapsed_text}",
        flush=True,
    )
    for component, bucket in summary["by_component"].items():
        print(
            "[token-component] "
            f"component={component} "
            f"requests={bucket['requests']} "
            f"input={bucket['prompt_tokens']} "
            f"output={bucket['completion_tokens']} "
            f"cost=${bucket['estimated_cost_usd']:.4f}",
            flush=True,
        )
    if output:
        print(f"[token-summary-json] {output}", flush=True)


if __name__ == "__main__":
    main()
