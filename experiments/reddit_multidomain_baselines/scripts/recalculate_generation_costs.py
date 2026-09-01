#!/usr/bin/env python3
"""Recalculate generation usage summaries and reports without rerunning APIs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import load_model_specs, read_json, summarize_usage, write_json
from run_generation import DEFAULT_OUTPUT_ROOT, write_generation_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--domains", nargs="*")
    parser.add_argument("--baselines", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = load_model_specs()
    report_paths = sorted(
        (args.output_root.expanduser().resolve() / "generation").glob(
            "*/*/*/generation_report.json"
        )
    )
    updated = 0
    for report_path in report_paths:
        report = read_json(report_path)
        baseline = str(report.get("baseline") or "")
        model = str(report.get("model") or "")
        domain = str(report.get("domain") or "")
        if args.baselines and baseline not in args.baselines:
            continue
        if args.models and model not in args.models:
            continue
        if args.domains and domain not in args.domains:
            continue
        model_spec = specs.get(model)
        if model_spec is None:
            print(f"[skip] unknown model={model} report={report_path}")
            continue

        usage_path = report_path.parent / "token_usage.jsonl"
        usage_summary = summarize_usage(usage_path, model_spec)
        old_cost = float(report.get("estimated_cost_usd") or 0.0)
        new_cost = round(float(usage_summary["estimated_cost_usd"]), 8)
        _update_report_usage(report, usage_summary, new_cost)
        print(
            f"[reprice] baseline={baseline} model={model} domain={domain} "
            f"old=${old_cost:.6f} new=${new_cost:.6f} "
            f"delta=${new_cost - old_cost:+.6f}"
        )
        if not args.dry_run:
            write_json(report_path.parent / "token_usage_summary.json", usage_summary)
            write_json(report_path, report)
        updated += 1

    if not args.dry_run:
        write_generation_summary(args.output_root.expanduser().resolve())
    print(f"[complete] reports={updated} dry_run={args.dry_run}")


def _update_report_usage(
    report: dict[str, Any], usage_summary: dict[str, Any], new_cost: float
) -> None:
    for key in (
        "prompt_tokens",
        "cached_prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "billable_output_tokens",
        "total_tokens",
        "unknown_cost_requests",
    ):
        report[key] = usage_summary[key]
    report["estimated_cost_usd"] = new_cost
    report["cost_accounting_version"] = 2
    report["cost_accounting_note"] = (
        "Provider-aware billed output; Gemini includes inferred thinking tokens."
    )


if __name__ == "__main__":
    main()
