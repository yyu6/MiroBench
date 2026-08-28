#!/usr/bin/env python3
"""Generate matched SynthPAI and OASIS baselines across Reddit domains.

The command is resume-safe.  It first creates deterministic real seed pools,
then runs every requested ``(baseline, model, domain)`` job.  Every job writes
``generation_report.json``; ``summary/generation_summary.csv`` aggregates wall
time, API usage/cost, generated threads, and generated comments.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from build_seed_pools import DEFAULT_DATA_ROOT, available_domains, build_domain
from common import (
    EXPERIMENT_ROOT,
    REPO_ROOT,
    count_generated_artifact,
    load_model_specs,
    read_json,
    summarize_usage,
    write_csv,
    write_json,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "reddit_multidomain_baselines"
BASELINES = ("oasis", "synthpai")


def parse_args() -> argparse.Namespace:
    specs = load_model_specs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--domains", nargs="*", help="Defaults to all discovered domains.")
    parser.add_argument("--models", nargs="+", choices=sorted(specs), default=sorted(specs))
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--max-seeds", type=int, default=150)
    parser.add_argument("--posts-per-run", type=int, default=5)
    parser.add_argument("--min-real-comments", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--oasis-agents", type=int, default=50)
    parser.add_argument("--oasis-hours", type=int, default=24)
    parser.add_argument("--oasis-rounds", type=int, default=12)
    parser.add_argument("--synthpai-config", default="configs/thread/thread_gpt4omini_city_country.yaml")
    parser.add_argument("--thread-retries", type=int, default=1)
    parser.add_argument("--run-retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=60.0)
    parser.add_argument("--force-seeds", action="store_true")
    parser.add_argument("--force-template", action="store_true")
    parser.add_argument("--force", action="store_true", help="Pass --force to the baseline generator.")
    parser.add_argument("--dry-run", action="store_true", help="Validate all generation plumbing without API calls.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining jobs after failures, then return non-zero if any failed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    model_specs = load_model_specs()
    domains = args.domains or available_domains(data_root)
    _validate_args(args, domains)
    output_root.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        build_domain(
            data_root=data_root,
            output_root=output_root / "inputs",
            domain=domain,
            max_seeds=args.max_seeds,
            posts_per_run=args.posts_per_run,
            min_real_comments=args.min_real_comments,
            seed=args.seed,
            force=args.force_seeds,
        )

    if not args.dry_run:
        _validate_credentials(args.models, model_specs)

    failures: list[dict[str, str]] = []
    for baseline in args.baselines:
        for model in args.models:
            for domain in domains:
                try:
                    run_job(
                        args=args,
                        output_root=output_root,
                        baseline=baseline,
                        model=model,
                        domain=domain,
                        model_spec=model_specs[model],
                    )
                except Exception as exc:
                    failure = {
                        "baseline": baseline,
                        "model": model,
                        "domain": domain,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    print(f"[failed] {failure}", file=sys.stderr, flush=True)
                    if not args.continue_on_error:
                        write_generation_summary(output_root)
                        raise

    write_generation_summary(output_root)
    if failures:
        raise SystemExit(f"{len(failures)} generation job(s) failed; inspect generation_report.json and logs.")
    print(f"[complete] summary={output_root / 'summary' / 'generation_summary.csv'}")


def run_job(
    *,
    args: argparse.Namespace,
    output_root: Path,
    baseline: str,
    model: str,
    domain: str,
    model_spec: dict[str, Any],
) -> None:
    job_root = output_root / "generation" / baseline / model / domain
    report_path = job_root / "generation_report.json"
    if report_path.exists() and not args.force and not args.force_template:
        previous = read_json(report_path)
        if previous.get("status") == "success" and bool(previous.get("dry_run")) == bool(args.dry_run):
            print(f"[skip] completed baseline={baseline} model={model} domain={domain}")
            return
    job_root.mkdir(parents=True, exist_ok=True)
    generated_root = job_root / "generated"
    usage_path = job_root / "token_usage.jsonl"
    log_path = job_root / "generation.log"
    seed_pool = output_root / "inputs" / "seed_pools" / f"{domain}.json"
    started_epoch = time.time()
    started_at = _iso_time(started_epoch)
    status = "success"
    error_text = ""

    env = _job_env(
        model=model,
        model_spec=model_spec,
        usage_path=usage_path,
        baseline=baseline,
        domain=domain,
        allow_missing_key=args.dry_run,
    )
    command: list[str]
    try:
        if baseline == "oasis":
            template_dir = output_root / "setup" / "oasis" / model / domain
            prepare_command = [
                sys.executable,
                str(EXPERIMENT_ROOT / "scripts" / "prepare_oasis_template.py"),
                "--seed-pool-json",
                str(seed_pool),
                "--output-dir",
                str(template_dir),
                "--domain",
                domain,
                "--model",
                model,
                "--base-url",
                str(model_spec["base_url"]),
                "--agents",
                str(args.oasis_agents),
                "--hours",
                str(args.oasis_hours),
                "--seed-posts",
                str(args.posts_per_run),
                "--seed",
                str(args.seed),
            ]
            if args.dry_run:
                prepare_command.append("--dry-run")
            if args.force_template:
                prepare_command.append("--force")
            _run_logged(prepare_command, env=env, log_path=log_path, label="prepare_oasis_template")
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_oasis_matched_seed_generator.py"),
                "--seed-post-pool-json",
                str(seed_pool),
                "--template-run-dir",
                str(template_dir),
                "--output-root",
                str(generated_root),
                "--model",
                model,
                "--base-url",
                str(model_spec["base_url"]),
                "--max-seeds",
                str(args.max_seeds),
                "--posts-per-run",
                str(args.posts_per_run),
                "--rounds",
                str(args.oasis_rounds),
                "--hours",
                str(args.oasis_hours),
                "--run-retries",
                str(args.run_retries),
                "--retry-delay",
                str(args.retry_delay),
                "--min-comments-per-post",
                "1",
            ]
        elif baseline == "synthpai":
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "run_synthpai_matched_seed_generator.py"),
                "--seed-post-pool-json",
                str(seed_pool),
                "--output-root",
                str(generated_root),
                "--synthpai-dir",
                str(REPO_ROOT / "SynthPAI"),
                "--config-path",
                args.synthpai_config,
                "--model",
                model,
                "--base-url",
                str(model_spec["base_url"]),
                "--max-seeds",
                str(args.max_seeds),
                "--posts-per-run",
                str(args.posts_per_run),
                "--seed",
                str(args.seed),
                "--thread-retries",
                str(args.thread_retries),
                "--retry-delay",
                str(args.retry_delay),
                "--min-comments-per-post",
                "1",
            ]
        else:  # pragma: no cover - argparse restricts the set.
            raise ValueError(f"Unsupported baseline: {baseline}")
        if args.dry_run:
            command.append("--dry-run")
        if args.force:
            command.append("--force")
        _run_logged(command, env=env, log_path=log_path, label=f"{baseline}_generator")
        if baseline == "oasis":
            _normalize_oasis_domain_metadata(generated_root, domain)
    except Exception as exc:
        status = "failed"
        error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        raise
    finally:
        ended_epoch = time.time()
        artifact_counts = count_generated_artifact(generated_root)
        usage_summary = summarize_usage(usage_path, model_spec)
        report = {
            "status": status,
            "dry_run": args.dry_run,
            "baseline": baseline,
            "model": model,
            "domain": domain,
            "model_key_env": model_spec.get("key_env"),
            "base_url": model_spec.get("base_url"),
            "pricing_basis": model_spec.get("pricing_basis"),
            "seed_pool": str(seed_pool),
            "generated_root": str(generated_root),
            "token_usage_log": str(usage_path),
            "log": str(log_path),
            "started_at": started_at,
            "ended_at": _iso_time(ended_epoch),
            "elapsed_seconds": round(ended_epoch - started_epoch, 3),
            "elapsed_minutes": round((ended_epoch - started_epoch) / 60.0, 3),
            "request_count": usage_summary["requests"],
            "prompt_tokens": usage_summary["prompt_tokens"],
            "cached_prompt_tokens": usage_summary["cached_prompt_tokens"],
            "completion_tokens": usage_summary["completion_tokens"],
            "total_tokens": usage_summary["total_tokens"],
            "estimated_cost_usd": round(float(usage_summary["estimated_cost_usd"]), 8),
            "unknown_cost_requests": usage_summary["unknown_cost_requests"],
            **artifact_counts,
            "error": error_text,
        }
        write_json(job_root / "token_usage_summary.json", usage_summary)
        write_json(report_path, report)
        print(
            f"[{status}] baseline={baseline} model={model} domain={domain} "
            f"threads={artifact_counts['thread_count']} comments={artifact_counts['comment_count']} "
            f"elapsed={report['elapsed_minutes']:.2f}m cost=${report['estimated_cost_usd']:.6f}",
            flush=True,
        )


def _job_env(
    *,
    model: str,
    model_spec: dict[str, Any],
    usage_path: Path,
    baseline: str,
    domain: str,
    allow_missing_key: bool,
) -> dict[str, str]:
    env = os.environ.copy()
    key_env = str(model_spec["key_env"])
    api_key = env.get(key_env, "").strip()
    if not api_key and allow_missing_key:
        # The legacy OASIS dry-run path validates a key before it short-circuits.
        # This value is never sent because all downstream commands receive
        # --dry-run; it keeps the smoke test free of credentials and API calls.
        api_key = "dry-run-placeholder"
    if api_key:
        # Both libraries use OpenAI-compatible clients but read different names.
        env["LLM_API_KEY"] = api_key
        env["OPENAI_API_KEY"] = api_key
    env["LLM_MODEL_NAME"] = model
    env["LLM_BASE_URL"] = str(model_spec["base_url"])
    env["TOKEN_USAGE_LOG_JSONL"] = str(usage_path)
    env["TOKEN_USAGE_AUTOPATCH"] = "1"
    env["TOKEN_USAGE_RUN_TAG"] = f"{baseline}:{model}:{domain}"
    env["TOKEN_USAGE_COMPONENT"] = f"{baseline}_{model}_{domain}"
    existing_python_path = env.get("PYTHONPATH", "")
    paths = [str(REPO_ROOT / "scripts")]
    if existing_python_path:
        paths.append(existing_python_path)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _run_logged(command: list[str], *, env: dict[str, str], log_path: Path, label: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(_shell_quote(part) for part in command)
    print(f"[run] {label}: {printable}", flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n$ {printable}\n")
        handle.flush()
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{label} exited with code {result.returncode}; log={log_path}")


def _normalize_oasis_domain_metadata(generated_root: Path, domain: str) -> None:
    """Remove the upstream matched-generator's credit-card-only labels.

    Its simulator prompts come from the prepared per-domain template, but the
    historical exporter writes ``credit_cards`` into a few bookkeeping fields.
    Correct those artifacts immediately after a successful run so downstream
    evaluation and provenance remain domain-accurate.
    """

    for run_dir in generated_root.glob("run_*_sampled_reddit"):
        config_path = run_dir / "simulation_config.json"
        if config_path.exists():
            config = read_json(config_path)
            config["graph_id"] = domain
            write_json(config_path, config)
        analysis_path = run_dir / "product_analysis.json"
        if analysis_path.exists():
            analysis = read_json(analysis_path)
            analysis["product_category"] = domain
            write_json(analysis_path, analysis)
        discussion_path = run_dir / "discussion.json"
        if discussion_path.exists():
            discussion = read_json(discussion_path)
            meta = dict(discussion.get("meta") or {})
            meta["product_category"] = domain
            discussion["meta"] = meta
            write_json(discussion_path, discussion)


def write_generation_summary(output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_root / "generation").glob("*/*/*/generation_report.json")):
        try:
            rows.append(read_json(path))
        except Exception:
            continue
    rows.sort(key=lambda row: (str(row.get("baseline")), str(row.get("model")), str(row.get("domain"))))
    total = {
        "jobs": len(rows),
        "successful_jobs": sum(row.get("status") == "success" for row in rows),
        "failed_jobs": sum(row.get("status") == "failed" for row in rows),
        "elapsed_seconds": round(sum(float(row.get("elapsed_seconds") or 0.0) for row in rows), 3),
        "estimated_cost_usd": round(sum(float(row.get("estimated_cost_usd") or 0.0) for row in rows), 8),
        "request_count": sum(int(row.get("request_count") or 0) for row in rows),
        "thread_count": sum(int(row.get("thread_count") or 0) for row in rows),
        "comment_count": sum(int(row.get("comment_count") or 0) for row in rows),
    }
    summary_root = output_root / "summary"
    write_csv(summary_root / "generation_summary.csv", rows)
    write_json(summary_root / "generation_summary.json", {"totals": total, "jobs": rows})


def _validate_args(args: argparse.Namespace, domains: list[str]) -> None:
    if not domains:
        raise SystemExit("No domains selected")
    if args.max_seeds < 1 or args.posts_per_run < 1:
        raise SystemExit("--max-seeds and --posts-per-run must be positive")


def _validate_credentials(models: list[str], specs: dict[str, dict[str, Any]]) -> None:
    missing = sorted({str(specs[model]["key_env"]) for model in models if not os.environ.get(str(specs[model]["key_env"]), "").strip()})
    if missing:
        raise SystemExit("Missing required API key environment variable(s): " + ", ".join(missing))


def _iso_time(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat()


def _shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "._/-=:" for char in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    main()
