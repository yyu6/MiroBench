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

# GEO domains whose config predates the multidomain adapter and keeps a bare name.
GEO_NATIVE_DOMAINS = ("camera", "cell_phone", "headphone", "laptop")
BASELINES = ("oasis", "synthpai", "geo")


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
    parser.add_argument(
        "--oasis-min-comments-per-post",
        type=int,
        default=0,
        help=(
            "Minimum comments required on every OASIS seed thread. The default "
            "records naturally empty OASIS threads instead of aborting the domain."
        ),
    )
    parser.add_argument("--synthpai-config", default="configs/thread/thread_gpt4omini_city_country.yaml")
    parser.add_argument(
        "--synthpai-python",
        type=Path,
        default=None,
        help=(
            "Python executable for SynthPAI. Defaults to SynthPAI/.venv/bin/python "
            "when present, otherwise the current interpreter."
        ),
    )
    parser.add_argument("--synthpai-min-comments-per-post", type=int, default=1)
    parser.add_argument("--thread-retries", type=int, default=1)
    parser.add_argument("--run-retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=60.0)
    parser.add_argument("--force-seeds", action="store_true")
    parser.add_argument("--force-template", action="store_true")
    parser.add_argument("--geo-planner", default="",
                        help="Planner for baseline=geo. Empty uses --models, "
                             "i.e. one model at both ends; the pinned v137ds arm "
                             "is --geo-planner gpt-5.4-mini with a DeepSeek writer.")
    parser.add_argument("--geo-shard-size", type=int, default=3)
    parser.add_argument("--geo-max-parallel", type=int, default=8,
                        help="Concurrent GEO shards. Memory-bound, not API-bound: "
                             "each shard is ~0.4GB once the domain profile is shared.")
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
    previous: dict[str, Any] = {}
    if report_path.exists() and not args.force and not args.force_template:
        previous = read_json(report_path)
        report_is_complete = args.dry_run or int(previous.get("thread_count") or 0) >= args.max_seeds
        if (
            previous.get("status") == "success"
            and bool(previous.get("dry_run")) == bool(args.dry_run)
            and report_is_complete
        ):
            print(f"[skip] completed baseline={baseline} model={model} domain={domain}")
            return
        if previous.get("status") == "success" and not report_is_complete:
            print(
                f"[resume] incomplete success report baseline={baseline} model={model} "
                f"domain={domain} threads={int(previous.get('thread_count') or 0)}/{args.max_seeds}"
            )
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
                str(args.oasis_min_comments_per_post),
            ]
        elif baseline == "synthpai":
            posts_per_run = effective_posts_per_run(
                baseline=baseline,
                model=model,
                requested=args.posts_per_run,
            )
            base_url = effective_base_url(
                baseline=baseline,
                model=model,
                configured=str(model_spec["base_url"]),
            )
            if posts_per_run != args.posts_per_run:
                print(
                    f"[model-override] baseline=synthpai model={model} "
                    f"posts_per_run={posts_per_run} (requested={args.posts_per_run})",
                    flush=True,
                )
            command = [
                str(_synthpai_python(args)),
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
                base_url,
                "--max-seeds",
                str(args.max_seeds),
                "--posts-per-run",
                str(posts_per_run),
                "--seed",
                str(args.seed),
                "--thread-retries",
                str(args.thread_retries),
                "--retry-delay",
                str(args.retry_delay),
                "--min-comments-per-post",
                str(args.synthpai_min_comments_per_post),
            ]
        elif baseline == "geo":
            # GEO owns its own domain configs, seed pools and pinned arm set, so
            # this delegates rather than reimplementing the flag list. The domain
            # name differs: the harness calls it `celebrity`, GEO's config is
            # `celebrity_geo`, because GEO's corpus adapter lives beside the
            # product-thread domains that already used the bare names.
            geo_domain = domain if domain in GEO_NATIVE_DOMAINS else f"{domain}_geo"
            planner = args.geo_planner or model
            command = [
                str(EXPERIMENT_ROOT.parent / "geo_v137ds" / "run_geo_domain.sh"),
                geo_domain,
                "--planner", planner,
                "--writer", model,
                "--shard-size", str(args.geo_shard_size),
                "--max-parallel", str(args.geo_max_parallel),
                "--tag-prefix", f"geo_{domain}_{model.replace('.','')}_{args.seed}",
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
        elif baseline == "geo":
            # GEO writes into artifacts/generalized_card/runs/<tag>/; the harness
            # expects generated/run_*_sampled_reddit under its own layout. The
            # export links the threads across, dedupes on source post, and writes
            # the report this job's evaluation reads -- including source_tags,
            # which is what lets the matched-pair test find the shards later.
            _run_logged(
                [
                    str(EXPERIMENT_ROOT.parent / "geo_v137ds" / "export_to_multidomain.sh"),
                    geo_domain,
                    "--writer", model,
                    "--planner", args.geo_planner or model,
                ],
                env=env, log_path=log_path, label="geo_export",
            )
    except KeyboardInterrupt:
        status = "interrupted"
        error_text = "KeyboardInterrupt: interrupted by user"
        raise
    except Exception as exc:
        status = "failed"
        error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        raise
    finally:
        ended_epoch = time.time()
        invocation_elapsed_seconds = ended_epoch - started_epoch
        previous_elapsed_seconds = 0.0
        if bool(previous.get("dry_run")) == bool(args.dry_run):
            previous_elapsed_seconds = float(previous.get("elapsed_seconds") or 0.0)
        cumulative_elapsed_seconds = previous_elapsed_seconds + invocation_elapsed_seconds
        artifact_counts = count_generated_artifact(generated_root)
        usage_summary = summarize_usage(usage_path, model_spec)
        report = {
            "status": status,
            "dry_run": args.dry_run,
            "baseline": baseline,
            "model": model,
            "domain": domain,
            "model_key_env": model_spec.get("key_env"),
            "base_url": effective_base_url(
                baseline=baseline,
                model=model,
                configured=str(model_spec["base_url"]),
            ),
            "configured_base_url": model_spec.get("base_url"),
            "pricing_basis": model_spec.get("pricing_basis"),
            "seed_pool": str(seed_pool),
            "generated_root": str(generated_root),
            "token_usage_log": str(usage_path),
            "log": str(log_path),
            "started_at": started_at,
            "ended_at": _iso_time(ended_epoch),
            "invocation_elapsed_seconds": round(invocation_elapsed_seconds, 3),
            "elapsed_seconds": round(cumulative_elapsed_seconds, 3),
            "elapsed_minutes": round(cumulative_elapsed_seconds / 60.0, 3),
            "min_comments_per_post": (
                args.oasis_min_comments_per_post
                if baseline == "oasis"
                else args.synthpai_min_comments_per_post
            ),
            "posts_per_run": effective_posts_per_run(
                baseline=baseline,
                model=model,
                requested=args.posts_per_run,
            ),
            "requested_posts_per_run": args.posts_per_run,
            "request_count": usage_summary["requests"],
            "prompt_tokens": usage_summary["prompt_tokens"],
            "cached_prompt_tokens": usage_summary["cached_prompt_tokens"],
            "completion_tokens": usage_summary["completion_tokens"],
            "reasoning_tokens": usage_summary["reasoning_tokens"],
            "billable_output_tokens": usage_summary["billable_output_tokens"],
            "total_tokens": usage_summary["total_tokens"],
            "estimated_cost_usd": round(float(usage_summary["estimated_cost_usd"]), 8),
            "unknown_cost_requests": usage_summary["unknown_cost_requests"],
            "cost_accounting_version": 2,
            "cost_accounting_note": (
                "Provider-aware billed output; Gemini includes inferred thinking tokens."
            ),
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
    if model.strip().lower() == "gemini-2.5-flash":
        # Gemini 2.5 Flash enables dynamic thinking by default.  Keep this
        # benchmark's generation cost/latency bounded by disabling it through
        # Google's OpenAI-compatible reasoning_effort mapping.
        env["LLM_REASONING_EFFORT"] = "none"
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


def _synthpai_python(args: argparse.Namespace) -> Path:
    if args.synthpai_python is not None:
        executable = args.synthpai_python.expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"SynthPAI Python executable not found: {executable}")
        return executable
    dedicated = REPO_ROOT / "SynthPAI" / ".venv" / "bin" / "python"
    if dedicated.is_file():
        return dedicated
    return Path(sys.executable)


def effective_posts_per_run(*, baseline: str, model: str, requested: int) -> int:
    """Return the artifact batch size, forcing Gemini SynthPAI to one seed/run."""

    if baseline == "synthpai" and model.strip().lower().startswith("gemini-"):
        return 1
    return requested


def effective_base_url(*, baseline: str, model: str, configured: str) -> str:
    """Avoid a legacy OpenAI-SDK double slash on Gemini's compatible endpoint."""

    if baseline == "synthpai" and model.strip().lower().startswith("gemini-"):
        return configured.rstrip("/")
    return configured


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
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "cached_prompt_tokens": sum(int(row.get("cached_prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "reasoning_tokens": sum(int(row.get("reasoning_tokens") or 0) for row in rows),
        "billable_output_tokens": sum(int(row.get("billable_output_tokens") or 0) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
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
    if args.oasis_min_comments_per_post < 0 or args.synthpai_min_comments_per_post < 0:
        raise SystemExit("minimum comments per post cannot be negative")


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
