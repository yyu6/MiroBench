#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.domain import REPO_ROOT  # noqa: E402
from generalized_card.core_contract import (  # noqa: E402
    CURRENT_EVALUATION_CORE_NAMES,
    verify_core_contract,
    verify_run_policy,
)


REQUIRED_THREAD_METRICS = (
    "self_bleu_4",
    "self_bertscore_mean_f1",
    "semantic_mean_cosine",
    "hard_disagree_rate",
    "polite_rate",
    "impolite_rate",
    "neutral_rate",
    "length_cv",
    "avg_depth",
    "structural_virality",
    "mean_story_probability",
    "emotion_entropy",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit, score, and matched-evaluate a generalized CARD run."
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--metric-parallel", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_root = REPO_ROOT / "artifacts" / "generalized_card" / "runs" / args.tag
    config = _load_json(run_root / "run_config.json")
    if not config:
        raise SystemExit(f"Run config not found: {run_root / 'run_config.json'}")
    # Evaluation may audit an immutable earlier generator policy. Generation
    # resume remains strict and cannot mix source versions in one run.
    verify_run_policy(config, operation="evaluate generation", allow_historical=True)
    verify_core_contract(CURRENT_EVALUATION_CORE_NAMES)
    generated = Path(config["generated_root"])
    cleaned = run_root / "cleaned"
    evaluation = run_root / "evaluation"
    matched = run_root / "matched_evaluation"
    audit_path = run_root / "output_audit.json"
    expected = int(config["max_posts"])
    posts_per_run = int(config["posts_per_run"])
    seed_pool = Path(config["seed_pool"])
    real_scores = Path(config["domain"]["real_scores_csv"])
    state_path = run_root / "run_state.json"
    state = _load_json(state_path)
    prior_elapsed = float(state.get("elapsed_seconds") or 0.0)
    started = time.monotonic()
    status = "evaluation_failed"
    return_code = 1
    try:
        _run(
            [
                sys.executable,
                str(PACKAGE_ROOT / "scripts" / "audit_output.py"),
                str(generated),
                "--output",
                str(audit_path),
                "--domain",
                str(config.get("domain_config") or config["domain"]["domain_id"]),
                "--seed-pool",
                str(seed_pool),
                *(
                    ["--domain-profile", str(config["domain_profile"])]
                    if config.get("domain_profile")
                    else []
                ),
            ]
        )
        _enforce_evaluable_audit(_load_json(audit_path), audit_path=audit_path)
        _stage_generated_snapshot(
            source=generated,
            target=cleaned,
            complete=lambda: _cleaned_complete(cleaned, expected),
            resume=args.resume,
        )
        _derived_stage(
            target=evaluation,
            complete=lambda: _score_csv_complete(
                evaluation / "revised_generated_thread_scores.csv", expected
            ),
            resume=args.resume,
            command=[
                sys.executable,
                str(
                    REPO_ROOT
                    / "scripts"
                    / "evaluation"
                    / "score_sampled_generated_runs.py"
                ),
                str(cleaned),
                "--output-dir",
                str(evaluation),
                "--device",
                args.device,
                "--metric-parallel",
                str(args.metric_parallel),
                "--expected-seeds",
                str(expected),
            ],
        )
        _derived_stage(
            target=matched,
            complete=lambda: _matched_complete(
                matched,
                generated_scores=evaluation / "revised_generated_thread_scores.csv",
                expected_rows=expected,
            ),
            resume=args.resume,
            command=[
                sys.executable,
                str(REPO_ROOT / "scripts" / "evaluate_matched_seed_group.py"),
                "--seed-post-pool-json",
                str(seed_pool),
                "--generated-scores-csv",
                str(evaluation / "revised_generated_thread_scores.csv"),
                "--real-scores-csv",
                str(real_scores),
                "--output-dir",
                str(matched),
                "--posts-per-run",
                str(posts_per_run),
            ],
        )
        _print_saved_matched_results(matched, sample_size=expected)
        _run(
            [
                sys.executable,
                str(PACKAGE_ROOT / "scripts" / "compare_content_profile.py"),
                "--tag",
                args.tag,
                "--domain",
                str(config.get("domain_config") or "camera"),
                "--runs-root",
                str(run_root.parent),
            ]
        )
        return_code = 0
        status = "evaluation_complete"
        current_artifact = run_root / "current_artifact.json"
        if not current_artifact.exists():
            _write_json(
                current_artifact,
                {
                    "stage": "initial_evaluation",
                    "root": str(cleaned),
                    "scores": str(evaluation / "revised_generated_thread_scores.csv"),
                    "matched": str(matched),
                    "content_profile": str(run_root / "content_profile_audit.json"),
                    "updated_at_epoch": time.time(),
                },
            )
        print(
            f"[evaluation-done] scores={evaluation / 'revised_generated_thread_scores.csv'}",
            flush=True,
        )
        print(
            f"[evaluation-done] matched={matched / 'matched_seed_group_eval.json'}",
            flush=True,
        )
        print(
            f"[evaluation-done] content={run_root / 'content_profile_audit.md'}",
            flush=True,
        )
    except KeyboardInterrupt:
        return_code = 130
        status = "evaluation_interrupted"
        print("[interrupted] completed evaluation stages remain resumable", flush=True)
    finally:
        elapsed = prior_elapsed + (time.monotonic() - started)
        state.update(
            {
                "status": status,
                "return_code": return_code,
                "elapsed_seconds": elapsed,
                "updated_at_epoch": time.time(),
            }
        )
        _write_json(state_path, state)
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "summarize_token_usage.py"),
                str(run_root / "logs" / "token_usage.jsonl"),
                "--output",
                str(run_root / "logs" / "token_usage_summary.json"),
                "--elapsed-seconds",
                str(elapsed),
            ],
            cwd=REPO_ROOT,
            check=False,
        )
    if return_code:
        raise SystemExit(return_code)


def _derived_stage(
    *, target: Path, complete: Any, resume: bool, command: list[str]
) -> None:
    if resume and complete():
        print(f"[evaluation-resume] complete={target}", flush=True)
        return
    if target.exists():
        shutil.rmtree(target)
    _run(command)
    if not complete():
        raise RuntimeError(
            f"Evaluation stage returned without complete output: {target}"
        )


def _stage_generated_snapshot(
    *,
    source: Path,
    target: Path,
    complete: Any,
    resume: bool,
) -> None:
    """Stage Writer output byte-for-byte; evaluation must not repair its input."""

    if resume and complete():
        print(f"[evaluation-resume] complete={target}", flush=True)
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    if not complete():
        raise RuntimeError(
            "Generated output changed neither text nor structure during staging, but "
            f"the staged cohort is incomplete or non-canonical: {target}"
        )


def _enforce_evaluable_audit(report: dict[str, Any], *, audit_path: Path) -> None:
    """Block contaminated output, but retain quality failures for evaluation."""

    if not report:
        raise RuntimeError(
            f"Output audit did not produce a readable report: {audit_path}"
        )
    if not report.get("evaluable"):
        raise RuntimeError(
            "Generated output failed the evaluation-integrity audit; "
            f"inspect {audit_path}"
        )
    if not report.get("healthy"):
        print(
            "[evaluation-audit-warning] continuing with measurable generation "
            "quality findings: "
            f"semantic_collision_posts={report.get('semantic_plan_collision_posts', 0)} "
            f"overconcentrated_perspective_posts="
            f"{report.get('overconcentrated_perspective_posts', 0)}",
            flush=True,
        )


def _run(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _count_discussion_posts(root: Path) -> int:
    total = 0
    for path in root.glob("run_*_sampled_reddit/discussion.json"):
        payload = _load_json(path)
        posts = payload.get("posts")
        if isinstance(posts, list):
            total += len(posts)
    return total


def _cleaned_complete(root: Path, expected_posts: int) -> bool:
    """Require both the expected sample size and canonical tree metadata."""
    if _count_discussion_posts(root) != expected_posts:
        return False
    for path in root.glob("run_*_sampled_reddit/discussion.json"):
        payload = _load_json(path)
        for post in payload.get("posts") or []:
            if not _comment_tree_metadata_matches(post.get("comments") or [], None, 0):
                return False
    return True


def _comment_tree_metadata_matches(
    comments: list[dict[str, Any]],
    parent_id: object,
    depth: int,
) -> bool:
    for comment in comments:
        if (
            comment.get("parent_comment_id") != parent_id
            or comment.get("depth") != depth
        ):
            return False
        if not _comment_tree_metadata_matches(
            comment.get("replies") or [],
            comment.get("comment_id"),
            depth + 1,
        ):
            return False
    return True


def _csv_row_count(path: Path) -> int:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except OSError:
        return 0


def _score_csv_complete(path: Path, expected_rows: int) -> bool:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return False
    if len(rows) != expected_rows:
        return False
    for row in rows:
        for metric in REQUIRED_THREAD_METRICS:
            try:
                value = float(row.get(metric, ""))
            except (TypeError, ValueError):
                return False
            if not math.isfinite(value):
                return False
    return True


def _matched_complete(
    root: Path,
    *,
    generated_scores: Path,
    expected_rows: int,
) -> bool:
    group_eval = root / "matched_seed_group_eval.json"
    matched_generated = root / "matched_generated_thread_scores.csv"
    matched_real = root / "matched_real_thread_scores.csv"
    if not group_eval.exists() or not generated_scores.exists():
        return False
    if group_eval.stat().st_mtime_ns < generated_scores.stat().st_mtime_ns:
        return False
    return (
        _score_csv_complete(matched_generated, expected_rows)
        and _csv_row_count(matched_real) == expected_rows
    )


def _print_saved_matched_results(matched_dir: Path, *, sample_size: int) -> None:
    group_eval = _load_json(matched_dir / "matched_seed_group_eval.json")
    if not group_eval:
        raise RuntimeError(f"Matched evaluation result is missing: {matched_dir}")
    statuses = [
        _metric_status(row, sample_size=sample_size)
        for row in group_eval.values()
        if isinstance(row, dict)
    ]
    if sample_size <= 1:
        print(
            "[evaluation-results] DESCRIPTIVE only (n=1); "
            "no PASS/PARTIAL/FAIL claim",
            flush=True,
        )
    else:
        print(
            "[evaluation-results] "
            f"PASS/PARTIAL/FAIL: {statuses.count('PASS')}/"
            f"{statuses.count('PARTIAL')}/{statuses.count('FAIL')}",
            flush=True,
        )
    for metric, row in group_eval.items():
        if not isinstance(row, dict):
            continue
        print(
            f"{metric:28s} {_metric_status(row, sample_size=sample_size):11s} "
            f"MWU={_format_metric_value(row.get('mwu_p_value'))} "
            f"KS={_format_metric_value(row.get('ks_p_value'))} "
            f"Cliff={_format_metric_value(row.get('cliffs_delta'))} "
            f"W={_format_metric_value(row.get('wasserstein_distance'))}",
            flush=True,
        )


def _metric_status(row: dict[str, Any], *, sample_size: int) -> str:
    if sample_size <= 1:
        return "DESCRIPTIVE"
    try:
        mwu = float(row.get("mwu_p_value"))
        ks = float(row.get("ks_p_value"))
    except (TypeError, ValueError):
        return "FAIL"
    if mwu > 0.05 and ks > 0.05:
        return "PASS"
    if mwu > 0.05 or ks > 0.05:
        return "PARTIAL"
    return "FAIL"


def _format_metric_value(value: object) -> str:
    try:
        return f"{float(value):.5g}"
    except (TypeError, ValueError):
        return "nan"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
