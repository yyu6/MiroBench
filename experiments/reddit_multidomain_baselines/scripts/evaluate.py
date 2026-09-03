#!/usr/bin/env python3
"""Run GEO's metric suite and compare every completed matched-seed baseline.

The script scores the fixed real-reference threads once per domain and each
completed generated job once.  It then writes per-job metric comparisons plus
an experiment-wide ``evaluation_summary.csv``.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import REPO_ROOT, read_json, write_csv, write_json


DEFAULT_RUN_ROOT = REPO_ROOT / "artifacts" / "reddit_multidomain_baselines"
METADATA_COLUMNS = {
    "thread_id",
    "post_slot",
    "seed_index",
    "source_raw_post_id",
    "source_product_dir",
    "source_file",
    "_run_id",
    "_source_sim_dir",
    "_metric_thread_id",
    "_product_dir",
}

# The benchmark specification has five families and exactly 12 core metrics.
# Scorers may emit additional diagnostics, but the statistical comparison is
# intentionally restricted to these predeclared outputs.
CORE_METRICS = {
    # Uniformity
    "self_bleu_4",
    "self_bertscore_mean_f1",
    "semantic_mean_cosine",
    # Expression
    "mean_story_probability",
    "emotion_entropy",
    # Tone
    "impolite_rate",
    "neutral_rate",
    "polite_rate",
    # Interaction
    "avg_depth",
    "hard_disagree_rate",
    "structural_virality",
    # Form
    "length_cv",
}

SUMMARY_KEY = ("baseline", "model", "domain", "test", "metric")
TWO_SAMPLE_TEST = "two_sample"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--domains", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--baselines", nargs="*")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--metric-parallel", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    reports = _load_completed_reports(run_root, args)
    if not reports:
        raise SystemExit(f"No completed generation reports selected under {run_root / 'generation'}")
    rows: list[dict[str, Any]] = []
    real_scores: dict[str, Path] = {}
    for report_path, report in reports:
        domain = str(report["domain"])
        reference_root = run_root / "inputs" / "real_reference" / domain
        if domain not in real_scores:
            real_scores[domain] = _score_artifact(
                artifact_root=reference_root,
                output_dir=run_root / "evaluation" / "real_reference" / domain,
                expected_seeds=_expected_seeds(reference_root),
                args=args,
            )
        generated_root = Path(str(report["generated_root"])).expanduser()
        generated_scores = _score_artifact(
            artifact_root=generated_root,
            output_dir=run_root / "evaluation" / report["baseline"] / report["model"] / domain / "generated",
            expected_seeds=_expected_seeds(reference_root),
            args=args,
        )
        if args.dry_run:
            continue
        comparison_dir = run_root / "evaluation" / report["baseline"] / report["model"] / domain
        comparison_rows = compare_score_csvs(real_scores[domain], generated_scores)
        for row in comparison_rows:
            row.update(
                {
                    "baseline": report["baseline"],
                    "model": report["model"],
                    "domain": domain,
                    "generation_report": str(report_path),
                    "test": TWO_SAMPLE_TEST,
                }
            )
        write_csv(comparison_dir / "metric_comparison.csv", comparison_rows)
        write_json(
            comparison_dir / "evaluation_manifest.json",
            {
                "real_scores_csv": str(real_scores[domain]),
                "generated_scores_csv": str(generated_scores),
                "comparison_csv": str(comparison_dir / "metric_comparison.csv"),
                "baseline": report["baseline"],
                "model": report["model"],
                "domain": domain,
                "real_thread_count": _csv_row_count(real_scores[domain]),
                "generated_thread_count": _csv_row_count(generated_scores),
            },
        )
        rows.extend(comparison_rows)
        if report["baseline"] == "geo":
            _geo_matched_pair(report, run_root, args)
    if not args.dry_run:
        summary_path = run_root / "summary" / "evaluation_summary.csv"
        kept, merged_rows = merge_evaluation_summary(
            summary_path,
            rows,
        )
        print(f"[summary] {len(rows)} rows written, {kept} earlier rows kept")
        write_json(
            run_root / "summary" / "evaluation_summary.json",
            {
                "comparison_rows": merged_rows,
                "comparison_count": len(merged_rows),
            },
        )
        print(f"[complete] {summary_path}")


def merge_evaluation_summary(
    path: Path,
    rows: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    """Merge one evaluation run without conflating distinct test protocols.

    Older two-sample rows had an empty ``test`` column. Normalize those rows
    before keying so a rerun replaces its own result, while a GEO
    ``matched_pair`` result for the same model/domain/metric remains separate.
    The returned full table is also used for JSON so CSV and JSON cannot drift.
    """

    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    # Migrate summaries produced before the benchmark was fixed at 12 metrics.
    # Otherwise the three retired diagnostic rows would survive every scoped
    # evaluation simply because they are not present in the incoming key set.
    existing = [row for row in existing if str(row.get("metric") or "") in CORE_METRICS]
    for row in [*existing, *rows]:
        row["test"] = str(row.get("test") or TWO_SAMPLE_TEST)
    incoming = {
        tuple(str(row.get(column, "")) for column in SUMMARY_KEY)
        for row in rows
    }
    kept = [
        row
        for row in existing
        if tuple(str(row.get(column, "")) for column in SUMMARY_KEY) not in incoming
    ]
    merged = kept + rows
    merged.sort(
        key=lambda row: tuple(
            str(row.get(column, ""))
            for column in ("domain", "baseline", "model", "test", "metric")
        )
    )
    write_csv(path, merged)
    return len(kept), merged


def _load_completed_reports(run_root: Path, args: argparse.Namespace) -> list[tuple[Path, dict[str, Any]]]:
    reports: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((run_root / "generation").glob("*/*/*/generation_report.json")):
        report = read_json(path)
        if report.get("status") != "success" or report.get("dry_run"):
            continue
        if args.domains and report.get("domain") not in args.domains:
            continue
        if args.models and report.get("model") not in args.models:
            continue
        if args.baselines and report.get("baseline") not in args.baselines:
            continue
        reports.append((path, report))
    return reports


def _geo_matched_pair(report: dict[str, Any], run_root: Path, args) -> None:
    """Also record GEO's matched-pair test, which the two-sample one cannot replace.

    two_sample scores generated threads against a shared real reference, which is
    what makes oasis, synthpai and geo comparable. matched_pair scores each
    generated thread against the real thread it was built from, which is the test
    GEO is designed for and the only one that answers "is this thread
    indistinguishable from its own source". Both belong in the summary; the
    `test` column keeps them apart.
    """

    tags = list(report.get("source_tags") or [])
    if not tags:
        print(f"[matched-pair] {report['model']}/{report['domain']}: report lists no "
              "source_tags; export_to_multidomain.sh records them", flush=True)
        return
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "geo_v137ds" / "matched_pair_table.py"),
        "--cohort", str(report["domain"]), str(report["model"]),
        "--out", str(run_root / "summary" / "evaluation_summary.csv"),
        "--tags", *tags,
    ]
    if args.dry_run:
        print("[matched-pair dry-run] " + " ".join(command[:8]) + " ...", flush=True)
        return
    subprocess.run(command, cwd=REPO_ROOT, check=False)


def _score_artifact(
    *,
    artifact_root: Path,
    output_dir: Path,
    expected_seeds: int,
    args: argparse.Namespace,
) -> Path:
    output_csv = output_dir / "revised_generated_thread_scores.csv"
    if output_csv.exists() and not args.force:
        print(f"[skip score] {artifact_root}")
        return output_csv
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluation" / "score_sampled_generated_runs.py"),
        str(artifact_root),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--metric-parallel",
        str(args.metric_parallel),
        "--expected-seeds",
        str(expected_seeds),
    ]
    print("[score] " + " ".join(_shell_quote(part) for part in command), flush=True)
    if args.dry_run:
        return output_csv
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    with (output_dir / "evaluation.log").open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Metric suite failed for {artifact_root}; log={output_dir / 'evaluation.log'}")
    return output_csv


def _expected_seeds(reference_root: Path) -> int:
    manifest = read_json(reference_root / "reference_manifest.json")
    return int(manifest.get("seed_count") or 0)


def compare_score_csvs(real_csv: Path, generated_csv: Path) -> list[dict[str, Any]]:
    real_rows = _read_csv(real_csv)
    generated_rows = _read_csv(generated_csv)
    metric_names = sorted(
        set(_numeric_columns(real_rows))
        & set(_numeric_columns(generated_rows))
        & CORE_METRICS
    )
    try:
        from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance
    except ImportError as exc:  # pragma: no cover - declared in pyproject.toml.
        raise RuntimeError("Evaluation requires scipy; install the repository dependencies first.") from exc
    output: list[dict[str, Any]] = []
    for metric in metric_names:
        real_values = _values(real_rows, metric)
        generated_values = _values(generated_rows, metric)
        if not real_values or not generated_values:
            continue
        try:
            ks = ks_2samp(real_values, generated_values)
            mwu = mannwhitneyu(real_values, generated_values, alternative="two-sided")
            wasserstein = wasserstein_distance(real_values, generated_values)
        except ValueError:
            continue
        output.append(
            {
                "metric": metric,
                "real_n": len(real_values),
                "generated_n": len(generated_values),
                "real_mean": _mean(real_values),
                "generated_mean": _mean(generated_values),
                "mean_difference_generated_minus_real": _mean(generated_values) - _mean(real_values),
                "wasserstein_distance": float(wasserstein),
                "ks_statistic": float(ks.statistic),
                "ks_p_value": float(ks.pvalue),
                "mwu_statistic": float(mwu.statistic),
                "mwu_p_value": float(mwu.pvalue),
                "cliffs_delta": _cliffs_delta(real_values, generated_values),
            }
        )
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if not str(row.get("thread_id") or "").startswith("__summary")
        ]


def _numeric_columns(rows: list[dict[str, str]]) -> set[str]:
    if not rows:
        return set()
    output: set[str] = set()
    for key in rows[0]:
        if key in METADATA_COLUMNS:
            continue
        values = _values(rows, key)
        if values:
            output.add(key)
    return output


def _values(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(str(row.get(key) or ""))
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _cliffs_delta(left: list[float], right: list[float]) -> float:
    """Return Cliff's delta for generated minus real distributions."""

    greater = 0
    lower = 0
    for right_value in right:
        for left_value in left:
            if right_value > left_value:
                greater += 1
            elif right_value < left_value:
                lower += 1
    return (greater - lower) / (len(left) * len(right))


def _csv_row_count(path: Path) -> int:
    return len(_read_csv(path))


def _shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "._/-=:" for char in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    main()
