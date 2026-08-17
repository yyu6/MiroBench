#!/usr/bin/env python3
"""Evaluate generated thread metrics against the matching real seed posts.

This is for controlled-Qwen runs where each generated post is drawn from a
real seed-post pool.  The generated scorer output has `_run_id` and
`thread_id`; with a fixed posts-per-run schedule we can map each generated
thread back to its source real Reddit thread and compare only that matched real
subset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "generalized_card"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.distribution_stats import evaluate_group_vs_real  # noqa: E402


DEFAULT_METRICS = [
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
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-post-pool-json", type=Path, required=True)
    parser.add_argument("--generated-scores-csv", type=Path, required=True)
    parser.add_argument(
        "--real-scores-csv",
        type=Path,
        default=Path(
            "artifacts/baselines/credit_cards_gpt4omini/real/thread_scores.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--posts-per-run", type=int, default=6)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    args = parser.parse_args()

    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    seeds = _load_seed_posts(args.seed_post_pool_json)
    generated = pd.read_csv(args.generated_scores_csv)
    real = pd.read_csv(args.real_scores_csv)
    real["thread_id_str"] = real["thread_id"].astype(str)

    matched_generated: list[pd.Series] = []
    matched_real: list[pd.Series] = []
    missing: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for row_index, row in generated.iterrows():
        seed_idx = _seed_index(row, posts_per_run=args.posts_per_run)
        if seed_idx < 0 or seed_idx >= len(seeds):
            missing.append(
                {
                    "row": int(row_index),
                    "seed_idx": int(seed_idx),
                    "reason": "seed_idx_out_of_range",
                }
            )
            continue
        seed = seeds[seed_idx]
        raw_id = str(seed.get("source_raw_post_id") or "")
        desired_product = _source_product_dir(seed)
        candidates = real[real["thread_id_str"].eq(raw_id)].copy()
        if candidates.empty:
            missing.append(
                {
                    "row": int(row_index),
                    "seed_idx": int(seed_idx),
                    "source_raw_post_id": raw_id,
                    "source_product": seed.get("source_product"),
                    "desired_product": desired_product,
                    "title": seed.get("title"),
                    "generated_comment_count": _safe_float(row.get("comment_count")),
                    "reason": "no_real_metric_row",
                }
            )
            continue

        exact = (
            candidates[candidates["product"].astype(str).eq(desired_product)]
            if "product" in candidates.columns
            else pd.DataFrame()
        )
        if not exact.empty:
            chosen = exact.iloc[0]
        else:
            chosen = candidates.iloc[0]
            ambiguous.append(
                {
                    "row": int(row_index),
                    "seed_idx": int(seed_idx),
                    "source_raw_post_id": raw_id,
                    "desired_product": desired_product,
                    "available_products": candidates.get(
                        "product", pd.Series(dtype=str)
                    )
                    .astype(str)
                    .tolist(),
                    "chosen_product": str(chosen.get("product")),
                }
            )

        generated_row = row.copy()
        real_row = chosen.copy()
        for item in (generated_row, real_row):
            item["matched_seed_idx"] = seed_idx
            item["matched_source_raw_post_id"] = raw_id
            item["matched_source_product"] = seed.get("source_product")
            item["matched_desired_product_dir"] = desired_product
        matched_generated.append(generated_row)
        matched_real.append(real_row)

    matched_generated_df = pd.DataFrame(matched_generated)
    matched_real_df = pd.DataFrame(matched_real)
    if "thread_id_str" in matched_real_df.columns:
        matched_real_df = matched_real_df.drop(columns=["thread_id_str"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched_generated_df.to_csv(
        args.output_dir / "matched_generated_thread_scores.csv", index=False
    )
    matched_real_df.to_csv(
        args.output_dir / "matched_real_thread_scores.csv", index=False
    )
    (args.output_dir / "matched_seed_missing_real_metrics.json").write_text(
        json.dumps(missing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "matched_seed_ambiguous_real_matches.json").write_text(
        json.dumps(ambiguous, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    group_eval = evaluate_group_vs_real(matched_real_df, matched_generated_df, metrics)
    sample_size = len(matched_generated_df)
    for row in group_eval.values():
        row["inferential_status"] = _status(row, sample_size=sample_size)
    (args.output_dir / "matched_seed_group_eval.json").write_text(
        json.dumps(group_eval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "matched_seed_group_eval.md").write_text(
        _render_markdown(
            group_eval=group_eval,
            metrics=metrics,
            seed_pool=args.seed_post_pool_json,
            generated_scores=args.generated_scores_csv,
            real_scores=args.real_scores_csv,
            generated_rows=len(generated),
            matched_rows=len(matched_generated_df),
            missing=missing,
            ambiguous=ambiguous,
            sample_size=sample_size,
        ),
        encoding="utf-8",
    )

    pass_count, partial_count, fail_count = _status_counts(
        group_eval, metrics, sample_size=sample_size
    )
    print(f"Output: {args.output_dir}")
    print(
        "Matched rows: "
        f"{len(matched_generated_df)} / generated rows: {len(generated)} / "
        f"missing real metrics: {len(missing)} / ambiguous: {len(ambiguous)}"
    )
    if sample_size <= 1:
        print("Inference: DESCRIPTIVE only (n=1); no PASS/PARTIAL/FAIL claim")
    else:
        print(f"PASS/PARTIAL/FAIL: {pass_count}/{partial_count}/{fail_count}")
    for metric in metrics:
        row = group_eval.get(metric, {})
        print(
            f"{metric:28s} {_status(row, sample_size=sample_size):11s} "
            f"MWU={_fmt(row.get('mwu_p_value'))} "
            f"KS={_fmt(row.get('ks_p_value'))} "
            f"Cliff={_fmt(row.get('cliffs_delta'))} "
            f"W={_fmt(row.get('wasserstein_distance'))}"
        )


def _load_seed_posts(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        posts = (
            data.get("seed_posts")
            or data.get("posts")
            or data.get("initial_posts")
            or []
        )
    elif isinstance(data, list):
        posts = data
    else:
        posts = []
    if not isinstance(posts, list) or not all(isinstance(item, dict) for item in posts):
        raise ValueError(f"No seed post list found in {path}")
    return posts


def _seed_index(row: pd.Series, *, posts_per_run: int) -> int:
    if "seed_index" in row and not pd.isna(row.get("seed_index")):
        try:
            return int(float(row.get("seed_index")))
        except (TypeError, ValueError):
            pass
    run_id = int(row.get("_run_id", 0))
    thread_id = int(row.get("thread_id", 0))
    return run_id * posts_per_run + thread_id - 1


def _source_product_dir(seed: dict[str, Any]) -> str:
    source_file = str(seed.get("source_file") or "")
    return Path(source_file).parent.name if source_file else ""


def _status(row: dict[str, Any], *, sample_size: int) -> str:
    if sample_size <= 1:
        return "DESCRIPTIVE"
    mwu = _safe_float(row.get("mwu_p_value"))
    ks = _safe_float(row.get("ks_p_value"))
    if mwu > 0.05 and ks > 0.05:
        return "PASS"
    if mwu > 0.05 or ks > 0.05:
        return "PARTIAL"
    return "FAIL"


def _status_counts(
    group_eval: dict[str, dict[str, Any]],
    metrics: list[str],
    *,
    sample_size: int,
) -> tuple[int, int, int]:
    statuses = [
        _status(group_eval.get(metric, {}), sample_size=sample_size)
        for metric in metrics
    ]
    return statuses.count("PASS"), statuses.count("PARTIAL"), statuses.count("FAIL")


def _render_markdown(
    *,
    group_eval: dict[str, dict[str, Any]],
    metrics: list[str],
    seed_pool: Path,
    generated_scores: Path,
    real_scores: Path,
    generated_rows: int,
    matched_rows: int,
    missing: list[dict[str, Any]],
    ambiguous: list[dict[str, Any]],
    sample_size: int,
) -> str:
    pass_count, partial_count, fail_count = _status_counts(
        group_eval, metrics, sample_size=sample_size
    )
    lines = [
        "# Matched-Seed Group Evaluation",
        "",
        f"- seed pool: {seed_pool}",
        f"- generated scores: {generated_scores}",
        f"- real scores: {real_scores}",
        f"- generated rows: {generated_rows}",
        f"- matched rows used: {matched_rows}",
        f"- missing real metric rows: {len(missing)}",
        f"- ambiguous real matches: {len(ambiguous)}",
        (
            "- inference: descriptive only (n=1); no PASS/PARTIAL/FAIL claim"
            if sample_size <= 1
            else f"- PASS/PARTIAL/FAIL: {pass_count}/{partial_count}/{fail_count}"
        ),
        "",
        "| Metric | MWU p | KS p | Cliff delta | Wasserstein | real median | gen median | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for metric in metrics:
        row = group_eval.get(metric, {})
        lines.append(
            f"| {metric} | {_fmt(row.get('mwu_p_value'))} | {_fmt(row.get('ks_p_value'))} | "
            f"{_fmt(row.get('cliffs_delta'))} | {_fmt(row.get('wasserstein_distance'))} | "
            f"{_fmt(row.get('real_median'))} | {_fmt(row.get('generated_median'))} | "
            f"{_status(row, sample_size=sample_size)} |"
        )
    if missing:
        lines.extend(["", "## Missing real metric rows", ""])
        for item in missing:
            lines.append(
                "- "
                f"seed_idx={item.get('seed_idx')} "
                f"id={item.get('source_raw_post_id')} "
                f"title={item.get('title')} "
                f"reason={item.get('reason')} "
                f"generated_comments={item.get('generated_comment_count')}"
            )
    return "\n".join(lines) + "\n"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: Any) -> str:
    number = _safe_float(value)
    if pd.isna(number):
        return "nan"
    return f"{number:.4g}"


if __name__ == "__main__":
    main()
