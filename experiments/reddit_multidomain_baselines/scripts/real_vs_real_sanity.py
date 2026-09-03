#!/usr/bin/env python3
"""Run repeated real-vs-real checks for the 12 MiroBench metrics.

By default, each repetition draws two independent bootstrap samples of 150
threads from a domain's fixed 150-thread real reference.  The script also
supports disjoint sampling when a score file contains at least 300 unique
threads.  Metric scoring is deliberately separate: this script reads cached
thread-level score CSVs and performs no model inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = REPO_ROOT / "artifacts" / "reddit_multidomain_baselines"
DEFAULT_DOMAINS = (
    "camera",
    "celebrity",
    "cellphone",
    "credit_cards",
    "game",
    "headphones",
    "health_issue",
    "laptop",
    "movies",
    "news",
    "sports",
    "tv_series",
)

# Keep this order aligned with the paper.
METRIC_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Uniformity",
        (
            "self_bleu_4",
            "self_bertscore_mean_f1",
            "semantic_mean_cosine",
        ),
    ),
    (
        "Expression",
        (
            "mean_story_probability",
            "emotion_entropy",
        ),
    ),
    (
        "Tone",
        (
            "polite_rate",
            "neutral_rate",
            "impolite_rate",
        ),
    ),
    (
        "Interaction",
        (
            "avg_depth",
            "hard_disagree_rate",
            "structural_virality",
        ),
    ),
    ("Form", ("length_cv",)),
)
METRICS = tuple(metric for _, metrics in METRIC_FAMILIES for metric in metrics)
FAMILY_BY_METRIC = {
    metric: family for family, metrics in METRIC_FAMILIES for metric in metrics
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--domains", nargs="*", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--sampling",
        choices=("bootstrap", "disjoint"),
        default="bootstrap",
        help=(
            "bootstrap: two independent samples with replacement; disjoint: "
            "sample 2n unique rows and split them into two groups"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: <run-root>/summary/real_vs_real_sanity",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        help=(
            "Optional tracked directory for compact metric/domain/overall "
            "summaries. Publishing requires all 12 default domains."
        ),
    )
    return parser.parse_args()


def score_path(run_root: Path, domain: str) -> Path:
    return (
        run_root
        / "evaluation"
        / "real_reference"
        / domain
        / "revised_generated_thread_scores.csv"
    )


def _finite(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def _thread_keys(frame: pd.DataFrame) -> pd.Series:
    if "source_raw_post_id" in frame and frame["source_raw_post_id"].notna().all():
        return frame["source_raw_post_id"].astype(str)
    if "_run_id" in frame and "thread_id" in frame:
        return frame["_run_id"].astype(str) + ":" + frame["thread_id"].astype(str)
    if "thread_id" in frame:
        return frame["thread_id"].astype(str)
    raise ValueError("Score CSV has no usable thread identifier column")


def load_domain_scores(path: Path, domain: str, sample_size: int, sampling: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached real scores for domain={domain}: {path}. "
            "Run run_real_sanity_check.sh without --skip-scoring first."
        )
    frame = pd.read_csv(path)
    if "thread_id" in frame:
        frame = frame[
            ~frame["thread_id"].astype(str).str.startswith("__summary")
        ].copy()
    missing = [metric for metric in METRICS if metric not in frame.columns]
    if missing:
        raise ValueError(f"domain={domain} score CSV is missing metrics: {missing}")
    frame["_sanity_thread_key"] = _thread_keys(frame)
    duplicated = frame["_sanity_thread_key"].duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, "_sanity_thread_key"].head(5).tolist()
        raise ValueError(f"domain={domain} has duplicate thread keys: {examples}")
    needed = sample_size if sampling == "bootstrap" else 2 * sample_size
    if len(frame) < needed:
        raise ValueError(
            f"domain={domain} has {len(frame)} scored threads; "
            f"sampling={sampling} requires at least {needed}"
        )
    return frame.reset_index(drop=True)


def sample_indices(
    row_count: int,
    sample_size: int,
    sampling: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if sampling == "bootstrap":
        return (
            rng.choice(row_count, size=sample_size, replace=True),
            rng.choice(row_count, size=sample_size, replace=True),
        )
    selected = rng.choice(row_count, size=2 * sample_size, replace=False)
    return selected[:sample_size], selected[sample_size:]


def compare_metric(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if not len(left) or not len(right):
        raise ValueError("A metric has no finite values in one sampled group")
    mwu = mannwhitneyu(left, right, alternative="two-sided")
    # These classifier-derived metrics contain many ties. The asymptotic KS
    # calculation is the stable supported choice for tied samples.
    ks = ks_2samp(left, right, method="asymp")
    # scipy's U is for left relative to right.  The reported signed effect is
    # right relative to left, matching generated-minus-real elsewhere.
    cliffs_delta = 1.0 - (2.0 * float(mwu.statistic) / (len(left) * len(right)))
    return {
        "valid_n_a": len(left),
        "valid_n_b": len(right),
        "group_a_mean": float(np.mean(left)),
        "group_b_mean": float(np.mean(right)),
        "group_a_median": float(np.median(left)),
        "group_b_median": float(np.median(right)),
        "mwu_statistic": float(mwu.statistic),
        "mwu_p_value": float(mwu.pvalue),
        "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue),
        "wasserstein_distance": float(wasserstein_distance(left, right)),
        "cliffs_delta": cliffs_delta,
        "abs_cliffs_delta": abs(cliffs_delta),
    }


def summarize_metrics(detail: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (domain, family, metric), group in detail.groupby(
        ["domain", "family", "metric"], sort=False
    ):
        rows.append(
            {
                "domain": domain,
                "family": family,
                "metric": metric,
                "source_thread_count": int(group["source_thread_count"].iloc[0]),
                "sample_size_per_group": int(group["sample_size_per_group"].iloc[0]),
                "repeat_count": len(group),
                "alpha": alpha,
                "mwu_p_ge_alpha_count": int(group["mwu_p_ge_alpha"].sum()),
                "mwu_p_ge_alpha_pct": 100.0 * float(group["mwu_p_ge_alpha"].mean()),
                "ks_p_ge_alpha_count": int(group["ks_p_ge_alpha"].sum()),
                "ks_p_ge_alpha_pct": 100.0 * float(group["ks_p_ge_alpha"].mean()),
                "both_p_ge_alpha_count": int(group["both_p_ge_alpha"].sum()),
                "both_p_ge_alpha_pct": 100.0 * float(group["both_p_ge_alpha"].mean()),
                "median_mwu_p_value": float(group["mwu_p_value"].median()),
                "median_ks_p_value": float(group["ks_p_value"].median()),
                "mean_wasserstein_distance": float(group["wasserstein_distance"].mean()),
                "median_wasserstein_distance": float(group["wasserstein_distance"].median()),
                "p95_wasserstein_distance": float(group["wasserstein_distance"].quantile(0.95)),
                "mean_abs_cliffs_delta": float(group["abs_cliffs_delta"].mean()),
                "median_abs_cliffs_delta": float(group["abs_cliffs_delta"].median()),
                "p95_abs_cliffs_delta": float(group["abs_cliffs_delta"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def summarize_domains(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain, group in detail.groupby("domain", sort=False):
        rows.append(
            {
                "domain": domain,
                "metric_count": int(group["metric"].nunique()),
                "repeat_count": int(group["repeat"].nunique()),
                "comparison_count": len(group),
                "mwu_p_ge_alpha_pct": 100.0 * float(group["mwu_p_ge_alpha"].mean()),
                "ks_p_ge_alpha_pct": 100.0 * float(group["ks_p_ge_alpha"].mean()),
                "both_p_ge_alpha_pct": 100.0 * float(group["both_p_ge_alpha"].mean()),
                "mean_abs_cliffs_delta": float(group["abs_cliffs_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.sample_size < 1 or args.repeats < 1:
        raise ValueError("--sample-size and --repeats must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    run_root = args.run_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_root / "summary" / "real_vs_real_sanity"
    )
    seed_rng = np.random.default_rng(args.seed)
    detail_rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}

    for domain in args.domains:
        path = score_path(run_root, domain)
        scores = load_domain_scores(path, domain, args.sample_size, args.sampling)
        sources[domain] = str(path)
        values = {
            metric: pd.to_numeric(scores[metric], errors="coerce").to_numpy(dtype=float)
            for metric in METRICS
        }
        keys = scores["_sanity_thread_key"].to_numpy(dtype=str)
        for repeat in range(1, args.repeats + 1):
            repeat_seed = int(seed_rng.integers(0, 2**32 - 1))
            left_indices, right_indices = sample_indices(
                len(scores),
                args.sample_size,
                args.sampling,
                np.random.default_rng(repeat_seed),
            )
            overlap = len(set(keys[left_indices]) & set(keys[right_indices]))
            for metric in METRICS:
                item = compare_metric(values[metric][left_indices], values[metric][right_indices])
                mwu_pass = item["mwu_p_value"] >= args.alpha
                ks_pass = item["ks_p_value"] >= args.alpha
                detail_rows.append(
                    {
                        "domain": domain,
                        "family": FAMILY_BY_METRIC[metric],
                        "metric": metric,
                        "repeat": repeat,
                        "base_seed": args.seed,
                        "repeat_seed": repeat_seed,
                        "sampling": args.sampling,
                        "source_thread_count": len(scores),
                        "sample_size_per_group": args.sample_size,
                        "sample_a_unique_threads": len(set(keys[left_indices])),
                        "sample_b_unique_threads": len(set(keys[right_indices])),
                        "cross_group_unique_thread_overlap": overlap,
                        **item,
                        "mwu_p_ge_alpha": mwu_pass,
                        "ks_p_ge_alpha": ks_pass,
                        "both_p_ge_alpha": mwu_pass and ks_pass,
                    }
                )

    detail = pd.DataFrame(detail_rows)
    metric_summary = summarize_metrics(detail, args.alpha)
    domain_summary = summarize_domains(detail)
    overall_summary = pd.DataFrame(
        [
            {
                "domain_count": int(detail["domain"].nunique()),
                "metric_count": int(detail["metric"].nunique()),
                "repeats_per_domain": args.repeats,
                "sample_size_per_group": args.sample_size,
                "sampling": args.sampling,
                "comparison_count": len(detail),
                "mwu_p_ge_alpha_pct": 100.0 * float(detail["mwu_p_ge_alpha"].mean()),
                "ks_p_ge_alpha_pct": 100.0 * float(detail["ks_p_ge_alpha"].mean()),
                "both_p_ge_alpha_pct": 100.0 * float(detail["both_p_ge_alpha"].mean()),
                "mean_abs_cliffs_delta": float(detail["abs_cliffs_delta"].mean()),
            }
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "repeated_real_vs_real_detail.csv"
    metric_path = output_dir / "repeated_real_vs_real_metric_summary.csv"
    domain_path = output_dir / "repeated_real_vs_real_domain_summary.csv"
    overall_path = output_dir / "repeated_real_vs_real_overall_summary.csv"
    detail.to_csv(detail_path, index=False)
    metric_summary.to_csv(metric_path, index=False)
    domain_summary.to_csv(domain_path, index=False)
    overall_summary.to_csv(overall_path, index=False)
    manifest = {
        "protocol": "repeated_real_vs_real",
        "domains": list(args.domains),
        "families": {family: list(metrics) for family, metrics in METRIC_FAMILIES},
        "metrics": list(METRICS),
        "sample_size_per_group": args.sample_size,
        "repeats_per_domain": args.repeats,
        "sampling": args.sampling,
        "sampling_note": (
            "Two independent samples with replacement from each fixed real reference."
            if args.sampling == "bootstrap"
            else "Two disjoint samples drawn without replacement within each repetition."
        ),
        "base_seed": args.seed,
        "alpha": args.alpha,
        "pass_rule": "p >= alpha",
        "score_sources": sources,
        "outputs": {
            "detail": str(detail_path),
            "metric_summary": str(metric_path),
            "domain_summary": str(domain_path),
            "overall_summary": str(overall_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.publish_dir:
        if set(args.domains) != set(DEFAULT_DOMAINS) or len(args.domains) != len(DEFAULT_DOMAINS):
            raise ValueError("--publish-dir requires the complete 12-domain set")
        publish_dir = args.publish_dir.expanduser().resolve()
        publish_dir.mkdir(parents=True, exist_ok=True)
        metric_summary.to_csv(
            publish_dir / "repeated_real_vs_real_metric_summary.csv", index=False
        )
        domain_summary.to_csv(
            publish_dir / "repeated_real_vs_real_domain_summary.csv", index=False
        )
        overall_summary.to_csv(
            publish_dir / "repeated_real_vs_real_overall_summary.csv", index=False
        )
        portable_manifest = dict(manifest)
        portable_manifest.pop("score_sources", None)
        portable_manifest.pop("outputs", None)
        portable_manifest["detail_output"] = (
            "The 28,800-row detail CSV is generated locally under the artifact run root."
        )
        portable_manifest["published_outputs"] = [
            "repeated_real_vs_real_metric_summary.csv",
            "repeated_real_vs_real_domain_summary.csv",
            "repeated_real_vs_real_overall_summary.csv",
        ]
        (publish_dir / "manifest.json").write_text(
            json.dumps(portable_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[published] {publish_dir}")
    print(f"[complete] detail={detail_path} rows={len(detail)}")
    print(f"[complete] metric_summary={metric_path} rows={len(metric_summary)}")
    print(f"[complete] domain_summary={domain_path} rows={len(domain_summary)}")
    print(overall_summary.to_string(index=False))


if __name__ == "__main__":
    main()
