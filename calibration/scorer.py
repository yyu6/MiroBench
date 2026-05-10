"""
Scorer for the calibration system.

Functions
---------
load_thread_metrics        : Load thread_metrics_summary.csv from a directory.
compute_real_baseline      : Compute median/mean/values for each metric from real data.
score_candidate            : Score a simulated candidate directory against a real baseline.
select_best_candidate      : Select the best candidate from a list of scored candidates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .stats import cliffs_delta, empirical_p_value, empirical_percentile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_METRICS = [
    "self_bleu_2", "self_bleu_3", "self_bleu_4",
    "self_bertscore_mean_f1", "semantic_mean_cosine",
    "mean_story_probability",
    "toxicity_mean", "toxicity_max", "toxicity_p90",
    "severe_toxicity_mean", "severe_toxicity_max", "severe_toxicity_p90",
    "obscene_mean", "obscene_max", "obscene_p90",
    "threat_mean", "threat_max", "threat_p90",
    "insult_mean", "insult_max", "insult_p90",
    "identity_attack_mean", "identity_attack_max", "identity_attack_p90",
    "aggression_score_mean", "aggression_score_max",
    "length_std", "length_iqr", "length_cv",
    "max_depth", "avg_depth", "avg_branching_factor", "structural_virality",
]

PRIMARY_CALIBRATION_METRICS = [
    "self_bleu_4",
    "self_bertscore_mean_f1",
    "semantic_mean_cosine",
    "mean_story_probability",
    "toxicity_mean",
    "aggression_score_mean",
    "length_cv",
    "avg_depth",
    "structural_virality",
]

_SUMMARY_CSV = "thread_metrics_summary.csv"
_SUMMARY_MEAN_THREAD_ID = "__summary_mean__"
_ROBUST_Z_CLIP = 5.0

_METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    "semantic_diversity": (
        "self_bleu_2",
        "self_bleu_3",
        "self_bleu_4",
        "self_bertscore_mean_f1",
        "self_bertscore_median_f1",
        "self_bertscore_top_k_mean_f1",
        "semantic_mean_cosine",
        "semantic_median_cosine",
        "semantic_top_k_mean_cosine",
        "semantic_p90_cosine",
    ),
    "story_anecdote": (
        "mean_story_probability",
    ),
    "tone_civility": (
        "toxicity_mean",
        "toxicity_max",
        "toxicity_p90",
        "obscene_mean",
        "obscene_max",
        "obscene_p90",
        "threat_mean",
        "threat_max",
        "threat_p90",
        "insult_mean",
        "insult_max",
        "insult_p90",
        "identity_attack_mean",
        "identity_attack_max",
        "identity_attack_p90",
        "aggression_score_mean",
        "aggression_score_max",
        "aggression_score_p90",
    ),
    "length_variation": (
        "length_std",
        "length_iqr",
        "length_cv",
    ),
    "structure": (
        "comment_count",
        "pair_count",
        "max_depth",
        "avg_depth",
        "avg_branching_factor",
        "structural_virality",
    ),
}

_SELECTION_METRIC_FAMILIES: dict[str, tuple[str, ...]] = {
    "semantic_core": (
        "self_bleu_4",
        "self_bertscore_mean_f1",
        "semantic_mean_cosine",
    ),
    "engagement_core": (
        "mean_story_probability",
        "avg_depth",
        "structural_virality",
    ),
    "length_core": (
        "length_cv",
    ),
    "guardrail_core": (
        "toxicity_mean",
        "aggression_score_mean",
    ),
}


# ---------------------------------------------------------------------------
# load_thread_metrics
# ---------------------------------------------------------------------------

def load_thread_metrics(directory: Path) -> pd.DataFrame:
    """Read thread_metrics_summary.csv from *directory* or all its subdirs.

    If *directory* contains a ``thread_metrics_summary.csv`` directly, that
    file is returned.  Otherwise every immediate subdirectory is scanned for
    the same file and the results are concatenated.  This lets callers pass
    either a single-product directory or a whole category directory (e.g.,
    ``data/raw/discussions/credit_cards``).

    Parameters
    ----------
    directory : Path
        Single-product dir or category dir.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If no ``thread_metrics_summary.csv`` is found anywhere.
    """
    directory = Path(directory)
    direct_csv = directory / _SUMMARY_CSV
    if direct_csv.exists():
        df = pd.read_csv(direct_csv)
        if "thread_id" in df.columns:
            df = df[df["thread_id"].astype(str) != _SUMMARY_MEAN_THREAD_ID]
        return df

    # Walk immediate subdirectories (one level deep)
    frames: list[pd.DataFrame] = []
    for sub in sorted(directory.iterdir()):
        if not sub.is_dir():
            continue
        sub_csv = sub / _SUMMARY_CSV
        if sub_csv.exists():
            df = pd.read_csv(sub_csv)
            if "thread_id" in df.columns:
                df = df[df["thread_id"].astype(str) != _SUMMARY_MEAN_THREAD_ID]
            df["_product_dir"] = sub.name
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No thread_metrics_summary.csv found in {directory} or its subdirs"
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# compute_real_baseline
# ---------------------------------------------------------------------------

def compute_real_baseline(real_dir: Path, metrics: list[str]) -> dict[str, dict]:
    """Compute baseline statistics from real discussion data.

    For each metric returns a dict with:
      - ``median``  : float
      - ``mean``    : float
      - ``values``  : list[float]  (NaN-dropped)

    Parameters
    ----------
    real_dir : Path
        Category dir (e.g., ``data/raw/discussions/credit_cards``) whose
        immediate subdirs each contain a ``thread_metrics_summary.csv``, OR a
        single product dir that contains the CSV directly.  All found CSVs are
        concatenated so that statistics are computed over the full dataset.
    metrics : list[str]
        Metric column names to include.

    Returns
    -------
    dict[metric_name -> {median, mean, values}]
    """
    df = load_thread_metrics(real_dir)
    return _baseline_from_df(df, metrics)


def compute_baseline_from_csv(csv_path: Path, metrics: list[str]) -> dict[str, dict]:
    """Compute baseline statistics from a pre-split CSV file.

    Same output format as :func:`compute_real_baseline` but reads directly from
    a thread-scores CSV (e.g. ``thread_scores_train.csv``) rather than walking
    a directory tree.
    """
    df = pd.read_csv(csv_path)
    return _baseline_from_df(df, metrics)


def _baseline_from_df(df: pd.DataFrame, metrics: list[str]) -> dict[str, dict]:
    """Build {metric: {median, mean, values}} from a DataFrame."""
    if "thread_id" in df.columns:
        df = df[df["thread_id"].astype(str) != _SUMMARY_MEAN_THREAD_ID].copy()
    baseline: dict[str, dict] = {}
    for metric in metrics:
        if metric not in df.columns:
            vals: list[float] = []
        else:
            vals = df[metric].dropna().tolist()

        arr = np.asarray(vals, dtype=float)
        baseline[metric] = {
            "median": float(np.median(arr)) if arr.size > 0 else float("nan"),
            "mean": float(np.mean(arr)) if arr.size > 0 else float("nan"),
            "values": vals,
        }
    return baseline


def _mad(values: np.ndarray, center: float) -> float:
    """Median absolute deviation around *center*."""
    if values.size == 0:
        return 0.0
    return float(np.median(np.abs(values - center)))


def _robust_metric_summary(
    real_values: list[float],
    sim_values: list[float],
    robust_z_clip: float = _ROBUST_Z_CLIP,
) -> dict[str, float | str]:
    """Summarize a metric by directly comparing simulated vs. real distributions."""
    real_arr = np.asarray(real_values, dtype=float)
    real_arr = real_arr[~np.isnan(real_arr)]
    sim_arr = np.asarray(sim_values, dtype=float)
    sim_arr = sim_arr[~np.isnan(sim_arr)]

    if real_arr.size == 0 or sim_arr.size == 0:
        return {
            "sim_median": float("nan"),
            "real_p10": float("nan"),
            "real_median": float("nan"),
            "real_p90": float("nan"),
            "percentile_rank": 0.5,
            "percentile_distance": 0.0,
            "robust_z": 0.0,
            "abs_robust_z": 0.0,
            "abs_robust_z_clipped": 0.0,
            "status": "missing",
        }

    sim_median = float(np.median(sim_arr))
    real_median = float(np.median(real_arr))
    real_p10 = float(np.percentile(real_arr, 10))
    real_p90 = float(np.percentile(real_arr, 90))
    percentile_rank = float(np.mean(real_arr <= sim_median))
    percentile_distance = abs(percentile_rank - 0.5) * 2.0
    mad = _mad(real_arr, real_median)
    robust_z = float((sim_median - real_median) / (mad + 1e-9))
    abs_robust_z = abs(robust_z)
    abs_robust_z_clipped = min(abs_robust_z, robust_z_clip)

    if sim_median > real_p90:
        status = "too_high"
    elif sim_median < real_p10:
        status = "too_low"
    else:
        status = "in_range"

    return {
        "sim_median": sim_median,
        "real_p10": real_p10,
        "real_median": real_median,
        "real_p90": real_p90,
        "percentile_rank": percentile_rank,
        "percentile_distance": percentile_distance,
        "robust_z": robust_z,
        "abs_robust_z": abs_robust_z,
        "abs_robust_z_clipped": abs_robust_z_clipped,
        "status": status,
    }


def _aggregate_group_scores(per_metric: dict[str, dict]) -> dict[str, dict[str, float | int]]:
    """Aggregate robust metric summaries into semantic/story/tone/length/structure groups."""
    group_scores: dict[str, dict[str, float | int]] = {}

    for group_name, group_metrics in _METRIC_GROUPS.items():
        items = [
            per_metric[m]
            for m in group_metrics
            if m in per_metric and per_metric[m].get("status") != "missing"
        ]
        if not items:
            continue

        quantile_fails = sum(1 for item in items if item.get("status") != "in_range")
        percentile_distances = [float(item.get("percentile_distance", 0.0)) for item in items]
        abs_robust_zs = [float(item.get("abs_robust_z_clipped", 0.0)) for item in items]

        group_scores[group_name] = {
            "metric_count": len(items),
            "quantile_fail_rate": quantile_fails / len(items),
            "mean_percentile_distance": float(np.mean(percentile_distances)),
            "mean_abs_robust_z": float(np.mean(abs_robust_zs)),
        }

    return group_scores


def _aggregate_metric_subset(
    per_metric: dict[str, dict[str, Any]],
    metrics: tuple[str, ...],
) -> dict[str, float | int]:
    """Aggregate a narrow subset of metrics for candidate selection."""
    items = [
        per_metric[m]
        for m in metrics
        if m in per_metric and per_metric[m].get("status") != "missing"
    ]
    if not items:
        return {
            "metric_count": 0,
            "out_of_range_count": 0,
            "out_of_range_rate": 0.0,
            "mean_percentile_distance": float("inf"),
            "max_percentile_distance": float("inf"),
            "mean_abs_robust_z": float("inf"),
            "max_abs_robust_z": float("inf"),
            "mean_abs_raw_robust_z": float("inf"),
            "max_abs_raw_robust_z": float("inf"),
        }

    out_of_range_count = sum(1 for item in items if item.get("status") != "in_range")
    percentile_distances = [float(item.get("percentile_distance", 0.0)) for item in items]
    abs_robust_zs = [float(item.get("abs_robust_z_clipped", 0.0)) for item in items]
    abs_raw_robust_zs = [float(item.get("abs_robust_z", 0.0)) for item in items]

    return {
        "metric_count": len(items),
        "out_of_range_count": out_of_range_count,
        "out_of_range_rate": out_of_range_count / len(items),
        "mean_percentile_distance": float(np.mean(percentile_distances)),
        "max_percentile_distance": float(np.max(percentile_distances)),
        "mean_abs_robust_z": float(np.mean(abs_robust_zs)),
        "max_abs_robust_z": float(np.max(abs_robust_zs)),
        "mean_abs_raw_robust_z": float(np.mean(abs_raw_robust_zs)),
        "max_abs_raw_robust_z": float(np.max(abs_raw_robust_zs)),
    }


# ---------------------------------------------------------------------------
# score_candidate
# ---------------------------------------------------------------------------

def score_candidate(
    sim_dir: Path,
    real_baseline: dict[str, dict],
    metrics: list[str],
    alpha: float = 0.05,
    ranking_metrics: list[str] | None = None,
) -> dict:
    """Score a simulated candidate against a real baseline.

    For each metric computes per-thread diagnostics and aggregate statistics.

    Parameters
    ----------
    sim_dir : Path
        Directory containing the simulated ``thread_metrics_summary.csv``.
    real_baseline : dict
        Output of :func:`compute_real_baseline`.
    metrics : list[str]
        Metric column names to evaluate.
    alpha : float
        Significance threshold for empirical p-value.

    Returns
    -------
    dict with keys:
      - ``fail_rate``     : fraction of (metric, thread) pairs that fail (p < alpha)
      - ``mean_abs_delta``: mean |Cliff's delta| across metrics
      - ``per_metric``    : dict[metric -> {real_median, generated_median, cliffs_delta,
                            fail_rate, direction, threads: [{value, empirical_p,
                            percentile, direction, pass}]}]
    """
    sim_df = load_thread_metrics(sim_dir)
    if ranking_metrics is None:
        ranking_metrics = PRIMARY_CALIBRATION_METRICS
    ranking_metric_set = set(ranking_metrics)

    total_pairs = 0
    total_fails = 0
    abs_deltas: list[float] = []
    ranking_total_pairs = 0
    ranking_total_fails = 0
    ranking_abs_deltas: list[float] = []
    per_metric: dict[str, Any] = {}
    robust_per_metric: dict[str, dict[str, Any]] = {}

    for metric in metrics:
        real_vals: list[float] = real_baseline.get(metric, {}).get("values", [])
        real_median: float = real_baseline.get(metric, {}).get("median", float("nan"))

        if metric in sim_df.columns:
            sim_vals = sim_df[metric].dropna().tolist()
        else:
            sim_vals = []

        # Per-thread diagnostics
        threads: list[dict] = []
        metric_fails = 0

        for val in sim_vals:
            gen_val = float(val)
            p = empirical_p_value(real_vals, gen_val)
            pct = empirical_percentile(real_vals, gen_val)
            passed = p >= alpha
            if not passed:
                metric_fails += 1
                direction = "too_high" if gen_val > real_median else "too_low"
            else:
                direction = "within_baseline"

            threads.append({
                "value": gen_val,
                "empirical_p": p,
                "percentile": pct,
                "direction": direction,
                "pass": passed,
            })

        # Metric-level aggregates
        n_sim = len(sim_vals)
        metric_fail_rate = (metric_fails / n_sim) if n_sim > 0 else 0.0

        # Cliff's delta: generated vs. real
        if real_vals and sim_vals:
            cd = cliffs_delta(
                [float(v) for v in sim_vals],
                [float(v) for v in real_vals],
            )
        else:
            cd = 0.0

        abs_deltas.append(abs(cd))
        if metric in ranking_metric_set:
            ranking_abs_deltas.append(abs(cd))

        # Direction of generated relative to real
        if abs(cd) < 0.147:  # negligible effect (Cohen's convention)
            metric_direction = "similar"
        elif cd > 0:
            metric_direction = "generated_higher"
        else:
            metric_direction = "generated_lower"

        robust_summary = _robust_metric_summary(real_vals, sim_vals)

        # Generated summary stats
        sim_arr = np.asarray(sim_vals, dtype=float)
        if sim_arr.size > 0:
            generated_summary = {
                "mean": float(np.mean(sim_arr)),
                "median": float(np.median(sim_arr)),
                "std": float(np.std(sim_arr)),
                "p05": float(np.percentile(sim_arr, 5)),
                "p25": float(np.percentile(sim_arr, 25)),
                "p75": float(np.percentile(sim_arr, 75)),
                "p95": float(np.percentile(sim_arr, 95)),
                "n": int(sim_arr.size),
            }
        else:
            generated_summary = {
                "mean": float("nan"), "median": float("nan"),
                "std": 0.0, "p05": float("nan"), "p25": float("nan"),
                "p75": float("nan"), "p95": float("nan"), "n": 0,
            }

        per_metric[metric] = {
            "real_median": real_median,
            "real_p10": robust_summary["real_p10"],
            "real_p90": robust_summary["real_p90"],
            "generated_summary": generated_summary,
            "generated_median": generated_summary["median"],
            "cliffs_delta": cd,
            "fail_rate": metric_fail_rate,
            "direction": metric_direction,
            "sim_median": robust_summary["sim_median"],
            "percentile_rank": robust_summary["percentile_rank"],
            "percentile_distance": robust_summary["percentile_distance"],
            "robust_z": robust_summary["robust_z"],
            "abs_robust_z": robust_summary["abs_robust_z"],
            "abs_robust_z_clipped": robust_summary["abs_robust_z_clipped"],
            "status": robust_summary["status"],
            "threads": threads,
        }
        robust_per_metric[metric] = per_metric[metric]

        total_pairs += n_sim
        total_fails += metric_fails
        if metric in ranking_metric_set:
            ranking_total_pairs += n_sim
            ranking_total_fails += metric_fails

    overall_fail_rate = (total_fails / total_pairs) if total_pairs > 0 else 0.0
    mean_abs_delta = float(np.mean(abs_deltas)) if abs_deltas else 0.0
    ranking_fail_rate = (
        ranking_total_fails / ranking_total_pairs if ranking_total_pairs > 0 else 0.0
    )
    ranking_mean_abs_delta = (
        float(np.mean(ranking_abs_deltas)) if ranking_abs_deltas else 0.0
    )
    robust_metrics = [
        info
        for metric_name, info in robust_per_metric.items()
        if metric_name in ranking_metric_set and info.get("status") != "missing"
    ]
    quantile_fail_rate = (
        sum(1 for info in robust_metrics if info.get("status") != "in_range") / len(robust_metrics)
        if robust_metrics else 0.0
    )
    mean_percentile_distance = (
        float(np.mean([float(info.get("percentile_distance", 0.0)) for info in robust_metrics]))
        if robust_metrics else 0.0
    )
    mean_abs_robust_z = (
        float(np.mean([float(info.get("abs_robust_z_clipped", 0.0)) for info in robust_metrics]))
        if robust_metrics else 0.0
    )
    group_scores = _aggregate_group_scores(robust_per_metric)
    selection_family_scores = {
        family_name: _aggregate_metric_subset(robust_per_metric, family_metrics)
        for family_name, family_metrics in _SELECTION_METRIC_FAMILIES.items()
    }
    mean_group_percentile_distance = (
        float(
            np.mean([
                float(info.get("mean_percentile_distance", 0.0))
                for info in group_scores.values()
            ])
        )
        if group_scores else mean_percentile_distance
    )
    candidate_thread_count = int(len(sim_df))

    return {
        "candidate_thread_count": candidate_thread_count,
        "ranking_metrics": list(ranking_metrics),
        "ranking_fail_rate": float(ranking_fail_rate),
        "ranking_mean_abs_delta": float(ranking_mean_abs_delta),
        "fail_rate": overall_fail_rate,
        "mean_abs_delta": mean_abs_delta,
        "quantile_fail_rate": float(quantile_fail_rate),
        "mean_percentile_distance": mean_percentile_distance,
        "mean_abs_robust_z": mean_abs_robust_z,
        "mean_group_percentile_distance": mean_group_percentile_distance,
        "group_scores": group_scores,
        "selection_family_scores": selection_family_scores,
        "per_metric": per_metric,
    }


# ---------------------------------------------------------------------------
# select_best_candidate
# ---------------------------------------------------------------------------

def candidate_selection_key(candidate: dict) -> tuple[float, ...]:
    """Return the ranking key used to compare scored candidates.

    During calibration, prefer robust distributional comparison against the
    real validation distribution. Selection is intentionally *not* driven by
    one compressed total score alone: semantic and engagement regressions
    should not be hidden by small wins on already-acceptable metrics.
    """
    if "quantile_fail_rate" in candidate:
        family_scores = candidate.get("selection_family_scores", {}) or {}
        guardrail = family_scores.get("guardrail_core", {})
        semantic = family_scores.get("semantic_core", {})
        engagement = family_scores.get("engagement_core", {})
        length = family_scores.get("length_core", {})
        return (
            int(guardrail.get("out_of_range_count", 0)),
            int(semantic.get("out_of_range_count", 0)),
            float(semantic.get("mean_percentile_distance", float("inf"))),
            float(semantic.get("max_percentile_distance", float("inf"))),
            float(semantic.get("mean_abs_raw_robust_z", semantic.get("mean_abs_robust_z", float("inf")))),
            float(semantic.get("max_abs_raw_robust_z", semantic.get("max_abs_robust_z", float("inf")))),
            int(engagement.get("out_of_range_count", 0)),
            float(engagement.get("mean_percentile_distance", float("inf"))),
            float(engagement.get("max_percentile_distance", float("inf"))),
            float(engagement.get("mean_abs_raw_robust_z", engagement.get("mean_abs_robust_z", float("inf")))),
            float(engagement.get("max_abs_raw_robust_z", engagement.get("max_abs_robust_z", float("inf")))),
            int(length.get("out_of_range_count", 0)),
            float(length.get("mean_percentile_distance", float("inf"))),
            float(length.get("max_percentile_distance", float("inf"))),
            float(length.get("mean_abs_raw_robust_z", length.get("mean_abs_robust_z", float("inf")))),
            float(candidate.get("quantile_fail_rate", float("inf"))),
            float(candidate.get("mean_percentile_distance", float("inf"))),
            float(candidate.get("mean_abs_robust_z", float("inf"))),
            float(candidate.get("ranking_mean_abs_delta", float("inf"))),
            float(candidate.get("ranking_fail_rate", float("inf"))),
            float(guardrail.get("max_percentile_distance", float("inf"))),
            float(guardrail.get("mean_abs_raw_robust_z", guardrail.get("mean_abs_robust_z", float("inf")))),
        )

    if (
        "group_mean_abs_cliffs_delta" in candidate
        or "group_overall_fail_rate" in candidate
    ):
        return (
            float(candidate.get("group_mean_abs_cliffs_delta", candidate["mean_abs_delta"])),
            float(candidate.get("group_overall_fail_rate", candidate["fail_rate"])),
            float(candidate["mean_abs_delta"]),
            float(candidate["fail_rate"]),
        )

    return (
        float(candidate["fail_rate"]),
        float(candidate["mean_abs_delta"]),
    )


def select_best_candidate(scored_candidates: list[dict]) -> dict:
    """Select the best candidate from a list of scored candidates.

    Selection criterion:
    - If robust distributional metrics are present, first minimize
      guardrail regressions, then semantic-core regressions, then
      engagement-core regressions, then overall robust totals.
    - Otherwise, if group-level validation metrics are present, minimise
      ``(group_mean_abs_cliffs_delta, group_overall_fail_rate, mean_abs_delta, fail_rate)``.
    - Otherwise minimise ``(fail_rate, mean_abs_delta)``.

    Parameters
    ----------
    scored_candidates : list[dict]
        Each element is the output of :func:`score_candidate`.

    Returns
    -------
    The candidate dict with the lowest ranking key for the active selection mode.

    Raises
    ------
    ValueError
        If *scored_candidates* is empty.
    """
    if not scored_candidates:
        raise ValueError("scored_candidates must be non-empty")

    return min(scored_candidates, key=candidate_selection_key)
