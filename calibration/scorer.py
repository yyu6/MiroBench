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

_SUMMARY_CSV = "thread_metrics_summary.csv"


# ---------------------------------------------------------------------------
# load_thread_metrics
# ---------------------------------------------------------------------------

def load_thread_metrics(directory: Path) -> pd.DataFrame:
    """Read thread_metrics_summary.csv from *directory*.

    Parameters
    ----------
    directory : Path
        Directory that contains ``thread_metrics_summary.csv``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    """
    csv_path = Path(directory) / _SUMMARY_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"thread_metrics_summary.csv not found in {directory}"
        )
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# compute_real_baseline
# ---------------------------------------------------------------------------

def compute_real_baseline(real_dir: Path, metrics: list[str]) -> dict[str, dict]:
    """Compute baseline statistics from real data.

    For each metric returns a dict with:
      - ``median``  : float
      - ``mean``    : float
      - ``values``  : list[float]  (NaN-dropped)

    Parameters
    ----------
    real_dir : Path
        Directory containing the real ``thread_metrics_summary.csv``.
    metrics : list[str]
        Metric column names to include.

    Returns
    -------
    dict[metric_name -> {median, mean, values}]
    """
    df = load_thread_metrics(real_dir)
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


# ---------------------------------------------------------------------------
# score_candidate
# ---------------------------------------------------------------------------

def score_candidate(
    sim_dir: Path,
    real_baseline: dict[str, dict],
    metrics: list[str],
    alpha: float = 0.05,
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

    total_pairs = 0
    total_fails = 0
    abs_deltas: list[float] = []
    per_metric: dict[str, Any] = {}

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

        # Direction of generated relative to real
        if abs(cd) < 0.147:  # negligible effect (Cohen's convention)
            metric_direction = "similar"
        elif cd > 0:
            metric_direction = "generated_higher"
        else:
            metric_direction = "generated_lower"

        # Generated median
        sim_arr = np.asarray(sim_vals, dtype=float)
        generated_median = float(np.median(sim_arr)) if sim_arr.size > 0 else float("nan")

        per_metric[metric] = {
            "real_median": real_median,
            "generated_median": generated_median,
            "cliffs_delta": cd,
            "fail_rate": metric_fail_rate,
            "direction": metric_direction,
            "threads": threads,
        }

        total_pairs += n_sim
        total_fails += metric_fails

    overall_fail_rate = (total_fails / total_pairs) if total_pairs > 0 else 0.0
    mean_abs_delta = float(np.mean(abs_deltas)) if abs_deltas else 0.0

    return {
        "fail_rate": overall_fail_rate,
        "mean_abs_delta": mean_abs_delta,
        "per_metric": per_metric,
    }


# ---------------------------------------------------------------------------
# select_best_candidate
# ---------------------------------------------------------------------------

def select_best_candidate(scored_candidates: list[dict]) -> dict:
    """Select the best candidate from a list of scored candidates.

    Selection criterion: minimise ``(fail_rate, mean_abs_delta)`` lexicographically.

    Parameters
    ----------
    scored_candidates : list[dict]
        Each element is the output of :func:`score_candidate`.

    Returns
    -------
    The candidate dict with the lowest fail_rate (ties broken by mean_abs_delta).

    Raises
    ------
    ValueError
        If *scored_candidates* is empty.
    """
    if not scored_candidates:
        raise ValueError("scored_candidates must be non-empty")

    return min(
        scored_candidates,
        key=lambda c: (c["fail_rate"], c["mean_abs_delta"]),
    )
