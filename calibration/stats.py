"""
Statistical functions for the calibration system.

Functions
---------
cliffs_delta              : Effect size between two samples.
empirical_p_value         : Two-sided empirical p-value for a generated value.
empirical_percentile      : Percentile of a generated value in the real distribution.
evaluate_group_vs_real    : Group-level evaluation of generated vs. real distributions.
diagnose_single_generation: Per-instance diagnostic for a single generated row.
compare_before_after      : Compare pre/post calibration group-level results.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# cliffs_delta
# ---------------------------------------------------------------------------

def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta effect size between two samples.

    Positive value means x tends to be greater than y.

    Parameters
    ----------
    x, y : sequences of numbers

    Returns
    -------
    float in [-1, 1]

    Raises
    ------
    ValueError if either input is empty.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("cliffs_delta requires non-empty inputs for both x and y")

    # Vectorised count: for each pair (xi, yj) determine sign(xi - yj)
    # Shape: (len(x), len(y))
    diff = x_arr[:, None] - y_arr[None, :]  # broadcast
    n_pairs = x_arr.size * y_arr.size
    delta = float((np.sum(diff > 0) - np.sum(diff < 0)) / n_pairs)
    return delta


# ---------------------------------------------------------------------------
# empirical_p_value
# ---------------------------------------------------------------------------

def empirical_p_value(real_values: Sequence[float], gen_value: float) -> float:
    """Two-sided empirical p-value of gen_value relative to real_values.

    Formula
    -------
    center   = median(real_values)            (NaNs dropped)
    gen_dist = |gen_value - center|
    real_dist = |real_i - center|  for each i
    p = (count(real_dist >= gen_dist) + 1) / (n + 1)

    Returns 1.0 if real_values is empty or all-NaN.
    """
    arr = np.asarray(real_values, dtype=float)
    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        return 1.0

    center = float(np.median(arr))
    gen_dist = abs(gen_value - center)
    real_dist = np.abs(arr - center)

    n = arr.size
    count = int(np.sum(real_dist >= gen_dist))
    return (count + 1) / (n + 1)


# ---------------------------------------------------------------------------
# empirical_percentile
# ---------------------------------------------------------------------------

def empirical_percentile(real_values: Sequence[float], gen_value: float) -> float:
    """Percentile of gen_value in the real distribution (0–100).

    Returns 50.0 if real_values is empty.
    """
    arr = np.asarray(real_values, dtype=float)
    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        return 50.0

    # scipy.stats.percentileofscore returns 0-100
    return float(scipy_stats.percentileofscore(arr, gen_value, kind="rank"))


# ---------------------------------------------------------------------------
# evaluate_group_vs_real
# ---------------------------------------------------------------------------

def evaluate_group_vs_real(
    real_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    metrics: List[str],
    alpha: float = 0.05,
) -> Dict[str, dict]:
    """Group-level evaluation of generated vs. real distributions.

    For each metric computes:
      - mwu_statistic, mwu_p_value  (Mann-Whitney U, two-sided)
      - ks_statistic,  ks_p_value   (Kolmogorov-Smirnov)
      - cliffs_delta
      - direction: "generated_higher" | "generated_lower" | "similar"
      - empirical_fail_rate: fraction of gen rows whose empirical_p_value < alpha

    Returns
    -------
    dict[metric_name -> dict]
    """
    results: Dict[str, dict] = {}

    for metric in metrics:
        real_vals = real_df[metric].dropna().to_numpy(dtype=float)
        gen_vals = gen_df[metric].dropna().to_numpy(dtype=float)

        # Mann-Whitney U (two-sided)
        if real_vals.size > 0 and gen_vals.size > 0:
            mwu_stat, mwu_p = scipy_stats.mannwhitneyu(
                real_vals, gen_vals, alternative="two-sided"
            )
            ks_stat, ks_p = scipy_stats.ks_2samp(real_vals, gen_vals)
            cd = cliffs_delta(gen_vals.tolist(), real_vals.tolist())
        else:
            mwu_stat, mwu_p = float("nan"), 1.0
            ks_stat, ks_p = float("nan"), 1.0
            cd = 0.0

        # Direction: cd > 0 => gen higher than real
        if mwu_p < alpha:
            direction = "generated_higher" if cd > 0 else "generated_lower"
        else:
            direction = "similar"

        # empirical_fail_rate: fraction of gen values that are individually flagged
        if gen_vals.size > 0:
            fail_count = sum(
                1 for v in gen_vals
                if empirical_p_value(real_vals.tolist(), float(v)) < alpha
            )
            empirical_fail_rate = fail_count / gen_vals.size
        else:
            empirical_fail_rate = 0.0

        results[metric] = {
            "mwu_statistic": float(mwu_stat),
            "mwu_p_value": float(mwu_p),
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p),
            "cliffs_delta": float(cd),
            "direction": direction,
            "empirical_fail_rate": empirical_fail_rate,
        }

    return results


# ---------------------------------------------------------------------------
# diagnose_single_generation
# ---------------------------------------------------------------------------

def diagnose_single_generation(
    real_df: pd.DataFrame,
    gen_row: pd.Series,
    metrics: List[str],
    alpha: float = 0.05,
) -> Dict[str, dict]:
    """Per-instance diagnostic for a single generated row.

    For each metric:
      - real_median
      - generated_value
      - empirical_p_value
      - percentile
      - direction: "within_baseline" | "too_high" | "too_low"
      - diagnosis_flag: "pass" | "fail"

    Returns
    -------
    dict[metric_name -> dict]
    """
    results: Dict[str, dict] = {}

    for metric in metrics:
        real_vals = real_df[metric].dropna().tolist()
        gen_val = float(gen_row[metric])

        real_median = float(np.median(real_vals)) if real_vals else float("nan")
        p_val = empirical_p_value(real_vals, gen_val)
        pct = empirical_percentile(real_vals, gen_val)

        if p_val < alpha:
            diagnosis_flag = "fail"
            direction = "too_high" if gen_val > real_median else "too_low"
        else:
            diagnosis_flag = "pass"
            direction = "within_baseline"

        results[metric] = {
            "real_median": real_median,
            "generated_value": gen_val,
            "empirical_p_value": p_val,
            "percentile": pct,
            "direction": direction,
            "diagnosis_flag": diagnosis_flag,
        }

    return results


# ---------------------------------------------------------------------------
# compare_before_after
# ---------------------------------------------------------------------------

def compare_before_after(
    before_results: Dict[str, dict],
    after_results: Dict[str, dict],
    alpha: float = 0.05,
) -> dict:
    """Compare pre/post calibration group-level evaluation results.

    Parameters
    ----------
    before_results, after_results : output of evaluate_group_vs_real

    Returns
    -------
    dict with keys:
      per_metric : dict[metric -> {improved, abs_delta_reduction, fail_rate_reduction}]
      summary    : {
          metrics_sig_different_before, metrics_sig_different_after,
          avg_abs_cliffs_delta_before, avg_abs_cliffs_delta_after,
          overall_fail_rate_before, overall_fail_rate_after,
          overall_pass_rate_before, overall_pass_rate_after,
      }
    """
    metrics = list(before_results.keys())
    per_metric: Dict[str, dict] = {}

    for metric in metrics:
        b = before_results[metric]
        a = after_results.get(metric, {})

        b_fail = b.get("empirical_fail_rate", 0.0)
        a_fail = a.get("empirical_fail_rate", 0.0)
        b_cd = abs(b.get("cliffs_delta", 0.0))
        a_cd = abs(a.get("cliffs_delta", 0.0))

        fail_rate_reduction = b_fail - a_fail
        abs_delta_reduction = b_cd - a_cd
        improved = (fail_rate_reduction > 0) or (abs_delta_reduction > 0)

        per_metric[metric] = {
            "improved": improved,
            "abs_delta_reduction": abs_delta_reduction,
            "fail_rate_reduction": fail_rate_reduction,
        }

    # Summary statistics
    def _sig(result_dict: dict) -> int:
        """Count metrics that are statistically significant."""
        count = 0
        for info in result_dict.values():
            # Use either mwu or ks p-value if available
            mwu_p = info.get("mwu_p_value", 1.0)
            ks_p = info.get("ks_p_value", 1.0)
            if mwu_p < alpha or ks_p < alpha:
                count += 1
        return count

    def _avg_abs_cd(result_dict: dict) -> float:
        cds = [abs(info.get("cliffs_delta", 0.0)) for info in result_dict.values()]
        return float(np.mean(cds)) if cds else 0.0

    def _avg_fail_rate(result_dict: dict) -> float:
        rates = [info.get("empirical_fail_rate", 0.0) for info in result_dict.values()]
        return float(np.mean(rates)) if rates else 0.0

    overall_fail_before = _avg_fail_rate(before_results)
    overall_fail_after = _avg_fail_rate(after_results)

    summary = {
        "metrics_sig_different_before": _sig(before_results),
        "metrics_sig_different_after": _sig(after_results),
        "avg_abs_cliffs_delta_before": _avg_abs_cd(before_results),
        "avg_abs_cliffs_delta_after": _avg_abs_cd(after_results),
        "overall_fail_rate_before": overall_fail_before,
        "overall_fail_rate_after": overall_fail_after,
        "overall_pass_rate_before": 1.0 - overall_fail_before,
        "overall_pass_rate_after": 1.0 - overall_fail_after,
    }

    return {"per_metric": per_metric, "summary": summary}
