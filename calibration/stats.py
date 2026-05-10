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


def _clean_array(values: Sequence[float]) -> np.ndarray:
    """Return a float array with NaNs removed."""
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def _median_absolute_deviation(values: np.ndarray, center: float | None = None) -> float:
    """Return the median absolute deviation (MAD)."""
    if values.size == 0:
        return float("nan")
    if center is None:
        center = float(np.median(values))
    return float(np.median(np.abs(values - center)))


def _quantile_error(
    real_values: np.ndarray,
    gen_values: np.ndarray,
    quantiles: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> float:
    """Mean absolute error across matched real/generated quantiles."""
    if real_values.size == 0 or gen_values.size == 0:
        return float("nan")
    real_q = np.quantile(real_values, quantiles)
    gen_q = np.quantile(gen_values, quantiles)
    return float(np.mean(np.abs(real_q - gen_q)))


def _bootstrap_wasserstein_reduction_ci(
    real_values: Sequence[float],
    before_values: Sequence[float],
    after_values: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI for the reduction in Wasserstein distance.

    Positive values mean the after-calibration distribution is closer to the
    real distribution than the before-calibration distribution.
    """
    real_arr = _clean_array(real_values)
    before_arr = _clean_array(before_values)
    after_arr = _clean_array(after_values)

    if real_arr.size == 0 or before_arr.size == 0 or after_arr.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    reductions = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        real_sample = rng.choice(real_arr, size=real_arr.size, replace=True)
        before_sample = rng.choice(before_arr, size=before_arr.size, replace=True)
        after_sample = rng.choice(after_arr, size=after_arr.size, replace=True)
        before_w = scipy_stats.wasserstein_distance(real_sample, before_sample)
        after_w = scipy_stats.wasserstein_distance(real_sample, after_sample)
        reductions[i] = before_w - after_w

    low, high = np.percentile(reductions, [2.5, 97.5])
    return float(low), float(high)


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
        real_vals = _clean_array(real_df[metric].tolist()) if metric in real_df.columns else np.asarray([], dtype=float)
        gen_vals = _clean_array(gen_df[metric].tolist()) if metric in gen_df.columns else np.asarray([], dtype=float)

        # Mann-Whitney U (two-sided)
        if real_vals.size > 0 and gen_vals.size > 0:
            mwu_stat, mwu_p = scipy_stats.mannwhitneyu(
                real_vals, gen_vals, alternative="two-sided"
            )
            ks_stat, ks_p = scipy_stats.ks_2samp(real_vals, gen_vals)
            cd = cliffs_delta(gen_vals.tolist(), real_vals.tolist())
            wasserstein = float(scipy_stats.wasserstein_distance(real_vals, gen_vals))
            quantile_error = _quantile_error(real_vals, gen_vals)
            real_mean = float(np.mean(real_vals))
            generated_mean = float(np.mean(gen_vals))
            real_median = float(np.median(real_vals))
            generated_median = float(np.median(gen_vals))
            median_gap = generated_median - real_median
            mad = _median_absolute_deviation(real_vals, center=real_median)
            mad_normalized_median_gap = float(median_gap / (mad + 1e-9))
        else:
            mwu_stat, mwu_p = float("nan"), 1.0
            ks_stat, ks_p = float("nan"), 1.0
            cd = 0.0
            wasserstein = float("nan")
            quantile_error = float("nan")
            real_mean = float("nan")
            generated_mean = float("nan")
            real_median = float("nan")
            generated_median = float("nan")
            median_gap = float("nan")
            mad_normalized_median_gap = float("nan")

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
            "wasserstein_distance": wasserstein,
            "quantile_error": quantile_error,
            "real_mean": real_mean,
            "generated_mean": generated_mean,
            "real_median": real_median,
            "generated_median": generated_median,
            "median_gap": median_gap,
            "abs_median_gap": abs(median_gap) if not np.isnan(median_gap) else float("nan"),
            "mad_normalized_median_gap": mad_normalized_median_gap,
            "abs_mad_normalized_median_gap": (
                abs(mad_normalized_median_gap)
                if not np.isnan(mad_normalized_median_gap)
                else float("nan")
            ),
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
    real_df: pd.DataFrame | None = None,
    before_df: pd.DataFrame | None = None,
    after_df: pd.DataFrame | None = None,
    metrics: List[str] | None = None,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 0,
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

        b_mwu_p = b.get("mwu_p_value", 1.0)
        a_mwu_p = a.get("mwu_p_value", 1.0)
        b_ks_p = b.get("ks_p_value", 1.0)
        a_ks_p = a.get("ks_p_value", 1.0)
        b_cd = b.get("cliffs_delta", 0.0)
        a_cd = a.get("cliffs_delta", 0.0)
        b_wasserstein = b.get("wasserstein_distance", float("nan"))
        a_wasserstein = a.get("wasserstein_distance", float("nan"))
        b_quantile_error = b.get("quantile_error", float("nan"))
        a_quantile_error = a.get("quantile_error", float("nan"))
        b_median_gap = b.get("median_gap", float("nan"))
        a_median_gap = a.get("median_gap", float("nan"))
        b_abs_median_gap = b.get("abs_median_gap", abs(b_median_gap))
        a_abs_median_gap = a.get("abs_median_gap", abs(a_median_gap))
        b_mad_gap = b.get("mad_normalized_median_gap", float("nan"))
        a_mad_gap = a.get("mad_normalized_median_gap", float("nan"))
        b_abs_mad_gap = b.get("abs_mad_normalized_median_gap", abs(b_mad_gap))
        a_abs_mad_gap = a.get("abs_mad_normalized_median_gap", abs(a_mad_gap))
        real_mean = b.get("real_mean", float("nan"))
        before_mean = b.get("generated_mean", float("nan"))
        after_mean = a.get("generated_mean", float("nan"))
        b_fail = b.get("empirical_fail_rate", 0.0)
        a_fail = a.get("empirical_fail_rate", 0.0)

        abs_delta_reduction = abs(b_cd) - abs(a_cd)
        wasserstein_reduction = (
            float(b_wasserstein - a_wasserstein)
            if not np.isnan(b_wasserstein) and not np.isnan(a_wasserstein)
            else float("nan")
        )
        quantile_error_reduction = (
            float(b_quantile_error - a_quantile_error)
            if not np.isnan(b_quantile_error) and not np.isnan(a_quantile_error)
            else float("nan")
        )
        abs_median_gap_reduction = (
            float(b_abs_median_gap - a_abs_median_gap)
            if not np.isnan(b_abs_median_gap) and not np.isnan(a_abs_median_gap)
            else float("nan")
        )
        abs_mad_gap_reduction = (
            float(b_abs_mad_gap - a_abs_mad_gap)
            if not np.isnan(b_abs_mad_gap) and not np.isnan(a_abs_mad_gap)
            else float("nan")
        )
        mean_gap_reduction = (
            float(abs(real_mean - before_mean) - abs(real_mean - after_mean))
            if not np.isnan(real_mean) and not np.isnan(before_mean) and not np.isnan(after_mean)
            else float("nan")
        )
        fail_rate_reduction = b_fail - a_fail
        strict_effect_improved = abs(a_cd) < abs(b_cd)
        strict_fail_improved = a_fail < b_fail
        improved = strict_effect_improved and strict_fail_improved
        closer_by_wasserstein = (
            bool(wasserstein_reduction > 0)
            if not np.isnan(wasserstein_reduction)
            else False
        )
        closer_by_quantile = (
            bool(quantile_error_reduction > 0)
            if not np.isnan(quantile_error_reduction)
            else False
        )
        closer_by_mean_gap = (
            bool(mean_gap_reduction > 0)
            if not np.isnan(mean_gap_reduction)
            else False
        )
        closer_by_abs_median_gap = (
            bool(abs_median_gap_reduction > 0)
            if not np.isnan(abs_median_gap_reduction)
            else False
        )
        closer_by_abs_mad_gap = (
            bool(abs_mad_gap_reduction > 0)
            if not np.isnan(abs_mad_gap_reduction)
            else False
        )

        bootstrap_ci_low = float("nan")
        bootstrap_ci_high = float("nan")
        if (
            real_df is not None
            and before_df is not None
            and after_df is not None
            and metric in real_df.columns
            and metric in before_df.columns
            and metric in after_df.columns
        ):
            bootstrap_ci_low, bootstrap_ci_high = _bootstrap_wasserstein_reduction_ci(
                real_df[metric].dropna().tolist(),
                before_df[metric].dropna().tolist(),
                after_df[metric].dropna().tolist(),
                n_bootstrap=bootstrap_iterations,
                seed=bootstrap_seed + metrics.index(metric) if metrics and metric in metrics else bootstrap_seed,
            )

        per_metric[metric] = {
            "before_mwu_p": b_mwu_p,
            "after_mwu_p": a_mwu_p,
            "before_ks_p": b_ks_p,
            "after_ks_p": a_ks_p,
            "before_cliffs_delta": b_cd,
            "after_cliffs_delta": a_cd,
            "abs_delta_reduction": abs_delta_reduction,
            "before_wasserstein_distance": b_wasserstein,
            "after_wasserstein_distance": a_wasserstein,
            "wasserstein_reduction": wasserstein_reduction,
            "before_quantile_error": b_quantile_error,
            "after_quantile_error": a_quantile_error,
            "quantile_error_reduction": quantile_error_reduction,
            "real_mean": real_mean,
            "before_generated_mean": before_mean,
            "after_generated_mean": after_mean,
            "mean_gap_reduction": mean_gap_reduction,
            "before_real_median": b.get("real_median", float("nan")),
            "before_generated_median": b.get("generated_median", float("nan")),
            "after_generated_median": a.get("generated_median", float("nan")),
            "before_median_gap": b_median_gap,
            "after_median_gap": a_median_gap,
            "before_abs_median_gap": b_abs_median_gap,
            "after_abs_median_gap": a_abs_median_gap,
            "abs_median_gap_reduction": abs_median_gap_reduction,
            "before_mad_normalized_median_gap": b_mad_gap,
            "after_mad_normalized_median_gap": a_mad_gap,
            "before_abs_mad_normalized_median_gap": b_abs_mad_gap,
            "after_abs_mad_normalized_median_gap": a_abs_mad_gap,
            "abs_mad_normalized_median_gap_reduction": abs_mad_gap_reduction,
            "bootstrap_ci_improvement_low": bootstrap_ci_low,
            "bootstrap_ci_improvement_high": bootstrap_ci_high,
            "bootstrap_ci_improvement_metric": "wasserstein_reduction",
            "before_fail_rate": b_fail,
            "after_fail_rate": a_fail,
            "fail_rate_reduction": fail_rate_reduction,
            "strict_effect_improved": strict_effect_improved,
            "strict_fail_improved": strict_fail_improved,
            "improved": improved,
            "closer_by_wasserstein": closer_by_wasserstein,
            "closer_by_quantile": closer_by_quantile,
            "closer_by_mean_gap": closer_by_mean_gap,
            "closer_by_abs_median_gap": closer_by_abs_median_gap,
            "closer_by_abs_mad_gap": closer_by_abs_mad_gap,
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

    def _avg_metric(result_dict: dict, key: str) -> float:
        vals = [
            float(info.get(key, float("nan")))
            for info in result_dict.values()
            if not np.isnan(float(info.get(key, float("nan"))))
        ]
        return float(np.mean(vals)) if vals else float("nan")

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
        "avg_wasserstein_distance_before": _avg_metric(before_results, "wasserstein_distance"),
        "avg_wasserstein_distance_after": _avg_metric(after_results, "wasserstein_distance"),
        "avg_quantile_error_before": _avg_metric(before_results, "quantile_error"),
        "avg_quantile_error_after": _avg_metric(after_results, "quantile_error"),
        "avg_abs_median_gap_before": _avg_metric(before_results, "abs_median_gap"),
        "avg_abs_median_gap_after": _avg_metric(after_results, "abs_median_gap"),
        "avg_abs_mad_normalized_median_gap_before": _avg_metric(
            before_results, "abs_mad_normalized_median_gap"
        ),
        "avg_abs_mad_normalized_median_gap_after": _avg_metric(
            after_results, "abs_mad_normalized_median_gap"
        ),
        "avg_generated_mean_before": _avg_metric(before_results, "generated_mean"),
        "avg_generated_mean_after": _avg_metric(after_results, "generated_mean"),
        "overall_fail_rate_before": overall_fail_before,
        "overall_fail_rate_after": overall_fail_after,
        "overall_pass_rate_before": 1.0 - overall_fail_before,
        "overall_pass_rate_after": 1.0 - overall_fail_after,
        "strict_improved_count": sum(1 for info in per_metric.values() if info.get("improved")),
        "closer_by_wasserstein_count": sum(
            1 for info in per_metric.values() if info.get("closer_by_wasserstein")
        ),
        "closer_by_quantile_count": sum(
            1 for info in per_metric.values() if info.get("closer_by_quantile")
        ),
        "closer_by_mean_gap_count": sum(
            1 for info in per_metric.values() if info.get("closer_by_mean_gap")
        ),
        "closer_by_abs_median_gap_count": sum(
            1 for info in per_metric.values() if info.get("closer_by_abs_median_gap")
        ),
        "closer_by_abs_mad_gap_count": sum(
            1 for info in per_metric.values() if info.get("closer_by_abs_mad_gap")
        ),
    }

    return {"per_metric": per_metric, "summary": summary}
