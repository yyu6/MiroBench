from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def cliffs_delta(candidate: Sequence[float], real: Sequence[float]) -> float:
    """Return candidate-minus-real Cliff's delta."""

    candidate_array = clean_array(candidate)
    real_array = clean_array(real)
    if not candidate_array.size or not real_array.size:
        raise ValueError("Cliff's delta requires two non-empty samples")
    differences = candidate_array[:, None] - real_array[None, :]
    pairs = candidate_array.size * real_array.size
    return float(
        (np.count_nonzero(differences > 0) - np.count_nonzero(differences < 0)) / pairs
    )


def distribution_stats(
    real: Sequence[float], candidate: Sequence[float]
) -> dict[str, float]:
    """Compare two non-empty distributions using the formal evaluation tests."""

    real_array = clean_array(real)
    candidate_array = clean_array(candidate)
    if not real_array.size or not candidate_array.size:
        raise ValueError("distribution comparison requires two non-empty samples")
    mwu_stat, mwu_p = scipy_stats.mannwhitneyu(
        real_array, candidate_array, alternative="two-sided"
    )
    ks_stat, ks_p = scipy_stats.ks_2samp(real_array, candidate_array)
    return {
        "mwu_statistic": float(mwu_stat),
        "mwu_p_value": float(mwu_p),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
        "cliffs_delta": cliffs_delta(candidate_array, real_array),
        "wasserstein_distance": float(
            scipy_stats.wasserstein_distance(real_array, candidate_array)
        ),
    }


def evaluate_group_vs_real(
    real_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    metrics: Sequence[str],
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Return formal group statistics for every requested metric."""

    results: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        real = column_values(real_frame, metric)
        candidate = column_values(candidate_frame, metric)
        if not real.size or not candidate.size:
            results[metric] = unavailable_stats()
            continue
        comparison = distribution_stats(real, candidate)
        real_mean = float(np.mean(real))
        candidate_mean = float(np.mean(candidate))
        real_median = float(np.median(real))
        candidate_median = float(np.median(candidate))
        median_gap = candidate_median - real_median
        real_mad = float(np.median(np.abs(real - real_median)))
        quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
        quantile_error = float(
            np.mean(
                np.abs(np.quantile(real, quantiles) - np.quantile(candidate, quantiles))
            )
        )
        normalized_median_gap = median_gap / (real_mad + 1e-9)
        results[metric] = {
            **comparison,
            "quantile_error": quantile_error,
            "real_mean": real_mean,
            "generated_mean": candidate_mean,
            "real_median": real_median,
            "generated_median": candidate_median,
            "median_gap": median_gap,
            "abs_median_gap": abs(median_gap),
            "mad_normalized_median_gap": normalized_median_gap,
            "abs_mad_normalized_median_gap": abs(normalized_median_gap),
            "direction": direction(comparison, alpha=alpha),
            "empirical_fail_rate": float(
                np.mean([empirical_p_value(real, value) < alpha for value in candidate])
            ),
        }
    return results


def clean_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def column_values(frame: pd.DataFrame, metric: str) -> np.ndarray:
    if metric not in frame.columns:
        return np.asarray([], dtype=float)
    return clean_array(frame[metric].tolist())


def empirical_p_value(real: Sequence[float], candidate: float) -> float:
    real_array = clean_array(real)
    if not real_array.size:
        return 1.0
    center = float(np.median(real_array))
    distance = abs(float(candidate) - center)
    count = np.count_nonzero(np.abs(real_array - center) >= distance)
    return float((count + 1) / (len(real_array) + 1))


def direction(comparison: dict[str, float], *, alpha: float) -> str:
    if comparison["mwu_p_value"] >= alpha:
        return "similar"
    return "generated_higher" if comparison["cliffs_delta"] > 0 else "generated_lower"


def unavailable_stats() -> dict[str, Any]:
    return {
        "mwu_statistic": float("nan"),
        "mwu_p_value": 1.0,
        "ks_statistic": float("nan"),
        "ks_p_value": 1.0,
        "cliffs_delta": 0.0,
        "wasserstein_distance": float("nan"),
        "quantile_error": float("nan"),
        "real_mean": float("nan"),
        "generated_mean": float("nan"),
        "real_median": float("nan"),
        "generated_median": float("nan"),
        "median_gap": float("nan"),
        "abs_median_gap": float("nan"),
        "mad_normalized_median_gap": float("nan"),
        "abs_mad_normalized_median_gap": float("nan"),
        "direction": "similar",
        "empirical_fail_rate": 0.0,
    }
