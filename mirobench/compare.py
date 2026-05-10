"""Statistical comparison of generated thread scores against real reference data."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

SKIP_COLS = {"dataset", "product", "thread_id", "dominant_emotion", "thread_title"}

FIELDS = [
    "domain", "model", "metric", "real_n", "sim_n",
    "real_mean", "real_median", "sim_mean", "sim_median",
    "mwu_p_value", "ks_p_value",
    "cliffs_delta", "abs_cliffs_delta", "cliffs_delta_interpretation",
    "wasserstein", "quantile_error", "empirical_fail_rate",
]


def compute_stats(real_vals: list[float], sim_vals: list[float]) -> dict[str, Any] | None:
    """Compute statistical comparison between real and simulated metric values.

    Returns a dict with MWU p-value, KS p-value, Cliff's delta, Wasserstein
    distance, quantile error, and empirical fail rate.
    """
    real = np.array(real_vals, dtype=float)
    sim = np.array(sim_vals, dtype=float)
    real, sim = real[~np.isnan(real)], sim[~np.isnan(sim)]
    if len(real) < 2 or len(sim) < 2:
        return None

    try:
        _, mwu_p = sp_stats.mannwhitneyu(real, sim, alternative="two-sided")
    except Exception:
        mwu_p = float("nan")
    try:
        _, ks_p = sp_stats.ks_2samp(real, sim)
    except Exception:
        ks_p = float("nan")

    more = sum(float(np.sum(real > s)) for s in sim)
    less = sum(float(np.sum(real < s)) for s in sim)
    n = len(real) * len(sim)
    cd = (more - less) / n
    abs_cd = abs(cd)
    if abs_cd < 0.147:
        interp = "negligible"
    elif abs_cd < 0.33:
        interp = "small"
    elif abs_cd < 0.474:
        interp = "medium"
    else:
        interp = "large"

    wd = float(sp_stats.wasserstein_distance(real, sim))

    real_q = np.percentile(real, np.arange(0, 101, 5))
    sim_q = np.percentile(sim, np.arange(0, 101, 5))
    qe = float(np.mean(np.abs(real_q - sim_q)))

    real_low = float(np.percentile(real, 2.5))
    real_high = float(np.percentile(real, 97.5))
    if len(sim) > 0:
        outside = np.sum((sim < real_low) | (sim > real_high))
        fail_rate = float(outside / len(sim))
    else:
        fail_rate = 0.0

    return {
        "real_n": len(real), "sim_n": len(sim),
        "real_mean": float(np.mean(real)), "real_median": float(np.median(real)),
        "sim_mean": float(np.mean(sim)), "sim_median": float(np.median(sim)),
        "mwu_p_value": mwu_p, "ks_p_value": ks_p,
        "cliffs_delta": cd, "abs_cliffs_delta": abs_cd,
        "cliffs_delta_interpretation": interp,
        "wasserstein": wd, "quantile_error": qe,
        "empirical_fail_rate": fail_rate,
    }


def read_metric_cols(csv_path: Path) -> dict[str, list[float]]:
    """Read a thread_scores.csv and return {metric_name: [values]}."""
    data: dict[str, list[float]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            tid = row.get("thread_id", "")
            if tid == "__summary_mean__":
                continue
            for k, v in row.items():
                if k in SKIP_COLS:
                    continue
                try:
                    data.setdefault(k, []).append(float(v))
                except (ValueError, TypeError):
                    pass
    return data


def compare_against_reference(
    sim_csv: Path,
    ref_csv: Path,
    domain: str = "",
    model: str = "",
) -> list[dict[str, Any]]:
    """Compare a simulated thread_scores.csv against a real reference.

    Returns a list of dicts (one per metric) with statistical comparisons.
    """
    sim = read_metric_cols(sim_csv)
    real = read_metric_cols(ref_csv)
    metrics = sorted(set(real.keys()) & set(sim.keys()))

    rows = []
    for m in metrics:
        s = compute_stats(real[m], sim[m])
        if s:
            s.update(domain=domain, model=model, metric=m)
            rows.append(s)
    return rows


def write_comparison_csv(rows: list[dict], output_path: Path) -> None:
    """Write comparison rows to CSV."""
    rows.sort(key=lambda r: (r.get("domain", ""), r.get("metric", "")))
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in FIELDS} for r in rows])
