#!/usr/bin/env python3
"""Consolidate all baseline results into a single CSV.

Auto-discovers baselines from two sources:
  1. artifacts/baselines/*/ — directories with baseline_config.json + vanilla/thread_scores.csv
  2. Manually registered calibration before_calibration dirs (for models that only ran calibration)

Usage:
    python scripts/evaluation/consolidate_baselines.py

Output: artifacts/baselines/all_baselines_summary.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sp_stats

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
BASELINES_DIR = ARTIFACTS / "baselines"
OUT_PATH = BASELINES_DIR / "all_baselines_summary.csv"

SKIP_COLS = {"dataset", "product", "thread_id", "dominant_emotion", "thread_title"}

# ── Domain name normalization ──
# baseline_config.json stores category_name like "credit_cards", "camera_product", etc.
# Normalize to short names for the summary CSV.
_DOMAIN_ALIASES = {
    "camera_product": "camera",
    "cell_phone_product": "cell_phone",
    "headphone_product": "headphone",
    "laptop_product": "laptop",
    "credit_cards": "credit_cards",
}

# ── Canonical real thread_scores.csv per domain ──
# Maps domain → path to FULL real thread_scores.csv.
# Auto-populated from the first gpt-4o-mini baseline discovered per domain.
_CANONICAL_REAL: dict[str, Path] = {}

# ── Extra baselines from calibration before_calibration dirs ──
# (domain, model, real_csv_relative, candidates_dir_relative)
# These are only used when there is no dedicated baseline dir for the model.
_CALIBRATION_BASELINES: list[tuple[str, str, str, str]] = [
    ("credit_cards", "gpt-5-mini",
     "baselines/credit_cards_20260426_164025/real/thread_scores.csv",
     "credit_card_calibration_runs_gpt5mini_sim/calibration_20260503_005000/before_calibration/candidates"),
    ("cell_phone", "gpt-5-mini",
     "baselines/cell_phone_product_20260501_223134/real/thread_scores.csv",
     "cell_phone_calibration_runs_gpt5mini_sim/calibration_20260503_235413/before_calibration/candidates"),
    ("laptop", "gpt-5-mini",
     "baselines/laptop_product/real/thread_scores.csv",
     "laptop_calibration_runs_gpt5mini_sim/calibration_20260503_074028/before_calibration/candidates"),
]

FIELDS = [
    "domain", "model", "metric", "real_n", "sim_n",
    "real_mean", "real_median", "sim_mean", "sim_median",
    "mwu_p_value", "ks_p_value",
    "cliffs_delta", "abs_cliffs_delta", "cliffs_delta_interpretation",
    "wasserstein", "quantile_error", "empirical_fail_rate",
]


# ── Stat computation ──

def compute_stats(real_vals: list[float], sim_vals: list[float]) -> dict[str, Any] | None:
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
    cd_interp = (
        "negligible" if abs_cd < 0.147
        else "small" if abs_cd < 0.33
        else "medium" if abs_cd < 0.474
        else "large"
    )

    wass = float(sp_stats.wasserstein_distance(real, sim))
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    qe = float(np.mean(np.abs(np.quantile(real, qs) - np.quantile(sim, qs))))
    p025, p975 = np.percentile(real, 2.5), np.percentile(real, 97.5)
    efr = float(np.sum((sim < p025) | (sim > p975)) / len(sim))

    return {
        "real_n": int(len(real)), "sim_n": int(len(sim)),
        "real_mean": float(np.mean(real)), "real_median": float(np.median(real)),
        "sim_mean": float(np.mean(sim)), "sim_median": float(np.median(sim)),
        "mwu_p_value": float(mwu_p), "ks_p_value": float(ks_p),
        "cliffs_delta": cd, "abs_cliffs_delta": abs_cd,
        "cliffs_delta_interpretation": cd_interp,
        "wasserstein": wass, "quantile_error": qe, "empirical_fail_rate": efr,
    }


def _is_summary_row(row: dict[str, str]) -> bool:
    """Return True for per-product aggregate rows that should be excluded."""
    tid = row.get("thread_id", "")
    return tid.startswith("__summary") or tid.startswith("__aggregate")


def read_metric_cols(csv_path: Path) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if _is_summary_row(row):
                continue
            for k, v in row.items():
                if k in SKIP_COLS:
                    continue
                try:
                    data.setdefault(k, []).append(float(v))
                except (ValueError, TypeError):
                    pass
    return data


def collect_sim_from_candidates(cands_dir: Path) -> dict[str, list[float]]:
    sim: dict[str, list[float]] = {}
    for csv_file in sorted(cands_dir.rglob("thread_metrics_summary.csv")):
        with open(csv_file) as f:
            for row in csv.DictReader(f):
                if _is_summary_row(row):
                    continue
                for k, v in row.items():
                    if k in SKIP_COLS:
                        continue
                    try:
                        sim.setdefault(k, []).append(float(v))
                    except (ValueError, TypeError):
                        pass
    return sim


def process_pair(
    domain: str, model: str,
    real: dict[str, list[float]], sim: dict[str, list[float]],
) -> list[dict[str, Any]]:
    metrics = sorted(set(real.keys()) & set(sim.keys()))
    rows = []
    for m in metrics:
        s = compute_stats(real[m], sim[m])
        if s:
            s.update(domain=domain, model=model, metric=m)
            rows.append(s)
    return rows


def _detect_model(baseline_dir: Path) -> str:
    """Detect model name from baseline_config.json or sim run_config.json."""
    # 1. Check baseline_config.json for llm_model field
    cfg = baseline_dir / "baseline_config.json"
    if cfg.exists():
        data = json.loads(cfg.read_text())
        if data.get("llm_model"):
            return data["llm_model"]

    # 2. Fall back to first sim run's run_config.json
    for rc in sorted(baseline_dir.rglob("run_config.json")):
        try:
            data = json.loads(rc.read_text())
            if data.get("llm_model"):
                return data["llm_model"]
        except Exception:
            pass

    return "gpt-4o-mini"  # default for old baselines without model field


def _detect_domain(baseline_dir: Path) -> str:
    """Detect domain from baseline_config.json."""
    cfg = baseline_dir / "baseline_config.json"
    if cfg.exists():
        data = json.loads(cfg.read_text())
        raw = data.get("category_name", "")
        return _DOMAIN_ALIASES.get(raw, raw)
    return baseline_dir.name


# ── Main ──

def main() -> None:
    all_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()  # (domain, model) pairs already processed

    # ── Phase 1: Auto-discover from artifacts/baselines/*/ ──
    print("Discovering baselines from artifacts/baselines/ ...")
    for d in sorted(BASELINES_DIR.iterdir()):
        if not d.is_dir():
            continue
        real_csv = d / "real" / "thread_scores.csv"
        sim_csv = d / "vanilla" / "thread_scores.csv"
        if not real_csv.exists() or not sim_csv.exists():
            continue

        domain = _detect_domain(d)
        model = _detect_model(d)

        # Track canonical real path per domain (first one wins)
        if domain not in _CANONICAL_REAL:
            _CANONICAL_REAL[domain] = real_csv

        print(f"  {domain:>15} / {model:<25}", end=" ")
        real = read_metric_cols(real_csv)
        sim = read_metric_cols(sim_csv)
        rows = process_pair(domain, model, real, sim)
        all_rows.extend(rows)
        seen.add((domain, model))

        r_n = len(next(iter(real.values()), []))
        s_n = len(next(iter(sim.values()), []))
        print(f"→ {len(rows)} metrics  (real={r_n}, sim={s_n})")

    # ── Phase 2: Calibration before_calibration dirs ──
    for domain, model, real_rel, cands_rel in _CALIBRATION_BASELINES:
        if (domain, model) in seen:
            continue  # already have a dedicated baseline
        real_path = ARTIFACTS / real_rel
        cands_path = ARTIFACTS / cands_rel
        if not real_path.exists() or not cands_path.exists():
            continue

        print(f"  {domain:>15} / {model:<25}", end=" (calibration) ")
        real = read_metric_cols(real_path)
        sim = collect_sim_from_candidates(cands_path)
        rows = process_pair(domain, model, real, sim)
        all_rows.extend(rows)
        seen.add((domain, model))

        r_n = len(next(iter(real.values()), []))
        s_n = len(next(iter(sim.values()), []))
        print(f"→ {len(rows)} metrics  (real={r_n}, sim={s_n})")

    # ── Write ──
    all_rows.sort(key=lambda r: (r["domain"], r["model"], r["metric"]))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    print(f"\nWrote {len(all_rows)} rows → {OUT_PATH}")
    from collections import Counter
    for (d, m), c in sorted(Counter((r["domain"], r["model"]) for r in all_rows).items()):
        print(f"    {d:>15} | {m:>25} | {c} metrics")


if __name__ == "__main__":
    main()
