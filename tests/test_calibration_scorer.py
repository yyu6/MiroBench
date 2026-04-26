"""
Tests for calibration/scorer.py — metric loading, real baseline, candidate diagnostics, selection.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibration.scorer import (
    load_thread_metrics,
    compute_real_baseline,
    compute_baseline_from_csv,
    score_candidate,
    select_best_candidate,
    DEFAULT_METRICS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_METRICS = ["toxicity_mean", "length_std", "max_depth"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts as a CSV to path."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_real_rows(n: int = 50, rng_seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(rng_seed)
    return [
        {
            "toxicity_mean": float(rng.uniform(0.01, 0.3)),
            "length_std": float(rng.uniform(5.0, 50.0)),
            "max_depth": float(rng.integers(1, 10)),
        }
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# TestLoadThreadMetrics
# ---------------------------------------------------------------------------


class TestLoadThreadMetrics:
    def test_reads_csv(self, tmp_path: Path) -> None:
        rows = _make_real_rows(10)
        csv_path = tmp_path / "thread_metrics_summary.csv"
        _write_csv(csv_path, rows)

        df = load_thread_metrics(tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "toxicity_mean" in df.columns

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_thread_metrics(tmp_path)

    def test_aggregates_category_subdirs(self, tmp_path: Path) -> None:
        """Category dir with multiple product subdirs is aggregated."""
        for product in ("product_a", "product_b"):
            sub = tmp_path / product
            sub.mkdir()
            _write_csv(sub / "thread_metrics_summary.csv", _make_real_rows(5))

        df = load_thread_metrics(tmp_path)
        assert len(df) == 10  # 5 + 5
        assert "_product_dir" in df.columns


# ---------------------------------------------------------------------------
# TestComputeRealBaseline
# ---------------------------------------------------------------------------


class TestComputeBaselineFromCsv:
    def test_reads_csv_directly(self, tmp_path: Path) -> None:
        rows = _make_real_rows(30)
        csv_path = tmp_path / "thread_scores_val.csv"
        _write_csv(csv_path, rows)

        baseline = compute_baseline_from_csv(csv_path, MINIMAL_METRICS)

        assert isinstance(baseline, dict)
        for metric in MINIMAL_METRICS:
            assert metric in baseline
            info = baseline[metric]
            assert "median" in info
            assert "mean" in info
            assert "values" in info
            assert len(info["values"]) == 30
            expected_median = float(np.median([r[metric] for r in rows]))
            assert info["median"] == pytest.approx(expected_median)

    def test_missing_column_returns_empty(self, tmp_path: Path) -> None:
        rows = [{"toxicity_mean": 0.1}]
        csv_path = tmp_path / "partial.csv"
        _write_csv(csv_path, rows)

        baseline = compute_baseline_from_csv(csv_path, MINIMAL_METRICS)
        assert baseline["toxicity_mean"]["values"] == [0.1]
        assert baseline["length_std"]["values"] == []
        assert baseline["max_depth"]["values"] == []


# ---------------------------------------------------------------------------
# TestComputeRealBaseline
# ---------------------------------------------------------------------------


class TestComputeRealBaseline:
    def test_returns_medians_and_values(self, tmp_path: Path) -> None:
        rows = _make_real_rows(50)
        csv_path = tmp_path / "thread_metrics_summary.csv"
        _write_csv(csv_path, rows)

        baseline = compute_real_baseline(tmp_path, MINIMAL_METRICS)

        assert isinstance(baseline, dict)
        for metric in MINIMAL_METRICS:
            assert metric in baseline
            info = baseline[metric]
            assert "median" in info
            assert "mean" in info
            assert "values" in info
            assert isinstance(info["values"], list)
            assert len(info["values"]) == 50
            # Median and mean must be close-ish for uniform-ish distributions
            expected_median = float(np.median([r[metric] for r in rows]))
            assert info["median"] == pytest.approx(expected_median)


# ---------------------------------------------------------------------------
# TestScoreCandidate
# ---------------------------------------------------------------------------


class TestScoreCandidate:
    def _setup_dirs(self, tmp_path: Path):
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        return real_dir, sim_dir

    def test_returns_diagnostic(self, tmp_path: Path) -> None:
        real_dir, sim_dir = self._setup_dirs(tmp_path)

        # Write 50-row real CSV
        real_rows = _make_real_rows(50)
        _write_csv(real_dir / "thread_metrics_summary.csv", real_rows)

        # Write 1-row sim CSV (values close to real median — should mostly pass)
        real_medians = {
            m: float(np.median([r[m] for r in real_rows])) for m in MINIMAL_METRICS
        }
        sim_rows = [real_medians.copy()]
        _write_csv(sim_dir / "thread_metrics_summary.csv", sim_rows)

        baseline = compute_real_baseline(real_dir, MINIMAL_METRICS)
        result = score_candidate(sim_dir, baseline, MINIMAL_METRICS)

        # Top-level keys
        assert "fail_rate" in result
        assert "mean_abs_delta" in result
        assert "per_metric" in result

        # Fail rate should be 0 (median values pass)
        assert result["fail_rate"] == pytest.approx(0.0)

        # per_metric structure
        for metric in MINIMAL_METRICS:
            assert metric in result["per_metric"]
            m_info = result["per_metric"][metric]
            assert "real_median" in m_info
            assert "generated_median" in m_info
            assert "cliffs_delta" in m_info
            assert "fail_rate" in m_info
            assert "direction" in m_info
            assert "threads" in m_info

            # Each thread entry
            for thread in m_info["threads"]:
                assert "value" in thread
                assert "empirical_p" in thread
                assert "percentile" in thread
                assert "direction" in thread
                assert "pass" in thread

    def test_extreme_sim_fails(self, tmp_path: Path) -> None:
        real_dir, sim_dir = self._setup_dirs(tmp_path)

        real_rows = _make_real_rows(50)
        _write_csv(real_dir / "thread_metrics_summary.csv", real_rows)

        # Sim with extreme values — should fail all metrics
        extreme_rows = [{"toxicity_mean": 999.0, "length_std": 999.0, "max_depth": 999.0}]
        _write_csv(sim_dir / "thread_metrics_summary.csv", extreme_rows)

        baseline = compute_real_baseline(real_dir, MINIMAL_METRICS)
        result = score_candidate(sim_dir, baseline, MINIMAL_METRICS)

        assert result["fail_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestSelectBestCandidate
# ---------------------------------------------------------------------------


class TestSelectBestCandidate:
    def _make_scored(self, fail_rate: float, mean_abs_delta: float) -> dict:
        return {
            "fail_rate": fail_rate,
            "mean_abs_delta": mean_abs_delta,
            "per_metric": {},
        }

    def test_selects_lowest_fail_rate(self) -> None:
        candidates = [
            self._make_scored(0.5, 0.1),
            self._make_scored(0.1, 0.5),
            self._make_scored(0.3, 0.2),
        ]
        best = select_best_candidate(candidates)
        assert best["fail_rate"] == pytest.approx(0.1)

    def test_breaks_tie_with_delta(self) -> None:
        # Two candidates with same fail_rate — pick lower delta
        candidates = [
            self._make_scored(0.2, 0.5),
            self._make_scored(0.2, 0.1),
        ]
        best = select_best_candidate(candidates)
        assert best["mean_abs_delta"] == pytest.approx(0.1)

    def test_single_candidate(self) -> None:
        candidates = [self._make_scored(0.3, 0.4)]
        best = select_best_candidate(candidates)
        assert best["fail_rate"] == pytest.approx(0.3)
