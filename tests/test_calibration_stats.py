"""
Tests for calibration/stats.py — statistical functions for the calibration system.
"""
import math
import numpy as np
import pandas as pd
import pytest

from calibration.stats import (
    cliffs_delta,
    empirical_p_value,
    empirical_percentile,
    evaluate_group_vs_real,
    diagnose_single_generation,
    compare_before_after,
)


# ---------------------------------------------------------------------------
# TestCliffsDelta
# ---------------------------------------------------------------------------

class TestCliffsDelta:
    def test_identical(self):
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        delta = cliffs_delta(x, y)
        assert abs(delta) < 1e-9, "Identical distributions should give delta=0"

    def test_x_greater_than_y(self):
        x = [10, 11, 12, 13, 14]
        y = [1, 2, 3, 4, 5]
        delta = cliffs_delta(x, y)
        assert delta == pytest.approx(1.0), "x all greater than y -> delta=1"

    def test_x_less_than_y(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 11, 12, 13, 14]
        delta = cliffs_delta(x, y)
        assert delta == pytest.approx(-1.0), "x all less than y -> delta=-1"

    def test_partial_overlap(self):
        x = [1, 2, 3, 4, 5, 6]
        y = [3, 4, 5, 6, 7, 8]
        delta = cliffs_delta(x, y)
        assert -1.0 < delta < 0.0, "Partial overlap with y higher -> negative delta"

    def test_positive_means_x_higher(self):
        x = [5, 6, 7]
        y = [1, 2, 3]
        delta = cliffs_delta(x, y)
        assert delta > 0, "Positive delta means x tends to be higher than y"

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError):
            cliffs_delta([], [1, 2, 3])
        with pytest.raises(ValueError):
            cliffs_delta([1, 2, 3], [])
        with pytest.raises(ValueError):
            cliffs_delta([], [])

    def test_return_in_range(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50).tolist()
        y = rng.normal(0.5, 1, 50).tolist()
        delta = cliffs_delta(x, y)
        assert -1.0 <= delta <= 1.0


# ---------------------------------------------------------------------------
# TestEmpiricalPValue
# ---------------------------------------------------------------------------

class TestEmpiricalPValue:
    def test_at_median_high_p(self):
        real = list(range(1, 101))  # 1..100
        center = np.median(real)
        # gen_value == center -> gen_dist=0, all real_dist>=0 -> p close to 1
        p = empirical_p_value(real, center)
        assert p > 0.9, "Value at median should yield high p-value"

    def test_extreme_low_p(self):
        real = list(range(1, 101))
        # gen far outside distribution
        p = empirical_p_value(real, 1000.0)
        assert p < 0.05, "Extreme outlier should yield low p-value"

    def test_between_0_and_1(self):
        rng = np.random.default_rng(7)
        real = rng.normal(0, 1, 100).tolist()
        p = empirical_p_value(real, 0.5)
        assert 0.0 < p <= 1.0

    def test_formula_matches_spec(self):
        # Verify formula: p=(count(real_dist>=gen_dist)+1)/(n+1)
        real = [1.0, 2.0, 3.0, 4.0, 5.0]
        gen_value = 5.0
        center = np.median(real)  # 3.0
        gen_dist = abs(gen_value - center)  # 2.0
        real_dists = np.abs(np.array(real) - center)
        count = np.sum(real_dists >= gen_dist)
        n = len(real)
        expected_p = (count + 1) / (n + 1)
        p = empirical_p_value(real, gen_value)
        assert p == pytest.approx(expected_p), "p-value must follow the spec formula"

    def test_empty_returns_1(self):
        p = empirical_p_value([], 5.0)
        assert p == 1.0

    def test_nan_handling(self):
        real = [1.0, float('nan'), 3.0, 4.0, 5.0]
        # Should not raise; NaNs are ignored
        p = empirical_p_value(real, 3.0)
        assert 0.0 < p <= 1.0


# ---------------------------------------------------------------------------
# TestEmpiricalPercentile
# ---------------------------------------------------------------------------

class TestEmpiricalPercentile:
    def test_below_all(self):
        real = [10, 20, 30, 40, 50]
        pct = empirical_percentile(real, 0)
        assert pct == pytest.approx(0.0)

    def test_above_all(self):
        real = [10, 20, 30, 40, 50]
        pct = empirical_percentile(real, 100)
        assert pct == pytest.approx(100.0)

    def test_at_median(self):
        real = [1, 2, 3, 4, 5]
        median = float(np.median(real))
        pct = empirical_percentile(real, median)
        assert 40.0 <= pct <= 60.0, "Median should be near 50th percentile"

    def test_empty_returns_50(self):
        pct = empirical_percentile([], 5.0)
        assert pct == pytest.approx(50.0)

    def test_return_in_range(self):
        rng = np.random.default_rng(99)
        real = rng.normal(0, 1, 100).tolist()
        pct = empirical_percentile(real, 0.0)
        assert 0.0 <= pct <= 100.0


# ---------------------------------------------------------------------------
# TestEvaluateGroupVsReal
# ---------------------------------------------------------------------------

class TestEvaluateGroupVsReal:
    def _make_frames(self):
        rng = np.random.default_rng(0)
        real_df = pd.DataFrame({
            "toxicity": rng.normal(0.1, 0.05, 100),
            "sentiment": rng.normal(0.5, 0.1, 100),
        })
        gen_df = pd.DataFrame({
            "toxicity": rng.normal(0.1, 0.05, 50),
            "sentiment": rng.normal(0.5, 0.1, 50),
        })
        return real_df, gen_df

    def test_returns_dict_per_metric(self):
        real_df, gen_df = self._make_frames()
        result = evaluate_group_vs_real(real_df, gen_df, ["toxicity", "sentiment"])
        assert isinstance(result, dict)
        assert "toxicity" in result
        assert "sentiment" in result

    def test_has_required_fields(self):
        real_df, gen_df = self._make_frames()
        result = evaluate_group_vs_real(real_df, gen_df, ["toxicity"])
        info = result["toxicity"]
        required = {
            "mwu_statistic", "mwu_p_value", "ks_statistic", "ks_p_value",
            "cliffs_delta", "direction", "empirical_fail_rate",
        }
        assert required.issubset(info.keys()), f"Missing keys: {required - info.keys()}"

    def test_divergent_detected(self):
        # Clearly different distributions
        rng = np.random.default_rng(1)
        real_df = pd.DataFrame({"score": rng.normal(0.1, 0.02, 200)})
        gen_df = pd.DataFrame({"score": rng.normal(0.9, 0.02, 100)})
        result = evaluate_group_vs_real(real_df, gen_df, ["score"], alpha=0.05)
        info = result["score"]
        assert info["mwu_p_value"] < 0.05, "Clearly different distributions should be flagged"
        assert info["direction"] in ("generated_higher", "generated_lower")

    def test_similar_not_flagged(self):
        rng = np.random.default_rng(2)
        real_df = pd.DataFrame({"score": rng.normal(0.5, 0.1, 200)})
        gen_df = pd.DataFrame({"score": rng.normal(0.5, 0.1, 200)})
        result = evaluate_group_vs_real(real_df, gen_df, ["score"], alpha=0.05)
        info = result["score"]
        # empirical_fail_rate should be low for matching distributions
        assert info["empirical_fail_rate"] < 0.3


# ---------------------------------------------------------------------------
# TestDiagnoseSingleGeneration
# ---------------------------------------------------------------------------

class TestDiagnoseSingleGeneration:
    def _make_real_df(self):
        rng = np.random.default_rng(10)
        return pd.DataFrame({
            "toxicity": rng.normal(0.2, 0.05, 100),
            "sentiment": rng.normal(0.5, 0.1, 100),
        })

    def test_returns_dict(self):
        real_df = self._make_real_df()
        gen_row = pd.Series({"toxicity": 0.2, "sentiment": 0.5})
        result = diagnose_single_generation(real_df, gen_row, ["toxicity", "sentiment"])
        assert isinstance(result, dict)

    def test_required_fields(self):
        real_df = self._make_real_df()
        gen_row = pd.Series({"toxicity": 0.2, "sentiment": 0.5})
        result = diagnose_single_generation(real_df, gen_row, ["toxicity"])
        info = result["toxicity"]
        required = {
            "real_median", "generated_value", "empirical_p_value",
            "percentile", "direction", "diagnosis_flag",
        }
        assert required.issubset(info.keys()), f"Missing keys: {required - info.keys()}"

    def test_extreme_fails(self):
        real_df = self._make_real_df()
        # Toxicity far above real distribution
        gen_row = pd.Series({"toxicity": 5.0, "sentiment": 0.5})
        result = diagnose_single_generation(real_df, gen_row, ["toxicity"])
        assert result["toxicity"]["diagnosis_flag"] == "fail"
        assert result["toxicity"]["direction"] == "too_high"

    def test_median_passes(self):
        real_df = self._make_real_df()
        median_tox = float(np.median(real_df["toxicity"]))
        median_sent = float(np.median(real_df["sentiment"]))
        gen_row = pd.Series({"toxicity": median_tox, "sentiment": median_sent})
        result = diagnose_single_generation(real_df, gen_row, ["toxicity", "sentiment"])
        # Values at median should pass
        for metric in ["toxicity", "sentiment"]:
            assert result[metric]["diagnosis_flag"] == "pass", (
                f"{metric} at median should pass"
            )
            assert result[metric]["direction"] == "within_baseline"

    def test_direction_too_low(self):
        real_df = self._make_real_df()
        gen_row = pd.Series({"toxicity": -100.0, "sentiment": 0.5})
        result = diagnose_single_generation(real_df, gen_row, ["toxicity"])
        assert result["toxicity"]["diagnosis_flag"] == "fail"
        assert result["toxicity"]["direction"] == "too_low"


# ---------------------------------------------------------------------------
# TestCompareBeforeAfter
# ---------------------------------------------------------------------------

class TestCompareBeforeAfter:
    def _make_eval_result(self, fail_rate, cliffs_d):
        return {
            "score": {
                "mwu_statistic": 1000,
                "mwu_p_value": 0.001,
                "ks_statistic": 0.4,
                "ks_p_value": 0.001,
                "cliffs_delta": cliffs_d,
                "direction": "generated_higher",
                "empirical_fail_rate": fail_rate,
            }
        }

    def test_returns_per_metric_and_summary(self):
        before = self._make_eval_result(0.4, 0.5)
        after = self._make_eval_result(0.1, 0.1)
        result = compare_before_after(before, after)
        assert "per_metric" in result
        assert "summary" in result

    def test_improvement_detected(self):
        before = self._make_eval_result(0.4, 0.5)
        after = self._make_eval_result(0.1, 0.1)
        result = compare_before_after(before, after)
        metric_info = result["per_metric"]["score"]
        assert metric_info["improved"] is True
        assert metric_info["fail_rate_reduction"] == pytest.approx(0.3)
        assert metric_info["abs_delta_reduction"] == pytest.approx(0.4)

    def test_regression_not_improved(self):
        before = self._make_eval_result(0.1, 0.1)
        after = self._make_eval_result(0.4, 0.4)
        result = compare_before_after(before, after)
        assert result["per_metric"]["score"]["improved"] is False

    def test_summary_counts(self):
        before = {
            "score": {
                "mwu_p_value": 0.001, "ks_p_value": 0.001,
                "cliffs_delta": 0.6, "empirical_fail_rate": 0.5,
            },
            "toxicity": {
                "mwu_p_value": 0.001, "ks_p_value": 0.001,
                "cliffs_delta": 0.7, "empirical_fail_rate": 0.4,
            },
        }
        after = {
            "score": {
                "mwu_p_value": 0.8, "ks_p_value": 0.7,
                "cliffs_delta": 0.05, "empirical_fail_rate": 0.1,
            },
            "toxicity": {
                "mwu_p_value": 0.8, "ks_p_value": 0.6,
                "cliffs_delta": 0.08, "empirical_fail_rate": 0.05,
            },
        }
        result = compare_before_after(before, after, alpha=0.05)
        summary = result["summary"]
        required_summary_keys = {
            "metrics_sig_different_before",
            "metrics_sig_different_after",
            "avg_abs_cliffs_delta_before",
            "avg_abs_cliffs_delta_after",
            "overall_fail_rate_before",
            "overall_fail_rate_after",
            "overall_pass_rate_before",
            "overall_pass_rate_after",
        }
        assert required_summary_keys.issubset(summary.keys())
        assert summary["metrics_sig_different_before"] == 2
        assert summary["metrics_sig_different_after"] == 0
        assert summary["overall_fail_rate_before"] == pytest.approx(0.45)
        assert summary["overall_fail_rate_after"] == pytest.approx(0.075)
        assert summary["overall_pass_rate_before"] == pytest.approx(1 - 0.45)
        assert summary["overall_pass_rate_after"] == pytest.approx(1 - 0.075)
