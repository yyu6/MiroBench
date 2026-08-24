# Calibration System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an iterative LLM-driven calibration loop that tunes persona distributions and prompt text so generated Reddit discussions become statistically indistinguishable from real discussions.

**Architecture:** A `calibration/` package with 9 modules. The orchestrator runs iterations: generate 5 candidate discussions → score with statistical tests → select best → LLM diagnoses gaps and proposes overlay changes. Before/after group evaluations (50 vs 50) use Mann-Whitney U, KS test, and Cliff's delta. During-calibration diagnostics use empirical p-values per thread, and candidate selection now prefers validation group-level distance when that summary is available.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy.stats, openai SDK. Existing simulation pipeline via subprocess.

**Spec:** `docs/design/2026-04-26-calibration-system-design.md`

## Implementation Update (2026-04-28)

- `run_discussion.py` loads calibration overlays for persona generation and persists
  the overlay in `simulation_config.json` for auditability. The old
  repo-local OASIS patch layer has been removed; the current run path uses the
  vanilla MiroFish/OASIS runtime directly.
- Calibration overlays are now sanitized against `KnobRegistry` before candidate execution and when resuming saved state. Unknown or invalid knobs are dropped and logged instead of silently surviving into `best_overlay.json`.
- Candidate selection now uses validation group-level summaries when available: minimize mean absolute Cliff's delta first, then overall empirical fail rate, with the original per-thread diagnostics kept as tie-breakers and debugging signals.
- `prompt.length_cv` is not a supported knob in the runtime. Length-diversity adjustments should be expressed through registered knobs only.

---

## File Map

### New Files (calibration package)

| File | Responsibility |
|---|---|
| `calibration/__init__.py` | Package marker |
| `calibration/__main__.py` | `python -m calibration` entry |
| `calibration/cli.py` | Argument parsing, wiring |
| `calibration/stats.py` | Statistical functions: Cliff's delta, empirical p-value, percentile, group eval, single diagnostic, before/after comparison |
| `calibration/registry.py` | Knob definitions, validation, defaults |
| `calibration/overlay.py` | Load/save/merge/validate/diff overlays |
| `calibration/scorer.py` | Run metric suite, read CSVs, call stats |
| `calibration/reasoner.py` | LLM prompt assembly, response parsing, variant generation |
| `calibration/runner.py` | Subprocess pool for candidate simulations |
| `calibration/orchestrator.py` | Main loop: phases 1–4 |
| `calibration/log.py` | Calibration log read/write/append |

### New Test Files

| File | Tests for |
|---|---|
| `tests/test_calibration_stats.py` | stats.py |
| `tests/test_calibration_registry.py` | registry.py |
| `tests/test_calibration_overlay.py` | overlay.py |
| `tests/test_calibration_scorer.py` | scorer.py |
| `tests/test_calibration_reasoner.py` | reasoner.py |
| `tests/test_calibration_runner.py` | runner.py |
| `tests/test_calibration_log.py` | log.py |
| `tests/test_calibration_orchestrator.py` | orchestrator.py |

### Modified Files

| File | Change |
|---|---|
| `pyproject.toml` | Add pandas, numpy, scipy to dependencies |
| `run_discussion.py` | Add `--overlay` CLI arg, load and pass overlay dict |
| `product_reddit_sim/persona_gen.py` | Accept overlay dict, apply persona distribution overrides |
| `product_reddit_sim/config_builder.py` | Persist overlay into `simulation_config.json` for auditability |

---

## Task 1: Statistical Functions (`stats.py`)

**Files:**
- Create: `calibration/__init__.py`
- Create: `calibration/stats.py`
- Create: `tests/test_calibration_stats.py`

### Step 1.1: Create package and write tests for `cliffs_delta`

- [ ] Create `calibration/__init__.py` (empty file)
- [ ] Write test file:

```python
# tests/test_calibration_stats.py
"""Tests for calibration.stats."""
from __future__ import annotations

import numpy as np
import pytest

from calibration.stats import cliffs_delta


class TestCliffsDelta:
    def test_identical_groups(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        delta = cliffs_delta(x, y)
        assert delta == pytest.approx(0.0, abs=0.01)

    def test_x_strictly_greater(self):
        x = [10.0, 11.0, 12.0]
        y = [1.0, 2.0, 3.0]
        delta = cliffs_delta(x, y)
        assert delta == pytest.approx(1.0)

    def test_x_strictly_less(self):
        x = [1.0, 2.0, 3.0]
        y = [10.0, 11.0, 12.0]
        delta = cliffs_delta(x, y)
        assert delta == pytest.approx(-1.0)

    def test_partial_overlap(self):
        x = [1.0, 2.0, 5.0, 6.0]
        y = [3.0, 4.0, 7.0, 8.0]
        delta = cliffs_delta(x, y)
        assert -1.0 < delta < 0.0  # x tends lower

    def test_positive_means_x_higher(self):
        x = [5.0, 6.0, 7.0, 8.0, 9.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        delta = cliffs_delta(x, y)
        assert delta > 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cliffs_delta([], [1.0, 2.0])
        with pytest.raises(ValueError):
            cliffs_delta([1.0], [])
```

- [ ] Run: `cd /Users/yaoningyu/Desktop/UIUC/GEO && python -m pytest tests/test_calibration_stats.py::TestCliffsDelta -v`
- [ ] Expected: FAIL — `ModuleNotFoundError: No module named 'calibration.stats'`

### Step 1.2: Implement `cliffs_delta`

- [ ] Write:

```python
# calibration/stats.py
"""Statistical functions for calibration evaluation."""
from __future__ import annotations

import numpy as np


def cliffs_delta(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """Cliff's delta effect size.

    Positive means x values tend to be higher than y values.
    Returns a value in [-1, 1].
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) == 0 or len(y_arr) == 0:
        raise ValueError("Both x and y must be non-empty")
    # Count pairwise comparisons
    n_x, n_y = len(x_arr), len(y_arr)
    more = 0
    less = 0
    for xi in x_arr:
        more += np.sum(xi > y_arr)
        less += np.sum(xi < y_arr)
    return float((more - less) / (n_x * n_y))
```

- [ ] Run: `python -m pytest tests/test_calibration_stats.py::TestCliffsDelta -v`
- [ ] Expected: All PASS

### Step 1.3: Write tests for `empirical_p_value` and `empirical_percentile`

- [ ] Append to `tests/test_calibration_stats.py`:

```python
from calibration.stats import empirical_p_value, empirical_percentile


class TestEmpiricalPValue:
    def test_value_at_median_has_high_p(self):
        real = [1.0, 2.0, 3.0, 4.0, 5.0]
        p = empirical_p_value(real, 3.0)  # exactly at median
        assert p > 0.5

    def test_extreme_value_has_low_p(self):
        real = [1.0, 2.0, 3.0, 4.0, 5.0]
        p = empirical_p_value(real, 100.0)
        assert p < 0.2

    def test_p_value_between_0_and_1(self):
        real = [1.0, 2.0, 3.0, 4.0, 5.0]
        for gen in [0.0, 3.0, 6.0, 100.0]:
            p = empirical_p_value(real, gen)
            assert 0.0 < p <= 1.0

    def test_formula_matches_spec(self):
        """Verify: center=median, gen_dist=|gen-center|, p=(count(real_dist>=gen_dist)+1)/(n+1)."""
        real = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gen_value = 5.5
        center = np.median(real)  # 3.0
        gen_dist = abs(gen_value - center)  # 2.5
        real_dist = np.abs(real - center)  # [2, 1, 0, 1, 2]
        expected_p = (np.sum(real_dist >= gen_dist) + 1) / (len(real) + 1)
        assert empirical_p_value(real, gen_value) == pytest.approx(expected_p)


class TestEmpiricalPercentile:
    def test_below_all(self):
        real = [10.0, 20.0, 30.0]
        pct = empirical_percentile(real, 1.0)
        assert pct == pytest.approx(0.0)

    def test_above_all(self):
        real = [10.0, 20.0, 30.0]
        pct = empirical_percentile(real, 100.0)
        assert pct == pytest.approx(100.0)

    def test_at_median(self):
        real = [1.0, 2.0, 3.0, 4.0, 5.0]
        pct = empirical_percentile(real, 3.0)
        assert 30.0 <= pct <= 60.0
```

- [ ] Run: `python -m pytest tests/test_calibration_stats.py -v -k "PValue or Percentile"`
- [ ] Expected: FAIL — `ImportError`

### Step 1.4: Implement `empirical_p_value` and `empirical_percentile`

- [ ] Add to `calibration/stats.py`:

```python
def empirical_p_value(
    real_values: list[float] | np.ndarray, gen_value: float
) -> float:
    """Two-sided empirical p-value: how far is gen_value from the real median?

    center = median(real_values)
    gen_dist = abs(gen_value - center)
    real_dist = abs(real_values - center)
    p = (count(real_dist >= gen_dist) + 1) / (len(real_values) + 1)
    """
    real = np.asarray(real_values, dtype=float)
    real = real[~np.isnan(real)]
    if len(real) == 0:
        return 1.0
    center = float(np.median(real))
    gen_dist = abs(gen_value - center)
    real_dist = np.abs(real - center)
    p = float((np.sum(real_dist >= gen_dist) + 1) / (len(real) + 1))
    return p


def empirical_percentile(
    real_values: list[float] | np.ndarray, gen_value: float
) -> float:
    """Percentile of gen_value within the real distribution (0-100)."""
    real = np.asarray(real_values, dtype=float)
    real = real[~np.isnan(real)]
    if len(real) == 0:
        return 50.0
    return float(np.sum(real <= gen_value) / len(real) * 100.0)
```

- [ ] Run: `python -m pytest tests/test_calibration_stats.py -v -k "PValue or Percentile"`
- [ ] Expected: All PASS

### Step 1.5: Write tests for `evaluate_group_vs_real`

- [ ] Append to `tests/test_calibration_stats.py`:

```python
import pandas as pd

from calibration.stats import evaluate_group_vs_real


class TestEvaluateGroupVsReal:
    def _make_dfs(self):
        rng = np.random.default_rng(42)
        real_df = pd.DataFrame({
            "self_bleu_2": rng.normal(0.05, 0.02, 50),
            "toxicity_mean": rng.normal(0.003, 0.001, 50),
        })
        gen_df = pd.DataFrame({
            "self_bleu_2": rng.normal(0.20, 0.03, 50),  # much higher
            "toxicity_mean": rng.normal(0.003, 0.001, 50),  # similar
        })
        return real_df, gen_df

    def test_returns_dict_per_metric(self):
        real_df, gen_df = self._make_dfs()
        metrics = ["self_bleu_2", "toxicity_mean"]
        result = evaluate_group_vs_real(real_df, gen_df, metrics)
        assert set(result.keys()) == set(metrics)

    def test_has_required_fields(self):
        real_df, gen_df = self._make_dfs()
        result = evaluate_group_vs_real(real_df, gen_df, ["self_bleu_2"])
        entry = result["self_bleu_2"]
        required = [
            "real_mean", "real_median", "generated_mean", "generated_median",
            "mwu_p_value", "ks_statistic", "ks_p_value", "cliffs_delta",
            "direction", "empirical_fail_rate",
        ]
        for field in required:
            assert field in entry, f"Missing field: {field}"

    def test_divergent_metric_detected(self):
        real_df, gen_df = self._make_dfs()
        result = evaluate_group_vs_real(real_df, gen_df, ["self_bleu_2"])
        entry = result["self_bleu_2"]
        assert entry["mwu_p_value"] < 0.05
        assert entry["cliffs_delta"] > 0  # generated higher
        assert entry["direction"] == "generated_higher"

    def test_similar_metric_not_flagged(self):
        real_df, gen_df = self._make_dfs()
        result = evaluate_group_vs_real(real_df, gen_df, ["toxicity_mean"])
        entry = result["toxicity_mean"]
        assert entry["mwu_p_value"] > 0.05 or abs(entry["cliffs_delta"]) < 0.3
```

- [ ] Run: `python -m pytest tests/test_calibration_stats.py::TestEvaluateGroupVsReal -v`
- [ ] Expected: FAIL — `ImportError`

### Step 1.6: Implement `evaluate_group_vs_real`

- [ ] Add to `calibration/stats.py`:

```python
from scipy import stats as sp_stats


def evaluate_group_vs_real(
    real_df: "pd.DataFrame",
    gen_df: "pd.DataFrame",
    metrics: list[str],
    alpha: float = 0.05,
) -> dict[str, dict]:
    """Group-level evaluation: generated vs real discussions.

    For each metric, computes MWU, KS, Cliff's delta, direction, and
    empirical fail rate.
    """
    import pandas as pd

    results: dict[str, dict] = {}
    for metric in metrics:
        real_vals = real_df[metric].dropna().values.astype(float)
        gen_vals = gen_df[metric].dropna().values.astype(float)
        if len(real_vals) == 0 or len(gen_vals) == 0:
            continue

        real_mean = float(np.mean(real_vals))
        real_median = float(np.median(real_vals))
        gen_mean = float(np.mean(gen_vals))
        gen_median = float(np.median(gen_vals))

        # Mann-Whitney U
        mwu_stat, mwu_p = sp_stats.mannwhitneyu(
            gen_vals, real_vals, alternative="two-sided"
        )

        # Kolmogorov-Smirnov
        ks_stat, ks_p = sp_stats.ks_2samp(gen_vals, real_vals)

        # Cliff's delta (generated vs real)
        delta = cliffs_delta(gen_vals.tolist(), real_vals.tolist())

        # Direction
        if abs(delta) < 0.05:
            direction = "similar"
        elif delta > 0:
            direction = "generated_higher"
        else:
            direction = "generated_lower"

        # Empirical fail rate
        fail_count = sum(
            1 for gv in gen_vals if empirical_p_value(real_vals, gv) < alpha
        )
        fail_rate = fail_count / len(gen_vals)

        results[metric] = {
            "real_mean": real_mean,
            "real_median": real_median,
            "generated_mean": gen_mean,
            "generated_median": gen_median,
            "mwu_p_value": float(mwu_p),
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p),
            "cliffs_delta": delta,
            "direction": direction,
            "empirical_fail_rate": fail_rate,
        }
    return results
```

- [ ] Run: `python -m pytest tests/test_calibration_stats.py::TestEvaluateGroupVsReal -v`
- [ ] Expected: All PASS

### Step 1.7: Write tests for `diagnose_single_generation`

- [ ] Append to `tests/test_calibration_stats.py`:

```python
from calibration.stats import diagnose_single_generation


class TestDiagnoseSingleGeneration:
    def test_returns_dict_per_metric(self):
        real_df = pd.DataFrame({"m1": np.random.default_rng(0).normal(5, 1, 50)})
        gen_row = {"m1": 5.0}
        result = diagnose_single_generation(real_df, gen_row, ["m1"])
        assert "m1" in result

    def test_required_fields(self):
        real_df = pd.DataFrame({"m1": np.random.default_rng(0).normal(5, 1, 50)})
        gen_row = {"m1": 5.0}
        result = diagnose_single_generation(real_df, gen_row, ["m1"])
        entry = result["m1"]
        for field in ["real_median", "generated_value", "empirical_p_value",
                       "percentile", "direction", "diagnosis_flag"]:
            assert field in entry

    def test_extreme_value_fails(self):
        real_df = pd.DataFrame({"m1": [1.0, 2.0, 3.0, 4.0, 5.0] * 10})
        gen_row = {"m1": 100.0}
        result = diagnose_single_generation(real_df, gen_row, ["m1"])
        assert result["m1"]["diagnosis_flag"] == "fail"
        assert result["m1"]["direction"] == "too_high"

    def test_median_value_passes(self):
        real_df = pd.DataFrame({"m1": [1.0, 2.0, 3.0, 4.0, 5.0] * 10})
        gen_row = {"m1": 3.0}
        result = diagnose_single_generation(real_df, gen_row, ["m1"])
        assert result["m1"]["diagnosis_flag"] == "pass"
        assert result["m1"]["direction"] == "within_baseline"
```

- [ ] Run: `python -m pytest tests/test_calibration_stats.py::TestDiagnoseSingleGeneration -v`
- [ ] Expected: FAIL

### Step 1.8: Implement `diagnose_single_generation`

- [ ] Add to `calibration/stats.py`:

```python
def diagnose_single_generation(
    real_df: "pd.DataFrame",
    gen_row: dict[str, float],
    metrics: list[str],
    alpha: float = 0.05,
) -> dict[str, dict]:
    """Per-instance calibration diagnostic for one generated thread."""
    results: dict[str, dict] = {}
    for metric in metrics:
        real_vals = real_df[metric].dropna().values.astype(float)
        gen_value = float(gen_row[metric])
        if len(real_vals) == 0:
            continue

        real_med = float(np.median(real_vals))
        p = empirical_p_value(real_vals, gen_value)
        pct = empirical_percentile(real_vals, gen_value)

        if p >= alpha:
            direction = "within_baseline"
        elif gen_value > real_med:
            direction = "too_high"
        else:
            direction = "too_low"

        results[metric] = {
            "real_median": real_med,
            "generated_value": gen_value,
            "empirical_p_value": p,
            "percentile": pct,
            "direction": direction,
            "diagnosis_flag": "fail" if p < alpha else "pass",
        }
    return results
```

- [ ] Run: `python -m pytest tests/test_calibration_stats.py::TestDiagnoseSingleGeneration -v`
- [ ] Expected: All PASS

### Step 1.9: Write tests for `compare_before_after`

- [ ] Append to `tests/test_calibration_stats.py`:

```python
from calibration.stats import compare_before_after


class TestCompareBeforeAfter:
    def test_returns_per_metric_and_summary(self):
        before = {
            "m1": {"mwu_p_value": 0.01, "ks_p_value": 0.02, "cliffs_delta": 0.5,
                    "empirical_fail_rate": 0.6},
        }
        after = {
            "m1": {"mwu_p_value": 0.3, "ks_p_value": 0.4, "cliffs_delta": 0.1,
                    "empirical_fail_rate": 0.1},
        }
        result = compare_before_after(before, after)
        assert "per_metric" in result
        assert "summary" in result
        assert "m1" in result["per_metric"]

    def test_improvement_detected(self):
        before = {
            "m1": {"mwu_p_value": 0.01, "ks_p_value": 0.02, "cliffs_delta": 0.5,
                    "empirical_fail_rate": 0.6},
        }
        after = {
            "m1": {"mwu_p_value": 0.3, "ks_p_value": 0.4, "cliffs_delta": 0.1,
                    "empirical_fail_rate": 0.1},
        }
        result = compare_before_after(before, after)
        m1 = result["per_metric"]["m1"]
        assert m1["improved"] is True
        assert m1["abs_delta_reduction"] == pytest.approx(0.4)
        assert m1["fail_rate_reduction"] == pytest.approx(0.5)

    def test_regression_not_marked_improved(self):
        before = {
            "m1": {"mwu_p_value": 0.3, "ks_p_value": 0.4, "cliffs_delta": 0.1,
                    "empirical_fail_rate": 0.05},
        }
        after = {
            "m1": {"mwu_p_value": 0.01, "ks_p_value": 0.02, "cliffs_delta": 0.6,
                    "empirical_fail_rate": 0.7},
        }
        result = compare_before_after(before, after)
        assert result["per_metric"]["m1"]["improved"] is False

    def test_summary_counts(self):
        before = {
            "m1": {"mwu_p_value": 0.01, "ks_p_value": 0.02, "cliffs_delta": 0.5,
                    "empirical_fail_rate": 0.6},
            "m2": {"mwu_p_value": 0.5, "ks_p_value": 0.6, "cliffs_delta": 0.05,
                    "empirical_fail_rate": 0.02},
        }
        after = {
            "m1": {"mwu_p_value": 0.3, "ks_p_value": 0.4, "cliffs_delta": 0.1,
                    "empirical_fail_rate": 0.1},
            "m2": {"mwu_p_value": 0.4, "ks_p_value": 0.5, "cliffs_delta": 0.04,
                    "empirical_fail_rate": 0.01},
        }
        result = compare_before_after(before, after)
        s = result["summary"]
        assert s["metrics_sig_different_before"] == 1  # m1
        assert s["metrics_sig_different_after"] == 0
```

- [ ] Run: `python -m pytest tests/test_calibration_stats.py::TestCompareBeforeAfter -v`
- [ ] Expected: FAIL

### Step 1.10: Implement `compare_before_after`

- [ ] Add to `calibration/stats.py`:

```python
def compare_before_after(
    before_results: dict[str, dict],
    after_results: dict[str, dict],
    alpha: float = 0.05,
) -> dict:
    """Compare before-calibration and after-calibration group evaluation results."""
    per_metric: dict[str, dict] = {}
    metrics = sorted(set(before_results.keys()) & set(after_results.keys()))

    for metric in metrics:
        b = before_results[metric]
        a = after_results[metric]
        abs_delta_reduction = abs(b["cliffs_delta"]) - abs(a["cliffs_delta"])
        fail_rate_reduction = b["empirical_fail_rate"] - a["empirical_fail_rate"]
        improved = (
            abs(a["cliffs_delta"]) < abs(b["cliffs_delta"])
            and a["empirical_fail_rate"] < b["empirical_fail_rate"]
        )
        per_metric[metric] = {
            "before_mwu_p": b["mwu_p_value"],
            "after_mwu_p": a["mwu_p_value"],
            "before_ks_p": b.get("ks_p_value"),
            "after_ks_p": a.get("ks_p_value"),
            "before_cliffs_delta": b["cliffs_delta"],
            "after_cliffs_delta": a["cliffs_delta"],
            "abs_delta_reduction": abs_delta_reduction,
            "before_fail_rate": b["empirical_fail_rate"],
            "after_fail_rate": a["empirical_fail_rate"],
            "fail_rate_reduction": fail_rate_reduction,
            "improved": improved,
        }

    sig_before = sum(1 for m in metrics if before_results[m]["mwu_p_value"] < alpha)
    sig_after = sum(1 for m in metrics if after_results[m]["mwu_p_value"] < alpha)
    avg_abs_delta_before = float(np.mean([abs(before_results[m]["cliffs_delta"]) for m in metrics])) if metrics else 0.0
    avg_abs_delta_after = float(np.mean([abs(after_results[m]["cliffs_delta"]) for m in metrics])) if metrics else 0.0
    avg_fail_before = float(np.mean([before_results[m]["empirical_fail_rate"] for m in metrics])) if metrics else 0.0
    avg_fail_after = float(np.mean([after_results[m]["empirical_fail_rate"] for m in metrics])) if metrics else 0.0

    summary = {
        "metrics_sig_different_before": sig_before,
        "metrics_sig_different_after": sig_after,
        "avg_abs_cliffs_delta_before": avg_abs_delta_before,
        "avg_abs_cliffs_delta_after": avg_abs_delta_after,
        "overall_fail_rate_before": avg_fail_before,
        "overall_fail_rate_after": avg_fail_after,
        "overall_pass_rate_before": 1.0 - avg_fail_before,
        "overall_pass_rate_after": 1.0 - avg_fail_after,
    }

    return {"per_metric": per_metric, "summary": summary}
```

- [ ] Run: `python -m pytest tests/test_calibration_stats.py -v`
- [ ] Expected: All PASS

### Step 1.11: Commit

- [ ] Run:
```bash
git add calibration/__init__.py calibration/stats.py tests/test_calibration_stats.py
git commit -m "feat(calibration): add stats module — Cliff's delta, empirical p-value, group eval, single diagnostic, before/after comparison"
```

---

## Task 2: Knob Registry (`registry.py`)

**Files:**
- Create: `calibration/registry.py`
- Create: `tests/test_calibration_registry.py`

### Step 2.1: Write tests

- [ ] Write:

```python
# tests/test_calibration_registry.py
"""Tests for calibration.registry."""
from __future__ import annotations

import pytest

from calibration.registry import KnobRegistry


class TestKnobRegistry:
    def test_loads_default_knobs(self):
        reg = KnobRegistry()
        names = reg.knob_names()
        assert len(names) > 0
        assert "persona.conflict_style_distribution" in names

    def test_all_knobs_have_required_fields(self):
        reg = KnobRegistry()
        for name in reg.knob_names():
            knob = reg.get(name)
            assert "layer" in knob
            assert knob["layer"] in ("persona", "prompt")
            assert "type" in knob
            assert "default" in knob
            assert "description" in knob

    def test_get_defaults(self):
        reg = KnobRegistry()
        defaults = reg.defaults()
        assert isinstance(defaults, dict)
        assert "persona.conflict_style_distribution" in defaults

    def test_validate_valid_overlay(self):
        reg = KnobRegistry()
        overlay = {
            "persona.conflict_style_distribution": {
                "calm": 0.2, "skeptical": 0.2, "blunt": 0.2,
                "sarcastic": 0.15, "argumentative": 0.15, "avoidant": 0.1,
            }
        }
        errors = reg.validate(overlay)
        assert errors == []

    def test_validate_unknown_knob(self):
        reg = KnobRegistry()
        errors = reg.validate({"nonexistent.knob": 42})
        assert len(errors) == 1
        assert "nonexistent.knob" in errors[0]

    def test_validate_bad_distribution_sum(self):
        reg = KnobRegistry()
        overlay = {
            "persona.conflict_style_distribution": {
                "calm": 0.5, "skeptical": 0.5, "blunt": 0.5,
                "sarcastic": 0.0, "argumentative": 0.0, "avoidant": 0.0,
            }
        }
        errors = reg.validate(overlay)
        assert any("sum" in e.lower() for e in errors)

    def test_validate_wrong_type(self):
        reg = KnobRegistry()
        overlay = {"persona.conflict_style_distribution": "not a dict"}
        errors = reg.validate(overlay)
        assert len(errors) >= 1

    def test_persona_knobs_only(self):
        reg = KnobRegistry()
        for name in reg.knob_names():
            knob = reg.get(name)
            assert knob["layer"] in ("persona", "prompt"), (
                f"Knob {name} has layer={knob['layer']}, expected persona or prompt"
            )

    def test_for_llm_context(self):
        reg = KnobRegistry()
        context = reg.for_llm_context()
        assert isinstance(context, str)
        assert "persona.conflict_style_distribution" in context
```

- [ ] Run: `python -m pytest tests/test_calibration_registry.py -v`
- [ ] Expected: FAIL

### Step 2.2: Implement `KnobRegistry`

- [ ] Write:

```python
# calibration/registry.py
"""Knob registry: declares all tunable parameters for calibration."""
from __future__ import annotations

import json
from typing import Any


_KNOB_DEFINITIONS: list[dict[str, Any]] = [
    # ── Persona layer ────────────────────────────────────────────────────
    {
        "name": "persona.conflict_style_distribution",
        "layer": "persona",
        "domain": "toxicity",
        "type": "distribution",
        "keys": ["calm", "skeptical", "blunt", "sarcastic", "argumentative", "avoidant"],
        "default": {"calm": 0.2, "skeptical": 0.25, "blunt": 0.15, "sarcastic": 0.15, "argumentative": 0.1, "avoidant": 0.15},
        "description": "Distribution of conflict styles across generated personas.",
    },
    {
        "name": "persona.primary_motivation_distribution",
        "layer": "persona",
        "domain": "toxicity",
        "type": "distribution",
        "keys": ["helping", "venting", "showing expertise", "correcting people",
                 "defending their own setup", "bargain-hunting", "complaining",
                 "validation-seeking", "joking around"],
        "default": {"helping": 0.15, "venting": 0.1, "showing expertise": 0.15,
                     "correcting people": 0.15, "defending their own setup": 0.1,
                     "bargain-hunting": 0.05, "complaining": 0.1,
                     "validation-seeking": 0.1, "joking around": 0.1},
        "description": "Distribution of primary motivations across personas.",
    },
    {
        "name": "persona.knowledge_style_distribution",
        "layer": "persona",
        "domain": "repetitiveness",
        "type": "distribution",
        "keys": ["beginner", "casual_user", "experienced_owner", "specialist",
                 "overconfident_half_expert"],
        "default": {"beginner": 0.15, "casual_user": 0.25, "experienced_owner": 0.25,
                     "specialist": 0.15, "overconfident_half_expert": 0.2},
        "description": "Distribution of knowledge styles across personas.",
    },
    {
        "name": "persona.stance_distribution",
        "layer": "persona",
        "domain": "toxicity",
        "type": "distribution",
        "keys": ["supportive", "neutral", "observer", "opposing"],
        "default": {"supportive": 0.25, "neutral": 0.4, "observer": 0.2, "opposing": 0.15},
        "description": "Distribution of stances across personas.",
    },
    {
        "name": "persona.sentiment_bias_min",
        "layer": "persona",
        "domain": "toxicity",
        "type": "float",
        "min": -1.0,
        "max": 0.0,
        "default": -0.3,
        "description": "Lower bound for sentiment_bias sampling range.",
    },
    {
        "name": "persona.sentiment_bias_max",
        "layer": "persona",
        "domain": "toxicity",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.4,
        "description": "Upper bound for sentiment_bias sampling range.",
    },
    # ── Prompt layer ─────────────────────────────────────────────────────
    {
        "name": "prompt.anti_paraphrase_instruction",
        "layer": "prompt",
        "domain": "repetitiveness",
        "type": "text",
        "default": "If several visible comments are already making the same point, your default should be to add a different lens, a different reason, or a direct challenge. Pure paraphrase should be relatively rare.",
        "description": "Instruction injected into action prompt to reduce comment repetitiveness.",
    },
    {
        "name": "prompt.tone_guidance",
        "layer": "prompt",
        "domain": "toxicity",
        "type": "text",
        "default": "Real threads mix tones. Some replies are calm and practical, some are firsthand datapoints, some are dismissive or sarcastic, and some are mildly aggressive.",
        "description": "Guidance for comment tone and aggression level.",
    },
    {
        "name": "prompt.structure_preference_weight",
        "layer": "prompt",
        "domain": "structure",
        "type": "text",
        "default": "Balance standalone answers and back-and-forth. If a thread only has one or two top-level takes, another top-level comment can be natural. If it already has several standalone takes but little interaction, reply to a visible comment instead.",
        "description": "Guidance for thread structure: top-level vs reply preference.",
    },
    {
        "name": "prompt.depth_soft_cap_instruction",
        "layer": "prompt",
        "domain": "structure",
        "type": "text",
        "default": "Once a visible chain is already around depth 3, prefer branching out or making a fresh top-level comment unless you are directly continuing that exact sub-argument.",
        "description": "Instruction for soft depth limit on reply chains.",
    },
    {
        "name": "prompt.few_shot_style_anchor",
        "layer": "prompt",
        "domain": "repetitiveness",
        "type": "text",
        "default": "",
        "description": "Optional style anchor text injected alongside few-shot examples.",
    },
    {
        "name": "prompt.consensus_handling",
        "layer": "prompt",
        "domain": "repetitiveness",
        "type": "text",
        "default": "Treat repeated consensus as a warning sign. If three visible comments already say some version of the same thing, a fourth version should usually be very short or should say something meaningfully different.",
        "description": "Instruction for handling consensus/repeated viewpoints.",
    },
]


class KnobRegistry:
    """Declares and validates all tunable calibration parameters."""

    def __init__(self) -> None:
        self._knobs: dict[str, dict[str, Any]] = {
            knob["name"]: knob for knob in _KNOB_DEFINITIONS
        }

    def knob_names(self) -> list[str]:
        return list(self._knobs.keys())

    def get(self, name: str) -> dict[str, Any]:
        return self._knobs[name]

    def defaults(self) -> dict[str, Any]:
        return {name: knob["default"] for name, knob in self._knobs.items()}

    def validate(self, overlay: dict[str, Any]) -> list[str]:
        """Validate an overlay dict against the registry. Returns error messages."""
        errors: list[str] = []
        for name, value in overlay.items():
            if name not in self._knobs:
                errors.append(f"Unknown knob: {name}")
                continue
            knob = self._knobs[name]
            knob_type = knob["type"]
            if knob_type == "distribution":
                if not isinstance(value, dict):
                    errors.append(f"{name}: expected dict, got {type(value).__name__}")
                    continue
                total = sum(value.values())
                if abs(total - 1.0) > 0.01:
                    errors.append(f"{name}: distribution values sum to {total}, expected 1.0")
                for v in value.values():
                    if v < 0:
                        errors.append(f"{name}: distribution values must be >= 0")
                        break
            elif knob_type == "float":
                if not isinstance(value, (int, float)):
                    errors.append(f"{name}: expected float, got {type(value).__name__}")
                    continue
                if "min" in knob and value < knob["min"]:
                    errors.append(f"{name}: {value} < min {knob['min']}")
                if "max" in knob and value > knob["max"]:
                    errors.append(f"{name}: {value} > max {knob['max']}")
            elif knob_type == "text":
                if not isinstance(value, str):
                    errors.append(f"{name}: expected str, got {type(value).__name__}")
        return errors

    def for_llm_context(self) -> str:
        """Render the registry as a text block for the calibration LLM prompt."""
        lines: list[str] = ["# Tunable Knobs\n"]
        for name, knob in self._knobs.items():
            lines.append(f"## {name}")
            lines.append(f"Layer: {knob['layer']}")
            lines.append(f"Domain: {knob.get('domain', 'general')}")
            lines.append(f"Type: {knob['type']}")
            lines.append(f"Description: {knob['description']}")
            if knob["type"] == "distribution":
                lines.append(f"Keys: {knob['keys']}")
            elif knob["type"] == "float":
                lines.append(f"Range: [{knob.get('min', '-inf')}, {knob.get('max', 'inf')}]")
            lines.append(f"Default: {json.dumps(knob['default'])}")
            lines.append("")
        return "\n".join(lines)
```

- [ ] Run: `python -m pytest tests/test_calibration_registry.py -v`
- [ ] Expected: All PASS

### Step 2.3: Commit

- [ ] Run:
```bash
git add calibration/registry.py tests/test_calibration_registry.py
git commit -m "feat(calibration): add knob registry — persona + prompt layer definitions with validation"
```

---

## Task 3: Overlay System (`overlay.py`)

**Files:**
- Create: `calibration/overlay.py`
- Create: `tests/test_calibration_overlay.py`

### Step 3.1: Write tests

- [ ] Write:

```python
# tests/test_calibration_overlay.py
"""Tests for calibration.overlay."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from calibration.overlay import load_overlay, save_overlay, merge_overlay, diff_overlay
from calibration.registry import KnobRegistry


class TestOverlay:
    def test_merge_with_empty_overlay(self):
        reg = KnobRegistry()
        defaults = reg.defaults()
        merged = merge_overlay(defaults, {})
        assert merged == defaults

    def test_merge_overrides_single_knob(self):
        reg = KnobRegistry()
        defaults = reg.defaults()
        overlay = {"persona.sentiment_bias_min": -0.5}
        merged = merge_overlay(defaults, overlay)
        assert merged["persona.sentiment_bias_min"] == -0.5
        # Other knobs unchanged
        assert merged["persona.sentiment_bias_max"] == defaults["persona.sentiment_bias_max"]

    def test_save_and_load_roundtrip(self, tmp_path):
        overlay = {"persona.sentiment_bias_min": -0.5, "prompt.tone_guidance": "Be blunt."}
        path = tmp_path / "overlay.json"
        save_overlay(overlay, path)
        loaded = load_overlay(path)
        assert loaded == overlay

    def test_load_nonexistent_returns_empty(self, tmp_path):
        loaded = load_overlay(tmp_path / "nope.json")
        assert loaded == {}

    def test_diff_no_changes(self):
        a = {"persona.sentiment_bias_min": -0.3}
        b = {"persona.sentiment_bias_min": -0.3}
        changes = diff_overlay(a, b)
        assert changes == {}

    def test_diff_detects_changes(self):
        a = {"persona.sentiment_bias_min": -0.3, "prompt.tone_guidance": "old"}
        b = {"persona.sentiment_bias_min": -0.5, "prompt.tone_guidance": "old"}
        changes = diff_overlay(a, b)
        assert "persona.sentiment_bias_min" in changes
        assert "prompt.tone_guidance" not in changes

    def test_diff_detects_additions(self):
        a = {}
        b = {"prompt.tone_guidance": "new"}
        changes = diff_overlay(a, b)
        assert "prompt.tone_guidance" in changes
```

- [ ] Run: `python -m pytest tests/test_calibration_overlay.py -v`
- [ ] Expected: FAIL

### Step 3.2: Implement overlay functions

- [ ] Write:

```python
# calibration/overlay.py
"""Overlay model: load, save, merge, validate, diff calibration overlays."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_overlay(path: Path) -> dict[str, Any]:
    """Load an overlay from a JSON file. Returns empty dict if not found."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_overlay(overlay: dict[str, Any], path: Path) -> None:
    """Save an overlay to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overlay, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_overlay(defaults: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge overlay on top of defaults. Overlay values take precedence."""
    merged = dict(defaults)
    merged.update(overlay)
    return merged


def diff_overlay(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Return knobs that differ between a and b. Values come from b."""
    changes: dict[str, Any] = {}
    all_keys = set(a.keys()) | set(b.keys())
    for key in all_keys:
        val_a = a.get(key)
        val_b = b.get(key)
        if val_a != val_b and val_b is not None:
            changes[key] = val_b
    return changes
```

- [ ] Run: `python -m pytest tests/test_calibration_overlay.py -v`
- [ ] Expected: All PASS

### Step 3.3: Commit

- [ ] Run:
```bash
git add calibration/overlay.py tests/test_calibration_overlay.py
git commit -m "feat(calibration): add overlay system — load/save/merge/diff"
```

---

## Task 4: Calibration Log (`log.py`)

**Files:**
- Create: `calibration/log.py`
- Create: `tests/test_calibration_log.py`

### Step 4.1: Write tests

- [ ] Write:

```python
# tests/test_calibration_log.py
"""Tests for calibration.log."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from calibration.log import CalibrationLog


class TestCalibrationLog:
    def test_new_log_is_empty(self, tmp_path):
        log = CalibrationLog(tmp_path / "log.json")
        assert log.entries() == []
        assert log.failed_strategies() == []

    def test_append_and_read(self, tmp_path):
        log = CalibrationLog(tmp_path / "log.json")
        entry = {
            "iteration": 0,
            "strategy_label": "baseline",
            "selection": {"beat_current_best": True},
        }
        log.append(entry)
        assert len(log.entries()) == 1
        assert log.entries()[0]["strategy_label"] == "baseline"

    def test_persistence(self, tmp_path):
        path = tmp_path / "log.json"
        log1 = CalibrationLog(path)
        log1.append({"iteration": 0, "strategy_label": "a", "selection": {"beat_current_best": True}})
        log1.append({"iteration": 1, "strategy_label": "b", "selection": {"beat_current_best": False}})

        log2 = CalibrationLog(path)
        assert len(log2.entries()) == 2

    def test_failed_strategies(self, tmp_path):
        log = CalibrationLog(tmp_path / "log.json")
        log.append({"iteration": 0, "strategy_label": "a", "selection": {"beat_current_best": True}})
        log.append({"iteration": 1, "strategy_label": "b", "selection": {"beat_current_best": False}})
        log.append({"iteration": 2, "strategy_label": "c", "selection": {"beat_current_best": False}})
        assert log.failed_strategies() == ["b", "c"]

    def test_trajectory(self, tmp_path):
        log = CalibrationLog(tmp_path / "log.json")
        log.append({
            "iteration": 0, "strategy_label": "a",
            "selection": {"beat_current_best": True},
            "best_fail_rate": 0.35,
        })
        log.append({
            "iteration": 1, "strategy_label": "b",
            "selection": {"beat_current_best": True},
            "best_fail_rate": 0.20,
        })
        traj = log.trajectory()
        assert len(traj) == 2
        assert traj[0]["best_fail_rate"] == 0.35
```

- [ ] Run: `python -m pytest tests/test_calibration_log.py -v`
- [ ] Expected: FAIL

### Step 4.2: Implement `CalibrationLog`

- [ ] Write:

```python
# calibration/log.py
"""Calibration log: per-iteration records with strategy tracking."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CalibrationLog:
    """Append-only log of calibration iteration records."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: list[dict[str, Any]] = []
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._entries = data.get("entries", [])

    def append(self, entry: dict[str, Any]) -> None:
        self._entries.append(entry)
        self._save()

    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def failed_strategies(self) -> list[str]:
        return [
            e["strategy_label"]
            for e in self._entries
            if not e.get("selection", {}).get("beat_current_best", True)
            and e.get("strategy_label")
        ]

    def trajectory(self) -> list[dict[str, Any]]:
        return [
            {k: v for k, v in e.items() if k != "candidates"}
            for e in self._entries
        ]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps({"entries": self._entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
```

- [ ] Run: `python -m pytest tests/test_calibration_log.py -v`
- [ ] Expected: All PASS

### Step 4.3: Commit

- [ ] Run:
```bash
git add calibration/log.py tests/test_calibration_log.py
git commit -m "feat(calibration): add calibration log — append-only iteration records with strategy tracking"
```

---

## Task 5: Runner (`runner.py`)

**Files:**
- Create: `calibration/runner.py`
- Create: `tests/test_calibration_runner.py`

### Step 5.1: Write tests

- [ ] Write:

```python
# tests/test_calibration_runner.py
"""Tests for calibration.runner."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from calibration.runner import run_candidates


class TestRunCandidates:
    def test_creates_candidate_dirs(self, tmp_path):
        overlays = [{"persona.sentiment_bias_min": -0.3}] * 2
        iter_dir = tmp_path / "iter_00"

        # Mock subprocess.run to simulate run_discussion.py creating output
        def fake_run(cmd, **kwargs):
            # Find --output-dir in cmd and create a dummy dir with metrics
            for i, arg in enumerate(cmd):
                if arg == "--output-dir":
                    out_dir = Path(cmd[i + 1])
                    sim_dir = out_dir / "credit_cards_fake"
                    sim_dir.mkdir(parents=True, exist_ok=True)
                    (sim_dir / "run_config.json").write_text("{}")
                    break
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("calibration.runner.subprocess.run", side_effect=fake_run):
            results = run_candidates(
                overlays=overlays,
                iter_dir=iter_dir,
                reference_run_config={
                    "input_file": "data/test.json",
                    "agents": 5, "hours": 1, "rounds": 1,
                    "seed_posts": 1, "seed": 42,
                },
                parallel=1,
                python=sys.executable,
                repo_root=tmp_path,
            )
        assert len(results) == 2
        for r in results:
            assert r["candidate_dir"].exists()
            assert (r["candidate_dir"] / "overlay.json").exists()

    def test_failed_simulation_marked(self, tmp_path):
        overlays = [{}]
        iter_dir = tmp_path / "iter_00"

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            return result

        with patch("calibration.runner.subprocess.run", side_effect=fake_run):
            results = run_candidates(
                overlays=overlays,
                iter_dir=iter_dir,
                reference_run_config={
                    "input_file": "data/test.json",
                    "agents": 5, "hours": 1, "rounds": 1,
                    "seed_posts": 1, "seed": 42,
                },
                parallel=1,
                python=sys.executable,
                repo_root=tmp_path,
            )
        assert results[0]["success"] is False
```

- [ ] Run: `python -m pytest tests/test_calibration_runner.py -v`
- [ ] Expected: FAIL

### Step 5.2: Implement `run_candidates`

- [ ] Write:

```python
# calibration/runner.py
"""Subprocess pool for running candidate simulations."""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .overlay import save_overlay


def run_candidates(
    overlays: list[dict[str, Any]],
    iter_dir: Path,
    reference_run_config: dict[str, Any],
    parallel: int = 1,
    python: str = sys.executable,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Run candidate simulations and return results.

    Each candidate gets its own subdirectory under iter_dir/candidates/.
    """
    candidates_dir = iter_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    for idx, overlay in enumerate(overlays):
        candidate_dir = candidates_dir / f"candidate_{idx}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = candidate_dir / "overlay.json"
        save_overlay(overlay, overlay_path)
        tasks.append({
            "candidate_id": idx,
            "candidate_dir": candidate_dir,
            "overlay_path": overlay_path,
            "overlay": overlay,
        })

    results: list[dict[str, Any]] = []
    if parallel <= 1:
        for task in tasks:
            result = _run_one(task, reference_run_config, python, repo_root)
            results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=min(parallel, len(tasks))) as pool:
            futures = {
                pool.submit(_run_one, task, reference_run_config, python, repo_root): task
                for task in tasks
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda r: r["candidate_id"])

    return results


def _run_one(
    task: dict[str, Any],
    reference_run_config: dict[str, Any],
    python: str,
    repo_root: Path | None,
) -> dict[str, Any]:
    """Run one candidate simulation as a subprocess."""
    candidate_dir = task["candidate_dir"]
    overlay_path = task["overlay_path"]
    sim_output_dir = candidate_dir / "sim_output"
    sim_output_dir.mkdir(parents=True, exist_ok=True)

    run_discussion = "run_discussion.py"
    if repo_root:
        run_discussion = str(repo_root / "run_discussion.py")

    input_file = reference_run_config["input_file"]
    if repo_root:
        input_file = str(repo_root / input_file)

    cmd = [
        python, run_discussion, input_file,
        "--agents", str(reference_run_config.get("agents", 30)),
        "--hours", str(reference_run_config.get("hours", 24)),
        "--rounds", str(reference_run_config.get("rounds", 24)),
        "--seed-posts", str(reference_run_config.get("seed_posts", 1)),
        "--seed", str(reference_run_config.get("seed", 42)),
        "--output-dir", str(sim_output_dir),
        "--overlay", str(overlay_path),
    ]

    hint = reference_run_config.get("hint")
    if hint:
        cmd.extend(["--hint", str(hint)])
    few_shot_source = reference_run_config.get("few_shot_source")
    if few_shot_source:
        cmd.extend(["--few-shot-source", str(few_shot_source)])
    few_shot_count = reference_run_config.get("few_shot_count")
    if few_shot_count is not None:
        cmd.extend(["--few-shot-count", str(few_shot_count)])
    few_shot_comments = reference_run_config.get("few_shot_comments")
    if few_shot_comments is not None:
        cmd.extend(["--few-shot-comments", str(few_shot_comments)])

    cwd = str(repo_root) if repo_root else None
    print(f"[candidate {task['candidate_id']}] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    # Find the created simulation directory inside sim_output
    sim_dir = None
    if sim_output_dir.exists():
        subdirs = [d for d in sim_output_dir.iterdir() if d.is_dir()]
        if subdirs:
            subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            sim_dir = subdirs[0]

    success = proc.returncode == 0 and sim_dir is not None
    stdout_path = candidate_dir / "stdout.log"
    stderr_path = candidate_dir / "stderr.log"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    return {
        "candidate_id": task["candidate_id"],
        "candidate_dir": candidate_dir,
        "sim_dir": sim_dir,
        "success": success,
        "returncode": proc.returncode,
    }
```

- [ ] Run: `python -m pytest tests/test_calibration_runner.py -v`
- [ ] Expected: All PASS

### Step 5.3: Commit

- [ ] Run:
```bash
git add calibration/runner.py tests/test_calibration_runner.py
git commit -m "feat(calibration): add runner — subprocess pool for candidate simulations"
```

---

## Task 6: Scorer (`scorer.py`)

**Files:**
- Create: `calibration/scorer.py`
- Create: `tests/test_calibration_scorer.py`

### Step 6.1: Write tests

- [ ] Write:

```python
# tests/test_calibration_scorer.py
"""Tests for calibration.scorer."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibration.scorer import (
    load_thread_metrics,
    compute_real_baseline,
    score_candidate,
    select_best_candidate,
    DEFAULT_METRICS,
)


class TestLoadThreadMetrics:
    def test_reads_csv(self, tmp_path):
        csv_path = tmp_path / "thread_metrics_summary.csv"
        csv_path.write_text(
            "thread_id,comment_count,self_bleu_2,toxicity_mean\n"
            "1,10,0.15,0.002\n"
            "2,8,0.20,0.001\n",
            encoding="utf-8",
        )
        df = load_thread_metrics(tmp_path)
        assert len(df) == 2
        assert "self_bleu_2" in df.columns

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_thread_metrics(tmp_path)


class TestComputeRealBaseline:
    def test_returns_medians_and_values(self, tmp_path):
        csv_path = tmp_path / "thread_metrics_summary.csv"
        csv_path.write_text(
            "thread_id,self_bleu_2,toxicity_mean\n"
            "a,0.05,0.003\n"
            "b,0.04,0.004\n"
            "c,0.06,0.002\n",
            encoding="utf-8",
        )
        baseline = compute_real_baseline(tmp_path, ["self_bleu_2", "toxicity_mean"])
        assert "self_bleu_2" in baseline
        assert "median" in baseline["self_bleu_2"]
        assert "values" in baseline["self_bleu_2"]
        assert len(baseline["self_bleu_2"]["values"]) == 3


class TestScoreCandidate:
    def test_returns_diagnostic(self, tmp_path):
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        (real_dir / "thread_metrics_summary.csv").write_text(
            "thread_id,self_bleu_2\n" + "\n".join(f"{i},{0.05 + i*0.001}" for i in range(50)),
            encoding="utf-8",
        )
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        (sim_dir / "thread_metrics_summary.csv").write_text(
            "thread_id,self_bleu_2\n1,0.06\n",
            encoding="utf-8",
        )
        baseline = compute_real_baseline(real_dir, ["self_bleu_2"])
        result = score_candidate(sim_dir, baseline, ["self_bleu_2"])
        assert "fail_rate" in result
        assert "per_metric" in result
        assert "self_bleu_2" in result["per_metric"]


class TestSelectBestCandidate:
    def test_selects_lowest_fail_rate(self):
        candidates = [
            {"candidate_id": 0, "fail_rate": 0.3, "mean_abs_delta": 0.2},
            {"candidate_id": 1, "fail_rate": 0.1, "mean_abs_delta": 0.3},
            {"candidate_id": 2, "fail_rate": 0.2, "mean_abs_delta": 0.1},
        ]
        best = select_best_candidate(candidates)
        assert best["candidate_id"] == 1

    def test_breaks_tie_with_delta(self):
        candidates = [
            {"candidate_id": 0, "fail_rate": 0.1, "mean_abs_delta": 0.5},
            {"candidate_id": 1, "fail_rate": 0.1, "mean_abs_delta": 0.2},
        ]
        best = select_best_candidate(candidates)
        assert best["candidate_id"] == 1
```

- [ ] Run: `python -m pytest tests/test_calibration_scorer.py -v`
- [ ] Expected: FAIL

### Step 6.2: Implement scorer

- [ ] Write:

```python
# calibration/scorer.py
"""Scoring: run metric suite, read CSVs, compute diagnostics."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .stats import cliffs_delta, diagnose_single_generation, empirical_p_value

DEFAULT_METRICS: list[str] = [
    # Repetitiveness
    "self_bleu_2", "self_bleu_3", "self_bleu_4",
    "self_bertscore_mean_f1", "semantic_mean_cosine",
    # Toxicity
    "toxicity_mean", "toxicity_max", "toxicity_p90",
    "severe_toxicity_mean", "severe_toxicity_max", "severe_toxicity_p90",
    "obscene_mean", "obscene_max", "obscene_p90",
    "threat_mean", "threat_max", "threat_p90",
    "insult_mean", "insult_max", "insult_p90",
    "identity_attack_mean", "identity_attack_max", "identity_attack_p90",
    "aggression_score_mean", "aggression_score_max",
    # Structure
    "length_std", "length_iqr", "length_cv",
    "max_depth", "avg_depth",
    "avg_branching_factor", "structural_virality",
]


def load_thread_metrics(directory: Path) -> pd.DataFrame:
    """Load thread_metrics_summary.csv from a directory."""
    csv_path = directory / "thread_metrics_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No thread_metrics_summary.csv in {directory}")
    return pd.read_csv(csv_path)


def compute_real_baseline(
    real_dir: Path, metrics: list[str]
) -> dict[str, dict[str, Any]]:
    """Compute real baseline: per-metric median and raw values."""
    df = load_thread_metrics(real_dir)
    baseline: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        if metric not in df.columns:
            continue
        values = df[metric].dropna().values.astype(float)
        baseline[metric] = {
            "median": float(np.median(values)) if len(values) > 0 else 0.0,
            "mean": float(np.mean(values)) if len(values) > 0 else 0.0,
            "values": values.tolist(),
        }
    return baseline


def score_candidate(
    sim_dir: Path,
    real_baseline: dict[str, dict[str, Any]],
    metrics: list[str],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Score one candidate simulation against the real baseline.

    Returns fail_rate, mean_abs_delta, and per-metric diagnostics.
    """
    df = load_thread_metrics(sim_dir)

    total_tests = 0
    total_fails = 0
    per_metric: dict[str, dict] = {}
    deltas: list[float] = []

    for metric in metrics:
        if metric not in df.columns or metric not in real_baseline:
            continue
        real_vals = np.array(real_baseline[metric]["values"])
        gen_vals = df[metric].dropna().values.astype(float)
        real_med = real_baseline[metric]["median"]

        if len(gen_vals) == 0 or len(real_vals) == 0:
            continue

        # Per-thread empirical p-values
        thread_diagnostics = []
        for gv in gen_vals:
            p = empirical_p_value(real_vals, float(gv))
            total_tests += 1
            if p < alpha:
                total_fails += 1
            pct = float(np.sum(real_vals <= gv) / len(real_vals) * 100.0)
            if p >= alpha:
                direction = "within_baseline"
            elif gv > real_med:
                direction = "too_high"
            else:
                direction = "too_low"
            thread_diagnostics.append({
                "value": float(gv),
                "empirical_p": p,
                "percentile": pct,
                "direction": direction,
                "pass": p >= alpha,
            })

        # Metric-level aggregates
        delta = cliffs_delta(gen_vals.tolist(), real_vals.tolist())
        deltas.append(abs(delta))
        metric_fail_rate = sum(1 for d in thread_diagnostics if not d["pass"]) / len(thread_diagnostics)

        gen_med = float(np.median(gen_vals))
        per_metric[metric] = {
            "real_median": real_med,
            "generated_median": gen_med,
            "cliffs_delta": delta,
            "fail_rate": metric_fail_rate,
            "direction": "too_high" if gen_med > real_med else "too_low" if gen_med < real_med else "similar",
            "threads": thread_diagnostics,
        }

    fail_rate = total_fails / total_tests if total_tests > 0 else 1.0
    mean_abs_delta = float(np.mean(deltas)) if deltas else 1.0

    return {
        "fail_rate": fail_rate,
        "mean_abs_delta": mean_abs_delta,
        "per_metric": per_metric,
    }


def select_best_candidate(
    scored_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the candidate with lowest fail rate, breaking ties by delta."""
    return min(
        scored_candidates,
        key=lambda c: (c["fail_rate"], c["mean_abs_delta"]),
    )
```

- [ ] Run: `python -m pytest tests/test_calibration_scorer.py -v`
- [ ] Expected: All PASS

### Step 6.3: Commit

- [ ] Run:
```bash
git add calibration/scorer.py tests/test_calibration_scorer.py
git commit -m "feat(calibration): add scorer — metric loading, real baseline, candidate diagnostics, selection"
```

---

## Task 7: Reasoner (`reasoner.py`)

**Files:**
- Create: `calibration/reasoner.py`
- Create: `tests/test_calibration_reasoner.py`

### Step 7.1: Write tests

- [ ] Write:

```python
# tests/test_calibration_reasoner.py
"""Tests for calibration.reasoner."""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from calibration.reasoner import build_reasoner_prompt, parse_reasoner_response, generate_variants
from calibration.registry import KnobRegistry


class TestBuildReasonerPrompt:
    def test_contains_required_sections(self):
        reg = KnobRegistry()
        prompt = build_reasoner_prompt(
            registry=reg,
            current_overlay={},
            current_diagnostic={"fail_rate": 0.3, "per_metric": {}},
            real_baseline={"self_bleu_2": {"median": 0.05}},
            trajectory=[],
            failed_strategies=[],
            metric_definitions="Test definitions",
        )
        assert "Tunable Knobs" in prompt
        assert "Test definitions" in prompt
        assert "fail_rate" in prompt


class TestParseReasonerResponse:
    def test_parses_valid_json(self):
        raw = json.dumps({
            "diagnosis": {"repetitiveness": "ok", "toxicity": "ok", "structure": "ok"},
            "strategy": "do something",
            "strategy_label": "test_strategy",
            "overlay_diff": {"persona.sentiment_bias_min": -0.4},
            "prompt_alternatives": {},
            "constraints": [],
        })
        result = parse_reasoner_response(raw)
        assert result["strategy_label"] == "test_strategy"
        assert result["overlay_diff"]["persona.sentiment_bias_min"] == -0.4

    def test_handles_missing_optional_fields(self):
        raw = json.dumps({
            "diagnosis": {},
            "strategy": "x",
            "strategy_label": "y",
            "overlay_diff": {},
        })
        result = parse_reasoner_response(raw)
        assert result["prompt_alternatives"] == {}
        assert result["constraints"] == []


class TestGenerateVariants:
    def test_returns_5_overlays(self):
        base_diff = {"persona.sentiment_bias_min": -0.4}
        prompt_alts = {
            "prompt.tone_guidance": ["alt1", "alt2"],
        }
        reg = KnobRegistry()
        current = reg.defaults()
        variants = generate_variants(current, base_diff, prompt_alts, reg, seed=42)
        assert len(variants) == 5

    def test_candidate_0_is_exact(self):
        base_diff = {"persona.sentiment_bias_min": -0.4}
        reg = KnobRegistry()
        current = reg.defaults()
        variants = generate_variants(current, base_diff, {}, reg, seed=42)
        expected = dict(current)
        expected.update(base_diff)
        assert variants[0] == expected

    def test_prompt_alternatives_applied(self):
        base_diff = {}
        prompt_alts = {
            "prompt.tone_guidance": ["alternative text 1", "alternative text 2"],
        }
        reg = KnobRegistry()
        current = reg.defaults()
        variants = generate_variants(current, base_diff, prompt_alts, reg, seed=42)
        # Candidates 3–4 should have the alternative prompt text
        assert variants[3]["prompt.tone_guidance"] == "alternative text 1"
        assert variants[4]["prompt.tone_guidance"] == "alternative text 2"
```

- [ ] Run: `python -m pytest tests/test_calibration_reasoner.py -v`
- [ ] Expected: FAIL

### Step 7.2: Implement reasoner

- [ ] Write:

```python
# calibration/reasoner.py
"""Calibration LLM: diagnosis, strategy proposal, overlay diff generation."""
from __future__ import annotations

import json
import random
from typing import Any

from openai import OpenAI

from .overlay import merge_overlay
from .registry import KnobRegistry


def build_reasoner_prompt(
    registry: KnobRegistry,
    current_overlay: dict[str, Any],
    current_diagnostic: dict[str, Any],
    real_baseline: dict[str, dict[str, Any]],
    trajectory: list[dict[str, Any]],
    failed_strategies: list[str],
    metric_definitions: str,
) -> str:
    """Assemble the full prompt for the calibration LLM."""
    sections = [
        "You are a calibration expert for a Reddit discussion simulator.",
        "Your goal: tune persona distributions and prompt text so generated "
        "discussions become statistically indistinguishable from real discussions.",
        "",
        "# Metric Definitions",
        metric_definitions,
        "",
        "# Tunable Knobs",
        registry.for_llm_context(),
        "",
        "# Current Overlay",
        json.dumps(current_overlay, indent=2),
        "",
        "# Current Diagnostic",
        f"Overall fail rate: {current_diagnostic.get('fail_rate', 'N/A')}",
        f"Mean absolute Cliff's delta: {current_diagnostic.get('mean_abs_delta', 'N/A')}",
        "",
        "Per-metric diagnostics:",
        json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "threads"}
             for k, v in current_diagnostic.get("per_metric", {}).items()},
            indent=2,
        ),
        "",
        "# Real Baseline Medians",
        json.dumps({k: v["median"] for k, v in real_baseline.items()}, indent=2),
        "",
        "# Trajectory (previous iterations)",
        json.dumps(trajectory, indent=2) if trajectory else "(first iteration)",
        "",
        "# Failed Strategies (do NOT re-propose these)",
        json.dumps(failed_strategies) if failed_strategies else "(none yet)",
        "",
        "# Instructions",
        "1. Diagnose which metrics are failing and why (too high / too low).",
        "2. Propose a strategy that addresses the failing metrics without "
        "regressing metrics that currently pass.",
        "3. Output an overlay_diff with specific knob changes.",
        "4. For any prompt knobs you change, also provide 2 alternative phrasings.",
        "",
        "Return a JSON object with this structure:",
        json.dumps({
            "diagnosis": {"repetitiveness": "...", "toxicity": "...", "structure": "..."},
            "strategy": "description of what to change and why",
            "strategy_label": "short_snake_case_label",
            "overlay_diff": {"knob.name": "new_value"},
            "prompt_alternatives": {"prompt.knob_name": ["alt1", "alt2"]},
            "constraints": ["do not regress metric X"],
        }, indent=2),
    ]
    return "\n".join(sections)


def call_reasoner(
    client: OpenAI,
    model: str,
    prompt: str,
) -> str:
    """Call the calibration LLM and return the raw response."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        response_format={"type": "json_object"},
        timeout=120.0,
    )
    return response.choices[0].message.content or ""


def parse_reasoner_response(raw: str) -> dict[str, Any]:
    """Parse the LLM response JSON into a structured dict."""
    data = json.loads(raw)
    return {
        "diagnosis": data.get("diagnosis", {}),
        "strategy": data.get("strategy", ""),
        "strategy_label": data.get("strategy_label", "unknown"),
        "overlay_diff": data.get("overlay_diff", {}),
        "prompt_alternatives": data.get("prompt_alternatives", {}),
        "constraints": data.get("constraints", []),
    }


def generate_variants(
    current_overlay: dict[str, Any],
    base_diff: dict[str, Any],
    prompt_alternatives: dict[str, list[str]],
    registry: KnobRegistry,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate 5 candidate overlays from the LLM's recommendation.

    - Candidate 0: exact recommendation
    - Candidates 1-2: numeric/distribution perturbations
    - Candidates 3-4: prompt alternatives (or numeric perturbations if none)
    """
    rng = random.Random(seed)

    # Candidate 0: exact
    exact = merge_overlay(current_overlay, base_diff)
    variants = [exact]

    # Candidates 1-2: perturb distribution knobs ±5-10%
    for _ in range(2):
        perturbed = dict(exact)
        for name, value in perturbed.items():
            knob_def = None
            try:
                knob_def = registry.get(name)
            except KeyError:
                continue
            if knob_def["type"] == "distribution" and isinstance(value, dict):
                perturbed[name] = _perturb_distribution(value, rng)
            elif knob_def["type"] == "float" and isinstance(value, (int, float)):
                jitter = value * rng.uniform(-0.1, 0.1)
                new_val = value + jitter
                if "min" in knob_def:
                    new_val = max(knob_def["min"], new_val)
                if "max" in knob_def:
                    new_val = min(knob_def["max"], new_val)
                perturbed[name] = round(new_val, 4)
        variants.append(perturbed)

    # Candidates 3-4: prompt alternatives or fallback to perturbations
    prompt_knobs = list(prompt_alternatives.keys())
    if prompt_knobs and all(len(v) >= 2 for v in prompt_alternatives.values()):
        for alt_idx in range(2):
            alt = dict(exact)
            for knob_name, alts in prompt_alternatives.items():
                if alt_idx < len(alts):
                    alt[knob_name] = alts[alt_idx]
            variants.append(alt)
    else:
        # Fallback: more numeric perturbations
        for _ in range(2):
            perturbed = dict(exact)
            for name, value in perturbed.items():
                knob_def = None
                try:
                    knob_def = registry.get(name)
                except KeyError:
                    continue
                if knob_def["type"] == "distribution" and isinstance(value, dict):
                    perturbed[name] = _perturb_distribution(value, rng)
            variants.append(perturbed)

    return variants


def _perturb_distribution(dist: dict[str, float], rng: random.Random) -> dict[str, float]:
    """Apply small random perturbations to a distribution, then renormalize."""
    perturbed = {}
    for key, val in dist.items():
        jitter = val * rng.uniform(-0.1, 0.1)
        perturbed[key] = max(0.0, val + jitter)
    total = sum(perturbed.values())
    if total > 0:
        perturbed = {k: round(v / total, 4) for k, v in perturbed.items()}
    return perturbed
```

- [ ] Run: `python -m pytest tests/test_calibration_reasoner.py -v`
- [ ] Expected: All PASS

### Step 7.3: Commit

- [ ] Run:
```bash
git add calibration/reasoner.py tests/test_calibration_reasoner.py
git commit -m "feat(calibration): add reasoner — LLM prompt assembly, response parsing, variant generation"
```

---

## Task 8: Orchestrator (`orchestrator.py`)

**Files:**
- Create: `calibration/orchestrator.py`
- Create: `tests/test_calibration_orchestrator.py`

### Step 8.1: Write test for orchestrator state management

- [ ] Write:

```python
# tests/test_calibration_orchestrator.py
"""Tests for calibration.orchestrator."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from calibration.orchestrator import CalibrationState, run_calibration_loop


class TestCalibrationState:
    def test_init_fresh(self, tmp_path):
        state = CalibrationState(output_dir=tmp_path / "cal_run")
        assert state.current_best_overlay == {}
        assert state.current_best_score is None
        assert state.completed_iterations == 0

    def test_save_and_load(self, tmp_path):
        out = tmp_path / "cal_run"
        state = CalibrationState(output_dir=out)
        state.current_best_overlay = {"persona.sentiment_bias_min": -0.4}
        state.current_best_score = {"fail_rate": 0.2, "mean_abs_delta": 0.15}
        state.completed_iterations = 3
        state.save()

        loaded = CalibrationState(output_dir=out)
        assert loaded.current_best_overlay == {"persona.sentiment_bias_min": -0.4}
        assert loaded.current_best_score["fail_rate"] == 0.2
        assert loaded.completed_iterations == 3
```

- [ ] Run: `python -m pytest tests/test_calibration_orchestrator.py -v`
- [ ] Expected: FAIL

### Step 8.2: Implement orchestrator

- [ ] Write:

```python
# calibration/orchestrator.py
"""Main calibration loop: phases 1-4."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .log import CalibrationLog
from .overlay import merge_overlay, save_overlay, diff_overlay
from .reasoner import (
    build_reasoner_prompt,
    call_reasoner,
    generate_variants,
    parse_reasoner_response,
)
from .registry import KnobRegistry
from .runner import run_candidates
from .scorer import (
    DEFAULT_METRICS,
    compute_real_baseline,
    load_thread_metrics,
    score_candidate,
    select_best_candidate,
)
from .stats import compare_before_after, evaluate_group_vs_real


class CalibrationState:
    """Persistent state for resume support."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.state_path = output_dir / "calibration_state.json"
        self.current_best_overlay: dict[str, Any] = {}
        self.current_best_score: dict[str, Any] | None = None
        self.current_best_diagnostic: dict[str, Any] | None = None
        self.completed_iterations: int = 0
        if self.state_path.exists():
            self._load()

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "current_best_overlay": self.current_best_overlay,
            "current_best_score": self.current_best_score,
            "completed_iterations": self.completed_iterations,
        }
        self.state_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.current_best_overlay = data.get("current_best_overlay", {})
        self.current_best_score = data.get("current_best_score")
        self.completed_iterations = data.get("completed_iterations", 0)


def run_calibration_loop(
    output_dir: Path,
    real_dir: Path,
    reference_run_config: dict[str, Any],
    max_iterations: int = 10,
    candidates_per_iter: int = 5,
    parallel: int = 1,
    calibration_model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    seed: int = 42,
    python: str = sys.executable,
    repo_root: Path | None = None,
    metrics: list[str] | None = None,
    metric_definitions: str = "",
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the full calibration loop (phases 1-4)."""
    from openai import OpenAI

    metrics = metrics or DEFAULT_METRICS
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = KnobRegistry()
    log = CalibrationLog(output_dir / "calibration_log.json")
    state = CalibrationState(output_dir)
    client = OpenAI(api_key=api_key, base_url=base_url)

    # ── Compute real baseline ────────────────────────────────────────────
    print("\n[calibration] Computing real baseline metrics...")
    real_baseline = compute_real_baseline(real_dir, metrics)
    baseline_path = output_dir / "real_baseline_metrics.json"
    baseline_path.write_text(
        json.dumps({k: {"median": v["median"], "mean": v["mean"]}
                     for k, v in real_baseline.items()},
                    ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Calibration iterations ───────────────────────────────────────────
    start_iter = state.completed_iterations
    for iteration in range(start_iter, max_iterations):
        print(f"\n{'='*60}")
        print(f"[calibration] Iteration {iteration}/{max_iterations - 1}")
        print(f"{'='*60}")

        iter_dir = output_dir / "iterations" / f"iter_{iteration:02d}"

        if iteration == 0:
            # No LLM reasoning — run with default overlay
            defaults = registry.defaults()
            overlays = [defaults] * candidates_per_iter
            strategy_label = "baseline"
            diagnosis_record: dict[str, Any] = {}
            overlay_diff: dict[str, Any] = {}
        else:
            # LLM reasoning
            prompt = build_reasoner_prompt(
                registry=registry,
                current_overlay=state.current_best_overlay,
                current_diagnostic=state.current_best_diagnostic or {},
                real_baseline=real_baseline,
                trajectory=log.trajectory(),
                failed_strategies=log.failed_strategies(),
                metric_definitions=metric_definitions,
            )
            print("[calibration] Calling calibration LLM...")
            raw = call_reasoner(client, calibration_model, prompt)
            parsed = parse_reasoner_response(raw)
            strategy_label = parsed["strategy_label"]
            overlay_diff = parsed["overlay_diff"]
            diagnosis_record = parsed["diagnosis"]

            # Save diagnosis
            iter_dir.mkdir(parents=True, exist_ok=True)
            (iter_dir / "diagnosis.json").write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # Generate 5 variant overlays
            overlays = generate_variants(
                current_overlay=state.current_best_overlay,
                base_diff=overlay_diff,
                prompt_alternatives=parsed.get("prompt_alternatives", {}),
                registry=registry,
                seed=seed + iteration,
            )

        # Save iteration overlay
        iter_dir.mkdir(parents=True, exist_ok=True)
        save_overlay(overlays[0], iter_dir / "overlay.json")

        # Run candidates
        print(f"[calibration] Running {len(overlays)} candidates (parallel={parallel})...")
        run_results = run_candidates(
            overlays=overlays,
            iter_dir=iter_dir,
            reference_run_config=reference_run_config,
            parallel=parallel,
            python=python,
            repo_root=repo_root,
        )

        # Score candidates
        scored: list[dict[str, Any]] = []
        for result in run_results:
            if not result["success"] or result["sim_dir"] is None:
                print(f"  [candidate {result['candidate_id']}] FAILED (skipped)")
                continue
            try:
                candidate_score = score_candidate(
                    result["sim_dir"], real_baseline, metrics
                )
                candidate_score["candidate_id"] = result["candidate_id"]
                candidate_score["sim_dir"] = str(result["sim_dir"])
                scored.append(candidate_score)
                print(
                    f"  [candidate {result['candidate_id']}] "
                    f"fail_rate={candidate_score['fail_rate']:.3f} "
                    f"mean_abs_delta={candidate_score['mean_abs_delta']:.3f}"
                )
            except Exception as exc:
                print(f"  [candidate {result['candidate_id']}] scoring error: {exc}")

        if not scored:
            print("[calibration] No candidates scored successfully. Skipping iteration.")
            state.completed_iterations = iteration + 1
            state.save()
            continue

        # Select best
        best = select_best_candidate(scored)
        beat_current = (
            state.current_best_score is None
            or best["fail_rate"] < state.current_best_score["fail_rate"]
            or (
                best["fail_rate"] == state.current_best_score["fail_rate"]
                and best["mean_abs_delta"] < state.current_best_score["mean_abs_delta"]
            )
        )

        if beat_current:
            state.current_best_overlay = overlays[best["candidate_id"]]
            state.current_best_score = {
                "fail_rate": best["fail_rate"],
                "mean_abs_delta": best["mean_abs_delta"],
            }
            state.current_best_diagnostic = best
            print(f"  → New best! fail_rate={best['fail_rate']:.3f}")
        else:
            print(f"  → No improvement. Keeping current best (fail_rate={state.current_best_score['fail_rate']:.3f})")

        # Log
        log_entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "strategy_label": strategy_label,
            "strategy_description": diagnosis_record.get("strategy", ""),
            "diagnosis": diagnosis_record,
            "overlay_diff_applied": overlay_diff,
            "candidates": [
                {
                    "candidate_id": s["candidate_id"],
                    "fail_rate": s["fail_rate"],
                    "mean_abs_delta": s["mean_abs_delta"],
                    "per_metric": {
                        k: {kk: vv for kk, vv in v.items() if kk != "threads"}
                        for k, v in s.get("per_metric", {}).items()
                    },
                }
                for s in scored
            ],
            "selection": {
                "winner": best["candidate_id"],
                "beat_current_best": beat_current,
                "best_fail_rate": best["fail_rate"],
                "best_mean_abs_delta": best["mean_abs_delta"],
            },
            "best_fail_rate": state.current_best_score["fail_rate"],
        }
        log.append(log_entry)

        # Save selection
        (iter_dir / "selection.json").write_text(
            json.dumps(log_entry["selection"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        state.completed_iterations = iteration + 1
        state.save()

    # ── Export best overlay ──────────────────────────────────────────────
    save_overlay(state.current_best_overlay, output_dir / "best_overlay.json")
    print(f"\n[calibration] Best overlay saved to {output_dir / 'best_overlay.json'}")

    # ── Write summary ────────────────────────────────────────────────────
    summary = {
        "total_iterations": max_iterations,
        "completed_iterations": state.completed_iterations,
        "best_fail_rate": state.current_best_score["fail_rate"] if state.current_best_score else None,
        "best_mean_abs_delta": state.current_best_score["mean_abs_delta"] if state.current_best_score else None,
        "successful_strategies": [
            e["strategy_label"] for e in log.entries()
            if e.get("selection", {}).get("beat_current_best")
        ],
        "failed_strategies": log.failed_strategies(),
        "trajectory": log.trajectory(),
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return summary
```

- [ ] Run: `python -m pytest tests/test_calibration_orchestrator.py -v`
- [ ] Expected: All PASS

### Step 8.3: Commit

- [ ] Run:
```bash
git add calibration/orchestrator.py tests/test_calibration_orchestrator.py
git commit -m "feat(calibration): add orchestrator — main calibration loop with state management and resume"
```

---

## Task 9: CLI (`cli.py` + `__main__.py`)

**Files:**
- Create: `calibration/cli.py`
- Create: `calibration/__main__.py`

### Step 9.1: Write CLI

- [ ] Write:

```python
# calibration/cli.py
"""CLI entry point for the calibration system."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    """Parse args and run the calibration loop."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Iterative LLM-driven calibration for Reddit discussion simulation."
    )
    parser.add_argument("--products-json", required=True, help="Product JSON file.")
    parser.add_argument("--real-dir", required=True, help="Real discussion directory.")
    parser.add_argument("--reference-run-dir", required=True, help="Reference simulation directory.")
    parser.add_argument("--iterations", type=int, default=10, help="Calibration iterations (default: 10).")
    parser.add_argument("--candidates", type=int, default=5, help="Candidates per iteration (default: 5).")
    parser.add_argument("--parallel", type=int, default=1, help="Max concurrent simulations (default: 1).")
    parser.add_argument("--calibration-model", default="gpt-4o-mini", help="LLM for calibration reasoning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--output-dir", default="artifacts/calibration_runs", help="Output directory.")
    parser.add_argument("--resume", action="store_true", help="Resume a previous run.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"],
                        help="Device for torch-based metrics.")
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    args = parser.parse_args()

    # Resolve paths
    repo_root = Path(__file__).resolve().parent.parent
    real_dir = Path(args.real_dir).resolve()
    reference_run_dir = Path(args.reference_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Load reference run config
    run_config_path = reference_run_dir / "run_config.json"
    if not run_config_path.exists():
        print(f"ERROR: run_config.json not found in {reference_run_dir}")
        sys.exit(1)
    reference_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))

    # Create timestamped output dir if not resuming
    if not args.resume:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / f"calibration_{ts}"

    # Load API key
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")

    # Load metric definitions if available
    metric_defs_path = repo_root / "docs" / "thread_metric_score_reference.md"
    metric_definitions = ""
    if metric_defs_path.exists():
        metric_definitions = metric_defs_path.read_text(encoding="utf-8")

    from .orchestrator import run_calibration_loop

    summary = run_calibration_loop(
        output_dir=output_dir,
        real_dir=real_dir,
        reference_run_config=reference_run_config,
        max_iterations=args.iterations,
        candidates_per_iter=args.candidates,
        parallel=args.parallel,
        calibration_model=args.calibration_model,
        api_key=api_key,
        base_url=base_url,
        seed=args.seed,
        python=args.python,
        repo_root=repo_root,
        metric_definitions=metric_definitions,
        device=args.device,
    )

    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Best fail rate: {summary.get('best_fail_rate', 'N/A')}")
    print(f"Successful strategies: {summary.get('successful_strategies', [])}")
    print(f"Failed strategies: {summary.get('failed_strategies', [])}")
```

- [ ] Write:

```python
# calibration/__main__.py
"""Allow `python -m calibration`."""
from calibration.cli import main

main()
```

### Step 9.2: Commit

- [ ] Run:
```bash
git add calibration/cli.py calibration/__main__.py
git commit -m "feat(calibration): add CLI and __main__ entry point"
```

---

## Task 10: Pipeline Modifications (`--overlay` support)

**Files:**
- Modify: `run_discussion.py`
- Modify: `product_reddit_sim/persona_gen.py`
- Modify: `product_reddit_sim/config_builder.py`

### Step 10.1: Add `--overlay` to `run_discussion.py`

- [ ] In `run_discussion.py`, add to `parse_args()`:

```python
    p.add_argument("--overlay", type=str, default=None,
                   help="Path to calibration overlay JSON (optional)")
```

- [ ] In `main()`, after `cli_args` dict is built (around line 147), add:

```python
    overlay = {}
    if args.overlay:
        import json as _json
        overlay = _json.loads(open(args.overlay, encoding="utf-8").read())

    cli_args["overlay"] = overlay
```

- [ ] Pass `overlay=overlay` to `generate_personas()` and ensure it's available to `build_config()`.
- [ ] Persist `cli_args["overlay"]` into `simulation_config.json` so each run keeps the overlay in its audit trail.

### Step 10.2: Add overlay support to `persona_gen.py`

- [ ] Modify `generate_personas()` signature to accept `overlay: dict | None = None`.
- [ ] In `_ensure_runtime_persona_diversity()`, check overlay for distribution overrides:

```python
    # At the start of _ensure_runtime_persona_diversity, after getting profiles:
    if overlay:
        # Apply conflict_style distribution override
        conflict_dist = overlay.get("persona.conflict_style_distribution")
        if conflict_dist:
            styles = list(conflict_dist.keys())
            weights = list(conflict_dist.values())
            for profile in profiles:
                if "conflict_style" not in profile or not profile["conflict_style"]:
                    profile["conflict_style"] = rng.choices(styles, weights=weights, k=1)[0]

        # Apply primary_motivation distribution override
        motivation_dist = overlay.get("persona.primary_motivation_distribution")
        if motivation_dist:
            motivations = list(motivation_dist.keys())
            weights = list(motivation_dist.values())
            for profile in profiles:
                if "primary_motivation" not in profile or not profile["primary_motivation"]:
                    profile["primary_motivation"] = rng.choices(motivations, weights=weights, k=1)[0]
```

### Step 10.3: Test overlay integration

- [ ] Run the existing test suite to confirm no regressions:
```bash
python -m pytest tests/ -v --ignore=tests/test_calibration_stats.py --ignore=tests/test_calibration_registry.py --ignore=tests/test_calibration_overlay.py --ignore=tests/test_calibration_log.py --ignore=tests/test_calibration_runner.py --ignore=tests/test_calibration_scorer.py --ignore=tests/test_calibration_reasoner.py --ignore=tests/test_calibration_orchestrator.py
```
- [ ] Expected: All existing tests PASS

### Step 10.4: Commit

- [ ] Run:
```bash
git add run_discussion.py product_reddit_sim/persona_gen.py product_reddit_sim/config_builder.py
git commit -m "feat: add --overlay support to pipeline — persona distributions and prompt text overrides"
```

---

## Task 11: Add Dependencies to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

### Step 11.1: Add pandas, numpy, scipy

- [ ] Add to the `dependencies` list in `pyproject.toml`:

```toml
dependencies = [
  "openai>=1.0.0",
  "python-dotenv>=1.0.1",
  "pandas>=2.0.0",
  "numpy>=1.24.0",
  "scipy>=1.10.0",
]
```

- [ ] Also add `calibration` to `[tool.setuptools.packages.find]`:

```toml
[tool.setuptools.packages.find]
include = ["product_reddit_sim", "calibration"]
```

### Step 11.2: Install and verify

- [ ] Run:
```bash
pip install -e .
python -c "from calibration.stats import cliffs_delta; print('OK')"
```

### Step 11.3: Run full test suite

- [ ] Run:
```bash
python -m pytest tests/ -v
```
- [ ] Expected: All PASS

### Step 11.4: Commit

- [ ] Run:
```bash
git add pyproject.toml
git commit -m "chore: add pandas, numpy, scipy dependencies and calibration package to build"
```

---

## Task 12: Full Integration Smoke Test

### Step 12.1: Verify all modules import cleanly

- [ ] Run:
```bash
python -c "
from calibration.stats import cliffs_delta, empirical_p_value, empirical_percentile, evaluate_group_vs_real, diagnose_single_generation, compare_before_after
from calibration.registry import KnobRegistry
from calibration.overlay import load_overlay, save_overlay, merge_overlay, diff_overlay
from calibration.scorer import load_thread_metrics, compute_real_baseline, score_candidate, select_best_candidate
from calibration.reasoner import build_reasoner_prompt, parse_reasoner_response, generate_variants
from calibration.runner import run_candidates
from calibration.orchestrator import CalibrationState, run_calibration_loop
from calibration.log import CalibrationLog
print('All imports OK')
"
```

### Step 12.2: Verify CLI help works

- [ ] Run:
```bash
python -m calibration --help
```
- [ ] Expected: Help text printed with all flags

### Step 12.3: Run full test suite one final time

- [ ] Run:
```bash
python -m pytest tests/ -v
```
- [ ] Expected: All PASS

### Step 12.4: Final commit

- [ ] Run:
```bash
git add -A
git commit -m "feat(calibration): complete calibration system — all modules, tests, and CLI"
```
