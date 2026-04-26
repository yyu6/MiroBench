# Calibration System Design

**Date:** 2026-04-26
**Revised:** 2026-04-26
**Status:** Approved
**Goal:** Iterative, LLM-driven calibration loop that tunes simulation prompts
and persona distributions so that generated Reddit discussions become
statistically indistinguishable from real discussions under independent
hypothesis tests.

---

## 1. Overview

The calibration system wraps around the existing simulation pipeline
(`run_discussion.py` → analyze → personas → config → simulate → export).

### 1.1 Phases

1. **Before-calibration evaluation** — compare 50 initial generated discussions
   against 50 real discussions using group-level statistical tests.
2. **Calibration loop** (default 10 iterations) — each iteration generates 5
   candidate discussions, diagnoses per-instance metric gaps via empirical
   p-values, selects the best candidate, and uses an LLM to propose persona
   and prompt changes.
3. **After-calibration evaluation** — compare 50 calibrated generated
   discussions against the same 50 real discussions using the same tests.
4. **Improvement analysis** — compare before vs after results per metric.

### 1.2 What the LLM Can Modify

The calibration LLM can modify two layers:

- **Persona layer** — distributions over conflict_style, primary_motivation,
  knowledge_style, stance; sentiment_bias range.
- **Prompt layer** — text blocks for anti-paraphrase instruction, tone
  guidance, structure preference weighting, depth soft cap language, few-shot
  style anchoring.

The LLM does **not** modify config-layer knobs (activity_level, time
multipliers, comments_per_hour, etc.). Those remain fixed at their reference
run values.

---

## 2. Metric Domains

Three domains, each with specific metrics from the existing evaluation suite:

### 2.1 Repetitiveness

| Metric | Source |
|---|---|
| `self_bleu_2`, `self_bleu_3`, `self_bleu_4` | `score_thread_self_bleu.py` |
| `self_bertscore_mean_f1` | `score_thread_self_bertscore.py` |
| `semantic_mean_cosine` | `score_thread_semantic_uniformity.py` |

### 2.2 Toxicity / Aggressiveness

| Metric | Source |
|---|---|
| `toxicity_mean`, `toxicity_max`, `toxicity_p90` | `score_thread_detoxify.py` |
| `severe_toxicity_mean`, `severe_toxicity_max`, `severe_toxicity_p90` | `score_thread_detoxify.py` |
| `obscene_mean`, `obscene_max`, `obscene_p90` | `score_thread_detoxify.py` |
| `threat_mean`, `threat_max`, `threat_p90` | `score_thread_detoxify.py` |
| `insult_mean`, `insult_max`, `insult_p90` | `score_thread_detoxify.py` |
| `identity_attack_mean`, `identity_attack_max`, `identity_attack_p90` | `score_thread_detoxify.py` |
| `aggression_score_mean`, `aggression_score_max` | `score_thread_detoxify.py` |

### 2.3 Thread Structure / Complexity

| Metric | Source |
|---|---|
| `length_std`, `length_iqr`, `length_cv` | `score_thread_structure.py` |
| `max_depth`, `avg_depth` | `score_thread_structure.py` |
| `avg_branching_factor` | `score_thread_structure.py` |
| `structural_virality` | `score_thread_structure.py` |

Metric selection is configurable; the domains and their member metrics can be
changed without modifying system architecture.

---

## 3. Package Structure

```
calibration/
├── __init__.py
├── cli.py                # CLI entry point
├── orchestrator.py       # Main iteration loop
├── registry.py           # Knob registry: persona + prompt tunable parameters
├── overlay.py            # Overlay model: load/save/merge/validate
├── scorer.py             # Run metric scripts, compute statistical tests
├── stats.py              # Statistical functions: Cliff's delta, empirical p-value, etc.
├── reasoner.py           # LLM diagnosis, strategy proposal, overlay diff
├── runner.py             # Subprocess pool for candidate simulations
└── log.py                # Calibration log: per-iteration records
```

---

## 4. Statistical Evaluation Framework

### 4.1 Core Functions (`stats.py`)

Implemented using pandas, numpy, scipy.stats:

```python
def cliffs_delta(x, y):
    """Cliff's delta effect size. Positive means x tends to be higher than y."""

def empirical_p_value(real_values, gen_value):
    """Two-sided empirical p-value measuring how far gen_value is from real median.

    center = median(real_values)
    gen_dist = abs(gen_value - center)
    real_dist = abs(real_values - center)
    p = (count(real_dist >= gen_dist) + 1) / (len(real_values) + 1)
    """

def empirical_percentile(real_values, gen_value):
    """Percentile of gen_value within the real distribution."""

def evaluate_group_vs_real(real_df, gen_df, metrics, alpha=0.05):
    """Group-level evaluation: 50 generated vs 50 real discussions."""

def diagnose_single_generation(real_df, gen_row, metrics, alpha=0.05):
    """Per-instance calibration diagnostic for one generated thread."""

def compare_before_after(before_results, after_results):
    """Improvement analysis comparing before and after calibration."""
```

Missing values are handled by dropping NaNs per metric.

### 4.2 Phase 1: Before-Calibration Group Evaluation

Compare 50 initial generated discussions against 50 real discussions.

For each metric, compute:

| Output Field | Description |
|---|---|
| `real_mean` | Mean of real values |
| `real_median` | Median of real values |
| `generated_mean` | Mean of generated values |
| `generated_median` | Median of generated values |
| `mwu_p_value` | Mann-Whitney U test p-value |
| `ks_statistic` | Kolmogorov-Smirnov test statistic |
| `ks_p_value` | Kolmogorov-Smirnov test p-value |
| `cliffs_delta` | Effect size; positive = generated higher than real |
| `direction` | `generated_higher` / `generated_lower` / `similar` |
| `empirical_fail_rate` | Fraction of generated values with empirical p < 0.05 |

Output: `before_calibration_group_eval.csv`

### 4.3 Phase 2: During-Calibration Single-Simulation Diagnostic

Given one generated discussion row and the real baseline, compute per-metric:

| Output Field | Description |
|---|---|
| `real_median` | Median of real values |
| `generated_value` | The generated thread's value |
| `empirical_p_value` | Two-sided empirical p-value (see formula above) |
| `percentile` | Percentile of generated value within real distribution |
| `direction` | `too_high` / `too_low` / `within_baseline` |
| `diagnosis_flag` | `fail` if empirical p-value < 0.05 |

This is the diagnostic the calibration LLM receives each iteration to
understand which metrics are out of distribution and in which direction.

Output: `single_generation_diagnostic.csv` (optional, per candidate)

### 4.4 Phase 3: After-Calibration Group Evaluation

Compare 50 calibrated generated discussions against 50 real discussions using
the same metrics and tests as Phase 1:

- Mann-Whitney U p-value
- KS test statistic and p-value
- Cliff's delta
- Empirical p-value fail rate

Output: `after_calibration_group_eval.csv`

### 4.5 Phase 4: Improvement Analysis

Compare before-calibration and after-calibration results. For each metric:

| Output Field | Description |
|---|---|
| `before_mwu_p` | Before MWU p-value |
| `after_mwu_p` | After MWU p-value |
| `before_ks_p` | Before KS p-value |
| `after_ks_p` | After KS p-value |
| `before_cliffs_delta` | Before Cliff's delta |
| `after_cliffs_delta` | After Cliff's delta |
| `abs_delta_reduction` | `abs(before_delta) - abs(after_delta)` |
| `before_fail_rate` | Before empirical fail rate |
| `after_fail_rate` | After empirical fail rate |
| `fail_rate_reduction` | `before_fail_rate - after_fail_rate` |
| `improved` | True if `abs(after_delta) < abs(before_delta)` AND `after_fail_rate < before_fail_rate` |

Overall summary:

| Summary Field | Description |
|---|---|
| `metrics_sig_different_before` | Count of metrics with MWU p < 0.05 before |
| `metrics_sig_different_after` | Count of metrics with MWU p < 0.05 after |
| `avg_abs_cliffs_delta_before` | Mean of `abs(cliffs_delta)` before |
| `avg_abs_cliffs_delta_after` | Mean of `abs(cliffs_delta)` after |
| `overall_pass_rate_before` | Fraction of (metric, thread) pairs with empirical p >= 0.05 before |
| `overall_pass_rate_after` | Fraction of (metric, thread) pairs with empirical p >= 0.05 after |
| `overall_fail_rate_before` | `1 - overall_pass_rate_before` |
| `overall_fail_rate_after` | `1 - overall_pass_rate_after` |

Output: `before_after_improvement_summary.csv`

A concise summary is also printed to the terminal.

---

## 5. Knob Registry

A declarative catalog of every tunable parameter across two layers.

### 5.1 Entry Format

```json
{
    "name": "persona.conflict_style_distribution",
    "layer": "persona",
    "domain": "toxicity",
    "type": "distribution",
    "keys": ["calm", "skeptical", "blunt", "sarcastic", "argumentative", "avoidant"],
    "constraints": "values sum to 1.0, each >= 0.0",
    "default": {"calm": 0.2, "skeptical": 0.25, "blunt": 0.15, "sarcastic": 0.15, "argumentative": 0.1, "avoidant": 0.15},
    "description": "Distribution of conflict styles across generated personas"
}
```

### 5.2 Layers

**Persona layer** (~8 knobs): distributions over `conflict_style`,
`primary_motivation`, `knowledge_style`, `stance`; `sentiment_bias` range.

**Prompt layer** (~6 knobs): text blocks for anti-paraphrase instruction, tone
guidance, structure preference weighting, depth soft cap language, few-shot
style anchoring.

### 5.3 Registry Responsibilities

1. **Validation** — reject LLM-proposed changes that are out of bounds.
2. **Documentation** — the LLM receives the registry as context so it knows
   what it can change.
3. **Diffing** — track what changed between iterations.

---

## 6. Overlay System

An overlay is a sparse JSON dict of knob overrides keyed by knob name. Only
changed knobs appear; everything else falls through to defaults.

```json
{
    "persona.conflict_style_distribution": {"calm": 0.1, "skeptical": 0.2, "blunt": 0.25, "sarcastic": 0.2, "argumentative": 0.15, "avoidant": 0.1},
    "prompt.anti_paraphrase_instruction": "If three or more visible comments express the same viewpoint, do NOT add another."
}
```

### 6.1 Key Operations

- `merge(base_defaults, overlay) → resolved_config` — full config for a run.
- `validate(overlay, registry) → errors` — type, range, distribution checks.
- `diff(overlay_a, overlay_b) → changes` — human-readable diff for the log.

### 6.2 Pipeline Integration

The existing pipeline modules (`persona_gen.py`, `oasis_reddit.py`) receive
light modifications to accept an optional `--overlay` path. When present,
overrides are read from it. When absent, behavior is unchanged — zero
regression risk.

---

## 7. Scorer (`scorer.py`)

### 7.1 Per-Candidate Scoring

For each candidate simulation:

1. Invoke `run_full_thread_metric_suite.py` as a subprocess.
2. Read `thread_metrics_summary.csv` from the candidate's output directory.

### 7.2 During-Calibration Diagnostic

For each candidate's threads, run `diagnose_single_generation()` against the
real baseline. The diagnostic tells the calibration LLM:

- Which metrics are failing (empirical p < 0.05)
- In which direction (too high / too low)
- The percentile position within the real distribution

### 7.3 Candidate Selection

Selection uses the empirical fail rate: the candidate with the **lowest
fraction of failing metrics** across its threads wins. Ties are broken by
the mean absolute Cliff's delta (lower is better).

To beat the current best, a candidate must have a strictly lower fail rate
(or equal fail rate with lower mean absolute delta). If no candidate beats
the current best, the current best carries forward and the strategies are
logged as unsuccessful.

---

## 8. Reasoner (Calibration LLM)

### 8.1 Interface

One LLM call per iteration. The prompt contains:

1. Metric definitions (full reference doc).
2. Knob registry (persona + prompt tunable parameters with types, ranges).
3. Real baseline metrics (reference distribution).
4. Current best candidate's per-metric diagnostic (empirical p-values,
   percentiles, fail flags, directions).
5. Per-domain summary: fail counts and dominant direction per domain.
6. Domain score trajectory across all prior iterations.
7. Calibration log — all previously tried strategies, success/failure status.

### 8.2 LLM Output Structure

```json
{
    "diagnosis": {
        "repetitiveness": "self_bleu_2 and self_bleu_3 are failing (too high, p=0.01). Comments are paraphrasing...",
        "toxicity": "All toxicity metrics pass. Slightly below real median but within baseline.",
        "structure": "avg_depth fails (too low, p=0.03). Threads are too flat..."
    },
    "strategy": "Tighten anti-paraphrase prompt to reduce self-BLEU. Shift structure preference toward replies to deepen threads.",
    "strategy_label": "reduce_paraphrase_deepen_threads",
    "overlay_diff": {
        "persona.conflict_style_distribution": {"calm": 0.15, "skeptical": 0.2, "blunt": 0.2, "sarcastic": 0.2, "argumentative": 0.15, "avoidant": 0.1},
        "prompt.anti_paraphrase_instruction": "If three or more visible comments express the same viewpoint..."
    },
    "prompt_alternatives": {
        "prompt.anti_paraphrase_instruction": [
            "Alternative phrasing 1...",
            "Alternative phrasing 2..."
        ]
    },
    "constraints": ["Do not regress toxicity metrics that currently pass"]
}
```

### 8.3 Candidate Variant Generation

The LLM produces one overlay diff. Five candidates are derived:

- **Candidate 0**: LLM's exact recommendation.
- **Candidates 1–2**: Distribution knob perturbations (±5–10% jitter on
  persona distributions), prompt text unchanged.
- **Candidates 3–4**: Same distribution knobs as candidate 0, but with the
  LLM-provided alternative prompt phrasings.

If the LLM did not change any prompt knobs, candidates 3–4 fall back to
distribution-only perturbations.

### 8.4 Model Configuration

CLI flag `--calibration-model` (default: `gpt-4o-mini`). The calibration LLM
is independent from the simulation LLM.

### 8.5 Strategy Memory

The reasoner prompt instructs: "Review the calibration log. Do not re-propose
strategies labeled as failed. If a domain's metrics currently pass, include a
constraint to preserve that."

---

## 9. Runner

### 9.1 Subprocess Pool

- Accepts a list of overlay configs and a `--parallel` flag (default 1, max 5).
- Each candidate gets its own output directory under
  `iterations/iter_XX/candidates/candidate_N/`.
- Invokes `run_discussion.py` as a subprocess with `--overlay`.
- Captures stdout/stderr per candidate for debugging.
- Failed simulations (non-zero exit) are excluded from scoring, not fatal.

---

## 10. Orchestrator

### 10.1 Main Loop

```
load or initialize calibration state:
    - real baseline (50 real threads, metrics computed once, cached)
    - knob registry
    - calibration log (empty or resumed)
    - current best overlay (defaults for fresh run, or loaded if resuming)

Phase 1: Before-calibration evaluation
    generate 50 discussions with default overlay
    run evaluate_group_vs_real() → before_calibration_group_eval.csv

Phase 2: Calibration loop
    for iteration in range(max_iterations):
        if iteration == 0:
            generate 5 candidates from default overlay (no LLM reasoning)
            score all → diagnose → select best → establish baseline
        else:
            feed reasoner: current best diagnostic, trajectory, log
            reasoner returns: diagnosis, strategy, overlay diff, prompt alternatives
            generate 5 variant overlays
            runner launches 5 candidates (parallel pool)
            scorer diagnoses each candidate
            select best candidate (lowest fail rate)
            if best < current best:
                update current best
                log: strategy succeeded
            else:
                keep current best
                log: strategy failed
        save iteration artifacts

Phase 3: After-calibration evaluation
    generate 50 discussions with best overlay
    run evaluate_group_vs_real() → after_calibration_group_eval.csv

Phase 4: Improvement analysis
    run compare_before_after() → before_after_improvement_summary.csv
    print terminal summary

export best_overlay.json
export calibration_summary.json
```

### 10.2 Resume Support

The orchestrator detects completed iteration artifacts and resumes from the
next incomplete iteration. Calibration log and current best are loaded from
disk.

---

## 11. Output Artifacts

### 11.1 Directory Structure

```
artifacts/calibration_runs/<run_id>/
├── calibration_log.json
├── calibration_summary.json
├── best_overlay.json
├── real_baseline_metrics.json
├── before_calibration_group_eval.csv
├── after_calibration_group_eval.csv
├── before_after_improvement_summary.csv
├── iterations/
│   ├── iter_00/
│   │   ├── overlay.json
│   │   ├── candidates/
│   │   │   ├── candidate_0/
│   │   │   │   ├── <simulation output files>
│   │   │   │   └── single_generation_diagnostic.csv
│   │   │   ├── candidate_1/
│   │   │   └── ...
│   │   ├── diagnosis.json
│   │   └── selection.json
│   ├── iter_01/
│   └── ...
├── before_calibration_runs/        # 50 runs for Phase 1
└── after_calibration_runs/         # 50 runs for Phase 3
```

### 11.2 Calibration Log Entry

```json
{
    "iteration": 3,
    "timestamp": "2026-04-27T14:32:01",
    "strategy_label": "reduce_paraphrase_deepen_threads",
    "strategy_description": "Tighten anti-paraphrase prompt...",
    "diagnosis": {
        "repetitiveness": "self_bleu_2 failing (too high, p=0.01)...",
        "toxicity": "All passing...",
        "structure": "avg_depth failing (too low, p=0.03)..."
    },
    "overlay_diff_applied": {
        "prompt.anti_paraphrase_instruction": "..."
    },
    "candidates": [
        {
            "candidate_id": 0,
            "variant_type": "exact",
            "fail_rate": 0.12,
            "mean_abs_cliffs_delta": 0.18,
            "per_metric": {
                "self_bleu_2": {
                    "value": 0.08,
                    "real_median": 0.05,
                    "empirical_p": 0.14,
                    "percentile": 72,
                    "direction": "within_baseline",
                    "pass": true
                },
                "avg_depth": {
                    "value": 1.5,
                    "real_median": 1.69,
                    "empirical_p": 0.03,
                    "percentile": 18,
                    "direction": "too_low",
                    "pass": false
                }
            }
        }
    ],
    "selection": {
        "winner": 0,
        "beat_current_best": true,
        "previous_best_fail_rate": 0.20,
        "new_best_fail_rate": 0.12
    },
    "trajectory_so_far": [
        {
            "iteration": 0,
            "fail_rate": 0.35,
            "mean_abs_cliffs_delta": 0.31,
            "failing_metrics": ["self_bleu_2", "self_bleu_3", "avg_depth", "toxicity_mean"]
        },
        {
            "iteration": 1,
            "fail_rate": 0.25,
            "mean_abs_cliffs_delta": 0.24,
            "failing_metrics": ["self_bleu_2", "avg_depth", "toxicity_mean"]
        }
    ],
    "failed_strategies_so_far": ["flatten_structure_and_soften_tone"]
}
```

### 11.3 Final Summary

`calibration_summary.json` contains the Phase 4 improvement analysis plus
calibration loop metadata:

```json
{
    "total_iterations": 10,
    "total_candidates_evaluated": 50,
    "best_iteration": 8,
    "successful_strategies": ["reduce_paraphrase_deepen_threads", "diversify_motivation_mix"],
    "failed_strategies": ["flatten_structure_and_soften_tone"],
    "before_calibration": {
        "metrics_sig_different": 8,
        "avg_abs_cliffs_delta": 0.31,
        "overall_fail_rate": 0.35,
        "overall_pass_rate": 0.65
    },
    "after_calibration": {
        "metrics_sig_different": 2,
        "avg_abs_cliffs_delta": 0.09,
        "overall_fail_rate": 0.08,
        "overall_pass_rate": 0.92
    },
    "per_metric_improvement": {
        "self_bleu_2": {
            "before_cliffs_delta": 0.45,
            "after_cliffs_delta": 0.08,
            "abs_delta_reduction": 0.37,
            "before_fail_rate": 0.72,
            "after_fail_rate": 0.06,
            "improved": true
        }
    },
    "trajectory": []
}
```

### 11.4 Exportable Overlay

`best_overlay.json` — the winning overlay config, usable in future runs:

```bash
python run_discussion.py products.json --overlay artifacts/calibration_runs/<run_id>/best_overlay.json
```

---

## 12. CLI

```bash
python -m calibration \
    --products-json data/processed/splits/credit_cards/product_descriptions_train.json \
    --real-dir data/raw/discussions/credit_cards/american_express_platinum_card \
    --reference-run-dir artifacts/simulations/credit_cards_20260420_150514 \
    --iterations 10 \
    --candidates 5 \
    --parallel 3 \
    --calibration-model gpt-4o-mini \
    --seed 42 \
    --output-dir artifacts/calibration_runs/ \
    --resume
```

| Flag | Default | Description |
|---|---|---|
| `--products-json` | required | Product JSON file for simulation |
| `--real-dir` | required | Real discussion directory for baseline metrics |
| `--reference-run-dir` | required | Reference simulation to clone run settings from |
| `--iterations` | 10 | Number of calibration iterations |
| `--candidates` | 5 | Candidates per iteration |
| `--parallel` | 1 | Max concurrent candidate simulations |
| `--calibration-model` | gpt-4o-mini | LLM for calibration reasoning |
| `--seed` | 42 | Base random seed |
| `--output-dir` | `artifacts/calibration_runs/` | Output root |
| `--resume` | false | Resume a partially completed run |

---

## 13. Pipeline Modifications

The existing pipeline requires light changes to support the `--overlay` flag:

1. **`run_discussion.py`** — add `--overlay` argument. Load overlay JSON and
   pass it to `generate_personas` and the simulation runtime.
2. **`persona_gen.py`** — read persona distribution overrides from overlay
   (conflict_style, motivation, knowledge_style, stance distributions).
3. **`oasis_reddit.py`** — read prompt text overrides from overlay
   (anti-paraphrase, tone guidance, structure preference text).

Each module falls through to its current hardcoded defaults when no overlay
is provided. Existing behavior is preserved exactly when `--overlay` is
absent.
