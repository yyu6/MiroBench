# Calibration System Design

**Date:** 2026-04-26
**Status:** Approved
**Goal:** Iterative, LLM-driven calibration loop that tunes simulation prompts,
persona distributions, and config knobs so that generated Reddit discussions
converge toward the statistical profile of real discussions.

---

## 1. Overview

The calibration system wraps around the existing simulation pipeline
(`run_discussion.py` → analyze → personas → config → simulate → export).
Each iteration:

1. Generates 5 candidate discussions from a mutated overlay config.
2. Scores each candidate against real-thread metrics.
3. Selects the best candidate by distance-to-real.
4. Uses an LLM to diagnose metric gaps, propose a strategy, and output a new
   overlay diff.
5. Carries forward the best config seen so far; logs all strategies (successful
   and failed) so they are not re-attempted.

Default: 10 iterations, configurable via CLI.

---

## 2. Metric Domains

Three domains, each with specific metrics drawn from the existing evaluation
suite:

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
├── registry.py           # Knob registry: tunable parameters, types, ranges
├── overlay.py            # Overlay model: load/save/merge/validate
├── scorer.py             # Run metric scripts, compute distance-to-real
├── reasoner.py           # LLM diagnosis, strategy proposal, overlay diff
├── runner.py             # Subprocess pool for candidate simulations
└── log.py                # Calibration log: per-iteration records
```

---

## 4. Knob Registry

A declarative catalog of every tunable parameter. Each entry:

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

### 4.1 Layers

**Persona layer** (~8 knobs): distributions over `conflict_style`,
`primary_motivation`, `knowledge_style`, `stance`; `sentiment_bias` range.

**Prompt layer** (~6 knobs): text blocks for anti-paraphrase instruction, tone
guidance, structure preference weighting, depth soft cap language, few-shot
style anchoring.

**Config layer** (~10 knobs): `activity_level` range, `comments_per_hour`
range, `posts_per_hour` range, `depth_soft_cap` value, time multipliers
(peak / off-peak / morning / work), `response_delay` ranges.

### 4.2 Registry Responsibilities

1. **Validation** — reject LLM-proposed changes that are out of bounds.
2. **Documentation** — the LLM receives the registry as context so it knows
   what it can change.
3. **Diffing** — track what changed between iterations.

---

## 5. Overlay System

An overlay is a sparse JSON dict of knob overrides keyed by knob name. Only
changed knobs appear; everything else falls through to defaults.

```json
{
    "persona.conflict_style_distribution": {"calm": 0.1, "skeptical": 0.2, "blunt": 0.25, "sarcastic": 0.2, "argumentative": 0.15, "avoidant": 0.1},
    "config.depth_soft_cap": 4,
    "prompt.anti_paraphrase_instruction": "If three or more visible comments express the same viewpoint, do NOT add another."
}
```

### 5.1 Key Operations

- `merge(base_defaults, overlay) → resolved_config` — full config for a run.
- `validate(overlay, registry) → errors` — type, range, distribution checks.
- `diff(overlay_a, overlay_b) → changes` — human-readable diff for the log.

### 5.2 Pipeline Integration

The existing pipeline modules (`persona_gen.py`, `config_builder.py`,
`oasis_reddit.py`) receive light modifications to accept an optional
`--overlay` path. When present, overrides are read from it. When absent,
behavior is unchanged — zero regression risk.

---

## 6. Scorer & Distance-to-Real

### 6.1 Scoring

For each candidate simulation:

1. Invoke `run_full_thread_metric_suite.py` as a subprocess.
2. Read `thread_metrics_summary.csv` from the candidate's output directory.
3. Aggregate per-thread metrics into a candidate-level summary (median across
   threads).

### 6.2 Real Baseline

Compute metrics for the real discussion directory on first run. Cache to
`real_baseline_metrics.json`. Contains per-metric median and IQR across all
real threads.

### 6.3 Distance Computation

Per-metric z-score-style distance:

```
distance_i = |candidate_median_i - real_median_i| / real_iqr_i
```

IQR-based normalization for outlier robustness. Epsilon fallback when IQR ≈ 0.

Per-domain aggregates:

```
d_repetitiveness = mean(distance_i for i in repetitiveness_metrics)
d_toxicity       = mean(distance_i for i in toxicity_metrics)
d_structure      = mean(distance_i for i in structure_metrics)
d_total          = mean(d_repetitiveness, d_toxicity, d_structure)
```

Equal domain weights by default.

### 6.4 Selection

Lowest `d_total` wins among the 5 candidates. To beat the current best, a
candidate must have `d_total < current_best_d_total`. If no candidate beats
the current best, the current best carries forward and the strategies are
logged as unsuccessful.

### 6.5 Score Reporting

Both raw metric values and distances are tracked everywhere — in the log,
trajectory, and final summary. The calibration LLM and anyone reviewing
results can see the actual values, the real reference values, and how far
apart they are.

---

## 7. Reasoner (Calibration LLM)

### 7.1 Interface

One LLM call per iteration. The prompt contains:

1. Metric definitions (full reference doc).
2. Knob registry (all tunable parameters with types, ranges, descriptions).
3. Real baseline metrics (reference distribution).
4. Current best overlay + its four distance scores + raw metric values.
5. Domain score trajectory across all prior iterations.
6. Calibration log — all previously tried strategies, success/failure status.
7. Current best candidate's raw per-thread metrics.

### 7.2 LLM Output Structure

```json
{
    "diagnosis": {
        "repetitiveness": "Self-BLEU is 0.15 above real median...",
        "toxicity": "Within acceptable range, slightly below real...",
        "structure": "avg_depth is 1.2 below real..."
    },
    "strategy": "Increase reply preference and raise depth cap...",
    "strategy_label": "deepen_threads_and_raise_edge",
    "overlay_diff": {
        "config.depth_soft_cap": 4,
        "persona.conflict_style_distribution": { "..." : "..." }
    },
    "prompt_alternatives": {
        "prompt.anti_paraphrase_instruction": [
            "Alternative phrasing 1...",
            "Alternative phrasing 2..."
        ]
    },
    "constraints": ["Do not regress d_repetitiveness above 0.35"]
}
```

### 7.3 Candidate Variant Generation

The LLM produces one overlay diff. Five candidates are derived:

- **Candidate 0**: LLM's exact recommendation.
- **Candidates 1–2**: Numeric/distribution knob perturbations (±5–10%
  jitter), prompt text unchanged.
- **Candidates 3–4**: Same numeric knobs as candidate 0, but with the
  LLM-provided alternative prompt phrasings.

If the LLM did not change any prompt knobs in a given iteration, candidates
3–4 fall back to numeric-only perturbations.

### 7.4 Model Configuration

CLI flag `--calibration-model` (default: `gpt-4o-mini`). The calibration LLM
is independent from the simulation LLM.

### 7.5 Strategy Memory

The reasoner prompt instructs: "Review the calibration log. Do not re-propose
strategies labeled as failed. If a domain improved in recent iterations,
include a constraint to preserve that gain."

---

## 8. Runner

### 8.1 Subprocess Pool

- Accepts a list of overlay configs and a `--parallel` flag (default 1, max 5).
- Each candidate gets its own output directory under
  `iterations/iter_XX/candidates/candidate_N/`.
- Invokes `run_discussion.py` as a subprocess with `--overlay`.
- Captures stdout/stderr per candidate for debugging.
- Failed simulations (non-zero exit) are excluded from scoring, not fatal.

---

## 9. Orchestrator

### 9.1 Main Loop

```
load or initialize calibration state:
    - real baseline metrics (compute once, cache)
    - knob registry
    - calibration log (empty or resumed)
    - current best overlay (defaults for fresh run, or loaded if resuming)

for iteration in range(max_iterations):
    if iteration == 0:
        generate 5 candidates from default overlay (no LLM reasoning)
        score all → select best → establish baseline distances
    else:
        feed reasoner: current best overlay, distances, trajectory, log
        reasoner returns: diagnosis, strategy, overlay diff, prompt alternatives
        generate 5 variant overlays
        runner launches 5 candidates (parallel pool)
        scorer evaluates each → distances
        select best candidate
        if best < current best:
            update current best
            log: strategy succeeded
        else:
            keep current best
            log: strategy failed
    save iteration artifacts

export best_overlay.json
export calibration_summary.json
```

### 9.2 Resume Support

The orchestrator detects completed iteration artifacts and resumes from the
next incomplete iteration. Calibration log and current best are loaded from
disk.

---

## 10. Output Artifacts

### 10.1 Directory Structure

```
artifacts/calibration_runs/<run_id>/
├── calibration_log.json
├── calibration_summary.json
├── best_overlay.json
├── real_baseline_metrics.json
├── iterations/
│   ├── iter_00/
│   │   ├── overlay.json
│   │   ├── candidates/
│   │   │   ├── candidate_0/     # Full simulation output
│   │   │   ├── candidate_1/
│   │   │   └── ...
│   │   ├── metrics.json
│   │   ├── distances.json
│   │   ├── diagnosis.json
│   │   └── selection.json
│   ├── iter_01/
│   └── ...
```

### 10.2 Calibration Log Entry

```json
{
    "iteration": 3,
    "timestamp": "2026-04-27T14:32:01",
    "strategy_label": "deepen_threads_and_raise_edge",
    "strategy_description": "Increase reply preference and raise depth cap...",
    "diagnosis": { "repetitiveness": "...", "toxicity": "...", "structure": "..." },
    "overlay_diff_applied": { "config.depth_soft_cap": 4 },
    "candidates": [
        {
            "candidate_id": 0,
            "variant_type": "exact",
            "scores": {
                "d_total": 0.38,
                "d_repetitiveness": 0.29,
                "d_toxicity": 0.45,
                "d_structure": 0.40,
                "raw": {
                    "self_bleu_2": 0.19,
                    "toxicity_mean": 0.0005,
                    "avg_depth": 2.27
                }
            }
        }
    ],
    "selection": {
        "winner": 0,
        "beat_current_best": true,
        "previous_best_d_total": 0.42,
        "new_best_d_total": 0.38
    },
    "trajectory_so_far": [
        {
            "iteration": 0,
            "d_total": 0.55,
            "d_repetitiveness": 0.50,
            "d_toxicity": 0.58,
            "d_structure": 0.57,
            "raw_medians": { "self_bleu_2": 0.22, "toxicity_mean": 0.0003, "avg_depth": 1.9 },
            "real_medians": { "self_bleu_2": 0.05, "toxicity_mean": 0.003, "avg_depth": 1.69 }
        }
    ],
    "failed_strategies_so_far": ["flatten_structure_and_soften_tone"]
}
```

### 10.3 Final Summary

`calibration_summary.json`:

```json
{
    "total_iterations": 10,
    "total_candidates_evaluated": 50,
    "baseline_distances": { "d_total": 0.55, "d_repetitiveness": 0.50, "d_toxicity": 0.58, "d_structure": 0.57 },
    "baseline_raw": { "self_bleu_2": 0.22, "toxicity_mean": 0.0003, "avg_depth": 1.9 },
    "final_distances": { "d_total": 0.28, "d_repetitiveness": 0.25, "d_toxicity": 0.30, "d_structure": 0.29 },
    "final_raw": { "self_bleu_2": 0.08, "toxicity_mean": 0.002, "avg_depth": 1.75 },
    "real_medians": { "self_bleu_2": 0.05, "toxicity_mean": 0.003, "avg_depth": 1.69 },
    "improvement_pct": { "d_total": 49.1, "d_repetitiveness": 50.0, "d_toxicity": 48.3, "d_structure": 49.1 },
    "best_iteration": 8,
    "successful_strategies": ["deepen_threads_and_raise_edge", "diversify_motivation_mix"],
    "failed_strategies": ["flatten_structure_and_soften_tone"],
    "trajectory": []
}
```

### 10.4 Exportable Overlay

`best_overlay.json` — the winning overlay config, usable in future runs:

```bash
python run_discussion.py products.json --overlay artifacts/calibration_runs/<run_id>/best_overlay.json
```

---

## 11. CLI

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

## 12. Pipeline Modifications

The existing pipeline requires light changes to support the `--overlay` flag:

1. **`run_discussion.py`** — add `--overlay` argument. Load overlay JSON and
   pass it to `analyze_products`, `generate_personas`, `build_config`.
2. **`persona_gen.py`** — read persona distribution overrides from overlay
   (conflict_style, motivation, knowledge_style, stance distributions).
3. **`config_builder.py`** — read config knob overrides from overlay
   (activity ranges, time multipliers, depth cap).
4. **`oasis_reddit.py`** — read prompt text overrides from overlay
   (anti-paraphrase, tone guidance, structure preference text).

Each module falls through to its current hardcoded defaults when no overlay
is provided. Existing behavior is preserved exactly when `--overlay` is
absent.
