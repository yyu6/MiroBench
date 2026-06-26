# SenseNova 6.7-flash-lite on MiroBench (credit_cards)

**Date**: 2026-06-26
**Domain**: credit_cards
**Baseline**: 2,650 real Reddit threads
**Device**: Apple Silicon MPS

## Test Overview

| Metric | Value |
|--------|-------|
| Model | SenseNova 6.7-flash-lite |
| API endpoint | token.sensenova.cn/v1 |
| Usable threads | 200 |
| Scorers completed | 9/9 |
| Core metrics present | 16/16 |
| Stance/disagreement checkpoint | included |
| Leaderboard match | 2/16 metrics (12.5%) |

## Scoring Coverage

All required scorer outputs were generated and summarized into
`credit_cards/thread_scores.csv`:

1. Disagreement / stance checkpoint
2. Self-BLEU
3. Structure
4. Self-BERTScore
5. Semantic uniformity
6. Story detection
7. Emotion
8. Politeness
9. Toxicity / detoxify

The submitted CSV contains 200 usable thread rows plus the optional
`__summary_mean__` row. All 16 leaderboard core metrics are present with numeric
values, including `hard_disagree_rate`.

## Core Metrics (16 metrics, credit_cards)

| Metric | Real mean | SenseNova mean | MWU p-value | Cliff's delta |
|--------|-----------|----------------|-------------|---------------|
| avg_depth | 1.9440 | 1.0038 | 4.47e-85 | +0.822 |
| emotion_entropy | 1.0870 | 1.2944 | 1.08e-04 | -0.164 |
| hard_disagree_rate | 0.0665 | 0.1138 | 1.24e-15 | -0.315 |
| impolite_rate | 0.3765 | 0.6107 | 3.00e-56 | -0.668 |
| length_cv | 0.7374 | 0.4054 | 5.87e-54 | +0.655 |
| mean_story_probability | 0.1781 | 0.1435 | 0.318 | -0.042 |
| neutral_rate | 0.1755 | 0.2055 | 1.69e-09 | -0.253 |
| obscene_mean | 0.0098 | 0.0002 | 7.47e-05 | -0.168 |
| polite_rate | 0.3383 | 0.0584 | 3.12e-66 | +0.726 |
| self_bertscore_mean_f1 | 0.4633 | 0.5376 | 1.94e-85 | -0.829 |
| self_bleu_4 | 0.0325 | 0.0404 | 1.20e-27 | -0.461 |
| semantic_mean_cosine | 0.3010 | 0.3023 | 0.837 | -0.009 |
| severe_toxicity_mean | 0.00008 | 0.000003 | 2.20e-07 | -0.219 |
| structural_virality | 1.6954 | 0.0067 | 2.27e-85 | +0.823 |
| threat_mean | 0.00029 | 0.00038 | 1.62e-76 | -0.784 |
| toxicity_mean | 0.0188 | 0.0022 | 9.74e-05 | -0.165 |

## Distributional Match (p > 0.05 vs real Reddit)

| Model | Matched Metrics | Rate |
|-------|-----------------|------|
| SenseNova 6.7-flash-lite | mean_story_probability (p=0.318), semantic_mean_cosine (p=0.837) | 2/16 (12.5%) |

## Statistical Summary

| Measure | SenseNova |
|---------|-----------|
| Avg Wasserstein distance | 0.2680 |
| Avg MWU p-value | 0.072 |
| Distributional match rate | 12.5% |

## Key Findings

1. **Flat thread structure**: avg_depth = 1.00 and structural_virality = 0.0067, far below real Reddit.
2. **Content/story score closest to real**: mean_story_probability passes the leaderboard MWU threshold.
3. **Semantic uniformity closest to real**: semantic_mean_cosine also passes the leaderboard MWU threshold.
4. **Tone mismatch remains large**: polite_rate is lower and impolite_rate is higher than real Reddit.
5. **Toxicity is lower overall**: toxicity_mean is below the real reference, while threat_mean remains distributionally different.

## Files

| File | Description |
|------|-------------|
| `credit_cards/thread_scores.csv` | 200 usable threads with per-thread scores |
| `credit_cards/mirobench_comparison.csv` | Local comparison against the real `credit_cards` reference |
