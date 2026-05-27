# 50-Thread Comparison: MiMo v2.5-pro vs SenseNova 6.7-flash-lite

**Date**: 2026-05-23
**Domain**: credit_cards
**Baseline**: 2,650 real Reddit threads
**Device**: Apple Silicon MPS

## Test Overview

| Metric | MiMo v2.5-pro | SenseNova 6.7-flash-lite |
|--------|---------------|--------------------------|
| Planned rounds | 95 | 70 |
| Successful threads | 47 (49%) | 53 (76%) |
| Failed (JSON parse) | 37 (39%) | 17 (24%) |
| API | token-plan-cn.xiaomimimo.com | token.sensenova.cn |
| Runtime | ~110 min | ~90 min |

## Core Metrics (15 metrics, credit_cards)

| Metric | Real | MiMo | SenseNova | MiMo Δ | SN Δ | Better |
|--------|------|------|-----------|--------|------|--------|
| avg_depth | 1.944 | 1.000 | 1.000 | +0.826 | +0.826 | Tie |
| structural_virality | 1.695 | 0.000 | 0.000 | +0.826 | +0.826 | Tie |
| length_cv | 0.737 | 0.336 | 0.299 | +0.717 | +0.740 | Tie |
| self_bertscore_mean_f1 | 0.463 | 0.511 | 0.519 | -0.474 | -0.624 | MiMo |
| obscene_mean | 0.0098 | 0.0001 | 0.0000 | +0.532 | +0.575 | Tie |
| self_bleu_4 | 0.033 | 0.038 | 0.042 | -0.385 | -0.528 | MiMo |
| toxicity_mean | 0.019 | 0.002 | 0.002 | +0.447 | +0.453 | Tie |
| severe_toxicity_mean | 0.0001 | 0.0000 | 0.0000 | +0.497 | +0.468 | Tie |
| polite_rate | 0.338 | 0.317 | 0.164 | +0.025 | +0.427 | MiMo |
| emotion_entropy | 1.087 | 0.866 | 0.743 | +0.288 | +0.372 | MiMo |
| neutral_rate | 0.175 | 0.143 | 0.308 | +0.104 | -0.409 | MiMo |
| mean_story_probability | 0.178 | 0.057 | 0.096 | +0.522 | +0.319 | SenseNova |
| threat_mean | 0.0003 | 0.0000 | 0.0000 | +0.388 | +0.304 | SenseNova |
| impolite_rate | 0.377 | 0.427 | 0.445 | -0.125 | -0.150 | Tie |
| semantic_mean_cosine | 0.301 | 0.312 | 0.321 | -0.048 | -0.091 | Tie |

## Distributional Match (p > 0.05 vs real Reddit)

| Model | Matched Metrics | Rate |
|-------|-----------------|------|
| MiMo v2.5-pro | polite_rate (p=0.759), neutral_rate (p=0.204), impolite_rate (p=0.128), semantic_mean_cosine (p=0.559) | 4/15 (27%) |
| SenseNova 6.7-flash-lite | impolite_rate (p=0.061), semantic_mean_cosine (p=0.255) | 2/15 (13%) |

## Statistical Summary

| Measure | MiMo | SenseNova | Winner |
|---------|------|-----------|--------|
| Avg Wasserstein distance | 0.2527 | 0.2691 | MiMo |
| Avg MWU p-value | 0.008 | 0.012 | MiMo |
| Distributional match rate | 27% | 13% | MiMo |
| Head-to-head wins | 7 | 2 | MiMo |

## Key Findings

1. **Both models produce flat threads**: avg_depth = 1.0 (real = 1.94), structural_virality = 0 (real = 1.7).
2. **MiMo is more naturalistic**: matches real Reddit on politeness, tone, and semantic similarity (4 distributional matches vs 2).
3. **SenseNova is more story-like**: better story probability (Δ=0.32 vs 0.52) and lower threat scores.
4. **Both lack lexical diversity**: self_bleu_4 higher than real (0.038/0.042 vs 0.033).
5. **JSON parsing failures**: MiMo 39% failure rate, SenseNova 24% — both fail on missing commas in nested JSON output.

## Files

| File | Description |
|------|-------------|
| `mimo-v2.5-pro/thread_scores.csv` | 47 threads × 57 metrics |
| `mimo-v2.5-pro/mirobench_comparison.csv` | 15 core metrics statistical comparison |
| `sensenova-6.7-flash-lite/thread_scores.csv` | 53 threads × 57 metrics |
| `sensenova-6.7-flash-lite/mirobench_comparison.csv` | 15 core metrics statistical comparison |
