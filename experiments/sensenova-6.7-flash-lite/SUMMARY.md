# SenseNova 6.7-flash-lite on MiroBench (credit_cards)

**Date**: 2026-05-23
**Domain**: credit_cards
**Baseline**: 2,650 real Reddit threads
**Device**: Apple Silicon MPS

## Test Overview

| Metric | Value |
|--------|-------|
| Model | SenseNova 6.7-flash-lite |
| API endpoint | token.sensenova.cn/v1 |
| Planned rounds | 70 |
| Successful threads | 53 (76%) |
| Failed (JSON parse) | 17 (24%) |
| Runtime | ~90 min |

## Core Metrics (15 metrics, credit_cards)

| Metric | Real | SenseNova | Δ (Cliff's) |
|--------|------|-----------|-------------|
| avg_depth | 1.944 | 1.000 | +0.826 |
| structural_virality | 1.695 | 0.000 | +0.826 |
| length_cv | 0.737 | 0.299 | +0.740 |
| self_bertscore_mean_f1 | 0.463 | 0.519 | -0.624 |
| obscene_mean | 0.0098 | 0.0000 | +0.575 |
| self_bleu_4 | 0.033 | 0.042 | -0.528 |
| toxicity_mean | 0.019 | 0.002 | +0.453 |
| severe_toxicity_mean | 0.0001 | 0.0000 | +0.468 |
| polite_rate | 0.338 | 0.164 | +0.427 |
| emotion_entropy | 1.087 | 0.743 | +0.372 |
| neutral_rate | 0.175 | 0.308 | -0.409 |
| mean_story_probability | 0.178 | 0.096 | +0.319 |
| threat_mean | 0.0003 | 0.0000 | +0.304 |
| impolite_rate | 0.377 | 0.445 | -0.150 |
| semantic_mean_cosine | 0.301 | 0.321 | -0.091 |

## Distributional Match (p > 0.05 vs real Reddit)

| Model | Matched Metrics | Rate |
|-------|-----------------|------|
| SenseNova 6.7-flash-lite | impolite_rate (p=0.061), semantic_mean_cosine (p=0.255) | 2/15 (13%) |

## Statistical Summary

| Measure | SenseNova |
|---------|-----------|
| Avg Wasserstein distance | 0.2691 |
| Avg MWU p-value | 0.012 |
| Distributional match rate | 13% |

## Key Findings

1. **Flat thread structure**: avg_depth = 1.0 (real = 1.94), structural_virality = 0 (real = 1.7) — no reply chains generated.
2. **Story-like tendencies**: lower story probability than real (Δ=0.32) and lower threat scores (Δ=0.30).
3. **Tone mismatch**: polite_rate much lower than real (0.164 vs 0.338), neutral_rate higher (0.308 vs 0.175).
4. **Low lexical diversity**: self_bleu_4 higher than real (0.042 vs 0.033).
5. **JSON parsing failures**: 24% failure rate — missing commas in nested JSON output.

## Files

| File | Description |
|------|-------------|
| `thread_scores.csv` | 53 threads × 57 metrics |
| `mirobench_comparison.csv` | 15 core metrics statistical comparison |
