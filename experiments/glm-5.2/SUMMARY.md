# GLM-5.2 on MiroBench (credit_cards)

**Date**: 2026-07-01
**Domain**: credit_cards
**Baseline**: 2,650 real Reddit threads
**Device**: Apple Silicon MPS

## Test Overview

| Metric | Value |
|--------|-------|
| Model | GLM-5.2 |
| Simulation engine | Single LLM via SenseNova API (no OASIS architecture) |
| API endpoint | SenseNova OpenAI-compatible API (`https://token.sensenova.cn/v1`) |
| Usable threads | 434 |
| Scorers completed | 9/9 |
| Core metrics present | 16/16 |
| Stance/disagreement checkpoint | included |
| Leaderboard match (MWU p>0.05) | 1/16 (6%) |

## Generation Provenance

Threads were generated with GLM-5.2 served through the SenseNova API
(`https://token.sensenova.cn/v1`) through the OpenAI-compatible chat
completions interface. This run did not use OASIS or an OASIS-derived agent
architecture; it used direct LLM prompting to generate Reddit-style posts and
comments. The client uses `SENSENOVA_API_KEY` for authentication. This
submission stores the generated threads and scorer outputs only; it does not
commit API keys or raw platform request logs.

## Scoring Coverage

1. ✅ Disagreement / Stance
2. ✅ Self-BLEU
3. ✅ Structure
4. ✅ Self-BERTScore
5. ✅ Semantic uniformity
6. ✅ StorySeeker
7. ✅ GoEmotions
8. ✅ Politeness
9. ✅ Detoxify

## Core Metrics

| Metric | Real Mean | Sim Mean | MWU p | Cliff's δ | Effect |
|--------|-----------|----------|-------|-----------|--------|
_(see mirobench_comparison.csv for full table)_

## Distributional Match

Metrics with `mwu_p_value > 0.05` (indistinguishable from real):
- `mean_story_probability` (p=0.561, negligible effect)

## Statistical Summary

| Measure | Value |
|---------|-------|
| Average Wasserstein distance | 0.2703 |
| Average MWU p-value | <0.05 |
| Match rate | 1/16 (6%) |
| Avg \|Cliff's δ\| | 0.400 |
| Effect breakdown | 2 neg / 5 small / 3 med / 6 large |

## Key Findings

GLM-5.2 generates flat discussion threads (avg_depth=1.0, structural_virality=0.0) — a known limitation shared by all current LLM-based generators. The model's self-BLEU scores (0.033) indicate moderate lexical diversity. Toxicity levels are very low (toxicity_mean=0.002), and politeness is high (polite_rate=0.53, impolite_rate=0.30). The model shows neutral emotional tone as the dominant emotion.

Story probability is the only core metric that passes the leaderboard MWU threshold. Thread structure remains a large mismatch: avg_depth and structural_virality are far below the real Reddit reference.

## Files

- `experiments/glm-5.2/credit_cards/thread_scores.csv`
- `experiments/glm-5.2/credit_cards/mirobench_comparison.csv`
