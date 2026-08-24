# Manual Credit-Card Calibration Playbook

## Goal

Use manual prompt editing to push as many core metric `MWU p-value` results above `0.05` as possible, then improve `KS p-value` as a second pass.

This playbook is tuned for:

- domain: `credit_card`
- real target: `artifacts/baselines/credit_cards_20260426_164025/real/thread_scores_test.csv`
- editable knobs:
  - `persona.generation_guidance`
  - `prompt.comment_style_guidance`

## Core Metrics

The manual loop should only watch these 13 metrics:

1. `self_bleu_4`
2. `self_bertscore_mean_f1`
3. `semantic_mean_cosine`
4. `mean_story_probability`
5. `toxicity_mean`
6. `severe_toxicity_mean`
7. `obscene_mean`
8. `threat_mean`
9. `aggression_score_mean`
10. `length_std`
11. `length_cv`
12. `avg_depth`
13. `structural_virality`

## Current Manual Baseline

From the current manual `Sheet8` run against the local real credit-card test set:

- `MWU p > 0.05`: `3 / 13`
- `KS p > 0.05`: `0 / 13`

Current `MWU p > 0.05` metrics:

- `toxicity_mean`
- `threat_mean`
- `aggression_score_mean`

Everything else is still significant.

## Direction Map

This is the most important part of manual tuning. Do not optimize in the wrong direction.

### Too High Now

These manual outputs are currently higher than real and should usually be pushed down:

- `self_bleu_4`
- `self_bertscore_mean_f1`
- `semantic_mean_cosine`
- `mean_story_probability`
- `avg_depth`
- `structural_virality`

### Too Low Now

These manual outputs are currently lower than real and should usually be pushed up:

- `obscene_mean`
- `severe_toxicity_mean`
- `length_std`
- `length_cv`

### Already Near the Boundary

These are already the closest to clearing `MWU > 0.05`, so do not over-correct them:

- `toxicity_mean`
- `threat_mean`
- `aggression_score_mean`

## Manual Search Order

Do not try to fix all 13 metrics at once. Use this order.

### Pass 1: Semantic Decompression

Target:

- `self_bleu_4`
- `self_bertscore_mean_f1`
- `semantic_mean_cosine`

Intent:

- reduce paraphrase
- reduce angle overlap
- reduce evidence-mode repetition

Typical knob moves:

- more persona-level disagreement motives
- more distinct evidence modes
- more different biases and repeated grievances
- stronger runtime anti-paraphrase rule

### Pass 2: Length Shape Recovery

Target:

- `length_std`
- `length_cv`

Intent:

- widen short / medium / long mixture
- stop the thread from sounding uniformly medium-length

Typical knob moves:

- explicit length buckets
- allow some one-liners and some clearly longer comments
- force different comment shapes by persona type

### Pass 3: Story Throttling

Target:

- `mean_story_probability`

Intent:

- keep lived detail, but reduce over-storying

Typical knob moves:

- cap anecdote share
- shorten anecdotes to one or two sentences
- replace many anecdotes with math, fine print, or correction comments

### Pass 4: Structure Cooling

Target:

- `avg_depth`
- `structural_virality`

Intent:

- keep some replies, but avoid over-threading

Typical knob moves:

- lower reply-prone persona share
- only reply to explicit numeric claims, denial/approval datapoints, or strong verdicts
- avoid gratuitous quote-reply chains

### Pass 5: Guardrail Touch-Up

Target:

- `toxicity_mean`
- `severe_toxicity_mean`
- `obscene_mean`
- `threat_mean`
- `aggression_score_mean`

Intent:

- keep bluntness realistic without tipping into attacks

Typical knob moves:

- allow rare mild frustration words
- keep no-threat / no-identity-attack rule
- do not chase more toxicity just because it clears one metric

## Acceptance Rule

For each manual iteration:

1. First maximize the count of core metrics with `MWU p > 0.05`.
2. If that count ties, maximize the count with `KS p > 0.05`.
3. If that still ties, prefer the candidate that improves:
   - `self_bleu_4`
   - `semantic_mean_cosine`
   - `length_cv`
   - `length_std`
   - `mean_story_probability`
4. Use `Wasserstein`, `quantile error`, and `abs median gap` only as tie-breakers among still-significant metrics.

## Per-Metric Intervention Table

| Metric | If still too significant | Default manual move |
| --- | --- | --- |
| `self_bleu_4` | comments still lexically repetitive | force more evidence-mode switching and more persona-specific anchor tokens |
| `self_bertscore_mean_f1` | comments still semantically paraphrasing each other | diversify motives, product use-cases, and stance templates |
| `semantic_mean_cosine` | threads still feel like one consensus voice | add stronger disagreement archetypes and orthogonal evaluation criteria |
| `mean_story_probability` | too much anecdote | reduce anecdote quota and shorten anecdotes |
| `toxicity_mean` | too low or too polite | allow mild bluntness in a small minority of comments |
| `severe_toxicity_mean` | unrealistically sanitized | allow rare sharper complaint framing, still no threats |
| `obscene_mean` | language too clean | allow rare mild expletive or blunt adjective |
| `threat_mean` | do not optimize upward directly | keep at current level unless it collapses |
| `aggression_score_mean` | too soft / conflict-free | allow skeptical pushback and dismissive short replies |
| `length_std` | lengths too uniform | widen the absolute span between very short and long comments |
| `length_cv` | not enough relative spread | mix one-liners, medium advice, and a smaller set of long posts |
| `avg_depth` | too many reply chains | reduce reply-prone share and reply only when triggered |
| `structural_virality` | tree too branchy | fewer corrective chains and fewer follow-up replies |

## Manual Loop

For each iteration:

1. Edit both knobs.
2. State one main hypothesis only.
3. Predict 3 to 5 metrics that should improve.
4. Run the simulation and score the 13 metrics.
5. Log:
   - what changed in persona guidance
   - what changed in style guidance
   - which metrics moved the right way
   - which metrics regressed
6. Keep the edit only if it improves the acceptance rule above.

## Do Not Do These

- Do not add more anecdotes when `mean_story_probability` is already too high.
- Do not add more reply triggers when `avg_depth` and `structural_virality` are already too high.
- Do not make all personas data-heavy; that will often help story but hurt semantic spread.
- Do not chase toxicity metrics aggressively; they are already closest to passing.
- Do not change five mechanisms at once; isolate one main hypothesis per iteration.

## Starter Overlay

The first manual overlay to try is stored in:

- [manual-credit-card-overlay-v1.json](/Users/yaoningyu/Desktop/UIUC/GEO/docs/design/manual-credit-card-overlay-v1.json)

It is designed to:

- keep strong semantic dispersion
- reduce over-storying
- recover length spread
- cool down reply depth a bit
- preserve mild realistic bluntness

## Suggested Log Template

Use this per iteration:

| iter | main hypothesis | expected winners | expected risks | MWU > 0.05 count | KS > 0.05 count | keep? | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |  |  |

