# Calibration System Design

**Date:** 2026-04-26  
**Revised:** 2026-05-04  
**Status:** Active  
**Scope:** Persona + prompt calibration for the Reddit discussion simulator

---

## 1. Goal

The current calibration system tries to make simulated discussion threads match
the real discussion distribution more closely on a fixed set of target
metrics.

It does **not** rewrite simulator source code during the calibration loop.

Instead, it performs an outer-loop search that:

1. reuses or computes a real/vanilla baseline
2. diagnoses mismatch against the **real validation split**
3. proposes candidate prompt patches
4. runs fresh simulations for those candidates
5. ranks candidates using per-metric distributional comparison statistics
6. commits the best phase-local patch into a cumulative overlay
7. runs a fresh post-calibration evaluation against the **real test split**

The core objective is:

- make generated thread-level metric distributions look like real validation
- push per-metric distributional comparison statistics toward `0`
- push per-metric `MWU` / `KS` p-values upward toward `> 0.05` when possible

---

## 2. Current Architecture

### 2.1 High-level pipeline

```text
real train / val / test
        |
        v
Phase 0: before-calibration baseline comparison
         (reuse existing vanilla scores, or optionally rerun fresh vanilla_oasis)
        |
        v
Phase 1: iterative calibration loop
        |
        |-- LLM 1: strategist / reasoner
        |-- LLM 2: text materializer / revision
        |-- candidate simulations through the vanilla MiroFish runtime
        |-- per-metric validation scoring
        |-- block-best accumulation into one cumulative overlay
        v
Phase 2: fresh after-calibration evaluation vs real test
        |
        v
Phase 3: before/after improvement analysis
```

### 2.2 Runtime backbone

The system now uses the restored vanilla MiroFish/OASIS Reddit runner for all
simulation runs. The old repo-local `geo_patched` runtime layer was removed
because the current run path did not execute it; calibration now changes only
the persisted persona/prompt overlay and post-generation revision inputs.

---

## 3. What Calibration Is Allowed to Modify

### 3.1 Two persisted slots

Calibration is currently allowed to edit exactly **two persisted overlay
fields**:

- `persona.generation_guidance`
- `prompt.comment_style_guidance`

These are not tiny knobs. They are the actual prompt patches injected into the
simulator.

### 3.2 What these two slots mean

`persona.generation_guidance` controls **who gets generated**:

- backgrounds
- recurring grievances
- knowledge level
- confidence mismatch
- product memories
- repeated biases
- lived experience hooks
- what tends to trigger a reply

`prompt.comment_style_guidance` controls **how they write**:

- comment shape
- reply behavior
- anecdote usage
- disagreement style
- length variation
- paraphrase avoidance
- visible-comment reaction behavior
- short-form social-media reply patterns

### 3.3 What the calibration LLM does not edit

The calibration loop does **not** currently let the LLM mutate:

- Python source code
- arbitrary runtime code paths
- arbitrary new executable fields
- the runtime structure defaults directly

Runtime structure realism is handled in the patched simulator code, not by the
calibration LLM inventing new runtime fields.

---

## 4. Real-data splits and their roles

### 4.1 Train split

Used only for qualitative grounding:

- few-shot thread examples
- real sample threads shown to the strategist/materializer
- concrete examples of how real discussions sound

It is **not** the ranking target for candidate selection.

### 4.2 Validation split

Used for **Phase 1 candidate ranking**.

This is the active reference distribution during calibration.

### 4.3 Test split

Used only for **fresh post-calibration evaluation** after the loop finishes.

---

## 5. Manual Calibration Schedule

The current active search policy is a **fixed 12-iteration manual-phase
schedule**.

There is no longer an active exploration/combination-heavy/final-integration
schedule in the current implementation.

### 5.1 Current 12-iteration block structure

The 12 edited iterations are partitioned into four deterministic 3-iteration
blocks:

1. `iter 1-3`: `story_persona_foundation`
2. `iter 4-6`: `diversity_style_decompression`
3. `iter 7-9`: `length_distribution_rebalancing`
4. `iter 10-12`: `conflict_tone_activation`

### 5.2 Current focus metrics by block

`story_persona_foundation`

- `mean_story_probability`

`diversity_style_decompression`

- `self_bleu_4`
- `self_bertscore_mean_f1`
- `semantic_mean_cosine`

`length_distribution_rebalancing`

- `length_cv`

`conflict_tone_activation`

- `toxicity_mean`
- `severe_toxicity_mean`
- `obscene_mean`
- `threat_mean`
- `aggression_score_mean`

### 5.3 Protected metrics

Each later block also protects the focus metrics already improved in earlier
blocks.

Example:

- the diversity block focuses on semantic repetition metrics
- but also preserves the earlier story block’s `mean_story_probability`

---

## 6. Block-wise Accumulation

### 6.1 Within-block behavior

Each 3-iteration block maintains its own `block_best`.

That means:

- `iter 2` compares against the current story-block incumbent, not a global
  exploration frontier
- `iter 5` compares against the current diversity-block incumbent
- `iter 8` compares against the current length-block incumbent
- `iter 11/12` compare against the current conflict-block incumbent

### 6.2 End-of-block commit

At the end of a block, the block-best overlay is **committed into the
cumulative overlay state**.

Later blocks always start from:

- the cumulative committed overlay
- then append new block-specific prompt text on top of it

### 6.3 Append-only text merge

The manual-phase merge policy is append-oriented for the two text knobs:

- earlier block text is preserved
- later block text is appended
- later blocks do not replace the earlier blocks wholesale

This means the calibration search is now structured as:

```text
story best
  + diversity best
  + length best
  + conflict best
```

rather than a single global-best-only overlay race.

### 6.4 Structured phase blocks inside the overlay

The cumulative overlay is now stored in a **structured phase-block form**
internally.

Instead of keeping one undifferentiated long text blob, manual-phase writes are
stored under an internal `_manual_phase_blocks` map such as:

- `story_persona_foundation`
- `diversity_style_decompression`
- `length_distribution_rebalancing`
- `conflict_tone_activation`

Each block stores its own:

- `phase_label`
- `phase_order`
- `persona.generation_guidance`
- `prompt.comment_style_guidance`

At runtime, these blocks are rendered back into the two real text knobs as
labeled sections, for example:

```text
[Story / Persona Foundation]
...

[Diversity / Style Decompression]
...
```

This gives later phases a clearer prompt structure and reduces the chance that
the model treats the entire cumulative overlay as one flat wall of text.

### 6.5 What later blocks actually inherit

Because of the structured overlay representation:

- `iter 4-6` start from the committed `story` block
- `iter 7-9` start from committed `story + diversity`
- `iter 10-12` start from committed `story + diversity + length`

So the search still uses block-wise accumulation, but the inherited state is
now better organized for the LLM.

---

## 7. Two-stage LLM loop

### 7.1 LLM 1: strategist / reasoner

The strategist decides:

- what the active block is failing on
- which causal mechanism should be tried next
- which 5 candidate directions to test

It sees:

- current cumulative / block-best overlay
- current diagnostic
- real validation summary
- same-block prior candidate outcomes
- failed strategies
- few-shot real threads
- current simulated thread examples
- active phase instructions
- per-metric statistic feedback from earlier candidates in the same block

### 7.2 LLM 2: text materializer / revision

The materializer writes the actual injected prompt text.

It sees essentially the same evidence as the strategist, plus:

- the strategist diagnosis
- the 5 candidate seeds themselves

The materializer is not a fallback expander. It is expected to:

- rewrite the strategist seed into better final runtime text
- produce complete text for both persisted knobs
- preserve the active block’s causal intent

### 7.3 Both knobs are always edited

Every manual-phase candidate must write both:

- `persona.generation_guidance`
- `prompt.comment_style_guidance`

There is no current one-sided candidate type in the active schedule.

One side can be dominant for a block, but both are always present.

---

## 8. Prompt philosophy in the current system

### 8.1 The system tells the LLM how to modify, not just what metric is bad

The reasoner/materializer prompts now explicitly encode:

- the active phase objective
- per-metric interpretation guidance
- what behavioral mechanisms should change
- what anti-patterns to avoid
- how previous same-block candidates performed
- what real few-shot examples actually look like

The LLM is not expected to infer calibration strategy from metric names alone.

### 8.2 Diversity and conflict are now explicitly more aggressive in calibration text

The current calibration prompts explicitly allow or encourage, where relevant:

- short social-media-shaped replies
- clipped fragments
- lowercase starts
- no-subject replies
- abrupt openings
- sharper disagreement
- less validation-first politeness
- less empathy-opener repetition

Importantly, these style goals now live in:

- `persona.generation_guidance`
- `prompt.comment_style_guidance`

and **not** in the core runtime prompt scaffolding.

### 8.3 Few-shot examples are part of the editing logic

Both strategist and materializer are told to inspect few-shot real threads and
use them as concrete style and structure references.

The intended behavior is:

- look at how real story usage appears
- look at how real disagreement appears
- look at how real short replies differ from long comments
- then encode that behavior into the two editable text slots

---

## 9. Current scoring logic

### 9.1 Active metrics are ranked per metric, not by block-average first

The current manual-phase scorer does **not** collapse the active block into one
average score before selection.

Instead, it compares candidates **metric by metric** inside the active focus
set.

### 9.2 Primary ranking statistics

For each active metric, the winner ranking prioritizes:

1. `Wasserstein distance`
2. `quantile_error`
3. `empirical_fail_rate`
4. `abs_median_gap`
5. `abs_cliffs_delta`
6. `MWU p-value`
7. `KS p-value`
8. `out_of_range`
9. `percentile_distance`
10. `abs_raw_robust_z`

This is a **distance-first, p-value-second, MAD/percentile-third** ordering.

### 9.3 Why p-values are not first

`MWU` / `KS` p-values are important, but in small Phase 1 candidate batches
they are treated as supporting evidence rather than the primary optimization
signal.

The current system explicitly treats:

- `Wasserstein`
- `quantile_error`
- `empirical_fail_rate`
- `abs_median_gap`
- `abs_cliffs_delta`

as the more reliable direction-of-improvement statistics for candidate ranking.

### 9.4 Robust diagnostics are still kept

The system still computes robust small-batch diagnostics such as:

- `out_of_range`
- `percentile_distance`
- `robust_z`

These remain useful as direction checks and sanity checks, but they are now
secondary in the manual-phase ranking order.

### 9.5 Protected metrics are checked after focus metrics

Once the active focus metrics are compared, the scorer checks whether the
candidate preserved earlier-block gains on protected metrics.

### 9.6 All-target snapshots are recorded per iteration winner

In addition to the active block’s focused metrics, every iteration winner now
gets a full **12-target-metric snapshot** recorded in logs/artifacts.

For each winner, the system stores for all 12 tracked metrics:

- `wasserstein`
- `quantile_error`
- `empirical_fail_rate`
- `abs_median_gap`
- `abs_cliffs_delta`
- `MWU p-value`
- `KS p-value`

This is written to:

- `iter_xxx/winner_target_metric_eval.json`
- `calibration_log.json`
- `diagnosis.json`

This makes it possible to inspect whether a candidate that won on its focused
metrics was already quietly damaging non-focused metrics in the same iteration.

### 9.7 Protected-metric non-regression guard

Later blocks are also checked by a **protected-metric regression guard**.

The guard does not require every earlier metric to keep strictly improving, but
it prevents obvious backsliding such as:

- large increases in `wasserstein`
- large increases in `quantile_error`
- large increases in `empirical_fail_rate`
- large increases in `abs_median_gap`
- large increases in `abs_cliffs_delta`
- losing a previously achieved `MWU > 0.05`
- losing a previously achieved `KS > 0.05`

Candidates that violate these protected-metric guardrails are ranked behind
candidates that preserve earlier block wins.

### 9.8 Final evaluation is still a fresh after-calibration run

The Phase 1 validation ranking is not the final result.

Final reporting still comes from:

- fresh post-calibration simulation
- comparison against the real test split
- before/after improvement analysis

---

## 10. Runtime vs calibration responsibility

The vanilla MiroFish runtime handles simulation execution. The calibration layer
handles the current adjustable behavior:

- tone
- anecdote frequency
- short-form style
- conflict intensity
- grammar roughness
- anti-template behavior

Keeping runtime behavior in MiroFish avoids carrying a second, unused 2000-line
patch layer in this repository.

---

## 11. Same-block historical feedback

The current system explicitly shows later iterations in a block what earlier
candidates in that same block did.

For example:

- `iter 2` sees `iter 1`
- `iter 3` sees `iter 1-2`
- `iter 5` sees `iter 4`
- `iter 6` sees `iter 4-5`

The prompt includes per-candidate, per-metric statistics for earlier
same-block candidates.

Each statistic is labeled relative to the current block incumbent as:

- `improved`
- `worsened`
- `mixed`

This is done for:

- `W`
- `Q`
- `fail`
- `|med|`
- `|cd|`
- `mwu_p`
- `ks_p`
- `oor`
- `pct`
- `raw_z`

The intended behavior is:

- reuse mechanisms that helped
- weaken or avoid mechanisms that hurt
- salvage partially useful directions from losing candidates

---

## 12. Logging and observability

The current system is much more explicit about what happened at each step.

Per iteration, it records:

- `reasoner_prompt.txt`
- `materializer_prompt.txt`
- raw strategist/materializer responses
- `diagnosis.json`
- candidate overlays
- candidate simulation outputs
- per-candidate scoring
- active watch metrics for the block
- winner selection details

Terminal output also prints:

- active block label
- focus metrics
- protected metrics
- current block-incumbent metric statistics
- candidate ranking tables
- winner metric statistics

This is meant to support direct debugging of:

- why a candidate won
- what the LLM was told
- which mechanism changed which metric

---

## 13. Resume and overlay-only evaluation

### 13.1 Resume

Resume is state-based.

The system persists:

- completed iteration count
- current cumulative overlay
- current block-best state
- current diagnostics
- candidate directories
- frontier/history/log artifacts

### 13.2 Overlay-only evaluation

The CLI also supports skipping calibration entirely and directly evaluating a
given overlay:

- `--evaluate-overlay-json`

This is useful for:

- evaluating one historical candidate overlay
- testing a chosen phase-end overlay such as `iter_012` winner

It can also reuse an existing Phase 0 result:

- `--before-group-eval-json`

so that only post-calibration evaluation and before/after comparison are
rerun.

---

## 14. Output artifacts

Each run writes a timestamped calibration directory under the configured output
root.

Important artifacts include:

- `calibration_state.json`
- `calibration_log.json`
- `best_overlay.json`
- `before_calibration_group_eval.json`
- `after_calibration_group_eval.json`
- `before_after_improvement_summary.json`
- `calibration_summary.json`
- per-iteration `diagnosis.json`
- per-candidate simulation outputs
- per-iteration `reasoner_prompt.txt`
- per-iteration `materializer_prompt.txt`

The most reusable artifact remains:

- `best_overlay.json`

---

## 15. CLI contract

The current default manual run looks like:

```bash
python3 -m calibration.cli \
  --real-train-csv ... \
  --real-val-csv ... \
  --real-test-csv ... \
  --vanilla-scores-csv ... \
  --few-shot-dir ... \
  --products-json ... \
  --iterations 12 \
  --seed-posts 4 \
  --calibration-model gpt-5-mini \
  --calibration-rounds 12 \
  --metric-parallel 3
```

Important notes:

- `train` is qualitative context only
- `validation` is the during-calibration ranking target
- `test` is the final fresh evaluation target
- all simulations use the vanilla MiroFish/OASIS runtime
- a fresh Phase 0 baseline can still be recomputed when requested

---

## 16. Current design summary

The current calibration system is best described as:

> a deterministic 12-iteration, block-wise accumulated, two-stage LLM prompt
> calibration loop that edits exactly two persisted text slots
> (`persona.generation_guidance` and `prompt.comment_style_guidance`), runs
> candidate simulations under GEO’s patched visible-comment Reddit runtime,
> ranks candidates per metric using distribution-first validation statistics,
> commits one block-best overlay at the end of each phase, and finally tests
> the resulting cumulative overlay on fresh post-calibration simulations
> against the real test split.

That is the current implemented architecture.
