# v97 worklog — keyboard surface and measured joints

Date: 2026-08-19

Status: zero-API gate complete. One paid seed-2 content gate is next. N=10 is
blocked until that gate passes artifact and content review.

Policy: `generalized-card-v2-keyboard-surface-measured-joints-v97-20260819`

## What v96 actually delivered

The v96 ten-thread run is the first complete, honestly evaluable sample under
the new content policy: 532 generated against 532 matched real comments,
coverage 1.00, `$3.71`, 49 minutes. **6 of 12 metrics pass.**

Passing: `semantic_mean_cosine` (p 0.970), `structural_virality` (0.909),
`avg_depth` (0.850), `emotion_entropy` (0.678), `mean_story_probability`
(0.521), `length_cv` (0.076, marginal).

Failing: `self_bertscore_mean_f1` (0.0002, |d| 1.00), `impolite_rate` (0.001),
`polite_rate` (0.006), `neutral_rate` (0.007), `self_bleu_4` (0.009),
`hard_disagree_rate` (0.014).

v96's story and emotion arms worked. Those two now pass and had not before.

## Four measured causes, not one theory

Each was measured in the v96 artifact before any code changed.

1. **Typography.** Zero of 532 comments used an ASCII apostrophe; 389 used a
   typographic one. The self-BLEU tokenizer splits `it’s` into three tokens and
   `it's` into one, so every contraction contributed a shared trigram. Em dashes
   appear 187 times against 3 real. Replaying the v96 text through the new
   per-speaker keyboard draw with the real scorer moves `self_bleu_4` MWU p from
   0.009 to 0.273 and KS from 0.052 to 0.787, and lowers `self_bertscore` by
   about 0.008.
2. **The adjudication frame on every slot.** "The question your turn settles"
   rendered on 532 of 532 slots. Its "that's the part that actually matters"
   output is in 18.4% of v96 comments and effectively absent from real text, and
   it is worst on the least adjudicative functions: personal_datapoint 29.1%,
   reaction 19.0%, against question_followup 8.1%. Three earlier releases
   reworded it; none withheld it.
3. **An inverted tone-length joint with a correct marginal.** Planned targets
   were 0.311 polite / 0.442 impolite against a real 0.308 / 0.404, but the plan
   put impolite on 74% of 120-250 word slots and 100% of slots over 250 words,
   where excluded real comments of that size are 72% polite. Realized long
   comments came out 87% impolite and 9% polite against a real 27% and 71%.
4. **Long slots asked for a shape that does not exist.** The 250w+ slots
   realized 0.61x their matched length and the 845-word slot 0.32x. It was not
   the token budget (1,500 tokens allowed) but the request: that slot was asked
   for one thesis in 40 beats, and the Planner saturates near nine beats however
   many are asked for. Real long comments are 6 paragraphs at the median and 14
   at p90, with lists in 12.6% and quoted excerpts in 26.7%; v96 had a blank
   line in 3.4% of comments against 33.8% real.

## Implementation

Three new focused modules and two gated renderers, all measured from
evaluation-excluded threads:

- `surface_typography.py` — per-class typographic/keyboard share, drawn once per
  speaker; applied in a wrapped `shape_writer_text_for_task` so validation, the
  thread ledgers, and persistence all see the same characters.
- `comment_structure.py` — paragraph, list, and quote layout per size band, with
  the paragraph count scaled inside the open-ended top band at its measured
  words per paragraph.
- `tone_length_fit.py` — P(tone | size band) plus an iterative proportional fit
  that keeps the template's tone counts exact.
- `semantic_realization.turn_settles_a_question` — gates the boundary line on
  both Writer paths.
- `long_form_planning` — beat ceiling 40 -> 12, with the minimum acceptable
  count capped where the Planner still delivers.

A min-cost assignment for the tone fit was implemented first and rejected: it
maximizes likelihood and lands in a corner, giving 100% polite in the top band
against a measured 72%. The proportional fit reproduces the measured dependence
without inventing a sharper one.

## Arms

| flag | v97 default | reproduces v96 |
|---|---|---|
| `--reddit-typography` | `on` | `off` |
| `--turn-frame` | `adjudicative_only` | `universal` |
| `--tone-length-fit` | `conditional` | `median` |
| `--long-form-layout` | `measured` | `beats_only` |

All four are written into `run_config.json` and verified by the resume-config
check. Domain profile schema 11 -> 14.

## Two defects the tests caught

- Without a measured conditional the tone fit fell back to a uniform prior,
  which removes the length dependence entirely and is worse than the median
  heuristic it replaces. It now falls back to the heuristic and records
  `median_no_profile`.
- The arm is re-read from the environment on every `configure_generator_backend`
  call, so selecting it in-process is not enough; the `universal` regression has
  to set the environment variable.

## Zero-API result

- 369 generalized-card tests pass, including the writer-prompt gate rendered
  through the configured backend rather than by calling it directly.
- Ruff clean over the whole tree.
- Source contract 98/98, no untracked active source, no unpinned local import,
  zero drift.
- Active and active-plus-legacy parity both healthy.
- Backend self-test passes with all four v97 arms.
- Domain profile rebuilds at schema 14 over 424 excluded threads, 0 seed
  overlap, all three new profiles available.
- Active shaper exercised directly: realized typographic apostrophe share 0.240
  over 200 speakers against the measured 0.271, four classes drawing
  independently.
- Exact seed-2 `--prepare-only` passed as
  `generalized_card_camera_gpt54_v97_keyboard_seed2_20260819_preflight_v1`; no
  API call.

## Verification gate before N=10

One paid seed-2 gate, then read every comment. The specific things to check,
because each is a prediction this version makes:

1. Keyboard punctuation share near 0.27 of apostrophe-bearing comments, not 1.0.
2. `that's the part / bit / only` at or near zero.
3. The 250w+ slots at a realized ratio well above 0.61, with blank lines.
4. Planned polite landing on long slots and realizing as polite.
5. Distinct 3-word openers above 0.77, polarity-token openers near 0.07.
6. `mean_story_probability` and `emotion_entropy` still in range — both pass now
   and four arms are changing the register they depend on.

n=1 MWU/KS values remain descriptive only.

## Paid seed-2 gate result — 2026-08-19

Tag `generalized_card_camera_gpt54_v97_keyboard_seed2_20260819_v1`: 45/45
comments in one attempt, 89 requests, `$0.3883`, 235 seconds, no retries or
degraded comments. All six predictions held. Full table in
`generalized_card/VERSION_LOG.md`; the headline numbers on the same thread across
three policies are self-BLEU 0.0350 -> 0.0306 -> **0.0273** against a real 0.0268,
and self-BERTScore 0.5306 -> 0.5299 -> **0.5074** against a real 0.4892.

Still open and not addressed by v97: `impolite_rate` realization (0.614 against a
real 0.222 with a correct 0.370 target), concreteness (domain vocabulary 0.156
against 0.556, digits 0.356 against 0.600, 10 distinct model designators against
40), bare declarative endings (0.044 against 0.244), and no `!` at all against a
real 0.044.

N=10 started as `generalized_card_camera_gpt54_v97_keyboard_n10_20260819_v1` with
the four arms at their v97 defaults.

## Held for the next version

A measured final-punctuation habit is written and tested but **not** in v97. It
adds `build_final_punctuation_profile` and `apply_final_punctuation_habit` to
`surface_typography`, and profile schema 15. Measured over 11,817 excluded
comments, the share of declarative endings left bare rather than ending in a
period is 0.533 micro, 0.308 short, 0.198 medium, 0.128 long, 0.092 very_long,
0.110 essay; generated output is at 0.044. Only a trailing period is dropped, at
a per-speaker draw against the band's rate, so a question mark or exclamation the
Writer chose is never touched. 8 focused tests.

It was reverted out of the tree because the N=10 run was already in flight and
every file it touches is hash-pinned; see the lesson dated 2026-08-19 on editing
a pinned core during a run. The working copy is preserved at
`scratchpad/v98_pending/` and should land as its own version with its own arm so
its effect is attributable.

## Two further causes measured during the N=10 run (read-only, held for v98)

**The boundary gate removed about half of the frame, not all of it.** Over the
first 250 v97 comments, the frame appears at 0.188 on slots that still receive
the boundary line (by design: correction, verdict, and advice turns) and at
**0.144 on slots that never see it**. Withholding the line moved
`personal_datapoint` from 0.291 to 0.141, which is the predicted halving, so the
gate works and there is a second source.

It is not the Planner's wording. Adjudicative phrasing appears in 9.2% of
`semantic_move` values, and the frame rate is *lower* on those slots (0.043) than
on the rest (0.167), so the Writer is not echoing the plan.

The second source is that **the focused Writer prompt has no mid-comment phrase
ledger**. `_focused_thread_ledger` renders comment openings, short utterances,
and semantic coverage. `semantic_realization.used_sentence_routes` and
`repeated_phrase_counts` — which are frequency-ranked, already bounded, and
would surface `that s the part that (used 7x)` — are only wired into the `full`
arm, and `focused` has been the active arm since v82. A late slot in the
91-comment thread was shown 24 openings and 21 short lines and nothing about the
phrase seven of its predecessors had already used.

Related: a new frame appeared that no ban would have caught, `and that was the
<part/point/only place>` in 8 comments, all of them `personal_datapoint` story
slots landing a narrative on a verdict.

**v98 candidates, in the order the evidence supports:**

1. Render a bounded, frequency-ranked reused-route ledger in the focused prompt.
   The functions exist; only the wiring is missing. v67 measured that a ban in
   190 prompts still produced 23 violations, so expect partial compliance, not
   zero.
2. The measured final-punctuation habit already written (see above).
3. `has a digit` 0.304 against 0.536 pooled. Before acting, note that the
   "has domain vocabulary" probe is unreliable here — see the lesson dated
   2026-08-19 — while the digit and distinct-designator counts are not.
4. `length_cv` is compressed from both ends, and the profile is now precise.
   Realized/target by matched target size over the first 250 v97 comments:

   | target | n | ratio |
   |---|---:|---:|
   | 1-7 w | 11 | 1.21 |
   | 7-10 w | 17 | **1.34** |
   | 10-15 w | 24 | **1.42** |
   | 15-25 w | 56 | 0.95 |
   | 25-40 w | 44 | 1.14 |
   | 40-200 w | 94 | 0.93-0.96 |
   | 200+ w | 4 | **0.71** |

   The middle is right; the two tails are not. A 11.5-word slot comes out at
   16.2 and a 271-word slot at 193.5. The short cue is currently only "Make one
   narrow local move and stop when that contribution is complete" plus "Do not
   pad past it", which is not enough for a model that will not write a
   seven-word comment.
