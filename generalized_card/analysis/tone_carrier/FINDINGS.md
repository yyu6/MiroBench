# The tone channel is one realization defect, and it is priced — 2026-08-25

Run `generalized_card_camera_gpt54_paper_20260825_v1`, N=50. Every harness
reproduces the shipped per-thread `polite_rate` **exactly** (max |diff| 0.00000,
both sides) before printing an edited number (E6).

## Why this and not `self_bleu_4`

Under Holm over the 24 tests (J2), acceptance is `min(p) > 0.05/24 = 0.00208`.
Five tests are below it today, and `self_bleu_4` is **not** one of them:

| test | p | |
|---|---:|---|
| polite_rate MWU | 0.00017 | 12x too small |
| self_bertscore KS | 0.00058 | |
| impolite_rate MWU | 0.00124 | |
| polite_rate KS | 0.00131 | |
| self_bertscore MWU | 0.00213 | on the line |
| *self_bleu_4 MWU / KS* | *0.0566 / 0.0392* | *passes Holm* |

## The assignment is right; the realization fails

The Planner assigns `tone_target=polite` to **588 of 1974 slots (29.8%)**
against real's `polite_rate` **0.3020**. Of the 578 that persisted, only
**235 (40.7%)** come out polite: 119 land `somewhat_polite`, **192 `impolite`**,
32 `neutral`.

## The carrier is an appreciative *evaluation*, not a thanks

Sentences drawn from real threads outside the 50 matched seeds that the shipped
classifier scores polite at P>0.90 look like:

> "Also the tilt and twist screen is so useful." ·
> "Remarkable quality for such a tiny combo." ·
> "I also loved their rangefinder style mirrorless bodies."

This resolves G53's paradox — the generator already writes gratitude at 1.58x and
affirm_other at 3.46x real's rate and gets no credit for it. The label is bought
by an appreciative statement **about the thing under discussion**.

## Priced, three ways (J7: each is an upper bound)

Prepending ONE real carrier sentence. `real` / `today` shown for reference.

| routing | polite MWU | impolite MWU | neutral MWU | self_bleu_4 MWU |
|---|---:|---:|---:|---:|
| today | 0.0002 | 0.0012 | 0.0137 | 0.0566 |
| random, 22% of all comments | **0.7122** | 0.1096 | **0.0010** ✗ | 0.1868 |
| **at comments that read impolite, 40%** | **0.6740** | **0.8254** | 0.0137 (unchanged) | **0.1566** |
| by Planner assignment, 70% of failing polite slots | **0.7328** | 0.0389 | **0.0015** ✗ | 0.0946 |

Levels at the best routing: `polite_rate` 0.1746 -> 0.2809 (real 0.3020),
`impolite_rate` 0.5349 -> 0.4258 (real 0.4220), `neutral_rate` untouched at
0.1143, `self_bleu_4` 0.0361 -> 0.0344 (real 0.0337).

**One mechanism takes the artifact from 5 failing Holm tests to 2, and both
survivors are `self_bertscore`.**

## The one open design question

`neutral_rate` is already short (0.1143 against 0.1577), so any carrier that
converts a neutral comment makes a failing metric worse. The routing that works
lands only where the comment would otherwise read impolite — but selecting on
the evaluation classifier's own label is forbidden (`ORIENTATION.md` s4:
distribution diagnostics never select a Writer candidate). The buildable
mechanism must reach those slots from Planner-side fields only. `stance`,
`comment_function`, `payload_type` and `real_tone_slot`-vs-`tone_target`
disagreement are all available on the slot and none has been tested yet.

Note also that the Planner assigns `impolite` to 912/1974 = 46.2% against real's
42.2%, so part of the impolite excess is in the assignment, not the realization.
