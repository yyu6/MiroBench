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

## The routing question, answered — 2026-08-25

Routing on the classifier's own label is forbidden (`ORIENTATION.md` s4). The
question was whether Planner-side fields carry the same signal. They do.

Within the 578 slots assigned `tone_target=polite` (base: impolite 0.332,
neutral 0.055), `tone_routing.py`:

| Planner field = value | n | impolite | neutral | lift |
|---|---:|---:|---:|---:|
| `story_mode = messy_multi_step_story` | 29 | 0.552 | **0.000** | 1.66 |
| `payload_type = personal_story` | 144 | 0.444 | **0.000** | 1.34 |
| `evidence_mode = firsthand_experience` | 148 | 0.432 | **0.000** | 1.30 |
| `story_mode = specific_personal_story` | 92 | 0.424 | **0.000** | 1.28 |
| `affect_role = admiration` | 113 | 0.416 | 0.018 | 1.25 |
| `comment_function = personal_datapoint` | 248 | 0.371 | 0.028 | 1.12 |
| `speaker_role = datapoint_only` | 259 | 0.371 | 0.031 | 1.12 |
| `surface_texture = abbrev_shorthand` | 223 | 0.372 | 0.027 | 1.12 |

A polite-assigned first-person experience slot reads impolite 42-55% of the
time and **never** reads neutral. That matters because `neutral_rate` is already
the shortest bucket (0.1143 against real 0.1577), so any carrier that converts a
neutral comment makes a failing metric worse.

Measured effect (`tone_rule_cf.py`, one real carrier sentence prepended, J7
upper bound):

| rule | routed slots | polite MWU | impolite MWU | neutral MWU | self_bleu_4 MWU |
|---|---:|---:|---:|---:|---:|
| today | — | 0.0002 | 0.0012 | 0.0137 | 0.0566 |
| narrow (the four zero-neutral values) | 148 (7.6%) | 0.0031 | 0.0054 | **0.0137 unchanged** | 0.0682 |
| **broad (all eight), at 100%** | **359 (18.4%)** | **0.2209** | **0.0364** | 0.0053 | 0.0830 |

Under Holm the bar is `p > 0.05/24 = 0.00208`. The broad rule clears it on all
four; the only Holm failures left would be `self_bertscore`'s two tests, which
is what v113 targets.

**Two risks to carry into the prediction.** The narrow rule costs `neutral_rate`
nothing but is too small to move `polite_rate`. The broad rule works but narrows
`neutral_rate`'s Holm margin from 6.6x to 2.5x, and N=150 multiplies power by
another sqrt(3) -- `neutral_rate` is the metric to watch, not `polite_rate`.

**An untested lever that would help instead of trading.** The Planner assigns
`impolite` to 912/1974 = **46.2%** of slots against real's 42.2%, so part of the
impolite excess is in the assignment rather than the realization. Correcting it
would relieve `impolite_rate` without spending any of `neutral_rate`.

**A predicate bug worth remembering.** The first version of this rule included
`story_mode.endswith("_story")`, which matches `no_story`. It silently collapsed
the rule to "every polite-assigned slot" and reproduced the earlier result
exactly, including the neutral damage the rule existed to avoid.

---

# CORRECTION — the carrier cannot be built as a cue (2026-08-26)

**Sections above over-read the ablation. Retracted here rather than left standing.**

The ablation prepended sentences the shipped classifier scores `polite` at
**P > 0.90**. A sentence the Writer produces instead is not that sentence. This
was audited before writing any code, and the mechanism does not survive it.

## 1. The existing register mechanism already works

`register_realization` (v101, on in the paper run) cues five positive-register
moves on polite slots. It is not inert — measured on the N=50 artifact's own
saved prompts, **81.6% of polite-assigned slots receive at least one cue** and
compliance is large:

| move | cued slots | realized \| cued | realized \| not cued | lift |
|---|---:|---:|---:|---:|
| own_thing | 210 | 0.662 | 0.098 | **6.76** |
| gratitude | 79 | 0.544 | 0.110 | **4.95** |
| any_intensifier | 283 | 0.675 | 0.272 | 2.48 |
| plain_verdict | 219 | 0.475 | 0.217 | 2.19 |
| love_like | 79 | 0.063 | 0.045 | 1.40 |

## 2. Prevalence is already matched; conversion is short everywhere

The module deliberately omits `intensified_positive` as "the same construction
twice". Tested directly — intensifier and positive verdict inside one sentence:

| bucket | real P(polite) | gen P(polite) | gen/real | real prev | gen prev |
|---|---:|---:|---:|---:|---:|
| conjunction, same sentence | 0.637 | 0.365 | **0.57** | 0.047 | **0.044** |
| both, different sentences | 0.595 | 0.419 | 0.70 | 0.043 | 0.032 |
| verdict only | 0.444 | 0.389 | 0.88 | 0.087 | 0.058 |
| intensifier only | 0.251 | 0.156 | 0.62 | 0.237 | 0.197 |
| neither | 0.146 | 0.097 | 0.66 | 0.587 | 0.670 |

Conjunction prevalence is **0.93x** — the generator is not under-producing it.
Every bucket converts at 0.57–0.88x. **Another surface-move cue cannot work**,
and that includes the carrier as designed: cueing the 359 routed slots into the
conjunction bucket would give them generated's own conjunction conversion of
0.365, against the 0.404 those slots already reach. Zero expected gain.

## 3. Hedging is not it either

| | real | generated |
|---|---:|---:|
| comments carrying a hedge | 0.206 | **0.186** |
| hedges per 100 words | 0.60 | **0.61** |

The generator hedges *less* than real. Stripping every hedge from generated
output moves `polite_rate` 0.147 -> 0.154 — **17 comments, 0.87%, against the
9.1% the gap needs**.

## 4. What survives: conversion is a steep function of length

| words | real P(polite) | gen P(polite) | gen/real |
|---|---:|---:|---:|
| 0–14 | 0.151 | 0.172 | 1.14 |
| 15–29 | 0.112 | 0.079 | 0.70 |
| **30–59** | **0.233** | **0.082** | **0.35** |
| 60–119 | 0.412 | 0.242 | 0.59 |
| 120+ | 0.623 | 0.419 | 0.67 |

Real politeness rises steeply with length, 0.151 -> 0.623. The generator is
short (realized/assigned 0.891) *and* converts worse inside every band, so the
two deficits multiply. The collapse is sharpest at 30–59 words, which is inside
v112's target band.

## Status

**No tone mechanism is buildable on this evidence.** Three hypotheses are dead:
more register cues, the omitted conjunction, and hedging. `polite_rate` remains
the furthest metric from passing and has no candidate. What has not been tested
is whether length repair alone moves it — v112 is the arm that would answer it,
and that is measurable on the same run rather than as separate work.

---

# The tone channel, re-derived from the ground up — 2026-08-26

Runs: `v110_length_transfer_n10_20260824_v1` (arms off) and
`v113_v112_gate_n10_20260826_v1` (v112+v113), the SAME ten seeds 2-11, plus the
matched real threads. Confound recorded: v110 ran `length_transfer=refit` and the
gate ran `v97`; commit 21e793c measured the refit arm firing 532/532 and moving
nothing, so it is believed inert but is not a controlled variable.

Every section below is produced by a script in this directory.

## 1. The open question from s4 is answered: length is NOT the tone lever

`v112_length_conversion.py` decomposes the polite gap into band occupancy and
within-band conversion:

| | actual polite | at real's length spread | at real's conversion |
|---|---:|---:|---:|
| real | 0.2595 | | |
| gate | **0.1377** | 0.1386 | **0.2545** |

**Occupancy explains 0%. Conversion explains ~100%.** v112 is not a tone arm and
should not be argued as one. It did move realization of `tone_target=polite` from
0.368 to 0.400, which is real but an order of magnitude short.

The band table shows where: generated matches real exactly below 30 words and
collapses above it.

| words | real P(polite) | gate P(polite) |
|---|---:|---:|
| 0-14 | 0.117 | 0.124 |
| 15-29 | 0.088 | 0.078 |
| 30-59 | 0.271 | **0.078** |
| 60-119 | 0.404 | **0.214** |
| 120+ | 0.708 | **0.347** |

## 2. Polite is a per-sentence lottery. Only the per-sentence rate matters

`sentence_accumulation.py` scores every sentence independently. Observed
P(>=1 polite sentence) tracks `1-(1-r)^k` at a ratio of 0.85-1.05 on **both**
sides and in every band, so sentences are effectively independent draws and the
comment label is the max. Sentences per comment are already matched (real 3.84,
gate 3.22). The whole defect is `r`:

| words | real r | gate r |
|---|---:|---:|
| 0-14 | 0.106 | 0.119 |
| 15-29 | 0.098 | 0.063 |
| 30-59 | 0.098 | **0.020** |
| 60-119 | 0.092 | **0.037** |
| 120+ | 0.134 | **0.034** |

Real's `r` is flat at ~0.10 at every length and every position, and rises to
0.262 in the last decile of a long comment. Generated's collapses. This kills the
"add one carrier sentence" framing outright: a carrier raises `r` by `1/k`, which
is 0.09 in a 120+ word comment.

## 3. It is not the vocabulary. Two independent tests

Conditioned on a sentence containing the same `register_realization` move
(`sentence_move_conversion.py`, 30+ word comments):

| move | real prev | real conv | gen prev | gen conv | ratio |
|---|---:|---:|---:|---:|---:|
| any_intensifier | 0.190 | 0.165 | 0.153 | 0.045 | **0.27** |
| plain_verdict | 0.091 | 0.331 | 0.071 | 0.086 | **0.26** |
| own_thing | 0.096 | 0.200 | 0.077 | 0.090 | 0.45 |
| *no move at all* | — | *0.056* | — | *0.020* | *0.36* |

Same word, a quarter of the conversion. And `where_the_mass_is.py` fits the
polite-discriminative vocabulary on real sentences by log-odds and applies it to
both sides: **summed prevalence of the top 45 tokens is 0.251 in real and 0.285
in generated, a ratio of 1.14.** The generator already over-produces real's
polite vocabulary. No lexical cue can work, and `register_realization` is not the
place to look. That is now the fifth and sixth dead lexical hypothesis.

## 4. The frame effect is real, large per sentence, and too rare to matter

`bare_assertion_frame.py`. A retraction of my own reading: the vivid examples
("great and still says almost nothing", "a pretty weak compromise") do describe a
true effect, and it is not the answer.

| bucket | real P(polite) | gate P(polite) | n(real) |
|---|---:|---:|---:|
| bare positive assertion | 0.471 | 0.361 | 70 |
| positive + contrast | 0.203 | **0.031** | 64 |
| positive + condition | 0.333 | **0.028** | 24 |
| no positive word at all | 0.073 | 0.035 | **1664** |

Contrast costs 6.5x and condition 12x — but those buckets are 5% of sentences.
Giving generated real's bare-assertion prevalence at its own conversion moves
`r` 0.0434 -> 0.0488 against real's 0.1034: **9% of the gap.** 78% of real's
polite sentences carry no move word at all, and that bucket is where the mass is.

## 5. What survives: invert the confusion matrix from the ASSIGNMENT side

Every hypothesis above attacks realization. But the Writer's failure is
*consistent*, not random, and that makes it a measurable transfer matrix
(`assignment_inversion.py`, gate; the v110 matrix is within a few points of it,
so C is stable across generator versions):

| assigned \ realized | polite | somewhat | neutral | impolite | n |
|---|---:|---:|---:|---:|---:|
| polite | 0.400 | 0.186 | 0.076 | 0.338 | 145 |
| somewhat_polite | 0.087 | 0.348 | 0.109 | 0.457 | 46 |
| neutral | 0.103 | 0.103 | 0.436 | 0.359 | 78 |
| impolite | 0.011 | 0.034 | 0.084 | **0.870** | 261 |

`det(C^T)=0.0387`, `cond=5.2`. Solving `C^T a = target` and projecting onto the
simplex removes **89%** of the four-way L2 gap. Nothing about the Writer changes;
the Planner asks for the mix that comes out right. This is the same inverse
calibration `length_calibration` already performs for word counts, and it lands
in exactly one function: `generation_distribution._target_tone_counts`, which is
what renders the Planner's exact tone quota.

## 6. The Lucas critique against s5, and the honest range

`inversion_feasibility.py` and `stratified_inversion.py`. C is measured at
today's mix, and the solution needs polite on 55.4% of slots against 27.4%.

Reassuring: P(realize polite | assign polite) is stable across
`comment_function`, `payload_type`, `evidence_mode` and `speaker_role` —
0.310 to 0.474 around a 0.384 base.

Not reassuring: it is **not** stable across `stance`. 261 of 289 polite
assignments sit on `agree` slots and realize at 0.402; the 17 on `uncertain`
realize at **0.059**. `agree` is only 34.3% of slots, so ~21pp of new polite has
to land on `mixed`/`uncertain`/`neutral`, where the only observation is that
0.059. `prompts.py:951` forbids polite on a disagreeing stance outright.

Stratifying by stance, refusing any cell with n<12, and forbidding polite on
disagree gives a range rather than a point:

| regime | polite share asked | polite | somewhat | neutral | impolite | gap closed |
|---|---:|---:|---:|---:|---:|---:|
| today | 0.273 | 0.1294 | 0.1180 | 0.1454 | 0.6072 | 0% |
| optimistic (unobserved cells at the pooled rate) | 0.662 | 0.2409 | 0.1287 | 0.1615 | 0.4680 | **89%** |
| pessimistic (unobserved polite cells at 0.059) | 0.556 | 0.1895 | 0.1466 | 0.2017 | 0.4612 | **55%** |
| real target | — | 0.2595 | 0.1174 | 0.1591 | 0.4640 | |

Per metric under the pessimistic solution: `polite_rate` bias -50.1% -> **-27.0%**,
`impolite_rate` +30.9% -> **-0.6%**, `neutral_rate` -8.6% -> +26.8%.

**`impolite_rate` is fully repaired in both regimes and it is today's smallest
p-value.** `neutral_rate` is spent to buy it, which is the trade s4 of the earlier
section warned about, and it is the metric to watch.

## 7. The single largest observed lever, independent of the inversion

The Planner assigns `impolite` to **330 of 330 disagreeing slots**, and
`disagree` is 31.2% of all slots. That one cell contributes 0.312 x 0.870 = 0.271
of the realized 0.607 impolite -- **45% of it** -- against a total excess of only
0.143 over real. Neighbouring observed cells put a `neutral` assignment on a
non-agree slot at 0.39-0.42 impolite rather than 0.87.

Real Reddit disagreement is not uniformly impolite. Nothing in the code forces
this pairing: `prompts.py:951` forbids only `polite` on a disagreeing stance, and
the tone class is the Planner's own choice under a quota rendered by
`_target_tone_counts`. This is measurable and buildable without touching the
Writer at all.

## Status

`polite_rate` had no candidate mechanism at the end of the previous session. It
now has one, priced at 55-89% depending on cells that have never been observed,
with `impolite_rate` -- the smallest p-value -- repaired in both regimes and
`neutral_rate` as the new metric at risk.

---

# The calibration run answered the cap question — 2026-08-26

`v117_calibration_20260826_v1`, 10 threads / 559 slots / $3.93 / 65 min, on the
zero-overlap calibration pool under `--tone-quota calibrate`. The flat quota did
what it was built for: assignment came out **137 / 137 / 140 / 145**, so every
cell of C has n>=137 instead of the old 289 / 92 / 156 / 522.

    REALIZATION_MATRIX refit (n=559)          shipped (n=1059, skewed mix)
    polite     .3942 .1971 .1022 .3066        .3841 .1938 .0900 .3322
    somewhat   .1241 .3650 .1460 .3650        .0761 .4130 .0978 .4130
    neutral    .1357 .1286 .2429 .4929        .0897 .0962 .4103 .4038
    impolite   .0069 .0069 .0966 .8897        .0096 .0307 .1054 .8544

## The Lucas critique is answered for the polite row

`tone_realization.POLITE_ASSIGNMENT_CAP` was pinned at 0.35 because
P(realize polite | assign polite) had only been observed on `agree` slots. Under a
flat quota the polite assignment spans every stance, and the rate is **0.3942**
against the 0.3841 measured at the old skewed mix. **C is approximately invariant
to the assignment mix**, which is what the cap existed to doubt. The cap can be
raised on evidence.

## But the neutral row is not stable, and it is now the binding constraint

`neutral -> neutral` fell **0.4103 -> 0.2429** and `neutral -> impolite` rose
0.4038 -> 0.4929 between the two fits, on comparable n (156 vs 140). That is the
one row that does not transfer, and it changes the answer:

| cap | polite | somewhat | neutral | impolite | L2 closed |
|---|---:|---:|---:|---:|---:|
| flat 0.245 (this run) | 0.1628 | 0.1717 | 0.1467 | 0.5188 | 0% |
| 0.30 | 0.1833 | 0.1555 | 0.1549 | 0.5064 | 23% |
| **0.35 (shipped)** | 0.1966 | 0.1518 | **0.1508** | 0.5009 | 35% |
| 0.40 | 0.2092 | 0.1463 | 0.1464 | 0.4980 | 45% |
| 0.50 | 0.2352 | 0.1372 | 0.1379 | 0.4897 | 63% |
| 0.59 (uncapped) | **0.2599** | 0.1422 | **0.1277** | 0.4702 | **67%** |
| real target | 0.2595 | 0.1174 | 0.1591 | 0.4640 | |

Two corrections to what the shipped matrix predicted:

* **The neutral damage at cap 0.35 is gone.** Predicted +23.7%; the refit says
  0.1508 against 0.1591, i.e. **−5.2%**.
* **Uncapped reaches 67%, not 86%**, and there costs `neutral_rate` −19.7% while
  landing `polite_rate` at +0.2% and `impolite_rate` at +1.3%.

So the trade is real and it is now between `polite`/`impolite` and `neutral`, not
between closure and extrapolation. **Which cap to ship is a decision, not a
measurement**, and it should be made against the reported-metric set rather than
against L2: cap 0.59 makes two of the three reported tone metrics near-exact and
the third fail; cap 0.35 leaves all three mid-range.

## Caveat that bounds all of the above

This matrix is fitted on the **calibration pool's** corpus, not the evaluation
pool's. The polite row transferring and the neutral row not transferring is
evidence that some rows are generator properties and some are corpus properties.
The shipped matrix has **not** been replaced, because replacing a matrix measured
on the evaluation-seed corpus with one measured on a different corpus trades one
known bias for an unknown one. That decision is left open.

## A defect this exposed in the harness

`fit_tone_matrix.py` globbed `cleaned/` only. A freshly generated run has
`generated/` and no `cleaned/` until it is evaluated, so the first invocation
printed a **matrix of all zeros with n=0** and then crashed on a divide. It now
falls back to `generated/` the way `gate_audit.py` already did, and refuses to
print a matrix when no slots were found rather than printing zeros.
