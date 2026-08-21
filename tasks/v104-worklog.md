# v104 worklog — why the tone pair has failed since v96

Reproduce every number here with
`python3 generalized_card/analysis/polite_sentence_diagnosis.py all`
(offline; re-runs the evaluation's own `Intel/polite-guard` checkpoint on CPU,
about eight minutes). Read `docs/ORIENTATION.md` first.

Measured on the shipped v103 artifact
`generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1`, against the
matched real threads and the 15k evaluation-excluded real comments.

## The standing failure

`polite_rate` and `impolite_rate` are two of the three metrics that carry a
**statistically real generator bias** against the Planner's own target
(`polite_rate` −0.1856, `impolite_rate` +0.1529, both Wilcoxon p = 0.002). They
have failed in every release since v96. Eight versions of surface-marker work
moved neither.

## What was rejected, and how fast

| hypothesis | test | verdict |
|---|---|---|
| length mix — generated comments are shorter | Kitagawa decomposition | **rejected**: fixing only the mix closes 6% of the polite gap and **−2%** of the impolite gap. The whole gap is the within-band conditional (84% / 94%). |
| question marks — real polite comments ask more | out-of-sample lift on 23k excluded real | **rejected**: lift **1.08**. The 1.91 seen in the matched 40w+ subset is small-sample. |
| `you`-modal ("you could try…") — real 0.115, generated 0.027 | out-of-sample lift | **kept but tiny**: lift 1.37, so closing the whole prevalence gap predicts **+0.009** on `polite_rate` against a gap of 0.182. |
| personal narrative — real polite comments are stories | StorySeeker × Polite Guard | **rejected**: story probability already matches (real 0.156, generated 0.161), and generated *high-story* comments are still 0.696 impolite against a real 0.416. The gap persists inside every story bin. |
| polite vocabulary | bag-of-words model, AUC 0.87–0.91 | **rejected, with the sign that matters**: the lexical model **over**-predicts politeness on generated (0.150 predicted, 0.106 actual) while being calibrated on real (0.294 / 0.288). Generated text has the polite words. The residual is not lexical. |
| the comments are near the decision boundary and a nudge would tip them | stored class probabilities | **rejected**: median margin −0.934, only **2.1%** within 0.10 of flipping, median P(impolite) among impolite-labelled generated **0.981**. Polite Guard is not the near-degenerate head `hard_disagree_rate` reads — it is confident. No sub-sentence edit was ever going to move it. |

That last row is the retrospective explanation for eight versions of failure.

## The mechanism, at sentence level

Scoring every sentence of every 40w+ comment separately:

| | matched real | generated |
|---|---|---|
| comment-level mean P(polite) | 0.450 | 0.171 |
| sentences individually P(polite) > 0.80 | 0.072 | **0.016** |
| mean P(polite), first / middle / last sentence | 0.111 / 0.114 / 0.142 | 0.067 / 0.033 / 0.040 |

**The deficit is flat across sentence position** — it is not the opener and not
the closing move. It is that generated text never writes a sentence that is
unambiguously appreciative on its own.

## One binary feature reconstructs both failing rates

Define a *carrier* as a comment holding at least one sentence at P(polite) > 0.80.

| | excluded real | generated |
|---|---|---|
| share that are carriers | **0.220** | **0.062** |
| P(comment polite \| carrier) | 0.808 | 0.667 |
| P(comment polite \| not) | 0.154 | 0.069 |
| reconstructed `polite_rate` | 0.298 (actual 0.298) | 0.106 (actual 0.106) |
| P(comment impolite \| carrier) | 0.147 | 0.242 |
| P(comment impolite \| not) | 0.524 | 0.648 |
| reconstructed `impolite_rate` | 0.441 (actual 0.441) | 0.623 (actual 0.623) |

Both rates reconstruct to three decimals from this one feature on both sides.
**Conversion is nearly intact (0.808 against 0.667); prevalence is 3.6x off.**

Holding generated's own conversion rates and moving prevalence 0.062 → 0.220:
`polite_rate` 0.106 → **0.201**, which is **52%** of the gap; `impolite_rate`
0.623 → **0.558**, which is **36%**.

## Causally verified on the real scorer

Insert one real short polite sentence into the 165 generated non-polite 40w+
comments and re-score:

| insertion | becomes polite | mean ΔP(polite) |
|---|---|---|
| front | **0.497** | +0.396 |
| after the first sentence | 0.285 | +0.205 |
| end | 0.291 | +0.180 |
| **control** — a non-polite real sentence at the end | 0.121 | +0.057 |

The control is the falsification: adding *any* real sentence is worth 0.121; the
appreciative one is worth 0.29–0.50.

Whole-run counterfactual at the measured per-band rate, position drawn from
real, touching 82 of 528 comments: `polite_rate` 0.106 → **0.163**,
`impolite_rate` 0.623 → **0.604**, mean words 48.2 → 49.5. Under-delivers
against the 0.201 estimate because the insertion position is random rather than
chosen, which is the honest lower bound for a first build.

## The profile a mechanism would ship

Share of comments carrying one, fitted on evaluation-excluded threads only:

| band | excluded real | generated | ratio |
|---|---|---|---|
| 0–15 | 0.189 | 0.053 | 0.28x |
| 15–30 | 0.160 | 0.048 | 0.30x |
| 30–60 | 0.212 | 0.043 | 0.20x |
| 60–120 | 0.328 | 0.080 | 0.24x |
| 120+ | 0.485 | 0.159 | 0.33x |

Monotone in length, and generated sits at roughly a quarter of real in **every**
band. Median carrier sentence is **9 words**.

## The open problem before any code

The harvested set is defined by a **classifier score**, and only **36.9%** of it
is covered by the seven named surface forms (`gratitude` 0.112,
`self_positive` 0.076, `praise_object` 0.066, `superlative` 0.052,
`affirm_other` 0.045, `well_wish` 0.040, `offer_help` 0.006). Shipping "write a
sentence Polite Guard likes" would be tuning to the metric, which this project
does not do. **The next step is to name the remaining 63%**, then ship the
families — the same way v100 named the closing *move* instead of banning the
adjudication frame's phrasing, and v102 named the token instead of the category.

Verbatim examples of what is missing, all from excluded real:

    I LOVE the A7r.
    Best Camera I have ever used!
    Good God that's an amazing shot.
    That's awesome!
    And nice NEX-5!
    I've upgraded since I bought it but I still use it as a backup camera!

Short, unqualified, unhedged. This is also the user's criterion-2 complaint in
its positive form: the missing beat is not another *thoughtful* sentence, it is
the plain enthusiastic one.


---

## What was built, 2026-08-21

Three arms, one release, one artifact so each is attributable — the same way
v97's four and v98's five were. `evaluative_register.py`, profile schema 20.

The carrier framing above localised the defect; it did not explain it. Naming
the forms is what did. The generator already writes the appreciative forms at or
above the real rate (`gratitude` 1.48x, `positive_predicate` 1.39x,
`bare_verdict` at parity) — they do not land, at a quarter to a tenth of real
precision. Reading matched pairs side by side gave three surface causes, and the
whole-corpus rates and the ablation are in `generalized_card/VERSION_LOG.md`
under v104.

Two things were tested and rejected on the way, both cheap:

- **The reuse ledger as a priming source.** It echoes the exact tics back to the
  Writer (`- that's the bit that (used 3x)`, `- The $200 part is nice, sure,
  but`), which looked like a self-feeding loop. Partitive lift **0.95x**, tag
  lift **1.09x**, and flat or lower where the ledger is present once position in
  the thread is controlled. v98's mechanism is untouched.
- **The named-form taxonomy as the mechanism.** Eleven forms, fitted on half the
  excluded threads and scored on the other half, reach only **0.420** recall of
  the carriers at 0.317 precision. `recommend_personally` (held precision 0.102),
  `me_too` (n=18) and `long_tenure` (held 0.116 against a fit 0.292 — it does not
  replicate) were dropped. A profile of "write a sentence Polite Guard likes"
  was **not** shipped: it is defined by a classifier score, and this project does
  not tune to the metric.

**What is still open after v104.** Even at full compliance the three edits close
28.1% of the polite gap. The carrier prevalence gap (0.220 real against 0.062)
is worth about 52% on its own and is not addressed here, because the forms that
make up 58% of it are still unnamed. That remains the next piece of work, and it
needs a better taxonomy, not a bigger regex.
