# Why `self_bleu_4` and `self_bertscore_mean_f1` fail — measured, 2026-08-25

Run: `generalized_card_camera_gpt54_paper_20260825_v1`, N=50, 1,974 comments.
Every number below is produced by a script in this directory. Each harness
reproduces the shipped artifact before printing an edited number (E6):
self_bleu_4 to 1e-16 (generated) and 0 (real); Self-BERTScore to 4e-4.

## 0. The premise that was wrong

The last three releases treated `self_bleu_4` and `self_bertscore` as one
"the generator repeats itself" defect. They are two unrelated defects, and
neither is repetition.

| | corr with mean comment length, within real | R² |
|---|---:|---:|
| `self_bleu_4` | **−0.479** | 0.24 |
| `self_bertscore_mean_f1` | +0.033 | 0.02 |

Shape of the bias (`shape.py`): `self_bertscore_top_k_mean_f1` is **lower** in
generated (0.6682 vs 0.6886, MWU 0.94) while the median is higher (MWU 0.0003);
`semantic_mean_cosine` is lower on mean, median, top-k and p90. Generated threads
have **fewer near-duplicate pairs and more topical spread than real**. The excess
is a uniform floor, not repetition. `self_bleu` excess falls monotonically with
n-gram order (bleu_2 +14.7%, bleu_3 +9.9%, bleu_4 +7.2%) — short n-grams, not
copied phrases.

## 1. Every failure is a pure location bias, not a shape mismatch

Recentre the generated per-thread distribution on real's mean and change nothing
else (`sb4_report.py`, `shape.py`):

| metric | MWU after de-biasing | KS | Cliff | sd ratio gen/real |
|---|---:|---:|---:|---:|
| self_bleu_4 | 0.446 | 0.549 | +0.089 | 0.65 |
| self_bertscore_mean_f1 | **0.970** | **0.967** | +0.005 | 1.00 |
| polite_rate | 0.833 | 0.179 | +0.025 | 0.68 |
| impolite_rate | 0.780 | 0.717 | +0.033 | 1.16 |
| neutral_rate | 0.702 | 0.396 | −0.045 | 1.03 |

There is no distributional-shape problem to solve. Five means need moving.

## 2. `self_bleu_4` is 80% a length artifact of the scorer's smoothing

`sentence_bleu` uses add-one smoothing at every order, `precision =
(overlap+1)/(total+1)`, and returns p=1.0 for an order where the hypothesis is
shorter than n. Both are functions of **token counts only**, so self_bleu_4
decomposes exactly and additively into a length floor plus a content excess
(`sb4_decomp.py`, `sb4_real.py`, `sb4_report.py`):

| component | real | generated | gap | share of the gap |
|---|---:|---:|---:|---:|
| **floor (length only)** | 0.01970 | 0.02164 | **+0.00194** | **80.0%** |
| excess (real n-gram overlap) | 0.01403 | 0.01451 | +0.00049 | 20.0% |
| self_bleu_4 | 0.03372 | 0.03615 | +0.00243 | 100% |

Generated comments carry 48.79 tokens against real's 57.76 (−15.5%): −11% in
words (realized/assigned 0.891) and −4.2% in tokens per word.

Exact counterfactuals (`len_cf.py`, floor recomputed in closed form, excess held
fixed — an upper bound per J7):

| scenario | self_bleu_4 | bias | MWU | KS |
|---|---:|---:|---:|---:|
| today | 0.03615 | +7.2% | 0.057 | 0.039 |
| v111 scope (assigned 35–100 → 1.0) | 0.03549 | +5.2% | 0.084 | 0.112 |
| every band ≥35 → 1.0 | 0.03528 | +4.6% | **0.109** | **0.112** |
| generated given real's per-thread floor | 0.03421 | +1.5% | **0.340** | **0.396** |

Realized/assigned by band on this artifact (`lenfid.py`): 1–9 **1.224**, 10–19
1.001, 20–34 0.992, 35–49 0.907, 50–69 0.881, 70–100 0.877, 101–150 0.910,
151–300 **0.802**, 301+ **0.699**. The 35–100 band v111 targets holds 41.3% of
assigned words. The >150 collapse is a separate defect and is worth more than
v111's band; note `--writer-max-tokens 260`.

## 3. `self_bertscore` is 76% caused by URLs the generator is forbidden to write

The observational regression could **not** identify this: a nine-feature battery
fitted on 536 real threads reaches R²=0.60 but predicts only +0.0049 of the
observed +0.0124 gap, with signs that contradict each other (`bert_drivers2.py`).
Regression is not identification.

Causal ablation instead — real text only, transformed toward the generator's
surface, rescored with the shipped scorer (`bert_ablate.py`, `bert_ablate2.py`,
`bert_ablate3.py`; 20 matched real threads, 2,029 pairs, deberta-xlarge-mnli,
no baseline rescaling, no idf):

| transform applied to REAL text | self_bertscore | move | share of the +0.0124 gap |
|---|---:|---:|---:|
| baseline | 0.4975 | — | — |
| sentences comma-joined | 0.4974 | −0.0001 | **−1%** |
| u/ or r/ mentions removed | 0.4976 | +0.0000 | 0% |
| `* _ \` ~` emphasis removed | 0.4974 | −0.0002 | −1% |
| quote markers / escapes removed | 0.4977 | +0.0001 | +1% |
| **all URLs removed** | **0.5069** | **+0.0094** | **+76%** |
| — machine media URLs only | 0.5002 | +0.0027 | +22% |
| — human reference URLs only | 0.5040 | +0.0064 | **+52%** |

Prevalence over the 1,951/1,953 matched comments: a URL appears in **4.41% of
real comments and 0.00% of generated ones**. 115 URLs: 21 machine media
attachments (preview.redd.it, i.imgur.com), 94 human references (dpreview,
youtube, bhphotovideo, mpb, lensrentals, canonrumors). `prompts.py:2722` tells
the Writer to "write a normal human reference sentence **without inventing a
URL**", and `prompts.py:1917` rewrites every URL in the visible thread to
`[link]`, so the Writer has never seen one.

The clause-packaging hypothesis is **dead**: comma-joining real text moves the
metric by −1%. It is not the mechanism, despite generated being 0.199 comma-free
against real's 0.411 within every length band (`surface.py`).

URLs also carry part of `self_bleu_4` (`url_bleu.py`): they are 3.06% of real's
tokens, and removing them from real moves the gap +0.00243 → +0.00105 and MWU
0.057 → 0.155. Real tokens/word 1.1954 → 1.1589 without them, against
generated's 1.1450 — so the URL channel is most of the tokens-per-word gap too.

## 4. What the target actually is

The reporting standard is Holm over the 24 tests (J2), so the binding constraint
is the single smallest p-value against 0.05/24 = 0.00208. Required reduction in
|Cliff's delta| from today (`holm_state.py` and the cap arithmetic):

| metric | today |δ| | N=50 raw | N=150 raw | **N=150 Holm** |
|---|---:|---:|---:|---:|
| self_bleu_4 | 0.222 | 0% | −41% | **−7%** |
| self_bertscore | 0.357 | −36% | −63% | **−42%** |
| polite_rate | 0.436 | −48% | −70% | **−53%** |
| impolite_rate | 0.375 | −39% | −65% | **−45%** |
| neutral_rate | 0.286 | −20% | −54% | **−28%** |

The URL channel alone is a 76% reduction on `self_bertscore`, against the 42%
needed. The length channel is a 20–57% reduction on `self_bleu_4`, against 7%.

## 5. The bias history says the work has been converging

| metric | v97 N=10 | v98 N=10 | v110 N=10 | this run N=50 |
|---|---:|---:|---:|---:|
| polite_rate | −72.4% | −70.0% | −51.5% | **−42.2%** |
| impolite_rate | +69.5% | +67.5% | +39.9% | **+26.8%** |
| neutral_rate | −53.0% | −50.2% | −19.7% | **−27.5%** |
| self_bleu_4 | +13.7% | +18.5% | +18.8% | **+7.2%** |
| self_bertscore | +4.9% | +4.4% | +2.6% | **+2.5%** |

Every bias is smaller than at any earlier release. The p-values got worse
because N went 10 → 50 and the tests gained √5 of power, not because the
generator regressed. Under Holm this artifact scores **9/12**.

## Unfinished

`url_delta.py` computes the exact per-thread URL effect over all 50 matched
threads (only pairs touching a URL-bearing comment change, so the thread delta
is exact). It was interrupted after seed 10; every thread scored so far has a
positive delta (+0.0015 to +0.0293). Finishing it turns section 3's mean-level
projection into a per-thread MWU/KS projection.

---

# The link arm, priced to its ceiling — 2026-08-26

Run `v113_v112_gate_n10_20260826_v1` against the matched real threads on the same
ten seeds 2-11. Scripts in this directory; the shipped scorer throughout
(deberta-xlarge-mnli, no baseline rescaling, no idf).

## 1. The arm fired and closed 24%, against the 42% s4 requires

The v113 gate wrote links at 4.32% of comments against real's 4.41% -- prevalence
matched -- yet applying the identical URL ablation to the gate's own output moves
`self_bertscore` by only 0.0038, a **24.1%** closure of the no-link gap, where
stripping URLs from real closed 76%. Same prevalence, a third of the effect.

## 2. Why: the URL mass, not the URL count

BERTScore is greedy token alignment with no idf, so a URL is worth its share of
its comment's tokens (`url_shape_gap.py`):

| | comments with a URL | URLs/comment | tokens/URL | URL tokens per carrying comment |
|---|---:|---:|---:|---:|
| real | 26 (4.92%) | 1.42 | 22.9 | **32.6** |
| gate | 23 (4.34%) | 1.22 | 14.8 | **18.0** |

v113's inventory reader ran `https?://\S+` straight through Reddit's `[url](url)`
markdown, so 166 of 690 entries were malformed and the drawn strings were
truncated. v114 fixed the reader.

**Rewriting the gate's links to their v114-clean form buys nothing** -- 23.7%
against the shipped 24.1%, because the malformed form carried the URL's characters
twice and the clean form carries them once.

## 3. The ceiling, measured (`url_mass_scaling.py`)

Adding inventory URLs to the gate's own link-carrying comments. Nothing else
changes -- no comment is shortened, no other slot is touched, so the move is
attributable.

| URL tokens per carrying comment | self_bertscore | bias | closure |
|---:|---:|---:|---:|
| 18.0 (today) | 0.5061 | +2.42% | 23.7% |
| 22.8 | 0.5057 | +2.32% | 26.8% |
| 24.7 | 0.5045 | +2.09% | 33.9% |
| 35.8 | 0.5034 | +1.87% | **40.9%** |
| *real's own 32.6* | | | *~39% interpolated* |

**Matching real's URL mass exactly tops the arm out near 39%, against the 42%
Holm needs at N=150** -- and that is a J7 upper bound, since the Writer complies
with a link offer at 0.958 and with a two-link offer at an unmeasured rate below
1.0. Going past 35.8 tokens means exceeding real's own link density, which is not
a legal arm.

**The link channel cannot carry `self_bertscore` alone. A second channel is
required.**

## 4. Two candidates killed, cheaply, and one integrity check passed

`url_shape_control.py`. Cutting the gate's link-carrying comments down to the
sentence holding the link closes 54.5%, which looked like a better arm than mass
matching -- but the controls kill it:

| variant | closure |
|---|---:|
| gate, as shipped | 24.1% |
| link comment -> its link sentence | 54.5% |
| **CONTROL: same count of random non-link cuts** | **22.5%** |
| link sentence, url also deleted | 85.5% |

The random-cut control passes (22.5% vs 24.1%: plain shortening does nothing), but
the last row exposes the rest: deleting the URL *as well* closes more than keeping
it, so what the variant actually creates is near-empty comments, and a degenerate
short text drags every pair it appears in. Generated already carries **more**
comments of <=10 words than real (0.1377 vs 0.1155), so this is not a real-versus-
generated gap at all. Not an arm.

**Length variance is not a channel either.** Real's comment lengths have sd 74.8
against generated's 58.2, which looks like a large gap -- but truncating real to
the gate's own maximum length moves `self_bertscore` by **0.0000**.

**Integrity: no URL was invented.** Of 24 distinct emissions, 18 match an
inventory entry exactly and 5 are the clean prefix of a *malformed* v113 inventory
entry -- the Writer behaved correctly and the inventory was the corrupt side. The
remaining one altered a YouTube `?si=` tracking parameter; the video id itself is
in the inventory and in the prompt the Writer received.

## 5. Where the second channel should be looked for

`surface_class_prevalence.py`, token share over the same ten seeds. Sorted by the
gap, the classes real carries and generated does not:

| class | real prev | gen prev | real tok% | gen tok% | gap |
|---|---:|---:|---:|---:|---:|
| parenthetical aside | 0.1723 | 0.0906 | 4.84% | 1.33% | **+3.51%** |
| digit run | 0.5644 | 0.5340 | 4.06% | 1.58% | **+2.49%** |
| alnum model code | 0.4564 | 0.3642 | 2.04% | 0.82% | +1.22% |
| url | 0.0492 | 0.0434 | 2.41% | 1.33% | +1.08% |
| price | 0.0739 | 0.0094 | 0.40% | 0.04% | +0.37% |

**The parenthetical gap is 3.2x the URL gap by token share and has never been
tested.** Underneath both sits the general form: generated carries 2,802 types and
1,174 hapax against real's 3,993 and 2,040 -- **43% fewer once-only tokens** on a
comparable token count. URLs are the most visible instance of that class, not the
class itself. `rare_token_ablation.py` prices parentheticals, digit runs and
hapax flattening, each against a random-token-removal control matched on the exact
number of tokens removed, because removing text also shortens it.
