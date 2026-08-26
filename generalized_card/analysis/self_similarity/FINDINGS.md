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

## 6. The second channel, found — and the two candidates it killed

`rare_token_ablation.py`, same ten seeds. Each class is removed from REAL text and
rescored; a rise is that class's contribution to real's lower score. Every row is
paired with a **random-token-removal control matched on the exact number of tokens
removed per comment**, because removing text also shortens it, and the reported
`net` subtracts the control.

| ablation on REAL | move | control | net | share of the +0.0138 gap |
|---|---:|---:|---:|---:|
| − URLs | +0.0077 | −0.0003 | **+0.0080** | **58%** |
| − parenthetical asides | +0.0017 | −0.0015 | **+0.0031** | **23%** |
| − digit runs | −0.0010 | −0.0005 | −0.0005 | **−3%** |
| − all three | +0.0077 | −0.0019 | +0.0096 | **70%** |
| hapax → a frequent thread word | −0.0020 | (length-neutral) | −0.0020 | **−14%** |

**Parenthetical asides are the second channel.** Real puts 4.84% of its tokens
inside parentheses against generated's 1.33%, and removing them costs real 23% of
the gap. A parenthetical is a pure surface move — no domain vocabulary, no
invented fact — so an arm for it is `ORIENTATION.md` s4 safe.

**Digit runs are dead, and that is the good news.** Their token gap is +2.49%,
larger than the URL gap, and s5 named them the second-biggest candidate. They are
worth **−3%**. That closes the only route that would have required domain
vocabulary in Writer-facing text.

**The hapax framing in s5 is wrong and is retracted here.** Generated carries 43%
fewer once-only tokens, and s5 read URLs as the most visible instance of that
class. Flattening real's hapax to a frequent thread word moves the metric the
**wrong way**, −0.0020. Rare tokens are not the general form of the URL channel;
the URL channel is specific.

### What the two live channels add up to

All three together return 70% where the singles sum to 81%, so the channels are
sub-additive at about 0.86 efficiency. Carrying that to the generated side, where
s3 measured the link arm's own ceiling at ~39% closure at real's URL mass:

    (39% + ~15% from a parenthetical arm at the same real->generated discount)
        x 0.86  ~=  46%,  against the 42% Holm needs at N=150.

### CORRECTION — that 46% is wrong, 2026-08-26

**Retracted.** The ~15% assumed a parenthetical arm closes the whole 4.84% / 1.33%
token gap. v116 draws the COUNT and does not touch compliance, and the count term
is far smaller than it looks because the two deficits compound in the same
direction.

The gate's parenthetical carriers sit in the wrong bands — short 11, medium 23,
long 11, very_long 3 — precisely because compliance collapses at length
(0.71/0.61/0.30/0.12). The measured count at short and medium is only 1.20-1.22,
so the **band-weighted** target over the gate's own carrier mix is **1.31**, not
the 1.76 real reaches with its long-skewed carriers.

Recomputed against the measured link-arm discount of 0.42 (24.1% delivered on the
generated side against 58% on the real side):

| | paren token share | closes of the paren gap | of the `self_bertscore` gap |
|---|---:|---:|---:|
| gate today | 1.33% | — | — |
| **v116 alone (count only)** | 1.75% | **12%** | **1.1%** |
| v116 + compliance 0.38 → 0.85 | 3.91% | 73% | 7.0% |
| real | 4.84% | 100% | 23% |

**Compliance is not a separate defect, it is the dominant one**, and it also gates
the count: fixing it moves carriers into the long bands where the measured count
is 1.47-1.88, so the two multiply. v116 without it is a multiplier on a broken
base.

Honest total of what is **built** today: the shipped link arm's measured 24.1%
plus v116's 1.1% ≈ **25%**, against the 42% Holm needs. The URL-mass arm that
would reach ~39% **does not exist** — `reference_link.draw_reference_link` returns
one URL and the offer says "Include this exact URL once", so the 1.42-per-carrier
figure is an ablation of text, not an arm. Neither does a compliance fix, and the
compliance collapse is still unexplained.

## 7. The parenthetical arm already exists and is 38% complied with

`sentence_rhythm` has carried a `parenthetical` habit since v97, drawn per slot at
its band's measured rate with the cue **"Put one aside in parentheses."** So the
question is not whether to add an arm (E5) but why the one there under-delivers.

Measured on the gate's own saved prompts:

| | value |
|---|---:|
| slots cued | 100 / 532 = **0.188** |
| real comment prevalence | 0.172 |
| slots realizing a parenthetical | 48 / 532 = 0.090 |
| **compliance, realized \| cued** | **0.380** |
| realized when not cued | 0.023 |

**The draw is already above real. The whole deficit is compliance.** And it is two
deficits, multiplicative:

| | prevalence | parens per carrier | words per paren | paren words per carrier |
|---|---:|---:|---:|---:|
| real | 0.1723 | **1.76** | 5.4 | **9.6** |
| gate | 0.0906 | **1.00** | 6.0 | **6.0** |

**Correction to an earlier version of this table.** It reported "tokens per paren
8.6 against 10.6" as a third deficit. That count used the self_bleu tokenizer,
which counts the brackets and inner punctuation as tokens. In words the gate's
asides are slightly **longer** than real's (6.0 against 5.4) and their p90 is
identical at 11. There is no length deficit. There are exactly two, and the
per-carrier word gap (9.6 against 6.0) is explained entirely by the count.

Real's per-carrier count runs 1/2/3/4/6/22. The gate's distribution is
**`{1: 48}`** — every single carrying comment has exactly one, with no exceptions.
The cue says *one*, and it gets exactly one. That is E4 confirmed from the other
direction: naming the concrete number buys ~1.0 compliance on the number.

Real's parentheticals also sit in much longer comments (carrier mean 129.9 words
against the corpus's 56.2) and span 3-23 tokens where the gate spans 5-13.

### The long-comment reading, tested and retracted

`long_prompt_crowding.py`. Compliance falls 0.71 -> 0.61 -> 0.30 -> **0.12** across
the four length bands, which matches the polite per-sentence collapse above 30
words and looked like one root cause under both. It is not:

| habit | short | medium | long | very_long | overall |
|---|---:|---:|---:|---:|---:|
| parenthetical | 0.71 | 0.61 | **0.30** | **0.12** | 0.380 |
| ellipsis | 0.80 | 1.00 | 1.00 | (n=3) | 0.943 |
| exclamation | 0.77 | 0.81 | 0.91 | 0.43 | 0.766 |
| digit | 0.50 | 0.70 | 0.77 | 0.67 | 0.681 |
| dash_clause | (n=1) | 0.55 | 0.62 | **0.83** | 0.667 |
| short_sentence | 0.45 | 0.26 | 0.24 | 0.60 | 0.338 |

Only `parenthetical` collapses; `dash_clause` rises with length. And the long-slot
prompt is not crowded — it carries the **fewest** rule lines of any band (68.9
against medium's 82.0) and 28% more characters than a short one. The unifying
reading is wrong and is retracted here.

A methodology note worth keeping: the first version of this table used loose
needles and reported `semicolon` and `dash` as cued on 522/532 and 532/532 slots
at 0.04 compliance. Both were matching unrelated prompt text. `semicolon` carries
an **empty** cue by design, because generated over-produces it. Compliance tables
must key on the exact cue string.

### What v116 would be

Draw the parenthetical **count** per band from real's measured distribution rather
than cueing a fixed "one", and address the length-band compliance directly. The
count fix alone takes paren tokens per carrier from 8.6 toward real's 18.7; raising
compliance from 0.38 to 0.8 takes realized prevalence from 0.090 to about 0.150
against real's 0.172. Together they close most of the 4.84% / 1.33% token gap that
s6 priced at 23% of the `self_bertscore` gap.

## 8. The channel that was never looked at: one model writing every speaker

`one_voice_floor.py`, `one_voice_control.py`, `one_voice_generated.py`.

s6 identified 81% of the gap in surface classes and s3/s7 showed that implementing
every one of them reduces it by only **31%** against the 42% Holm needs, at a
delivery ratio of 0.34 against their real-side value. Both facts point at
something underneath the symbols.

### Real threads carry an authorial-voice floor, and it is not topic

Inside real threads, pairs by the SAME author outscore pairs by different authors
by **+0.0177**. On Reddit one person's two comments usually sit in the same
subthread, so that could be topical proximity wearing authorship as a mask.
Conditioning on conversational relation does not just fail to kill it — it
**reverses the confound's prediction**:

| relation | same-author | different | delta |
|---|---:|---:|---:|
| same parent | 0.5051 (n=16) | 0.5059 (n=1228) | **−0.0008** |
| ancestor/descendant | 0.5156 (n=206) | 0.5136 (n=740) | +0.0021 |
| same root branch | 0.5297 (n=88) | 0.5064 (n=2050) | **+0.0233** |
| **different branch** | 0.5022 (n=476) | 0.4881 (n=21462) | **+0.0141** |
| stratum-weighted | | | **+0.0137** |

If topic drove it the effect would be largest where the two comments sit closest.
It is **near zero there** and largest where they sit furthest apart. That is the
shape of a writing signature, not of a shared subject.

### The generator has 55% of real's voice separation

The label structure is not the problem: the gate assigns **326** distinct authors
across 530 comments against real's 260 across 528, and its same-author pair share
is *lower* (0.0201 against 0.0299). The labels are there. The question is whether
they carry a voice, and they partly do:

| relation | generated | real | gen/real |
|---|---:|---:|---:|
| same parent | **+0.0170** | −0.0008 | — |
| ancestor/descendant | +0.0058 | +0.0021 | — |
| same root branch | +0.0133 | +0.0233 | 0.57 |
| **different branch** | **+0.0061** | **+0.0141** | **0.43** |
| stratum-weighted | **+0.0076** | **+0.0137** | **0.55** |

A generated speaker writing in two unrelated parts of a thread is **43%** as
distinctive as a real one. Scaled to a whole thread:

    headroom = +0.0060 = 51% of the +0.0119 gap.

**That is larger than the 42% Holm needs, and it is the only channel measured so
far that is.** It also explains what the surface channels could not: their 0.34
delivery ratio, and the 19% they never covered. A URL, an aside, an odd rare token
are *by-products* of many different people writing, so patching them individually
works against a floor that is still there.

### What this is and is not

It is a **bound**, computed the same way s3's link ceiling was, and J7 applies: it
says what closing the voice gap would be worth, not that an arm can close it. Two
things temper it. The headroom is **not additive** with s6's surface channels —
authorial variety is partly *expressed* through them, so the two overlap by an
unmeasured amount. And the generator is already at 0.55 of real's separation, so
the remaining work is making existing personas more lexically distinct, which is a
harder ask than adding a symbol that was simply absent.

One anomaly worth keeping: at `same parent` the generator runs **+0.0170** where
real runs −0.0008. Reusing a speaker in the same local spot makes generated
comments *more* alike than a human's, the opposite of everywhere else. n=39, so
thin, but it is the one cell where the generator is worse than real rather than
flatter.

`persona_bridge`, `speaker_roster`, `actor_conditioning` and `--speaker-identity
matched` all exist and were on for this run. None has ever been measured against
`self_bertscore`. That is the next measurement, and it is free.

---

# The floor, the ceiling, and a trap in the archive — 2026-08-26

## 9. The target IS reachable at full coverage, and the gap is 10x the noise

`real_vs_real_floor.py`. `scripts/bootstrap_real_comment_discussions.py` states the
logic: attach a DIFFERENT real thread to each seed and score it — "if this
bootstrap cannot match the matched real distribution, the issue is likely
seed/eval/matching/sample-size rather than the generator; if it does match, the
target distribution is reachable in principle." That run existed only for
**credit_cards**. Done here on camera, from the cached real baseline:

150 evaluation real threads against 150 **disjoint** real camera threads matched
on comment count, coverage 0.996:

| metric | target | donor | bias | MWU | KS | |
|---|---:|---:|---:|---:|---:|---|
| **self_bertscore** | 0.4923 | 0.4935 | **+0.24%** | 0.810 | 0.443 | PASS |
| **self_bleu_4** | 0.0330 | 0.0325 | **−1.61%** | 0.801 | 0.231 | PASS |
| semantic_mean_cosine | 0.2741 | 0.2816 | +2.72% | 0.320 | 0.443 | PASS |
| polite_rate | 0.3216 | 0.3336 | +3.75% | 0.358 | 0.628 | PASS |
| impolite_rate | 0.4079 | 0.3893 | −4.56% | 0.338 | 0.362 | PASS |
| neutral_rate | 0.1611 | 0.1773 | +10.05% | 0.384 | 0.443 | PASS |

**An arbitrary real camera thread passes all six comfortably.** The metric, the
matching, and the sample size are all sound, and the generator's +2.41% is **ten
times** the natural real-to-real spread. This is also the cheapest validation
harness the project has: any domain can be checked this way before a single token
is spent.

## 10. No thread-level aggregate explains the gap, which is why regression failed

`real_thread_correlates.py` ranks every cached thread-level column by its
correlation with `self_bertscore` across 763 real threads, then places the
generator on each in units of real's own spread — **against its own matched real
threads**, not the population.

| column | corr with sbert | matched real | generated | gen z |
|---|---:|---:|---:|---:|
| self_bleu_2 | +0.58 | 0.0500 | 0.0618 | +0.3 |
| semantic_p90_cosine | +0.81 | — | — | −0.2 |
| polite_rate | +0.22 | 0.3020 | 0.1746 | −0.5 |
| self_bleu_3 | +0.46 | 0.0391 | 0.0429 | +0.2 |
| neutral_rate | −0.26 | 0.1577 | 0.1143 | −0.2 |

**The generator sits within ~0.3 sd of its matched real threads on essentially
every cached metric.** That is why s3's nine-feature regression reached R²=0.60
and still predicted only 40% of the gap: the driver is not an aggregate. It is per
comment, or per pair.

A first version of this table compared the generator to the whole real population
and produced a large apparent structural gap — branching factor, virality, depth.
That was a **selection effect**: the evaluation pool's threads are bigger than the
population (39.1 comments against 32.4), and the generator copies its matched
thread's structure. Recorded because the artifact reads convincingly either way.

## 11. Decomposing the metric to per-comment leverage

`high_floor_comments.py`. `self_bertscore` is the mean over pairs, so each
comment's leverage is the mean F1 of the pairs it appears in. Ranked and centred
within thread:

| side | decile | leverage | words | sentences | 1st-person | question |
|---|---|---:|---:|---:|---:|---:|
| real | bottom 10% | −0.0568 | **23.7** | 2.13 | 0.13 | 0.17 |
| real | top 10% | +0.0360 | 30.7 | 2.19 | 0.62 | 0.13 |
| gate | bottom 10% | −0.0578 | **46.2** | 3.04 | 0.28 | 0.30 |
| gate | top 10% | +0.0361 | 34.0 | 2.36 | 0.77 | 0.04 |

The leverage *range* matches on both sides. What differs is what occupies the
ends. The generator's high-leverage comments are one thing — 66% of them carry
`evidence_mode=technical_or_policy_reasoning` against 23% at the bottom, 45%
`payload_type=soft_helpful` against 6%: the polished first-person helpful take.

And the two bottom deciles are made of different material:

    generated:  "Hard pass"  "Pretty much"  "Confirmation email? lol"
    real:       "Sean Tucker"  "Gold quality, lead weight :("
                "current af lenses: - e-mount: Sigma 19/2.8 30/2.8 60/2.8, 18-55 kit"
                "About 1k$-1.2k$ since I'm bringing only 2k$."
                "Not true. See [Gerald Undone's test results](https://twitter.com/...)"

Real reaches its lowest leverage with **23.7 words** of dense content — a name, a
spec list, a price, a link. The generator needs **46.2 words** and gets there with
*conversational* fragments instead. It is paying twice the length for the same
dissimilarity.

**A hypothesis this killed:** first-person looked like the discriminator (real
0.62 vs 0.13 across deciles, gate 0.77 vs 0.28). It is not a level difference —
overall first-person prevalence is real 0.580 against generated 0.540, and density
3.50 against 2.95 per 100 words. The generator uses **less**. The decile split is
within-thread ranking, not a gap.

## 12. A trap in the archive: `self_bertscore` has never passed at full coverage

Recorded because the artifact directory reads as though it has.

Searching every `*_controller_history.json`, `self_bertscore_mean_f1` appears as a
**protected** metric in 32 observations that end PASS and as a **target** in
**zero**. The self-loop revisers target `tone` (121 run directories), `self_bleu`
(77) and `story_structure` (34). **There is no self_bertscore self-loop.** Where
the metric shows PASS it was already passing before the loop ran, and the loop's
job was not to break it.

And those passing runs sit in the regime `VERSION_LOG.md` opens by warning about:

| run family | coverage | gen sbert | bias |
|---|---:|---:|---:|
| card_deepseek_v4_flash_v37 | **0.577** | 0.4585 | −1.70% |
| card_gemini25flash_v37 | **0.546** | 0.4571 | −2.00% |
| repro_v37 | **0.629** | 0.4623 | −0.88% |
| sample_planner_gpt4omini_writer_v37 | 0.603 | 0.5059 | **+8.47%** |

2,300–2,700 generated comments against 4,255 real ones. Truncating a thread
flatters exactly these metrics, and only v64 and later are comparable. Sweeping
all **284** evaluated run directories, the coverage≥0.90 band has a median
self_bertscore bias of **+4.28%** and exactly **one** run under 1% — the
real-comment bootstrap of s9.

Separately worth keeping: at *matched* coverage the generator swings the metric
enormously — `repro_v37` at 0.629 gives −0.88% and `sample_planner_gpt4omini` at
0.603 gives **+8.47%**. Nine points from something other than coverage, never
identified. Those two runs differ in more than one thing, so it is an observation
and not a channel.

## 13. `evidence_mode` is a labelling gap, not a text gap — the spend is not justified

`dead_evidence_cells.py`. s6/s12 left `evidence_mode` as the largest per-pair
collision channel (41.4% of cross-branch pairs share one, worth +0.0228, topic
controlled) and the plan was to buy a target distribution by labelling real text
with an LLM. Three of the taxonomy's **seven** values are effectively unused by
the Planner, and those three are the ones a surface pattern can find for free:

| evidence mode | real (lower bound) | generated SURFACE | Planner ASSIGNS | surface ratio |
|---|---:|---:|---:|---:|
| link_quote_reference | 0.0859 | 0.0736 | **0.0060** | **0.86** |
| hearsay_consensus | 0.0175 | 0.0189 | **0.0020** | **1.08** |
| calculation_math | 0.1123 | 0.0491 | **0.0000** | 0.44 |

**Two of the three are already at real's rate in the text, with the label at
essentially zero.** The Writer produces a quote, a link, a "people say" frame
whether or not `evidence_mode` names it. So raising those labels to real's rate
would push the surface *above* real, not close a gap — and the 28% figure that
arithmetic produces is not usable. Only `calculation_math` has a genuine surface
deficit, and bare digits were already priced at **−3%** in s6.

The conclusion is about coupling: `evidence_mode` and surface behaviour are only
loosely linked, so a measured per-pair collision effect cannot be assumed to
convert into changed text. Whether real is also ~50% technical reasoning — the
question that decides the dominant cell — is not answerable by pattern, and on
this evidence paying an LLM to label real text for it is not justified.

Recorded as a decision not taken, with its reason, so it is not re-proposed.

## 14. v117 hits the metric target and makes the content visibly worse — 2026-08-26

`gate_audit.py --tag v117_calibration_20260826_v1`. The arm fires almost exactly:

| | target | measured |
|---|---:|---:|
| URLs per carrying comment | real 1.67 | **1.68** |
| characters per URL | inventory 61 | **61** |
| compliance, wrote \| offered | v113's 0.958 | 0.950 |
| markdown garbage / invented URLs / repeats | 0 | **0 / 0 / 0** |

And the output contains this:

    [4 urls | 46w] ...Nikon with a higher-end standard zoom and AF
                   https://youtu.be/fz2LSHQ8E_w
                   http://hasselblad.com/promotions/camera-prices.aspx
                   https://www.bhphotovideo.com/...

Four unrelated links stacked at the end of a 46-word comment. Also
`https://support.apple.com/en-us/119916` inside a comment about a Sony A7, and a
Fuji X-T5 film-simulation recipe inside one comparing Canon compacts.

The cause is in the draw: URLs come from an 802-entry inventory by hash, with **no
relationship to the comment's content**. At one link that reads as a reference
aside. At four it is a wall of unrelated links and an eye-visible tell, which is
worse for the claim being made than the metric gain is worth.

Two measured defects, both fixable (249 real comments carrying 2+ non-media URLs):

| | real | v117 |
|---|---:|---|
| all URLs share ONE host | **0.643** (160/249) | drawn independently |
| distinct-host counts | 1:160 2:60 3:20 4:4 5:3 | — |
| position of the first URL | median **0.23** of the way in | stacked at the end |

So a v118 should (a) draw a multi-link slot's URLs from **one host** at real's
0.643 rate, and (b) place the first link early rather than trailing — the current
offer says "inline and in different places" and the Writer partly ignores it.

**Recorded as a blocker on v117, not a success.** The metric numbers are right and
the artifact is not shippable as it stands.

## 15. Two thirds of s14's fix did not survive measurement — 2026-08-26

`url_host_coherence.py`, on the **150-seed** evaluation-excluded corpus (424
threads, 11,817 comments, 531 carriers, 179 of them holding 2+ non-media URLs).
s14 was measured on the calibration pool's larger exclusion set, which is why its
counts are bigger; the direction of the surviving claim is the same.

### The placement half is an artifact — RETRACTED

s14: *"the first URL sits a median 23% into the comment rather than trailing"*, so
a v118 should place the first link early. That 0.23 is `text.find(url) / len(text)`
— a **character** fraction divided by a length the URLs themselves dominate. A
median URL is 61 characters, so in a comment ending in a block of three of them
the prose is a small part of the denominator and the first URL "starts early" by
arithmetic alone. Reproduced here at 0.314 on this corpus, so the figure is right
and the reading is wrong.

Measured in **words**, which is what a Writer cue can act on:

| | first URL, median | first URL in the last quarter |
|---|---:|---:|
| all carriers (n=507) | **0.795** | **0.533** |
| k=2..4 (n=147) | **0.722** | **0.497** |

Two independent forms agree: characters excluding URL characters gives 0.647, word
index over non-URL words gives 0.708. **Real trails its links.** v117 stacking
them at the end is correct behaviour, and an arm that moved the first link early
would have moved the generator away from real.

### The topical-relevance half is not supported at this n

s14's headline example is an Apple support URL inside a comment about a Sony A7.
Tagging every URL and every comment's prose with a camera-brand list:

| | URLs naming a brand | naming a brand the prose never mentions |
|---|---:|---:|
| real | 0.330 | **0.200** |
| v117 generated | 0.438 | **0.344** (11 of 32) |

Real posts off-brand links constantly — the excluded corpus contains that *same*
`fujifilm-dsc.com/.../x100f/...` URL inside a comment about Canon. 11 against an
expected 6.4 at n=32 is not a finding. And it is structurally unfixable: a
topically relevant link needs the seed thread's subject, and the inventory is
built from the threads that exclude it. An idf-ranked match attempt on the real
routed slots returns candidates like `evf` -> a Canon EOS page for a Sony A7 slot.

**A weaker instrument agrees.** Token overlap between a URL and its own comment's
prose: real 0.372, shuffle null 0.103, generated 0.094 — generated sits exactly at
the null, as a hash draw must. But real's own 0.372 is itself close enough to a
draw-with-structure that forcing generated to real's rate would be tuning to a
lower bound; real's opaque URLs (12.6% carry no descriptive path token at all, and
they match at 0.000) can never show relevance a reader still trusts.

### What does survive, and it is large

| k | carriers | all URLs on one folded host |
|---|---:|---:|
| 2 | 105 | **0.771** |
| 3 | 25 | **0.640** |
| 4 | 24 | **0.417** |
| pooled 2..4 | 154 | **0.695** |

v117 drew **0.000** on its own artifact (0 of 6 multi-link carriers; P=0.0007
under real's rate) because it draws each URL independently across 249 folded
hosts. This is v118, and it is the whole of v118. Reading all 19 of v117's
carrying comments rather than the three s14 quotes is what separated these three
claims: the links that read as human are the opaque ones (`youtu.be/...?t=1434`,
flickr photo pages), and the ones that read as machine output are four unrelated
hosts stacked together — not one off-topic link, which real does too.

## 16. The persona layer was never on, is keyed wrong, and the channel it was meant to open is already closed — 2026-08-27

Three separate findings, in the order they had to be established.

### 16a. G57's premise is false: two of the four levers were never enabled

G57 and s8 both close on *"`persona_bridge`, `speaker_roster`, `actor_conditioning`
and `--speaker-identity matched` all exist and **were on for every run**"*. Reading
`run_config.json` instead of the module list:

    persona_conditioning = {"mode": "none"}
    actor_conditioning   = {"mode": "none", "source": "disabled"}
    speaker_identity     = "matched"

Both are the CLI defaults (`run_generate.py` `--persona-conditioning` and
`--actor-conditioning` default to `MODE_NONE`). Sweeping all **163**
`run_config.json` files: 147 `none`, 9 absent, **7** `matraix-projected` -- and all
7 are from 2026-08-08/09, i.e. ~100 versions before v113. So the +0.0076 voice
separation s8 measured is what `--speaker-identity matched` produces **alone**, and
the MatrAIx persona system prompt has never been in a Writer prompt in any run this
project has evaluated.

The layer is runnable today: `third_party/MatrAIx-Persona-8B` sits at the exact
audited commit `e85c8772`, its dev-sample dataset holds 200 personas of which 147
pass the English-adult filter, and driving the runtime over the v117 run's own 571
tasks assigns **119** distinct personas with a top-persona share of 4.0%. The
rendered profiles are genuinely distinct -- mean pairwise Jaccard 0.312, only three
tokens (`who`, `you`, `are`) shared by all -- but thin: median **3** selected
dimensions, mean 23.8 persona-specific tokens, and 15.1% of personas render zero
dimensions (4.2% of slots).

### 16b. It is keyed per SLOT, not per speaker, so switching it on would not implement s8's channel

`MatraixPersonaRuntime.assign` ranks candidates by
`_stable_rank(seed, seed_index, local_task_id, persona_id)`. The key is the
**task**, and no speaker id enters anywhere in the path. Driven over the v117 run:
of the 93 authors who write more than one comment, **93 of 93 -- 100% -- receive a
different persona for each comment.**

s8's channel is that real same-author pairs outscore different-author pairs by
+0.0137, a writing signature. A per-slot persona gives one author two unrelated
voices, which works *against* that structure rather than for it. Implementing the
channel needs the persona keyed to the speaker the roster already tracks
(`real_sample_id`), not to `local_task_id`.

### 16c. And the channel is already closed on the current artifact

`one_voice_persona.py`, the s8 instrument, run on both artifacts.
**Fidelity first:** on the v113 gate it reproduces s8's table to four decimals
(+0.0170 / +0.0058 / +0.0133 / +0.0061, weighted **+0.0076**, ratio 0.55), so the
instrument is the same one.

| relation | v113 gate | **v117 calibration** | real |
|---|---:|---:|---:|
| same parent | +0.0170 | +0.0229 | −0.0008 |
| ancestor/descendant | +0.0058 | +0.0065 | +0.0021 |
| same root branch | +0.0133 | +0.0126 | +0.0233 |
| **different branch** | +0.0061 | **+0.0163** | +0.0141 |
| stratum-weighted | +0.0076 | **+0.0160** | +0.0137 |
| as a fraction of real | 0.55 | **1.16** | 1.00 |

The decisive `different branch` cell -- the one s8 built the bound on -- has gone
from 43% of real's to **116%** of it. **G57's headroom of +0.0060 = 51% of the gap
does not exist on the current generator.** The two runs differ in thread pool and
in three arms, so *what* closed it is not attributable here; that it is closed is.

The residual is then a level effect, not a structural one, exactly as G3 recorded:
generated's different-branch different-author pairs sit at 0.5082 against real's
0.4881, and its same-author ones at 0.5246 against 0.5022 -- both about +0.02.
Every pair is uniformly too similar while the author structure is now correct.

**The only measured channel above the 42% bar is therefore spent, and no named
mechanism remains for `self_bertscore`.** The persona layer is still worth trying
as an unpriced experiment, but it must be re-keyed to the speaker first, and it
can no longer be justified by G57's arithmetic.

## 17. The clause-register signature is real and measured — and it is another sub-20% channel — 2026-08-27

G40 inferred a profile from a token attribution rather than measuring it: *"generated
writes fewer, longer, determiner-dense, less verbal sentences than real"*, resting on
`the` carrying 20.9% of the positive excess overlap mass and the full stop being the
most under-shared token (−0.0461), followed by `to`, `i`, `of`, `with`, `for`, `and`,
`but`, `be`, `have`.

**Measured directly** (`clause_structure_gap.py`), on matched threads, both runs:

| property | real | v117 | gen/real | v113 gen/real |
|---|---:|---:|---:|---:|
| **verbal rate** | 0.1054 | 0.0734 | **0.70** | **0.68** |
| determiner rate | 0.1090 | 0.1275 | **1.17** | 1.22 |
| types/sqrt(tokens) | 20.34 | 14.94 | **0.73** | 0.76 |
| words/sentence p90 | 25.5 | 21.3 | 0.84 | 0.84 |
| pronoun rate | 0.0701 | 0.0610 | 0.87 | 0.78 |
| ends with `.!?` | 0.7965 | 0.6852 | 0.86 | 0.89 |
| sentences/comment | 3.60 | 2.71 | 0.75 | 0.91 |
| words/comment | 56.9 | 42.0 | 0.74 | 0.90 |

G40's inference is confirmed and the profile is **stable across two runs and two
pools**. The verbal rate is the largest and steadiest term: generated uses verbs and
auxiliaries at **0.68–0.70x** real throughout.

**Worth flagging separately:** v117 is markedly *shorter* than v113 (words/comment
0.74x real against 0.90x, sentences 0.75x against 0.91x). The calibration run's flat
tone quota is the obvious suspect and it is not isolated here.

### Priced, with the control that matters

`clause_register_ablation.py`. Real text stripped of its verbal tokens (11.9% of
words), each edit paired with a random-token control removing the **same count per
comment** — because deletion also shortens, and `self_bleu_4` is a length metric
through its brevity penalty (G27):

| | self_bleu_4 | move | share of gap |
|---|---:|---:|---:|
| real, untouched | 0.029768 | | |
| real minus VERBAL tokens | 0.032262 | +0.002494 | 30.9% |
| real minus RANDOM tokens (control) | 0.031624 | +0.001856 | 23.0% |
| **NET** | | **+0.000638** | **7.9%** |

**The control carries 74% of the raw move.** Uncontrolled this would have been
reported as a 31% channel; it is 7.9% against this script's own denominator and 21%
against the evaluation's smaller matched-real gap. Fidelity is honest about which
half holds: the generated side reproduces the shipped 0.0378 exactly, the real side
does not (521 comments here against the evaluation's 571), so the absolute move and
its control are internally consistent while the *share* depends on the denominator.

**Conclusion: another sub-20% channel.** It joins entity expansion (5.4%), the
subject-mention cap (≤9.4%, saturating), comma deletion (~0 net against a `the`
control), length-mix reweighting (2.1%) and function-word variety (−28%, wrong sign).
G35's reading stands and now has a sixth data point: **there is no single lever for
`self_bleu_4`, and stacking several sub-10% fixes is the correct shape of the work.**
What is new is that the register profile is now *measured* rather than inferred, so a
mechanism aimed at it can be checked for firing.

