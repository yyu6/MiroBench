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
