# Predictions for v128 — conversation-shaped reference viewpoints. Written BEFORE the run.

2026-08-28. Control is **v125b on seeds 2–11** (v126c is disqualified: enabling
actor conditioning cut the author repeat rate from 23.9% to 6.8%, G116).

## The defect, in seven measured steps

1. The Planner's only window into real discourse is the reference-viewpoint
   block: 18–36 rows from the evaluation-excluded corpus, each carrying up to
   300 characters of **real text**. Real text reaching the Planner is therefore
   already established practice, not a new liberty.
2. `retrieve_reference_viewpoints` caps each source thread at **2 rows**
   (`if source_counts[source] >= 2: return False`) to stop one reference
   discussion becoming a semantic template.
3. Measured consequence: for seed 2 the window is **36 rows from 22 distinct
   threads**, and **30 of 36 are depth=0** — top-level replies to an OP.
   Seeds 3 and 4 give 24/36 and 22/36 threads at 47% and 67% depth-0.
4. So the Planner has seen thousands of *opening statements* and almost no
   *reply*. It has never seen a rebuttal that quotes its parent, an author
   returning to defend a claim, a joke landing on someone's assertion, a link
   post, or a reaction to a link.
5. The matched real thread supplies no content either — only
   `depth / parent / words / surface`.
6. And that surface label destroys the rest: `surface_only_label` tests
   `words >= 70` **before** it tests for a quote or a link, so of 2,028 real
   comments carrying one, **1,243 (61.3%)** reach the Planner labelled
   `long_turn`, `ordinary_turn`, `short_question` or `micro`.
7. The generator therefore plans what it has been shown: a list of independent
   opening statements. Read against its matched real thread, ours is twelve
   people each nominating a different thing to check, with **nobody disagreeing
   with anybody**; the real one is a four-turn quoted argument plus a joke, an
   unanswered factual question, a link, and a reaction to the link.

This single defect predicts both failing metrics: every comment performing the
same speech act is high `self_bertscore`, and performing it in the same
evaluative-checklist register is high `self_bleu_4`.

## What the arm does

`--reference-scope conversation`; `comment` reproduces v125b byte-for-byte.

**Additive.** The existing 36 topically-ranked rows are untouched — they supply
domain content. The arm appends **2–3 complete conversation fragments**: every
retained reference from one excluded thread, rendered in reply order with its
parent structure, chosen for structural richness (contains a depth>=1 reply, and
where available a quote, a link, or a returning author).

The fragments are chosen **topically DISTANT** from the seed. That is stronger
than the cap it works alongside, not weaker: a distant thread has no content
worth copying, so the Planner can only take the discourse shape from it, which
is the thing being transferred. Nothing about the matched real thread is
revealed, and no constant is written down — the fragments are whatever the
configured domain's own excluded threads do, so the arm follows the domain.

## Compliance gate — OFFLINE, before the run is priced (E15, lessons 2026-08-27)

v126 cost $0.81 to discover a rendering failure that four offline lines would
have caught. This arm does not get priced until these pass in a test:

1. The rendered Planner prompt contains a conversation block.
2. It carries **>= 2 fragments**, each with **>= 4 comments**, each containing
   **at least one depth>=1 reply**.
3. Fragment source threads are disjoint from the seed pool (no evaluation
   thread can appear).
4. Mean topical overlap between fragment text and the seed is **below** that of
   the 36 ranked rows — proving the fragments were chosen for structure, not
   content.

## Predictions — judged as written

**Arm objective, measured on the arm's own output before any metric is read**
(G87/G88). All against v125b:

| quantity | v125b | prediction |
|---|---|---|
| comments carrying a quote or link | 3.38% | **>= 6%** (real 8.44%) |
| comments carrying a URL | 0.00% | **> 0%** |
| authors posting 2+ times | 23.9% | **>= 27%** (real 32.9%) |
| longest exchange between one author pair | ~1 | **>= 2** |

**If none of these four moves, the arm did not fire and the metrics are void.**

**Primary.** `self_bertscore` Cliff **+0.90 -> <= +0.60**, `self_bleu_4`
**+0.38 -> <= +0.30**. Note the resolution limit that four paid runs taught this
session: the sd of Cliff d at N=10 is **0.265**, so a move smaller than ~0.53
cannot be distinguished from zero. A result inside that band is reported as
"no measurable effect", not as a small win.

**Guard.** No metric currently above p>0.05 may fall below it. `avg_depth`,
`structural_virality` and `mean_story_probability` are the ones a change to
thread shape could plausibly break; if any leaves p>0.05 the arm is rejected
whatever the priority metrics do.

**Null result.** If the four objective quantities move and the two metrics do
not, then discourse structure is not what the metrics measure, and the honest
next step is to stop treating them as a content problem.

## Cost

<= $5.00 at N=10. Do not spend it until the offline gate above is green.
