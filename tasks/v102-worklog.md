# v102 worklog — the `hard_disagree_rate` diagnosis

Status: **diagnosis complete, no code shipped.** Reproduce every number with

```bash
.venv/bin/python generalized_card/analysis/disagreement_diagnosis.py all
```

Run under diagnosis: `generalized_card_camera_gpt54_v101_register_n10_20260820_v1`
(N=10, 9/0/3, `hard_disagree_rate` PASS at MWU 0.1735 with **Cliff +0.37**).

Read `docs/ORIENTATION.md` first. This file holds the evidence; nine hypotheses
are rejected below and two survive.

---

## 0. What the metric actually is

Read from `scripts/evaluation/score_thread_disagreement.py`, not recalled:

- One **pair per comment**: `(parent text, comment text)`. A root comment's
  parent is the post's title plus selftext — so root comments are pairs too, and
  they are **37% of all pairs**.
- `hard_disagree_rate` = share of pairs whose **argmax** over
  `{disagree, neutral, agree}` is `disagree`.
- The saved head expects 1536 features: 768 RoBERTa pooled + 768 relation. The
  graph path is absent, so a zero vector is used.
- `is_usable_pair` needs parent ≥ 3 words and reply ≥ 2 words.

**Trap found: `pair_count` in `matched_*_thread_scores.csv` is not the stance
pair count.** It is `n(n-1)/2` from the pairwise metrics (45 comments → 990). The
stance pair count is ≈ the comment count. Do not read that column for this metric.

---

## 1. The head is nearly degenerate, and the gap is a uniform translation

All three class probabilities sit inside ≈ [0.26, 0.41] — a softmax barely off
uniform. The metric is an argmax on a knife edge, so it moves on a probability
shift of a few thousandths.

| corpus (reply pairs) | n | mean p(disagree) | margin p10 | med | p90 | rate |
|---|---:|---:|---:|---:|---:|---:|
| generated | 349 | 0.3244 | −0.0919 | −0.0370 | +0.0213 | 0.2235 |
| matched-real | 349 | 0.3157 | −0.0958 | −0.0551 | +0.0090 | 0.1433 |
| excluded-real | 6,966 | 0.3198 | −0.0990 | −0.0496 | +0.0194 | 0.1797 |

Shifting **every** matched-real margin by a constant reproduces the generated
rate: +0.010 → 0.1920, +0.015 → 0.2178, +0.020 → 0.2579. So the defect is a
**uniform ≈ +0.017 translation of the whole decision margin**, not a subset of
bad pairs. This is the same signature `self_bertscore_mean_f1` has (+0.02 on
every pair, flat under trimming) and it is why no single phrase explains it.

---

## 2. The whole gap is the reply-pair conditional; root pairs already match

| corpus | n | root share | P(d \| root) | P(d \| reply) | overall |
|---|---:|---:|---:|---:|---:|
| generated | 526 | 0.337 | **0.0621** | **0.2235** | 0.1692 |
| matched-real | 476 | 0.267 | **0.0630** | **0.1433** | 0.1218 |
| excluded-real | 11,119 | 0.374 | 0.0667 | 0.1797 | 0.1375 |

- **Root pairs are matched to three decimal places** (0.0621 vs 0.0630).
- Reply pairs are **1.56×** real.
- Generated and matched real have **exactly 349 reply pairs each** — the matched
  sampler's reply structure is exact, so this is not a coverage artifact.
- Giving generated the real reply conditional and keeping everything else:
  0.1692 → **0.1160**, i.e. the reply conditional is 100% of the gap.

Structure is not the cause: generated carries *more* root pairs (0.337 vs 0.267)
and roots are the low-disagreement kind, so the mix works slightly in its favour.

---

## 3. What the head reads: the reply, and stance tokens inside it

A TF-IDF surrogate fitted on excluded-real reply pairs, grouped 5-fold by thread:

| input | vocab | AUC | margin R² |
|---|---:|---:|---:|
| reply only | 12,308 | **0.740** | 0.191 |
| parent only | 19,035 | 0.579 | 0.009 |
| parent + reply | 31,019 | 0.702 | 0.155 |

The parent alone is barely above chance and adding it **lowers** AUC. The
`disagree` class is keyed by **explicit stance tokens, agreement ones included**:

`agree(+0.215) agreed(+0.132) yup(+0.128) yeah(+0.123) exactly(+0.120)
true(+0.108) yep(+0.099) absolutely(+0.093) yes(+0.090) disagree(+0.084)
right(+0.079) wrong(+0.070)`

The head does not separate agreement from disagreement well. It separates
**"this comment takes an explicit position on the parent"** from "this comment is
just talking", and then splits that mass near-arbitrarily. That is why planned
stance is *anti*-correlated with the label in the artifact: planned `agree` slots
are labelled disagree 0.255 of the time, planned `disagree` slots 0.181, planned
`uncertain` 0.018.

The parent is not irrelevant, though — a 2×2 crossover on the real scorer:

| condition | rate |
|---|---:|
| generated reply × its own parent | 0.2235 |
| generated reply × a shuffled generated parent | 0.1719 |
| generated reply × a random real parent | 0.1404 |
| generated reply × an empty parent | 0.1662 |
| real reply × its own parent | 0.1377 |
| real reply × a shuffled real parent | 0.1039 |
| real reply × an empty parent | 0.0942 |

Destroying the relation costs generated −0.052 and real −0.034, and **the gap
survives every control** (+0.072 even with no parent at all). So ≈ 80% of the gap
is intrinsic to the reply text and ≈ 20% is the relation being tighter.

---

## 4. SURVIVES — the assigned opener is not realized, and `polarity_token` is
the highest-disagreement opener

`opener_type` is scheduled per slot from the domain profile's measured shares and
**the instruction does reach the Writer prompt at exactly those shares** (checked
by grepping the 532 saved prompts: `polarity_token` 28/532 = 0.0526 against a
profile share of 0.0526). It is then not obeyed:

| planned opener | n | obeyed | top realized |
|---|---:|---:|---|
| `quote` | 18 | 1.000 | quote:18 |
| `conditional` | 15 | 1.000 | conditional:15 |
| `first_person` | 100 | 0.960 | first_person:96 |
| `polarity_token` | 28 | 0.893 | polarity_token:25 |
| `question` | 21 | 0.476 | question:10, content_phrase:9 |
| `content_phrase` | 224 | **0.460** | content_phrase:103, noun_phrase:49, first_person:26, **polarity_token:21** |
| `imperative` | 10 | 0.400 | content_phrase:6, imperative:4 |
| `noun_phrase` | 59 | 0.254 | content_phrase:42, noun_phrase:15 |
| `discourse_marker` | 38 | **0.184** | **polarity_token:19**, content_phrase:11, discourse_marker:7 |
| `address` | 13 | 0.077 | content_phrase:9 |

Realized against measured share:

| opener | measured | realized | ratio |
|---|---:|---:|---:|
| `polarity_token` | 0.0526 | **0.1274** | **2.42** |
| `discourse_marker` | 0.0726 | 0.0247 | 0.34 |
| `address` | 0.0249 | 0.0038 | 0.15 |
| `conditional` | 0.0293 | 0.0589 | 2.01 |
| `link` | 0.0053 | 0.0000 | 0.00 |

`content_phrase` → `noun_phrase` and `noun_phrase` → `content_phrase` are a
harmless confusion between two content-bearing classes. The damaging leak is
narrow: **36 of 349 reply slots (10.3%) prepend an agreement token they were not
assigned**, 18 from `discourse_marker` and 16 from `content_phrase`.

Why those two cells: *"Open with a short conversational connective before the
point"* names a category, and the Writer resolves it to `Yeah,`. Real
`discourse_marker` openers are `thanks / lol / haha / oh / well / honestly /
personally / also / and / but / so`. This is the project's standing failure mode
— **prose describing a register is not obeyed; a drawn concrete surface act is**
(`TONE_DEFINITIONS["polite"]` realized 19.3%, `sentence_rhythm` realized within
0.016 of its measured rate).

`polarity_token` is also the **highest-disagreement opener there is**, reply pairs:

| opener | real prev | real P(d) | gen prev | gen P(d) |
|---|---:|---:|---:|---:|
| `polarity_token` | 0.068 | **0.457** | 0.140 | **0.612** |
| `content_phrase` | 0.403 | 0.169 | 0.350 | 0.131 |
| `first_person` | 0.172 | 0.110 | 0.238 | 0.145 |
| `discourse_marker` | 0.095 | 0.120 | 0.032 | 0.000 |

Prevalence contribution of the opener mix: **+0.0128** of the +0.0802 reply gap
holding real conditionals; **−0.0326** holding generated's own. So 16–41%.

### The causal measurement

An exact ablation harness re-scores text with the evaluator's own scorer classes.
Fidelity first: it reproduces the shipped artifact on all 526 pairs, label
agreement **1.0000**, max |Δp| **0.000000**. Then, on reply pairs:

| edit | rate | slots edited |
|---|---:|---:|
| baseline (v101 as shipped) | 0.2235 | 0 |
| **obey the plan** — strip only the *unassigned* polarity openers | **0.1862** | 36 |
| strip *every* polarity opener (over-correction below the real rate) | 0.1633 | 49 |
| matched real | 0.1433 | — |

**Obeying an instruction the pipeline already renders is worth −0.0373, i.e. 47%
of the reply-pair gap.** Guardrail checked with the exact `self_bleu_4` scorer on
the same edit: 0.03330 → 0.03297, down in 10 of 10 threads. No harm.

Caveat, stated plainly: this is **textual surgery, not regeneration**. A Writer
told to open differently writes a different sentence, not the same sentence minus
`Yeah,`. Treat 0.1862 as the direction and the order of magnitude, not a forecast.

---

## 5. SURVIVES — generated replies echo the parent's words 1.4–1.6× as often

Share of the reply's content words that also occur in its parent (stop-worded):

| corpus | reply pairs | mean echo |
|---|---:|---:|
| generated | 349 | **0.2145** |
| matched-real | 349 | 0.1542 |
| excluded-real | 6,966 | 0.1367 |

In real text P(disagree) rises **monotonically** with echo across all six bins —
0.150, 0.156, 0.188, 0.205, 0.207, 0.232 — and generated's within-bin
conditionals track real's closely (+0.007, −0.002, +0.047, +0.001, +0.037,
+0.048). The prevalence is what differs: real puts 33% of replies in the lowest
echo bin, generated 15%; real 12% in the highest, generated 27%.

Counterfactual: generated's own conditionals at the real echo distribution →
**0.1972** against a shipped 0.2235 and an excluded-real 0.1797 — **55% of the
gap is prevalence.**

Not a length artifact: conditioned on both parent-length and reply-length bands,
generated echo is **1.27–2.04× real in all ten populated cells**, never at parity.

Reading the text says the same thing. Generated replies stay inside the parent's
frame and deliver a judgement on it; real replies use the parent as a springboard
and go elsewhere — a joke about proprietary memory cards, a remark about the
*person* ("You seem very extreme in all your opinions however haha"), their own
purchase. This is the user's own complaint, criterion 2: *很容易去讨论同一个话题.*

The existing `context_transform` arm does **not** fix it. Mean echo by transform:
`parent_hidden` 0.259, `normal` 0.236, `parent_gist` 0.211, `parent_jittered`
0.202, `minor_detail_focus` 0.175 — all far above real's 0.137, and the highest
is the mode where the Writer **never sees the parent text**. The echo comes from
the plan's parent-local topic, not from copying visible text.

---

## 6. REJECTED — nine hypotheses, with what rejected each

1. **Graph-feature asymmetry.** The scorer defaults to `--graph-author reply` and
   looks up RGCN author embeddings; generated authors are synthetic. *Rejected:*
   coverage is 0.0000 on generated and 0.0039 on real (88 of 22,809 pairs). Both
   are effectively zero-graph.
2. **Root-pair title asymmetry.** Real root pairs get `title\n\nselftext` as the
   parent, generated get `post["content"]`. *Rejected:* generated `content`
   already begins with the title, so the two are equal after whitespace collapse.
3. **Environment drift between the real tables and the generated ones.**
   *Rejected:* re-scoring all 476 matched-real pairs today reproduces the stored
   labels exactly — agreement 1.0000, max |Δp| 0.000000.
4. **The metric is relational.** *Mostly rejected:* reply-only surrogate AUC
   0.740 against parent-only 0.579, and the gap is +0.072 even with an empty
   parent. The relation is worth ≈ 20%, not the mechanism.
5. **The adjudication frame** (v100's target). *Rejected:* stripping every frame
   span hits **11 of 349** generated replies and moves the rate **−0.0029**.
6. **Contrastives** (`but / however / though / actually / still`). *Rejected:*
   stripping them **raises** the generated rate **+0.0143**.
7. **The closing sentence.** *Rejected:* dropping the last sentence **raises**
   generated **+0.0258** while barely moving real (−0.0048).
8. **Hedges.** *Rejected:* 10 generated hits, **0.0000** move. (It does record a
   criterion-2 tell: generated hedges on 2.9% of replies against a real 17.6%.)
9. **Acknowledgement tokens anywhere in the reply.** *Rejected as the primary
   term:* prevalence is generated 0.413 against matched-real 0.394 — nearly
   equal — and the gap is present in **both** cells (with-ack +0.055, without-ack
   +0.101). Only the **opening position** carries it.
10. **A thread-wide lexical over-coupling shared with `self_bleu_4` and
    `self_bertscore_mean_f1`.** *Rejected:* within-thread pairwise Jaccard is
    generated 0.0411 against matched-real 0.0335 (1.23×), but excluded-real
    threads average 0.0441 — there is no thread-wide excess. The coupling is
    specifically parent→child, so it does not unify these three metrics.

Measured but **not** to be acted on: stripping negations moves generated −0.0201
on 105 slots. That is the family the politeness work already established
generated uses *less* of than real (−0.767 impolite-vocabulary excess against a
+8.381 polite-vocabulary deficit). Suppressing it would trade this metric against
`polite_rate` and `impolite_rate`.

Also checked and clean: the opener defect does **not** explain the politeness
trio. Real polite rate by opener runs 0.18–0.47 and generated 0.02–0.15 in
**every** class — the tone gap is flat across openers, so it is a separate defect.

---

## 7. What v102 should be, and what it is worth

**Arm `--opener-realization {measured,off}`; `off` reproduces v101.**

1. Measure the real opening-connective vocabulary on the evaluation-excluded
   corpus and **draw one per `discourse_marker` slot**, naming it — "open with
   `honestly`, then the point" — instead of describing the category.
2. On every slot whose assigned opener is **not** `polarity_token`, render an
   explicit token-level prohibition that names the tokens (`yeah / yep / yes /
   no / nope / fair / right / exactly / agreed / true / same`). The project has a
   track record with named-token suppression: v98 took the semicolon 0.109 →
   0.023 and the dash clause 0.299 → 0.071.
3. Leave `polarity_token` slots alone. They are obeyed at 0.893 and their
   measured share **is** the target — this must not become a global ban.

**Predictions, written before any run, with the population named.** The rule
reaches 504 of 532 slots (every slot not assigned `polarity_token`); the excess
lives on 36 of 349 reply slots; compliance on a named-token rule has run 0.3–0.7
in this project.

| quantity | v101 | predicted v102 | real |
|---|---:|---:|---:|
| realized `polarity_token` share | 0.1274 | 0.06–0.08 | 0.0526 |
| realized `discourse_marker` share | 0.0247 | 0.04–0.06 | 0.0726 |
| reply-pair `hard_disagree_rate` | 0.2235 | 0.19–0.20 | 0.1433 |
| thread `hard_disagree_rate` | 0.1692 | 0.145–0.155 | 0.1218 |
| `hard_disagree_rate` Cliff | +0.37 | +0.15 to +0.25 | — |

**It does not reach |Cliff| ≤ 0.10 on its own, and it should not be sold as if it
does.** The parent-echo term is the other half of the gap and has no mechanism
yet. Guardrails: `polite_rate` and `impolite_rate` must not move (the tone gap is
flat across opener classes, so any movement means the rule leaked into register);
`self_bleu_4` moved −0.0003 under the surgery, so anything larger than noise
there is unexplained and must be read before shipping.

---

## 8. My own errors in this diagnosis, and what they cost

- **The first ablation harness failed fidelity and I nearly read it.** I took the
  reply text from `discussion.json` instead of the scorer's own pair record, so
  it kept the newlines the scorer's `clean_text` collapses. 11.2% of labels
  flipped and the pooled rate moved 0.1692 → 0.1730 — comparable to a third of
  the effect being measured. Fixed by reading `pair["reply_text"]`; fidelity is
  now asserted **before** any edited number is printed.
- **I double-counted the real corpus.** One Reddit post can sit under two product
  folders, so a naive read of `data/raw/discussions/` inflates the matched set
  1.24× and the corpus 1.32×. Rates were unaffected (the duplicates are exact)
  but every weighted figure was. Deduplicated by `(thread_id, reply_id)`.
  **`generalized_card/analysis/politeness_diagnosis.py` does not dedupe** — its
  conditionals are safe, its pooled weights are not.
