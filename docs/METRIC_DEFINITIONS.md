# Thread-level metrics — definitions

All twelve metrics used in the evaluation suite, as implemented.

All five metrics are **thread-level scalars**: each generated thread yields one
value, each matched real thread yields one value, and the two samples are
compared with a two-sided Mann–Whitney U test and a two-sided Kolmogorov–Smirnov
test. A metric passes when **both** p > 0.05 (distributions indistinguishable).

Notation: a thread `T` has comments `c_1 … c_n` in traversal order; a reply pair
`(p → r)` is a comment together with its parent.

---

## 1. Politeness rates — `polite_rate`, `neutral_rate`, `impolite_rate`

**Instrument.** `Intel/polite-guard` (HuggingFace), a four-way **single-label**
sequence classifier over the classes

```
polite   somewhat_polite   neutral   impolite
```

**Per comment.** Each comment is classified independently; the predicted label
is the argmax over the four class probabilities. No threshold is applied — every
comment receives exactly one label.

**Per thread.** For each class `L`,

```
L_rate(T) = |{ c ∈ T : pred(c) = L }| / n
```

so the four rates sum to 1 by construction. Three of the four are reported as
metrics (`somewhat_polite_rate` is recorded but not in the headline set).

**Interpretation.** The *distribution of interpersonal stance* a thread displays.
`polite_rate` is the share of comments a politeness classifier reads as warm or
appreciative; `impolite_rate` the share it reads as dismissive or hostile;
`neutral_rate` the share it reads as affectively flat. Real Reddit threads mix
all three; a generator that is uniformly courteous shows an inflated
`polite_rate` and a depressed `impolite_rate`.

**Caveat for the paper.** These are classifier outputs, not human judgements.
`polite-guard` is a confident four-way classifier and its decision for a
multi-sentence comment can hinge on whether a single sentence reads as
appreciative, so the rates should be read as *the behaviour of a fixed detector*
rather than as ground-truth politeness.

---

## 2. `hard_disagree_rate`

**Instrument.** The local `Stance_Rel` checkpoint — a RoBERTa encoder plus a
trained stance classification head over

```
disagree   neutral   agree
```

The saved head expects a 1536-dimensional input (768 RoBERTa pooled features
plus 768 relation/graph features). The original graph-inference path is not
distributed with the checkpoint, so this evaluator supplies a **zero relation
feature** and uses the trained RoBERTa and stance weights alone. This must be
stated in the paper: the head is used out of its full training configuration.

**Per pair.** Every parent→reply pair inside a thread is encoded as a text pair
and assigned an argmax stance label.

**Per thread / corpus.**

```
hard_disagree_rate = |{ pairs with pred = "disagree" }| / |pairs|
```

It is a *rate over reply pairs*, not over comments — a thread with no replies
contributes no pairs. The companion `mean_disagree_probability` averages the
soft `disagree` probability over the same pairs.

**Interpretation.** How often a reply takes an opposing position to the comment
it answers. It measures **conflict structure**, not tone: a polite, well-argued
rebuttal counts as disagreement, and a rude agreement does not.

**Known behaviour.** In this project the stance head is close to degenerate on
Reddit text and keys strongly on agreement tokens near the *start* of a reply,
which makes `hard_disagree_rate` behave partly as an opener metric. Report it
with that limitation.

---

## 3. `emotion_entropy`

**Instrument.** `SamLowe/roberta-base-go_emotions`, a **multi-label** classifier
over the 27 GoEmotions categories plus `neutral` (28 labels).

**Per comment.** All 28 sigmoid probabilities are recorded. The comment's
**dominant emotion** is the single highest-probability label.

**Per thread.** Let `D` be the multiset of dominant emotions across the thread's
`n` comments, and `count(e)` the number of comments whose dominant emotion is
`e`. Then

```
emotion_entropy(T) = − Σ_e  (count(e)/n) · ln(count(e)/n)
```

i.e. **Shannon entropy in nats over the empirical distribution of dominant
emotions**, computed over observed emotions only (zero counts are dropped). A
normalised variant `emotion_entropy / ln(28)` is also recorded.

Two companions come from the same pass:

```
dominant_emotion_share(T) = max_e count(e) / n
emotion_shift_rate(T)     = |{ i : dom(c_i) ≠ dom(c_{i+1}) }| / (n − 1)
```

**Interpretation.** `emotion_entropy` is the **affective variety** of a thread.
It is high when many different emotions take turns being dominant, and low when
one emotion dominates. It is deliberately computed over *dominant labels*, not
over averaged probabilities, so it measures how varied the thread's comments are
from each other rather than how emotional they are on average. `emotion_shift_rate`
captures the same variety in sequence, and `dominant_emotion_share` its converse.

**Note on scale.** Entropy in nats depends on the number of *distinct* dominant
emotions realised, so it is sensitive to thread length; comparing generated
against a **size-matched** real thread, as this evaluation does, controls for
that.

---

---

## 4. `self_bleu_4` — surface-form uniformity

**No model.** Pure n-gram arithmetic.

**Tokenisation.** `[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?|[^\w\s]`, lowercased —
words keep internal apostrophes and hyphens; punctuation is its own token.

**Per pair.** BLEU is directional (hypothesis-side precision + brevity penalty),
so each unordered pair is scored **symmetrically**:

```
pairBLEU(a,b) = ½ · [ BLEU(a | ref=b) + BLEU(b | ref=a) ]
```

with, for `max_order = 4`,

```
p_k  = (clipped_overlap_k + 1) / (hyp_ngrams_k + 1)      k = 1..4   ← add-one smoothing
BP   = 1                        if len(hyp) >  len(ref)
     = exp(1 − len(ref)/len(hyp))  otherwise
BLEU = BP · exp( (1/4) · Σ_k ln p_k )
```

`clipped_overlap` is standard BLEU clipping: each hypothesis n-gram counts at
most as many times as it appears in the reference.

**Per thread.**

```
self_bleu_4(T) = mean over all C(n,2) unordered pairs of pairBLEU
```

`self_bleu_2` and `self_bleu_3` are the same quantity at `max_order` 2 and 3.

**Interpretation.** How much comments in one thread **reuse each other's
wording**. Higher = more templated. Reporting orders 2/3/4 separates *local word
choice* (2) from *phrase reuse* (4) — in this project the 2-gram order fails far
worse than the 4-gram one, which localises the defect at the two-token scale.

**Paper note.** The add-one smoothing means an empty overlap gives a small
positive score, so absolute values are not comparable to unsmoothed BLEU
implementations. Only generated-vs-matched-real comparisons under the *same*
implementation are meaningful.

---

## 5. `self_bertscore_mean_f1` — contextual-embedding uniformity

**Instrument.** BERTScore with `microsoft/deberta-xlarge-mnli`, **layer 40**,
`idf = False`, `rescale_with_baseline = False`, CPU. Hash recorded in every
output: `microsoft/deberta-xlarge-mnli_L40_no-idf_version=0.3.12(hug_trans=4.48.0)`.

**Per pair.** For every unordered comment pair, BERTScore performs **greedy
token alignment** between the two token sequences' contextual embeddings: each
token in A is matched to its most cosine-similar token in B and vice versa.
Precision is the mean over A's tokens, recall the mean over B's, F1 their
harmonic mean.

**Per thread.**

```
self_bertscore_mean_f1(T)   = mean over all pairs of F1
self_bertscore_median_f1(T) = median over all pairs
self_bertscore_top_k_mean_f1(T) = mean over the k highest-F1 pairs
```

**Interpretation.** Whether comments **say the same kind of thing in the same
kind of way**, at the level of contextual word meaning rather than exact tokens.
It is the strictest of the three uniformity metrics: two comments can share no
n-grams and still align well if their tokens sit in similar contexts.

**Why mean and top-k differ.** `mean` reads the *typical* pair, `top_k` the
*most similar* pairs. In this project `top_k` passes while `mean` fails, which
says the defect is a raised **floor** — no genuinely unrelated pairs — rather
than a few near-duplicates. Report both.

---

## 6. `semantic_mean_cosine` — semantic uniformity

**Instrument.** `sentence-transformers/all-mpnet-base-v2`, embeddings
L2-normalised (`normalize_embeddings=True`).

**Per pair.** Cosine similarity between the two comments' sentence embeddings.
With normalised vectors this is a dot product.

**Per thread.**

```
semantic_mean_cosine(T)     = mean over all C(n,2) pairs
semantic_median_cosine(T)   = median
semantic_top_k_mean_cosine(T) = mean of the k largest
semantic_p90_cosine(T)      = 90th percentile
```

**Interpretation.** Whether comments in a thread are **about the same thing**.
This is the topical axis, and it is deliberately separate from `self_bleu` and
`self_bertscore`: a thread can be topically varied and lexically templated, or
the reverse.

**Why it matters for the paper.** In this project every cosine metric passes
while every BERTScore metric fails. Since one scores sentence-level *meaning*
and the other token-level *alignment*, that split is the evidence that the
remaining defect is realization, not planning — the threads are spread correctly
in meaning and too close in wording.

---

## 7. `mean_story_probability` — personal-narrative content

**Instrument.** `mariaantoniak/storyseeker`, a RoBERTa binary classifier for
"does this online post contain a story". The HuggingFace config exposes generic
labels; verified by sanity check as `LABEL_0 = not_story`, `LABEL_1 = story`.

**Per comment.** The `story` class probability.

**Per thread.**

```
mean_story_probability(T) = mean over comments of P(story)
story_rate(T)             = |{ c : argmax = story }| / n
```

`mean_story_probability` uses the **soft probability**, `story_rate` the hard
label. The soft version is the headline metric because it is stable on threads
where many comments sit near the decision boundary.

**Interpretation.** How much of a thread is **first-person anecdote** rather
than advice, questions, or specification talk. Real Reddit threads carry a
characteristic share of personal stories; a generator that answers every slot
with a tidy recommendation shows a depressed value.

---

## 8. `length_cv` — comment-length dispersion

**No model.** Length is whitespace token count, `len(text.split())`.

```
length_cv(T) = std(lengths) / mean(lengths)
```

with companions `length_std` and `length_iqr`.

**Interpretation.** How **unevenly sized** the comments are. The coefficient of
variation is used rather than the raw standard deviation so that threads of
different typical comment length are comparable. Real threads mix one-line
reactions with long detailed replies; a generator that writes every comment to a
similar length shows a depressed CV.

---

## 9. `avg_depth` — reply-tree depth

**No model.** Computed on the comment tree by BFS, with **top-level comments at
depth 1**:

```
depth(c) = 1                    if c replies to the post
         = depth(parent) + 1    otherwise

avg_depth(T) = mean over all comments of depth(c)
```

with companion `max_depth`.

**Interpretation.** How **deep the conversation nests**. A thread of 50
independent top-level replies has `avg_depth = 1`; a long back-and-forth chain
pushes it up. It captures conversational structure, not content.

---

## 10. `structural_virality` — Wiener index of the reply tree

**No model.** The **average shortest-path distance between all unordered pairs
of comments**, on the reply tree treated as an **undirected** graph:

```
structural_virality(T) = ( Σ_{i<j} d(c_i, c_j) ) / C(n,2)
```

where `d` is graph distance (edge = parent–child link). Pairs in disconnected
components are skipped. This is the Wiener index of Goel et al. (2016), the
standard measure distinguishing **broadcast** from **viral** diffusion.

**Interpretation.** Low values = a **star** (everyone replies to the post
directly). High values = a **deep, branching** conversation where two random
comments are many hops apart. It is complementary to `avg_depth`: a single long
chain and a bushy tree can share an average depth but differ in virality.

---

## 11. What these twelve are for

The twelve span four axes that no single one of them can capture:

| axis | metrics | model? |
|---|---|---|
| **uniformity** — do the comments repeat each other | `self_bleu_4` (n-grams), `self_bertscore_mean_f1` (token alignment), `semantic_mean_cosine` (sentence meaning) | 2 of 3 |
| **behaviour** — what kind of speech acts occur | `polite_rate`, `neutral_rate`, `impolite_rate`, `hard_disagree_rate`, `emotion_entropy`, `mean_story_probability` | all |
| **structure** — the shape of the reply tree | `avg_depth`, `structural_virality` | none |
| **form** — how the comments are sized | `length_cv` | none |

The three uniformity metrics are deliberately not redundant: they operate at the
n-gram, contextual-token, and sentence level respectively, and in this project
they **disagree in sign** — the threads are more semantically spread than real
while being more lexically uniform. That disagreement is only visible because
all three are reported.

The six behavioural metrics exist because a generator can match a real thread on
wording and still be obviously synthetic: uniformly courteous, uniformly
agreeable, uniformly flat, and never telling a personal story.

The three model-free metrics (`length_cv`, `avg_depth`, `structural_virality`)
are the cheapest and the hardest to game — they depend only on the reply graph
and on token counts, so they cannot be moved by rewriting text.

---

## 12. Statistical protocol

For each metric, the generated threads and their **matched real threads** (same
source post, same target size) give two samples of thread-level values.

- Mann–Whitney U, **two-sided** — do the two samples differ in location?
- Kolmogorov–Smirnov, **two-sided** — do they differ in distribution shape?
- Pass = both p > 0.05.
- Effect size = Cliff's delta, `d = (#{x>y} − #{x<y}) / (n_gen · n_real)`.

The standard deviation of Cliff's `d` under the null is
`(2/N)·√((2N+1)/12)` — **0.265 at N=10**, **0.116 at N=50**. A p-value at N=10
is therefore a weak instrument; effect sizes are the stable quantity and
p-values should be reported alongside N.
