# Thread-level behavioural metrics — definitions

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

## 4. What the five together are for

They separate three axes that a surface-similarity metric cannot:

| axis | metric |
|---|---|
| interpersonal stance | `polite_rate`, `neutral_rate`, `impolite_rate` |
| conflict structure | `hard_disagree_rate` |
| affective variety | `emotion_entropy` |

A generator can match a real thread on wording and still fail all five, by being
uniformly courteous, uniformly agreeable, and uniformly flat. They exist to make
that failure visible.

---

## 5. Statistical protocol

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
