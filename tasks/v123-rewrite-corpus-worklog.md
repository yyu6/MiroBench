# v123 — mining the 7,420 verified rewrites (opened 2026-08-28)

## Why this exists

The user pushed back twice, correctly:

1. I had *counted* the self-loop reviser's success rate (G85) but **never opened
   it to see what its rewrites actually do**. Counting a mechanism is not
   learning from it.
2. My proposed fix — write N candidates at write time, score each, keep the best
   — is **the reviser relocated, not a new method**. Explicitly rejected.

The ask: use the reviser's *evidence* to make the **Planner and Writer** produce
better text in the first pass.

## The asset nobody had used

Sweeping `artifacts/` for reviser reports yields **9,174 accepted rewrites**,
**7,420 unique**, and **every one has `gap_reduction > 0`** — i.e. each is a
verified before/after pair that provably moved thread Self-BLEU toward its real
target. Mean pair-Self-BLEU drop **0.00455**. Saved to
`scratchpad/rewrite_pairs.json` for analysis.

This is the largest untapped resource in the project: thousands of labelled
examples of "this text was too similar; this text was less similar; same
meaning".

## What the reviser actually does, measured (not read off its docstring)

Over 4,000 pairs:

| property | value |
|---|---|
| token-set Jaccard old→new | median **0.584** |
| length change | median **+1 token** |
| first-3-word opener changed | **86.9%** |
| vocabulary retained, first third | **0.392** |
| vocabulary retained, rest | **0.546** |

**It keeps 58% of the vocabulary and nearly the exact length, and concentrates
its edits at the start of the comment.** It is forbidden from adding facts,
changing stance, or changing who is replying to whom — so the entire
`self_bleu_4` movement is achieved **without changing meaning at all**.

## First hypothesis tested — and rejected

**H: the opener is the locus, so a Writer rule about openers would capture it.**

Rewrites that changed the first three words average `gap_reduction` **0.000538**
against **0.000457** for those that did not — a ratio of only **1.18x**
(n=5,976 vs 1,146). The head-vs-tail Jaccard split confirms edits *cluster* at
the start, but the gain does not follow the opener.

**Conclusion: the win is a distributed re-path of the whole sentence, not a
swappable opener.** An opener-rule arm would have captured ~18% of a small
effect. Recorded so nobody builds it.

## Parallel exploration launched (4 agents, each anchored to a prior finding)

1. **Mine the corpus for writer-time rules** — top vs bottom decile of
   `gap_reduction`, surface deltas correlated with the win, positional patterns.
   Anchored to E4 (name a concrete token, not a category).
2. **Planner-side levers** — field cardinality per thread, which assignment
   collisions actually predict text similarity, and G35's lane partition
   (`forbidden_decision_subjects` on 532/532 slots).
3. **Where real threads get their diversity** — author structure, missing comment
   categories, the low-similarity tail, topic drift. Excluded corpus only.
4. **The selfbert tail-repair script** — the sibling targeting the embedding
   metric; what it changes about *meaning* rather than wording, since lexical
   variation alone cannot move an embedding metric.

Each is anchored to an established result, not a guess. Results and verdicts
appended below as they land.

## Agent 1 result, and the verification that changed half of it

The corpus-mining agent returned a ranked rule list. Its central finding is
**methodologically strong and I am adopting it**: a matched within-source test
(1,832 sources with ≥2 competing rewrites, so the input is held constant) shows

| held constant | varied | P(higher gain) | z |
|---|---|---|---|
| 4-gram novelty (reordering) | **unigram novelty (new words)** | **62.9%** | **+7.1** |
| unigram novelty | 4-gram novelty (reordering) | 49.0% | −0.5 |

**Reordering clauses and swapping connectives buys nothing. Replacing content
words is the active ingredient.** This independently confirms my own opener test
(1.18x, weak) and explains it: the reviser's *declared* moves — `connector_swap`
is 28% of all rows — are house style, not mechanism.

It also correctly flagged four deltas that are large but inert within-source
(commas +32% z=−2.0, colons +84% z=+1.4, ellipsis removal fails the level test
p=0.7, longer sentences has the wrong-signed level correlation). Those are the
moves a naive diff-reading would have encoded.

### But the rules were mined inside the rewrite corpus, which is two domains
### (cameras AND credit cards) and is not the target distribution.

The project rule is to falsify on the evaluation-EXCLUDED real corpus before
building. Prevalence of each proposed ban, real (15,294 excluded camera
comments) vs ours (528, v122):

| rule | REAL | v122 | verdict |
|---|---:|---:|---|
| `the ___ part` nominalization | **2.4%** | **21.2%** | **BAN — 8.8x overuse, our clearest fingerprint** |
| `feels like` | 0.3% | 3.4% | **BAN — 11x** |
| `a lot of/more/less` | 3.7% | 9.1% | **BAN — 2.5x** |
| banned opener token list | **27.2%** | 32.2% | **DO NOT BAN** — real does it nearly as often |
| gratitude formula | **0.3%** | 0.2% | **DO NOT BAN** — we already match; its high collision in the corpus is a credit-card-domain artifact |

**Two of the agent's seven rules would have made us less like real, not more.**
The opener-token ban was rule 2, targeting a drop to <5% against a real rate of
27.2% — that would have been a visible tell and a likely tone-metric regression.
This is the second time this session that free falsification on the excluded
corpus killed an arm I was about to build (the first was G92's rhetorical
question).

### Opener sharing: real but a third the claimed size

The agent reported 44.4% of comments share their first three words. That is
measured **across the whole corpus**; the metrics are computed **per thread**, so
the within-thread rate is what matters:

| | REAL (excl) | v122 |
|---|---:|---:|
| duplicate first-3-words, within thread | 5.9% | **8.5%** |
| duplicate FIRST WORD, within thread | 50.6% | **63.3%** |

A real +2.7pp / +12.7pp gap, worth acting on — but a third the headline size, and
the first-word gap is the bigger one.

### Carried forward to the build

1. Ban `the ___ part` / `the ___ bit` / `the ___ thing` / `the ___ side` (8.8x).
2. Ban `feels like` (11x) and `a lot of|more|less` (2.5x).
3. Enforce within-thread first-WORD variety (63.3% → target ~50%), not an
   opener-token blacklist.
4. Instruct lexical substitution — "different noun, different verb" — explicitly
   **not** clause reordering or connective swapping, which the matched test
   prices at zero.
