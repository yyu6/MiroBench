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
