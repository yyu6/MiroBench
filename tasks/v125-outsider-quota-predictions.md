# Predictions for v125 — the topical-outsider quota. Written BEFORE the run.

2026-08-28. Control is **v122 on the same seeds 2–11** (v124 is rejected, G98,
and its ledger is off here).

## What it is

`--outsider-quota measured` on top of v122's flags. The Planner is told that a
measured share of slots must not answer the post at all, with named channels
(`offtopic_noise`, `joke`, `side_tangent`, `meta_or_template`), a **long**
sub-quota, and an explicit refusal to satisfy it with thanks.

## Why this, after v124 failed

G98 closed the Planner-repair door: making the repair instruction concrete did
not widen the plan, because the move generator has no wider vocabulary to reach
for. **This arm does not ask the Planner to find something new in the same
space — it hands it a different space.** The channels already exist and were
selected 0 times in 532 slots.

## The arm's own objective — checked FIRST, per G88/G98

| quantity | v122 | predicted |
|---|---:|---|
| slots with `comment_function=offtopic_noise` or an outsider `payload_type` | ~0% | **8–14%** |
| of those, `length_bucket` long/very_long | 0 | **≥25% of them** |
| comments with thread-affinity < 0.10 | 4.14% | **7–11%** (real: 11.43%) |
| comments ≥40 words with affinity < 0.10 | **0.0%** | **>0** |

**If the outsider share stays under 4%, the arm did not fire and nothing
downstream may be read** — that is a wiring/compliance failure, not a result.

## Downstream predictions

| metric | v122 | predicted | reasoning |
|---|---:|---|---|
| `self_bertscore` | +5.09% | **+3.8% to +4.7%** | G97 prices the outsider third at ~30% of the gap; the Writer dilutes |
| `self_bleu_4` | +20.54% | **+16% to +20%** | off-topic text shares fewer n-grams |
| `semantic_mean_cosine` | +7.48% | **the risk metric** — could improve or overshoot below real | it is the one a diversity arm overshoots first |
| `mean_story_probability` | −1.00% | ±8pp | anecdote is one named outsider channel |
| tone metrics | — | **uninterpretable below ~25pp** (G98) | do not attribute |
| cost | $4.14 | **$4.0–4.8** | no extra calls, only different plans |

## Decision rule, fixed in advance

- **Ship** if the outsider share reaches ≥8% **and** `self_bertscore` improves
  ≥1.0pp **and** no currently-safe metric flips to FAIL.
- **Reject as inert** if the outsider share stays <4%: the Planner ignored a
  named, quantified instruction, which would be a finding about instruction
  compliance, not about diversity.
- **Reject as mispriced** if the share reaches ≥8% but `self_bertscore` moves
  <0.5pp. That would falsify G97's attribution of ~1/3 of the gap to missing
  outsiders and send the search back to the on-topic core.
- **Watch:** `semantic_mean_cosine` (Cliff +0.20) overshooting *below* real, and
  `avg_depth` / `structural_virality` (both Cliff ≤0.02) — an outsider slot that
  changes reply structure would move them.

## What it cannot do

G97: **~2/3 of the gap is on-topic comments being too close to each other.**
This arm addresses the other third only. Even full success leaves the majority
open, and G98 just closed the most obvious route to that majority.
