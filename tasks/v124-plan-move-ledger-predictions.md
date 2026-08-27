# Predictions for v124 — the spent-move ledger. Written BEFORE the run.

2026-08-28, per ORIENTATION §4 step 6. Control is **v122 on the same seeds 2–11**.

## What it is

`--plan-move-ledger spent_moves` on top of v122's flags. One change: when the
Planner is asked to repair a **semantic collision**, it is now shown the moves
the thread has already spent and required to name an unused one.

## Why this and not something else

| door | verdict |
|---|---|
| prompt wording | closed — G28 |
| decoding / sampler | closed — G86 |
| write-time retry | closed — G88 (38% rewritten, both metrics worse) |
| post-hoc reviser | ruled out by the user; also G85 caps it at |Cliff| −0.0123 |
| **the plan itself** | **open — G94** |

G94: within-thread cosine runs **plan 0.3289 > output 0.3064 > real 0.2892**,
reproduced independently, plan above output in 9/10 threads. The Writer is
already diluting the plan's excess by ~57%. G96: the Planner *detects* the
collision, repairs 3 times, fails, and ships — **111 slot instances** across 22
warnings, collision_rate at surrender up to **0.667**.

The instruction is the defect: "change the decision lens, stance, evidence role"
is a **category** (E4: 0.23 compliance vs ~1.0 for a named token) and it never
says what is already taken.

## Predictions

| quantity | v122 | predicted | reasoning |
|---|---:|---|---|
| plan-quality warnings | 22 | **8–16** | the repair now has a concrete target |
| unresolved slot instances | 111 | **40–80** | same |
| plan-move cosine | 0.3289 | **0.30–0.32** | this is the arm's own objective — **if it does not move, the arm failed at its own job (the G88 lesson)** |
| `self_bertscore` | +5.09% | **+3.5% to +4.7%** | the Writer dilutes ~57% of a plan change |
| `self_bleu_4` | +20.54% | **+17% to +20%** | a smaller, indirect effect |
| `polite_rate` | −42.63% | unchanged ±5pp | no tone field is touched |
| the 7 safe metrics | all P≥0.35 | **none flips to FAIL** | content-only change; tree shape untouched |
| cost | $4.14 | **$4.2–5.0** | repair calls are a small share |

## Decision rule, fixed in advance

- **Ship** if `self_bertscore` improves ≥1.0pp **and** no currently-safe metric
  flips to FAIL.
- **Reject as mis-instructed** if plan-move cosine barely moves (<0.005). That
  means naming the spent moves did not widen the plan, and the next lever is
  G97's outsider quota, not more Planner repair.
- **Reject as diluted** if plan-move cosine drops ≥0.02 but `self_bertscore`
  does not improve. That would mean the plan is no longer the binding
  constraint and G94's chain is broken downstream.
- **Watch:** `mean_story_probability` sits at parity (Cliff 0.00), so any push
  away from `personal_datapoint` moves it; and `semantic_mean_cosine` (+7.48%,
  Cliff +0.20) is the metric a diversity arm is most likely to overshoot.

## What it cannot do

G97 measured that **~1/3** of the gap is missing topical outsiders — comments
that do not answer the OP at all, especially **long** ones (we ship 0.0% of
those against real's 6.1% of low-affinity comments being ≥40 words). This arm
does not address that third at all. Even a full success leaves it open.

n=10 detects only |Cliff| > ~0.6; noise floor sd 2.94% / 13.7% (G76).
