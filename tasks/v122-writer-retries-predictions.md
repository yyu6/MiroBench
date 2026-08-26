# Predictions for v122 — turning on the guard that was already there

Written **before** the run, per ORIENTATION §4 step 6. 2026-08-28.

## What this arm is

Not new code. `--writer-retries 1` on top of v119's flags. G87: the write-time
guard battery flagged **264 problems across 532 slots (40.2%)** on v119 and
rewrote **1 slot**, because `total_attempts = max(1, writer_retries + 1)` and
the CLI defaults `--writer-retries` to **0**. Every paid run in the archive used
0. The corrective `retry_note` is fully built — it carries the offending
excerpt and a specific instruction ("contribute a different implication,
evidence role, stance, or decision lens") — and reaches the prompt at
`prompts.py:1203`. It has never been invoked on a paid run.

## Why this and not the alternatives

- Prompt end: closed by G28 (inputs separate, output does not follow).
- Sampler end: closed by G86 (entire design space = 1.65% of base on
  `self_bertscore`, inside one sd of thread noise).
- Post-hoc reviser (G85): **ruled out by the user** — the Writer's own output
  must pass. This arm is write-time, which is what the user asked for.

## The defect this should hit, measured

Generated vs matched-real pair-F1 distribution (2,965 / ~4,000 sampled pairs,
project's own scorer):

| quantile | generated | real | gap |
|---|---:|---:|---:|
| p5 | 0.4143 | 0.4078 | +0.0065 |
| p25 | 0.4848 | 0.4650 | +0.0198 |
| p50 | 0.5192 | 0.4967 | +0.0225 |
| p75 | 0.5500 | 0.5248 | +0.0252 |
| p95 | 0.5967 | 0.5688 | +0.0279 |
| p99 | 0.6481 | 0.6041 | +0.0440 |

**A near-uniform upward shift, widening with the quantile — not a missing tail.**
The single sharpest statement of it: **share of pairs below 0.5 is real 53.4%,
generated 35.2%.** We are short of mutually-unrelated comment pairs. A retry that
rewrites a draft flagged as too close to the thread is the mechanism that
manufactures exactly those pairs.

## Predictions

| quantity | v119 | predicted | reasoning |
|---|---:|---|---|
| slots taking >1 attempt | 1 (0.2%) | **150–214 (28–40%)** | the guard already flagged 214; retries let them act |
| `self_bertscore` | +4.33% | **+2.0% to +3.5%** | 56 `semantic_overlap_high` flags now rewrite; this is the metric they target |
| `self_bleu_4` | +18.85% | **+10% to +16%** | only 9 `lexical_overlap_high` flags, so a smaller move than selfbert |
| share of pairs < 0.5 | 0.352 | **0.38–0.45** | toward real's 0.534, not reaching it |
| cost | $3.85 | **$4.80–$5.60** | ≤214 extra writer calls, +40% worst case |
| `polite_rate` / `impolite_rate` | −38.1% / +19.8% | **unchanged ±5pp** | no tone guard is in the retry set |

## Decision rule, fixed in advance

- **Ship** if `self_bertscore` improves ≥1.0pp **and** `self_bleu_4` improves
  ≥3pp, with no metric that passed at n=10 flipping to FAIL.
- **Try `--writer-retries 2`** if the direction is right but the move is under
  half of the predicted band, and the attempt histogram shows retries were
  actually consumed.
- **Reject** if the retry rate is under 10% (the guard is not reaching the loop —
  a wiring bug, not a mechanism result) **or** if `self_bertscore` fails to
  improve at all despite ≥25% of slots retrying. In that second case the guard's
  own thresholds are miscalibrated and the finding is that the battery detects
  the wrong thing.
- **Watch for**: word count drift >15% (a retry loop that just shortens text),
  and `guard_degraded` count rising (retries exhausting into degraded output).

## What this cannot answer

n=10 detects only |Cliff| > ~0.6. `self_bleu_4` currently sits at Cliff +0.36 and
will read PASS either way; judge it on the **relative deviation and the sub-0.5
pair share**, not its pass/fail. Per-thread noise floor is sd 2.94% / 13.7%
(G76).

---

## Live observation during the run (2026-08-28 04:20)

**The mechanism is confirmed live.** First thread (seed 2, 45 slots): attempts
histogram `{1: 35, 2: 10}` — **22.2% of slots retried**, against v119's 0.2%.
This clears the "retry rate under 10% = wiring bug" reject condition
immediately; the guard reaches the loop and drives a real second call.

**Rewrites are substantive, not paraphrase.** Read three directly. The clearest:

- A1: *"the RX100 VII only makes sense if you're happy paying for the small body
  and giving up zoom-free simplicity. If you want a compact camera, the GR IIIx
  or X100F is the cleaner long-term bet."* → flagged
  `semantic_overlap_high: thread_mean_cosine=0.3994; target=0.211`
- A2: *"the Ricoh GR IIIx only pays off long-term if you're already happy living
  without zoom. Fine little camera."*

A broad three-body comparison became one specific verdict, and much shorter.
That is the corrective note working as designed.

**But a limitation is already visible and must shape the reading.** In all three
inspected cases the *second* attempt still carries problems (`length_too_long`,
`semantic_overlap_high` again, `uncertainty_frame_unwanted`,
`missing_concrete_anchor`). With `retries=1` there is exactly one extra chance,
and a second draft that still exceeds the band is accepted anyway.

**Consequence for the pre-registered decision rule:** this raises the prior on
the "direction right but move undersized" branch, whose stated response is
`--writer-retries 2`. Recording it now, before the metrics land, so that choice
is not a post-hoc rationalisation of a disappointing number.

---

## VERDICT (2026-08-28) — REJECTED as a pairwise fix, KEPT as the best N=150 candidate

Paid N=10, same seeds 2–11, **$4.14** (predicted $4.80–5.60).

### Scorecard against the pre-registered predictions

| quantity | predicted | actual | |
|---|---|---:|---|
| slots retried | 28–40% | **38.0%** (202/532) | HIT |
| `guard_degraded` | ~0 | **0** | HIT |
| cost | $4.80–5.60 | **$4.14** | slightly under |
| word drift | <15% | **+2.7%** (62.2→63.9) | HIT |
| `self_bertscore` | +2.0 to +3.5% | **+5.09%** (from +4.33%) | **MISS — wrong direction** |
| `self_bleu_4` | +10 to +16% | **+20.54%** (from +18.85%) | **MISS — wrong direction** |
| tone unchanged ±5pp | unchanged | **impolite moved 15.6pp** | **MISS — but in our favour** |

**5 of 7 quantities behaved as predicted. Both priority metrics moved the wrong
way.** By the rule fixed in advance (ship needs ≥1.0pp and ≥3pp of improvement),
this is a **reject**, and it triggers the second reject condition verbatim: no
`self_bertscore` improvement despite ≥25% of slots retrying.

### Why it failed, measured

The retry does not achieve **its own objective**. On the 60 slots where both
attempts reported a semantic diagnostic, `thread_mean_cosine` moved
**0.4528 → 0.4499 (−0.0030)** and only **33/60 = 55%** of rewrites moved it down
— a coin flip. Only **50/202 = 25%** of second drafts came back clean. The guard
**detects** convergence accurately; its corrective note **cannot remove it**.

**`--writer-retries 2` is NOT indicated.** The pre-registered escalation required
the direction to be right. Escalating a step that moves its own objective by
−0.003 at 55% accuracy buys cost and more of the same.

### But it is the strongest release on the board

Expected passes at N=150: **v113 2.96, v119 3.03, v122 3.48.**
`impolite_rate` P(pass) 0.24→0.38, `neutral_rate` 0.19→0.37, `length_cv`
0.18→0.38 (Cliff −0.34→**0.00**).

**And the reason is not what it looks like (G90).** Retried comments are 47.3%
impolite vs untreated 48.3% — **−1.1pp**, which cannot explain a 15.6pp metric
move. What changed is the **spread across threads**: per-thread impolite rates
went `20 32 36 42 44 50 50 62 71 77` → `21 21 28 32 42 43 45 59 59 70`. The
metric is the *mean of per-thread rates*, so pulling in the outlier threads moves
it. This also explains why the pairwise metrics did not follow: they are means
over comment **pairs within** a thread, so across-thread spread is invisible.

### Next step

Target **per-thread outliers** explicitly and measure on the **per-thread
distribution**, not the pooled share. That is the one lever demonstrated to move
this scoreboard. `self_bertscore` and `self_bleu_4` now have **all three
generation-side doors closed** (G28 prompt, G86 sampler, G88 write-time loop);
what remains for them is outside the current generator.
