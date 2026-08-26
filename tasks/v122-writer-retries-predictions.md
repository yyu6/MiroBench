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
