# v121 candidate — the sampler as a `self_bertscore` / `self_bleu_4` lever

Opened 2026-08-27. Priority set by the user: **`self_bertscore` and `self_bleu_4`
come first**, ahead of tone. Written before any result, per ORIENTATION §4 step 6.

## Why this direction, and the correction that produced it

The user asked whether v117 was better on these two metrics. **It was, and my
earlier framing was wrong.** I had dismissed the comparison because
`v117_calibration` used seed indices 0–9 against v113/v119's 2–11. That argument
kills the *pass-count* comparison and I stand by it — but it does not touch the
*relative deviation*, which is computed against each run's own matched real and
is what the metric actually tests:

| run | `self_bertscore` | `self_bleu_4` |
|---|---:|---:|
| v117 (seeds 0–9) | **+1.59%** Cliff +0.34 | **+8.67%** Cliff +0.26 |
| v113 (seeds 2–11) | +2.41% Cliff +0.64 | +12.96% Cliff +0.30 |
| v119 (seeds 2–11) | +4.33% Cliff +0.94 | +18.85% Cliff +0.36 |

**One half of the user's reading survives, one does not.** On `self_bertscore`
v117 is genuinely best. On `self_bleu_4` it is not: v117's *generated* value
(0.0378) is the **worst** of the three (v113 0.0314, v119 0.0331); it only looks
best because its seed set's real baseline was higher (0.0348 vs 0.0278). A
relative deviation flatters a run whose real threads happen to be repetitive.

**The clean, same-seed number is the one to act on.** v113 → v119 differ only by
`--tone-quota inverted`:

```
self_bertscore   +2.41%  ->  +4.33%     turning the quota off recovers +1.92pp
self_bleu_4     +12.96%  ->  +18.85%    turning the quota off recovers +5.89pp
```

**Consequence: my "keep v119" call (G83) is reversed under the new priority.**
It rested on `impolite_rate`'s hard-fail conversion being worth more than the
pairwise cost. With the user ranking the pairwise metrics first, that trade is
backwards and the quota should default off.

## The hypothesis

G28 (`prompt_convergence_diagnosis.py`) is the wall every recent mechanism has
hit: the Writer's inputs already separate (prompt line-Jaccard 0.3516 → 0.2481
with depth) while realized text similarity stays flat (r = 0.320), so
convergence is produced **inside the Writer, downstream of the prompt**.

Every mechanism tried since has been input-side. **G28's rejection covers input
levers only.** The sampler sits downstream of the prompt — precisely where G28
localises the defect — and has never been varied:

```
writer_temperature():   micro/short -> 0.88
                        offtopic_noise/reaction -> 0.95
                        else -> 0.82          # 371/532 = 69.7% of v119 slots
writer_extra_body():    None for every non-local profile
                        -> top_p, frequency_penalty, presence_penalty never set
```

Three temperatures, keyed on length and function, and **blind to depth** — while
G3/G26 put the whole defect in the reply population, with depth bins [2,4) and
[4,7) carrying 82.7%. The lever is orthogonal to the defect's own structure.

**H1.** Pairwise self-similarity responds to sampler settings at fixed prompt.
**H2 (the domain-adaptive form, only if H1 holds).** A depth-conditioned
temperature schedule, solved per domain against that domain's own real
depth-diversity curve, moves generated toward real. The per-domain curves are
already measured and on disk —
`analysis/reply_diversity_ceiling_calibration_results.json`:

```
camera      0-1: 0.6049  1-2: 0.5605  2-4: 0.5652  4-7: 0.5681  7+: 0.5729
cell_phone  0-1: 0.6111  1-2: 0.5888  2-4: 0.5806  ...
```

Each domain solves its own schedule from its own excluded corpus. Nothing is
hardcoded for cameras.

## The test now running (zero API cost)

`analysis/self_similarity/decoding_diversity_probe.py`. The run's **own saved
Writer prompts** (`generation_records.json`) are replayed through a local
`Qwen3-8B-4bit` at seven settings, each moving exactly one knob from the
production baseline. Every setting reads identical prompts, so the input side is
held exactly fixed and only the sampler varies — the contrast G28's
observational r = 0.320 could not make. Scored with the project's own
`score_thread_self_bertscore` and `score_thread_self_bleu`.

Settings: `base_T0.82`, `temp_T1.00`, `temp_T1.15`, `topp_0.85`, `freqpen_0.6`,
`freqpen_1.2`, `prespen_0.6`. Threads `seed002` (44 slots) and `seed003` (38).

**What it can and cannot answer.** The local model is not `gpt-5.4-mini`, so **no
magnitude here transfers** — this establishes existence and sign only. A knob
that cannot move pairwise similarity on any model is not a candidate; one that
moves it cleanly earns a paid arm. Word count is reported alongside, because a
knob that merely shortens comments would move both metrics for a reason that is
not diversity.

**Known risk on the `frequency_penalty` arm.** G27 ablated the comma (the largest
single 1-gram excess term, 16.5% of positive mass) at 30/50/100% and closed only
10.6/14.7/9.7%, with a `the` control closing the same — "because precision is a
ratio". A penalty shifts the whole token distribution rather than deleting one
type, so it is not squarely inside that rejection, but **that is my inference,
not a measurement**, and confidence on this arm is lower than on temperature.

## Predictions, recorded before reading any output

1. **Temperature moves both metrics downward, monotonically.** T=1.15 lands below
   base on `self_bertscore` and `self_bleu_4`. If it does not, H1 is dead and the
   sampler route closes with it.
2. **`frequency_penalty` moves `self_bleu_4` more than `self_bertscore`** — it
   acts on token reuse, which is what BLEU precision counts, while BERTScore
   works over contextual embeddings.
3. **`top_p` 0.85 moves both *upward*** (tighter nucleus = more typical text =
   more convergent). Included as a sign check: if every knob moves the same
   direction regardless of what it does, the probe is measuring an artifact.
4. Word count stays within ~15% of base for every setting. If a winning setting
   also shortens comments sharply, the win is confounded and must be re-read.

## Status

- [x] Correct the v117 comparison; record which half survives
- [x] Establish the same-seed cost of the tone quota (+1.92pp / +5.89pp)
- [x] Confirm G28's rejection is input-side only, and that the sampler is untried
- [x] Verify local inference is wired (mlx_lm 0.31.3, Qwen3-8B-4bit, 6.7s/slot)
- [ ] **Running:** seven-setting sweep, ~64 min
- [ ] Judge against the four predictions above
- [ ] If H1 holds: solve the per-domain depth schedule and build the arm

## Legality check against ORIENTATION §4 "What may never happen" — done before building

This matters more than usual here, because the sampler arm aims at **the same
target as a mechanism this project already killed**. G20 retired a
depth-conditioned reply-diversity *guard* (`reply_diversity_guard_diagnosis.py`)
for violating rule 4: it would have re-drafted a comment whose embedding
similarity to the thread exceeded a real-derived ceiling — a distribution
diagnostic selecting a Writer candidate.

A sampler schedule is not that. It changes the sampling distribution **before**
generation; nothing is scored, rejected, re-drawn or chosen among. All six rules:

| rule | status |
|---|---|
| Writer never sees matched evaluation comment text | unaffected — no text is added to the prompt |
| Domain profile from excluded threads only, zero seed overlap | schedule solved from the excluded corpus, same as every profile field |
| Nothing tuned against final test-set p-values | **the binding constraint.** The schedule is solved against the excluded corpus's own depth curve. It must never be adjusted by watching N=10 eval output — that would be tuning on the test set |
| Distribution diagnostics never select a Writer candidate | **clean, and this is the point** — no candidate is scored or rejected. This is exactly where G20's guard died and this arm does not |
| Every matched structural slot preserved | unaffected |
| No domain vocabulary in Writer-facing rule text | unaffected — a temperature is not text |

So the sampler reaches G3's target through the one door that is still open. That
is the strongest argument for the direction, independent of what the probe says.

**Constraint carried into the build:** the per-depth schedule is solved once,
from excluded data, and frozen before the paid run. If the first gate misses, the
fix is a better *derivation*, not a re-fit against the gate's own p-values.

## Timing correction (2026-08-27 23:37)

The smoke test measured 6.7s for one generation and I sized the sweep at ~64 min
from it. Real throughput is **~20s/slot** (45 slots in ~15 min), so the seven-
setting sweep is **~3h15m**, not one hour. The smoke test's single generation
returned a short comment; it was not representative of the token budget the
sweep actually spends. Same class of error as E13 — a probe sized to prove the
plumbing was read for a quantity it could not support, this time a rate.

No redesign: results are written after **each** setting, and the settings run in
declaration order, so `base` / `temp_T1.00` / `temp_T1.15` complete first and
prediction 1 — the primary hypothesis — is answerable at roughly the one-hour
mark, before the penalty arms finish.

## Free finding while the sweep runs: the defect has a depth SHAPE, and it is a rebound

`reply_diversity_guard_diagnosis.py --run v119...` (mpnet cosine, the cheap
embedding space — a replication of G3's property in a different model, per G19,
not a re-measurement of `self_bertscore` itself).

Per-comment max-similarity-to-anything-else in the same thread, by own depth:

```
                 [0,1)    [1,2)    [2,4)    [4,7)    [7,+)
generated       0.6369   0.5747   0.6215   0.6373   0.6224
real (matched)  0.6072   0.5743   0.5806   0.5616   0.5794
excess          +0.0153  +0.0087  +0.0124  +0.0397  +0.0253
```

### The legality trap I nearly walked into

The row above labelled "real (matched)" **is the evaluation target set**. Fitting
a schedule to it would violate ORIENTATION §4 rule 3 — "nothing is tuned against
final test-set p-values; calibrate on excluded reference data". Diagnosing
against it is fine; **deriving a parameter from it is not.** The legal source is
`reply_diversity_ceiling_calibration_results.json`, built from evaluation-excluded
threads only.

And the two sources **disagree**, which is exactly why the rule exists:

```
                    [0,1)    [1,2)    [2,4)    [4,7)    [7,+)
excluded (legal)   0.6049   0.5605   0.5652   0.5681   0.5729   n=4153/2610/3064/1461/320
  step                      -0.0444  +0.0047  +0.0029  +0.0048
matched (10 thr)   0.6072   0.5743   0.5806   0.5616   0.5794   n=1677/1221/1744/1037/212
  step                      -0.0329  +0.0063  -0.0190  +0.0178
```

The **dip at [4,7)** that makes the matched excess look largest (+0.0397) is not
in the excluded corpus, which plateaus (+0.0029). On 10 threads that dip is very
likely noise. **Had I calibrated to the matched curve I would have built a
schedule aimed at a seed-specific artifact — and it would have looked like it
worked, on the same threads that produced it.**

### The real target, stated legally

Against the excluded corpus, real does one thing after the root: **it drops ~0.044
and then holds a plateau** (+0.005, +0.003, +0.005 across the remaining bins).

Generated drops further (−0.062, root → [1,2)) and then **rebounds +0.063**, back
to its own root level (0.5747 → 0.6373). Real moves +0.008 over the same span.

**The defect is not "too similar everywhere". It is a failure to hold the plateau
— a rebound in the reply population.** That is the shape a depth schedule has to
produce, and it is why a depth-blind sampler cannot produce it: 69.7% of slots
draw at one temperature regardless of where they sit in the tree.

This is independent of the sweep. It holds whatever the sweep says, and it is
what the arm would be built against if H1 survives.
