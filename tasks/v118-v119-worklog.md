# v118–v119 worklog — the blocker shrank, the cap was decided, and the last named channel closed

Session 2026-08-27. **Paid spend: $0.** Everything below is free: cached artifacts,
the evaluation-excluded corpus, and simulation. Two shipped changes, five
retractions, one new acceptance measurement.
Handoff: `.claude/handoffs/2026-08-27-*.md`.

## 1. Two thirds of the v117 blocker did not survive measurement

G61 blocked v117 on three grounds. Measured on the 150-seed evaluation-excluded
corpus (424 threads, 11,817 comments, 531 carriers, 179 with 2+ non-media URLs):

| claim | verdict |
|---|---|
| multi-link slots should share a host | **stands** — real 0.771/0.640/0.417 at k=2/3/4, 0.695 pooled over 2≤k≤4; v117 drew **0.000** (0 of 6, P=0.0007) |
| the first link should sit ~23% in | **RETRACTED** — that is a *character* fraction over a length URLs dominate. In words: median **0.795** all carriers, **0.722** at k=2..4, and 53% put the first URL in the last quarter. Real trails |
| the URLs are topically unrelated | **not supported** — real names an off-prose brand in **0.200** of URLs, generated **0.344** at n=32. The corpus holds the same `fujifilm-dsc/x100f` URL inside a Canon comment |

What separated them was reading **all 19** of v117's carrying comments instead of
the three the handoff quoted (G65). The links that read as human are the opaque
ones — `youtu.be/...?t=1434`, flickr photo pages; 12.6% of real URLs carry no
descriptive path token at all. The tell is four unrelated hosts stacked together,
not one off-topic link.

**The user then ruled irrelevant URLs acceptable for now (G64)**, with an
LLM-minted or live-fetched URL as the future instrument. So only host coherence
was built.

## 2. v118 — one host per multi-link slot

`--reference-link-host {off,measured}`, default off, byte-identical when off.
Simulated on the run-built 699-URL inventory over 8,000 slots:

| k | v117 (`off`) | v118 | real |
|---|---:|---:|---:|
| 2 | 0.045 | **0.781** | 0.771 |
| 3 | 0.003 | **0.673** | 0.640 |
| 4 | 0.000 | **0.427** | 0.417 |

All 699 URLs still drawn; top host share 0.170 → 0.189, so the repeated-n-gram
guardrail on `self_bleu_4` holds. Schema 22 → 23.

## 3. v119 — the tone cap, and the objective that mattered more

The 0.35 cap hedged a cell that rested on n=17. **The calibration run discharged
it**: every row of C now has n≥137 measured across every stance, and the polite row
moved only 0.3841 → 0.3942.

The larger error was the objective. `invert_tone_rates` minimised the **four-class**
L2, but `somewhat_polite` is never reported, so error parked there is free. Robust
worst-metric closure (worst of the three reported metrics, worst over both known
matrices):

| cap | 0.34 | 0.50 | 0.54 | **0.56** | 0.60 | 0.62 |
|---|---:|---:|---:|---:|---:|---:|
| four-way L2 (old) | **−481%** | −235% | −99% | −31% | 33% | 33% |
| reported three (new) | 13% | 54% | 61% | **65%** | 42% | 33% |

So the shipped configuration would have driven `neutral_rate` several times worse
than doing nothing. `--tone-quota inverted` has never been run, so it never fired.

## 4. G57 is retracted, and with it the last named `self_bertscore` channel

**The persona layer was never on.** Every modern `run_config.json` carries
`persona_conditioning.mode = "none"` and `actor_conditioning.mode = "none"` — both
CLI defaults. Of 163 run configs: 147 none, 9 absent, 7 `matraix-projected`, all
from 2026-08-08/09. So s8's +0.0076 is `--speaker-identity matched` alone.

**And it is keyed per slot.** `_stable_rank(seed, seed_index, local_task_id,
persona_id)` — no speaker id anywhere. Of the 93 authors writing 2+ comments,
**93 of 93** would get a different persona per comment, which works against the
one-author-one-voice structure the channel is about.

**And the channel is already closed.** The s8 instrument — fidelity-checked, it
reproduces the v113 gate to four decimals — on the v117 artifact:

| | v113 gate | **v117** | real |
|---|---:|---:|---:|
| different branch | +0.0061 | **+0.0163** | +0.0141 |
| stratum-weighted | +0.0076 | **+0.0160** | +0.0137 |
| fraction of real | 0.55 | **1.16** | 1.00 |

G57's +0.0060 = 51% headroom does not exist. The residual is a level effect: every
pair uniformly ~+0.02 too similar, with the author structure now correct (G3).

## 5. Where the four named metrics actually stand at N=150

P(pass), raw rule / Holm, at today's bias:

| metric | bias | raw | Holm |
|---|---:|---:|---:|
| self_bleu_4 | +8.67% | 0.45 | 0.87 |
| emotion_entropy | −9.99% | 0.32 | 0.72 |
| self_bertscore | +1.59% | 0.28 | 0.73 |
| neutral_rate | −33.8% | 0.01 | 0.10 |
| **polite_rate** | −47.2% | **0.00** | **0.00** |
| **impolite_rate** | +49.7% | **0.00** | **0.00** |

**The tone trio is the emergency, not `self_bertscore`.** A perfect generator caps
at 0.92–0.95 raw, so 0.90 is the practical ceiling. `emotion_entropy` swings sign
across runs (−1.54% / +5.54% / −9.99%) and needs a better estimate before pricing.

## 6. What a large N=10 p-value is worth

P(pass at N=150) given the N=10 reading, against the unconditional rate:

| metric | uncond. | p≥0.8 at N=10 |
|---|---:|---:|
| self_bertscore | 0.20 | **0.34** |
| self_bleu_4 | 0.26 | **0.39** |
| emotion_entropy | 0.40 | **0.52** |
| polite_rate | 0.20 | **0.32** |

The direction of the intuition holds — a high p beats a marginal one — and the
magnitude does not. It roughly doubles the odds and leaves them near a third.
N=10 is too underpowered for its p-value to narrow the posterior much.
