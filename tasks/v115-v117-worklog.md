# v115–v117 worklog — three count/quota arms, the reachability proof, and one content regression

Session 2026-08-26. Paid spend **$3.93** (one 10-thread calibration run,
**571 generation records / 559 scored comments** -- the loader drops 12 -- 65 min).
Every number below is quoted with the count it was computed on. Three arms built, tested, committed, all three verified firing.
Handoff: `.claude/handoffs/2026-08-26-geo-v115-v117-*.md`.

## Why this session's most important result is not an arm

`analysis/self_similarity/real_vs_real_floor.py`. 150 evaluation real camera
threads against 150 **disjoint** real camera threads matched on comment count,
coverage 0.996:

| metric | target | donor | bias | MWU | KS | |
|---|---:|---:|---:|---:|---:|---|
| self_bertscore | 0.4923 | 0.4935 | **+0.24%** | 0.810 | 0.443 | PASS |
| self_bleu_4 | 0.0330 | 0.0325 | **−1.61%** | 0.801 | 0.231 | PASS |
| semantic_mean_cosine | 0.2741 | 0.2816 | +2.72% | 0.320 | 0.443 | PASS |
| polite_rate | 0.3216 | 0.3336 | +3.75% | 0.358 | 0.628 | PASS |
| impolite_rate | 0.4079 | 0.3893 | −4.56% | 0.338 | 0.362 | PASS |
| neutral_rate | 0.1611 | 0.1773 | +10.05% | 0.384 | 0.443 | PASS |

An arbitrary real thread passes all six comfortably. The metric, the matching and
the sample size are sound; the generator's +2.41% is **10x** the natural spread.
This is the logic `scripts/bootstrap_real_comment_discussions.py` was written for,
which had only ever been run on credit_cards. **Run it on any new domain before
spending.**

## And the archive says the opposite, wrongly

Across every `*_controller_history.json`, `self_bertscore` appears as a self-loop
TARGET in **zero** observations and as a *protected* metric in the 32 that end
PASS: **there is no selfbert self-loop.** The four run families carrying those PASS
observations sit at coverage **0.546, 0.577, 0.603 and 0.629** -- the truncation
regime `VERSION_LOG.md` opens by warning about. (Those four are what was checked
individually; the claim is about them, not about every run in `artifacts/`.)

Sweeping all **284** evaluated run directories: the coverage>=0.90 band has a
median `self_bertscore` bias of **+4.28%** and exactly **one** run under 1% -- the
real-comment bootstrap, which is not a generator. The 0.5-0.7 band has 224 runs and
61 of them under 1%, which is the artifact of the same effect.

## v115 — invert the tone quota

Six realization-side hypotheses are dead. The last one closes the lexical route:
fitting the polite-discriminative vocabulary by log-odds on real sentences and
applying it to both sides, generated already carries real's top-45 tokens at
**1.14x** real prevalence, and conditioned on the same move word it converts at
0.26–0.45x. There is no surface feature left to name.

What survives: polite is a **per-sentence lottery** (observed P(>=1 polite
sentence) tracks `1-(1-r)^k` at ratio 0.85–1.05 on both sides, sentences per
comment already match), so the whole defect is the per-sentence rate. And the
Writer's failure is consistent, which makes it a transfer matrix. `C^T a = target`
solved under the simplex constraint removes **86%** of the four-way L2 gap on the
shipped matrix. (An earlier figure of 89% came from a projected-gradient solver of
mine whose projection onto `{simplex, a0<=cap}` was wrong -- it also produced
solutions *worse* than the status quo at low caps, which is what exposed it. 86% is
the SLSQP value and the one to quote.)

Landed in one function: `generation_distribution.template_tone_rates`, reaching
both the rendered quota and the `planner_distribution` slot schedule.

## v116 — draw how many asides

The v113 gate's per-carrier parenthetical distribution was literally **`{1: 48}`**
— 48 carrying comments, every one holding exactly one, no exceptions — because the
cue read *"Put one aside in parentheses."* E4 confirmed from the other direction.
Measured over 15,559 evaluation-excluded comments the count runs 1.20 at short,
1.47 at long, 1.88 at very_long, 3.58 at essay.

Result on the calibration run: counts written `{1:40, 2:8, 3:3}` -- no longer all
1s -- and compliance **0.585** against the v113 gate's 0.380. **That comparison
crosses pools** (different threads, different band mix, and the tone quota was flat
here), so the direction is evidence and the magnitude is not. Words per
parenthetical 5.3 here against 5.4 measured on the matched seeds: the length was
never the defect, which corrected a table I had already published (it counted
brackets and inner punctuation as tokens).

## v117 — draw how many links, and the claim it forced me to resize

"The link arm buys 23%" came from the 35.8-token point on the closure curve. That
point needs **2.2 URLs per carrier** and real has 1.67, so it is not a legal arm.
Re-measured on 15,559 excluded comments (847 carriers, against the 26 the matched
threads offer): real holds **1.666** URLs and **34.3** URL tokens per carrier where
the gate held 1.00 and 18.0. Routing is not the defect — the gate routed 4.51% of
slots against the matched threads' own 4.92% carrying rate. `MAX_LINKS_PER_SLOT=4`
takes the target to **1.513** on the 150-pool exclusion set (the calibration pool's
own exclusion set measures 1.628 capped, so the number is pool-specific), worth
about **17%** of the gap on the measured closure curve, not 23%.

Result: **1.68** URLs per carrier against real's 1.67, **61** characters per URL
against the inventory's 61, compliance 0.950, and zero markdown garbage, invented
URLs or repeats — the v114 inventory fix is confirmed.

## The regression that blocks v117

The same output contains four unrelated links stacked at the end of a 46-word
comment, `https://support.apple.com/...` inside a comment about a Sony A7, and a
Fuji X-T5 film-simulation recipe inside one comparing Canon compacts. URLs are
drawn from an 802-entry inventory by hash **with no relation to the comment's
content**: tolerable at one link, an eye-visible tell at four.

Measured on 249 real comments carrying 2+ non-media URLs: **64.3% have ALL their
URLs on one host**, and the first URL sits a median **23%** into the comment
rather than trailing. A v118 draws a multi-link slot from one host at that rate
and places the first link early. **v117 must not ship until then.**

## What the calibration run bought

The flat quota worked as designed. Over generation records the assignment came out
**141/141/143/146** (571 slots); over the 559 comments the matrix is fitted on it is
**137/137/140/145**, so every cell of C has n>=137 against the old 289/92/156/522. The polite row transfers (0.3841 → **0.3942**) even
though it is now measured across every stance, which answers the Lucas critique
the 0.35 cap existed to hedge. The **neutral row does not** transfer
(neutral→neutral 0.4103 → 0.2429), and it is now the binding constraint: cap 0.59
lands polite at +0.2% and impolite at +1.3% while costing neutral −19.7%.

The shipped matrix was deliberately **not** replaced. One row transferring and one
not says some rows are generator properties and some are corpus properties, and
swapping a matrix measured on the evaluation-seed corpus for one measured on a
different corpus trades a known bias for an unknown one.

## Where `self_bertscore` actually stands

Honest arithmetic on what is built: v117 ~17% of the gap, v115 ~3%. **Assuming
they add**, which is not established -- the surface channels were measured
sub-additive at ~0.86 efficiency -- gap 0.0119 → ~0.0095 against the **0.0069**
Holm needs at N=150. Not enough, and both terms are J7 upper bounds.

The one channel measured large enough is unbuilt: authorial voice separation.
Generated sits at **0.55** of real's (stratum-weighted +0.0076 against +0.0137;
in the decisive different-branch cell +0.0061 against +0.0141). Headroom
**+0.0060 = 51% of the gap** — the only single channel above the 42% bar.
`persona_bridge`, `speaker_roster`, `actor_conditioning` and
`--speaker-identity matched` all exist and were on for every run, and **none has
ever been measured against `self_bertscore`.** Free, and next.

## Decisions not taken, with reasons

- **The evidence_mode labelling spend is dropped.** It was the largest per-pair
  collision channel (+0.0228, topic controlled), but two of the three cheap cells
  are already matched *in the text* with the label at zero: link/quote 0.86x real,
  hearsay 1.08x real. The Writer produces those moves regardless of the label, so
  raising the labels would push the surface above real.
- **Digit runs are dead**: the controlled ablation moves real `self_bertscore` by
  **−0.0005**, i.e. −3.6% of the 0.0138 denominator `rare_token_ablation.py` used
  or −4.2% of the 0.0119 local baseline -- negative either way, despite a token gap
  larger than URLs'. That closes the only route that would have needed domain
  vocabulary in Writer-facing text.
- **The rare-token reading of the URL channel is dead**: flattening real's hapax to
  a frequent thread word moves the metric the wrong way (−0.0020).
