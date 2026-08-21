# Lessons

## 2026-08-21 — Repairing a realization is what makes the plan's own error measurable

**What happened.** v102/v103 took the opener from 18% obeyed to 97% obeyed. The
metric it was aimed at, `hard_disagree_rate`, did not land on target — it
**overshot**, Cliff +0.37 → −0.23, thread mean 0.1569 → 0.0920 against a real
0.1208.

Decomposed by pair kind, the reply conditional was repaired exactly as the
ablation predicted (+0.080 over real → +0.022) and the **root conditional broke**:
0.0621 → 0.0284 against a real 0.0630, having been matched to three decimals
before. Realized polarity-token openers turned out to be **inverted** against
real — 0.0847 on roots against a real 0.0224, 0.0507 on replies against 0.0685.

`opener_profile` measures one **pooled marginal** and nothing makes the
assignment respect the root/reply conditional. That defect is older than v102.
It was invisible in v101 because the control was 18% obeyed and the leak happened
to push polarity openers onto replies, which pointed the right way by accident.

**Why:** a control that is mostly ignored cannot be wrong in any way the output
can show. Obedience is what converts a scheduling error into a measurable one. So
the first run after a realization fix is not only a test of the fix — it is the
first honest test of everything upstream of it.

**How to apply.**
- **After a realization fix, re-audit the schedule it now obeys**, not just the
  metric it was aimed at. Assume the schedule has been wrong and was hidden.
- A mechanism that lands *past* its target rather than short of it is evidence
  the target itself is mis-specified. Overshoot is a different diagnosis from
  under-delivery and points upstream, not at the cue.
- When a profile is a **marginal**, ask what conditional the assignment ignores.
  `register_realization` is measured per register for exactly this reason;
  `opener_profile` was not measured per pair kind, and that is the whole defect.

## 2026-08-21 — A pooled rate can match perfectly while the metric fails

**What happened.** On the v103 N=10, pooled over all 526 parent→reply pairs,
generated `hard_disagree_rate` reads **0.1198 against a matched real 0.1218**.
That looks like a solved metric. The reported metric is **0.0920 against 0.1208**
and its Cliff is −0.23.

The metric is a **mean of per-thread rates**, not a pooled rate over pairs. The
pooled figure is dominated by the largest thread (183 of 526 pairs), where the
fix landed almost perfectly; the unweighted thread mean gives a 7-comment thread
the same weight, and several small threads had collapsed to 0.00–0.03.

It also hid a two-sided error: the root conditional was **0.035 below** real and
the reply conditional **0.022 above**, and pooling let them cancel.

**Why:** the aggregation the metric uses is part of the metric's definition. Any
other aggregation is a different quantity that happens to share a name.

**How to apply.**
- **Compute the metric the way the scorer computes it** before drawing any
  conclusion — thread mean here, not a pooled pair rate. `ORIENTATION.md` §2 has
  warned about pass-by-cancellation since v96; this is the same failure in a
  decomposition rather than in a headline.
- A pooled number that looks better than the reported one is a **warning**, not
  a reassurance: it usually means two errors of opposite sign, or one big unit
  carrying the average.
- Always split a matched decomposition by the strata the metric can weight
  differently — here root vs reply, and small threads vs large.

## 2026-08-21 — The gate thread can be the one the fix happens to fit

**What happened.** The v102 large-thread gate ran on `i1o51h` and
`hard_disagree_rate` came out at 3.0% relative error, the closest the metric had
ever been. At N=10 the same mechanism overshot to Cliff −0.23, and `i1o51h` was
the thread it fit **best** — 0.1758 against a real 0.1697, while other threads
fell to 0.0222 against 0.0571 and 0.0270 against 0.1351.

The gate protocol exists to catch content defects on a big thread before N=10,
and it did its job — reading its comments is what found the stance contradiction.
But a single thread cannot tell you whether a mechanism lands *on* its target or
*past* it, because it has no distribution to overshoot within.

**Why:** the gate is a content instrument with one distance reading attached.
One reading has no variance, so "it landed" and "it landed here by luck" are
indistinguishable.

**How to apply.**
- Read the gate as **content evidence plus one sample**, and never let a good
  relative error on it stand in for an effect size.
- Where a mechanism could overshoot — anything suppressive, or anything that
  removes a behaviour rather than adding one — say so before the gate and treat
  the gate's distance number as unable to answer it.
- The N=10 is what tests direction *and* magnitude. Budget for it as the real
  test, not as confirmation.


## 2026-08-21 — A drawn value can contradict a control the plan already owns

**What happened.** v102 draws a slot's opening word from the register's measured
distribution. For `polarity_token` slots that word is `yes` / `yeah` / `no` /
`agreed` — and a polarity token **carries a stance**, which the Planner had
already assigned in the `stance` field. Nothing connected the two, so the draw
could contradict the plan, and on the gate it did on **2 of 10 polarity slots**:
both `stance=agree`, both told to open with `no`. The comments read "no, I'd just
check my RAW…" — text that agrees with `no,` bolted on the front.

**Every metric on that run looked good.** `hard_disagree_rate` reached 3.0%
relative error, its best ever. The defect was found only by reading the 23
comments the arm had touched, and it was found because the user asked whether
there was really nothing left to fix.

**Why:** a measured distribution is a marginal. The moment a drawn surface form
carries semantics that another control already assigns, the draw has to be
conditioned on that control or the two will disagree at exactly the rate the
marginal implies. `discourse_marker` had no such problem because `thanks / oh /
well / and / so` carry no stance.

**How to apply.**
- **Before drawing a surface form, ask what else in the plan already determines
  it.** Tone, stance, story mode and payload are all assigned upstream; a draw
  that touches any of them must be conditioned on it, the way the register draw
  is conditioned on the assigned tone.
- Condition by **restricting the family and renormalising inside it**, not by
  reweighting. The plan picks the family; the measurement keeps picking the token.
- Leave an escape: a register whose measured table has no token of the required
  family should fall back to the full draw rather than withhold the cue, so a
  sparse domain loses precision instead of losing the mechanism.
- **A clean metric table is not a clean run.** Read the comments the mechanism
  touched, every time. The gate protocol says "content *and* distance" and the
  content half is the one that catches a contradiction.

## 2026-08-21 — Do not use the median to wave away a mean-based metric

**What happened.** On the v102 gate `mean_story_probability` rose on the 23 slots
the arm touched, 0.076 → 0.157. I checked the median (0.063), saw three high
comments, and wrote it off as "3 comments of 23, not resolvable".

The metric is a **mean over every comment in the thread**. It is outlier-sensitive
by construction, so "the median is low" does not answer the question — three
genuinely new high-story comments move the metric exactly as much as the mean says
they do. The right check is **paired, per slot**, and it says something different:
story rose in **15 of 23** slots and comments over 0.5 went **1 → 3**, against the
other 160 slots where it rose in 89 (a coin flip) and comments over 0.5 fell
18 → 15. That is a directional signal at p ≈ 0.21 — still unresolved, but for the
honest reason (n=23) rather than the wrong one.

**Why:** the summary statistic used to dismiss an effect has to match the
statistic the metric is built from. A median rebuts a claim about the typical
comment; it does not rebut a claim about a mean.

**How to apply.**
- **Match the statistic to the metric.** `mean_story_probability` is a mean;
  `polite_rate` is a share; `emotion_entropy` is a function of a histogram. Argue
  in the metric's own currency.
- When runs are **paired by slot**, compare per slot — direction counts and
  over-threshold counts — rather than comparing two aggregates.
- "Not resolvable at this n" is a fine conclusion. Reaching it through the wrong
  statistic is not, because the next version inherits the reasoning.


## 2026-08-20 — Naming the token gets ~1.0 compliance where naming the category gets 0.23

**What happened.** `opener_profile` schedules a grammatical entry type per slot
and `_opener_rule` renders it. On v101 the instruction reached the Writer at
exactly the measured share and was ignored one way: `discourse_marker` — "open
with a short conversational connective before the point" — was realized 0.231 of
the time and turned into `Yeah,` on half its slots, and the realized
`polarity_token` share ran 2.4x its measured one.

The fix was not a stronger rule. **A prose prohibition was already there** — "Do
not open with a bare agreement or disagreement token" had been appended to every
non-polarity slot since v96, reached **504 of 532 v101 prompts, and was violated
on 9.1% of them.** I found that by grepping the saved prompts before writing
code, and it changed the design: v102 draws the concrete word from the register's
measured distribution and names it, and replaces the categorical ban with the ten
measured tokens.

On the gate: `discourse_marker` realization **0.231 → 0.923**, realized
`polarity_token` share **0.1559 → 0.0538** against a measured 0.0526, and **0 of
158 reply slots** prepended an unassigned polarity token against 19 in v101.

**Why:** a category name is a variable the model binds to its own default. A
token is a constant. This is the same finding as `TONE_DEFINITIONS["polite"]` at
19.3% and v98's "Use no semicolons" at 0.109 → 0.023, but the gap here is wider
than either, because the instruction was competing with a default the model
reaches for constantly.

**How to apply.**
- **Before writing a rule, grep the saved prompts for it.** A rule that is
  already rendered and already ignored will not work better a second time. The
  saved prompts are in each run's `generation_records[].prompt`.
- When a control is not realized, check whether it names a **category** or an
  **act**. Categories lose to the model's default; concrete tokens do not.
- A control that is faithfully *scheduled* can still be entirely unrealized.
  Verify the realization, not the assignment — the share in the profile and the
  share in the prompt both looked correct while the output was 2.4x off.

## 2026-08-20 — A flat conditional does not mean a lever cannot move the metric

**What happened.** I measured P(polite | opener class) on real and generated and
found the gap flat in every class — real 0.18-0.47, generated 0.02-0.15
throughout — and concluded the opener could not move `polite_rate`. I wrote that
into v102's guardrails: "`polite_rate` and `impolite_rate` must not move;
movement means the rule leaked into register."

They moved, favourably: on the gate `polite_rate` went 0.0820 → 0.1075 and
`neutral_rate` 0.1148 → 0.1398. A natural experiment attributes it exactly — the
polite gain is entirely on the arm's own 23 slots (0.000 → 0.174) and the neutral
gain entirely on the other 163 (0.113 → 0.160).

**Why:** a flat conditional rules out the feature explaining the *existing* gap.
It does not rule out **moving the prevalence between classes with different base
rates.** `discourse_marker` is the most polite class in real text at 0.466 and
`polarity_token` one of the least; shifting slots between them moves the metric
without changing any conditional. This is the prevalence-versus-conditional
distinction the v99 politeness work is built on — I applied it correctly to
`hard_disagree_rate` and forgot it one metric later, in the same document.

**How to apply.**
- "The gap is flat in every cell" answers *is this feature the cause of the gap*.
  It does not answer *would changing the mix of cells move the metric*. They are
  different questions and need different arithmetic.
- Before writing a guardrail, compute the counterfactual: cell base rates times
  the prevalence change. Here it was 23 slots moving into a class whose real
  polite rate is 0.466 — visibly non-zero, and I never did the multiplication.
- A guardrail that fires favourably is still a wrong guardrail. Record why it was
  wrong, or the next version inherits the reasoning.


## 2026-08-20 — An ablation harness that has not reproduced the artifact is a random number generator

**What happened.** Building an exact ablation harness for `hard_disagree_rate`, I
took each generated reply's text from `discussion.json` instead of from the
scorer's own pair record. The scorer runs `clean_text`, which collapses newlines;
`discussion.json` keeps them, and RoBERTa's BPE tokenizes `\n` as a token. The
harness re-scored the shipped run at 0.1730 against a recorded 0.1692, with
**11.2% of the labels flipped** and max |Δp| = 0.069. That error is a third the
size of the real-vs-generated gap I was about to measure with it.

I chased the wrong cause first — batch size, then the graph feature, then a torch
version difference — and only found it after re-scoring the **real** tables and
getting a byte-exact reproduction, which proved the environment was innocent and
the input was not.

**Why:** every ablation reports a *difference*, and a difference between a
faithful number and an unfaithful one is meaningless. The check that catches it
is cheap and the one that does not is expensive: comparing to a plausible number
tells you nothing, comparing to the artifact tells you everything.

**How to apply.**
- **Reproduce the artifact exactly before printing a single edited number**, and
  make it an assertion in the harness rather than something you remember to do.
  `disagreement_diagnosis.py ablate` refuses to print the ablations if label
  agreement is below 1.0.
- **Take the scorer's input from the scorer's own output when it records it.**
  The stance results store `parent_text` and `reply_text` post-`clean_text`.
  Re-deriving them from the source is re-implementing a normalizer.
- When a reproduction is off, **test the innocent side first**. Re-scoring the
  unchanged real tables cost one CPU minute and eliminated three hypotheses.

## 2026-08-20 — One Reddit post can sit under two product folders

**What happened.** `data/raw/discussions/camera_product/*/` is organised by
product, and a thread that mentions two products is stored under both. Reading
the per-comment classifier tables by globbing products double-counts those pairs:
**1.24× on the ten matched threads and 1.32× on the camera corpus**. My first
per-thread table showed real thread `1lt0yq3` with 70 stance pairs against a
`comment_count` of 45 — an impossibility, since there is at most one pair per
comment, and that impossibility was the only reason I caught it.

Rates were unaffected, because the duplicates are exact copies, but every pooled
or weighted figure was, and a duplicated thread silently gets double weight in
any corpus-level average.

**Why:** the directory layout encodes a product→thread relation that is
many-to-many, and every loader in this repo treats it as one-to-many.

**How to apply.**
- **Dedupe by `(thread_id, reply_id)`** — or `(thread_id, comment_id)` — when
  reading anything under `data/raw/discussions/`.
- `generalized_card/analysis/politeness_diagnosis.py` does **not** dedupe. Its
  conditionals and lifts are safe; its pooled counts and weighted shares are not.
- **A count that cannot be true is the cheapest bug detector there is.** Pairs
  exceeded comments, and that was the whole signal. Sanity-check the arithmetic
  of a table before reading its conclusion.

## 2026-08-20 — A metric name can lie about what the metric measures

**What happened.** `hard_disagree_rate` reads as "how often a reply disagrees
with its parent". It is not that. The local Stance_Rel wrapper has no graph path,
its three class probabilities all sit inside ≈[0.26, 0.41], and a surrogate
fitted on real data reaches AUC 0.740 from the **reply text alone** against 0.579
from the parent alone. Its `disagree` class is keyed by explicit stance tokens
with **agreement ones weighted most heavily** — `agree`, `agreed`, `yup`,
`yeah`, `exactly`. In the shipped artifact, slots the Planner assigned
`stance=agree` are labelled disagree **0.255** of the time and slots assigned
`stance=disagree` only **0.181**.

Anyone reasoning from the name would have gone looking for a way to make replies
disagree less, and would have moved the plan's stance marginal — which is not
where the defect is.

**Why:** the metric is an argmax over a nearly uniform softmax, so it measures
"does this comment take an explicit position" far more than "which position".

**How to apply.**
- **Read the scorer, then read what the classifier responds to on real data** —
  the name is a third-hand summary of neither.
- For any argmax metric, **look at the decision margin distribution** before
  theorising. A near-degenerate head means the metric moves on surface tokens and
  a uniform translation, not on semantics.
- The mechanisms that move such a metric target **what the classifier responds
  to**, not the concept in its name. This is the same rule the politeness work
  arrived at, restated for a second metric.


## 2026-08-20 — Reason at the granularity the mechanism operates at, three times over

**What happened.** One version line produced the same error three times, each
time costing a wrong prediction or a regression.

1. **The v99 prediction.** The arm fires only where the plan assigned `polite`,
   which is 25% of slots. The prediction was computed from a corpus-wide baseline
   to a corpus-wide target: 0.258 -> 0.45 for `any_intensifier`. On the slots the
   cue actually reaches, the baseline was already 0.565 — above the real corpus
   rate of 0.373 — so there was nothing there to fix. It came out at 0.204.
2. **The cue wording.** "what you ended up keeping" and "how long you have had
   it, what it did or did not do" are event descriptions, and the model answered
   with narrative. `mean_story_probability` went from 0.8% relative error to
   29.2%. Real text uses the same possessive as a bare state at story probability
   0.279; generated reached 0.510.
3. **Excluding `gratitude`.** It was dropped because generated output ran 1.25x
   real — a figure comparing generated-on-all-slots against
   real-on-all-comments, while the cue fires on one register. Conditioned
   properly, real polite micro comments thank at 0.330 against a generated 0.100,
   and real polite short at 0.165 against a generated **0.000**. Five
   planned-polite micro slots on the gate fell from 0.600 realized polite to
   0.000 once intensifiers replaced thanks.

A fourth, in the analysis rather than the code: an attribution split used
`real_word_count` read back from `discussion.json`, which is **not persisted** and
is 0 on every row, so it silently fell back to the generated length instead of the
matched length the cue keys on. Redone on `length_bucket`, which is persisted, the
conclusion reversed: medium slots had improved (planned-polite 0.105 -> 0.222)
while micro had collapsed.

**Why:** every one of these compares a number measured over one population
against a mechanism that acts on a different one. The comparison always looks
reasonable, because both numbers are real; what is wrong is that they are not
about the same slots.

**How to apply.**
- **Before predicting, write down the population the mechanism acts on**, and
  measure the baseline on exactly that population. If a cue fires on 25% of slots,
  a corpus-wide prediction is arithmetically impossible to hit.
- **Before excluding a move because it is "already over-produced", condition on
  the register and band the cue would fire in.** A pooled ratio can point the
  opposite way from every cell inside it.
- **Read a cue back as an instruction and ask what a model would return.** "What
  you ended up keeping" asks for a history. "What you have" asks for a state.
- **Check that a field exists in the artifact before splitting on it.** A silent
  fallback to a different field produces a clean-looking table with the opposite
  meaning. `real_word_count` is 0 in `discussion.json`; `length_bucket` is the
  persisted proxy.
- Corollary for reporting: at n=9 and n=13 a 3-comment swing is noise. One claim
  in this line ("the verdict suppression strips polite appraisal from long slots")
  was reported from exactly that and had to be retracted — real polite comments
  close on that pattern at 0.010–0.029, so the suppression costs almost nothing.

## 2026-08-20 — Condition on the feature; if the gap is flat in every cell, the feature is not the cause

**What happened.** The politeness trio had a carried-forward v99 plan with a
verified-looking causal claim: warmth-marker rate against thread-level
`polite_rate`, r = +0.727 over 412 real threads, monotone across quintiles. The
plan was to schedule warmth markers per slot at their measured rates.

Before writing it, the same relationship was measured **per comment** instead of
per thread, using the real per-comment polite-guard labels that already existed in
the repo:

| | marker presence | P(polite \| marker) | P(polite \| none) |
|---|---:|---:|---:|
| excluded real | 0.308 | 0.627 | 0.173 |
| matched real | 0.284 | 0.652 | 0.144 |
| generated | 0.178 | **0.213** | 0.039 |

Moving presence to the real level while the conditional stays where it is
predicts `polite_rate` 0.070 -> 0.088, against a real 0.288. The thread-level
correlation was real and the mechanism it implied was wrong, because a thread-level
r cannot distinguish "generated has fewer markers" from "generated markers do not
land".

Three more hypotheses died the same way, each by conditioning and finding the gap
unchanged inside every cell: warmth-as-concession (contrastives *raise* P(polite)
in real text), first-person lived experience (every experience feature lifts
1.4-2.2x against warmth's 3.56x, and the gap is a flat 3-4x in all four cells of
the warmth x experience table), and a dismissive-adjudicative register
(excluded-real P(polite) is 0.293 with it and 0.315 without).

One of those rejections reversed a priority the previous worklog had recorded.
"Negative markers at 3x real" had been written down as something to suppress; a
TF-IDF model fitted on real text decomposed the gap as a **+8.381 polite-vocabulary
deficit against a -0.767 impolite-vocabulary excess** -- generated text uses
*less* of the impolite vocabulary than real. Suppressing it would have moved the
metric the wrong way.

**Why:** a correlation at the aggregation level of the metric is not a mechanism.
The metric is a thread mean of per-comment classifier decisions, so the mechanism
lives per comment, and only a per-comment measurement can separate prevalence
from conditional probability.

**How to apply.**
- **Condition on the candidate feature and compare the cells.** If the
  generated/real gap is roughly constant inside every cell, the feature is not
  the cause, however good its aggregate correlation looks.
- Measure at the level the metric is computed at. A thread-level r on a
  thread-mean-of-per-comment metric is one aggregation step too coarse.
- Before adding a lexical family, check its current rate. Two of the moves that
  looked obviously missing -- `gratitude` and the whole negative register -- were
  already at or above the real rate.
- Look for the per-comment labels before scoring anything. All of this ran on
  `politeness_results.json` files that were already in `data/raw/discussions/`;
  no model was re-run.

## 2026-08-20 — A pinned hash is not provenance if the source was never committed

**What happened.** The project has four traceability mechanisms and I had been
treating them as redundant: `run_config.json` records every arm and the domain
profile's SHA-256; `core_contract.py` pins the SHA-256 of all 101 active source
files and refuses to run on drift; `HISTORICAL_GENERATION_POLICY_VERSIONS`
records every released policy string; and git holds the tree.

Checked, rather than assumed:

```
HEAD = a34abc6  -> core_contract.py at HEAD names v96 as current
git log -- generalized_card/generalized_card/sentence_rhythm.py   -> empty
git log -- generalized_card/generalized_card/length_calibration.py -> empty
```

**v97 and v98 existed only in the working tree.** Two shipped versions, one of
them the source of the N=10 result being quoted as the project's current state,
with no recoverable source tree. The run directory holds `generated/`, `logs/`
and `run_config.json` — no source snapshot. And
`HISTORICAL_GENERATION_POLICY_VERSIONS` stores version *strings* with no
per-version file hashes, so it can identify an old artifact but cannot
reconstruct the code that made it.

The three non-git mechanisms all describe the tree. Only git *is* the tree.
Pinning a hash proves the file has not changed since you pinned it; it does not
store the file. They looked redundant because each one felt like provenance.

Two versions also cannot be separated after the fact — the working tree
interleaved v97 and v98 edits, so v97's standalone tree is unrecoverable and the
two share one commit boundary.

**Why:** the checks that felt like traceability were all integrity checks
(has this drifted?) rather than storage (can I get it back?). And the one that
was storage was the one nobody ran.

**The near miss is the instructive part.** `repin_core_contract.py` already
refused to pin a file that `git ls-files` did not know about, and it reported
`untracked active: 0` the entire time. `git ls-files` lists *tracked* files, and
a staged-but-never-committed file is tracked. A check written against the wrong
git command reads exactly like a check that passes.

**How to apply.**
- **Commit at every version boundary, before the paid run, not after.** The
  commit is what makes `run_config.json`'s policy string mean something.
- When asked whether something is traceable, run `git log -- <the new file>`.
  Do not reason from the presence of a hash table.
- Any mechanism that answers "has this changed?" is not an answer to "can this
  be recovered?" Keep the two questions separate.
- **A rule that depends on remembering is not a mechanism.** This was already
  covered by prose in `AGENTS.md` ("preserve run configuration, source
  provenance...") and it still happened twice. The fix was
  `source_provenance.py`: `run_generate.py` now refuses to start when any file
  defining the version is missing from `HEAD`, checked before the first API call.
  Add the gate where the cost is incurred, not the paragraph.
- **Test the belief, not the mock.** The provenance tests each build a real
  throwaway git repository, because the defect was a wrong belief about what a
  git command reports. A mock would have encoded the same wrong belief and
  passed.

## 2026-08-20 — An archive that grows by addendum stops being readable as a spec

**What happened.** `tasks/HANDOFF.md` had become the project's entry point at
1,350 lines and 72 KB. It grows newest-addendum-first, so a reader hits nineteen
dated addenda before reaching `# 1. THE GOAL` at line 510 — and that numbered
body still described v79-era state, quoted a test count from v81, and carried a
`self_bertscore` conclusion in §13 that had since been measured and rejected.
Everything needed to onboard was in the file; none of it was findable, and some
of it was wrong.

**Why:** an append-only evidence log and a current-state spec have opposite
maintenance rules. The log must never be rewritten, or the evidence trail
breaks. The spec must always be rewritten, or it lies. One file cannot do both,
and when it tries, the newest content wins the top of the file while the stalest
content keeps the authoritative-sounding headings.

**How to apply.**
- `docs/ORIENTATION.md` is the spec: goal, judging standard, metric
  interpretation, method, current state, discipline. It is **rewritten in
  place** every time one of those changes, and it carries a "last verified" date
  plus the list of checks behind it.
- `tasks/HANDOFF.md` and `tasks/v<N>-worklog.md` stay append-only evidence.
- When a spec claim is retracted, record the retraction **in the spec**, next to
  the claim it replaces — not only in the new worklog. Three retractions now sit
  in `ORIENTATION.md` §6 for exactly this reason.

## 2026-08-19 — At N=150 the target is an effect size, and "all 12 pass" is a coin flip

**What happened.** The stated goal is 12 metrics at p>0.05 over 150 threads, and
progress had been tracked as "metrics passing at N=10". Simulating the actual
pair of tests the evaluator runs (MWU and KS, both required, alpha 0.05) over
normal populations at a known Cliff's delta:

| true \|d\| | pass at N=10 | pass at N=50 | pass at N=150 |
|---:|---:|---:|---:|
| 0.00 | 0.95 | 0.94 | 0.94 |
| 0.10 | 0.94 | 0.84 | 0.72 |
| 0.15 | 0.93 | 0.70 | 0.37 |
| 0.20 | 0.88 | 0.56 | 0.14 |
| 0.25 | 0.87 | 0.38 | 0.04 |
| 0.50 | 0.48 | 0.01 | 0.00 |

Two consequences.

**The N=10 pass count is a weak signal.** A metric at \|d\|=0.25 passes 87% of
the time at N=10 and 4% at N=150. Of v96's six passing metrics, only three
(`semantic_mean_cosine` 0.02, `structural_virality` 0.04, `avg_depth` 0.06) are
safe at 150. `emotion_entropy` at 0.12 is a coin flip and
`mean_story_probability` at 0.18 is 18%, despite its p of 0.521.

**Twelve metrics times two tests is its own problem.** Even with a *perfect*
generator at \|d\|=0, each metric passes both tests 94% of the time, so all
twelve pass together with probability 0.94^12 = **0.52**. At \|d\|=0.05 it is
0.18. The stated goal is therefore at best a coin flip against a flawless
generator, and one or two metrics failing in any given run is expected rather
than diagnostic.

**How to apply.**
- Track \|Cliff's delta\| per metric, with 0.10 as the working ceiling. Use the
  p-value to report, not to steer.
- Do not read a single metric's failure at N=150 as a regression without
  checking its effect size against the previous version.
- Raise the multiplicity issue with the user before the final run so the
  reporting standard is decided in advance rather than after seeing which
  metrics failed.

## 2026-08-19 — Check the probe's word list before believing the gap it reports

**What happened.** The matched content audit reported "has domain vocabulary"
at 0.156 generated against 0.556 real on the v97 seed-2 thread, and the same
diagnostic had helped motivate concreteness work in v91 and v96. Counting what
the generated thread actually names: `rx100` 12 times, `x100f` 9, `gr iiix` 9,
`af` 7, `ricoh` 3, plus `autofocus`, `aps-c`, `low light`, `sony`, `canon`.

The probe is `config.technical_terms + config.protected_entity_terms`, and
`Ricoh` is **not in `protected_entity_terms`** even though it is the brand of one
of the three cameras named in the seed post, and no model designator is in either
list. So the probe counts 7 of 45 comments where the thread is in fact dense with
domain-specific names. The audit labels these "weak surface probes... never
semantic ground truth", and this is what that warning is for.

The gaps in the same table that do not depend on a hand-written list —
`has a digit` 0.356 against 0.600, and 10 distinct model designators against 40 —
are the real signal, and they say something different: the generated thread stays
on the products in the seed while real commenters keep bringing in other bodies.

**How to apply.** Before treating a diagnostic gap as a target, read the
predicate that produced it and count the same property a second way. If the probe
is a keyword list, print what the text actually contains and check the list
covers it. Prefer the list-free version of the same measurement when one exists.

## 2026-08-19 — A hash-pinned core makes editing during a run an outage

**What happened.** With a `--posts-per-run 5 --max-posts 10` run in flight, I
started building the next change and edited three pinned core files. The run
spawns one `run_generator_backend.py` subprocess per batch, and
`load_generator_backend` calls `verify_core_contract`, which **raises** on any
hash mismatch. The first batch had already been paid for; the second batch would
have aborted on a core-contract error before its first API call.

Re-pinning would have been worse than the crash: the second batch would then have
run on different source from the first, inside one run tag, which is exactly the
policy mixing `RUN_INDEX` and `run_config.json` exist to prevent. The fix was to
save the new work aside and restore the three files to the state the run started
from, then confirm `repin_core_contract.py` reported zero drift.

**How to apply.** While a multi-batch run is in flight, treat every file in
`CORE_FILES` as frozen. Develop the next version in a scratch copy or a worktree
and apply it after the run finishes. If a core file has already been touched,
restore it rather than re-pinning: a run whose batches used different source is
not a run, and the cost of discarding the edit is far below the cost of an
unattributable result.

## 2026-08-19 — A tokenizer is part of the metric, so typography is content

**What happened.** `self_bertscore` and `self_bleu_4` had been treated as content
problems for six releases. One character count over the v96 output answered a
large part of both in a minute: **zero of 532 generated comments contained an
ASCII apostrophe** and 389 contained a typographic one, while the domain's
evaluation-excluded real comments use the typographic form in 27% of the comments
that use an apostrophe at all. `score_thread_self_bleu.TOKEN_PATTERN` reads
`it's` as one token and `it’s` as three, so every generated contraction
contributed a `<word> ’ s` trigram shared across the whole thread that no real
comment produces. Em dashes appeared 187 times against 3 in the matched real
text.

Replaying the same text through a per-speaker keyboard draw, scored with the real
scorer, moved `self_bleu_4` MWU p from 0.009 to 0.273 and KS from 0.052 to 0.787.
No content changed at all.

**How to apply.** Read the metric's tokenizer, not just its aggregation. Then
count, in the generated text, every surface class that tokenizer treats as a
distinct token: punctuation form, casing, whitespace, markup. A model's default
typography is a systematic thread-wide signal, and a systematic signal is exactly
what a pairwise self-similarity metric measures. Diff the character inventory of
generated against real output before theorising about meaning.

## 2026-08-19 — Matching a marginal can hide a joint that is exactly backwards

**What happened.** `polite_rate` failed at p=0.006 with a **correct target**: the
Planner's polite share was 0.311 against a real 0.308. The failure was where the
labels landed. `_tone_cost` ranked candidate slots by distance from each tone
class's median length, which put `impolite` on 74% of 120-250 word slots and 100%
of slots over 250 words. In the same domain's excluded threads, comments over 250
words are 72% polite and 23% impolite. The realized output followed the plan: 87%
impolite and 9% polite above 120 words against a real 27% and 71%.

Three tone metrics were failing on one placement rule, and no amount of work on
the marginal could have found it.

**A second error inside the fix.** The first implementation costed every
(slot, label) pair at `-log P(label | band)` and consumed pairs in ascending
cost. That maximizes total likelihood, which drives an assignment problem to a
corner: it produced 100% polite in the top band against a measured 72% and 98%
impolite in the `short` band against a measured 48%. Fitting a conditional
subject to two fixed margins is iterative proportional fitting, not min-cost
assignment. A cost function that "prefers" the right direction is not the same
statement as reproducing a measured distribution.

**How to apply.**
- When a metric fails and its target is right, print the realized joint against
  the measurement, per band. `tone_length_joint` is now in every schedule for
  exactly this reason.
- Before writing a placement heuristic, write down which distribution it is
  supposed to reproduce and check that the algorithm reproduces *that*.
- A hard compatibility rule written from intuition ("a warm turn cannot be a
  micro reaction") is a hypothesis. 25.1% of real comments under ten words are
  labelled polite, because a short thank-you is one.

## 2026-08-19 — An unreachable request is not a stronger request

**What happened.** Long slots realized 0.61x their matched length, and the
845-word slot 0.32x. Two earlier releases raised the beat budget for exactly this
symptom, ending at one beat per 21 words with a ceiling of 40, which asked that
slot for 40 connected beats on one thesis.

The Planner does not supply beats above about nine however many are asked for.
Measured over the v96 slots that carried a beat plan: asked ~6 it returned 5.2
and the slot realized 0.95x; asked ~9, 8.1 and 0.91x; asked ~12, 8.3 and 0.74x;
asked 14-40, 9.5 and 0.60x. The largest plan any slot received in the whole run
was 26. Raising the ceiling past the saturation point bought nothing and
generated plan-repair traffic that could not succeed.

The shape being asked for did not exist either. Real long comments are not one
thesis: median paragraph count rises to 6 in the top size band with a p90 of 14,
and words per paragraph is nearly flat inside a band while the paragraph count
scales with length. v96 output had a blank line in 3.4% of comments against 33.8%
of real ones.

**How to apply.** When a control does not produce its effect, measure what the
model returns when the control is dialled up, not only what the output does. If
the response saturates, the control is at its ceiling and the remaining gap needs
a different mechanism. And check the target shape exists in the reference data
before asking for more of it.

## 2026-08-19 — A frame reworded three times needed to be withheld once

**What happened.** "The question your turn settles: ..." rendered on 532 of 532
v96 Writer prompts, and the "that's the part that actually matters" family it
produces was in 18.4% of comments against effectively zero in 30,643 tokens of
matched real text. v73 reworded the line, v74 rebuilt the prompt around it, v75
added a route lock. Every release kept rendering it on every slot.

Broken out by planned function, the frame was worst where it least belongs:
personal_datapoint 29.1% and reaction 19.0%, against question_followup 8.1% and
verdict_evaluation 12.3%. A slot told to report an experience *and* told which
question it settles converts the experience into an adjudication. The variable
nobody had varied was whether the slot gets the line at all.

**How to apply.** If a prompt artifact survives two rewordings, stop rewording
and check the population it is rendered to. Break the artifact rate down by the
control that selects the slot; if it is highest where the instruction is least
appropriate, the instruction is being imposed rather than followed, and the fix
is a gate, not better wording.

## 2026-08-18 — A distribution target is not a terminal per-slot truth

**What happened.** A direct reply planned a coherent gratitude close. The
held-out aggregate template then overwrote only its affect label with neutral,
and validation rejected the contradiction created by that overlay. Across three
whole-post attempts, different story, length, social, and density combinations
failed after 130 Planner repair calls, leaving no text to evaluate.

**How to apply.** Give each field one owner. Compile aggregate targets and the
Planner's semantic choice into one dependent contract before evaluation, and
record any projection. Treat residual content quality as an observable warning,
not schema corruption. Only a state that cannot be safely persisted may abort a
post. Completion is an engineering invariant; p-value matching remains an
empirical result that cannot be promised before scoring complete output.

## 2026-08-18 — Slot-local repair still loses state if it replaces the whole object

**What happened.** Three repairs targeted only S9, but each returned and replaced
the entire plan. One candidate fixed long-form capacity, the next fixed its
story/evidence contract while erasing the long-form beats, and the last restored
the beats while reverting the evidence contract. Every required field had been
correct at least once, yet no whole candidate contained both repairs when the
budget ended.

**How to apply.** Once a slot has one remaining repair diagnostic with a known
field boundary, merge only that field from the candidate and preserve healthy
state. Use whole-object replacement while multiple semantic contracts must
change together. Persist raw and applied candidates separately so the merge is
auditable and never mistake one historical candidate passing for proof that a
stochastic future repair is reliable.

## 2026-08-17 — Hard feasibility cannot share one scalar with soft quality

**What happened.** Targeted Planner repair added collision, story, capacity,
and other issue weights into one score. A candidate that fixed S15's blocking
story contract gained a semantic collision worth 10 points, so it lost to the
unrealizable story conflict worth 8. The post later failed with the very
contract the repair had already removed once.

**How to apply.** Compare repair candidates lexicographically: first minimize
the number of contracts the downstream stage cannot jointly realize, then
optimize collision and other quality diagnostics. Persist both candidates and
the comparison ranks; an issue summary without the rejected plan is not enough
to audit the decision.

## 2026-08-17 — A classifier label is not automatically a semantic invariant

**What happened.** `polite` is produced by a four-way classifier over realized
text, but Planner validation redefined it as only agreement, personal evidence,
reaction, or positive verdict. Two unresolved polite pairings then aborted a
186-comment run even though politeness is the lowest-priority metric and the
Writer had not been called.

**How to apply.** Use scorer outputs at the level they measure. A surface
register can guide and diagnose Planner routing without becoming a hard claim
about comment function. Reserve blocking validation for combinations no single
output can realize, such as `no_story + firsthand_experience` or a long slot
forced into a low-information payload.

## 2026-08-17 — A prompt schema example is executable data

**What happened.** The direct-reply Planner schema described
`development_plan` as “none for a short slot; otherwise …”. The model copied
that explanation as the field value in 104 short slots, and the Writer later
treated it as planned content. A sentence intended as documentation became a
shared semantic beat across more than half the thread.

**How to apply.** JSON examples must contain valid literal values, not prose
instructions. Put conditional rules outside the schema, normalize sentinel
values, and reconcile a field against anonymous structural capacity before it
reaches the Writer.

## 2026-08-17 — Matching marginals is not enough; assign a feasible joint contract

**What happened.** Story, tone, and affect counts each matched their template,
but they were assigned independently. The result included `approval+impolite`,
`neutral-affect+polite`, and story slots carrying gratitude closes. The Writer
could not satisfy both sides of those pairs, so one label collapsed even though
every marginal count looked correct.

**How to apply.** Preserve the measured marginal totals while assigning them
through explicit cross-field compatibility. Verify both completeness and the
joint contingency table before generation. A quota is fulfilled only when the
Planner can build one coherent semantic move for the whole row.

## 2026-08-17 — Normalization may repair metadata, not author semantics

**What happened.** Post-parse code silently converted incompatible substantive
payloads to `soft_helpful` and rewrote every gratitude/relief problem to one
canned semantic move. That hid Planner failures, inflated helpful/explainer
register, and introduced repetition after the Planner had already made its
decision.

**How to apply.** Deterministic normalization may canonicalize enums or clear a
field that has no structural capacity. If a repair changes role, evidence,
claim, reply increment, or payload meaning, send the conflict back to the
Planner and fail closed when bounded repair is exhausted.

## 2026-08-17 — Git is the version switch; dead experiment controllers are not

**What happened.** Metric-guided Writer retry code, candidate ranking, a
blocking repetition arm, and several CLI knobs remained after the public v81
path required them all to be disabled. They added hundreds of lines, obscured
the actual single-realization policy, and encouraged evaluation metrics to leak
into generation-time selection.

**How to apply.** Preserve old behavior with commits and per-run source/config
snapshots. Once an arm is disproven and unreachable in the current policy,
remove its implementation and historical-only tests. Keep only current safety
recovery, and make distribution metrics observational rather than selective.

## 2026-08-16 — A scheduled label is not a coherent plan

**What happened.** The frozen distribution schedule overwrote `story_mode`,
tone, and affect after Planner output, but left `payload_type`, role, stance,
function, and evidence mode untouched. In v79 this produced 21 explicit
`no_story + personal_story` rows and 9 polite rows whose semantic job was advice,
correction, or analysis. The Writer was then asked to satisfy contradictory
instructions, and the metric was blamed on Writer realization alone.

**How to apply.** Validate cross-field contracts after every deterministic
overlay, at the last point before the Writer. A field-level target is not met
merely because the desired label appears in JSON. Replay the validator on prior
generation records to quantify how often the defect actually occurred before
paying for another run.

## 2026-08-16 — Reproducibility includes rejected experiment arms

**What happened.** Several behavior flags were written to `run_config.json` but
omitted from resume/extension comparison, and v77-v79 reused v76's policy string.
That made artifacts readable but allowed incompatible behavior to share lineage.

**How to apply.** Keep one authoritative experiment-field list, use it for every
lineage check, bump the policy for behavioral changes, and provide an explicit
`off` arm that restores the prior behavior. Git preserves deleted source; the
version log explains why it was deleted and what replaced it.

## 2026-08-16 — Read the scorer before theorising about the metric

**What happened.** I told the user the story allocation was correct and should
not be changed, on the strength of `mean_story_probability` showing PASS at n=10.
The user asked me to check how the metric is actually computed. It is
StorySeeker's P(story) **averaged over every comment in the thread** — and per
thread the generator overshoots real by 1.5–2.4×. It "passes" only because four
threads overshoot and six undershoot, and MWU/KS are unpaired tests that cannot
see per-thread mismatch.

Extending the check to all twelve metrics changed the picture of the whole
project. Of the four metrics that pass, only `avg_depth` and
`structural_virality` match per thread, and both are fixed by the matched sampler
rather than won by generation. Meanwhile `self_bertscore`, which has never passed
in any version, has a 6.9% mean relative error with 10/10 threads inside ±20% —
it fails on a *uniform* +0.03 offset, which is the signature of one global
constant, not of content quality. Four releases had been aimed at it as if it
were a content problem.

**How to apply.** Before designing anything for a metric, open its scorer and
write down what it computes and how it aggregates. Then check whether a passing
metric passes *per thread* or by cancellation, and whether a failing metric fails
by magnitude or by consistent sign. Those two questions change what the fix even
looks like.

## 2026-08-16 — Never approximate a metric that is cheap to compute

**What happened.** I wrote my own self-BLEU (mean of each comment's maximum
4-gram overlap against any other) to judge four runs, and reported that a change
had closed 87% of the `self_bleu_4` gap. `score_thread_self_bleu.py` is pure
n-gram, needs no model, and runs in seconds. Run against it, the same change
moved the metric from 0.03775 to 0.03750 — nothing. I had to retract the claim in
the next message.

This is the mirror of the earlier "measured pooled comment-level CV instead of
per-thread `length_cv`" error: same failure, opposite direction. A statistic that
*resembles* a metric can differ from it by an order of magnitude in effect size,
because aggregation is most of what a metric is.

**How to apply.** Use the real scorer. If it needs a model and that is too slow
for an inner loop, say explicitly that the proxy is a proxy, and never state a
headline result from it.

## 2026-08-13 — Read the whole related codebase before diagnosing, not slices of it

**What happened.** Diagnosing why generated long comments came out at ~0.72x their
matched length, I read `prompts.py`, `backend.py`, and `length_policy.py` in
partial line ranges and surveyed the 9,290-line
`scripts/sampling_generator/run_sampled_reddit_generator.py` only through an
outline and greps. From those slices I concluded the beat budget in
`long_form_planning.expected_development_beats` (one beat per 80 words) was the
cause, changed it to one per 35 words, spent an API run, and measured no
improvement (0.70 -> 0.72).

The real cause was visible only by reading a file I had never opened in full:
`reply_planning.render_direct_reply_planner_prompt` has its own compact JSON
schema that simply **omits** `development_plan`. Every long slot at depth >= 1
(33 of 77 long slots) therefore received no development guidance at all. The
beat budget was never the binding constraint for those slots.

**Why it matters.** A partial read produces a plausible-looking cause. Acting on
it costs a full generation run plus an evaluation run, and the null result is
ambiguous — it looks like "the fix was too weak" rather than "the fix was aimed
at the wrong thing." I also reported an improvement (0.70 -> 0.84) that was an
artifact of a double-counting bug in my own analysis script, and had to retract it.

**How to apply.**
- Before proposing a cause for behavior in this repo, read every file on the
  active path end to end, not the region a grep pointed at. In this codebase
  that means the CLI, the backend adapter, *every* prompt builder (there is more
  than one per role — root planner, direct-reply planner, low-info writer), the
  shared generator, and the policy modules.
- When several prompt builders exist for one role, diff their schemas against
  each other. A field present in one and absent in another is a likely defect.
- Verify an analysis script against a ground-truth count before trusting its
  aggregates. `generation_records.json` has one record per comment *and* each
  record nests its replies, so recursing into `replies` double-counts.
- Change one mechanism per API run, and predict the expected magnitude before
  spending the run so a null result is interpretable.

## 2026-08-14 — "No metric depends on it" is not a reason to delete a control

**What happened.** Rebuilding the Writer prompt (v74), I dropped the
semantic-difference contract with the documented reasoning that no currently
passing metric depended on it. No metric *measured* plan echo, so nothing
objected. Plan echo — the Writer reproducing its own `semantic_move` verbatim —
went from 10.2% to 25.8% of slots in that release, and to 34.7% among comments of
25+ words. The Writer had become a transcription layer.

**How to apply.** Before removing a control, ask what it *does*, not which metric
covers it. If nothing measures the failure mode it prevents, that is a reason to
add a measurement, not a licence to delete the control. Corollary: a control that
exists only to prevent a failure no metric watches is exactly the one that will
be deleted by a metric-driven cleanup.

## 2026-08-14 — Apply a fix to every path, or the run is unattributable

**What happened.** v74's focused prompt was gated on
`not backend.should_use_low_info_writer(task)`, so 106 of 522 slots (20%) kept the
old ~15,468-character prompt. Blocks the new tests asserted absent were still
present in the run, and the release's headline result could not be cleanly
attributed to the change.

**How to apply.** After changing a prompt or rule, grep for every renderer of the
same control and confirm each one got it. In v75 the realization rule was
rendered on both the focused and low-info paths, with a test on each.

## 2026-08-15 — Verify the completeness checker is itself complete

**What happened.** To avoid changing two variables in one run, I wrote a script to
diff a run's config against every argparse default. It reported five differences
and I trusted it. It matched argparse dest names (`plan_quality_repairs`) against
config leaf keys (`repair_rounds`); those names differ, so it silently skipped
that field. The next run changed two variables — the intended route lock and
`plan-quality-repairs` 0 → 3 — costing $3.75 in 914 unintended planner repairs and
confounding every diversity metric in the result.

**How to apply.** A "systematic" check that maps between two naming schemes must
be validated against a known-positive case before being trusted. Better: diff the
two runs' `run_config.json` trees directly against each other, rather than
inferring intent from defaults. Never describe a check as exhaustive without
having tested that it catches something it should catch.

## 2026-08-15 — Dry-run every command before handing it to the user

**What happened.** Three consecutive commands failed in the user's shell: one used
a `--seed-pool` flag that does not exist, one omitted
`--writer-hard-recovery-rounds 0` (default 2, every prior run used 0), and one
reused a tag that could no longer `--resume` because I had re-pinned core hashes
after that tag's preflight wrote its config. Each failure cost the user a round
trip, and I had asserted correctness by reasoning rather than by execution.

**How to apply.** Run the command on a throwaway tag with `--prepare-only`, then
delete the directory. Separately verify whatever `--prepare-only` skips —
in `run_generate.py` it returns at :715, before the API-key check at :719, so the
credential path needs its own test. Reasoning about a CLI is not verification.

## 2026-08-16 — Measure where the target behaviour actually comes from, not where it feels like it should

**What happened.** Generated comments carry 0.08 specification-shaped tokens each
against a real 0.55, so P4a licensed the speaker to invent facts about their own
equipment and history, gated on `first_person_experience_slot`. Run v76b measured
it on seed 8 and concreteness moved the *wrong* way: 0.05 → 0.02 overall, and
0.083 → 0.024 on the licensed slots themselves.

The hypothesis was refutable before the run and I never tested it. Across the ten
matched real threads, 78 of 114 spec-carrying comments (68%) contain no
first-person frame at all: real concreteness is how someone who deals with the
subject talks about it, not personal narrative. The gate selected the wrong
slots. Worse, replacing a vague blanket ban with an explicit "about the product
under discussion, name only what is visible above" made the binding constraint
*sharper* on exactly the detail real comments are full of.

**A second error in the same change.** The licensed rule read "what you shot or
set it to", "your own gear". That is camera vocabulary in a system whose standing
requirement is domain generality, and no test caught it because every test ran on
the camera domain.

**How to apply.**
- Before licensing or banning a behaviour, measure which real comments exhibit
  it and what they have in common. One `grep` over the held-out threads would
  have killed the first-person gate for the cost of a minute.
- Prefer a signal that holds on *every* thread. Specification-shaped tokens range
  from 0% of comments (seed 1) to 64% (seed 5) by thread; quantities (real 12.3×
  generated) and proper nouns (1.85×) separate real from generated on all ten.
  A thread-dependent signal is a domain-specific fix wearing a general name.
- Assert the absence of domain vocabulary in any rule text that ships to the
  Writer. `test_the_named_rule_carries_no_domain_vocabulary` now does this.
- Replacing a vague prohibition with a precise one can *tighten* it. When the
  goal is to loosen, check the replacement against the behaviour you want to see
  more of, not only against the behaviour you meant to keep banning.

## 2026-08-16 — Look for the defect in the output before theorising about the cause

**What happened.** Three releases of work — the plan-echo route lock (v75), the
own-fact license (v76), the speaker roster (P4b) — were aimed at `self_bleu_4`
and `self_bertscore` on theories about *why* comments resemble each other. None
of them measured what the resemblance actually consisted of.

One document-frequency count over the generated comments answered it in seconds.
The 4-gram "that's the part" appears in 12 of 186 generated comments (6.5%),
alongside "yeah that's the", "the part that makes", "that's the bit". In the
matched real thread the most-shared 4-gram appears in 3 of 200 (1.5%) and there
is effectively no shared phrasing at all. The same frame family was recorded at
20% of comments back at policy v72 and 0 times in 39,265 real tokens, and it
survived a rewording (v73), a prompt rebuild (v74) and a route lock (v75).

`template_phrase_reused` already fires on it: 38 slots in v76a, 40 in v76b, about
21%. Every one was discarded — all 186 slots ran exactly one attempt and 85 were
accepted through `accepted_first_pass_distribution_diagnostics`.

**How to apply.** For any distributional metric, first ask what the generated
text is doing that the real text is not, at the level the metric measures.
`self_bleu` is n-gram overlap, so count n-grams. Do that before designing a
mechanism, and check whether an existing validator already detects it — this one
had been detecting it, and being ignored, for four releases.

## 2026-08-15 — Honour the user's stated priorities

**What happened.** The user said more than once that politeness was
de-prioritised and that `self_bleu`, `emotion`, `mean_story_probability` and
`length_cv` were the real targets. I kept opening analyses with politeness tables
and eventually ran a whole natural experiment on it, and the user objected.

**How to apply.** Re-read the stated priority order before choosing what to
measure and what to lead a report with. A metric being *interesting* or
*explanatory* does not promote it above what the user asked for.

## 2026-08-17 — A proposition-only handoff is not a complete Planner contract

**What happened.** The focused Writer path retained the Planner's semantic move
and the controls tied directly to named metrics, but dropped the comment
function, payload, speaker role, voice, evidence, stance, and other discourse fields.
The plan could say “rant” or “bare reaction” and the default Writer would never
see that assignment, so its generic helpful-answer prior remained free to win.

**How to apply.** Audit a Planner→Writer boundary by behavioral dimension, not
only by checking that the main text field survives. For every default Writer
path, prove end to end that it receives (1) what the turn says, (2) what social
and discourse act it performs, and (3) the structural/surface constraints. Keep
each control once; compactness is not permission to discard authority.

The same boundary audit must include indirect paths. In v81, matched-real words
were not placed directly in the Prompt, yet a shared “surface” classifier turned
`thanks` into a gratitude tone label that did reach the Prompt. A derived label
is still data leakage when its source is semantic matched text.

## 2026-08-17 — A high acceptance percentage is not structural completeness

**What happened.** The v80 seed-8 artifact reported 186 generation records and
185 comments. One hard Writer failure was persisted as `comment=null`, while the
post wrapper printed `policy=persist_valid_comments`. The output audit used a
minimum accepted-share threshold, so 185/186 = 99.46% could still proceed to
evaluation even though the repository contract requires every matched slot.

The immediate S99 failure was also internally deterministic: the scheduled quote
opener required copying the exact visible parent line, while `parent_copy` was a
hard failure. Retrying the same contradictory contract three times could not fix
it.

**How to apply.** Treat structural coverage as equality, never a quality
percentage: planned slots = Writer records = generated comments = rendered tree
nodes, with zero skips. Enforce it before persistence and again before evaluation.
When a Prompt intentionally requests a syntax that resembles a guard violation,
define a narrow, auditable exception for that syntax rather than disabling the
guard. Here the exception requires an explicitly scheduled markdown excerpt and
an independent response; unscheduled or whole-parent copying still fails.

## 2026-08-17 — A validator after deterministic normalization may be theatre

**What happened.** Plan quality appeared to validate invalid perspectives and
wrong branch routes, and to repair perspective concentration. On the active
path, deterministic metadata normalization ran first: invalid perspectives were
mapped to `seed_local`, and tree topology overwrote both branch and perspective
ownership. Two checks therefore compared already-canonical values, while a
slot-local concentration repair asked the Planner to change a field that the
next normalization pass overwrote again.

**How to apply.** Trace every quality check together with the mutations before
and after it. If normalization owns a field, record the normalization event and
test the normalizer; do not retain a downstream check that cannot see the raw
value. A collection-level consequence of structural ownership may remain an
audit warning, but it must not trigger an impossible per-slot LLM repair.

## 2026-08-17 — A report called “matched” still needs every row's join audited

**What happened.** The content-profile tool correctly loaded the matched real
comment text for lexical comparisons, then silently loaded every domain
GoEmotions and StorySeeker file for its model-scored rows. The heading still
said REAL target, so plausible-looking aggregate numbers could have driven the
next Prompt change even though they described a different population.

**How to apply.** Treat diagnostics as production scientific code. Trace every
reported row back to its source key, add a distractor thread to the join test,
and store evidence provenance in the output. When n=1 is used for qualitative
debugging, override misleading saved test statuses with `descriptive_only_n1`;
do not let a mathematically returned p-value become an inferential claim.

## 2026-08-17 — Separate target selection from realization before changing either

**What happened.** A generated metric gap was being discussed as a generic
Planner/Writer problem. Reconstructing the exact excluded-real template draws
showed that their distribution passes all 12 MWU and KS tests at both N=10 and
N=150. The historical n=1 output then made the missing stage visible: polite
target 0.249 was close to real 0.232 but Writer output was 0.059; story target
0.128 was close to real 0.111 but Writer output was 0.249. Emotion entropy was
the opposite case: generated 1.5358 almost exactly realized target 1.5359, while
that single high-variance target draw differed from real 1.9459.

**How to apply.** Persist every stochastic upstream target and report
real→target separately from target→output. Validate a distribution sampler as a
distribution, never by its per-thread correlation or one draw. Only change the
Planner sampler when its multi-thread target distribution fails; change the
Planner/Writer contract when the output is insensitive to a good target.

## 2026-08-17 — Evaluation must not repair the artifact it claims to measure

**What happened.** The formal path audited Writer output, then sent it through a
1,177-line cleanup program before scoring. Even with GPT disabled, that program
could normalize tree metadata and strip or naturalize text. Separately, the core
contract pinned active evaluator hashes whose files were untracked, and one
dirty calibration module whose pinned bytes were not present in the commit.
Both systems could say “healthy” without proving that a checkout would reproduce
the scored artifact or the scorer.

**How to apply.** Reject contaminated output; do not clean it into eligibility.
Stage accepted generation byte-for-byte. A source hash is provenance only when
the source is recoverable, so verify both git tracking and the transitive local
import closure. Keep legacy revisers out of default active parity when they are
not part of the experiment.

## 2026-08-17 — Import closure does not prove dynamic runtime closure

**What happened.** The first provenance audit followed package imports and
reported zero omissions, but active runners also launched Python files by path
and the backend dynamically imported the token tracker after changing
`sys.path`. Those files affected execution while remaining outside the claimed
active source set.

**How to apply.** Treat dynamic imports and subprocess entry points as explicit
contract edges. Pin and track those runtime sources, and make the closure audit
follow sibling-script imports in addition to package-relative imports. Report
declared, active, tracked, and unpinned counts separately.

## 2026-08-17 — Audit rendered Prompts by path and visible target

**What happened.** The normal focused Writer Prompt had no repeated long lines,
but the low-information branch still rendered the same move and controls through
multiple old blocks. Separately, root plans legitimately stored the shared enum
value `answers_parent`, and the Writer displayed it even when its visible target
was the seed post. Source-level searches did not expose either contradiction as
clearly as rendering representative root, reply, substantive, and low-info
Prompts.

**How to apply.** Prompt review must cover every routing branch and inspect the
final rendered text. Preserve raw Planner values for audit, but translate them
at the Writer boundary when their human-facing meaning depends on whether the
visible target is a post or parent. A specialized Writer path should reuse the
same compact contract primitives unless evidence requires extra information.

## 2026-08-17 — Replay every slot; representative Prompts do not prove routing

**What happened.** Representative root/reply and substantive/low-information
Prompt renders looked coherent, but the router checked `utterance_mode` before
`payload_type`. In the 186-slot thread, six `soft_helpful` tasks and one
`correction` therefore entered a low-information Prompt that explicitly banned
advice, explanation, and caveats. Each individual Prompt was internally
well-formed; the task-to-Prompt choice was wrong.

**How to apply.** A Prompt audit must enumerate every recorded task through the
actual route predicate, not only render one example per intended branch. Treat
payload semantics as the authority for specialized low-information paths;
surface shape may refine an eligible route but must not downgrade a
substantive contract. Report the route matrix and assert that every selected
payload belongs to the path's allowed set.

## 2026-08-17 — Recompute dependent controls after the authoritative contract

**What happened.** `real_tone_slot` was derived before later surface overrides
and restoration of Planner-owned fields. A task could finish as a neutral
datapoint or correction while retaining `pure_acknowledgement` from an earlier
social shape. The validator also enforced gratitude in only one direction:
gratitude affect required a social reaction, but a `gratitude_reply` or
`social_close` did not require gratitude/relief and a compatible payload.

**How to apply.** Document which fields are authoritative and which are derived.
After the last authoritative mutation, recompute every Writer-facing dependent
control once. Express coherence contracts bidirectionally when either label can
activate Writer behavior, and test both valid and reverse-invalid cases before
the Writer boundary.

## 2026-08-17 — Git is the ablation archive; active Prompts do not preserve bugs

**What happened.** `own-fact-license off` was called byte-compatible with an old
policy, so the active default deliberately kept a known fact ban. But the
equipment helper still offered invented gear on 78 of 186 Prompts, and 61 then
revoked that permission with a personal-experience ban. Passing tests asserted
the historical wording rather than the absence of the conflict.

**How to apply.** An ablation flag may preserve a meaningful alternative, not a
known contradictory implementation. Once the old bytes are committed and
versioned, remove the contradiction from the active path. Test the rendered
Prompt combination across all slots, not each helper sentence in isolation.

## 2026-08-17 — Structural matching must not smuggle in semantic persona

**What happened.** Matched speaker grouping was a valid leakage-safe join, but
the same flag also assigned each speaker an invented tenure, use case, and kit.
That made it impossible to enable realistic recurring authors without adding
unsupported biography and another source of Prompt conflict.

**How to apply.** Keep matched structure and generated semantics in separate
types and controls. A structural speaker owns only anonymous slot membership and
OP status; Planner-owned evidence, story, voice, and affect supply what the
person says. If a structural mode is requested, fail on an impossible join
instead of silently degrading to a different structure.

## 2026-08-17 — A contract fix must cover every Planner route

**What happened.** v89 repaired the root Planner's contradiction between a
required story and a blanket no-anecdote rule, but the specialized direct-reply
Planner kept the same ambiguity in different wording. Root Prompt tests passed,
yet reply story slots still had to guess whether any synthetic personal event
was allowed.

**How to apply.** When a semantic contract is shared across root and reply
planning, define its invariant once and render it on every route. Test the final
Prompt of each route for both permission and prohibition; a source-level fix in
one large Prompt is not evidence that the Planner→Writer contract is globally
coherent.

## 2026-08-17 — A global permission must preserve the slot gate

**What happened.** The `named` arm correctly licensed only substantive slots,
but its system Prompt applied the concreteness instruction globally. This
silently bypassed the resolver for micro/short comments and conflicted with
their visible-only, no-extra-fact user rules. Per-slot unit tests did not catch
it because they changed the module flag after the system Prompt had already
been configured.

**How to apply.** A system-level exception should authorize a lower-level rule,
not repeat the behavior globally. Configure the real environment before
capturing the system Prompt in tests, then combine it with both a licensed and
an unlicensed final user Prompt. Measure the gate's selected share against the
real property it is meant to reproduce before turning the arm on.

## 2026-08-17 — An off flag must stop planning data that will be dropped

**What happened.** `domain-claim=off` was designed as a delivery-only ablation:
the Planner still assigned a fact, then the Writer registry discarded it. That
kept an old experiment comparable but violated the active handoff invariant—the
Planner could organize visible fields around content the Writer never received.
It also paid input/output tokens for an intentionally unused field.

**How to apply.** Trace optional information from Prompt request through
normalization, storage, and final consumer. When a mode disables delivery,
either remove the upstream request or preserve the data through the handoff; do
not plan-and-drop. Keep the schema stable with a literal empty value, enforce it
after parsing, and state where the complete semantic contribution must live.

## 2026-08-18 — Topology owns whether reply fields are applicable

**What happened.** A root slot's long-form repair supplied every required beat,
but the model also filled `reply_delta_type=social_close`. The social validator
correctly rejected that reply contract, even though the anonymous structure said
the slot had no parent. A generic root Prompt still contained an entire obsolete
direct-reply rule set after direct replies had moved to their own Planner.

**How to apply.** Treat parent topology as the authority for reply-only fields.
Normalize inapplicable controls before semantic-quality selection, log nonempty
overrides, and keep the stable schema with literal empty values. When routing a
case to a specialized Prompt, remove the old case-specific prose and helpers
from the generic Prompt; otherwise duplication becomes both token waste and a
source of contradictory model output.

## 2026-08-18 — Permission to be concrete is not a factual supply chain

**What happened.** The `named` Writer arm passed every offline Prompt gate but
the paid v95 thread still had 5 distinct model designators against 40 real and
20% digit-bearing comments against 60%. Most Writer turns had only a broad
permission plus one or two seed tokens. The same rule also forbade reusing a
name from an earlier comment, so a thread about three products was pressured to
paraphrase their names away.

**How to apply.** Audit what information is actually available at the final
consumer, not just whether a permission sentence exists. Repeating the shared
subject is discourse cohesion; repeating the same fact is redundancy. Supply
bounded, source-traceable particulars upstream and test their survival through
normalization, registries, prompt rendering, anchors, validation, and logs.

## 2026-08-18 — Immediate-parent novelty does not protect a deep branch

**What happened.** Every reply had a new delta label relative to its parent, yet
a nine-turn chain repeatedly reframed the same fixed-lens/long-term boundary.
Local pairwise novelty was true while branch-level semantic novelty was false.

**How to apply.** For a deep reply, make the whole ancestor chain a compact
exclusion ledger. Reusing the entity is allowed, but the new fact, test,
condition, consequence, or decision boundary must differ from every ancestor,
not only from the immediately preceding sentence.

## 2026-08-19 — v98

### Measure the causal claim on real data before you build the fix, not after

`sentence_rhythm` was fully written, tested, wired, pinned, and self-tested
before a falsification test was run on the hypothesis that produced it. The test
took four minutes and rejected the claim: real comments that differ in the
measured typing habits are only 0.003-0.011 lower in function-word cosine than
ones that share them, against a generated-vs-real gap of 0.134, and real
comments with *uneven* sentence lengths are slightly **more** alike, not less.

The module was still worth keeping for the metrics it does move, so nothing was
wasted, but that was luck. The rule: when the mechanism is "X causes metric M",
the reference corpus can usually answer whether X and M covary **before** any
code is written. Do that first.

### A gap that survives trimming is a distribution shift, not a tail

Reading the six highest-BERTScore generated pairs made the cause look obvious --
two comments making the same argument in different words. Trimming said
otherwise: the gap is +0.0163 untrimmed and +0.0154 after dropping the top 20%
of pairs on both sides. It was uniform.

Reading the highest-scoring **real** pairs is what explained the asymmetry: real
threads reach F1 > 0.74 through shared image URLs, not shared content. Always
read the control's extremes, not only the treatment's.

### A prohibition can be wider than the thing it protects

v96's `no_story` instruction banned "past action, event, before/after change" to
stop narrative. StorySeeker scores narrative *sequence*. The ban removed tense
from 85% of a thread: `have` at 11% of its real rate, `will` at 1%, and a
lexicon of 2,670 distinct types against a real 3,645. That single instruction is
the whole `self_bertscore_mean_f1` gap.

Before writing a prohibition, name what the metric actually keys on and bar that,
not the nearest larger category.

### Grep the rendered prompts for pairs of rules that contradict each other

`writer_grounding` exists because the fact ban was written in eight places that
disagreed. It happened again: 247 of 532 v97 prompts (46.4%) carried both "Be
particular rather than general" and "no past action, event". The prompts are
stored verbatim in `generation_records.json[].prompt`, so this check costs
nothing and should run every version.

### `from .module import CONSTANT` breaks every arm switch

Third time. `backend.py` imported `ACTIVE_RHYTHM_PROFILE` by value, which would
have frozen the empty dict at import time. Cross-module mutable state is always
`from . import module` and read as `module.NAME` at call time. The v97 lesson
said the same thing about `TONE_LENGTH_FIT_ENABLED`.

### A test that installs state before `configure_generator_backend` is testing nothing

`configure_generator_backend` reinstalls every profile from
`GENERALIZED_DOMAIN_PROFILE` and re-reads every arm from `os.environ` on each
call. State set before it is silently replaced. Fixtures must take the state as
a parameter and install it after configuration; arms must go through
`mock.patch.dict(os.environ, ...)`.

### Pinned sources must be `git add`ed before `repin_core_contract.py --write`

The script refuses to pin an untracked file ("active sources must be recoverable
from git") and exits 2 without writing anything -- including the entries that
were not blocked. Add the new files first, then re-pin once.
