# v98 worklog — drawn typing rhythm and length calibration

Date: 2026-08-19

Policy: `generalized-card-v2-drawn-typing-rhythm-length-calibration-v98-20260819`

## What v97 left

7 PASS / 1 PARTIAL / 4 FAIL over ten matched threads. The four the user ranked
first: `self_bleu_4` (passing but weak, MWU 0.186), `self_bertscore_mean_f1`
(0.0003, Cliff 0.96), `length_cv` (0.021, Cliff -0.62, a regression from v96),
and `emotion_entropy` (passing at 0.326 but Cliff -0.27 and W 0.223, which will
not survive N=150).

## Two hypotheses measured and rejected before the third was implemented

This is the part worth keeping. `self_bertscore` had three plausible causes and
the first two were wrong.

**Rejected: length spread.** Every unordered comment pair of the six smallest
threads was scored with the evaluator's own BERTScore and binned by the pair's
length ratio. The gap concentrates entirely in same-length pairs (+0.0363 at
log-ratio < 0.35, **-0.0236** above 2.2 -- generated is already *more* diverse
than real for very unequal pairs). But reweighting the generated pairs onto the
real length-ratio mix moves the mean only 0.5090 -> 0.5057, one fifth of the
0.0163 gap.

**Rejected: a duplication tail.** Trimming the top of both distributions leaves
the gap untouched -- +0.0163 untrimmed, +0.0154 after dropping the top 20% of
pairs on each side. Reading the highest-F1 generated pairs looked damning (two
comments making the same argument in different words), but reading the real ones
explained the asymmetry: real threads reach F1 > 0.74 through **shared image
URLs**, not shared content. The Planner is clean -- zero exactly duplicated
`semantic_move` values over 532 slots, at most 1.3% of in-thread plan pairs above
0.35 content-word Jaccard.

**Rejected: the surface register.** Same-length pairs differ most in
function-word cosine (real 0.368, generated 0.502). A falsification test on the
excluded real corpus killed the causal claim: real comments that *differ* in the
measured typing habits are only 0.003-0.011 lower in function-word cosine than
ones that share them, against a 0.134 gap, and real comments with *uneven*
sentence lengths are slightly **more** alike, not less. `sentence_rhythm` was
already written when this test ran. It was kept -- for the metrics it does move
-- and its docstring now leads with the rejection.

## What survived

A uniform lexical narrowing:

| | real | generated |
|---|---:|---:|
| distinct word types | 3,645 | 2,670 |
| types / sqrt(tokens) | 21.02 | 15.95 |
| hapax rate | 0.502 | 0.427 |
| top-500 type coverage | 0.783 | 0.830 |

Per-comment type-token ratio at a fixed 30 tokens is *higher* in generated text
(0.891 against 0.866). No comment is individually thin; the thread draws from a
smaller lexicon, which lifts every pair equally -- which is exactly the flat
trimmed-mean signature.

The cause is one instruction applied to 453 of 532 slots. v96's `no_story` text
bans tense, not narrative: "no past action, event, before/after change, or
then/after pacing".

| on `no_story` slots | real | generated |
|---|---:|---:|
| past-tense verb | 0.543 | 0.181 |
| future / `'ll` | 0.226 | 0.031 |
| present perfect | 0.167 | 0.031 |

`have` at 11% of its real rate, `will` at 1%, `they` at 11%, `to` at 54%. The
fallback register is a timeless conditional: `the` 147%, `if` 225%, `whether`
1800%, `matters` 2900%, `part` 1370%.

It was also a live prompt contradiction. Under `--own-fact-license named` the
non-story grounding rule is "Be particular rather than general", and **247 of the
532 rendered v97 prompts (46.4%) carried that line and the tense ban together**
-- the same defect `writer_grounding` exists to prevent. `story_scope` bars the
second event and the then/after pacing StorySeeker actually scores, and permits
one completed past fact and ordinary future.

## Implementation

Three new focused modules, one renderer change, one ledger wiring:

- `story_scope.py` -- what a `no_story` slot is barred from. Two arms.
- `length_calibration.py` -- inverts the fitted transfer function
  `log(realized) = 0.3835 + 0.8925 * log(asked)` (n=532, R2 0.894) so the cue
  asks for what realizes the slot. Only the number in the cue is calibrated.
- `sentence_rhythm.py` -- seven habits drawn per slot at each band's measured
  rate, from evaluation-excluded threads.
- `surface_typography.apply_final_punctuation_habit` -- the staged v98 work,
  landed. A trailing period only.
- `prompts._focused_thread_ledger` -- renders the reused mid-comment routes the
  `full` arm has had since v66.

## Arms

| flag | v98 default | reproduces v97 |
|---|---|---|
| `--no-story-scope` | `sequence` | `tense` |
| `--length-calibration` | `measured` | `off` |
| `--sentence-rhythm` | `measured` | `off` |
| `--final-punctuation` | `measured` | `off` |
| `--route-ledger` | `on` | `off` |

Domain profile schema 14 -> 15.

## Reproducibility note on `length_calibration`

This is the one profile in the system that is **not** domain-measured. The
transfer function is a property of the model and the prompt, not of the domain,
so it is a recorded constant rather than a domain-profile entry. Both the target
(`task.real_word_count`) and the realized count (`comment.word_count`) are in
every generation record, so any future run's artifact refits it without
regenerating anything. Refit when the model changes, not when the domain does.

`sentence_rhythm` and `final_punctuation` are fully domain-adaptive: measured per
domain from that domain's evaluation-excluded threads, no written-down
frequencies, bands with fewer than 40 samples omitted rather than guessed, and
`test_cue_carries_no_domain_vocabulary` asserts the cues carry none.

## Zero-API result

- 446 tests pass (v97: 369). Ruff clean.
- 100/100 pins, zero drift, no untracked active source, no unpinned local import.
- Both parity scopes healthy.
- Self-test passes with all five arms on **and** with all five off.
- Schema-15 profile rebuilds over 424 excluded threads, 0 seed overlap, all five
  sub-profiles available.
- All 532 v97 slots re-rendered through the v98 prompt: rhythm rule on 532/532,
  realized habit rates track the measurement per band, calibrated ask asserted on
  every slot, prompt +474 chars (+8.5%).

## Verification gate before N=10

One paid seed-2 run, then read every comment. Each item is a prediction this
version makes:

1. Past-tense verbs near 0.54 of comments, not 0.18; `will`/`'ll` present at all.
2. Distinct word types up from 2,670-equivalent; types/sqrt(tokens) toward 21.
3. Exclamation marks present at roughly the band rates, not zero.
4. Realized/target near 1.0 across bands, especially 11-15w and 251-400w.
5. `mean_story_probability` still in range -- the loosened scope is the risk here.
6. No invented personal experience. The grounding rule is unchanged and licensed,
   but this is the failure mode the tense ban was incidentally suppressing.

## Paid seed-2 gate result — 2026-08-19

Tag `generalized_card_camera_gpt54_v98_rhythm_seed2_20260819_v1`: 45/45 comments
in one attempt, 85 requests, `$0.3699`, no retries and no degraded comments.

**The central prediction failed.** Past-tense rate on `no_story` slots moved
0.175 -> 0.220 against a real 0.622, and `self_bertscore` on this thread moved
0.4959 -> **0.5005** against a real 0.4878 — the wrong way.

The instruction reached the model: 41 of 45 prompts carry the `sequence` scope
and **0 of 45** carry the old tense ban. It was simply not obeyed as a
behavioural change. Splitting the 41 `no_story` slots by their gating fields
does not rescue it either — `allow_first_person_frame=True` slots are at 0.278
against `False` at 0.174, both far under 0.622.

**What this establishes, and it is the reusable part:** removing a prohibition
is not the same as asking for the behaviour. The exclamation mark had never been
banned and v97 still produced zero in 532 comments; v98 *asked* and got 0.133.
Past tense was un-banned and did not appear. A habit the model does not reach
for by default has to be requested, not merely permitted.

**What did move, on the same 45 comments:**

| | real | v97 | v98 |
|---|---:|---:|---:|
| types / sqrt(tokens) | 14.44 | 13.87 | **14.22** |
| self_bleu_4 (proxy) | 0.0454 | 0.0430 | **0.0245** |
| exclamation | 0.044 | 0.000 | 0.133 |
| parenthetical | 0.178 | 0.022 | 0.133 |
| no final punctuation | 0.222 | 0.044 | 0.133 |
| semicolon | 0.022 | 0.111 | **0.000** |
| dash-joined clause | 0.089 | 0.244 | **0.000** |
| future / `'ll` | 0.222 | 0.044 | 0.089 |
| digit | 0.600 | 0.356 | **0.267** |
| length_cv | 0.877 | 0.871 | 0.997 |

n=1 thread; these are descriptive, not tests.

### Three defects the gate exposed, all fixed before N=10

1. **The pacing clause contradicted the skeleton.** `infer_surface_skeleton`
   names the slot's own sentence count on 348 of 532 slots. A 115-word slot was
   told "about 12 sentences" and "about 16 words" per sentence — 192 words
   against a 133-word ask. The band median is now withheld wherever the
   slot-specific count exists.
2. **The digit cue regressed the digit rate**, 0.356 -> 0.267 against a real
   0.600. It was phrased conditionally ("if this turn has a number in it
   already"), which is an easy out. It now asks directly and points at the
   numbers the slot is licensed to name.
3. **Dash suppression drove the rate to exactly zero** against a real 0.089,
   because the habit was suppression-only and nothing asked for it on the slots
   the measurement licenses. It is two-sided now. The semicolon stays
   suppression-only and the reason is recorded in the table.

### Still open

- `self_bertscore_mean_f1` has **no validated fix**. The `no_story` change
  removed a real prompt contradiction (247 of 532 v97 prompts carried two rules
  a slot cannot both satisfy) and is worth keeping on those grounds, but it is
  not the mechanism. The lexical-narrowing measurement stands; what closes it
  does not.
- The length calibration's long band went 0.919 -> 0.709. The rhythm cue
  shortens comments, which shifts the transfer function the calibration
  inverts. This is the interaction flagged before the gate ran. n=10 slots in
  that band is too thin to refit from; refit from the N=10 artifact's 532.

## 2026-08-20 — the level of analysis was wrong, and the user was right to say so

Three word-level hypotheses for `self_bertscore_mean_f1` were tested against 22
evaluation-excluded real threads scored with the evaluator's own BERTScore. All
three are **rejected**, and two of them by the wrong sign:

| predictor | r with thread self-BERTScore |
|---|---:|
| types / sqrt(tokens) | +0.077 |
| distinct entity types per comment | +0.182 |
| past-tense rate | +0.395 |
| first-person rate | +0.640 |
| proper-noun rate | -0.428 |
| length CV | -0.281 |

Lexical breadth does not predict low self-similarity among real threads. The
generated narrowing (2,670 types against 3,645, 0.438x entity diversity, 10/10
threads) is **real and is not the cause of the metric**. Building the entity
licensing change on it would have been the fourth wasted mechanism.

n=22 with a self-BERTScore range of only 0.46-0.52 attenuates every correlation,
so a moderate effect is not excluded. A strong one is, and the sign is wrong.

### What the metrics actually share

Re-measured at the level of *what each comment does* rather than which words it
uses, over the ten matched threads with the real scorers:

| speech act | real | generated | ratio |
|---|---:|---:|---:|
| gives advice | 0.090 | 0.008 | **0.08** |
| complains | 0.026 | 0.122 | **4.64** |
| jokes / reacts | 0.041 | 0.109 | 2.64 |
| pushes back | 0.102 | 0.188 | 1.85 |
| hedges | 0.207 | 0.147 | 0.71 |
| reports own use | 0.115 | 0.068 | 0.59 |
| asks a question | 0.179 | 0.184 | 1.03 |

Speech-act *variety* is fine: mean thread act-entropy 1.589 real against 1.617
generated, 1.41 acts per comment against 1.47. The **mix** is not. The generated
thread has over-corrected past the "helpful assistant" failure of v29-v72 and
out the other side into a thread of complainers.

The emotion scorer says the same thing on the same comments: admiration
0.079 -> 0.017, love 0.035 -> 0.004, joy 0.027 -> 0.006, optimism 0.024 -> 0.002,
while annoyance goes 0.012 -> 0.076. Every warm label collapsed and one negative
label sextupled. That is `emotion_entropy`, `polite_rate`, `impolite_rate`, and
`neutral_rate` as **one defect**, not four.

### It is a realization failure, not a planning one

| planned tone | n | -> realized polite | -> realized impolite |
|---|---:|---:|---:|
| polite | 143 | 0.168 | **0.538** |
| somewhat_polite | 46 | 0.435 | 0.457 |
| neutral | 79 | 0.063 | **0.582** |
| impolite | 261 | 0.015 | 0.912 |

Plan marginal polite 0.270 / impolite 0.493 against a real 0.288 / 0.443 — the
Planner is close to correct. Realized 0.066 / 0.722. The Writer hits `impolite`
at 0.912 and cannot hit anything else, including `neutral`.

### What `Intel/polite-guard` keys on, over all 24,029 labelled real comments

| feature | P(polite \| feature) | base | lift | real | generated |
|---|---:|---:|---:|---:|---:|
| thanks / positive evaluation | 0.673 | 0.300 | **2.24** | 0.118 | 0.079 |
| advice marker | 0.410 | 0.300 | 1.37 | 0.175 | 0.100 |
| addresses "you" | 0.368 | 0.300 | 1.23 | 0.391 | 0.444 |

Median length is 56 words for `polite` against 29 for `impolite`, so the length
effect recorded in an earlier session is real but secondary; v97's
`tone_length_fit` already places polite on the long slots and realization is
still 0.066.

The generator under-produces the two features with the highest lift and
over-produces complaint. Reading the planned-polite comments the classifier
called impolite confirms it is not a classifier artefact in the usual sense —
they are first-person narratives with no addressee and no evaluation ("I had a
little Sony in my bag for a while and it was weirdly the easiest camera to keep
on me"), while every one it called polite carries an explicit affirmation or
recommendation ("I'm with you on wanting both", "If low light is a real
priority, I'd lean Sony here").

### The next mechanism, not yet built

Restore positive evaluation and advice as *schedulable moves* at their measured
real rates, the way `sentence_rhythm` schedules typing habits — rather than
leaving them suppressed by the v29-v72 anti-assistant guards. `advisor_max_share`
is 0.28 and realized advice is 0.008, so the guard is not the binding
constraint; the Writer simply is not asked. Same finding as the exclamation
mark: **removing a prohibition is not the same as asking.**

Do not build it before testing the causal claim on the reference corpus. That
rule has now saved two mechanisms in one session.

## 2026-08-20 process audit — what is and is not a bug

Asked directly whether the pipeline still has defects. Every arm and every stage
was checked against the artifacts rather than by reading code.

### Not bugs (checked and clean)

- **All nine arms take effect.** Each was flipped through `os.environ` and 102
  prompts plus their shaped text re-rendered and diffed: sentence_rhythm 102
  prompts changed, length_calibration 94, route_ledger 20, no_story_scope 97,
  turn_frame 62, long_form_layout 40, final_punctuation 5 shaped, typography 16
  shaped. `tone_length_fit` shows 0 only because tone is assigned at *planning*
  time and the probe replays frozen tasks.
- **`tone_length_fit` really ran in v97.** Planned polite by size band is
  0.220 / 0.136 / 0.225 / 0.431 / 0.600 / 0.538 against a measured conditional of
  0.251 / 0.162 / 0.263 / 0.520 / 0.638 / 0.720. The placement is correct; only
  the realization fails.
- **`distribution_stats` is correct.** Two-sided Mann-Whitney, two-sample KS,
  Cliff's delta as candidate-minus-real. Worth noting rather than fixing: the
  tests are *unpaired* while the data is paired by seed, which makes N=10
  p-values optimistic. That is why |Cliff| <= 0.10 is the target for N=150, not
  a small-N p-value.
- **Only three task fields are always empty** (`distribution_assignment`,
  `tone_overlay_slot`, `tone_overlay_instruction`) and `degraded_from` is empty
  because nothing degraded. No mechanism is silently dead.

### Real defects found

1. **The evaluation loses comments unevenly, breaking exact size matching.**
   `is_usable_comment` drops anything under two words, and the real side also
   carries `[deleted]`/`[removed]`. On `post04_seed011` the generated thread is
   scored at **24 comments against a real 22**; two other threads lose one
   generated comment each. Pooled it is 529 generated against 528 real, so the
   aggregate effect is negligible, but `--exact-matched-thread-size` is the whole
   design and `avg_depth`, `structural_virality`, `length_cv`, `self_bleu_4` and
   `self_bertscore_mean_f1` are all size-sensitive. Dropped items are genuine
   one-word comments -- "Rubbish", "Indeed.", "Nikon." on the real side, "Fair",
   "Exactly", "Agreed" on the generated side. **Not yet fixed.**

2. **The slot distribution schedule is never persisted.**
   `build_slot_distribution_schedule` records `tone_length_fit` and
   `tone_length_joint`, and neither reaches `discussion.json`. The arm is in
   `run_config.json`, so the run is reproducible, but the realized tone-by-size
   joint cannot be audited from an artifact without recomputing it.
   **Not yet fixed.**

3. **The length cue pointed the wrong way between 35 and 100 words. Fixed.**
   The "do not trim toward a medium-length answer" cue was gated on a
   written-down `real_words > 100`, but realized/target crosses 1.0 near 35, so
   every slot in between was told "Do not pad past it" while its measured error
   was undershoot. On the v98 seed-2 gate the 56-80 word slots realized 0.48,
   0.51, 0.54, 0.57, 0.57, 0.68, 0.74 of target while slots over 100 words were
   fine at a median 0.97. The cue now reads `ask_multiplier`, the same curve the
   calibration inverts, so the ask and the cue cannot disagree again.

### On refitting the length transfer function

The slope is stable -- 0.8925 over the 532 v97 slots, 0.8989 over the 45 v98
ones -- so the shape did not move. The intercept appears to shift, but the same
thread scored under v97 fits an intercept of **-0.057** against the N=10
aggregate's **+0.384**: thread-to-thread variance in the fit is larger than the
v97-to-v98 difference. Refitting from 45 comments would be fitting noise. Refit
from the N=10 artifact's 532 slots instead.

## The v99 mechanism, verified at thread level before any code was written

`polite_rate` and `impolite_rate` are thread-level metrics, so the causal claim
has to be tested at thread level. Over **412 evaluation-excluded real threads**
with at least 12 comments:

| thread-level predictor | r with polite_rate | r with impolite_rate |
|---|---:|---:|
| warmth-marker rate | **+0.727** | **-0.601** |
| mean comment length | +0.417 | -0.143 |
| advice-marker rate | +0.334 | -0.249 |
| negative-marker rate | -0.092 | +0.213 |

The dose-response is monotone across warmth quintiles:

| warmth rate | polite_rate | impolite_rate |
|---:|---:|---:|
| 0.067 | 0.183 | 0.513 |
| 0.144 | 0.231 | 0.488 |
| 0.210 | 0.305 | 0.447 |
| 0.277 | 0.381 | 0.372 |
| 0.434 | 0.490 | 0.269 |

This is the first predictor in the whole investigation that survives its own
falsification test. Contrast the three that did not: lexical breadth (+0.077),
entity diversity (+0.182), past tense (+0.395), all against
`self_bertscore_mean_f1`.

Where the generator sits, over the ten matched threads:

| | real | generated | ratio |
|---|---:|---:|---:|
| warmth markers | 0.186 | 0.143 | 0.77 |
| advice markers | 0.175 | 0.100 | 0.57 |
| negative markers | 0.047 | 0.141 | **3.00** |

Length is already handled by `tone_length_fit` (r +0.417, the second-strongest
predictor), which is why the *plan* marginal is nearly correct. The remaining
levers are the three surface rates above, and they are schedulable by exactly
the mechanism `sentence_rhythm` already implements -- per-slot draws at a
measured per-band rate, whose realized fidelity is within 0.008 of the
measurement over 3,000 draws.

One caveat to carry into the design: at its own warmth rate of 0.143 the
quintile table predicts a polite_rate near 0.23, and the generator realizes
0.066. So raising the three rates is necessary but the residual says the markers
are also being *used* differently -- "that's a nice idea but ..." is not the same
move as "that's a nice idea". Measure the realized rate after the change rather
than assuming the quintile curve transfers.

**Do not build this until the v98 N=10 result is in.** Two of the five v98 arms
already push warmth-adjacent surface (exclamation, final punctuation) and the
N=10 output is the only honest measurement of where the rates land under v98.

## 2026-08-20 — reading the pinned core, and a retraction

Instructed to read every related file before diagnosing further. Doing so on the
tone/warmth chain (`generation_distribution`, `planner_distribution`, the
`prompts` renderers, and the pinned core's `engine/writer_validation.py` and
`engine/vocabulary.py`) found one thing the artifacts alone could not, and
**retracted one of my own claims**.

### Retraction: "gives advice 0.090 -> 0.008 (0.08x)" was a probe artifact

That number came from a probe missing the forms the generator actually uses. Six
independent probes on the same 532/532 comments:

| probe | real | generated | ratio |
|---|---:|---:|---:|
| the original speech-act probe | 0.090 | 0.008 | 0.08 |
| the warmth-study probe | 0.175 | 0.100 | 0.57 |
| the core's own `generic_advice_frame` | 0.128 | 0.195 | **1.53** |
| second-person modals only | 0.103 | 0.045 | 0.44 |
| `I'd` | 0.051 | 0.160 | **3.15** |

Token by token, the generator gives advice constantly — just in a narrow set of
forms:

| token | real | generated |
|---|---:|---:|
| check | 0.019 | **0.194** |
| i'd | 0.051 | **0.160** |
| worth | 0.011 | 0.036 |
| you should | 0.011 | **0.000** |
| you could | 0.008 | **0.000** |
| consider | 0.021 | **0.000** |
| make sure | 0.006 | **0.000** |
| you can | 0.090 | 0.043 |

So the defect is **register, not absence**: the ordinary second-person advice
forms are gone and one form, "check", carries the load at ten times its real
rate. This is the same lesson as the unreliable domain-vocabulary probe of
2026-08-19, and it is why a headline number must be decomposed before it becomes
a design.

### The template-phrase cap is working, not misfiring

`engine/vocabulary.CAPPED_TEMPLATE_PHRASE_FAMILIES` caps nine opening frames at
`--template-phrase-reuse-budget 4` per thread, and `template_phrase_reused` is a
**blocking** guard in `has_blocking_guard_failure`. `generic_advice_frame`,
`first_person_experience_frame` and `gpt_good_to_know_frame` are all capped, so
this looked like a direct suppressor of the warm register.

It is not. Measured per thread, the *generated* side is the one that exceeds the
budget (advice frames 5, 5, 5, 5, 6, 26 against a budget of 4), driven by the
"check" over-use above. Real threads exceed it only on the 186-comment thread.
The cap is doing its job against a real over-production; raising it would make
the register worse.

Two real threads do exceed it though — `uncertainty_frame` 7 and 8 on 38- and
91-comment threads, and 12 on the 186-comment one — so a flat budget of 4 is
wrong at large thread sizes regardless. It should scale with comment count.

### Warmth decomposed

| token | real | generated | ratio |
|---|---:|---:|---:|
| great | 0.068 | 0.019 | 0.28 |
| love | 0.045 | 0.006 | 0.12 |
| thank you | 0.009 | 0.000 | - |
| awesome / glad / good luck | 0.006 each | 0.000 | - |
| exactly | 0.015 | 0.049 | 3.25 |
| nice | 0.024 | 0.036 | 1.46 |

Warmth really is under-produced, and it is concentrated: `great` and `love`
carry most of the real mass and are at 0.28x and 0.12x. The generator
substitutes "exactly" and "nice".

### Eye-visible tells, for the indistinguishability criterion

Ranked by absolute log rate ratio over tokens with at least 12 occurrences:

| token | real occurrences | generated |
|---|---:|---:|
| https / www / com / amazon | 37 / 25 / 31 / 23 | **0 / 0 / 0 / 0** |
| will | 109 | **1** |
| their | 74 | **0** |
| we | 26 | **0** |
| might | 27 | **0** |
| handling | 0 | 35 |
| usable | 0 | 27 |
| framing | 0 | 23 |
| settle | 0 | 22 |
| tells / sits / everyday | 0 each | 19 / 18 / 19 |

**No generated comment contains a link.** Real threads carry them in 7.6% of
comments. That is a difference a reader sees immediately and no metric in the
suite penalizes.

`settle` at 22 occurrences against 0 real is the adjudication frame still
leaking after v97's gate and v98's route ledger.

## v98 N=10 result — 2026-08-20

Tag `generalized_card_camera_gpt54_v98_rhythm_n10_20260820_v1`, seeds 2-11 (the
same seeds as v97, so the comparison is paired). 532 comments, coverage 1.00,
982 requests, `$3.7047`, 50.1 minutes, zero degraded comments.

```
[evaluation-results] PASS/PARTIAL/FAIL: 8/1/3
self_bleu_4                  PASS        MWU=0.12122 KS=0.41752 Cliff=0.42 W=0.0051401
self_bertscore_mean_f1       FAIL        MWU=0.00058284 KS=0.0002165 Cliff=0.92 W=0.021694
semantic_mean_cosine         PASS        MWU=0.62318 KS=0.78693 Cliff=-0.14 W=0.01944
hard_disagree_rate           PASS        MWU=0.28974 KS=0.41752 Cliff=0.29 W=0.050781
polite_rate                  FAIL        MWU=0.012578 KS=0.012341 Cliff=-0.67 W=0.21595
impolite_rate                FAIL        MWU=0.001008 KS=0.0002165 Cliff=0.88 W=0.27282
neutral_rate                 PARTIAL     MWU=0.020989 KS=0.052448 Cliff=-0.62 W=0.086035
length_cv                    PASS        MWU=0.47268 KS=0.78693 Cliff=0.2 W=0.070022
avg_depth                    PASS        MWU=0.96976 KS=1 Cliff=0.02 W=0.022154
structural_virality          PASS        MWU=0.96967 KS=1 Cliff=0.02 W=0.026822
mean_story_probability       PASS        MWU=0.67758 KS=0.99446 Cliff=-0.12 W=0.049507
emotion_entropy              PASS        MWU=0.57075 KS=0.41752 Cliff=-0.16 W=0.1865
```

8/1/3 against v97's 7/1/4.

| metric | v97 MWU | v98 MWU | v97 Cliff | v98 Cliff |
|---|---:|---:|---:|---:|
| length_cv | 0.021 FAIL | **0.473 PASS** | -0.62 | **+0.20** |
| emotion_entropy | 0.326 | **0.571** | -0.27 | **-0.16** |
| self_bleu_4 | 0.186 | 0.121 | 0.36 | 0.42 |
| self_bertscore | 0.0003 | 0.0006 | 0.96 | 0.92 |
| polite / impolite / neutral | unchanged | unchanged | unchanged | unchanged |

### Which arms worked

Measured on the 532 comments against their 532 matched real ones:

| habit | real | v97 | v98 |
|---|---:|---:|---:|
| semicolon | 0.024 | 0.109 | **0.023** |
| dash-joined clause | 0.075 | 0.299 | **0.071** |
| ellipsis | 0.079 | 0.017 | **0.081** |
| exclamation | 0.079 | 0.000 | **0.064** |
| no final punctuation | 0.192 | 0.041 | 0.246 |
| digit | 0.562 | 0.299 | 0.457 |
| parenthetical | 0.171 | 0.055 | 0.086 |

`sentence_rhythm` and `final_punctuation` delivered: four habits land on their
measured rate and three moved most of the way. `emotion_entropy` improving
alongside the exclamation mark going 0.000 -> 0.064 is the predicted mechanism.

`length_calibration` plus the crossover fix delivered the length tail:
realized/target by band went essay 0.738 -> **0.985**, very_long 0.873 ->
**0.987**, and `length_cv` 0.862 -> 0.981 against a real 0.959, with threads
below real going 9/10 -> 6/10. The `short` band overshot downward (1.071 ->
0.857) and is the remaining length defect.

`no_story_scope` **did nothing and should be reverted to `tense`**: past-tense
rate 0.289 -> 0.288, `will` 0.015 -> 0.019, vocabulary V/sqrt(N) 15.95 -> 16.20
against a real 21.02. It also introduced new repeated 4-grams (`. before that ,`
0->4, `i was wrong to` 0->4). It removed a genuine prompt contradiction, which is
worth keeping on correctness grounds, but it buys no metric.

### self_bleu_4: an exact ablation harness, and every candidate rejected

`SB.pairwise_self_bleu_for_order([SB.tokenize(t) for t in texts], 4)` over the
comments the evaluator keeps reproduces the reported number exactly (v97 MWU
0.18588 Cliff +0.36, v98 MWU 0.12122 Cliff +0.42). Using it:

| ablation on v98 | mean | MWU p | Cliff |
|---|---:|---:|---:|
| real | 0.0278 | | |
| as-is | 0.0330 | 0.121 | +0.42 |
| normalize every typographic apostrophe | 0.0333 | 0.121 | +0.42 |
| drop "check ..." openings | 0.0330 | 0.140 | +0.40 |
| drop the "that's the part" frame | 0.0335 | 0.121 | +0.42 |
| drop yeah / basically / actually / honestly | 0.0328 | 0.121 | +0.42 |
| drop the first sentence of every comment | 0.0380 | 0.017 | +0.64 |

**No phrase drives it.** The apostrophe result matters: v97 already spent that
gain, and normalizing further changes nothing.

Across 160 evaluation-excluded real threads it is a length metric first:

| predictor | r with self_bleu_4 |
|---|---:|
| share of comments <= 15 words | +0.783 |
| mean comment length | -0.723 |
| distinct entity types per comment | -0.587 |
| types / sqrt(tokens) | -0.386 |

Generated already matches on length (52.9 against 55.8 words; short share 0.216
against 0.205). Regressing self_bleu_4 on length and entity diversity over those
160 threads gives R2 0.527, and

    self_bleu_4 = 0.04964 - 0.000288*meanWords - 0.00127*entityTypesPerComment

with a **partial r of only -0.097** for entity diversity once length is
controlled. The model predicts a +0.0025 gap at the generated thread's position
against an observed +0.0052, so length and entities together explain about half
of it and entity diversity alone about a third. Entity diversity is a real but
modest lever here -- and it is *not* the `self_bertscore` lever (partial-free r
there was +0.182, the wrong sign).

### Standing conclusion

Four of the five v98 arms are worth keeping and one is not. The remaining four
problem metrics are `self_bertscore_mean_f1` and the politeness trio, and the
trio is one cause with a thread-level verified mechanism (warmth-marker rate,
r +0.727, monotone across quintiles). `self_bertscore` still has no verified
mechanism after four rejected hypotheses.
