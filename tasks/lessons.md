# Lessons

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
