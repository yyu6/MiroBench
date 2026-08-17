# Lessons

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
