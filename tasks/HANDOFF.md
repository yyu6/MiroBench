# Handoff — synthetic Reddit thread generation (generalized_card)

> **Start at [`../docs/ORIENTATION.md`](../docs/ORIENTATION.md), not here.**
> That file is the current spec: goal, judging standard, metric interpretation,
> method, and working discipline. **This file is an evidence archive**, ordered
> newest addendum first. Its older numbered sections carry stale state and at
> least one claim that was later retracted (see `ORIENTATION.md` §6 for the
> retractions). Where the two disagree, `ORIENTATION.md` wins.

## 2026-08-25 v109 / v110 addendum — two live arms, two nulls, and a causal-identification error

Full detail in
[`.claude/handoffs/2026-08-25-geo-v109-v110-two-live-arms-two-nulls-and-the-causal-instrument-problem.md`](../.claude/handoffs/2026-08-25-geo-v109-v110-two-live-arms-two-nulls-and-the-causal-instrument-problem.md);
rules in `docs/DECISIONS.md` **G24-G49**. $4.94 paid this session across two
gates. **Neither release is promoted.** Both arms fired at 100%, verified in the
saved prompts, so both nulls are mechanism failures rather than plumbing
failures.

**The target, quantified (G42).** `p ~ 0.5-0.6` needs **~90% gap closure at
N=150, ~75% at N=50, ~50-75% at N=10**, simulated over the 763 real camera
threads. This retires the whole 5-10% mechanism class that the previous six
releases came from, and it is the first number in this project that says what is
worth building at all.

**Every candidate was priced before anything was built.** Length composition
31-37% (`self_bleu_4`) / 14-26% (`self_bertscore`); absent links 8.8% / 10.8%;
markdown emphasis 3.6% / 3.8%. Killed: full Planner de-duplication <=2.4% /
<=1.8% (G45), entity variety <=9.4% and saturating (G40), "generated writes
fewer, longer sentences" (words per sentence 15.15 against 15.54, G44),
`no end punctuation` as a tell (generated has **more**, 54 against 34), and seven
further surface features where generated matches or exceeds real.

**v109 -- per-slot referent spread (G36-G41).** The naming-shape defect closed
almost exactly: mentions per distinct designator **4.286 -> 2.333** against a
matched real 2.432, distinct designators 21 -> 69 (real 118), pooled 3-/4-gram
precisions moved from above real to below it. And the same arm **causally
worsened both priority metrics**, established by a randomised within-run contrast
(the draw keys on Planner traversal order, so treatment is content-independent):
`self_bertscore` pair F1 0.5033 / 0.5101 / 0.5289 and cosine 0.1889 / 0.2144 /
0.2525 by treatment count, monotone. The untreated half of the thread sits at
real on both. Length explains about a third of it, shared names 16%, and the rest
is the cue's own prescribed speech act -- slots given the same rhetorical
instruction converge. It also raised P(story) per treated comment (0.1298 against
0.0884) while the thread-level guardrail passed, which is why a rate-drawn arm
must be read on its own subpopulation.

**v110 -- refit length transfer (G46-G49), and the error worth carrying
forward.** `length_calibration` inverts a fit of `realized ~ asked`. Refitting it
on the object that actually governs the current system gave R2 0.879 across four
runs and looked decisive. It was not: `asked` is a **deterministic function of**
`real_word_count`, which also drives the layout guidance, the beat count, the
surface skeleton and the token ceiling, so the regression has **no identifying
variation**. The N=10 gate broke that collinearity for the first time and
measured the true elasticity of realized length with respect to the asked
number: **-0.02 at 50-99 assigned words and 0.11 above 100**, against the 1.21
the mechanism assumed. Realized/assigned moved 0.8896 -> 0.8957 against a
predicted 0.97-1.02. **Every "ask for more words" mechanism is dead**, which
retrospectively explains why v96 and v97's prompt-wording work only moved the
250w+ ratio 0.61 -> 0.71. `self_bleu_4` did not move at all (gap +0.0049 ->
+0.0049, MWU 0.121 -> 0.1212, Cliff +0.42 both, 5/10 threads, Wilcoxon p=0.695).
`self_bertscore` moved 0.0188 -> 0.0148 nominally but on 7/10 threads at
Wilcoxon p=0.064, with the channel that was supposed to produce it demonstrably
inoperative -- so it is not claimed.

**Three corrections to earlier claims.** v108's `--semantic-coverage-nonrepeat`
measures **~0.0007** on seed 8 (G39), so the previous handoff's "best
single-thread self_bertscore result yet" was thread noise. My own G24 claim that
the v103->v108 comparison was "isolated" was wrong -- six arms differed (G25).
G34's generalisation that these metrics cannot see manner of speaking was
retracted (G35): it rested on between-real-thread slopes, and generated sits at
the 0.0 percentile of real big threads, so it was extrapolation.

**Two infrastructure facts found by reading the artifacts.** `--writer-retries`
defaults to **0**, so the entire Writer validation loop is inert -- 65 of 186
slots on the v109 gate failed their own validator and every one shipped
unretried, including 32 `template_phrase_reused`. And there are **three** Writer
prompt templates, not two; `_low_info_writer_prompt` carries no anchors or
equipment block, capping any rate-drawn offer mechanism at ~83% of slots.

---

## 2026-08-19 v97 keyboard-surface/measured-joints addendum

The v96 N=10 run is the evidence this version is built on, and it is the first
complete honest sample under the new content policy: coverage 1.00, 532 generated
against 532 matched real comments, `$3.71`, 49 minutes. **6 of 12 metrics pass.**
`semantic_mean_cosine` 0.970, `structural_virality` 0.909, `avg_depth` 0.850,
`emotion_entropy` 0.678, `mean_story_probability` 0.521, `length_cv` 0.076. The
failures are `self_bertscore_mean_f1` 0.0002 (|d| 1.00), `impolite_rate` 0.001,
`polite_rate` 0.006, `neutral_rate` 0.007, `self_bleu_4` 0.009,
`hard_disagree_rate` 0.014. v96's story and emotion arms worked; both now pass.

Four causes were measured in that artifact before any code changed.

Typography is a metric, not a cosmetic. Zero of 532 comments used an ASCII
apostrophe and 389 used a typographic one, and the self-BLEU tokenizer reads
`it's` as one token and `it’s` as three, so every generated contraction added a
thread-wide shared trigram. Replaying the v96 text through a per-speaker keyboard
draw calibrated on 11,817 excluded comments moves `self_bleu_4` MWU p from 0.009
to 0.273 and KS from 0.052 to 0.787, and lowers `self_bertscore` by about 0.008.

The adjudication frame was on 532 of 532 slots. Its output is in 18.4% of v96
comments and effectively absent from real text, and it is worst where it least
belongs: personal_datapoint 29.1%, reaction 19.0%, against question_followup
8.1%. v73, v74, and v75 each reworded that line; none withheld it. v97 renders it
only for correction, verdict, and advice turns and never for a story slot, which
withholds it from 68.0% of the v96 slots and 76% of the observed instances.

The tone marginal was right and the joint was inverted. Planned targets were
0.311 polite and 0.442 impolite against a real 0.308 and 0.404, but the plan put
impolite on 74% of 120-250 word slots and 100% of slots over 250 words, where
excluded real comments of that size are 72% polite. Realized comments over 120
words came out 87% impolite and 9% polite against a real 27% and 71%.
`tone_length_fit` now fits P(tone | size band), measured over 15,294 excluded
comments, by iterative proportional fitting so both margins stay exact.

Long slots asked for a shape that does not exist. 250w+ slots realized 0.61x
their matched length and the 845-word slot 0.32x. The token budget allowed 1,500
tokens; the request was the problem. That slot was asked for one thesis in 40
beats, and the Planner saturates near nine however many are asked. Real long
comments are 6 paragraphs at the median and 14 at p90, with words per paragraph
nearly flat inside a band. v96 had a blank line in 3.4% of comments against
33.8% real.

Four named arms, each recorded in `run_config.json` and each reproducing v96 at
its legacy value: `--reddit-typography off`, `--turn-frame universal`,
`--tone-length-fit median`, `--long-form-layout beats_only`. Domain profile
schema 11 -> 14.

The zero-API gate is complete: 369 tests, Ruff, 98/98 pins, both parity scopes,
self-test with all four arms, a schema-14 profile rebuild over 424 excluded
threads with 0 seed overlap, direct proof that the active shaper runs, and exact
seed-2 `--prepare-only`. Run one paid seed-2 gate next and check the six
predictions in `tasks/v97-worklog.md`. Do not start N=10 before that passes.

## 2026-08-18 v96 selective-facts/ancestor-novelty addendum

The paid v95 seed-2 gate completed all 45 comments in one attempt for `$0.3481`,
confirming that compiled non-terminal Planner contracts fixed progression. It
did not produce real-matched content. Exact n=1 diagnostics were self-BLEU
`0.0350` vs real `0.0268`, self-BERTScore `0.5306` vs `0.4892`, story
probability `0.1015` vs `0.2321`, emotion entropy `1.6572` vs `1.9687`, 20% vs
60% comments with digits, 15.6% vs 55.6% with domain vocabulary, and 5 vs 40
distinct model designators. N=1 p-values are not inferential. Length CV, average
depth, and structural virality matched.

The active trace found no safe factual path under `domain-claim=off`. The root
Planner saw excluded reference text but had to drop facts; the direct-reply
Planner saw no reference rows; and the Writer often received only `Sony` or
`VII`. Its `named` rule then incorrectly prohibited repeating any name another
comment used, even though real threads repeat the product name while changing
the fact. Deep replies excluded only their immediate parent, so S37--S45 kept
returning to the same fixed-lens boundary under different delta labels.

v96 adds a distinct `domain-claim=selective` policy. A deterministic schedule
uses only anonymous slot capacity and evaluation-excluded reference surface
roles; only scheduled slots can retain one Planner-restated general fact after
JSON parsing. `planned` and `off` remain reproducible arms. Selective direct
replies receive their own excluded reference window, and every deep reply sees
compact semantic coverage from its full ancestor chain. Raw reference wording
never reaches the Writer. Planned facts now enter Writer anchors; a personal
story gets either that claim or a rotating excluded-reference equipment
shortlist, never two independent factual sources.

The `named` rule now allows normal product-name reuse and blocks only repetition
of the same fact/amount. This is a content hypothesis, not a guaranteed metric
pass. The free gate is complete: 316 tests, full Ruff, 95/95 clean pins, both
parity scopes, selective/named self-test, and exact seed-2 prepare-only all
pass. Run one paid seed-2 gate next. Do not start N=10 unless it passes artifact,
qualitative, repetition, specificity, story/emotion, and exact n=1 review.

## 2026-08-18 v95 compiled/non-terminal-plan addendum

The paid v94 seed-2 gate failed all three whole-post attempts: 152 requests,
1,031,377 input tokens, 76,450 output tokens, 541 seconds, and `$0.9608`. It
produced no evaluable discussion. The attempts stopped on different content
contracts: S20/S22, then S43, then S18. Persisted audits show 130 of the 152
requests were Planner quality repairs.

This disproves the v94 reliability claim. The remaining problem was not one
field merge but ownership: held-out aggregate story/affect controls were
overlaid after semantic planning, then dependent fields were judged against the
new labels. S43 was a coherent gratitude close before the overlay; changing
only affect to neutral created the conflict that stopped the post.

v95 adds a deterministic, domain-neutral contract compiler before quality
evaluation. It preserves the Planner's semantic move and only reconciles the
dependent evidence/payload/function/role route required by fixed story,
social-close, micro, and substantive-capacity controls. Every reconciliation is
audited. The direct-reply Prompt now receives default `no_story` explicitly.

Soft plan diagnostics get at most one repair per slot. Missing development
beats use the existing Writer capacity fallback and cannot abort. Any residual
content-contract problem is logged as `unresolved_plan_contract_warning` and
continues; malformed/missing schema, API/safety/empty output, and exact coverage
remain hard. Replaying all 19 saved v94 batches leaves zero terminal contract
conflicts. See `tasks/v95-worklog.md`.

The v95 zero-API gate is complete: 307 tests, Ruff, 95/95 source pins, both
parity scopes, backend self-test, and the exact seed-2 `--prepare-only` pass.
Even after runtime completion is observed, do not claim 12-metric success
without running the unchanged scorers: pipeline progression is an engineering
property, but MWU/KS p-values cannot be honestly guaranteed before observing
generated text.

## 2026-08-18 v94 state-preserving-repair addendum

The paid v93 N=10 run completed seeds 0 and 1 and stopped on seed 2 S9 after
100 requests, 305 seconds, and `$0.3992`. The v93 topology fix was active and
worked; the new failure was a different bug in repair state management.

Persisted attempts show an alternating sequence. Repair 1 supplied five
development beats but retained `no_story + firsthand_experience`; repair 2
changed the evidence to `small_observation` but erased the beats; repair 3
restored the beats but changed evidence back to `firsthand_experience`. The
backend replaced the whole plan after every accepted candidate, so fixing one
field could undo a prior fix in another field.

v94 uses field-scoped merge only when a slot has exactly one remaining repair
issue with a known field boundary. For `long_form_capacity`, only
`development_plan` is applied. Multiple blocking issues still use whole-plan
replacement. Audit rows now distinguish the raw model candidate from the
applied candidate and list `repair_merge_fields`. Exact v93-log replay changes
S9 blocking `1 -> 0`, preserves five beats, preserves
`evidence_mode=small_observation`, and leaves no S9 issue.

Do not resume or formally evaluate the partial v93 or failed v94 artifacts. The
v94 seed-2 gate exhausted three attempts and is superseded by v95. See
`tasks/v94-worklog.md` and `tasks/v95-worklog.md`.

## 2026-08-18 v93 root/reply-boundary addendum

The paid v92 N=10 run completed seeds 0 and 1, then stopped on seed 2 root S9
with `long_form_capacity`. The saved attempt payloads prove that long-form
planning was not the remaining failure: the first S9 repair supplied all five
required beats. It was rejected only because the root row also emitted
`reply_delta_type=social_close`, creating an unrelated blocking social contract.

v93 clears all reply-only fields from structurally root plans before quality
selection and records any nonempty override. Direct replies are unchanged. The
root schema now requires literal `none`; its duplicate direct-reply definitions,
contrast prose, and obsolete parent-contract renderer were deleted. Replaying
the actual v92 candidate moves repair rank `(1,46) -> (0,41)`, leaves zero
blocking issues, and preserves the five beats.

Do not resume the v92 tag for formal evaluation: seeds 0 and 1 already use its
older root Prompt. Start a fresh v93 N=10 tag. Offline acceptance is complete:
300 generalized tests, 3 scorer tests, Ruff, both parity scopes, 93/93 clean
pins, actual-candidate replay, backend self-test, and exact N=10 prepare-only all
pass. See `tasks/v93-worklog.md`.

## 2026-08-17 v92 lossless-domain-claim addendum

The v91 completion audit found another active Planner→Writer gap. With
`--domain-claim off`, both Planner paths still assigned a separate domain fact,
but `backend.py` deliberately did not register it for the Writer. This wasted
Prompt/output tokens and allowed the visible semantic fields to depend on an
invisible claim.

v92 makes off-mode symmetric. Root and direct reply Planners set
`domain_claim=none`, omit the claim-specific knowledge/rule blocks, and put the
whole contribution in `semantic_move`, `detail_focus`, and `domain_intent`.
Normalization forcibly clears a returned claim in case the model ignores the
schema. Planned mode remains unchanged and reproducible.

v92's offline acceptance was complete: 299 generalized tests, 3 focused scorer
tests, Ruff, both parity scopes, 93/93 clean pins, rendered root/direct Prompt
checks, named/off backend self-test, and exact seed-8 prepare-only. Its paid
N=10 attempt later completed two threads and exposed the v93 root/reply boundary
bug; do not resume it for formal evaluation. See `tasks/v92-worklog.md` and the
v93 addendum above.

## 2026-08-17 v91 slot-gated concreteness addendum

The zero-API audit found that `--own-fact-license named` could not safely be
turned on as implemented. The per-slot resolver correctly limited the license
to substantive comments, but the system Prompt unconditionally told every
Writer call to name particulars and give amounts. Micro/short comments therefore
received a global pressure to add facts while their user Prompt prohibited
non-visible names and numbers.

v91 reduces the system addition to one conditional authorization: only an
explicitly licensed per-comment Prompt may override the generic visibility ban.
The behavioral instruction appears once, at the substantive slot. The same
boundary now protects the retained legacy `own` arm.

This gate is data-scaled rather than arbitrary. It selects 110/186 seed-8 slots
(59.14%); matched real comments contain a digit at 59.68%, versus 31.35% in v80
generated text. Matched real also has 118 distinct model designators versus 29
generated. At that stage the next run would have used policy
`generalized-card-v2-slot-gated-fact-license-v91-20260817`, a fresh v91 tag,
`--own-fact-license named`, and `--domain-claim off`. See
`tasks/v91-worklog.md`. v91 was superseded by v92 before an API call. Its offline
acceptance was complete: 297 generalized tests,
3 focused scorer tests, Ruff, both parity scopes, 93/93 pins, named backend
self-test, exact 186-slot Prompt replay, and named-mode prepare-only all pass.

## 2026-08-17 v90 reply-story-grounding addendum

The zero-API completion audit found one residual v89 Prompt conflict. The root
Planner explicitly allowed the ordinary synthetic personal sequence required by
a scheduled story, but the specialized direct-reply Planner did not. It required
an actual event sequence while only stating the prohibitions on source
participant details and invented seed facts.

v90 moves the conservative synthetic-story boundary into one shared Planner
definition and renders it on both root and direct-reply paths. It allows a
non-verifiable first-person sequence around the visible or generic local point;
it still forbids invented product facts and externally checkable outcomes. The
Writer already enforced the same distinction, so this closes Planner→Writer
contract drift without broadening factual permission.

Do not spend under v89 or v90. v90 was superseded before a paid call when the
completion audit found the global/per-slot conflict in the pending concreteness
arm.
Offline v90 acceptance was complete: 295 generalized tests, 3 focused scorer
tests, Ruff, backend self-test, both parity scopes, 93/93 clean pins, and exact
seed-8 `--prepare-only` all pass. See `tasks/v90-worklog.md`.

## 2026-08-17 v89 Planner-repair addendum

The paid v88 seed-8 command did not reach Writer generation. It stopped after
24 Planner requests, 116 seconds, and `$0.1805`; therefore it produced no
discussion that can be evaluated. The offset-8 batch retained blocking plans
for S10, S13, and S15 after bounded repair.

This was not solved by raising retries. Current-source and exact-log review
found that scalar repair scoring preferred a Writer-incompatible story plan
over a coherent plan with one semantic collision. The last S15 candidate in
the v88 audit reduced the batch's blocking conflicts but was rejected because
its total score was 56 versus the selected 54. v89 ranks blocking contract
count before quality score. Polite role mismatch is now low-weight Planner
feedback rather than a hard semantic contract, because the actual scorer is a
four-way classifier over realized comment text and politeness is the user's
lowest-priority metric. Story/affect/capacity coherence remains hard.

The root Planner Prompt also no longer simultaneously requires a scheduled
personal story and bans the synthetic personal sequence needed to plan it. Its
boundary now matches the Writer's existing conservative story rule: an ordinary
non-verifiable sequence is allowed, externally checkable product facts are not.

v89 remains reproducible and its offline acceptance is complete: 294
generalized tests, 3 scorer tests, Ruff, backend self-test, both parity scopes,
93/93 clean pins, and exact seed-8 `--prepare-only` all passed. It was
superseded by v90 before any v89 API call because the same story boundary had
not yet reached the direct-reply Planner.

## 2026-08-17 v88 structural-speaker/grounding addendum

v87 is reproducible but was superseded before any paid run. At that stage v88
became the required policy with `--speaker-identity matched`; the v88 paid
attempt later failed before Writer generation and is historical now.

The completion audit found that the formal v87 command still preserved a known
Prompt contradiction for historical ablation: under `own-fact-license off`, 78
of 186 replayed Prompts displayed an invented equipment permission, 144 banned
personal experiences, and 61 contained both. v88 renders the equipment block
only under the explicit `own` license; replay is now 0 equipment blocks and 0
permission/revocation conflicts. The ordinary off-mode fact rule remains.

The existing matched-speaker arm also was not a clean structural control. It
grouped repeated source authors correctly, but then attached rotating tenure,
use case, and kit claims. v88 deletes those biography fields and the associated
kit helper. A speaker now contains only anonymous ID, OP membership, owned slot
IDs, and anonymous-account status. The Prompt may show the same anonymous
speaker's earlier generated turns, but no source username or invented identity
claim. Its continuity instruction explicitly yields to each turn's assigned
voice and affect.

On current seed 8, 186 matched slots become 97 generated speaker groups: 80
named-source groups plus 17 deleted/anonymous one-shots. Named groups average
2.112 turns, recurring groups cover 66.7% of comments, and the maximum is 10
turns. An active-wrapper integration test verifies the grouping, not merely the
roster helper. See `tasks/v88-worklog.md`.

This fixes current structural fidelity and Prompt conflict; it does not prove
metric movement. The paid v88 attempt failed in Planner and was superseded by
v89 before any Writer content existed.

## 2026-08-17 v87 route-contract addendum

v87 is retained for provenance and superseded by v88 before any paid run.

A zero-API replay rendered all 186 frozen v80 Writer tasks in their original
long-thread order. It found that utterance shape was checked before payload:
six `soft_helpful` plans and one `correction` were sent to the low-information
Prompt, whose hard rules prohibit exactly the advice, explanation, and caveat
those payloads need. v87 requires an explicitly low-information payload before
any short-shape routing rule can apply. The same replay now selects 25 low-info
tasks, all in the allowed payload set and all `no_story`.

The replay also exposed a brittle Prompt implementation: focused paths built the
full blackboard, then parsed two sections from its rendered string, with
effectively unbounded short-line history. v87 builds only two bounded ledgers
from source records, removes openings already displayed next to them, and does
not tell a social-close Writer both to perform and avoid the same semantic move.
Exact-duplicate output validation still retains the full generated history.
Low-info Prompt mean/max fell from roughly 7,581/9,397 to 6,579/7,921
characters; there were no repeated long Prompt lines or repeated required
semantic moves after the change.

Finally, Writer-dependent tone controls are recomputed after the final Planner
contract, and gratitude/social-close coherence is validated in both directions.
This prevents stale `pure_acknowledgement` instructions and blocks a neutral or
meta `gratitude_reply` before Writer generation. See `tasks/v87-worklog.md`.

All evidence so far is offline contract evidence, not output-quality evidence.
The paid seed-8 run must still be reviewed at 186/186 coverage for natural
Reddit voice, repetition, generic helpfulness, affect/profanity, story
realization, and descriptive 12-metric distance. Only sufficient N may support
MWU/KS conclusions.

## 2026-08-17 v86 Prompt-boundary addendum

This policy is retained for provenance and is superseded by v87 before any paid
run. No v85 or v86 paid artifact exists, so the policy bump does not mix an
existing run.

The final zero-API Prompt audit found two active issues. First, root comments
could show `reply relation: answers_parent` even though no parent was visible;
the Writer boundary now renders `relation to post: answers_post` while leaving
the Planner record untouched. Second, the low-information path repeated its
move and controls through a private-slot block, semantic contract, local-move
line, full blackboard, payload guidance, and per-slot guidance. It now uses the
compact focused discourse contract and bounded repetition ledger once, while
retaining the low-information and factual hard rules.

Reviser-only Prompt helpers were split into `legacy_reviser_prompts.py`, cutting
214 lines from active `prompts.py`. AST comparison showed zero changes to the
retained active functions and identical migrated legacy functions. See
`tasks/v86-worklog.md`. Formal metric success remains unproven until v86 n=1
content review and then sufficient-N matched evaluation.

## 2026-08-17 target/realization and evaluation-integrity addendum

The free target audit changes the diagnosis. The evaluation-excluded reference
template sampler passes both MWU and KS for all 12 metrics at N=10 and N=150.
Low per-thread correlation is expected because this is deliberately an
unpaired distribution draw, not matched-test fitting. Do not change the sampler
because one n=1 template differs from its matched real thread.

Every new post now persists the exact selected aggregate template in
`thread_plan.reference_metric_template`. The content report shows real,
Planner target, generated output, target−real, and generated−target for all 12
metrics, with separate stage statistics. On old v80 seed 8 this attributes the
large gaps correctly: politeness and story are Writer-realization failures;
emotion entropy nearly equals its selected target even though that one target
draw is low relative to matched real. Legacy sequence-only logs are accepted
only if the mapping is complete and unique; resume ambiguity is a hard error.

`run_evaluate.py` no longer runs deterministic cleanup before scoring. After
the integrity audit it creates a byte-identical snapshot and fails on
noncanonical metadata instead of repairing it. Formal stats and scorer CLIs are
now tracked/pinned current sources rather than transitive untracked or dirty
calibration dependencies. Default parity covers active generation/evaluation
only; revisers are legacy and not part of the current research workflow.
Dynamic runtime edges are explicit too: backend/audit runners and token
tracking/summarization are among the 67 active tracked sources (92 declared
pins total), and the closure audit follows sibling scripts.

Finally, n=1 is `DESCRIPTIVE` from the matched evaluator through console and
content reports. Ignore the mathematical p=1 values returned for singleton
samples; they cannot establish a pass. The pending paid step is a fresh v93
N=10 generation and evaluation with exact 10/10 structural coverage.

## 2026-08-17 exact-matched content-audit addendum

`run_evaluate.py` now writes `content_profile_audit.json` and `.md` after the
formal matched evaluation. This fixes a load-bearing bug in the old
`compare_content_profile.py`: lexical comments were matched to the seed, but
the real emotion/story target pooled the whole domain. Never quote those old
full-domain rows as a matched n=1 target.

The new report uses exact seed ID plus product directory for real scorer rows,
shows all 12 paired distances, joins persisted Planner controls to generated
per-comment classifier outputs, and exposes repetition phrases and planned
helpful/advice mass. Regex surface probes are explicitly weak diagnostics, not
semantic classifiers. At n=1 every metric status is `descriptive_only_n1`; only
the later multi-thread run may interpret MWU/KS. See `tasks/v85-worklog.md`.

## 2026-08-17 v85 current-path audit addendum

Policy `generalized-card-v2-auditable-plan-controls-v85-20260817` is preserved
for provenance but superseded by v86 before any paid run. This pass used
historical commits only as provenance; every decision
was re-established from current definitions, current call sites, the current
v80 artifact, and current tests.

The important new finding is not another Prompt heuristic. Fixed template
story/tone/affect/opener values are applied before plan-quality evaluation, so
the existing social-contract validator does catch semantic plans that become
incoherent after assignment. What was missing was evidence of whether the
Planner obeyed those values before enforcement. `planning_quality.jsonl` now
records `initial_slot_contract_overrides` plus all repair-time overrides.

The same path audit found three false repair/validation surfaces. Structural
branch metadata rewrites every slot's `perspective_id` and `branch_id` before
evaluation, so invalid-perspective and branch-route checks could never fail;
perspective concentration could be observed but never changed by a slot-local
Planner retry. The first two are deleted; concentration remains a warning but no
longer spends repair calls. Retired tone overlays and the old
`constructive_polite_helpful` label are removed from current Writer/finalizer
logic while compatibility fields remain for artifact deserialization. See
`tasks/v85-worklog.md`.

This is primarily simplification, cost correctness, and diagnostic validity.
It does not establish that the 12 metrics improved; that requires the new n=1
content/contract run and then a multi-thread matched evaluation.

## 2026-08-17 v83 matched-text isolation addendum

The v82 audit was extended through every callback that receives the anonymous
matched comment body. It found that first-person/uncertainty regexes and
`story_rant`/`side_tangent` real-surface labels still derived semantic controls
from evaluation wording. v83 removes those paths. Matched data now contributes
only tree linkage, speaker structure when explicitly enabled, word scale,
punctuation, dominant link/quote form, capitalization/identifier form, and other
non-semantic surface shape. Planner story, stance, voice, function, payload,
affect, and tone are the sole semantic authority.

v83 is historical; use the current policy named above. See
`tasks/v83-worklog.md` for that release's evidence.

## 2026-08-17 v82 completion-audit addendum

The v81 post-implementation field-survival audit found one remaining default-path
defect before a paid v81 run: focused Writer prompts did not carry the Planner's
discourse role. They had the proposition and tone/story/affect controls, but
silently omitted function, payload, speaker role, voice, evidence, content angle,
stance, detail, decision intent, reply relation, and the explicit local
exclusion. This can turn a planned rant, correction, datapoint, question, or
bare reaction into the model's generic helpful/customer-service answer.

v82 adds one compact, deduplicated discourse contract rather than restoring the
old full prompt. An end-to-end test now starts at raw Planner JSON and proves the
fields survive normalization, matched-slot expansion, finalization, and the
default focused Writer path. The same audit removed a second leak: matched-real
words such as `thanks` and `appreciate` could assign gratitude tone through the
shared surface classifier. Only typography remains matched-text-derived; social
meaning is Planner-owned. At that stage v82 superseded v81. Detailed
evidence is in `tasks/v82-worklog.md`; formal metric success remains unmeasured.

## 2026-08-17 v81 current-state addendum

This addendum is authoritative for the current implementation. Historical
implementations and old TODO hypotheses are **not design evidence**: use the
current active path, scorer implementations, current run artifacts, and fresh
offline checks. Git commits and each run's reproducibility/source snapshot are
the mechanism for reproducing an old version; do not keep dead runtime branches
or contradictory prompt rules merely to preserve an old arm.

v81 fixes the root defects found in the v80 large-thread run:

- story/no-story is now a joint semantic/evidence contract before the Writer;
- direct replies plan with fixed story, tone, affect, and opener controls in
  their JSON contract instead of receiving those labels after semantic design;
- copied short-slot `development_plan` schema prose is removed;
- deterministic normalization no longer rewrites bad plans into the same
  gratitude move or `soft_helpful` payload;
- tone, affect, and story marginals are assigned jointly for feasibility;
- focused, low-info, and full Writer prompts no longer repeat static metric or
  tone blocks, and neutral affect no longer conflicts with impolite tone;
- metric-guided Writer retries, candidate ranking, blocking repetition guard,
  dead CLI flags, and their old tests were deleted. Distribution metrics are
  diagnostics; only non-persistable output has bounded recovery.

Verification: complete `generalized_card/tests` suite **259 passed**; backend
self-test passed for `camera_product`; 72 source pins have 0 missing and 0
drifted entries. Exact metric definitions, v80 evidence, and the implementation
audit are in `tasks/v81-worklog.md`. Formal success still requires a new
multi-thread matched evaluation; n=1 is only the content/contract diagnostic.

Written 2026-08-16. **This supersedes the 2026-08-15 handoff**, which is preserved
at `tasks/HANDOFF-20260815.md`. Several of its load-bearing claims were measured
this session and turned out to be wrong; §6 lists every correction.

Read in this order:

1. this file, end to end
2. `tasks/todo.md` — the re-prioritised plan
3. `tasks/lessons.md` — 8 recorded mistakes, 3 of them from this session
4. `tasks/generator_audit.md` — the older evidence base, with §6 here as its errata

Every number here is measured from run artifacts. Where a claim is uncertain, or
where the measurement cannot separate a cause, it says so.

---

# 1. THE GOAL

Generate synthetic Reddit threads that are statistically indistinguishable from
real ones across **12 thread-level metrics**, using `generalized_card/`, a
domain-configured Planner–Writer implementation of CARD.

## What "real" means here — the user's own framing, authoritative

> 我们要模仿的是说话方式，而不是真正的 content …… 只要能做好就行，因为我这个指标
> 其实就是衡量人们是怎么样说话、怎么样讨论的。

Not factual accuracy. **How people talk and discuss.** The user decomposed this
into four dimensions, and they map onto the metrics:

| user's dimension | metrics | state |
|---|---|---|
| 1. semantic is dispersed | `semantic_mean_cosine` | passes at n=10, but by cancellation (§4.3) |
| 2. low lexical overlap | `self_bleu_4`, `self_bertscore_mean_f1` | bertscore has never passed in any version |
| 3. stories told in first person | `mean_story_probability` | passes at n=10 by cancellation; per-thread 1.5–2.4× too high |
| 4. tone and emotion are varied | `emotion_entropy`, `polite_rate`, `impolite_rate`, `neutral_rate`, `hard_disagree_rate` | all fail, all in the same direction |

**Dimension 4 is the largest and most coherent failure.** See §5.

## The judging standard

- A metric is matched only if **MWU p > 0.05 AND KS p > 0.05, and the p-value is
  comfortably large.** Barely above 0.05 does not count.
- The user rejected N-based extrapolation: 不希望用根据 N 的大小来测试的方法，除非
  publicly scientifically proved. Do not argue "this would pass at N=150".
- Final target: **150 threads per domain.**
- **Print findings in chat, not only in MD files.**
- The user runs the paid commands: 修改完之后，我来负责测试。

## Standing constraints

- **Domain-generalised, never domain-specific.** This bit me this session: a rule
  I shipped said "your own gear … what you shot or set it to", which is camera
  vocabulary. Every test ran on the camera domain so nothing caught it.
  `test_the_named_rule_carries_no_domain_vocabulary` now asserts against it.
- **Every behaviour change gets an ablation flag**, `off` reproduces the previous
  release byte-for-byte, and the flag is recorded in `run_config.json`.
- **Style exemplars must be modified real comments**, never verbatim. Not built.
- Politeness *as a topic* is de-prioritised, but `polite_rate` / `impolite_rate` /
  `emotion_entropy` are dimension 4 and are the biggest gap. The distinction that
  matters: the work is **"make the assigned register actually appear"**, not
  "make the comments polite".

---

# 2. WHERE THE CODE IS

Branch `generator/v75-writer-realizes-planner-move`. The v80 work starts from
`4633af7`; use `git log` for the resulting implementation commit.

```
generalized_card/generalized_card/
  backend.py              2.6k lines  adapter and current Planner/Writer lifecycle
  prompts.py              2.7k lines  root/reply Planner + focused/full/low-info Writer
  writer_quality.py        270 lines  diagnostics + hard-output recovery only
  writer_grounding.py      323 lines  fact/grounding rules in one place
  speaker_roster.py        230 lines  NEW this session: who is speaking, across their turns
  generation_distribution.py 576     TONE_DEFINITIONS, AFFECT_INSTRUCTIONS, social targets
  task_distribution.py     260 lines  which task fields survive the surface rebalancer
  core_contract.py         520 lines  72 pinned file hashes + policy versions
scripts/sampling_generator/
  run_sampled_reddit_generator.py 2.1k  the CARD facade the adapter patches
  engine/*.py                            12 modules, all pinned
generalized_card/scripts/
  run_generate.py         1.4k lines  CLI, config record, subprocess env
  run_evaluate.py                     clean → score → matched-evaluate
  repin_core_contract.py   NEW this session: walks the whole CORE_FILES table
```

**All of `generalized_card/generalized_card/*.py`, `generalized_card/scripts/
run_generate.py` and every `scripts/sampling_generator/**` file is hash-pinned.**
A change anywhere means re-pinning. Use the new script; do not hand-edit hashes:

```bash
python3 generalized_card/scripts/repin_core_contract.py          # report drift
python3 generalized_card/scripts/repin_core_contract.py --write  # re-pin
```

It exists because hand re-pinning means listing the files you *remember*
changing. That is the same shape as the config-diff script that silently skipped
`plan_quality.repair_rounds` and cost a confounded run.

---

# 3. HOW EACH METRIC IS ACTUALLY MEASURED

The user's instruction: 对于每一个 metrics，你都必须清楚它们到底是怎么衡量的才可以.
This section exists because a previous claim ("the story allocation is correct,
do not change it") was wrong precisely from not reading the scorer.

| metric | scorer | what it computes |
|---|---|---|
| `self_bleu_4` | `score_thread_self_bleu.py` | pure n-gram, **no model**. Runs in seconds offline. |
| `self_bertscore_mean_f1` | `score_thread_self_bertscore.py` | BERTScore F1 between comment pairs in a thread |
| `semantic_mean_cosine` | `score_thread_semantic_uniformity.py` | mean pairwise cosine of sentence embeddings |
| `hard_disagree_rate` | `score_thread_disagreement.py` | share of parent→child pairs classed as hard disagreement |
| `polite_rate`, `impolite_rate`, `neutral_rate` | `score_thread_politeness.py` | **Intel/polite-guard, 4-way single-label.** `pred_label = argmax` over {polite, somewhat_polite, neutral, impolite}; each rate is the share of comments with that label. `somewhat_polite` is a real class that absorbs mass but is **not reported** as a metric. |
| `length_cv` | `score_thread_structure.py` | per-thread word-count coefficient of variation |
| `avg_depth`, `structural_virality` | `score_thread_structure.py` | tree shape only — determined by the matched sampler, not by generation |
| `mean_story_probability` | `score_thread_storyseeker.py` | **mariaantoniak/storyseeker**, RoBERTa. P(story) per comment, then the **mean over every comment in the thread**. LABEL_1 = story. Not the same as `story_rate`, which thresholds at 0.5. |
| `emotion_entropy` | `score_thread_go_emotions.py` | **SamLowe/roberta-base-go_emotions**, 28 labels, sigmoid, threshold 0.5. Each comment's `dominant_emotion = argmax`. The metric is the **Shannon entropy of the histogram of dominant emotions** across the thread. To raise it you need more distinct dominant emotions, spread more evenly. |

Consequences worth holding onto:

- **`self_bleu_4` is free to compute.** Never approximate it. This session I wrote
  my own approximation, it disagreed with the real scorer by an order of
  magnitude in effect size, and I reported a win that did not exist (§6.4).
- **`emotion_entropy` is about the *variety of argmax labels*, not about
  intensity.** v79 has 13 distinct dominant emotions but `neutral` takes 48.4%
  and `approval` 20.1%. Flattening that histogram is the lever.
- **`mean_story_probability` averages over *all* comments**, so it moves when
  non-story comments start sounding narrative, not only when story slots change.

---

# 4. CURRENT STATE

## 4.1 Runs

| tag | flags that differ | comments | cost |
|---|---|---|---|
| `…v75_ownwords_20260815_v2` | 10 threads, **`plan_quality.repair_rounds=3`** (unintended) | 522 | $5.99 |
| `…v76a_baseline_seed8_20260815_v1` | seed 8 only, repairs=0, all new flags off | 186 | $0.76 |
| `…v76b_ownfacts_seed8_20260815_v1` | `--own-fact-license own` | 186 | $0.77 |
| `…v77_repguard_seed8_20260816_v1` | `--repetition-guard blocking`, retry-limit 1 | **172** | $0.86 |
| `…v78_frameguard_seed8_20260816_v1` | + whole-comment frame check, retry-limit 2 | **182** | $0.88 |
| `…v79_nodrop_seed8_20260816_v1` | + style-residual retention | **186** | $0.90 |

All five seed-8 runs have `repair_rounds=0`, so **the v75 confound is resolved**
for seed 8. It is still unresolved at 10 threads; that only matters if you want a
clean 10-thread baseline.

Seed 8 is the largest thread in the pool (185 real comments). The pool's seed
range is `[start-seed-index, start-seed-index + max-posts)`, so seed 8 alone is
`--start-seed-index 8 --max-posts 1`.

## 4.2 The five seed-8 runs against real seed 8

Real values come from `…v75…/matched_evaluation/matched_real_thread_scores.csv`.

```
metric                     REAL     v76a     v76b      v77      v78      v79
self_bleu_4              0.0283   0.0377   0.0354   0.0389   0.0354   0.0375
self_bertscore_mean_f1   0.4887   0.5241   0.5192   0.5171   0.5181   0.5227
semantic_mean_cosine     0.1865   0.2110   0.2183   0.1954   0.2009   0.2267
hard_disagree_rate       0.1697   0.3279   0.2967   0.2982   0.2809   0.3516
polite_rate              0.2324   0.0870   0.0710   0.0698   0.0889   0.0543
impolite_rate            0.4649   0.6739   0.7158   0.7151   0.6556   0.6957
neutral_rate             0.1622   0.0978   0.0820   0.1163   0.1389   0.0978
length_cv                0.8951   0.8639   0.8304   0.9147   0.8746   0.8593
avg_depth                3.6000   3.5978   3.5683   3.4128   3.5111   3.5978
structural_virality      4.5608   4.5663   4.5153   4.3326   4.4887   4.5663
mean_story_probability   0.1114   0.1944   0.2610   0.1935   0.1633   0.2152
emotion_entropy          1.9459   1.5443   1.4402   1.4986   1.4591   1.6572
```

**These five runs cannot be ranked.** Any two runs differ in ~99% of planner
fields (branch_goal 100%, semantic_move 99%), and there is **no
same-config-twice run to estimate that noise**. Mean relative error is 25.0%
(v78) to 38.8% (v76b) and that spread may be entirely noise.

What *is* readable is the **sign, which is identical in all five runs**:

- `polite_rate` 3–4× too low
- `impolite_rate` ~1.5× too high
- `hard_disagree_rate` ~1.8× too high
- `emotion_entropy` too low, i.e. too concentrated
- `mean_story_probability` 1.5–2.4× too high
- `self_bertscore` +0.03, every run, every thread

`length_cv`, `avg_depth` and `structural_virality` are within ~5% everywhere.

## 4.3 The most important structural finding: passing ≠ matched

MWU and KS are **unpaired** tests over 10 thread values. A metric can pass while
every individual thread is wrong, provided the errors cancel. Measured on v75:

```
metric                    v75    mean |gen-real|/real   threads within 20%   reading
avg_depth                PASS            0.5%                10/10           matched per thread
structural_virality      PASS            0.8%                10/10           matched per thread
semantic_mean_cosine     PASS           19.4%                 5/10           PASSES BY CANCELLATION
mean_story_probability   PASS           46.4%                 3/10           PASSES BY CANCELLATION
self_bertscore           FAIL            6.9%                10/10           close per thread, fails on consistent sign
length_cv                PART           11.7%                 8/10           close per thread
hard_disagree_rate       PART          188.9%                 0/10           wrong per thread
impolite_rate            FAIL           70.4%                 0/10           wrong per thread
polite_rate              FAIL           65.2%                 1/10           wrong per thread
neutral_rate             PART           63.4%                 0/10           wrong per thread
emotion_entropy          FAIL           39.2%                 3/10           wrong per thread
self_bleu_4              FAIL           34.9%                 2/10           wrong per thread
```

Two conclusions:

1. **Of the four metrics that "pass", only two are real**, and both are
   sampler-determined tree shape. **Zero metrics are currently won by generation
   quality.** Any claim of the form "we pass 4 of 12" should be read this way.
2. **`self_bertscore` is not a large error — it is a small, perfectly consistent
   one.** 6.9% mean error, 10/10 threads inside ±20%, and it fails only because
   all ten overshoot by ~+0.03. That is the signature of **one global constant
   offset**, not of content. This reframes the metric that has never passed.

---

# 5. THE DIAGNOSIS THAT SHOULD DRIVE THE NEXT CHANGE

## 5.1 Dimension 4 is one root cause, not five failures

Measured on v79, joining each slot's assigned `tone_target` to polite-guard's
`pred_label` (184 aligned slots):

```
assigned          n    realized impolite    realized as assigned
impolite         90         93%                    93%
polite           46         59%                    13%
neutral          35         34%                    34%
somewhat_polite  13         38%                    54%
                                overall realization 59.2%
```

The Writer can produce `impolite` and essentially nothing else. Assigned-polite
slots collapse into impolite 59% of the time. **One register per thread explains
polite ↓, impolite ↑, hard_disagree ↑, and emotion_entropy ↓ simultaneously.**

## 5.2 Two candidate causes measured and eliminated this session

**Not length.** `TONE_SCOPE_HINTS` at `generation_distribution.py:508-518`
records that polite-guard's polite class is length-driven in real data (52% of
60–120 word comments, 64% above 120). It does not transfer:

```
                 60–120 words     120+ words
real polite          52%             64%
v79 polite            6%              0%
```

Generated long comments are 73–88% impolite. **Giving polite slots more length
will not work**; that item can be struck from the plan.

**Not insufficient agreement.** Real comments carry *more* negation than
generated (41.5% vs 31.2% on seed 8) and are still scored polite. What differs:

```
surface                       REAL     v79    ratio
warm / appreciation marker   14.0%   11.8%    0.84
emotional endpoint            2.5%    1.1%    0.43
hedge                        18.0%   12.9%    0.72
decision-framing noun         0.5%    4.3%    8.60
```

`TONE_DEFINITIONS["polite"]` (`generation_distribution.py:480-489`) currently
**forbids two of the three surfaces the real data uses**:

> "Do not hedge the positive judgement into a maybe, and do not
> use customer-service phrasing or a template thank-you."

and the 8.6× on decision-framing nouns says the Writer substitutes analysis for
feeling. The comment block above `TONE_DEFINITIONS` documents the reasoning for
the hedge ban: a softener reading was predicted to collapse into
`somewhat_polite`. The measured collapse is into **impolite**. The prediction was
wrong, so the rule should go.

## 5.3 `self_bertscore`: one global signature, and a concrete candidate

The offset is uniform (§4.3), so look for something identical across every
generated comment. Typography is the strongest candidate available:

```
of comments containing any apostrophe:
   real   17.6% use only curly ’
   v78   100.0% use only curly ’
overall curly-typography rate:  real 11–13%   generated 72–74%
straight apostrophe inside a word: real 51%   generated 0%
```

**Every generated comment carries the same typographic fingerprint.** This is
model-emitted, not from `gpt_cleanup` (verified in an earlier session: identical
pre/post). A deterministic post-step fixes it with zero API cost and can be
verified offline over the whole corpus.

This item currently sits in P6, rated lowest priority. **That ranking is wrong**
— it is the only mechanism on the board whose shape matches how `self_bertscore`
actually fails.

## 5.4 `mean_story_probability` is too HIGH, not too low

Corrected from the previous handoff. Real per-thread `story_rate` ranges 0.000
(seeds 0, 3, 5) to 0.275 (seed 6), mean 0.110. Generated overshoots on seed 8 by
1.5–2.4×. Separately measured: of 559 real comments across the ten threads, 32
narrate a personal experience (5.7%), and **32 of 32 are first person**.

So the user's dimension 3 is right in form — when a real commenter tells a
story it is first person, always — but stories are a **minority** of real
comments, and the generator currently tells too many of them.

---

# 6. ERRATA — claims in the previous handoff that measurement refuted

Load-bearing corrections. Do not re-derive these.

**6.1 "P0's first item: add the 6 missing metrics to the Writer's distribution
target (`run_generate.py:488`)" — the location is a record, not a wire.**
`run_generate.py:523` writes a `metrics` list into `run_config.json` for the
reader. The real target is hard-coded in
`generation_diversity.build_thread_distribution_target:40-43`. It feeds
`joint_target_distance`, which `writer_quality.py` uses only to **rank
candidates**. Under `--writer-retries 0` there is one candidate, so the list has
no effect at all. Five of the six missing metrics also need transformer
classifiers inside the generation process; only `length_cv` is free.

**6.2 "`LENGTH_BUCKET_BOUNDS["very_long"] = (120, 220)` caps `length_cv`" — dead
at runtime.** It is read only by `backend.py:2426` inside
`_retry_note_for_problems`, i.e. on a retry. With `--writer-retries 0` retries
were ~never taken. Every prompt gets `soft_length_guidance` instead. The
generator audit's own "confirmed dead" section was right and the later section
was wrong. `length_cv` is within 3.5% of real anyway.

**6.3 "The story allocation is correct, do not change it."** Wrong. See §5.4.

**6.4 "The plan-echo route lock and the frame guard closed most of the
`self_bleu` gap."** Wrong, and the error was mine twice over. I wrote my own
self-BLEU approximation (max overlap against any other comment) instead of
running `score_thread_self_bleu.py`, which is free. The real metric moved
0.03775 → 0.03750 across the whole frame intervention: **nothing**. The frame
family "that's the part" did drop from 8.1% to 2.7% of comments, so the text
defect is real — but it and `self_bleu_4` are close to independent.

**6.5 "`allocate_story_and_affect` is a no-op auditor (B7)."** It is a
*deliberate* auditor. `generation_distribution.py:108-114` documents that a
post-Planner allocator used to force incompatible affects onto coherent plans and
was removed on purpose. Not a bug.

**6.6 "P4a: license the speaker's own kit and history."** Shipped as
`--own-fact-license own`, measured in v76b, **refuted**: specification tokens
0.05 → 0.02 per comment against a real 0.54, and 0.083 → 0.024 on the licensed
slots themselves. Two reasons, both measurable in advance: 78 of 114
spec-carrying real comments (68%) carry no first-person frame, so the gate
selected the wrong slots; and replacing a vague blanket ban with an explicit
"about the product under discussion, name only what is visible above" made the
binding constraint **sharper** on exactly the detail real comments are full of.

**6.7 "Concreteness means specifications."** Thread-dependent, so not
generalisable: spec-carrying comments are 0% of seed 1 and 64% of seed 5. What
holds on **all ten** threads is quantities (real 12.3× generated) and proper
nouns (real 1.85×). Any concreteness rule must be phrased that way.

---

# 7. WHAT THIS SESSION CHANGED IN THE CODE

One commit: `e9a9fbe`, on top of `67e4e9b`. 14 files, +2032/−75. 266 tests pass,
backend self-test passes, contract drift none.

## 7.1 New modules

**`generalized_card/generalized_card/writer_grounding.py`** — the fact/grounding
rules, which were previously smeared across eight places that disagreed with each
other:

```
prompts.py:1341   focused writer      "Name a product, model, or number only if visible above."
prompts.py:2756+  _story_fact_safety_rule, three branches
prompts.py:1503   system prompt       "Do not invent facts, specifications, numbers…"
prompts.py:113    _own_equipment_block "…do not invent a specification, price, measurement…"
prompts.py:2786   _metric_guidance_block (low-info path only)
prompts.py:~1434  low-info hard rules
prompts.py:1517   mask_specifics       $900 → [amount], 3+ digits → [number]
engine/vocabulary.py:226 (pinned core)  "…product details unless they are visible in the prompt"
```

Measured over the 522 v75 slots before the extraction: 443 slots (84.9%) carried
the blanket ban, 249 (47.7%) carried an "Equipment you may claim as your own"
permission, and **170 (32.6%) carried both — a permission and its revocation in
the same prompt.** All 249 equipment blocks closed by forbidding any
specification about that equipment.

Three modes, all reproducible:
- `off` — v75 verbatim. Verified byte-identical: fingerprint sha256
  `7257a066cf9fc05f80862d0d89ae54d597ea550777fc99baf2cbb96e4a9c32ca` over all 522
  slots, before and after the refactor.
- `own` — the refuted personal-history license. Kept as an arm, not a
  recommendation.
- `named` — the correction. Domain-neutral wording, gated on `substantive_slot`
  (≥25 real words and not micro/short) rather than on a first-person frame.
  **Never run.**

**`generalized_card/generalized_card/speaker_roster.py`** — a thread has people,
not slots. `run_sampled_reddit_generator.py:1408` built the author name as a pure
function of the slot index, so a 186-comment thread was 186 people who each spoke
once. The matched real threads are 559 comments from 265 named authors (2.11
each), with 68% of comment mass written by someone who spoke more than once; seed
8's busiest author wrote 10.

The structure is a **join, not a new sampling policy**: `real_sample_id` already
binds each slot to one matched real comment, and that comment has an author.
`selected_matched_comments` is deterministic (no rng), so the adapter recomputes
the same list. Verified on seed 8: `real_word_count` agrees for 186 of 186 slots,
and the rebuilt roster reproduces 80 named speakers at 2.11 comments each,
busiest 10 — identical to counting the raw jsonl directly.

Leakage: the real author string is used **only** as a grouping key. It is never
stored on a `Speaker`, never rendered, never written to an artifact. A test
asserts this. Equipment is keyed by speaker instead of slot index (previously the
same participant got a different kit in each of their turns), and
`_speaker_identity_block` shows a speaker their own earlier comments in this
thread. **Never run.**

**`generalized_card/scripts/repin_core_contract.py`** — see §2.

## 7.2 Flags added

All default to the previous behaviour, all recorded in `run_config.json`, all
plumbed env var → `backend.py` module attr → CLI flag.

| flag | values | state |
|---|---|---|
| `--own-fact-license` | `off` \| `own` \| `named` | `own` refuted; `named` repaired in v91, not yet measured |
| `--speaker-identity` | `off` \| `matched` | **never run** |
| `--repetition-guard` | `off` \| `blocking` | run in v77/78/79; works mechanically, no metric effect |

## 7.3 The repetition guard, and what it taught

`writer_quality.py` gained `REPETITION_DIAGNOSTIC_PROBLEMS` (promoting
`template_phrase_reused`, `opener_family_reused`, `opening_reused` out of the
advisory set under the guard) and a new whole-comment `repeated_frame:` check.

Why the new check exists: the core's `template_phrase_signature`
(`engine/writer_validation.py:154-178`) reads only `tokens[:28]`. Of the 15
comments in v76a carrying the "that's the part" family it saw **4**; the rest sat
at token 20, 52, 62, 80. The new check reads the whole comment and normalises
curly apostrophes; verified on the actual run output it catches 15 of 15. It is
kept separate from the core signature because that signature also decides
`first_person_frame_unwanted` and `uncertainty_frame_unwanted`, which genuinely
are about how a comment opens.

Result: frame 8.1% → 2.7% of comments (real 0.0%), top shared 4-gram 6.5% → 2.7%
(real 1.5%), novel entities 27 → 45 (real 96). `self_bleu_4` unchanged (§6.4).

## 7.4 Two comment-loss bugs fixed

Both cost real comments, and a shortened thread also damages `avg_depth` and
`structural_virality`, two of the four metrics that "pass".

1. **v77 lost 14 of 186.** Promoting the repetition codes made them
   non-distribution failures, so `consider_distribution_candidate`
   (`backend.py:2160-2179`) returned early and `best_distribution_candidate`
   stayed `None`; exhaustion then returned `skip: True`. The previous handoff
   listed *two* drop paths; I verified `REPAIRABLE_WRITER_PROBLEMS` and forgot
   `backend.py:2205`.
2. **v78 still lost 4.** Their residue was `missing_concrete_anchor` or
   `question_mark_unwanted` — both advisory, both silently accepted on attempt 1.
   Rejecting on attempt 5 what attempt 1 would have kept is not a stricter
   policy, only an inconsistent one.

Fix: `only_style_problems()` plus an `accepted_style_residual_after_repair` path
that retains the best candidate when every residual problem is one the run would
have tolerated at first pass. Tests assert it never fires over `exact_duplicate`,
`empty`, `parent_copy` or `placeholder_literal`. v79: **186 of 186**, and
`rejected_distribution_repair_exhausted` no longer appears.

## 7.5 Other edits

- `engine/model.py` (pinned): `CommentTask.speaker_id`, default `""`.
- `run_sampled_reddit_generator.py` (pinned): author name uses `speaker_id` when
  present, otherwise the old slot-indexed name; `speaker_id` carried onto the
  comment dict.
- `task_distribution.py`: `speaker_id` added to `PLANNER_AND_SLOT_INVARIANTS` so
  the surface rebalancer cannot rewrite it — the same omission that lost
  `semantic_move` in 347 of 347 reply slots.
- `core_contract.py`: `writer_grounding` and `speaker_roster` are now pinned and
  verified by `run_generate.py`. Policy version is
  `generalized-card-v2-own-fact-license-v76-20260815`; v74 and v75 are in
  `HISTORICAL_GENERATION_POLICY_VERSIONS`.
- ~60 new tests across `test_generalized_card.py` and
  `test_planner_field_survival.py`.

## 7.6 Known debt

**The policy version was not bumped after v76.** v77, v78 and v79 have different
Writer behaviour but the same policy string. The flags are in each
`run_config.json` so the behaviour is recoverable, but the next change should
bump the version and move `…own-fact-license-v76-20260815` into
`HISTORICAL_GENERATION_POLICY_VERSIONS`, or those three runs become
un-evaluatable. `run_evaluate.py` passes `allow_historical=True`, so evaluation
of existing runs is safe either way; `--resume` on generation is not.

## 7.7 v80 continuation — implementation complete, paid run pending

The active generation path, all policy modules, all twelve scorers, the
evaluation aggregator, and the existing v79 records were read end to end before
this change. The new policy string is
`generalized-card-v2-planner-contract-coherence-v80-20260816`.

Free replay over v79's 186 Planner records found **30 contradictions that the old
quality gate accepted**: 21 `no_story + personal_story` plans and 9 `polite`
plans attached to incompatible roles/functions. The StorySeeker join independently
showed that planned story slots contributed only about 25% of total story
probability; 25 of 167 `no_story` comments were still classified as stories.

Implemented:

- validate the whole story/tone semantic contract after the frozen schedule is
  applied, before Writer execution;
- render an explicit non-narrative rule for every `no_story` Writer path;
- replace the refuted polite length/anti-hedge theory with measured social cues;
- show direct-reply planners sibling delta/novelty coverage;
- record every behavioral field in resume/extension/upgrade comparisons;
- add `--social-contract-coherence` and `--reply-sibling-visibility`; `off`
  preserves the pre-v80 arms and both values are recorded in `run_config.json`;
- remove only functions proven unreferenced by repository-wide reference and AST
  audits. Active reviser helpers remain; the user-owned evaluation/cleanup files
  outside `generalized_card/` were not touched.

The curly-apostrophe counterfactual was also run for free on 40 comments / 780
pairs. Self-BERTScore moved 0.52947 -> 0.52381, toward real but far short of the
full ~0.034 gap. Do not ship a held-out-test-calibrated typography transform;
this is a secondary hypothesis requiring evaluation-excluded calibration.

Verification before any paid generation: 270 full generalized tests passed,
the backend self-test passed, CLI help renders, Ruff passed, all 72 core pins
matched, and scoped `git diff --check` passed. No claim is made that a metric or
p-value improved; v80 has not yet generated new text.

---

# 8. THE PLAN

Full detail in `tasks/todo.md`. Re-ordered by **which measured gap it moves**,
not by where a code defect happens to be. The old P0–P6 numbering is retained in
`tasks/todo.md` for traceability, with each item marked kept / struck / demoted.

## A — realize the assigned register  ← the largest gap, dimension 4

Targets `polite_rate`, `impolite_rate`, `neutral_rate`, `emotion_entropy`,
`hard_disagree_rate` together, because §5.1 shows they are one failure.

- [ ] Remove the hedge and thank-you prohibitions from
      `TONE_DEFINITIONS["polite"]` (`generation_distribution.py:480-489`). The
      comment above the table records the prediction that justified them; §5.2
      shows the prediction was wrong.
- [ ] License the emotional endpoint explicitly (real 2.5% vs generated 1.1%).
- [ ] Suppress decision-framing nouns in the Writer's own rules (real 0.5% vs
      generated 4.3%, an 8.6× overshoot).
- [ ] Do **not** add length to polite slots — measured and eliminated in §5.2.
- [ ] Offline acceptance check before any paid run: re-render the v79 prompts and
      confirm the banned surfaces are gone. Then a seed-8 run, judged on
      **tone realization rate (59.2% baseline) and emotion_entropy**, not on
      p-values.
- [ ] Ablation flag, `off` byte-identical.

## B — the global typographic signature  ← the only lever aimed at `self_bertscore`

- [ ] Deterministic post-generation normalisation: curly → straight quotes and
      apostrophes at a rate matched to the domain's real corpus, rather than
      100% curly on every comment. Real: 51% of comments carry a straight
      apostrophe inside a word, generated 0%.
- [ ] Free, offline-verifiable over the whole corpus, no prompt work, no API.
- [ ] Then re-score `self_bertscore` on an existing run — the scorer does not
      need regeneration, only re-cleaning. **This makes the hypothesis testable
      for $0.**

## C — bring `mean_story_probability` down

- [ ] Generated overshoots 1.5–2.4× on seed 8. Real per-thread `story_rate` is
      0.000 on three of ten threads. The per-thread target already scales from
      the matched template (`generation_distribution.py:129-134`), so check
      whether the overshoot is allocation or realization before changing
      allocation — non-story comments sounding narrative would also do it, since
      the metric averages over every comment (§3).

## D — the two built-but-never-run arms

- [ ] `--own-fact-license named` — targets quantities and proper nouns, the two
      concreteness signals that hold on all ten threads.
- [ ] `--speaker-identity matched` — targets `self_bertscore` via voice
      variation. Note §5.3 may explain that metric more cheaply; run B first.

## E — reply-planner sibling visibility (was P3)

- [ ] Every depth ≥ 1 batch takes `render_direct_reply_planner_prompt`, which
      renders no prior-plan ledger, no coverage summary, no sibling contract.
      Verified on seed 2: depths 3–8 are single-slot batches and tasks 38–45 are
      nine near-duplicate moves. Plausibly feeds dimension 1 and 4.

## Struck

- **Plan-echo validator (old P1).** Echo is at 0.0% and the route lock that fixed
  it moved no metric. Nothing to guard.
- **Length for polite slots (part of old P5).** Eliminated in §5.2.
- **`LENGTH_BUCKET_BOUNDS` 220-word ceiling (part of old P6).** Dead at runtime,
  §6.2.
- **`--own-fact-license own`.** Refuted, §6.6. Kept only as a reproducible arm.

## Rejected: two writers supervising each other

The user asked. Recommendation stands: **no LLM critique loop.** Cost doubles and
critique-driven revision pushes text toward the balanced, hedged register that is
already the failure mode. The stronger argument: the system has ~20 validators and
ignores most of them. The useful form of supervision is a deterministic
discriminator, which is what §7.3's frame check is.

---

# 9. OPERATING RULES

## Before diagnosing

- **Read every file on the active path end to end.** Not grep hits. In this
  codebase that means the CLI, the backend adapter, **every** prompt builder
  (root planner / direct-reply planner / focused writer / low-info writer — there
  is more than one per role, and their schemas have contradicted each other
  twice), the generator facade, the engine modules, and the policy modules.
- **Read the scorer before theorising about a metric.** §3 exists because two
  wrong conclusions came from not doing this.
- **Never approximate a metric that is cheap to compute.** `self_bleu_4` needs no
  model and runs in seconds.
- Subagent reports and prior handoffs can be wrong. Two of three full-file reads
  in an earlier session contained a materially wrong claim, and §6 lists six more
  from the last handoff. Re-verify load-bearing claims against run artifacts.

## Before changing

- **One mechanism per API run**, and predict the magnitude first so a null result
  is interpretable. Write the prediction down.
- **Do not use old implementations as design authority.** Reproduce them from
  git or the run's source snapshot when needed; delete disproven, unreachable
  controllers from the current path.
- **Distribution diagnostics never select a Writer candidate.** Only output
  that cannot be persisted may receive bounded hard recovery.
- **Apply the change to every path.** v74 converted only the focused writer and
  left 106 of 522 slots on the old prompt, which made that release
  unattributable.
- **No domain vocabulary in Writer-facing rule text.** Every test runs on camera,
  so nothing else will catch it.

## Before handing the user a command

- **Dry-run on a throwaway tag with `--prepare-only`**, then delete the tag. This
  caught two real errors this session: a missing `choices` entry, and the fact
  that the adapter files are pinned under different registry names than I assumed.
- **Separately verify what `--prepare-only` skips** — it returns before the
  API-key check.
- **Re-pin with the script, then confirm the drift list is exactly the files you
  edited.**
- Run `PYTHONPATH=generalized_card .venv/bin/python -m pytest -q generalized_card/tests`
  (**259 pass** at v81) and the camera-product backend self-test.

## Interpreting a single-thread run

- n=1 gives no p-value and a degenerate Cliff's delta. **Only the relative error
  against the matched real thread is readable.**
- Two runs of the same config have never been made, so **run-to-run noise is
  unknown** and any single-thread ranking of two configs is unsupported. If a
  future decision depends on ranking, spend $0.9 on a repeat of an existing
  config first.
- Content diagnostics computed directly on the text (frame rate, comment count,
  realization rate) are far more trustworthy at n=1 than metric deltas.

---

# 10. COMMANDS AND PATHS

## Environment

API keys live in `third_party/MiroFish/.env` as **`LLM_API_KEY`**;
`run_generate.py` loads that file itself but still pass `--api-key-env LLM_API_KEY`.

## Generation, fresh N=10 v93 (~$5.5–$7, roughly 2–3 hours)

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag generalized_card_camera_gpt54_v93_named_n10_20260818_v1 \
  --domain camera --model gpt-5.4-mini \
  --base-url https://api.openai.com/v1 --api-key-env LLM_API_KEY \
  --pool-size 150 --max-posts 10 --posts-per-run 1 \
  --start-seed-index 0 --sampling-seed 42 \
  --context-dropout-rate 0.42 --context-jitter-rate 0.32 \
  --plan-quality-repairs 3 --writer-hard-recovery-rounds 2 \
  --post-retry-limit 1 \
  --domain-claim off --writer-prompt focused --writer-route-lock own_words \
  --social-contract-coherence on --reply-sibling-visibility on \
  --own-fact-license named --speaker-identity matched --resume \
  2>&1 | tee /tmp/generalized_card_v93_named_n10_generation.log
```

This creates ten one-post run directories for seeds 0–9 and safely skips
completed posts when the identical command is resumed. `plan-quality-repairs
3` gives joint contract conflicts a bounded Planner repair budget. Hard Writer
recovery handles only empty,
duplicate/copy, placeholder, or planner-skeleton output; it does not optimize a
metric.

The seed pool is derived, not a flag:
`artifacts/generalized_card/seed_pools/camera_product_150_seed42.json`.

## Evaluation — zero API cost, CPU only

```bash
python3 -u generalized_card/scripts/run_evaluate.py \
  --tag generalized_card_camera_gpt54_v93_named_n10_20260818_v1 --device cpu
```

Cheap single metric, no model, seconds:

```bash
cd scripts/evaluation && python3 score_thread_self_bleu.py \
  <run>/generated/run_NN_sampled_reddit/discussion.json --output /tmp/sb.json
```

## Artifacts per run

```
artifacts/generalized_card/runs/<TAG>/
  run_config.json                        full config incl. every ablation flag
  generated/run_NN_sampled_reddit/
    generation_records.json              per slot: the 62 task fields, prompt, raw, comment
    discussion.json
  cleaned/run_NN_.../
    politeness_results.json              per comment pred_label  (no text — join by comment_id)
    go_emotions_results.json             per comment dominant_emotion
    storyseeker_results.json
  logs/writer_distribution_control.jsonl attempts, final_status, per-attempt problems
  evaluation/revised_generated_thread_scores.csv    the 12 metrics, per thread
  matched_evaluation/matched_real_thread_scores.csv real values for the matched seeds
  content_profile_audit.json              exact-matched content/realization diagnostics
  content_profile_audit.md                human-readable form of the same report
```

Real ground truth: `data/raw/discussions/camera_product/<product>/<product>.comments.jsonl`,
filtered by `post_id == seed_pool.seed_posts[].source_raw_post_id`.

## Reconstructing tasks offline

`generation_records.json[].task` carries exactly the 62 `CommentTask` dataclass
fields, so `CommentTask(**record["task"])` round-trips. That is how every offline
prompt-rendering check in this session was built — no API, full corpus.

---

# 11. RECOMMENDED FIRST MOVE

The free checks are complete. Run a fresh N=10 tag under v93 with social
contract, sibling visibility, named concreteness, and matched speakers on. Do
not reuse the v92 tag because its first two discussions have the older root
Prompt. Require exact 10/10 coverage, then evaluate the 12 metrics and inspect
tone realization, repetition, story mass, emotion, and customer-service advice
before adding another mechanism.

---

# 12. 2026-08-17 v84 ADDENDUM — INCOMPLETE THREADS ARE NOT ARTIFACTS

The paid v80 seed-8 run had 186 planned/recorded slots but only 185 rendered
comments. S99 was skipped after three `parent_copy` failures. Its scheduled
quote opener instructed the Writer to reproduce the exact parent line, so the
Prompt and hard guard contradicted each other. This is current-path evidence,
not a conclusion copied from an older implementation: the quote scheduler,
Writer Prompt, hard guard, persistence wrapper, output audit, and evaluation
gate were all traced in the present source.

v84 makes two independent guarantees:

1. A quote opener asks for a short exact markdown excerpt. `parent_copy` is
   waived only for that scheduled form, only when the excerpt occurs in the
   visible parent, is not the whole nontrivial parent, and is followed by an
   independent reply. All ordinary copying remains unpersistable.
2. Writer coverage must be exact before `generate_one_post_slot` can call
   `replace_or_append_post`. If bounded same-slot recovery is exhausted, the
   wrapper raises a recoverable error. With the default `post_retry_limit=1`,
   the run fails visibly and writes no incomplete post; setting a larger limit
   is an explicit whole-post cost choice. Output audit separately requires exact
   record/rendered coverage for any artifact that has generation records.

The old v80 artifact now audits as `evaluable=false`, with
`accepted_share=0.9946`, `skipped_generation_slots=1`, and
`complete_structural_coverage=false`. This is important because a missing
comment changes the pair population for Self-BLEU, Self-BERTScore, and semantic
cosine, can change length CV, and can remove a tree node that affects depth and
virality. A 12-metric comparison is not meaningful until this invariant holds.

Historical code is retained only for provenance and reproducibility. Do not use
an older implementation as design authority; re-establish every causal claim
from the current path, current scorer, and current artifact.

---

# 13. 2026-08-19 v98 ADDENDUM — THE `self_bertscore` CAUSE, AFTER TWO REJECTIONS

Read `tasks/v98-worklog.md` for the full evidence. This section records only what
a future session must not re-derive.

## The metric is lexical breadth, not topic and not repetition

`self_bertscore_mean_f1` had been the worst metric in the suite since v72 and
three plausible causes were measured. Two were rejected:

- **Length spread.** Reweighting generated comment pairs onto the real pairs'
  length-ratio mix closes 0.0033 of the 0.0163 gap.
- **A duplication tail.** Trimming the top 20% of pairs on both sides leaves the
  gap at +0.0154 against +0.0163 untrimmed. It is a uniform shift.
- **The surface register.** Real comments that differ in typing habits are only
  0.003-0.011 lower in function-word cosine than ones that share them, against
  a 0.134 generated-vs-real gap.

What is left is that the generated thread uses 2,670 distinct word types where
the matched real threads use 3,645, at types/sqrt(tokens) 15.95 against 21.02 —
while per-comment type-token ratio at a fixed 30 tokens is *higher* in the
generated text. Individual comments are fine; the thread's lexicon is small.

## One instruction caused it

453 of 532 slots are `no_story`. v96's instruction for them banned tense, not
narrative. On those slots: past-tense verbs 0.181 against a real 0.543, future
0.031 against 0.226, present perfect 0.031 against 0.167. `have` at 11% of its
real rate, `will` at 1%. The fallback is a timeless conditional register —
`the` 147%, `if` 225%, `whether` 1800%, `matters` 2900%.

It was simultaneously a prompt contradiction: **247 of the 532 rendered v97
prompts (46.4%) carried both "Be particular rather than general" (the licensed
grounding rule under `--own-fact-license named`) and "no past action, event".**
This is exactly what `writer_grounding` was built to eliminate, recurring in a
different pair of rules.

## Two checks that cost nothing and should run every version

1. **Grep the rendered prompts for contradicting rule pairs.** They are stored
   verbatim in `generation_records.json[].prompt`.
2. **Test the causal claim on the reference corpus before writing the fix.** If
   the mechanism is "X causes metric M", the excluded real threads can usually
   say whether X and M covary. `sentence_rhythm` was fully built and pinned
   before its hypothesis was falsified this way.

## The one non-domain-adaptive profile

`length_calibration` holds a fitted transfer function
(`log(realized) = 0.3835 + 0.8925*log(asked)`, n=532, R2 0.894). It is a
property of the model and the prompt, not of the domain, so it is a recorded
constant rather than a domain-profile entry. Both the target
(`task.real_word_count`) and the realized count (`comment.word_count`) are in
every generation record, so any run's artifact refits it with no regeneration.
**Refit when the model changes.** Every other v98 profile is measured per domain
from that domain's evaluation-excluded threads.
