# Generalized CARD Version Log

## Read this before quoting any pass count

Pass counts are only comparable between runs with equal **structural coverage**
(generated comments / matched real comments over the same seeds). Measured over
the 10-seed pool, whose matched real threads hold 515 comments:

| version | coverage | metrics passing |
|---|---:|---:|
| v19 | **0.45** | 11 |
| v14 | 0.60 | 9 |
| v16 | 0.67 | 9 |
| v15 | 0.68 | 9 |
| v34 | 0.67 | 8 |
| v33 | 0.71 | 8 |
| v64 | **1.01** | 6 |

Coverage and pass count move together, and the mechanism is not subtle:
`self_bleu_4`, `self_bertscore_mean_f1`, and `semantic_mean_cosine` are means over
all comment pairs, so a shorter thread has fewer pairs and lower mean pairwise
overlap. `length_cv` shifts as well. **Truncating a thread flatters exactly the
metrics this work is trying to match.**

So v19's 11/12 is a truncation artifact, not a target: it generated 231 comments
where the real threads held 515. v34 generated 76 comments against a 186-comment
real thread. v64 was the first version to generate complete threads, which is why
its 6/12 is the first honest measurement rather than a regression. **Only v64 and
later are comparable to one another.**

The historical runs therefore violated this repository's own rule in
`AGENTS.md`: "For first-pass Planner-Writer generation, preserve every matched
structural slot. Never shrink, cap, or omit a matched thread by default."
`RUN_INDEX.md` reports comment counts per run; check them against the seed pool
before drawing any comparison.


One entry per generator policy version: what changed, why, which run tested it,
and what happened. Git history, this file, `core_contract.py`'s historical policy
set, and each run's `run_config.json` together form the provenance chain.

Machine-generated companion: [`RUN_INDEX.md`](RUN_INDEX.md), rebuilt with

```bash
PYTHONPATH=generalized_card python3 generalized_card/scripts/build_version_log.py
```

## Recording a version

Before any run that changes behavior:

1. Bump `GENERALIZED_V2_GENERATION_POLICY_VERSION` in `core_contract.py` and move
   the previous value into `HISTORICAL_GENERATION_POLICY_VERSIONS`.
2. Recompute the pinned `CORE_FILES` hashes.
3. Add an entry below **before** spending the API call, stating the hypothesis
   and the predicted direction, so a null result stays interpretable.
4. Use a run tag containing the version number.
5. After evaluating, fill in the result and regenerate `RUN_INDEX.md`.

---

## v93 — structural root/reply boundary (2026-08-18)

Policy ID: `generalized-card-v2-root-reply-boundary-v93-20260818`.

The paid v92 N=10 run completed seeds 0 and 1, then stopped in seed 2 on root
S9. Its initial 108-word slot lacked a required long-form plan. All three S9
repairs supplied the requested five connected beats. The first otherwise-valid
candidate also used `reply_delta_type=social_close` on this structurally root
slot; the social validator correctly rejected that combination. A later attempt
collapsed the long turn into a narrow question, and the third repeated the root
`social_close`, so the old empty-development plan remained selected.

v93 makes anonymous topology authoritative before quality selection. Root plans
clear `reply_delta`, `reply_delta_type`, and `reply_novelty_anchor`, with every
nonempty override recorded in `control_normalizations`. Direct replies preserve
all three fields. The root Planner schema now requests literal `none`, and 138
lines of duplicate direct-reply definitions, contrast rules, and the unreachable
parent-contract renderer were removed from `prompts.py`; direct replies already
use their dedicated compact Planner Prompt.

Replaying the actual first v92 S9 repair under v93 changes its rank from
`(1, 46)` to `(0, 41)`, removes the only blocking issue, and preserves all five
development beats. Offline acceptance: 300 generalized tests and 3 focused
Self-BERTScore tests pass; Ruff is clean; active and active-plus-legacy parity
are healthy; 93/93 pins are clean; and the exact N=10 command completed
`--prepare-only` as
`generalized_card_camera_gpt54_v93_named_n10_20260818_preflight_v1` with no API
call. Because v92 already generated two threads with the old root Prompt, the
formal N=10 evaluation must use a fresh v93 tag rather than mix policies.

---

## v92 — lossless `domain-claim=off` planning (2026-08-17)

Policy ID: `generalized-card-v2-lossless-domain-claim-off-v92-20260817`.

The post-v91 Planner→Writer audit found that `--domain-claim off` disabled only
delivery. Both root and direct-reply Planners still spent Prompt and output
tokens assigning a fact that `backend.py` then withheld from the Writer. The
Planner could consequently build `semantic_move`, `detail_focus`, or
`domain_intent` around information absent at realization—the exact handoff gap
this generator is supposed to eliminate.

v92 makes `off` mean off at both stages. Root and reply schemas require the
literal `none`, the domain-knowledge/claim instruction is absent, and a compact
rule requires the complete contribution to live in fields the Writer receives.
Normalization also clears a claim if the model ignores the rule. `planned`
mode retains the prior claim path unchanged. This removes redundant Planner
Prompt/output mass without weakening semantic planning from visible seed,
parent, branch, and evaluation-excluded discourse patterns.

The next configuration remains `--domain-claim off --own-fact-license named`:
the Planner hands over the whole semantic move, while the slot-gated Writer adds
varied local particulars instead of receiving one separately injected fact.
Expected effects are fewer abstract/incomplete moves and lower Planner cost;
12-metric movement remains unmeasured until the paid diagnostic.

Offline acceptance is complete: 299 generalized tests and 3 focused
Self-BERTScore scorer tests pass; Ruff is clean; active and
active-plus-legacy parity are healthy; all 93 source pins have zero missing,
untracked, unpinned-import, or drift findings. Rendered root and direct-reply
Prompt tests cover both flag values, normalization clears a noncompliant
off-mode claim, and the exact named/off seed-8 public command completed
`--prepare-only` as
`generalized_card_camera_gpt54_v92_named_seed8_20260817_preflight_v2` without an
API call. No v92 content or metric result is claimed yet.

---

## v91 — slot-gated concreteness permission (2026-08-17)

Policy ID: `generalized-card-v2-slot-gated-fact-license-v91-20260817`.

The pre-run completion audit found that the built-but-unrun `named`
concreteness arm was not safe to enable. Its slot resolver licensed only
substantive comments (at least 25 anonymous words and not micro/short), but its
system Prompt unconditionally told every comment to name particulars and give
amounts. A micro reaction therefore received a global pressure to invent detail
and a per-comment rule allowing names/numbers only when visible.

v91 makes the system sentence an authorization boundary only: per-comment
instructions may override the generic visibility ban for an explicitly
licensed turn. The actual name/amount instruction remains once, in the
substantive user Prompt. Unlicensed micro/short turns retain their visible-only
rule, and the legacy `own` permission receives the same slot-gated treatment.

The gate has a meaningful held-out structural scale. On the exact 186-slot
seed-8 skeleton it licenses 110 slots (59.14%); the matched real thread has a
digit in 59.68% of comments, while v80 generated only 31.35%. Real model
designators were 118 versus 29 generated. The next diagnostic should therefore
use `--own-fact-license named` with `--domain-claim off`: varied particulars are
realized locally instead of injecting one Planner fact across nearly every
comment.

Expected direction: more varied names and quantities, lower designator
concentration, and less abstract/advisory prose. This may move Self-BLEU,
Self-BERTScore, and semantic cosine through varied content, but n=1 can only
diagnose that mechanism; formal distribution claims still require sufficient N.

Offline acceptance: 297 generalized tests plus 3 focused Self-BERTScore tests,
Ruff, healthy active and active-plus-legacy parity, 93/93 clean source pins, and
a named-mode backend self-test. Full 186-slot Prompt replay produced exactly
110 licensed Prompts, each with one behavior instruction; all 76 unlicensed
Prompts had zero, the system contained one conditional authorization and no
behavior duplicate, and no invented-equipment block appeared. Exact named-mode
seed-8 `--prepare-only` passed under
`generalized_card_camera_gpt54_v91_named_seed8_20260817_preflight_v1`. No v91
API call has been made.

---

## v90 — one story-grounding boundary for both Planner paths (2026-08-17)

Policy ID: `generalized-card-v2-reply-story-grounding-v90-20260817`.

The post-v89 completion audit found that the synthetic-story repair covered the
root Comment Planner but not the specialized direct-reply Planner. Direct
replies were still required to plan an actual first-person event sequence while
also being forbidden to carry a source participant's detail or invent a fact
about the seed. That left the model to guess whether an ordinary synthetic
personal sequence was permitted—the same ambiguity that helped make the v88
root story slot unrealizable.

v90 defines that Planner boundary once and renders it on both root and direct
reply paths: an ordinary, non-verifiable first-person sequence may be
synthesized around a visible or generic local point, but product facts and other
externally checkable outcomes may not be invented. The Writer's existing
off-mode story rule already has this boundary, so v90 makes Planner and Writer
agree; it does not expand the Writer's factual license. A regression test checks
that the direct-reply Prompt contains both halves of the rule and still forbids
inventing seed facts.

Expected result: scheduled direct-reply stories should no longer consume repair
attempts or fail because the Planner interprets the factual boundary as a ban on
all personal sequence. This is a completion/reliability fix; movement on story
realization and the 12 metrics remains a paid-run question.

Offline acceptance: 295 generalized tests plus 3 focused Self-BERTScore tests,
Ruff, matched-speaker backend self-test, active and active-plus-legacy parity,
and 93/93 source pins with zero drift or closure gaps. Exact seed-8
`--prepare-only` passed under
`generalized_card_camera_gpt54_v90_preflight_seed8_20260817_v1`. No v90 API call
has been made.

---

## v89 — realizability-first Planner repair (2026-08-17)

Policy ID: `generalized-card-v2-realizability-first-planner-v89-20260817`.

The first paid v88 seed-8 attempt stopped after 24 Planner requests, before any
Writer call: 116 seconds and $0.1805. The offset-8 batch ended with blocking
contracts on S10, S13, and S15. The audit proved three distinct causes.

- Candidate selection used one scalar issue score. A repaired S15 removed its
  story conflict but introduced a semantic collision (weight 10); because the
  story conflict weighed 8, the realizable candidate was rejected and the
  impossible plan was retained. v89 ranks candidates first by the number of
  Writer-blocking contract issues, then by aggregate quality.
- The root Planner required scheduled stories and firsthand evidence while a
  blanket rule banned hidden anecdotes and all facts absent from a title-only
  seed. v89 states the same synthetic, non-verifiable personal-sequence
  boundary already enforced by the Writer, without licensing externally
  checkable product facts.
- A `polite` classifier target was treated as semantic truth and could abort a
  186-comment post unless the plan agreed and used one of three functions.
  Polite-Guard scores realized surface text, so v89 retains this pairing as
  low-weight anti-customer-support feedback but removes it from the blocking
  social contract. Story, affect/social-close, surface capacity, and long-form
  coherence remain blocking.

Planner audit rows now include JSON-safe initial, candidate, recovered, and
selected plan snapshots plus the before/candidate repair ranks. This closes the
v88 observability gap: its log recorded issue labels and scores but not the plan
whose acceptance was being decided.

Expected result for the replacement seed-8 run: no termination from the known
S10/S13 polite pairings; an S15 repair that reduces blocking contracts is kept
even if collision remains as a logged warning. This is a reliability fix, not a
claim that any of the 12 metric distributions improved. Paid content and metric
evidence remain pending.

Offline acceptance: 294 generalized tests plus 3 focused scorer tests, Ruff,
matched-speaker backend self-test, active and active-plus-legacy parity, 93/93
source pins with zero drift or closure gaps, and exact v89 seed-8
`--prepare-only`. No v89 API call has been made.

---

## v88 — structural speakers without invented biography (2026-08-17)

Policy ID: `generalized-card-v2-structural-speakers-grounding-v88-20260817`.

Completion audit before the paid v87 run found two current Prompt/structure
problems. First, `--own-fact-license off` still rendered an invented equipment
permission before the conservative fact rule revoked personal experience. A
full 186-task replay measured 78 equipment blocks, 144 personal-experience bans,
and 61 Prompts containing both. Preserving that contradiction solely as a
historical ablation violated the active Prompt rules; git already preserves it.
v88 renders an equipment shortlist only for the explicit legacy `own` license,
and the same replay now measures zero equipment blocks and zero conflicts.

Second, `speaker-identity matched` mixed a valid matched structural join with
invented kit, tenure, and use-case biography, so it could not safely be the
default. v88 deletes those semantic fields and their kit-filter helper. The
roster retains only anonymous speaker ID, OP status, slot IDs, and anonymous
account status. The Writer may see only its own earlier generated turns and an
instruction to keep factual self-claims consistent while still following the
current turn's assigned voice and affect. Real author strings never cross the
boundary. Matched recurring-speaker structure is now the default; `off` remains
the one-author-per-slot structural ablation.

Current seed-8 structural audit: 186 slots form 97 generated speaker groups,
including 80 named-source groups and 17 anonymous one-shots; named groups
average 2.112 turns, recurring groups own 66.7% of comment mass, and the busiest
group has 10 turns. The active expander integration test proves repeated source
authors receive the same anonymous `speaker_id`.

Expected directions are fewer grounding contradictions, less fake persona
boilerplate, and more realistic participant continuity. Self-BERT or other
metric movement is a hypothesis, not a result. Offline acceptance: 292
generalized tests plus 3 focused scorer tests, Ruff, matched-speaker backend
self-test, active and active-plus-legacy parity, 93/93 source pins, full Prompt
replay, and exact v88 seed-8 `--prepare-only`. No API call was made.

Paid result: the formal seed-8 attempt failed before Writer generation after 24
Planner requests (`$0.1805`, 116 seconds). No discussion artifact exists and no
content or 12-metric conclusion can be drawn. v89 supersedes v88 for rerun.

---

## v87 — payload-safe Writer routing and final-contract refresh (2026-08-17)

Policy ID: `generalized-card-v2-payload-safe-writer-routing-v87-20260817`.

Hypothesis before the paid run: a short surface shape must not send a
substantive Planner payload into the low-information Writer, whose correct hard
rules prohibit advice, explanation, and caveats. Building the focused ledger
directly from bounded records should also remove duplicated exclusions without
weakening exact-duplicate persistence checks. Expected qualitative directions
are more faithful short corrections/helpful turns, fewer contradictory social
assignments, and less repeated Prompt context. No generated content or
12-metric improvement is claimed before a paid artifact is scored.

Evidence and changes:

- Replayed all 186 recorded v80 tasks through every current Writer route. The
  old routing condition selected 32 low-information slots, including six
  `soft_helpful` payloads and one `correction`. v87 gates short utterance modes
  by payload semantics first; the replay now selects 25/186, all from the
  explicit low-information payload set and all `no_story`.
- Focused/low-information Prompts no longer build a full five-section thread
  blackboard and parse two sections back from its rendered text. They construct
  bounded semantic and short-line ledgers from source records, omit exact
  openings already shown nearby, and avoid restating the same social-close move
  as both required and already covered. Replay found no exact duplicate long
  lines and no duplicated required semantic move.
- Writer-facing tone controls are recomputed after every Planner-owned role,
  payload, voice, and surface contract is final. A stale
  `pure_acknowledgement` can no longer survive on a neutral datapoint or
  correction.
- The social-reaction validator is now bidirectional. Gratitude/relief already
  required a gratitude reaction; v87 also rejects a `gratitude_reply` or
  `social_close` paired with neutral affect, an explanatory/meta payload, a
  non-reaction function, or a story.

Offline acceptance: 290 generalized tests plus 3 focused scorer tests pass;
Ruff, camera backend self-test, active and active-plus-legacy parity pass; all
93 declared pins agree with zero missing, untracked-active, unpinned-import, or
drift entries. The exact seed-8 v87 command passed `--prepare-only` with no API
call. v87 was superseded by v88 before a paid generation.

---

## v86 — compact low-information Writer and root-relation clarity (2026-08-17)

Policy ID: `generalized-card-v2-root-relation-prompt-v86-20260817`.

Hypothesis before the paid run: low-information slots should realize their
assigned reaction, rant, question, acknowledgement, or bare answer more often
when the Writer sees one compact discourse contract rather than several
overlapping copies. Root comments should not be told that they answer a parent
that does not exist. Expected qualitative directions are fewer generic helpful
or customer-service-shaped turns, less Planner-language echo, and more faithful
short social/affective realization. No 12-metric improvement is claimed until a
new artifact is scored.

Changed:

- For a root slot, the focused Writer now receives `relation to post` and values
  such as `answers_post`; direct replies retain `reply relation` and parent
  values. The persisted Planner plan is unchanged, preserving audit evidence.
- The low-information Writer now uses the same compact discourse contract and
  bounded semantic/short-utterance ledger as the focused substantive path.
  Duplicate private-slot, semantic-contract, local-move, full-blackboard,
  placeholder, payload, tone, story, affect, and length renderings were removed.
  Its low-information and grounding hard rules remain.
- Reviser-only Prompt adaptation and Self-BLEU revision diagnostics moved from
  active `prompts.py` to `legacy_reviser_prompts.py`. AST hashes prove every
  migrated function is identical and every retained active Prompt function is
  unchanged apart from the two v86 Writer edits.

Offline acceptance: 286 generalized tests plus 3 focused scorer tests, Ruff,
camera backend self-test, active and legacy parity, and 93/93 pins with zero
active untracked or unpinned local imports. v86 was superseded by v87 before a
paid generation.

---

## v85 — auditable Planner controls and dead-path pruning (2026-08-17)

Policy ID: `generalized-card-v2-auditable-plan-controls-v85-20260817`.

This release is a current-path simplification and observability pass before the
next paid run. It does not claim a direct improvement to any of the 12 metric
values.

Changed:

- The existing slot-schedule override events are now retained in
  `planning_quality.jsonl`, both for the initial Planner response and across
  bounded repair attempts. This exposes whether the Planner originally obeyed
  each fixed story/tone/affect/opener contract; the post-override semantic
  coherence checks remain unchanged.
- `perspective_concentration` remains an audit/strict warning but no longer
  triggers a slot-local LLM repair. Structural branch ownership overwrites
  `perspective_id` before every evaluation, so such a repair could not change
  the concentration and only spent requests.
- Removed two other validations that cannot fire on the active path:
  `invalid_perspective` is deterministically canonicalized first, and
  `branch_route_conflict` compared a topology-owned branch ID with the same ID
  after normalization. The effective concentration, branch-goal, reply-delta,
  social-contract, capacity, and collision checks remain.
- Removed the retired tone-overlay control from current Writer inputs. Its two
  dataclass/persistence fields remain solely so historical records deserialize;
  current code neither assigns nor consumes them.
- Removed the unreachable `constructive_polite_helpful` finalizer branches and
  the unused scalar `projected_metric`; the live batched projection path remains.
- Replaced the old print-only content comparison with a pinned, tested matched
  audit automatically run after evaluation. The old tool matched lexical text
  correctly but compared generated emotion/story against the entire domain
  corpus. The new join uses the exact seed ID and product directory for real
  per-comment model rows, reports all 12 paired distances, Planner→Writer
  realization, repetition contributors, and explicitly weak helpful/profanity
  surface probes in machine-readable JSON and Markdown.
- Persist the exact evaluation-excluded reference metric template atomically in
  each post's `thread_plan`. The content audit now decomposes every metric into
  real → Planner target and Planner target → Writer output, with separate
  MWU/KS/Cliff/Wasserstein statistics. Legacy logs are accepted only when their
  post alignment is provably unambiguous.
- Replace pre-score cleanup with a byte-identical scoring snapshot. The output
  audit must reject bad Writer text or tree metadata; evaluation no longer
  edits, deletes, or normalizes the artifact it claims to measure.
- Move the active metric suite and formal distribution statistics behind small,
  pinned generalized modules. The matched evaluator and all scorer CLIs are now
  tracked in git; the recoverability audit checks both git tracking and the
  transitive local-import closure. Default parity excludes legacy revisers.
- Treat n=1 as descriptive at every output layer. MWU/KS numbers remain visible,
  but neither the matched evaluator nor `run_evaluate` can print a false
  `12/12 PASS` for one thread.

A zero-API audit separated target choice from realization. Across both the
10-thread diagnostic set and all 150 matched seeds, the selected excluded-real
Planner templates pass both MWU and KS on all 12 metrics. That is evidence that
the distribution sampler is working, not permission to tune against final test
p-values. On the historical v80 n=1 thread, for example, polite target/real are
0.249/0.232 but Writer output is 0.059; story target/real are 0.128/0.111 but
Writer output is 0.249. The next paid run should therefore test Writer
realization rather than rewrite the sampler.

Expected paid-run effects are bounded and falsifiable: fewer impossible Planner
repair requests, explicit counts of initial fixed-contract disagreement, and no
`tone overlay: none` Prompt noise. Content/metric success still requires the new
large-thread artifact followed by a sufficient-N matched evaluation.

Offline acceptance: all 285 generalized tests pass; the focused scorer test adds
3 more passes; Ruff passes on every active changed source; the camera-product
backend self-test passes; all 92 declared pins agree, all 67 active pins are git
tracked, and the active local-import closure has zero omissions, including
dynamically imported/launched runners and token tooling. The v80 185/186
artifact remains rejected, its content report replays under the strict legacy
join, and the exact seed-8 configuration passes `--prepare-only` without an API
call.

---

## v84 — complete Writer coverage and quote-safe recovery (2026-08-17)

Policy ID: `generalized-card-v2-complete-writer-coverage-v84-20260817`.

A full replay of the paid v80 seed-8 artifact found 186 Writer tasks but only
185 rendered comments. S99 exhausted three attempts because its scheduled quote
opener said to copy the exact parent line while `parent_copy` correctly remained
a hard failure. The generator then persisted the shortened thread under
`policy=persist_valid_comments`, and the output audit still considered its
99.46% accepted share evaluable. That silently changes comment-pair metrics and
the sampled tree, so it is not a valid matched-thread artifact.

Changed:

- Quote openers now request a short exact markdown excerpt, never the whole
  parent. A `parent_copy` finding is waived only when the Planner explicitly
  assigned `opener_type=quote`, the quoted tokens are a strict excerpt of the
  visible parent, and at least six words of independent reply remain. Ordinary
  parent copying is still a hard failure.
- Exact Writer coverage is now a pre-persistence invariant. After bounded
  same-slot hard recovery, any missing, skipped, or malformed record raises a
  recoverable post error; the incomplete post never reaches atomic persistence.
  The default still performs no hidden whole-post retry or additional API spend.
- Output audit independently rejects any recorded post whose planned slots,
  generation records, generated records, and rendered comments are not exactly
  equal, even when `accepted_share` exceeds the old threshold. This also protects
  evaluation of historical artifacts.
- Removed the unreachable `omit_without_backfill` branch and corrected run
  metadata to describe bounded Planner schema recovery followed by hard failure.

Expected metric effect before a paid run: no direct claim of better content
quality. The required effect is measurement validity: every evaluated generated
thread has exactly the matched structural slots, so Self-BLEU, Self-BERTScore,
semantic cosine, length CV, depth, virality, story, emotion, and tone metrics are
not computed on a silently shortened sample. The shorter quote instruction may
also reduce parent-line repetition, but that is a secondary hypothesis.

Offline acceptance so far: the updated audit rejects the existing v80 artifact
at 185/186 despite `accepted_share=0.9946`; focused coverage/quote/audit tests
pass; the complete suite passes 266 tests; Ruff and the camera-product backend
self-test pass; all 72 source pins agree. The exact formal seed-8 command passed
`--prepare-only` under the v84 policy with no API calls, and its temporary run
directory was moved to Trash so the formal tag remains available.

---

## v83 — matched-text semantic isolation (2026-08-17)

Policy ID: `generalized-card-v2-matched-text-semantic-isolation-v83-20260817`.

The v82 completion audit was extended from final Prompt strings back through
every expander callback that receives an anonymous matched-real body. Three
remaining paths still derived semantic controls from evaluation wording:

- lexical first-person and uncertainty markers temporarily licensed those
  frames before Planner restoration;
- a long anonymous slot was labelled `story_rant`, regardless of its plan;
- lexical prefixes such as `side note`, `unrelated`, `FWIW`, and `BTW` assigned
  a `side_tangent` real-surface shape, and `!template` assigned template meaning.

Changed:

- Matched wording can no longer license first-person or uncertainty. The two
  dead regex classifiers were replaced by one explicit false boundary; the
  Planner's story/evidence/stance contract remains the sole authority.
- Real-surface inference now uses only deleted/moderator metadata, word scale,
  question punctuation, dominant link/quote form, and identifier typography.
  Its neutral structural labels are `long_turn`, `full_answer`, and
  `compact_identifier_turn`, never story/rant/tangent labels.
- The generalized anchor builder explicitly discards `real_body`; facts still
  come only from seed, generated parent, and Planner/domain claim controls.

Expected direction before a paid run: fewer hidden Planner conflicts and fewer
comments whose story, uncertainty, gratitude, or tangent behavior mirrors the
matched evaluation comment rather than the planned slot. The tree and length
signals remain identical. Formal metric result is pending.

Offline acceptance: semantic-marker isolation tests pass, the complete suite
passes 263 tests, Ruff and backend self-test pass, and all 72 pins agree.

---

## v82 — focused Planner discourse handoff (2026-08-17)

Policy ID: `generalized-card-v2-focused-discourse-contract-v82-20260817`.

The post-v81 completion audit found one remaining Planner→Writer break in the
default path. The focused Writer received the planned proposition plus dedicated
tone/story/affect controls, but not the planned comment function, payload form,
speaker role, voice, evidence basis, content angle, stance, detail, decision intent,
reply relation, or local exclusion. A planned rant, correction, datapoint, or
bare reaction could therefore fall back to the model's generic helpful answer.

Changed:

- Add one compact, deduplicated discourse contract to the focused Writer. It
  carries those fields once without restoring the old full prompt, static metric
  guidance, overlapping surface paraphrases, or bulky payload instructions.
- Add an end-to-end contract test from raw Planner JSON through normalization,
  matched-slot expansion, finalization, and focused Prompt rendering. A valid
  `rant + ranter + hard_disagree` slot must retain each planned control exactly
  once.
- Replace the shared surface-texture classifier on the generalized path. Matched
  comment typography may shape typography, but words such as `thanks` and
  `appreciate` may no longer assign gratitude tone or a
  `pure_acknowledgement`; social meaning remains Planner-owned.

Predicted direction before a paid run: fewer generic customer-service/helpful
turns; more faithful rants, corrections, questions, datapoints, and terse social
moves; greater lexical and emotional variety, moving Self-BLEU/Self-BERTScore
and emotion-related rows toward real data. Story allocation and tree structure
are unchanged. Formal result: pending a new artifact and multi-thread evaluation.

Offline acceptance: the focused contract and matched-text isolation tests pass,
prompt size remains below the existing focused/full ratio gate, and the complete
suite passes 262 tests.

---

## v81 — joint story/affect handoff and prompt-residue removal (2026-08-17)

Policy ID: `generalized-card-v2-joint-story-affect-handoff-v81-20260817`.
The implementation commit is the git entry that adds this section; every paid
artifact additionally stores its exact source/config snapshot.

v80 showed that making the Writer instruction stronger was not enough. The
direct-reply Planner saw fixed social labels as prose but did not return them in
its schema, 61 slots used firsthand evidence against a 17-story quota, and 104
short replies copied the `development_plan` schema example into a real plan.
Post-parse normalization then hid bad plans by rewriting them to one repeated
gratitude sentence or to `soft_helpful`.

Changed:

- Story is now a bidirectional Planner invariant. `no_story` rejects firsthand
  evidence and personal-story payloads; a story slot requires firsthand,
  personal-datapoint semantics. Unresolved story/surface/long-form contracts
  stop before the Writer instead of being logged and persisted.
- Direct replies receive story, tone, affect, and opener controls as structured
  per-slot contracts. A no-story row cannot choose the explicitly narrative
  `corroborating_datapoint` route.
- Short slots deterministically clear any copied development-plan prose. Both
  Planner schemas now use literal `none` and explicitly require it below the
  long-form threshold. Root and direct-reply prompts use the same dynamic beat
  capacity function as validation; the conflicting 35-word/16-beat prose was
  removed.
- Removed semantic post-parse rewrites. Gratitude/relief and substantive-slot
  conflicts go through targeted Planner repair; no shared canned semantic move
  and no automatic `soft_helpful` conversion remain.
- Tone/affect marginals are paired jointly before planning. On the frozen v80
  seed-8 template the new schedule assigned every label while reducing
  `approval+impolite` 10→0 and `neutral-affect+polite` 27→2.
- The focused Writer renders the tone definition once, gives neutral affect a
  non-conflicting instruction, and omits known schema defaults from its
  semantic ledger. Impolite and amusement contracts explicitly permit
  non-targeted profanity and natural laughter tokens, respectively, without
  requiring a fixed phrase.
- First-pass distribution resampling is disabled at the public CLI. Repetition,
  Self-BLEU, Self-BERTScore, and semantic cosine are collection diagnostics;
  only non-persistable Writer failures retain bounded recovery.

Offline acceptance before the first run:

- v80 replay: 104 short development residues removed; 59 latent story-contract
  conflicts detected rather than passed through.
- the v80 template's 186 tone/affect assignments remain complete with zero
  unassigned labels and zero story/social-close collisions.
- expected direction: story probability down toward the frozen template;
  emotion realization and entropy stability improve; shared prompt scaffolding,
  Self-BLEU, Self-BERTScore and helpful/explainer register decrease. Structure
  is unchanged because every matched slot and parent edge is preserved.
- complete test suite: 259 passed; backend self-test passed; 72 pinned source
  files report zero missing and zero drifted entries.

Formal acceptance still requires a multi-thread matched evaluation. An n=1 run
is only a content and contract diagnostic.

## v80 — coherent Planner social contracts (2026-08-16)

Tag: pending; do not start with a paid run.

Measured diagnosis on the existing v79 seed-8 artifact:

- Only 17 comments were assigned a story mode, and they contributed about 25%
  of the thread's total StorySeeker probability. Among 167 `no_story` comments,
  25 were still classified as stories. The highest-scoring rows retained
  `payload_type=personal_story` or a temporal firsthand plan after the schedule
  overwrote only `story_mode`.
- Of 46 `polite` slots, only 6 realized as polite; 27 realized as impolite. The
  Planner prompt already requires an agreeing personal datapoint, reaction, or
  positive verdict, but mismatching roles/functions survived because the
  post-normalization quality gate did not check that contract.
- On 40 existing comments (780 unordered pairs), changing only curly apostrophes
  to ASCII moved Self-BERTScore 0.52947 -> 0.52381. Curly double quotes had
  effectively no effect. This is a real but secondary global signature, not an
  explanation of the full 0.034 v79-vs-real gap.

Changed before any API call:

- Plan-quality validation now rejects `no_story + personal_story` and incoherent
  polite role/stance/function combinations, so targeted Planner repair operates
  on the whole semantic contract instead of relabeling one field after planning.
- Every `no_story` Writer path now explicitly forbids a temporal event sequence
  while still allowing one firsthand observation.
- Polite guidance now follows the observed real-discussion cues: ordinary
  hedging and brief thanks are allowed, an emotional endpoint is required, and
  repeated abstract decision-framing is discouraged. The refuted generated-data
  length hint was removed.
- Direct-reply planning now exposes sibling coverage, including already committed
  sibling delta types and novelty anchors.
- Both interventions have explicit ablations:
  `--social-contract-coherence off` and
  `--reply-sibling-visibility off` restore the pre-v80 arms, and both fields are
  part of the recorded and resume-checked experiment identity.
- Resume/extension/upgrade checks share one experiment-field list that includes
  every behavior flag. The prior implementation wrote those flags to the run
  record but omitted them from lineage comparison.
- Removed proven-unreferenced helpers and stale tone-example rewrites; generalized
  Planner prose no longer assumes every domain is equipment/products.

Predicted direction: planned-social-contract realization above v79's 59.2%,
`no_story` StorySeeker mass down materially, polite realization above 13%, and no
change to matched tree structure. Validate plan-contract counts and prompt
snapshots before a paid run; evaluate p-values only after a comparable multi-seed
run.

## v68-v79 provenance correction (recorded 2026-08-16)

The narrative log previously stopped at v67 even though run artifacts and the
historical policy set continued through v76. The durable record is:

- v68: domain-claim/entity generalization.
- v69: scheduled opener grammar; evaluated on ten threads at 8/12, with the
  cancellation caveats described in the handoff.
- v70: domain-claim field survival; the recorded smoke was not fully evaluated.
- v71: Planner-owned reply move and single-parent exclusion; ten-thread result
  4/12.
- `v72_noclaim` was an experiment tag, not a policy version: its run config
  correctly retained the v71 policy string. It scored 7/12.
- v73: affirmative affect and uncapped anonymous slot shape; 8/12.
- v74: focused Writer prompt; 7/12.
- v75: Writer realizes the Planner move in its own words; the evaluated repeat
  scored 4/12.
- v76: own-fact-license experiment arms.
- v77, v78, and v79 changed repetition/recovery behavior but incorrectly reused
  the v76 policy string. They are retained as artifact tags, not claimed as
  reproducible policy releases, and must not be ranked from their one-thread
  12/12 p-value output.

---

## v64 — calibrated tone registers and length scale (2026-08-13)

Tag: `generalized_card_camera_gpt54_v64_tone_smoke10_20260813_v1` (10 threads, 521 comments)

Changed:
- `TONE_CLASSES` extended to the classifier's full four-way partition. The
  reported metrics stay polite/impolite/neutral, but planning over three classes
  had renormalized the missing `somewhat_polite` mass onto the reported three.
- `TONE_DEFINITIONS` rewritten from measurements on 11,817 evaluation-excluded
  camera comments rather than a generic notion of manners.
- `_tone_cost` reversed: polite now routes to longer slots, matching the observed
  distribution, instead of the shortest compatible slot.
- Writer's blanket ban on acknowledgement and first-person framing scoped so it
  no longer cancels the tone control it sits next to.
- `allow_first_person_frame` no longer forced off for a no-story polite slot.
- Beat budget moved from one beat per 80 words to one per 35.

Result: **6/12 pass**, down from v34's 8/12.
- Improved: neutral_rate PARTIAL→PASS, semantic_cosine 0.21→0.31,
  avg_depth and structural_virality to p=1.00, planner→writer tone contract
  fidelity 40.1%→54.7%, somewhat_polite rate 0.269→0.124 against a real 0.125.
- Regressed: self_bleu_4 PASS→FAIL, self_bertscore PASS→PARTIAL,
  emotion_entropy PASS→FAIL, impolite_rate worse.
- polite_rate did not move (0.068→0.048 against a real 0.297).

Diagnosis of the regression: the tone text prescribed sentence structure
("Lead with the disagreement"), which gave every same-register comment a shared
entry route and inflated within-thread lexical and semantic similarity. The beat
change had almost no effect because it was aimed at the wrong constraint.

## v65 — tone-compatible reply increments and reply development plans (2026-08-13)

Tag: `generalized_card_camera_gpt54_v65_bigthread_seed78_20260813_v1`
(1 thread, seed_index 78, 197 comments, $1.95, 24 min)

Hypothesis: polite could not be realized because the Planner's schema could not
express a warm reply at all. `REPLY_DELTA_TYPES` held seven values, six of them
inherently critical, so 92% of polite-planned slots were planned as
`speaker_role=advisor` delivering a technical adjudication — content no tone
control can turn warm.

Changed:
- Added `corroborating_datapoint`, `useful_extension`, and
  `endorsement_with_reason`, and gated the allowed set per tone register.
- Propagated the new vocabulary to every consumer: the direct-reply planner
  schema and rules, the root planner schema and rules, the reply-delta contract
  block, the Writer's `realization_by_type` route lock, and
  `planning_quality.reply_increment_problem`, which had been rejecting the new
  types as "generic agreement".
- Joint tone/affect assignment, so a polite slot can no longer receive
  disapproval, anger, or disappointment.
- `development_plan` added to the direct-reply planner, which had omitted the
  field entirely. Every long slot at depth ≥ 1 (33 of 77) was receiving no
  development guidance and was realized at ~0.72x its matched length.
- Per-slot beat requirements now stated on each row in both planners.
- All sentence-structure prescriptions removed from the tone guidance.

Predicted direction: polite fidelity up from 6%; advisor share down from 72%;
long-slot ratio up from 0.72; self_bleu_4 and emotion_entropy recovered to at
least v34 levels now that the shared entry routes are gone.

Result: every predicted plan-level change landed.

| | v64 | v65 |
|---|---:|---:|
| advisor share of slots | 72% | 9% |
| stance=agree | 14% of polite slots | 64% of all slots |
| supportive delta types | absent | 57% of replies |
| long slots with a development_plan | 30% | 95% |
| ... of those at depth >= 1 | 0 of 33 | 12 of 12 |
| long-slot length ratio (100+ words) | 0.72 | 0.87 |
| polite contract fidelity | 6% | 14% |
| generated polite_rate | 0.048 | 0.117 |

Caveat: v65 ran one thread at seed_index 78 while v64 ran seeds 0-9, so the
realization numbers are not a clean A/B. The plan-level counts are unambiguous
because they measure the fields that were changed. Seed 78 is now the fixed
iteration thread so later versions compare against this row directly.

Remaining defect: 64% of polite-planned slots are still classified impolite.
Inspecting the text separates the two groups cleanly by the *valence of the
concrete object*, not by stance, role, or length (misses average 80 words, hits
57):

- Miss: "that's a genuinely awkward spot", "the body stopped seeming so alien",
  "if the body doesn't put the buttons where your fingers expect, it never
  really settles in", "what broke for me was...". The slot agrees with its
  parent but corroborates a *friction*.
- Hit: "one thing that helped me was...", "it genuinely made the body feel less
  intimidating", "that was the bit that clicked for me", "Appreciate that".
  The concrete object is a *resolution or benefit*.

`corroborating_datapoint` accounts for 28 of the 78 misses: the Writer confirms
the parent's difficulty rather than a positive outcome. The supportive delta
definitions are valence-neutral, so a warm register attached to a
friction-shaped anchor still reads as complaint.

Deferred: making the supportive increments valence-bearing for polite slots.
Politeness was deprioritized in favour of diversity and emotion.

## v66 — held-out entity inventory, unseeded route lock, route ledger, beat rate (2026-08-13)

Tag: pending. Same seed as v65 (seed_index 78) so the comparison is a real A/B.

Priorities reset: diversity (`self_bleu_4`, `self_bertscore_mean_f1`,
`semantic_mean_cosine`) and `emotion_entropy` matter most; politeness least.
Against that ordering, the v65 thread stood at:

| metric | real | v65 | verdict |
|---|---:|---:|---|
| semantic_mean_cosine | 0.2825 | 0.2490 | already past real |
| emotion_entropy | 1.9394 | 2.1037 | already past real |
| avg_depth / structural_virality | 2.244 / 3.971 | 2.250 / 3.971 | matched |
| **self_bleu_4** | 0.0264 | 0.0338 | too repetitive |
| **self_bertscore_mean_f1** | 0.5026 | 0.5188 | too similar |
| **length_cv** | 0.9456 | 0.8515 | too narrow |

Reading the matched real thread rather than only its statistics produced the
main finding. Over the same 197 slots:

| | real | v65 |
|---|---:|---:|
| repeated 4-gram share | 0.0545 | 0.0790 |
| **distinct camera models named** | **117** | **23** |
| most frequent model's share of mentions | 0.03 | 0.29 |
| no-end-punctuation share | 0.183 | 0.091 |

The Writer's rule "named entities may appear only when visible in the discussion
or in the visible factual anchors" is correct for claims *about* the seed, but it
also means all 197 comments can only ever name the two or three products the seed
mentions. `the sony a7 iv` appeared 9 times, `sony a7` 22, `the a7` 19. Real
commenters name their own gear instead, which is what spreads entity mass.

Changed:
- **E1** New `entity_inventory` module and profile field (schema 10): equipment
  designators learned by brand adjacency over the 424 evaluation-excluded
  threads, then counted in every form. 63 clean designators for camera. Offered
  to the Writer only on slots whose plan already licenses first-person
  experience, rotated by slot so mass spreads, excluding anything already visible
  in the slot, and licensed strictly as the speaker's own gear.
- **D1** `_semantic_route_lock` said "make this the part that changes the
  parent"; the Writer echoed "that's the part that…" 18 times. Reworded so the
  scaffolding no longer contains the construction it asks for.
- **D5** `used_sentence_routes` ranked by recency and carried no counts, so the
  entrenched routes were pushed out of the ledger by recent one-offs. Now ranked
  by reuse with counts attached.
- **L1** `WORDS_PER_REALIZED_BEAT` 35 → 21 and the cap 16 → 24, from the measured
  realized rate (246/12, 179/8, 134/6 words per beat).

Also recorded: 190 of 197 v65 prompts already listed `that s the part` as a
repeated four-gram and 23 comments used it anyway, and the running self-BLEU
plateaued at 0.0335 by comment #62 while the calibrated band only contracted
enough to flag it at #182. Prompt-level exclusion lists do not work here, and the
existing guard cannot detect the problem while it is still fixable. If v66 does
not close the self-BLEU gap, those two are the next targets rather than more
prompt wording.

Predicted direction: distinct models 23 → 50+ with top-model share well under
0.29; repeated 4-gram share moving from 0.0790 toward the real 0.0545;
`self_bleu_4` gap and `self_bertscore` gap both shrinking; long-slot ratio from
0.87 toward 1.0 and `length_cv` from 0.8515 toward 0.9456; `semantic_cosine` and
`emotion_entropy` holding.

Tag: `generalized_card_camera_gpt54_v66_entity_seed78_20260813_v1`
(same seed as v65, 195 comments, $1.99, 23 min)

Result: **the changes did not land.** Every predicted magnitude missed.

| | real | v65 | v66 | predicted |
|---|---:|---:|---:|---|
| repeated 4-gram share | 0.0545 | 0.0790 | 0.0755 | toward 0.0545 |
| distinct models | 117 | 23 | **27** | 50+ |
| top model share | 0.032 | 0.289 | **0.308** | well under 0.29 |
| distinct 3-word openers | 0.888 | 0.772 | 0.810 | — |
| "that's the part/bit" | 0 | 24 | **19** | down |
| long-slot ratio | — | 0.87 | **0.88** | toward 1.0 |
| thread word CV | 0.943 | 0.856 | **0.823** | toward 0.946 |

Why E1 missed, measured the same way as the earlier exclusion-list check: 129 of
195 prompts (66%) offered the equipment shortlist, and only **21 of those 129
(16%)** named an offered item. The Writer ignores an optional affordance in this
prompt exactly as it ignores an exclusion list.

Why D1 only half worked: `that's the part` fell 18 → 9, but `that's the bit` rose
6 → 10, and a new frame `the rest of the` appeared 11 times. Removing the seeded
wording made the model reach for a synonym; the underlying rhetorical act was
untouched.

Why L1 missed: the Planner supplied 7.0 of 7.8 requested beats, so planning
complied, but doubling the beat budget produced no extra length (0.87 → 0.88) and
the CV fell. One 282-word slot collapsed to 92 words despite 12 planned beats.

## Cross-cutting conclusion after v64-v66

Compliance depends on the *kind* of control, not on its wording:

| control | kind | compliance |
|---|---|---|
| `tone_target=impolite` | planned categorical field | 86% |
| `development_plan` present vs absent | planned field, presence | 0.76 → 0.88 ratio |
| `tone_target=polite` | planned categorical field | 14% |
| equipment shortlist | prompt affordance | 16% |
| beat count doubled | planned field, magnitude | no effect |
| repeated-n-gram exclusions | prompt rule | ~0 (23 violations after being shown) |
| opener/route exclusions | prompt rule | ~0 |

The Writer follows *what kind of thing to say*. It does not follow *how much*,
*how not to*, or *which optional resource to use*. The writer prompt averages
23,000 characters and 84 bulleted rules to produce a ~270-character comment, so
nothing in the rule mass is being attended to.

**Therefore: adding or rewording prompt text cannot fix diversity or length.**
Three runs now support that. The remaining levers are structural:

1. Radical prompt reduction — cut the Writer prompt from ~23k to ~3k characters
   containing only the plan and the visible context. Untested, cheapest, and it
   attacks the common cause of every non-compliance above.
2. Act on the guard that already exists. `lexical_overlap_problem` computes the
   evaluator's exact self-BLEU per candidate against a held-out band;
   `writer_local_repair_rounds` is currently 0. v19, the historical 11/12 run,
   had it at 2. This touches the `AGENTS.md` prohibition on best-of-N for a
   distribution metric and needs an explicit decision.
3. Deterministic non-LLM surface transforms after generation. Effective but it is
   post-hoc text editing, which is what this project set out to avoid.

Also fixed for the future: the running self-BLEU band contracts with progress, so
a uniformly slightly-too-repetitive thread is only flagged at ~92% completion.
Any variant of lever 2 must compare against the final target from early on.

## v67 — bounded thread blackboard (2026-08-13)

Tag: pending. Same seed as v65 and v66 (seed_index 78).

This measures the cause of the three preceding failures rather than another
control. Section sizes in the largest v66 writer prompt (67,284 characters):

| section | chars | share |
|---|---:|---:|
| **Structured thread blackboard** | **43,728** | **65.0%** |
| Hard rules | 5,171 | 7.7% |
| Per-slot instructions | 4,583 | 6.8% |
| Planner intent | 3,023 | 4.5% |
| One-shot semantic contract | 2,802 | 4.2% |
| Semantic route lock | 777 | 1.2% |
| Visible discussion | 576 | 0.9% |
| equipment shortlist | 341 | 0.5% |

Inside the blackboard, `semantic_coverage_entries` alone was 30,148 characters
across 140 entries: 69% of the blackboard and 45% of the whole prompt. Its
purpose is to hold down semantic repetition, and `semantic_mean_cosine` is the
one diversity metric already past real — so the largest block in the prompt was
over-serving the metric that needed nothing while crowding out everything else.

Every ledger cap scaled with thread length, so the blackboard grew without bound:

| slot | prompt | blackboard share |
|---:|---:|---:|
| 0 | 10,839 | 16% |
| 20 | 29,342 | 50% |
| 60 | 35,229 | 71% |
| 140 | 53,492 | **81%** |

By comment 140 the slot's own assignment was 19% of its prompt. That is why a
341-character affordance drew 16% uptake, why 190 prompts listing a banned
four-gram produced 23 violations, and why doubling the beat budget did nothing.

Changed:
- Every ledger capped at a constant instead of scaling with thread length.
- `semantic_coverage_entries` reduced to `move` and `boundary` per entry, capped
  at 24, and ranked by lexical relevance to the current slot rather than
  recency, so what survives the cap is what this slot could actually duplicate.
- `used_sentence_routes` capped at 20 (already frequency-ranked from v66).
- Earlier-comment tail 12 → 8 entries with 5 tags instead of 11.
- The short-line exclusion ledger is kept complete only for slots that could
  reproduce a short line; a long slot cannot, and it was pure prompt mass there.
  This preserves the exact-duplicate invariant exactly where it applies.
- `_tone_discourse_guidance_block` renders the assigned register and the one it
  drifts into, not all four.

Verified offline against the v66 tasks, no API call:

| | before | after |
|---|---:|---:|
| blackboard mean / max | 31,544 / 44,230 | **8,230 / 9,667** |
| writer prompt mean / max | 46,166 / 67,284 | **22,852 / 31,903** |
| plan share at slot 60 | 29% | 54% |
| plan share at slot 140 | 19% | **53%** |

Predicted direction: this is a compliance fix, so the controls that previously
missed should move without being re-worded — equipment uptake above 16%,
`that's the part/bit` below 19, long-slot ratio above 0.88, `length_cv` above
0.823, and `self_bleu_4` below v66's 0.0338. `semantic_mean_cosine` is the
metric most at risk, since its ledger shrank the most; it had headroom
(0.249 against a real 0.283) and is expected to rise but stay under real.

Tag: `generalized_card_camera_gpt54_v67_bounded_seed78_20260813_v1`
(seed_index 78, 197 comments, $1.48, 20 min. One earlier attempt was killed
externally at planning slot 131; a single-post run persists atomically at post
completion, so that attempt's $0.30 of planning was lost and it restarted.)

Result: **the compliance hypothesis holds.** Controls that had missed for two
versions moved without a single word of their wording changing.

| | real | v65 | v66 | v67 |
|---|---:|---:|---:|---:|
| writer prompt mean | — | — | 46,269 | **22,115** |
| writer prompt max | — | — | 67,284 | **31,009** |
| equipment uptake | — | — | 16% | **26%** |
| long-slot ratio | — | 0.93 | 0.88 | **0.99** |
| distinct models named | 117 | 23 | 27 | **39** |
| most frequent model's share | 0.032 | 0.289 | 0.308 | **0.175** |
| repeated 4-gram share | 0.0545 | 0.0790 | 0.0755 | **0.0707** |
| thread word CV | 0.946 | 0.856 | 0.823 | 0.846 |
| longest generated comment | 413 | 246 | 304 | 302 |

Long-slot ratio here is the mean of per-slot `generated/real`. An earlier note
reported 0.87 for v65 using the ratio of bucket means, which is a different
statistic; per-slot means are used throughout this table.

Still open after v67:
- `that's the part/bit` is 21, against 24 and 19 — flat. Shrinking the prompt did
  not touch it, because the shared *rhetorical act* is what produces it, not
  prompt pressure. This is the deferred D4 fix: schedule a `rhetorical_form`
  across slots the way tone, affect, and story are scheduled, instead of letting
  every slot invent its own `opening_style`.
- Distinct models 39 against a real 117. Uptake tripled the entity spread but the
  affordance is still optional. Making the equipment a *planned field* the
  Planner writes into the slot contract should close more of it, since planned
  categorical fields are the only control type this Writer reliably follows.
- No-end-punctuation share 0.066 against a real 0.183. Real comments drop final
  punctuation about one time in five; generated output is too polished.
  `surface_texture=no_punct_fragment` exists and is under-scheduled.
- Thread word CV 0.846 against 0.946, with the top still truncated at 302 words
  against a real 413.

Operational note: `--posts-per-run` persists at post granularity, so an
interrupted single-post run of a 197-comment thread loses all of its spend. Worth
changing to slot-level atomicity before many more long-thread iterations.

### v67 on the comparable 10-thread pool

Tag: `generalized_card_camera_gpt54_v67_smoke10_20260813_v1`
(10 threads, 520 comments, coverage 1.01, $2.97, 36 min)

**v67 did not improve the pass count: 5/12 against v64's 6/12.** Both runs have
coverage 1.00, so this is the only valid comparison available.

| metric | v64 p | v67 p | v64 \|d\| | v67 \|d\| | real noise p90 \|d\| | v67 closer on |
|---|---:|---:|---:|---:|---:|---:|
| semantic_mean_cosine | 0.307 | **0.791** | 0.28 | **0.08** | 0.48 | — |
| emotion_entropy | 0.016 | 0.049 | 0.65 | 0.53 | 0.44 | — |
| self_bertscore_mean_f1 | 0.017 | 0.002 | 0.64 | 0.82 | 0.44 | **7/10** |
| self_bleu_4 | 0.014 | 0.011 | 0.66 | 0.68 | 0.40 | 5/10 |
| length_cv | 0.017 | 0.021 | 0.64 | 0.62 | 0.46 | 4/10, **0/2 large** |
| hard_disagree_rate | 0.162 | 0.023 | 0.38 | 0.61 | 0.44 | — |
| mean_story_probability | 0.850 | 0.345 | 0.06 | **0.26** | 0.43 | 5/10, **0/2 large** |
| avg_depth | 1.000 | 0.909 | 0.01 | 0.04 | 0.44 | — |
| structural_virality | 1.000 | 0.970 | 0.01 | 0.02 | 0.44 | — |

**Cliff's delta saturates under systematic bias, so it is the wrong progress
metric.** `self_bertscore` shows this exactly: its delta rose 0.64 to 0.82 while
the per-thread magnitude improved on 7 of 10 threads. On thread 38jlgz the gap
went from -0.0156 to +0.0015 — far smaller, but it flipped to positive, and once
every generated thread sits on the same side of real, delta approaches 1
regardless of how small the gaps are. Use delta to predict whether a test will
pass at a given N; use the mean absolute gap or Wasserstein to track progress.

Attributing the three changes:
- **Bounded blackboard: keep.** `semantic_mean_cosine` delta 0.28 to 0.08,
  `emotion_entropy` 0.65 to 0.53, `self_bertscore` magnitude better on 7/10.
- **L1, beat divisor 35 to 21: revert or soften.** It raised the long-slot length
  ratio to 0.99 but `length_cv` got worse on both large threads and on seed 78,
  because lengthening mid-size comments compresses the spread the metric measures.
- **E1, equipment plus first-person licensing: gate it.**
  `mean_story_probability` delta went 0.06 to 0.26 with gaps up to +0.19, worst on
  the large threads. Offering own-gear anecdotes on `no_story` slots produces
  content StorySeeker scores as narrative. The offer should require
  `story_mode != no_story` or `evidence_mode == firsthand_experience` rather than
  the bare `allow_first_person_frame` flag that v64 set true for polite slots.

`hard_disagree_rate` degraded across v65-v67 (0.070, then delta 0.38 to 0.61) and
has no attributed cause yet.
