# CARD Core Parity Contract

Policy version: `card-paper-v37-domain-neutral-profile-v4-20260807`

Revision policy: `generalized-card-revision-v7-bidirectional-exact-no-fail-20260811`

Generation policy: `generalized-card-v2-one-shot-content-depth-v34-20260812`

The revision policy binds the CARD-style revision chain, deviation-driven
coverage, deterministic controller memory, and native history replay.
Generator lineage is recorded separately.
The default `generalized-v2` profile pins the stronger domain-neutral generator
at `scripts/sampling_generator/run_sampled_reddit_generator.py`; the explicit
`card-snapshot` profile pins the July 4 historical snapshot. Artifacts cannot be
silently mixed or relabeled across these generator profiles.

`generalized-v2` is CARD-derived, not byte-identical to the July 4 snapshot.
Six of the 22 audited generation functions contain later changes for tone
targeting, reply-tree repair, lexical writer guards, and richer global memory;
the other 16 are AST-identical. The adapter preserves all functions from the
selected backend and separately audits every replacement boundary. Exact
historical-snapshot claims require the explicit `card-snapshot` profile.

This contract distinguishes the shared CARD algorithm from domain adaptation
and from later generalized extensions. `generalized_card/core_contract.py`
pins the shared source hashes, generalized adapters, and orchestration entry
points and fails before an API call if any of them drift.
`generalized_card/backend.py` additionally compares every top-level function
before and after adapter configuration. Any changed function outside the
declared domain-boundary allowlist is a fatal error.

| Area | Shared CARD behavior | Generalized change allowed |
|---|---|---|
| Thread planning | Shared structural sampling, comment-count matching, tree planning, role/payload/tone/story/length controls, balancing and pressure | Evaluation-seed comments provide structure and anonymous word-count/surface signals only. Viewpoint text is retrieved from evaluation-excluded reference threads, then abstracted by the private Planner. Twelve domain-neutral decision lenses replace topic-phrase labels. |
| Plan validation | Planner JSON normalization and retry path | `reference_id` is retained; semantic collisions, claim reuse, invalid lenses, and perspective concentration trigger structured Planner-only repair before Writer calls. Returned IDs are restricted to the requested global slot range. Enum-only mistakes such as a branch ID in `perspective_id` are deterministically normalized to the schema fallback and audited without changing semantic content. Each failed semantic slot is regenerated independently, while healthy slots remain fixed; every local replacement is rescored against the complete thread ledger, and the collision-rate threshold uses the full thread denominator. Reference reuse is recorded as a warning rather than a repair target. |
| Comment writing | Shared context aperture/dropout/jitter, blackboard, low-information path, claim/opening/template budgets, role/payload/function/voice/texture balancing, and seed/planner factual anchors; dropout/jitter use the recorded CARD run values `0.42/0.32` | CARD surface balancing runs under invariants that preserve the domain-neutral semantic move, claim, perspective, tree slot and matched length signal. For anonymous long slots, the Planner supplies a connected development sequence that is retained through normalization, task creation, Writer prompting, and provenance records. Used sentence-entry routes and prior semantic contributions cover the full generated thread within bounded prompt memory. Neither matched-real nor reference-comment text is shown to the Writer. |
| Self-BLEU target selection | Quantile-gap profile selection and every matched-real deviation above the profile threshold | No controller override |
| Self-BLEU local search | Profile rewrite share, rewrite budget, ranked comment pool, best-of-N width, exact thread insertion, gap and overshoot gates | No algorithm override |
| Tone target selection | Four-metric gap vector and CARD stage selection | Domain-neutral tone and stance prompt wording |
| Tone local search | Per-thread rewrite share/budget, candidate pool, best-of-N scoring, stance and Self-BLEU protection | Domain entity/number validation |
| Reviser prompts | Full CARD context, profile/stage instructions, candidate slate, exact contribution fields, prior rewrites, failure feedback and thread phrase inventory | Static finance wording is replaced; Self-BLEU receives an additional 1/2-gram diagnostic |
| Story and structure | Supported-direction deviating threads, deviation-derived text/move capacity, exact local and full-collection gates | Domain story wording and protected entities |
| Round controller | Full cleanup, full rescoring, matched-seed evaluation, target improvement, protected metrics, accept/rollback | Paths, seed pool and real-score inputs |
| Revision memory | Deterministic round summaries, same-input failed-strategy avoidance, bounded prompt feedback, immutable per-round snapshots | Domain labels in prompts only |
| Resume | Accepted rounds chain forward; rejected rounds roll back; partial thread work stays under the same prefix | Recoverable Planner/output failures retry the same unfinished post and are appended to `_generation_failures/post_retries.jsonl`; completed posts, cumulative cost, and elapsed time remain unchanged. Permanent configuration/authentication errors and user interruption still stop safely. |

## Coverage Rule

`max_threads=0` remains in effect, so every thread selected by the matched-real
gap rule is visited. In generalized deviation-driven mode, the ranked comment
pool spans every eligible comment and the rewrite/move capacity is derived from
the observed matched-real gap and the available thread content. Exact local
gap, undershoot, factual-preservation and target-reached gates determine how
many edits are retained. The best-of-N width for one selected comment remains
finite because it defines the candidate search slate; it does not omit a
deviating thread or impose a fixed number of selected comments.

## Semantic Source Isolation

The historical credit-card generator abstracted the matched real comment text
corresponding to each seed. The cross-domain protocol cannot expose evaluation
answers to the Planner. It therefore freezes a reference-viewpoint bank from
threads outside the complete seed pool, verifies zero source-post overlap, and
retrieves a source-diverse set for each seed. The Planner sees those non-test
comments and emits only abstract slot controls. The Writer sees the abstract
controls, visible seed/parent content, and a metadata-only generated-thread
blackboard. Planner repair history is written to `logs/planning_quality.jsonl`.
Output auditing checks semantic-plan concentration and copying against both the
matched evaluation thread and the non-test reference bank.

## First-Pass Distribution Contract

The V34 generation policy retains V33's CARD task-distribution controller before
the Writer while preserving generalized semantic and structural invariants.
The controller consumes the configured advisor, question, micro, social,
gratitude and tone shares; those command-line values are no longer diagnostic
no-ops. It may change surface role, payload, function, voice, utterance shape,
texture and tone shape. It may not change the slot ID, tree anchor, matched
word-count signal, length bucket, semantic move, local topic, stance, evidence,
claim, perspective or domain intent. A substantive matched slot cannot be
collapsed into a reaction or social-noise payload.

Story and affect are allocated after this surface pass. Non-neutral affect uses
a deterministic, domain-neutral realization channel and clearer label-specific
expression so that labels do not all collapse into neutral wording. Length
remains a soft matched-real signal: long slots receive a Planner-authored set of
connected semantic beats, but generation is not accepted or rejected by a hard
word-count interval. Lexical route memory is also prompt-side guidance, not a
regex content classifier or metric-driven best-of-N loop.

## Generalized Extensions

Self-BERT, exact semantic cosine, length CV and emotion-entropy controllers are
additional metric stages. They follow the same full-collection acceptance and
rollback pattern, but they are not part of the historical three-round
Self-BLEU plus seven-stage tone snapshot. Results should identify these stages
as generalized CARD extensions rather than historical core stages.
`run_full_revise.py` therefore defaults to `--revision-profile card-core`,
which runs Self-BLEU followed by Tone. The explicit
`--revision-profile extended` option adds the extension stages without changing
the core stage implementations. Its seven-round budget is global. CARD metric
order breaks ties, while coverage-first scheduling gives each still-required,
supported stage one proposal before any stage receives a second. Story and
Structure are scheduled and accounted for separately.

The extensions do not claim a safe repair in every mathematical direction.
Self-BLEU, Self-BERT and story repair generated excess; structure repair raises
depth/virality deficits. Semantic cosine, length CV, emotion entropy and the
coupled tone controller support both directions. Unsupported reverse
directions are logged as monitor-only rather than sent to an ineffective
reviser.

## Invalidated Camera Revision Prefix

The camera prefixes `diversity_2bbb235140_a00` and
`diversity_2f770d79a4_a00` were produced by obsolete adapters. The former
changed CARD's local coverage policy; the latter overrode profile/progress/
protected decisions and used reduced prompts. Their intermediate rounds remain
available for audit but must not be reported as core-parity results. The policy
version is part of new lineage hashes, so corrected runs use a new revision
workspace and cannot silently resume or mix with those artifacts.

The generation run `generalized_card_camera_gpt54_pw50_core_v4` predates formal
generator policy versioning. Its content and metrics remain valid under the
label `legacy generalized Planner-Writer`. The revision workspace
`generalized_card_camera_gpt54_core_v4_newselfloop_v1` preserves that source
label but contains an obsolete revision policy. It must not be described as
newly generated by either pinned profile or as revision-v5 output.
