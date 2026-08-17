# v85 worklog — current-path control audit

Date: 2026-08-17

Policy: `generalized-card-v2-auditable-plan-controls-v85-20260817`

## Scope and evidence rule

This pass did not borrow historical implementations as design authority. Git
and old artifacts remain useful for reproduction and empirical comparison, but
each code decision below was checked against the current CLI, backend monkey
patches, shared generator, Prompt renderers, persistence model, output audit,
tests, and current v80 seed-8 artifact.

## Findings

1. `tone_overlay_slot` and `tone_overlay_instruction` have no assignment in the
   current source. All 186 v80 seed-8 task records also contain empty overlay
   values. Older records can contain values, so the dataclass and persistence
   fields must remain to deserialize them; current Writer inputs need not read
   them.
2. `constructive_polite_helpful` is not a current tone class and has no
   assignment. Its two finalizer disjuncts were unreachable.
3. `projected_metric` had zero source, CLI, or test callers. The active reviser
   uses the batched `project_candidate_metrics`, which produces the same kinds
   of exact projections more efficiently.
4. Fixed story/tone/affect/opener assignments are applied before
   `evaluate_plan_batch`. Therefore the current social-contract check sees the
   final labels and catches incompatible payload/function/evidence choices.
   The defect was lost telemetry: the `events` parameter already existed but
   the backend did not pass it.
5. Perspective and branch metadata are deterministically canonicalized before
   every evaluation. `invalid_perspective` and `branch_route_conflict` could not
   fire on the active path. `perspective_concentration` could fire, but a
   slot-local retry could not change topology-owned perspective IDs.
6. A Prompt-only concern about `very_long + no_story + no anchors` was tested
   rather than assumed. It occurred in 0/186 seed-8 slots: all 17 very-long
   slots were stories, and all 44 long/no-story slots had visible anchors.
   Slots above 100 words also require a validated Planner development plan.

## Changes

- Removed current Writer reads and Prompt text for the retired overlay; retained
  only compatibility fields in `CommentTask` and persistence.
- Removed the old unreachable tone-label conditions and unused scalar metric
  projection function.
- Added `initial_slot_contract_overrides` and `slot_contract_overrides` to every
  plan-quality report.
- Made perspective concentration non-repairable while retaining it in reports
  and strict warnings.
- Removed dead invalid-perspective/branch-route evaluation and their unused
  public parameters and score weights.
- Rewrote the TODO's old B-item list to reflect current code instead of carrying
  disproved hypotheses forward.

## Prompt review

A rendered current focused Writer prompt was 3,454 characters for a
tone+affect+story+rant contract. It contained zero repeated non-trivial lines;
tone target, story realization, and affect role each appeared once. Both focused
and full prompts contain no tone-overlay rule after this change. This confirms
the simplification without claiming that one synthetic fixture predicts model
quality.

## Expected result and evaluation

This release does not claim that Self-BLEU, Self-BERTScore, emotion entropy, or
story probability will improve by itself. Expected observable effects are:

- no impossible perspective-only targeted repair requests;
- explicit counts and examples of the Planner's initial disagreement with fixed
  real-distribution contracts;
- no retired overlay label or fixed `tone overlay: none` line in Writer prompts;
- unchanged exact structural-completeness rejection of the v80 185/186 artifact.

The next paid n=1 run must be judged on 186/186 coverage, override/repair logs,
per-comment plan realization, repetition, affect, story mass split by planned
story status, and customer-service/helpful-default rate. Formal 12-metric
matching still requires multiple matched threads; n=1 cannot establish MWU/KS
p-values.

## Verification

- Before version pinning, behavior tests passed 228/229; the only failure was
  the intentional source-hash drift guard.
- After pinning, the complete `generalized_card/tests` suite passes: 266/266.
- Ruff passes on every changed production/test Python file with only the shared
  facade's intentional dynamic-export `F401`/`F821` pattern excluded.
- Camera-product backend self-test passes with the v85 policy and current domain
  profile.
- All 72 source pins have zero missing and zero drifted entries.
- Current output audit still rejects the v80 artifact at 185/186 slots with
  `evaluable=false`.
- The exact seed-8 configuration passes `--prepare-only`; no API call was made.
  Its temporary run directory was moved recoverably to
  `/Users/yaoningyu/.Trash/generalized_card_camera_gpt54_v85_audit_seed8_20260817_v1_prepare_only_final`.

## Matched content-audit repair

Continuation on 2026-08-17 found that the existing zero-API content comparison
did not honor its own matched claim for two important rows. Lexical comments
were joined to the seed correctly, but real GoEmotions and StorySeeker values
were pooled from every camera thread. On v80 seed 8 that printed real emotion
entropy/story probability as 2.1394/0.1556; the exact matched thread values are
1.9459/0.1114.

The print-only 355-line script is now a 49-line CLI over three focused modules:
229 lines of orchestration/rendering, 235 lines of artifact joins, and 479 lines
of content analysis. The report is automatically written after formal
evaluation and includes:

- all 12 paired metric values and distances, with n=1 status forced to
  `descriptive_only_n1` regardless of saved p-values;
- exact-matched per-comment emotion/story properties;
- assigned-versus-realized tone, affect, and story splits;
- repeated n-gram contributors and opener concentration;
- separately labeled Planner helpful/advice shares and weak matched-side
  lexical probes, never a regex claim about semantic naturalness.

Five focused tests cover the matched join, n=1 interpretation, control/model
joins, evidence boundary, and incomplete-cohort rejection. The complete suite
is now 271/271, Ruff passes, and 76/76 pinned sources have no drift. A direct
CLI replay on v80 produced the new JSON/Markdown successfully;
`run_evaluate` itself still correctly refuses
to re-evaluate that artifact because it has only 185/186 structural coverage.
