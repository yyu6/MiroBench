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

The print-only 355-line script is now a 49-line CLI over focused orchestration,
artifact-join, and content-analysis modules. The report is automatically
written after formal evaluation and includes:

- all 12 paired metric values and distances, with n=1 status forced to
  `descriptive_only_n1` regardless of saved p-values;
- exact-matched per-comment emotion/story properties;
- assigned-versus-realized tone, affect, and story splits;
- repeated n-gram contributors and opener concentration;
- separately labeled Planner helpful/advice shares and weak matched-side
  lexical probes, never a regex claim about semantic naturalness.

Seven focused content tests cover the matched join, n=1 interpretation,
control/model joins, stage statistics, evidence boundaries, legacy alignment,
and incomplete-cohort rejection. The complete suite is now 285/285, Ruff
passes, and 92/92 declared source hashes have no drift. A direct
CLI replay on v80 produced the new JSON/Markdown successfully;
`run_evaluate` itself still correctly refuses
to re-evaluate that artifact because it has only 185/186 structural coverage.

## Planner target → Writer output decomposition

A zero-API reconstruction used the current excluded-real domain profile, the
frozen 150-seed pool, the exact matched real metric rows, and the same
`select_thread_template` function used at generation time. The selected target
distribution passes both non-paired tests on all 12 metrics at N=10 and N=150.
At N=150, selected-target means / matched-real means were:

| metric | selected target | matched real | MWU p | KS p |
|---|---:|---:|---:|---:|
| self_bleu_4 | 0.0333 | 0.0330 | 0.4943 | 0.5322 |
| self_bertscore_mean_f1 | 0.4933 | 0.4923 | 0.8852 | 0.2911 |
| semantic_mean_cosine | 0.2723 | 0.2741 | 0.9697 | 0.8165 |
| hard_disagree_rate | 0.1016 | 0.1209 | 0.1001 | 0.1391 |
| polite_rate | 0.3467 | 0.3216 | 0.1936 | 0.1391 |
| impolite_rate | 0.3775 | 0.4079 | 0.1613 | 0.3617 |
| neutral_rate | 0.1691 | 0.1611 | 0.6436 | 0.7250 |
| length_cv | 0.9507 | 0.9394 | 0.4960 | 0.4425 |
| avg_depth | 2.1042 | 2.2437 | 0.2641 | 0.2911 |
| structural_virality | 2.1721 | 2.2554 | 0.7019 | 0.4425 |
| mean_story_probability | 0.1364 | 0.1357 | 0.9623 | 0.2911 |
| emotion_entropy | 1.4728 | 1.5318 | 0.4373 | 0.5322 |

This is an audit, not a generation control and not test-set tuning. It refutes
the hypothesis that the current Planner target sampler is the main root cause.

The content report schema is now v3. Each new post atomically stores the exact
reference template; the JSONL also includes `seed_key`. Old sequence-only logs
are supported only when sequence indices form one complete unique mapping, so
an interrupted/resumed log cannot be silently joined to the wrong post.
Replaying v80 seed 8 produces the following causal examples:

- polite: real 0.2324 → target 0.2486 → generated 0.0595;
- story: real 0.1114 → target 0.1281 → generated 0.2488;
- Self-BERT: real 0.4887 → target 0.4591 → generated 0.5208;
- emotion entropy: real 1.9459 → target 1.5359 → generated 1.5358.

## Evaluation and source-integrity simplification

The active evaluator no longer uses the delete/normalize cleanup program. It
audits the Writer artifact, copies it byte-for-byte into the scoring snapshot,
and rejects noncanonical tree metadata without repair. Focused generalized
modules now own metric orchestration and the exact MWU/KS/Cliff/Wasserstein
definitions. A comparison fixture confirmed every returned statistic matches
the previous formal implementation.

The matched evaluator, nine scorer CLIs, and summarizer were pinned but not
tracked before this pass; they are now git sources. `repin_core_contract.py`
also checks active git tracking and recursively audits local package imports.
The default parity scope is active generation/evaluation and does not load
revisers. Current result: 92 declared hashes agree, 67 active sources are
tracked, and zero local imports are missing from the active pins.

A final runtime-edge audit found dependencies that an import-only package walk
could not prove: generation/evaluation backend and output-audit runners, plus
the dynamically imported token tracker and subprocess token summarizer. They
are now explicit active pins and git sources. The closure walk also follows
sibling-script imports.

Singleton MWU/KS values are now labeled `DESCRIPTIVE` in the matched evaluator,
Markdown, JSON rows, `run_evaluate` console, and content report. v80 replay no
longer prints the mathematically true but scientifically false `12/12 PASS`.

Verification after these changes: 285/285 generalized tests, 3/3 focused
Self-BERT scorer tests, Ruff on every active changed Python source, all scorer
CLI `--help` imports, camera backend self-test, active parity, strict pin/import
audit, v80 content and matched-evaluator replay, and exact v85 seed-8
`--prepare-only`. No API call was made.
