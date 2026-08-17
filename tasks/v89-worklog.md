# v89 worklog — realizability-first Planner repair

Date: 2026-08-17

Policy: `generalized-card-v2-realizability-first-planner-v89-20260817`

## Trigger

The formal v88 seed-8 run failed before Writer generation:

- 24 Planner requests;
- 190,038 input tokens, 13,507 output tokens;
- `$0.1805` estimated cost;
- 116 seconds elapsed;
- no persisted discussion and therefore no content or 12-metric result.

The failure was in the depth-0 batch S9–S16. The final selected report retained
`social_contract_conflict` on S10, S13, and S15.

## Exact reconstruction

The current raw thread, matched structural sampler, reference-template selector,
and slot scheduler were replayed without an API call. The batch contracts were:

- S10: 50-word ordinary root, `tone_class=polite`, affect `confusion`;
- S13: 51-word ordinary root, `tone_class=polite`, affect `neutral`;
- S15: 146-word long root, `story_mode=messy_multi_step_story`,
  `tone_class=impolite`, affect `disapproval`.

The old audit did not store actual plan payloads, but its attempt sequence is
sufficient to prove the selection bug. A late S15 repair removed S15's story
conflict, leaving the two polite conflicts, but its aggregate score was 56
against the selected 54 and it was rejected. Collision had a larger scalar
weight than story coherence.

## Changes

1. `PlanQualityReport.repair_rank` is `(blocking_issue_count, issue_score)`.
   Story/affect social contracts, surface capacity/density, and long-form
   capacity remain blocking. Candidate selection uses this rank.
2. Polite role mismatch is a separate `tone_role_mismatch` diagnostic with
   weight 1. It still triggers bounded targeted repair and explicitly warns
   against customer-support routing, but it does not block Writer generation.
3. The root Planner's blanket hidden-anecdote ban is replaced by the same
   conservative synthetic-story boundary already used by the Writer. This
   resolves the direct conflict with fixed story and firsthand-evidence fields.
4. `planning_quality.jsonl` now stores JSON-safe initial, recovered, candidate,
   and selected plan snapshots plus before/candidate ranks. No matched comment
   body is included.
5. The terminal error now says “unresolved blocking slot plans”; it no longer
   mislabels every exhausted Planner output as an internally unrealizable
   system contract.

## Review evidence so far

- 34 generation-distribution tests pass.
- 194 generalized-card integration tests pass.
- The new regression constructs the exact scoring pathology: a story repair
  removes one blocking conflict while adding a duplicate-claim collision. v89
  accepts the realizable candidate, retains the collision diagnostic, and logs
  initial/candidate/selected evidence.
- Ruff passes on the changed Python sources and tests.

## Final offline acceptance

- Full `generalized_card/tests`: 294 passed.
- Focused Self-BERT scorer tests: 3 passed.
- Ruff: passed on every changed Python source and test.
- Camera backend self-test: passed with matched speakers, focused Writer,
  `own_words`, social coherence on, sibling visibility on, and own-fact license
  off. Hugging Face was forced offline so the test used the local model cache.
- Active parity and active-plus-legacy parity: healthy.
- Core closure: 93 declared pins, zero missing files, untracked active files,
  unpinned local imports, or drift.
- Exact seed-8 prepare-only:
  `generalized_card_camera_gpt54_v89_preflight_seed8_20260817_v4`; policy and
  all requested behavior flags match, and no API call was made.

The remaining gate is the paid v89 seed-8 run, followed by content audit and
descriptive single-thread metrics. Formal MWU/KS remains a sufficient-N test.
