# v94 worklog — state-preserving Planner repair

Date: 2026-08-18

Policy: `generalized-card-v2-state-preserving-plan-repair-v94-20260818`

## Trigger

The paid v93 N=10 run stopped on seed 2 (`1lt0yq3`), batch offset 8, root S9:

- completed posts before failure: 2;
- target comments in failed post: 45;
- requests: 100;
- input/output tokens: 356,771 / 34,168;
- estimated cost: `$0.3992`;
- elapsed: 305 seconds;
- terminal issue: `S9:long_form_capacity`.

The command used `--post-retry-limit 1`. Current code defines this as one total
attempt, so the recoverable post wrapper correctly exited without a second
whole-post attempt. Completed posts remained persisted, but the partial v93
artifact is not a formal N=10 sample.

## Persisted-attempt reconstruction

The full `planning_quality.jsonl` row for `1lt0yq3`, offset 8, was inspected.
S9 had 108 anonymous words and required about five development beats.

1. Initial S9: zero beats plus `no_story + firsthand_experience`; both
   `long_form_capacity` and `social_contract_conflict` were blocking.
2. Repair 1: five beats, but the story/evidence conflict remained. It was still
   an improvement and was selected.
3. Repair 2: `evidence_mode=small_observation` removed the social conflict, but
   `development_plan` became empty. It was selected because its remaining
   long-form issue had a lower score.
4. Repair 3: five beats returned, but evidence reverted to
   `firsthand_experience`. Its whole-plan rank was worse, so it was rejected.

The failure is therefore repair-state loss, not inability to produce beats,
root/reply confusion, insufficient retry count, or a Writer failure.

## Change

Added the focused `plan_repair.py` policy module. A field-scoped merge is used
only when exactly one repair diagnostic remains and that diagnostic has an
explicit field boundary. The current mapping is:

- `long_form_capacity -> development_plan`.

When multiple repair issues remain, the existing full-plan repair path is
unchanged. The repair Prompt says once that only the named field may change.
Returned drift is ignored in code, so correctness does not depend on Prompt
obedience. Each attempt records `candidate_plan`, `applied_candidate_plan`, and
`repair_merge_fields`.

## Verification

- Exact v93 final-candidate replay: blocking `1 -> 0`; five beats retained;
  `evidence_mode=small_observation` retained; no S9 issue.
- Active-wrapper regression reproducing all four v93 S9 states: pass.
- Focused policy tests for single-field and multi-conflict behavior: pass.
- Generalized test suite: 304 passed.
- Ruff on changed Python files: pass.
- Backend self-test with `GENERALIZED_CARD_DOMAIN=camera_product`: pass.
- Active and active-plus-legacy parity: healthy; 94/94 source pins clean.
- Exact seed-2 gate configuration completed `--prepare-only` as
  `generalized_card_camera_gpt54_v94_named_seed2_20260818_preflight_v1`.
- No API call was made during diagnosis or verification.

## Next paid gate

Use a fresh v94 tag and seed 2 only. Set `--post-retry-limit 2`, which means at
most two total attempts (one whole-post retry), after the three finite repairs
available to each failing Planner slot. Only after this 45-comment gate reaches
`Done` should a fresh v94 N=10 begin. Formal evaluation must not combine v92,
v93, and v94 outputs.
