# v93 worklog — structural root/reply boundary

Date: 2026-08-18

Policy: `generalized-card-v2-root-reply-boundary-v93-20260818`

## Paid trigger

The fresh v92 N=10 run completed seeds 0 and 1, then failed before Writer
generation for seed 2:

- failed batch: root S9–S16 at offset 8;
- terminal contract: `S9:long_form_capacity`;
- 95 requests, 322,849 input tokens, 33,058 output tokens;
- `$0.3740` estimated cost and 300 seconds elapsed;
- seeds 0 and 1 persisted; no seed-2 discussion persisted.

## Exact reconstruction

S9 is a 108-word anonymous root with `long_turn` capacity. The initial plan had
no `development_plan`. Each of its three targeted repairs did supply five
connected beats. The first candidate retained the local timing/availability
move and all five beats, but also emitted `reply_delta_type=social_close` even
though `parent_sample_id` was empty. That created a new blocking
`social_contract_conflict`; scalar quality moved from 46 to 49, so the candidate
was rejected and the initial long-form failure survived.

This is not evidence that five beats are insufficient or that
`long_form_capacity` should become nonblocking. The candidate was semantically
realizable after removing metadata that cannot apply to a root.

## Change and simplification

1. Anonymous parent topology now clears `reply_delta`, `reply_delta_type`, and
   `reply_novelty_anchor` on roots before every initial/recovery/repair quality
   evaluation. A direct reply preserves them.
2. Nonempty root overrides are persisted in `control_normalizations` with
   reason `root_slot_has_no_reply_contract`.
3. The root Planner schema requests literal `none` for all three stable fields
   and explains that its legacy `reply_relation` vocabulary describes relation
   to the seed post, not a parent.
4. Direct replies already route to `render_direct_reply_planner_prompt`, so the
   generic root Prompt no longer renders delta definitions, reply contrast
   tests, sibling rules, or parent-local contracts.
5. `_reply_delta_type_definitions` and `_render_reply_delta_contracts` had no
   remaining caller after that cleanup and were deleted. Production changes add
   50 lines and remove 138, a net reduction of 88 lines.

## Actual-candidate replay

The first paid v92 S9 repair was replayed from
`planning_quality.jsonl` through the current semantic model and quality scorer:

- selected initial rank: `(1, 46.0)`;
- repaired candidate rank under v93: `(0, 41.0)`;
- blocking issues: `[(9, long_form_capacity)] -> []`;
- normalization: only root `reply_delta_type=social_close` was cleared;
- all five development beats were retained byte-for-byte.

## Offline acceptance

- Full `generalized_card/tests`: 300 passed.
- Focused Self-BERTScore scorer tests: 3 passed.
- Ruff on changed source and tests: passed.
- Active and active-plus-legacy parity: healthy, with no unexpected functions.
- Core closure: 93 pins, zero missing, untracked, unpinned-import, or drift
  findings.
- Exact v93 named/off N=10 `--prepare-only`:
  `generalized_card_camera_gpt54_v93_named_n10_20260818_preflight_v1`; seed
  range 0–9, runs 10, posts per run 1, policy and behavior flags match, and no
  API call was made.

## Next paid step

Use a fresh v93 tag. Do not policy-upgrade or resume the v92 tag for formal
evaluation because its completed seed 0 and seed 1 discussions were planned by
the older root Prompt. After v93 generation, run the unchanged evaluation and
inspect all 12 metrics plus content/realization diagnostics.
