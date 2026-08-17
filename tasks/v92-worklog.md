# v92 worklog — lossless domain-claim off mode

Date: 2026-08-17

Policy: `generalized-card-v2-lossless-domain-claim-off-v92-20260817`

## Trigger and full trace

The formal configuration uses `--domain-claim off` because v71 delivered a
separate fact to 508/522 comments and produced 157 additional semantic-overlap
flags. The current trace showed that off-mode was only a Writer registry gate:

1. root and direct-reply Prompts still required `domain_claim`;
2. normalization retained the model's claim;
3. planning quality and the earlier-plan ledger saw it;
4. `plan_comment_move_batch` skipped registry insertion when mode was off;
5. `claim_for_task` therefore returned empty at every Writer path.

The old behavior was experimentally intentional but wrong for the active
quality path. A semantic move could depend on a fact that vanished before
realization, and Planner tokens were spent on unused output.

## Change

1. `planner_claims_enabled` is the single mode predicate shared by root Prompt,
   direct-reply Prompt, normalization, and delivery.
2. Off-mode schemas retain the stable field but require literal `none`.
3. Claim-specific reference-knowledge and substantive-claim prose is omitted
   in off mode.
4. Both Planner routes require the complete contribution in `semantic_move`,
   `detail_focus`, and `domain_intent`—fields the Writer actually receives.
5. Normalization clears a returned claim when off, so model noncompliance cannot
   recreate hidden state.
6. Planned mode keeps its earlier Prompt, normalization, and registry behavior.

## Prediction

Combined with v91 named concreteness, v92 separates responsibilities cleanly:
the Planner hands over the complete semantic action; the Writer adds varied
slot-local particulars only where the real structural capacity permits. Expect
lower Planner Prompt/output mass and fewer abstract moves whose missing premise
causes generic explanatory prose. Direct content properties and all 12 metrics
remain unmeasured until paid output exists.

## Verification status

- Focused root/reply Prompt and normalization tests: 17 passed across the two
  focused invocations. The rendered tests cover planned and off mode on both
  Planner routes, assert the off schema/rule, and reject the claim-only prose.
- Full `generalized_card/tests`: 299 passed.
- Focused Self-BERTScore scorer tests: 3 passed.
- Ruff on every changed Python source and test: passed.
- Active and active-plus-legacy parity: healthy, with no unexpected generator
  or backend functions.
- Core closure: 93 declared pins, zero missing files, untracked active files,
  unpinned local imports, or drift.
- Named/off camera backend self-test passed as part of the public preflight.
- Exact seed-8 named/off `--prepare-only` passed as
  `generalized_card_camera_gpt54_v92_named_seed8_20260817_preflight_v2`; the
  recorded policy and flags match, and no API call was made.

## Paid outcome

The later v92 N=10 attempt made 95 requests and spent an estimated `$0.3740`.
Seeds 0 and 1 persisted; seed 2 stopped on root S9 before Writer generation with
an unresolved `long_form_capacity` contract. The new plan snapshots made the
v93 diagnosis possible: the repair had supplied all five beats but carried an
inapplicable root `reply_delta_type=social_close`. See `tasks/v93-worklog.md`.
The partial v92 output must not be mixed with v93 for formal statistics.
