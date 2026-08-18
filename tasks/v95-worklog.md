# v95 worklog — compiled, non-terminal Planner content contracts

Date: 2026-08-18

Policy: `generalized-card-v2-nonfatal-compiled-plan-contract-v95-20260818`

## Trigger and measured failure

The v94 seed-2 gate used three total attempts and still produced no discussion.
Its 152 requests included 130 targeted Planner quality repairs. It consumed
1,031,377 input tokens, 76,450 output tokens, 541 seconds, and `$0.9608`.

The terminal combinations differed on every attempt:

1. S20 `no_story + firsthand_experience`; S22 had either five beats with a
   low-information payload or a substantive payload with no beats.
2. S43 was a natural gratitude close whose affect was overwritten to neutral.
3. S18 had six development beats but a `meta_or_template` payload on a
   118-word anonymous slot.

More whole-post attempts therefore sampled new failures rather than recovering
one transient fault.

## Root cause

The held-out template owns aggregate story, tone, affect, and opener targets.
The Planner owns the local semantic move plus dependent evidence and discourse
route. Code overlaid template labels after the Planner returned, but did not
compile the dependent fields with those labels. The validator then treated a
statistical target as per-slot semantic truth and aborted when the two owners
disagreed.

The direct-reply path had an additional Prompt/handoff gap: schedule defaults
were not merged into `slot_controls`. Most reply slots displayed
`Story contract: unassigned`; normalization later forced `no_story`.

## Implementation

`planner_contract.py` now compiles dependent controls before quality scoring:

- a scheduled story receives firsthand evidence and a story/datapoint route;
- `no_story` removes narrative evidence/payload while retaining the local move;
- micro slots receive a micro reaction route;
- social closes receive a coherent affect/role/function/payload route;
- ordinary/long slots cannot retain a whole-comment low-information payload.

The compiler does not change `semantic_move`, `local_topic`, `detail_focus`,
`decision_boundary`, branch ownership, or reply novelty. Every changed field,
before/after value, reason, slot, and repair attempt is appended to
`control_normalizations`.

Direct reply controls now merge the schedule's default `story_mode=no_story`
before rendering the Prompt. Planner content diagnostics receive at most one
repair per slot unless a compiled within-plan conflict remains. Missing
long-form beats are non-terminal because `render_development_guidance` already
derives a fallback beat count from the anonymous slot capacity.

After bounded repair, residual content-contract diagnostics are stored under
`unresolved_plan_contract_warning` and continue. Missing S# rows, malformed
JSON, API/auth/safety failures, empty Writer output, and incomplete exact slot
coverage remain terminal.

## Zero-API verification

- All 19 saved v94 selected batch states replayed through the compiler:
  social/surface terminal conflicts `3 -> 0`.
- The complete second v94 attempt prefix (43 slots) has zero compiled contract
  conflicts; without embedding-only collision checks it would need at most nine
  soft repair calls rather than the observed 130 across three restarts.
- Cross-product test covers story/no-story, neutral/gratitude, side/gratitude
  roles, meta/story/advice payloads, and micro/ordinary/long capacities.
- A residual unsupported short-story contract is explicitly tested to warn and
  continue rather than raise.
- Complete generalized test suite: 307 passed.
- Ruff over every changed Python source and test: clean.
- Source contract: 95/95 files present and clean, no untracked active source and
  no unpinned local imports.
- Active and active-plus-legacy parity: both healthy, with no unexpected core or
  backend overrides.
- Backend self-test with `GENERALIZED_CARD_DOMAIN=camera_product`: pass.
- The exact seed-2 configuration that failed under v94 completed
  `--prepare-only` under v95 as
  `generalized_card_camera_gpt54_v95_named_seed2_20260818_preflight_v1`; no API
  call was made.

These gates prove the active path and non-terminal contract policy, not provider
availability or realized-text quality. Completion does not prove the 12 metrics:
it makes a complete, structurally honest sample available to the unchanged
scorers for the first time in these failed gates.
