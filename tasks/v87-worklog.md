# v87 worklog — payload-safe Writer routing

Date: 2026-08-17

Policy: `generalized-card-v2-payload-safe-writer-routing-v87-20260817`

Status: superseded by v88 before any paid generation; retained for provenance.

## Scope and evidence method

This pass did not use historical implementations as design authority. It used
the current route predicates, finalizer, Prompt builders, plan-quality
validator, tests, and the 186 recorded task contracts from the paid v80 seed-8
artifact. Git history remains provenance only.

A temporary zero-API audit replayed every task in original order through the
current Writer Prompt. It recorded route, parent visibility, payload, story,
tone, affect, prompt size, repeated long lines, and repeated semantic moves.
No generated response was requested.

## Finding 1 — short shape overrode substantive meaning

`should_use_low_info_writer` returned early for a short low-information
`utterance_mode` before checking `payload_type`. The 186-task replay selected 32
low-information routes, including six `soft_helpful` tasks and one `correction`.
That Prompt correctly prohibits advice, explanation, and caveats, so the route
made the Planner contract unrealizable.

The predicate now first excludes every payload outside the explicit
low-information set, then uses utterance shape and length only within that set.
After the change:

- 186/186 tasks render;
- 25 use low-info and 161 use focused substantive;
- low-info payloads are 12 `narrow_question`, 6 `fragment_datapoint`, 5
  `bare_answer`, 1 `low_info_reaction`, and 1 `meta_or_template`;
- all 25 are `no_story`.

The `meta_or_template` row is a frozen v80 task with contradictory
`gratitude_reply` metadata. Current v87 planning now rejects that contract; a
replay of already-finalized historical task JSON deliberately does not rewrite
the evidence.

## Finding 2 — focused ledger construction was indirect and unbounded

Focused paths rendered the complete five-section thread blackboard and parsed
two headings back out of the resulting string. Short slots also requested an
exclusion limit based on the entire previous-comment count, so the effective
history was unbounded. Some exact lines appeared both in the nearby openings
block and the parsed ledger; three social-close tasks repeated the same
`semantic_move` as both the required route and already-covered content.

Focused and low-info paths now construct only the two required ledgers directly
from comment records. Short-line history is capped at 32 for short outputs and
12 otherwise; semantic coverage is capped at 8 and 16 respectively. The 24
visible openings are deduplicated from the short ledger. For an assigned
`social_close`, an identical generic social move is not rendered as already
covered. Full exact-duplicate validation remains in persistence and still sees
all comments.

Replay after the change found zero exact duplicate long lines and zero repeated
required semantic moves. Low-information Prompt mean/max moved from about
7,581/9,397 to 6,579/7,921 characters. The audit's provisional 6,000-character
warning is not an acceptance criterion: fixed necessary blocks alone account
for roughly 4,200 characters, and there is no output evidence for further
blind truncation.

## Finding 3 — derived tone state could outlive its source contract

The shared expander derived `real_tone_slot` before later surface overrides and
before generalized code restored Planner-owned role, payload, and voice. The
Writer could therefore receive `pure_acknowledgement` for a final neutral
datapoint or correction.

The finalizer now removes stale `gratitude_social` texture when the final task
is not a social acknowledgement, then recomputes the Writer-facing tone slot
from the final contract. Voice alone is not treated as proof of gratitude; only
the final gratitude role or gratitude/relief affect qualifies.

The plan validator is also bidirectional. Gratitude/relief already required a
no-story gratitude reaction. v87 additionally rejects
`speaker_role=gratitude_reply` or `reply_delta_type=social_close` unless affect,
function, payload, role, and story form the same social-reaction contract. Tone
class is intentionally not forced to polite: politeness is the user's lowest
priority and no authoritative real joint distribution justifies that extra
constraint.

## Verification

- Targeted routing/social/ledger tests: 7 passed.
- Full `generalized_card/tests`: 290 passed.
- Focused Self-BERT scorer tests: 3 passed.
- Ruff passes on all changed Python sources and tests.
- Camera backend self-test passes.
- Active parity and active-plus-legacy parity are healthy with no unexpected
  adapter functions.
- Core closure: 93 declared pins, 0 missing, 0 untracked active sources, 0
  unpinned local imports, and 0 drift.
- Exact seed-8 v87 `--prepare-only` records the v87 policy and makes no API call.

## Remaining acceptance work

No claim is made that Self-BLEU, Self-BERT, emotion entropy, story probability,
or any other output metric improved: v87 has no generated artifact yet. The
next paid seed-8 run is a qualitative and Planner→Writer realization diagnostic,
not an inferential pass. If and only if it has exact 186/186 coverage and
credible natural-content results, the unchanged policy should advance to a
sufficient-N matched evaluation for the 12 metrics and MWU/KS tests.
