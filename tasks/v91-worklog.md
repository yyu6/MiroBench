# v91 worklog — slot-gated concreteness permission

Date: 2026-08-17

Policy: `generalized-card-v2-slot-gated-fact-license-v91-20260817`

## Trigger

The final pre-run audit traced the never-run `named` arm from backend
configuration through `slot_license`, all Writer Prompt paths, and the configured
system Prompt. The resolver licenses only substantive slots: anonymous matched
length at least 25 words and neither `micro` nor `short`. The system Prompt did
not preserve that gate; it told every call to name particulars and give amounts.

Consequently, a micro reaction combined a global instruction to add detail with
a per-comment instruction that names/numbers may appear only when visible and
that no extra fact fits. The existing license tests changed the module flag
after configuration, so they exercised the user Prompt but never the configured
system Prompt.

## Current-data evidence

On the exact v80 seed-8 structural records:

- total slots: 186;
- `named`-eligible substantive slots: 110 (59.14%);
- matched-real comments containing a digit: 59.68%;
- v80 generated comments containing a digit: 31.35%;
- distinct model designators: matched real 118, generated 29;
- top designator share: matched real 13.94%, generated 46.27%.

The gate's capacity therefore matches the real numeric surface density closely;
the problem was that the system sentence bypassed the gate.

## Change

1. The named-mode system sentence now states only that an explicitly licensed
   per-comment turn may override the preceding visibility rule.
2. The name/amount behavior remains once in the substantive user Prompt.
3. Unlicensed micro/short turns retain the visible-only entity rule.
4. The retained `own` mode uses the same conditional system authorization for
   personal history instead of a global permission.
5. Regression tests configure the backend from the environment, inspect the
   actual system Prompt, and compare licensed substantive versus unlicensed
   micro final user Prompts.

## Predicted result

Run `domain-claim=off` with `own-fact-license=named`. The Writer should introduce
more varied particulars locally without injecting one Planner fact into most
comments. Expected direct properties are higher digit/domain-name coverage and
lower designator concentration. The hypothesized 12-metric effects are lower
Self-BLEU/Self-BERTScore and possibly lower semantic concentration; none is
claimed before paid output.

## Verification status

- Focused grounding/concreteness tests: 19 passed.
- Full `generalized_card/tests`: 297 passed.
- Focused Self-BERTScore scorer tests: 3 passed.
- Ruff on every changed Python source and test: passed.
- Active and active-plus-legacy parity: healthy.
- Core closure: 93 declared pins, zero missing files, untracked active files,
  unpinned local imports, or drift.
- Named-mode camera backend self-test: passed with focused Writer, `own_words`,
  social coherence on, sibling visibility on, and matched speakers.
- Exact 186-slot named Prompt replay: 110 licensed and 76 unlicensed; every
  licensed Prompt contains the behavior once, every unlicensed Prompt zero
  times, the system has one conditional authorization and no behavior duplicate,
  and no invented-equipment block appears.
- Exact seed-8 named-mode prepare-only:
  `generalized_card_camera_gpt54_v91_named_seed8_20260817_preflight_v1`; policy
  and requested behavior flags match, and no API call was made.

No v90 or v91 API call has been made.
