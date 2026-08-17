# v88 worklog — structural speakers and conflict-free grounding

Date: 2026-08-17

Policy: `generalized-card-v2-structural-speakers-grounding-v88-20260817`

## Why v87 was not sent directly to a paid run

The completion audit inspected the exact current configuration rather than
stopping at v87's passing tests. Two enabled/disabled arms were not clean:

1. `own-fact-license off` intentionally retained a historical blanket ban, but
   `_own_equipment_block` still rendered an invented equipment permission for
   first-person-shaped tasks.
2. `speaker-identity matched` recovered a valid structural join and then added
   invented kit, tenure, and use-case biography. Therefore the formal v87
   command left it off and kept the known one-author-per-slot mismatch.

Neither behavior should remain in the active path merely to reproduce an old
arm. Git and the v87 policy ID already preserve that version.

## Grounding replay

The 186 frozen v80 tasks were re-finalized and rendered through the current v87
focused/low-info routes with the exact camera profile and `own-fact-license
off`. Before editing:

- equipment permission block: 78/186 Prompts;
- `or personal experiences` ban: 144/186;
- permission and ban together: 61/186.

v88 makes the equipment block reachable only when the explicit legacy `own`
license is active. `off` keeps its conservative fact rule without displaying a
permission; `named` permits ordinary particulars but does not assign a fake
kit. `equipment_closing_clause` rejects unlicensed calls so another path cannot
silently recreate the conflict.

After editing, the full replay remains 186/186 with 25 low-info and 161
substantive routes, while equipment blocks and permission/ban conflicts are both
zero.

## Structural speaker boundary

The current speaker module and every call site were audited. The matched author
string is now used only as a grouping key. Deleted/removed accounts remain
separate one-shot groups. A `Speaker` contains only:

- anonymous generated `speaker_id`;
- OP membership;
- owned matched slot IDs;
- whether the source account was anonymous.

Deleted fields: `display_name`, `kit`, `tenure`, and `use_case`, together with
the tenure ladder, kit sizing, inventory allocation, and kit-filter helper. The
Prompt no longer claims a fixed biography or kit. A returning speaker sees up
to three of its own earlier generated turns and is told to keep factual
self-claims consistent, while the current Planner-owned voice and affect remain
authoritative.

Matched anonymous participation is now the default. `off` remains an explicit
one-author-per-slot ablation. If matched roster selection unexpectedly returns
no slots, generation fails instead of silently claiming matched mode while
falling back to one-shot authors.

Current seed-8 audit against the authoritative raw thread:

- 186 slots;
- 97 generated speaker groups;
- 80 named-source groups and 17 anonymous one-shots;
- 2.112 turns per named group;
- 35 recurring groups covering 66.7% of all slots;
- maximum 10 slots for one speaker;
- zero source author strings stored in the roster summary.

An integration test runs the configured active expander on repeated-author
rows and proves that the resulting `CommentTask.speaker_id` values recur on the
correct slots. This is in addition to helper, OP, deleted-account, leakage,
field-survival, and Prompt-memory tests.

## Simplification and verification

The production diff deletes substantially more code than it adds: invented
speaker biography and kit routing are removed rather than hidden behind another
condition.

- Full `generalized_card/tests`: 292 passed.
- Focused Self-BERT scorer tests: 3 passed.
- Ruff passes on all changed Python sources and tests.
- Camera backend self-test passes with `speaker-identity matched`.
- Active and active-plus-legacy parity are healthy.
- Core closure: 93 declared pins, zero missing, untracked active, unpinned
  imports, or drift.
- Exact v88 seed-8 `--prepare-only` records matched speakers and the v88 policy;
  no API call was made.

## Remaining evidence

No generated v88 artifact exists. Natural continuity, repetition, emotion,
story, Self-BERT, and the other 12 metrics remain hypotheses until paid seed 8;
formal MWU/KS remains a sufficient-N gate.
