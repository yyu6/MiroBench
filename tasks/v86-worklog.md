# v86 worklog — Prompt-boundary simplification

Date: 2026-08-17

Policy: `generalized-card-v2-root-relation-prompt-v86-20260817`

## Evidence before editing

No paid v85 artifact existed. The current focused Writer Prompt was rendered
offline for polite, somewhat-polite, neutral, and impolite representative
slots. It contained no exact duplicate long lines. The remaining issues were
path-specific rather than generic Prompt length.

The first active contradiction was visible in real v80 plans: among 25 root
slots with no `local_parent_task_id`, 13 carried `answers_parent`; the focused
Writer rendered that as a reply relation even though it displayed the seed
post, not a parent. The raw Planner value remains useful audit evidence, so v86
translates only the Writer-facing label/value to `relation to post` and
`answers_post`/`challenges_post`.

The low-information Writer still rendered the same assignment through several
overlapping blocks: route lock, private slot, required local move, semantic
difference contract, full blackboard, payload guidance, and per-slot guidance.
The old v74 evidence says 106/522 slots used this branch, so it was not a rare
fallback. This was the one active Prompt path that had not received the compact
focused treatment.

## Changes

- Reused `_focused_slot_contract` for low-information function, payload, role,
  voice, evidence, angle, stance, detail, intent, relation, and exclusion.
- Reused `_focused_thread_ledger`, retaining semantic coverage and exact short
  utterance exclusions while dropping full control/distribution blackboard
  mass.
- Removed the duplicate semantic contract and required-local-move rendering;
  the early route lock remains authoritative.
- Kept low-information amount limits, anti-helpfulness rules, actor boundary,
  realization-in-own-words rule, visible-entity restriction, and output-only
  rule.
- Moved four legacy reviser Prompt functions plus their constants into a
  separately pinned `legacy_reviser_prompts.py`. Active `prompts.py` fell from
  2,718 to roughly 2,500 lines before the v86 low-info edit.

## Verification

- AST comparison after the split: 62 retained Prompt functions were identical;
  the only changed retained functions were `writer_prompt`,
  `_focused_slot_contract`, and `_low_info_writer_prompt`. Removed from active
  Prompt only
  `adapt_card_reviser_prompt`, `selfbleu_ngram_diagnostic`,
  `insert_reviser_guidance`, and `_ngram_tokens`; every retained function hash
  outside those three was unchanged. All four migrated functions matched their
  old AST hashes.
- Focused root/reply tests verify the visible relation target.
- The low-information regression test verifies the compact contract, retained
  short-utterance ledger, own-words rule, no legacy overlay, and a prompt under
  6,000 characters.
- `PYTHONPATH=generalized_card .venv/bin/python -m pytest -q
  generalized_card/tests`: 286 passed; focused Self-BERT scorer tests: 3 passed.
- Ruff passed for every changed Python source and regression test.
- Camera backend self-test passed. Active and active-plus-legacy parity were
  both healthy with no unexpected adapter functions.
- Core closure: 93 declared pins, 0 missing, 0 untracked active sources, 0
  unpinned local imports, and 0 drift.
- The exact v86 seed-8 command completed `--prepare-only`: `run_config.json`
  records `generalized-card-v2-root-relation-prompt-v86-20260817`, and the run
  directory contains only configuration plus the domain profile—no generated
  output, request log, or token-usage file.

## Paid acceptance still pending

The seed-8 run is a qualitative and realization diagnostic, not an inferential
metric pass. It must have 186/186 structural coverage and be reviewed for
Planner→Writer tone/affect/story realization, repeated framing, spontaneous
emotion/profanity, personal story mass, and generic helpful/customer-service
defaults. Only a later sufficient-N matched run can test MWU/KS p-values.
