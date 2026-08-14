# Generator slimming and refactor

Goal: make the generation engine readable before changing its behavior. The
9,290-line generator is unreadable, so every past diagnosis was made from
fragments. The audit in `tasks/generator_audit.md` lists 14 findings; several
turned out to be wrong precisely because the code was too large to hold at once.

Ordering rule: **shrink first, fix second.** No behavior change lands until the
engine is legible.

## Verification tools (built first, used at every step)

- [x] `scratchpad/reach.py` — runtime reachability under the generalized-v2
      monkey patches. Models the three ways a definition dies: replaced by an
      adapter patch and never invoked, only reachable from the card-snapshot
      profile branch, or only read by `safe_getsource` for provenance.
- [x] `scratchpad/why.py` — prints the root path that keeps a definition alive.
- [x] `scratchpad/defsnap.py` — hashes every surviving top-level definition.
      Pure deletion and pure moves must leave every survivor byte-identical.
- [x] Test baseline: `cd generalized_card && python3 -m pytest tests/ -q`
      → **209 passed in 14.24s** (must be run from `generalized_card/`; from the
      repo root the package resolves as a namespace and collection fails).
- [x] Original archived: the pre-refactor generator is byte-identical to
      `artifacts/.../v70_smoke10_.../generated/_reproducibility/generator_source_snapshot.py`
      (sha256 `5cf5828b…`, the hash pinned in `core_contract.py`).

## Measured starting point

| File | Lines | Live def lines | Dead def lines |
|---|---|---|---|
| `scripts/sampling_generator/run_sampled_reddit_generator.py` | 9,290 | 3,961 | **4,251** |
| `generalized_card/generalized_card/backend.py` | 2,289 | — | — |
| `generalized_card/generalized_card/prompts.py` | 2,786 | — | — |

135 of 263 generator definitions execute on a generalized-v2 run.

## Stage 1 — delete dead code in place ✅

- [x] Deleted 126 unreachable definitions and 9 unused module constants.
- [x] Retired the CARD-parity symbol list entries with no referent.
- [x] Proven: 0 unintended changes to survivors, 209 tests pass, preflight
      self-test passes.

## Stage 2 — split into cohesive modules ✅

- [x] `scripts/sampling_generator/` is a package; 13 engine modules, largest 454
      lines.
- [x] Pure moves: 165 of 166 surviving definitions byte-identical.
- [x] Every engine file pinned in `core_contract.py` and checked on load.

**Result: 9,290 → 2,059 in the facade, 4,780 across the whole engine.**

The facade stays at 2,059 because a definition can only move out if it never
calls a patched name. See `tasks/refactor_map.md` for the list of 26 that
cannot.

## Stage 3 — dissolve the monkey-patch layer (not started)

The remaining 2,059 lines are one module only because `backend.py` rebinds 67
module attributes and Python resolves bare calls through the calling module's
globals. Dissolving that is the only way to split the spine.

- [ ] Replace each patched attribute with an explicit dependency.
- [ ] Drop the `card-snapshot` profile and `_generalize_instruction_text`'s
      40-entry substitution table with it.
- [ ] This one changes behavior risk, not just layout, so it needs a
      stub-provider run proving prompts are byte-identical before and after.

## Stage 4 — re-audit the findings, then fix ✅

- [x] Re-verified all 14 findings against v70 data. See the re-audit section of
      `tasks/generator_audit.md`. Six confirmed; four dropped or deferred.
- [x] Implemented all six. **211 tests pass** (2 new), preflight self-test passes.
- [x] Policy bumped to
      `generalized-card-v2-planner-owned-reply-move-single-parent-exclusion-v71-20260813`.

| # | Change | Where | Predicted signal on the next run |
|---|---|---|---|
| A | a reply keeps the Planner's `semantic_move`; the novelty anchor no longer replaces it | `task_distribution.py` `PLANNER_OWNED_TASK_FIELDS`, facade slot expansion, reply planner prompt | reply `semantic_move` word count 7.7 → ~20, `== reply_novelty_anchor` 61/61 → ~0; `self_bleu_4` and `self_bertscore` down toward real |
| B | the parent's proposition is excluded once, not three times verbatim | facade `must_not_do` / `avoid_repeating`, `prompts._semantic_route_lock` | parent overlap 0.129 → toward the real 0.197; `hard_disagree_rate` δ 0.61 → lower |
| C | every opening used in the thread stays in the ledger | facade `generate_post_from_tasks` | openings visible per prompt 18 → thread length; `self_bleu_4` down |
| D | the reply planner interpolates the `claim_family` enumeration | `reply_planning.py` | reply `miscellaneous` 61/61 → a spread over 16 families; the per-thread family cap starts governing replies |
| E | length floor and ceiling track the matched slot instead of its bucket | `engine/writer_validation.py` | 0-30-word slots stop running 1.20x long, the 200+ tail stops running 0.57x short; `length_cv` 0.874 → toward 1.038 |
| F | entry grammars a slot cannot carry are never scheduled, and the quota is re-spent on writable types | `planner_distribution.py`, planner prompt | opener realization 43.8% → higher; `question` 0/23 and `imperative` 0/10 no longer waste quota |

**One caveat on E, stated because it changes what to expect.** `real_slot_too_short`
and `length_too_long` are *repairable* diagnostics, not blocking failures, and the
v70 run set `writer_local_repair_rounds=0` and `writer_slot_retry_limit=0`. With
repair disabled a corrected bound is logged but never acted on. To make E move
`length_cv`, the next run needs at least one local repair round — a run-config
choice with an API cost, so it is yours to make, not mine.

## Stage 5 — domain adapter and learned directions (not started)

Asked for, not yet built. Current state of the domain profile, measured:

**Already learned from evaluation-excluded real threads**: opener shares,
entity inventory, metric calibration templates, lexical-quality calibration,
surface behaviour observations, and 2,557 abstracted reference viewpoints.

**Still hand-written**, and this is what blocks `news` / `sports` / `reddit`:

- `configs/domains/camera.json` carries 9 hand-written `topic_facets`, 16
  `technical_terms`, and 14 `protected_entity_terms`.
- `perspectives` are **not domain-derived at all**: `domain_profile.py:73` calls
  `universal_viewpoints()`, a fixed P01-P12 decision lens in
  `planning_quality.py`. Every domain gets the same twelve directions, and
  `max_perspective_share=0.34` is enforced against them.
- `backend._generalize_instruction_text` holds a 40-entry hand-written
  substitution table.

- [ ] An LLM domain adapter that induces facets, terms, protected entities, and
      perspectives from the excluded threads, so a domain config needs only
      domain_id, display name, community context, and data paths.
- [ ] Let the Planner read what each real comment actually does — its stance,
      register, and direction — rather than picking from a fixed lens.
- [ ] Retire the 40-entry substitution table once the adapter supplies it.
