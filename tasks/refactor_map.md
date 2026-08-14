# Refactor map

Where every removed or moved definition went, so a later diagnosis can be traced
back. Ordered by stage.

**Archive of the pre-refactor generator**: byte-identical copy at
`artifacts/generalized_card/runs/generalized_card_camera_gpt54_v70_smoke10_20260813_v1/generated/_reproducibility/generator_source_snapshot.py`
(sha256 `5cf5828b90441466ac167e68a33dd4080bbbb9f6128ed194bc5616cb341e64c3`, the
hash that `core_contract.py` pinned before this work). Nothing below is lost.

## How "dead" was decided

Not by reading. `scratchpad/reach.py` builds the call graph the way the runtime
actually resolves names, which is what reading kept getting wrong:

1. `backend.py` rebinds 67 module attributes. After `module.g = new_g`, a
   generator function calling `g(...)` reaches the replacement, so the original
   `def g` body is dead unless `configure_generator_backend` captured it first
   **and** the replacement invokes it.
2. A capture is not an invocation. Three captured originals are passed to a
   replacement that discards them:
   `rebalance_card_surfaces` opens with `del core_rebalance`,
   `_substantive_safe_degraded_task` with `del original`, and
   `_evaluator_aligned_lexical_overlap_check` never takes one.
3. Five prompt builders are wrapped only in the `card-snapshot` branch. That
   profile loads a different, frozen file, so under `generalized-v2` they never
   ran.
4. `write_reproducibility_snapshot` passed five functions to `safe_getsource`.
   Reflection reads a function object without running it, so that reference kept
   `build_low_info_writer_prompt` and every `gpt_*` guidance block looking alive
   while none of them could reach the Writer.
5. `backend.py` is not the only caller. `prompts.py` receives the loaded
   generator as a parameter named `backend` and calls ten functions on it,
   including `controls_for_task` and `render_sampled_plan_block`. Scanning only
   `backend.py` wrongly condemned those — caught by the test suite, then fixed
   in the analyzer.

## Stage 1 — dead code deleted in place

`scripts/sampling_generator/run_sampled_reddit_generator.py`: **9,290 → 4,759
lines**, 126 of 263 definitions removed. Verified by `defsnap.py`: every
surviving definition is byte-identical, and `pytest generalized_card/tests/`
holds at **209 passed**, plus the real preflight
(`run_generator_backend.py --self-test`) passes end to end.

### Deleted: CARD surface rebalancing (~1,450 lines, 73 definitions)

`rebalance_tasks_for_diversity` and everything only it reached:
`rebalance_tone_shapes`, `rebalance_selfbert_answer_shapes`,
`rebalance_micro_short_tasks`, `rebalance_social_noise_tasks`,
`rebalance_gratitude_tasks`, `cap_complete_answer_tasks`,
`cap_question_like_tasks`, `cap_repetitive_selfbert_shapes`,
`restyle_complete_answer_task`, `demote_*`, `make_*_task`, `can_convert_to_*`,
`*_priority`, `*_budget`, and the whole tone-target/overlay family
(`assign_thread_tone_targets`, `apply_tone_target`, `apply_tone_overlay`,
`make_reddit_polite_task`, `make_calm_tone_task`, `soften_harsh_tone_task`,
`is_metric_visible_polite_task`, `polite_marker_family_instruction`, …).

**Replaced by**: `task_distribution.rebalance_card_surfaces`, which reports and
returns the tasks unchanged, followed by
`generation_distribution.allocate_story_and_affect`. The Planner owns these
controls now, so a second surface pass would overwrite it. Its `core_rebalance`
parameter, which it deleted on entry, was removed from both signature and call.

### Deleted: CARD prompt construction (~900 lines, 29 definitions)

`build_writer_prompt`, `build_planner_prompt`,
`build_comment_move_planner_prompt`, `build_low_info_writer_prompt`,
`build_minimal_context_prompt`, `render_parent_context_for_writer`,
`render_seed_context_for_writer`, `render_matched_real_sample`,
`render_matched_real_rows_for_move_planner`, `render_concrete_anchor_block`,
`render_gpt_thread_memory`, `render_gpt_distribution_pressure`,
`gpt_metric_guidance_block`, `gpt_tone_discourse_guidance_block`,
`gpt_payload_specific_guidance_block`, `gpt_placeholder_guidance_block`,
`gpt_comment_tags`, `gpt_discourse_shape_for_comment`, `tone_shape_guidance`,
`speaker_role_guidance`, `utterance_mode_guidance`, `surface_texture_guidance`,
`real_surface_shape_guidance`, `voice_guidance`, `real_length_guidance_line`,
`writer_length_rule`, `is_advice_like_comment`, `is_question_like_comment`,
`is_social_ack_comment`, `is_story_like_comment`, `is_blunt_like_comment`.

**Replaced by**: `prompts.py` (`writer_prompt`, `planner_prompt`,
`comment_planner_prompt`, `render_parent_context`, `render_seed_context`).

Worth stating plainly, because the audit flagged it and this confirms it: the
`gpt_*` blocks were the most detailed per-metric Writer guidance in the file and
**not one word of them ever reached a model**. They survived only because
`write_reproducibility_snapshot` fed them to `safe_getsource`.

### Deleted: real-comment classification replaced by domain-neutral versions

`infer_real_tone_slot`, `infer_real_comment_social_overrides`,
`real_text_allows_first_person_frame`, `real_text_allows_uncertainty_frame`,
`is_real_comment_usable`, `mask_high_salience_context_terms`,
`extract_product_anchors`, `extract_term_anchors`,
`build_concrete_anchors_for_task`.

**Replaced by**: `backend._generic_real_tone_slot`,
`_structural_real_comment_overrides`, `_allows_first_person`,
`_allows_uncertainty`, `_anchor_builder`, and `prompts.mask_specifics` /
`extract_*_anchors`. The deleted versions carried finance-era word lists.

### Deleted: provider and I/O plumbing replaced by the adapter

`chat_completion_text`, `parse_json_object`, `uses_max_completion_tokens`,
`preflight_openai_compatible_endpoint`, `load_or_init_discussion`,
`load_real_thread_bank`, `find_matched_real_thread`, `sanitize_writer_text`,
`retry_note_for_problems`, `guard_fallback_retry_note`,
`has_realistic_long_helpful_anchor`, `lexical_overlap_problem`,
`degraded_task_for_guard_failure`, `lightly_jitter_context_text`,
`strengthen_task_dropout`, `dropout_transform_for_role`,
`grounded_context_transform`.

**Replaced by**: the corresponding `backend.py` patches.

### Deleted: the generator's own self-test (1,185 lines)

`run_self_test` and `FakeOpenAIClient`. `backend.py` replaces the attribute with
`_run_generalized_self_test`, so both the `run_generate.py` preflight subprocess
and `run_generation_harness.py` already ran the generalized version.

## Stage 1 — edits outside the generator

| File | Change | Why |
|---|---|---|
| `backend.py` `CORE_ALGORITHM_SYMBOLS` | dropped 8 rebalancer names | read with `getattr` before patching, so a name here must exist |
| `backend.py` `configure_generator_backend` | 5 prompt-builder captures moved into the `card-snapshot` branch | only that profile wraps them; the generalized engine no longer defines them |
| `backend.py` `configure_generator_backend` | dropped the `original_degraded_task` and `original_lexical_overlap_check` captures | neither replacement invokes them |
| `backend.py` `lexical_overlap_problem` patch | made unconditional | `writer_quality.py` reads it off the module; it was guarded on an original that no longer exists |
| `backend.py` `_substantive_safe_degraded_task` | signature `(module, original)` → `()` | body opened with `del module; del original` |
| `task_distribution.rebalance_card_surfaces` | dropped the `core_rebalance` parameter | discarded on entry |
| generator `write_reproducibility_snapshot` | resolves prompt builders by name at run time, records `(absent)` when unbound | it named five functions that no longer exist, and recorded pre-patch sources that never rendered anything |
| `core_contract.py` | re-pinned `generator_generalized_v2`, `generator_adapter`, `task_distribution` | intentional reviewed change |

`GENERALIZED_V2_GENERATION_POLICY_VERSION` is deliberately **not** bumped: no
generation behavior changed, and run provenance still records each file's
sha256. It bumps when behavior does.

## Stage 1 — tests updated (3)

- `test_generator_adapter_changes_only_declared_core_extensions`: dropped
  `rebalance_tasks_for_diversity` from the expected changed-symbol list.
- `test_degraded_retry_preserves_normalized_planner_slot`: calls
  `_substantive_safe_degraded_task()` with no arguments.
- `test_matched_real_length_overrides_low_info_fragment_shape` and
  `test_direct_reply_receives_parent_semantic_exclusion_without_replanning`:
  now configure the backend instead of using the raw module. Both were asserting
  on unpatched CARD behavior that never runs.

That last change surfaced a real asymmetry worth carrying into the audit: on the
configured path a reply keeps the Planner's `decision_boundary` but its
`semantic_move` is still overwritten by `reply_novelty_anchor`. Only one of the
two fields is planner-owned.

## Stage 1b — unused module state

Removed nine module constants no live code read: `REAL_SURFACE_SHAPES`,
`CONTEXT_TRANSFORMS`, `CALM_TONE_SHAPES`, `HARSH_TONE_SHAPES`,
`POLITE_REAL_TONE_SLOTS`, `CONCRETE_PRODUCT_PATTERNS` (the finance-era product
regex bank) and `TONE_TARGET_INSTRUCTIONS` (CARD's polite/calm instruction
table), plus the unused `tempfile` and `Counter` imports.

`UTTERANCE_MODES` and `SURFACE_TEXTURES` went with them and were **restored**:
no run-time code reads them, but `test_all_writer_control_paths_are_free_of_finance_prompt_residue`
enumerates them to sweep every writer-prompt variant, which is worth keeping.
They now sit next to the `infer_*` functions that produce those values, with a
comment saying they document a field domain rather than drive behavior.

## Stage 2 — the engine split into a package

`scripts/sampling_generator/` is now a package. The facade keeps the pipeline
the adapter patches; each engine module owns one concern.

| File | Lines | Holds |
|---|---|---|
| `run_sampled_reddit_generator.py` (facade) | 2,059 | the pipeline spine: `main`, planning, slot expansion, post generation, the Writer lifecycle and validation |
| `engine/slot_inference.py` | 454 | payload/voice/role/utterance/texture/tone inference for a matched real slot |
| `engine/persistence.py` | 380 | discussion bundle, markdown, summaries, global memory, run manifest, `task_to_dict` |
| `engine/cli.py` | 352 | argument parsing, client construction, seed-pool loading, failure records |
| `engine/thread_structure.py` | 295 | thread targets, tree ordering, matched-comment selection, planner batching |
| `engine/writer_validation.py` | 254 | length floors, opener and template signatures, placeholder and anchor checks |
| `engine/vocabulary.py` | 250 | field enumerations, length-bucket bounds, system prompts |
| `engine/writer_request.py` | 172 | plan block and control rendering, temperature, token ceilings, text cleanup |
| `engine/anchors.py` | 146 | concrete-anchor extraction and deduplication |
| `engine/context_policy.py` | 126 | context transform choice and the default reply delta |
| `engine/model.py` | 107 | `SeedPost`, `ThreadTarget`, `BranchPlan`, `CommentTask` |
| `engine/util.py` | 124 | text, vocabulary, JSON, and RNG helpers |
| `engine/parent_alignment.py` | 60 | nearest generated ancestor and parent-aligned task fields |

### What the facade still holds, and why it is 2,059 lines

A definition can only leave the facade if it does **not** call a name the
adapter patches. Python resolves a bare call through the calling module's
globals, so `backend.py` rebinding `module.chat_completion_text` reaches only
code that lives in the facade. Twenty-six definitions call at least one patched
name — `main`, `plan_thread`, `plan_comment_move_batch`,
`expand_matched_real_sample_to_tasks`, `generate_post_from_tasks`,
`generate_writer_text_with_guards`, `validate_writer_text` and the rest of the
spine — so they stay together. `GENERATOR_NAME` and `CLAIM_FAMILIES` stay for
the same reason: the adapter rebinds both, while `SYSTEM_PROMPTS` is only
mutated in place and could move.

Shrinking the facade further means dissolving the patch layer itself, which is
Stage 3 and a behavior-risk change, not a move.

### Verification

- **165 of the 166 surviving definitions are byte-identical** to the originals.
  The one exception is `write_reproducibility_snapshot`, edited on purpose.
- `pytest generalized_card/tests/` holds at **209 passed** after every step.
- `run_generator_backend.py --self-test` passes end to end.
- All 13 engine files are pinned in `core_contract.py` and checked on load, so
  drift anywhere in the engine still fails closed. Verified by appending one
  comment to `engine/util.py` and watching the self-test refuse to run.

