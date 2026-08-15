# Plan: fix what blocks a real-looking discussion (post-v74)

Diagnosis is in `tasks/generator_audit.md` under "v74 content audit" and
"v75 root-cause audit". Every number below is measured from the v74 run
artifacts, not inferred.

## The finding that reorders everything

The validation layer measures, logs, and then accepts. It is not a tuning
problem; it is off.

- `--writer-retries 0` -> 1 model call per slot. 519/522 slots ran exactly 1 attempt.
- 231/522 slots (44.3%) were accepted through
  `accepted_first_pass_distribution_diagnostics` on a **known-failing** attempt.
- Only 4 codes can block anything: `exact_duplicate`, `parent_copy`,
  `placeholder_literal`, `planner_skeleton_residue`. The other 16 are advisory.
- `recovered_after_exhaustion` is False 522/522.
- `joint_target_distance` is **worse** for the accept-anyway class (0.667) than
  the clean class (0.498): the path systematically keeps the worse candidate.
- Two validators that fire and are ignored map straight onto the failing
  metrics: `missing_concrete_anchor` 84x (concreteness), `lexical_overlap_high`
  79x (self_bleu).

So: turn on what exists before adding anything new.

## P0 - turn the validation layer on  (no new concepts)

- [ ] The Writer's distribution target carries only two metrics
      (`run_generate.py:488`): `["self_bleu_4", "semantic_mean_cosine"]`, and the
      policy field next to it says `"single Writer realization; distribution
      metrics are diagnostic"`. `polite_rate`, `impolite_rate`, `neutral_rate`,
      `emotion_entropy`, `mean_story_probability` and `length_cv` are dropped at
      the Writer boundary. Six of the twelve metrics have no writer-side control
      loop at all - which is the structural reason they are the ones failing.
- [ ] Raise `--writer-retries` above 0 and re-check cost per thread.
- [ ] Move `lexical_overlap_high`, `missing_concrete_anchor`, and
      `template_phrase_reused` out of `SINGLE_STAGE_DIAGNOSTIC_PROBLEMS`
      (`writer_quality.py:28-40`) so they can force a retry.
- [ ] Fix B4: `shape_writer_text_for_task` (`run_sampled_reddit_generator.py:1997-1999`)
      picks `micro_options[local_task_id % 6]` deterministically, and repair never
      changes `local_task_id`, so `exact_duplicate` recurs forever. Two comments
      were permanently lost in v74 and it poisons `self_bleu` (similarity 0.9999998).
- [ ] Add `empty` to the core blocking set (`run_sampled_reddit_generator.py:1688-1707`).

## P1 - plan-echo guard  (the v73/v74 regression)

- [ ] New validator: first-sentence 4-gram overlap against `task.semantic_move`.
      Register in `HARD_REALIZATION_PROBLEMS`, keep it OUT of the single-stage set.
- [ ] Remove `task.semantic_move` from `has_task_anchor_overlap`'s anchor source
      (`writer_validation.py:247-274`). Today copying the plan verbatim *satisfies*
      the anchor check, so echo is rewarded.
- [ ] Extend the audit `evaluable`/`healthy` gate (`audit.py:217-228`) to see plan
      echo. A thread of pure echoes currently scores healthy.

## P2 - stop the Planner writing the comment  [DONE, v75 change 1]

Shipped ahead of P0/P1 because tracing the acceptance path showed an echo
validator added first would have dropped up to 130 slots: a code outside
`REPAIRABLE_WRITER_PROBLEMS` returns `skip: True` at `backend.py:2022`, repair
exhaustion does the same at `backend.py:2205`, and the only channel that reaches
the focused prompt (`retry_note`) is empty under `--writer-retries 0`. Details in
`tasks/generator_audit.md`, "v75 change 1".

- [x] `--writer-route-lock own_words|say_only`; `say_only` reproduces v74.
- [x] Route lock no longer says "Say this, and only this".
- [x] Reply schema no longer demands a finished sentence; scale clause kept.
- [x] `_realization_rule` restores the counterweight, on the focused **and**
      low-info paths (106/522 slots take the latter).
- [x] `micro_reaction` pool widened and text-keyed: two v74 comments were being
      dropped by a deterministic collision.
- [x] 224 tests pass; the route lock previously had zero test coverage.
- [ ] Constrain the planner's register further: `prompts.py:694-695` addresses the
      planner as the participant ("what happened when you personally used X"), and
      `reply_planning.py:71` defines `corroborating_datapoint` as "your own
      concrete experience". These are why 19.3% of moves open with "I". Left for
      after measuring change 1, so the two are separable.

## P2-old - original notes

- [ ] `reply_planning.py:255` asks for "a full sentence stating what this reply
      asserts - not a bare noun phrase". Root schema (`prompts.py:572`) asks for
      "one concrete but non-verbatim action". The two contradict; reply slots echo
      at 25.1%, root slots at 6.4%. Align reply on the root wording.
- [ ] Delete `"- Say this, and only this: "` (`prompts.py:2266`).
- [ ] Restore the semantic-difference contract dropped by `_focused_writer_prompt`.
      It was cut because "no metric depended on it" - no metric measures plan echo.
- [ ] Constrain the planner's register: no first person, no finished sentence.
      19.3% of 522 moves begin with "I". Source: `prompts.py:694-695` addresses the
      planner as the participant ("what happened when you personally used X"),
      and `reply_planning.py:71` defines `corroborating_datapoint` as "your own
      concrete experience".

## P3 - give the reply planner sibling visibility, and dedupe moves

- [ ] Every depth>=1 batch takes `render_direct_reply_planner_prompt`, which
      renders no prior-plan ledger, no coverage summary, no sibling contract, no
      branch goal, no R# rows. Each row sees only its parent. Verified on seed 2:
      depths 3,4,5,6,7,8 are single-slot batches, and tasks 38-45 are the nine
      near-duplicate moves.
- [ ] Add a `semantic_move` similarity check. There is none anywhere today; the
      whole-plan `semantic_collision` check is diluted to ~10% token mass by
      `development_plan`, exempts parent-child pairs via `_dependent_variation`,
      and when it does fire it only warns (`backend.py:1531-1557`).
- [ ] Fix the beat-budget contradiction: prompt says one beat per 35 words capped
      at 16 (`prompts.py:733-738`); validator demands `round(words/21)-1`
      (`long_form_planning.py:29-30`). A 300-word slot: prompt 8-9, validator 13.
      Following the prompt guarantees failure, and the surplus beats are what
      dilute the collision detector.

## P4 - persistent speaker identity  (story + concreteness + emotion)

- [ ] There is no speaker identity today: `speaker_role` is a 10-value enum,
      `persona_conditioning=none`, and gear is a 4-item shortlist keyed by
      `slot_index` with no continuity.
- [ ] Split the grounding rule, which currently conflates two different things:
      facts about the seed product must stay grounded; facts about **my own kit
      and history** should be free to invent. Today both are banned
      (`prompts.py:113-115`, `prompts.py:1286`), which is why 57 story slots
      contain a spec 5% of the time, a time marker 2%, a place 4%.

## P5 - polite: undo a design decision the data refuted

`generation_distribution.py:473-478` records the reasoning behind the current
`polite` definition:

> A softener-and-hedge reading of "polite" produced the tentative register the
> classifier scores as somewhat_polite, so the distinction is made explicit here
> rather than left to the model's prior.

So `TONE_DEFINITIONS["polite"]` (`generation_distribution.py:480-489`) now says
*"Do not hedge the positive judgement into a maybe, and do not use
customer-service phrasing or a template thank-you"*, and
`prompts.py:2791-2793` repeats it.

The prediction was that polite would collapse into `somewhat_polite`. The
measured collapse is into **impolite**, 65% of its slots, with polite realized
7.4% (recorded at `prompts.py:1243-1244`). Stripping hedges, thank-yous and
customer-service phrasing does not yield warmth; it yields flat assertion, which
polite-guard scores as impolite. The hypothesis was wrong, so the rule should go.

- [ ] Restore hedges and gratitude as licensed polite surfaces.
- [ ] Give polite slots real length. `TONE_SCOPE_HINTS`' own comment
      (`generation_distribution.py:508-511`) records that the classifier's polite
      class is length-dependent - 52% of 60-120 word comments, 64% above 120 -
      and generated comments have a median of 33 words. This lever is measured
      and currently unused. Interacts with the 220-word ceiling in P6.
- [ ] tone_target is scheduled independently of `semantic_move`: 4.8% of slots are
      `impolite` over a helpful move. Assigned polite 26.2% -> realized 8%.
- [ ] `_substitution_rule` (`prompts.py:2726`) decides the first-person ban from
      `tone_target` alone and never reads `task.allow_first_person_frame`, which
      `run_sampled_reddit_generator.py:1022` already computed from the matched real
      comment. 84 slots (16.1%) are banned from a frame their real counterpart used.
- [ ] Dead tone machinery to remove or wire up: `tone_overlay_slot` /
      `tone_overlay_instruction` are read in five places and assigned nowhere;
      `_delexicalize_tone_examples` (`backend.py:1086-1106`) matches three strings
      that no longer exist in `TONE_DEFINITIONS`; `tone_target ==
      "constructive_polite_helpful"` (`run_sampled_reddit_generator.py:1299-1308`)
      can never be true since `TONE_CLASSES` has four other values.

## P5b - story: the allocation is right, the instruction is not

Correcting an over-claim: `specific_personal_story` is 44 of 79 story slots
(56%), not rare. The per-thread story **count** is scaled from the matched real
thread's own `story_rate`, which is why `mean_story_probability` passes. The
allocation is correct and should not change.

The content is the problem. `_story_instruction`
(`generation_distribution.py:323-329`) asks for "a setting, an action, a small
friction or change, and a local reaction", and `_story_fact_safety_rule`
(`prompts.py:2707-2714`) then forbids "a product, specification, price,
measurement, date, policy, link, diagnosis, or externally checkable outcome" -
i.e. every category that would make those specific. The system prompt
(`prompts.py:1448`) adds "realize only a **qualitative** synthetic context".

- [ ] "I've done that in a packed room before" is the *compliant* output. Measured
      across 57 story slots: 5% contain a spec, 2% a time marker, 4% a place.
- [ ] Fix by scoping the ban to claims about the seed product (see P4), not to the
      speaker's own history. A consequence is currently banned as an "externally
      checkable outcome"; a story without a consequence is not a story.

## P6 - surface realism and the unconverted path  (lowest priority per user)

- [ ] Convert `_low_info_writer_prompt`: 106/522 slots, mean 15,468 chars, dumps
      11 internal labels, produced 9-word outputs.
- [ ] `LENGTH_BUCKET_BOUNDS["very_long"] = (120, 220)`
      (`engine/vocabulary.py:195`). Real threads exceed 220 words in 10/10.
- [ ] Drop the single-paragraph instruction (39.1% of prompts) - real
      multi-paragraph rate is 32.8%, generated 3.1%.
- [ ] Straight apostrophes: 68.1% of generated comments carry curly typography vs
      10.9% real; model-emitted, identical pre/post cleanup.

## Rejected: two writers supervising each other

Not as an LLM critique loop. Two reasons. Cost doubles, and critique-driven
revision pushes text toward the balanced, hedged, "on the other hand" register,
which is the exact failure mode. The stronger reason: the system already has 20
validators and ignores 18 of them. Adding an LLM to generate more advice that
gets ignored is the wrong move.

The useful form of "supervision" is a deterministic discriminator on the checks
above - free, reproducible, aimed at the measured gaps. If an LLM goes in the
loop later, the shape is re-voicing a draft as a different speaker with a
different kit, with no critique language in the prompt, measured on its own.

## Sequencing

P0+P1 first: one variable each, both cheap, both independently attributable.
P2+P3 next as one change with an ablation flag. P4-P6 after, one at a time.
Every step keeps a `--writer-prompt`/`--domain-claim` style flag so the previous
version stays reproducible.
