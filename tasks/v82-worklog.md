# Generalized CARD v82 worklog — 2026-08-17

## Why this follow-up exists

After v81 passed its unit, pin, self-test, and prepare-only gates, the final
Planner-field audit traced every default Writer path instead of stopping at task
construction. It found that `focused`—the public default—received the semantic
move, decision boundary, reply delta, tone, affect, story, opener, length, and
grounding controls, but omitted the Planner's discourse role:

- `comment_function`
- `payload_type`
- `speaker_role`
- `voice`
- `evidence_mode`
- `content_angle`
- `stance`
- `detail_focus`
- `domain_intent`
- `reply_relation`
- `avoid_repeating`

The full and low-information paths already carried these controls. This was a
focused-path information-loss bug, not evidence that the Planner itself needed
another semantic rewrite. It directly permits planned rants, corrections,
datapoints, and reactions to collapse into generic helpful advice.

The same trace exposed an indirect matched-text leak. The shared
`infer_surface_texture` treated lexical gratitude markers in an evaluation
comment as `gratitude_social`; `_generic_real_tone_slot` then promoted that to
`pure_acknowledgement`. Although matched wording was never copied into the
Prompt, its social meaning still overrode the Planner through a derived label.

## Fix

`prompts._focused_slot_contract` renders those fields once in a short block.
Exact duplicate values are suppressed. The change does not restore the full
semantic contract, static metric guidance, five overlapping surface label
paraphrases, or the bulky payload guidance block, so the v74 prompt-size and
anti-repetition rationale remains intact.

The generalized adapter now owns surface-texture inference. Matched text may
contribute only typography: link/quote presence, emoji or `/s` notation,
capitalization, missing terminal punctuation, and messy punctuation. Gratitude,
joke, and tangent textures come only from the Planner's payload/speaker role.

## Verification

- A focused prompt test constructs a valid
  `rant + ranter + hard_disagree + impolite` slot and asserts that every compact
  contract row occurs exactly once.
- A separate end-to-end test starts with raw Planner JSON, calls the configured
  normalizer, expands a real anonymous matched slot, finalizes its `CommentTask`,
  and renders the public focused Writer prompt. It proves the planned fields are
  unchanged at the task boundary and present exactly once at the Writer boundary.
  Its anonymous matched body intentionally contains `thanks`, `appreciate`, and
  `good to know`; the resulting rant remains `surface_texture=plain` and never
  receives a `pure_acknowledgement` tone slot.
- Focused/full prompt-size regression remains covered by the existing suite.
- No matched real comment wording is passed to the Writer, no Writer candidate
  ranking is reintroduced, and no metric-driven local retry is added.
- Complete `generalized_card/tests`: **262 passed**.
- Ruff passes on every changed generalized production and test module.
- Backend self-test: `[generalized-self-test] PASS domain=camera_product`.
- Core contract: 72 files checked, 0 missing, 0 drifted.
- The exact v82 seed-8 command passed `--prepare-only` under policy
  `generalized-card-v2-focused-discourse-contract-v82-20260817`; no API calls
  were made and the formal run tag remains unused.

## Expected result and limitation

The content-level expectation is fewer assistant/customer-service turns and
more faithful rants, corrections, questions, datapoints, and low-polish social
moves. That may improve Self-BLEU, Self-BERTScore, hard disagreement, and emotion
distribution by removing one shared helpful-answer default. It does not change
the story quota, matched word-count signal, or reply-tree sampler. These are
predictions, not measured outcomes; the 12-metric target still requires a new
multi-thread evaluation.
