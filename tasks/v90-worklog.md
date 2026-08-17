# v90 worklog — reply-story grounding parity

Date: 2026-08-17

Policy: `generalized-card-v2-reply-story-grounding-v90-20260817`

## Trigger

The v89 zero-API completion audit compared the complete specialized
direct-reply Planner with root planning and Writer grounding. Root planning and
the Writer explicitly allowed an ordinary, non-verifiable synthetic personal
sequence in scheduled story slots. Direct-reply planning required an actual
event sequence, prohibited carrying source-participant details, and prohibited
inventing seed facts, but never stated the allowed middle ground.

This was a current-path issue, not a historical inference: every depth >= 1
batch routes through `render_direct_reply_planner_prompt` when its parent plan
is committed.

## Change

1. Define `SYNTHETIC_STORY_PLANNER_BOUNDARY` once in `reply_planning.py`.
2. Render the same boundary in both root and direct-reply Planner Prompts.
3. Keep the factual boundary narrow: synthetic non-verifiable personal
   sequence is allowed; product facts, measurements, dates, links, diagnoses,
   and externally checkable outcomes are not.
4. Keep the Writer license unchanged. Its off-mode story contract already made
   this distinction.
5. Add a rendered direct-reply Prompt regression covering the permission, the
   externally-checkable boundary, and the existing seed-fact prohibition.

## Adjacent-boundary review

The final review initially classified direct Planner `domain_claim` handling
under `--domain-claim off` as an intentional Writer-boundary ablation. v92
corrected that conclusion: even though no claim permission reached the Writer,
planning a fact and then dropping it still created a handoff gap and wasted
Planner Prompt/output tokens. See `tasks/v92-worklog.md`.

## Verification status

- Focused route-lock/grounding tests: 17 passed.
- Full `generalized_card/tests`: 295 passed.
- Focused Self-BERTScore scorer tests: 3 passed.
- Ruff on every changed Python source and test: passed.
- Camera backend self-test: passed with matched speakers, focused Writer,
  `own_words`, social coherence on, sibling visibility on, and own-fact license
  off. Hugging Face was forced offline to use the local model cache.
- Active and active-plus-legacy parity: healthy.
- Core closure: 93 declared pins, zero missing files, untracked active files,
  unpinned local imports, or drift.
- Exact seed-8 prepare-only:
  `generalized_card_camera_gpt54_v90_preflight_seed8_20260817_v1`; the policy and
  requested behavior flags match, and no API call was made.

No v89 or v90 API call has been made.
