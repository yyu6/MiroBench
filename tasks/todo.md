# Plan — make the discussion read like people talking

Rewritten 2026-08-16. Read `tasks/HANDOFF.md` first; this file is the task list,
that file is the evidence.

The previous version of this file ordered work by **where a code defect lives**.
This one orders it by **which measured gap it moves**, because three of the last
four paid runs fixed a real code defect and moved no metric. The old P0–P6 items
are all still listed, at the bottom, marked kept / struck / demoted, so nothing is
silently dropped.

## The ordering principle

The user's target is how people talk, decomposed into four dimensions. Mapping
them onto the 12 metrics and onto the per-thread evidence in `HANDOFF.md` §4.3:

| dimension | metrics | per-thread state |
|---|---|---|
| 1 semantic dispersed | `semantic_mean_cosine` | 5/10 threads within 20% — passes by cancellation |
| 2 low lexical overlap | `self_bleu_4`, `self_bertscore` | bleu 2/10; bertscore 10/10 within 20% but fails on a uniform +0.03 |
| 3 stories in first person | `mean_story_probability` | 3/10; overshoots 1.5–2.4× |
| 4 tone and emotion varied | `emotion_entropy`, polite/impolite/neutral, `hard_disagree_rate` | 0–3/10 on every one |

Only `avg_depth` and `structural_virality` are genuinely matched per thread, and
both are fixed by the matched sampler rather than won by generation.

## v82 implementation status — 2026-08-17

Detailed evidence and exact scorer definitions are in `tasks/v81-worklog.md`;
the completion-audit fix is recorded in `tasks/v82-worklog.md`.
This supersedes the older idea that a repetition warning should resample one
comment: collection-level metrics are diagnostic in first-pass generation.

- [x] Remove copied short-slot `development_plan` prose before the Writer.
- [x] Make story/no-story a bidirectional Planner contract, including
      `evidence_mode`; stop unresolved hard contracts before the Writer.
- [x] Put story/tone/affect/opener controls into direct-reply planning rather
      than adding them only after its semantic plan is written.
- [x] Remove the canned gratitude semantic rewrite and automatic
      `soft_helpful` conversion; targeted Planner repair owns those choices.
- [x] Jointly pair tone, affect, and story marginals. On the frozen v80 large
      template all labels remain assignable and the measured contradictory
      pairs collapse sharply.
- [x] Remove the duplicated focused-Writer tone block and the neutral-affect
      instruction conflict.
- [x] Disable per-comment distribution retries and repetition best-of-N at the
      v81 public CLI; retain bounded recovery only for non-persistable output.
- [x] Restore the compact Planner discourse contract on the default focused
      Writer path. Function, payload, role, voice, evidence, content angle, stance,
      detail, intent, reply relation, and exclusion now survive once; a raw-plan
      end-to-end test prevents a planned rant from collapsing into generic help.
- [x] Remove matched-text tone leakage from surface inference: links, quotes,
      capitalization, emoji, and punctuation remain anonymous shape, but lexical
      gratitude no longer creates a `pure_acknowledgement` contract.
- [x] Finish source-pin/version updates and full code review: 262 tests pass,
      backend self-test passes, and all 72 source pins have zero drift.
- [ ] Run one large n=1 content/contract diagnostic with no metric-driven
      retries, then run a multi-thread matched evaluation for formal p-values.

---

## A — realize the assigned register   [dimension 4, largest gap]

**Why.** Measured on v79, 184 aligned slots: assigned `impolite` realizes at 93%,
assigned `polite` at 13% with 59% collapsing into impolite. Overall realization
59.2%. One register per thread explains polite ↓, impolite ↑, hard_disagree ↑ and
emotion_entropy ↓ **simultaneously** — these are not four problems.

**What the data already eliminated** (do not redo):
- *Not length.* Real polite is 52% of 60–120 word comments; generated is 6%, and
  0% above 120 words. Generated long comments are 73–88% impolite.
- *Not insufficient agreement.* Real comments carry more negation than generated
  (41.5% vs 31.2%) and are still scored polite.

**What differs** (seed 8, real vs v79): warm markers 14.0% vs 11.8%; emotional
endpoint 2.5% vs 1.1%; hedge 18.0% vs 12.9%; decision-framing nouns 0.5% vs 4.3%.

**Tasks**
- [x] Delete the hedge and thank-you prohibitions from `TONE_DEFINITIONS["polite"]`
      (`generation_distribution.py:480-489`). The block above the table records
      the prediction that motivated them — collapse into `somewhat_polite`. The
      measured collapse is into **impolite**, so the prediction was wrong.
- [x] License the emotional endpoint explicitly ("I love it", "never looked
      back"): real 2.5%, generated 1.1%.
- [x] Cut decision-framing nouns from the Writer's own rule text — 8.6× overshoot,
      and the Writer is substituting analysis for feeling.
- [x] Check `_affect_instruction` rotation (`generation_distribution.py:448-470`)
      still reaches the prompt on the focused path; the affect rewrite was
      bundled into v73 and never cleanly attributed.
- [x] Ablation flag; `off` restores the pre-v80 contract and is recorded in
      `run_config` as `social_contract_coherence`.
- [x] **Offline gate before any paid run:** re-render the v79 prompts from
      `generation_records.json[].task` and confirm the banned surfaces are gone
      and the new ones present.
- [ ] **Judge the paid run on tone realization rate (59.2% baseline) and
      `emotion_entropy`**, not on p-values. n=1 has no p-value.

---

## B — the global typographic signature   [dimension 2, free]

**Why.** `self_bertscore` has never passed in any version, but §4.3 shows it is
not a large error: 6.9% mean relative error, 10/10 threads inside ±20%, failing
only because all ten overshoot by a near-uniform +0.03. That is the signature of
one global constant, not of content.

The strongest available candidate: **every generated comment carries the same
typography.** Of comments containing an apostrophe, 100% of generated use only
curly `’`, against 17.6% of real. Curly overall: generated 72–74%, real 11–13%.
Straight apostrophe inside a word: real 51%, generated 0%. Verified
model-emitted, identical before and after `gpt_cleanup`.

**Tasks**
- [x] Do not add deterministic apostrophe normalization: the actual-scorer
      counterfactual below explains only a minority of the gap, so this is not a
      justified primary fix and would add post-processing without fixing talk.
- [x] Run a no-regeneration counterfactual first: on 40 v79 comments / 780 pairs,
      curly apostrophes -> ASCII moved 0.52947 to 0.52381. This explains only a
      minority of the ~0.034 gap, so do not implement a held-out-calibrated
      normalizer as the primary fix.
- [x] Address the current rough-surface gap upstream: impolite slots explicitly
      allow ordinary non-targeted profanity and amusement slots allow a natural
      laughter token. Neither is required or hard-coded to one phrase.
- [ ] Re-measure the other surface gaps after v81, all measured on seed 8:
      paragraph breaks real 25.5% vs generated 2.8%; no final punctuation 24.0%
      vs 6.6%; URLs 4.5% vs 0%; `lol/haha` 3.0% vs 0%; ALLCAPS 19.5% vs 7.7%.
      These are prompt-level, not post-processing, so keep them separate from the
      typography change if attribution matters.

---

## C — bring `mean_story_probability` down   [dimension 3]

**Why.** Generated overshoots real by 1.5–2.4× on seed 8. Real per-thread
`story_rate` ranges 0.000 (seeds 0, 3, 5) to 0.275 (seed 6), mean 0.110. The
previous handoff said the allocation was correct and should not change; that was
wrong.

**Before changing allocation**, note how the metric is computed: StorySeeker's
P(story) averaged over **every** comment in the thread, not only story slots. So
non-story comments drifting narrative would produce the same overshoot. The
per-thread story count already scales from the matched template
(`generation_distribution.py:129-134`).

**Tasks**
- [x] Score story-mode slots and no-story slots separately in an existing run to
      see which class carries the overshoot. Offline, the per-comment
      probabilities are already in `cleaned/*/storyseeker_results.json`.
- [x] Diagnose realization as the main failure: planned story slots supplied
      only ~25% of total story probability; 25/167 `no_story` comments were
      classified as stories. Add a Writer no-sequence contract and reject
      `no_story + personal_story` plans before writing.
- [x] v81 root fix: no-story also rejects `firsthand_experience`; scheduled
      stories require a coherent personal-datapoint evidence plan in both root
      and direct-reply planners. v80 replay exposes 59 latent conflicts.
- [x] Keep first person for the slots that do tell stories: every scheduled
      story task sets `allow_first_person_frame=True`, while the joint contract
      requires firsthand personal-datapoint semantics.

---

## D — the two arms that are built but never run

- [ ] **`--own-fact-license named`.** Domain-neutral concreteness: name things
      and give quantities. Gated on `substantive_slot` (≥25 real words, not
      micro/short). Targets the two signals that separate real from generated on
      all ten threads — quantities 12.3×, proper nouns 1.85×. Note the sibling
      arm `own` was refuted; see HANDOFF §6.6.
- [ ] **`--speaker-identity matched`.** 265 named participants over 559 real
      comments, 2.11 each, 68% of comment mass from someone who speaks more than
      once; the generator gives every comment a distinct one-shot author. Targets
      `self_bertscore` through voice variation. **Run B first** — it may explain
      the same metric for free.

---

## E — reply-planner sibling visibility   [was P3, kept]

- [x] Every depth ≥ 1 batch takes `render_direct_reply_planner_prompt`
      (`prompts.py:336-381` routes there; batches never mix depths, so all of
      them qualify). It renders no prior-plan ledger, no coverage summary, no
      sibling contract, no branch goal, no R# rows. Each row sees only its
      parent. Verified on seed 2: depths 3–8 are single-slot batches and tasks
      38–45 are nine near-duplicate moves that could not see each other.
- [x] Add `--reply-sibling-visibility`; the `on` arm renders every sibling and
      already committed delta/novelty object, while `off` restores old rows.
- [x] Do not add a stricter `semantic_move` gate without new evidence. v80 seed
      8 semantic cosine is already near real, sibling visibility now exposes
      competing reply increments, and a stronger gate could over-disperse it.
      The existing whole-plan
      `semantic_collision` check cannot catch it: `plan_similarity` is a Jaccard
      over all `SEMANTIC_FIELDS` including `development_plan`, so a ~20-token
      move is ~10% of the token mass; `_dependent_variation` exempts parent–child
      pairs. Reopen only if v81 text repeats sibling moves or semantic cosine
      moves above real.
- [x] Fix the beat-budget contradiction: root and direct-reply prompts now use
      the dynamic count rendered from `expected_development_beats`; no separate
      35-word/16-beat rule remains in prompt prose.

---

## F — turn on more of the validation layer   [was P0, demoted]

Resolved in v81 by simplifying the policy instead of promoting more metrics:

- [x] Distribution and repetition findings are diagnostics and never resample,
      rank, select, or drop a persistable comment.
- [x] `missing_concrete_anchor` remains an audit signal, not a retry trigger.
- [x] Empty text is explicitly classified as a hard realization failure even
      when the shared Writer omits the problem code; bounded hard recovery
      handles it before persistence.
- [x] Delete the old repairability sets, candidate ranking, repetition arm, and
      dead CLI parameters rather than retaining an unreachable controller.

**Corrected from the previous version of this file:** "add the 6 missing metrics
to the Writer's distribution target (`run_generate.py:488`)" — that line is a
record written into `run_config.json`, not a wire. The real target is hard-coded
in `generation_diversity.build_thread_distribution_target:40-43`. In v81 the
target is audit-only and no longer ranks candidates. Five of the six also need
transformer classifiers inside the generation process. See HANDOFF §6.1.

---

## G — simplify without erasing history

- [x] Preserve v68-v80 behavioral provenance in `VERSION_LOG.md` and compare all
      behavior fields during resume/extension/policy upgrade.
- [x] Run repository-wide reference and AST audits before deletion.
- [x] Remove unreferenced reviser prompt builders and stale helpers while keeping
      every active reviser entry point.
- [x] Re-pin all changed generalized sources and commit only the scoped files;
      unrelated dirty worktree content belongs to other sessions.

---

## Struck, with the measurement that struck them

- **Plan-echo validator (was P1).** Echo is at 0.0% since v75 and the route lock
  that achieved it moved no metric. Nothing left to guard.
- **Length for polite slots (was in P5).** Real polite is length-driven; the
  effect does not transfer. Generated 60–120 word comments are 6% polite, 120+ are
  0%. HANDOFF §5.2.
- **`LENGTH_BUCKET_BOUNDS["very_long"] = (120, 220)` (was in P6).** Read only by
  `_retry_note_for_problems`, i.e. on a retry, which under `--writer-retries 0`
  almost never happened. Dead. `length_cv` is within 3.5% of real anyway.
- **`--own-fact-license own` (was P4a).** Refuted in v76b: 0.05 → 0.02
  specification tokens per comment against a real 0.54. 68% of real
  spec-carrying comments have no first-person frame, so the gate was wrong.
  Retained only as a reproducible arm.
- **B7 "`allocate_story_and_affect` is a no-op auditor".** Not a bug; it is a
  deliberate auditor, documented at `generation_distribution.py:108-114`.

## Still open from the old audit, unranked

`B2` beat-budget contradiction (now in E), `B3` `allow_first_person_frame`
computed and never read by `_substitution_rule`, `B4` `tone_overlay_*` read in
five places and assigned nowhere, `B5` `_delexicalize_tone_examples` matches
strings that no longer exist, `B6` `constructive_polite_helpful` unreachable,
`B8` template overrides swallowed by `apply_slot_distribution_schedule`, `B9`
dead validations, `B11` `perspective_id` repair impossible but budgeted, `B12`
repair feedback references a block the reply prompt lacks, `B13`
`--writer-hard-recovery-rounds` never exercised.

None of these has a measured link to a failing metric. Fix them when touching the
surrounding code, not as a campaign.

## Sequencing

B (free, testable with no generation) → A (largest gap, offline-gated) → C
(diagnose before changing) → D → E. One mechanism per paid run, prediction
written down first, `off` byte-identical, dry-run before handing over a command.
