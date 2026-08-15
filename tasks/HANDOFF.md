# Handoff — synthetic Reddit thread generation (generalized_card)

Written 2026-08-15, after run v75. Read this **plus** `tasks/generator_audit.md`
(the evidence) and `tasks/todo.md` (the plan) before touching anything.

Every number here is measured from run artifacts, not inferred. Where a claim is
uncertain it says so.

---

# 1. THE GOAL AND THE JUDGING STANDARD

Generate synthetic Reddit threads that are statistically indistinguishable from
real ones across **12 thread-level metrics**, using `generalized_card/` (a
domain-configured Planner–Writer implementation of CARD).

## The judging standard (user's own words, authoritative)

- **A metric is "matched" only if MWU p > 0.05 AND KS p > 0.05, and the p-value
  is comfortably large.** Barely above 0.05 does not count.
- The user explicitly **rejected N-based extrapolation**: "我不希望你用根据 N 的
  大小来测试的方法，除非这个方法是 publicly scientifically proved". Do not argue
  "this would pass at N=150".
- Only two questions matter: **(1) is the p-value big enough now, (2) content-wise
  how real does it actually look.**
- The p-values that count as genuinely matched look like `avg_depth` p=1.00,
  `semantic_cosine` p=0.97, `mean_story_probability` p=0.97.
- Final target: **150 threads per domain.** Current runs are 10-thread smoke
  tests at ~$2–6 each.

## Priority order (user stated repeatedly)

1. **Language diversity** — `self_bleu_4`, `self_bertscore`, `semantic_cosine`
2. **Emotion** — `emotion_entropy`, `mean_story_probability`
3. **Length** — `length_cv` (explicitly named a priority)

**Politeness is DE-PRIORITIZED.** The user has said this more than once and got
annoyed when I kept analyzing it: "politeness不是重要的，最重要的是其他除了
politeness 的". `polite_rate`/`impolite_rate`/`neutral_rate` still appear in the
12 metrics and still feed `emotion_entropy`, but **do not spend analysis turns on
politeness itself.**

## "Content" means how people talk

"content并不是内容的准确度，而是他们说话方式" — not factual accuracy, but
speech style: word choice, perspective, concreteness, register.

## Other standing constraints

- **Must be domain-generalized, not domain-specific.** A domain adapter is
  expected; better ideas welcome.
- **Style exemplars must be modified real comments**, never verbatim, to avoid
  leakage: "给writer 看到的example 是真实评论修改之后的". Not built yet.
- **Version records must be preserved and reproducible.** Every behavior change
  gets an ablation flag; the previous version must stay runnable.
- **Print findings and plans in chat**, not only in MD files.
- The user runs the API commands: "修改完之后，我来负责测试。"

---

# 2. CURRENT STATE

## Run history, matched-seed evaluation, same 10 real threads

| run | PASS/PART/FAIL | genuinely matched | cost | note |
|---|---|---|---|---|
| v67 | — | — | — | healthy baseline for plan-echo (0.4%) |
| v69 | 8/12 | 4 | — | |
| v71 | 4/12 | — | $3.37 | regression: `domain_claim` bundled with 6 other fixes |
| v72 | 7/12 | 4 | $3.34 | `--domain-claim off` ablation confirmed the cause |
| v73 | 8/12 | 4 | $3.37 | |
| v74 | **7/2/3** | 5 | $2.24 | focused writer prompt (−57% prompt size) |
| **v75** | **4/3/5** | **3 strong** | **$5.99** | route lock fix + an **unintended** second variable |

## v75 full result (current head)

```
self_bleu_4              FAIL     MWU=0.0091   KS=0.0123   Cliff=0.70
self_bertscore_mean_f1   FAIL     MWU=0.00033  KS=0.00022  Cliff=0.96
semantic_mean_cosine     PASS     MWU=0.97     KS=0.99     Cliff=0.02   <-- best ever
hard_disagree_rate       PARTIAL  MWU=0.037    KS=0.17     Cliff=0.56
polite_rate              FAIL     MWU=0.041    KS=0.0123   Cliff=-0.55
impolite_rate            FAIL     MWU=0.00077  KS=0.00022  Cliff=0.90
neutral_rate             PARTIAL  MWU=0.0091   KS=0.052    Cliff=-0.70
length_cv                PARTIAL  MWU=0.038    KS=0.17     Cliff=-0.56
avg_depth                PASS     MWU=1.00     KS=1.00     Cliff=0.00
structural_virality      PASS     MWU=0.97     KS=1.00     Cliff=-0.02
mean_story_probability   PASS     MWU=0.97     KS=0.99     Cliff=-0.02  <-- best ever
emotion_entropy          FAIL     MWU=0.038    KS=0.0123   Cliff=-0.56
```

Genuinely matched (large p): `avg_depth`, `semantic_mean_cosine`,
`mean_story_probability`, `structural_virality`. Three of those four are
structural and constrained by the matched sampler; `semantic_cosine` and
`mean_story_probability` are real generation wins.

**`self_bertscore` has never passed in any version.** It is the single most
stubborn metric (Cliff 0.94–0.96 throughout).

## CRITICAL: v75 is confounded — two variables changed

I intended one change (`--writer-route-lock own_words`). A second slipped in:

```
plan_quality.repair_rounds:  v67/v69/v72/v73/v74 = 0     v75 = 3
```

I omitted `--plan-quality-repairs 0`; the argparse default is 3. Consequences:
- planner repair attempts **0 → 914**
- total requests **680 → 1595** (the 915 delta is almost exactly the repairs)
- cost **$2.24 → $5.99**
- plan `collision_rate` **0.075 → 0.045**

**Any v75 improvement in diversity/story metrics may come from the plan repairs,
not the route lock.** `mean_story_probability` p=0.57→0.97 is most likely the
repairs. To disentangle, rerun v75's code with `--plan-quality-repairs 0`.

## What IS cleanly attributable to the route lock

Plan echo (longest contiguous shared word run ≥12 between a slot's
`semantic_move` and its own comment):

| run | all | comments ≥25w | reply slots | root slots |
|---|---:|---:|---:|---:|
| v67 | 0.4% | 0.0% | 0.6% | 0.0% |
| v73 | 10.2% | 11.7% | 15.2% | 0.0% |
| v74 | **25.8%** | **34.7%** | 32.3% | 12.7% |
| **v75** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |

Plan repairs cannot affect whether a writer copies *its own* plan, and both root
and reply went to zero. This one is the route lock.

## Content diagnostics, generated vs real, all 10 threads

Measured on `generated/` (verified identical to `cleaned/` — cleanup is a no-op).

| tell | real | v74 | v75 |
|---|---:|---:|---:|
| analytic "reviewer voice" frame | 0.15% | 7.23% | **2.29%** |
| comment word-count median | 34.6 | 33.3 | **36.7** |
| max comment words | 262 | 153 | **169** |
| word-count CV (pooled) | 1.01 | 0.87 | 0.86 |
| specs per comment (`f/2.8`, `28mm`, `ISO 1600`, `$900`) | **0.55** | 0.06 | 0.08 |
| novel brand/model tokens per thread | **47.3** | 5.8 | 6.6 |
| contains a URL | 4.5% | 0% | 0% |
| contains an emoji | 2.1% | 0% | 0% |
| `lol/haha/lmao` | 0.6% | 0% | 0% |
| ALL-CAPS word | 22.7% | 6.6% | — |
| blank-line paragraph break | **32.8%** | 3.1% | — |
| ends without final punctuation | 26.4% | 8.4% | — |
| curly `’ “ ” —` typography | 10.9% | 68.1% | **79.4%** (worse) |
| straight `'` inside a word | 44.3% | 0.7% | — |
| meta-discourse about the thread | **0.7%** | 6.9% | **6.9%** (unchanged) |
| imperative opener | 4.7% | 1.65% | 0.48% (worse) |

**The concreteness gap is the largest unaddressed content defect** and it maps
directly onto `self_bertscore`.

## Negative result worth keeping

The novel-entity deficit does **not** explain `self_bertscore`. Seed 6 introduced
*more* novel entities than its real thread and still overshot by 0.034; seed 7
had 6 vs 34 entities and matched to 0.001. The overshoot is a near-uniform
**+0.033 offset on 9/10 threads** — a shared register signature applied evenly,
not topical narrowness. Cliff=0.96 comes from consistent sign, not magnitude.

## self_bleu_4: the PARTIAL→FAIL flip in v75 is NOT a real regression

| | v74 | v75 |
|---|---:|---:|
| threads where generated > real | **10/10** | **10/10** |
| mean excess over real | **+0.0088** | **+0.0089** |

Identical. The metric has always failed in the same direction with the same
magnitude; the PARTIAL/FAIL label flip is distributional noise in an unpaired
test at n=10. Do not "fix" a regression that did not happen.

## gpt_cleanup does nothing

**3 of 521 comments changed (0.6%), all trailing-whitespace strips.** It invokes
gpt-5.4-mini over all 10 threads for zero substantive effect. Verified by diffing
`discussion.json.pre_gpt_cleanup` against `discussion.json`. Candidate for
removal; at minimum do not attribute anything to it.

---

# 3. CODEBASE AUDIT — FINDINGS BY SUBSYSTEM

All line numbers are from the working tree as of 2026-08-15. Three full-file
reads were done (planner path, story/tone path, writer validation path) and their
load-bearing claims were independently re-verified against run artifacts. Where
two reads disagreed, the disagreement is noted with the resolution.

## 3.1 The writer validation layer is effectively OFF (biggest structural finding)

Measured from `logs/writer_distribution_control.jsonl` (522 rows, v74):

| fact | value |
|---|---|
| `--writer-retries 0` → attempts per slot | **519/522 ran exactly 1** |
| accepted via `accepted_first_pass_distribution_diagnostics` | **231/522 (44.3%)** — accepted on a *known-failing* attempt |
| `recovered_after_exhaustion` | **False 522/522** |
| `selected_attempt == 1` | 519/522 (99.4%) |
| `joint_target_distance`, clean vs accept-anyway | 0.498 vs **0.667** — the accept-anyway path keeps the *worse* candidate |

Problems that fire and are then ignored (v74, 525 attempts):
`missing_concrete_anchor` 84, `lexical_overlap_high` 79, `template_phrase_reused`
60, `real_slot_too_short` 24, `low_info_too_long` 22, `question_mark_unwanted` 18,
`opening_reused` 15, `opener_family_reused` 11.

Mechanism: `generalized_card/generalized_card/backend.py:1994-2020` accepts any
candidate whose problems are all "single-stage diagnostics".
`writer_quality.py:28-40` `SINGLE_STAGE_DIAGNOSTIC_PROBLEMS` holds 9 codes;
`writer_quality.py:41` `SINGLE_STAGE_DIAGNOSTIC_PREFIXES` holds the 3
distribution prefixes; `length_policy.py:10-25` holds 4 more soft-length codes.

**Resolution of a contradiction between two subagent reads:** the CARD core's
`has_blocking_guard_failure` (`run_sampled_reddit_generator.py:1688-1707`) *does*
list `uncertainty_frame_unwanted`, `question_mark_unwanted`,
`template_phrase_reused`, `missing_concrete_anchor` etc. as blocking. But the
adapter's `_blocking_guard_check` (`backend.py:1808-1820`) filters exactly those
out before delegating. **Only 4 codes can block anything:** `exact_duplicate`,
`parent_copy`, `placeholder_literal`, `planner_skeleton_residue`. Run data
settles it (attempts=1 for 519/522 despite 233 failing attempts).

## 3.2 Six of twelve metrics never reach the Writer's control loop

`generalized_card/scripts/run_generate.py:488`:

```python
"writer_distribution_controller": {
    "metrics": ["self_bleu_4", "semantic_mean_cosine"],
    "target": "same-size evaluation-excluded real metric template",
    "candidate_policy": "single Writer realization; distribution metrics are diagnostic",
```

`generation_diversity.build_thread_distribution_target` carries only those two.
`polite_rate`, `impolite_rate`, `neutral_rate`, `emotion_entropy`,
`mean_story_probability`, `length_cv` are **dropped at the writer boundary** —
they exist only as a prompt sentence with nothing verifying them. This is the
structural reason they are the failing ones.

## 3.3 No validator ever reads `semantic_move` back

There is **no** check comparing a comment to its own plan. Searched for
`plan_echo|echo_detect|instruction_leak|verbatim_plan|...` — zero hits outside
tests.

Worse, the one function that compares output to plan text **rewards** the echo:
`scripts/sampling_generator/engine/writer_validation.py:247-274`
`has_task_anchor_overlap` includes `task.semantic_move` in its anchor source, so a
verbatim copy trivially clears `hits >= 2` and *passes* the anchor check. It is
reached from `backend.py:2388` inside `_long_helpful_anchor`.

Note: `long_helpful_too_generic` fired **0 times** in v74 and `has_anchor` has
four escape routes, so removing `semantic_move` from that source is a
correctness cleanup with **no measurable effect** — do not oversell it.

The audit's leak detector (`audit.py:24-27` `PROMPT_LEAK_PATTERNS`) matches the
literal field *name* `semantic_move`, not its value, so a thread of pure echoes
scores `evaluable: True, healthy: True`.

## 3.4 The reply planner is blind to its siblings

`prompts.py:336-381` routes to `render_direct_reply_planner_prompt` when
`is_direct_reply_batch` is true. Because batches never mix depths
(`engine/thread_structure.py:87-114`) and ordering is breadth-first, **every
depth ≥ 1 batch takes this path.**

`reply_planning.py:128-308` renders **no** prior-plan ledger, **no** coverage
summary, **no** sibling contract, **no** branch goal, **no** R# reference rows.
Each row sees only its own parent's move-to-exclude.

Verified on seed 2: depth sequence is `0×31, 1×4, 2×2, 3, 4, 5, 6, 7, 8, 9×2` —
depths 3–8 are **single-slot batches**. Tasks 38–45 are nine near-duplicate moves
all saying "the fixed lens made me reach for the camera more often". They could
not see each other.

## 3.5 There is no `semantic_move` de-duplication anywhere

The whole-plan `semantic_collision` check exists but cannot catch this:
- `planning_quality.py:565-577` `plan_similarity` is a Jaccard over **all** of
  `SEMANTIC_FIELDS` including `development_plan` (up to 40 beats × 220 chars), so
  a ~20-token `semantic_move` is ~10% of the token mass. Two plans with
  byte-identical moves score ≈0.05–0.27 against a threshold of 0.72.
- `planning_quality.py:542` → `_dependent_variation` (`:944-957`) **exempts
  parent–child pairs outright** when 3 loosely-defined fields differ.
- `duplicate_claim` (`:534-541`) requires an **exact** `claim_key` match.
- When a collision *is* caught, `backend.py:1531-1557` only **warns** and
  continues.
- `STOPWORDS` (`:101-158`) contains exactly the parser's fallback strings, so two
  plans that both fell back produce empty token sets and can never collide.

## 3.6 Prompt contradictions, measured over 522 rendered v74 prompts

| finding | share |
|---|---|
| competing register/style instructions per prompt | mean **7.8** (min 7, max 8) |
| prompt asks for a first-person story **and** bans a "first-person frame" | **9.8% (51 slots)** |
| bans a first-person frame | 45.4% |
| ...while real Reddit uses one | 51.6% |
| `tone=impolite` over a *helpful* semantic_move | 4.8% |
| instructs a single/uneven **paragraph** | 39.1% |
| bans unlisted products/numbers | **79.7%** |
| slot's own proposition restated in its prompt | mean 1.43×, max 6 |

## 3.7 The concreteness ban — why specs are 0.06/comment vs a real 0.55

Every path forbids naming anything not already visible:
- `prompts.py:1286` (focused, the production path): `- Name a product, model, or
  number only if it is visible above.`
- `prompts.py:1132` (full), `prompts.py:1390` (low-info): equivalents
- `prompts.py:2697-2714` `_story_fact_safety_rule`: forbids "a product,
  specification, price, measurement, date, policy, link, diagnosis, or
  **externally checkable outcome**"
- `prompts.py:1447` system prompt: "Do not invent facts, specifications, numbers,
  products, links, measured outcomes, or policies" + "realize only a
  **qualitative** synthetic context"
- `prompts.py:110-116` the one escape hatch (own equipment) still bans "a
  specification, price, measurement, or test result", caps at **one** model name,
  and is gated to first-person slots
- `prompts.py:1461-1466` `mask_specifics` scrubs `$900 → [amount]`,
  `ISO 1600 → ISO [number]` from the visible context on jittered slots (jitter
  rate 0.32), so those slots have **zero** licensed numbers

**The fix direction (P4): split the ban.** Facts about the *seed product* must
stay grounded; facts about the *speaker's own kit and history* should be free to
invent. That is where real Reddit texture comes from, and it is currently banned
along with everything else.

## 3.8 Story: the allocation is correct, the instruction is not

Correcting an over-claim from one subagent read: `specific_personal_story` is
**44 of 79 story slots (56%)**, not rare. Per-thread story count is scaled from
the matched real thread's own `story_rate`, which is why
`mean_story_probability` passes. **Do not change the allocation.**

The content is the problem. `generation_distribution.py:323-329` asks for "a
setting, an action, a small friction or change, and a local reaction", then
`_story_fact_safety_rule` forbids every category that would make those specific.
Measured across 57 story slots: **5% contain a spec, 2% a time marker, 4% a
place.** "I've done that in a packed room before" is the *compliant* output. A
consequence is banned as an "externally checkable outcome" — and a story without
a consequence is not a story.

## 3.9 There is no persistent speaker identity (deepest structural gap)

- `speaker_role` is a per-slot label from a 10-value enum
- `persona_conditioning = none` (inert; `persona_bridge.py` is dead weight)
- own equipment is a rotating 4-item shortlist keyed by **slot index**, with no
  continuity across slots (`prompts.py:79-116`, `_own_equipment_block`)
- `perspectives` are NOT domain-derived: `domain_profile.py:73` calls
  `universal_viewpoints()`, a fixed P01–P12 lens shared by every domain, enforced
  with `max_perspective_share=0.34`

A real thread is N people with histories; this is N slots with role labels. All
comments read like one author because they **are** one author. This is the
leading hypothesis for `self_bertscore`'s uniform +0.033 offset.

## 3.10 Confirmed bugs

**B1 — `micro_reaction` was silently dropping comments. [FIXED in v75]**
`run_sampled_reddit_generator.py:1997-1999` overwrote any over-length micro slot
with `micro_options[local_task_id % 6]`. One v74 thread had **10** micro slots
against a pool of 6, so tasks 26/32/116 all resolved to `"This"`; the first won
and the other two raised `exact_duplicate` every repair round (repair never
changes `local_task_id`). **Two comments permanently lost**, plus a sibling
`self_bleu` similarity of 0.9999998.

**B2 — beat-budget contradiction (NOT fixed).** `prompts.py:733-738` tells the
planner one beat per 35 words capped at 16; `long_form_planning.py:29-30` demands
`round(words/21) - 1`. For a 300-word slot: prompt says 8–9, validator wants 13.
Following the prompt *guarantees* `long_form_capacity` failure, and the surplus
beats are exactly what dilutes the collision detector (3.5).

**B3 — `allow_first_person_frame` is computed and never used (NOT fixed).**
`run_sampled_reddit_generator.py:1022` derives it from the matched real comment
and stores it. `prompts.py:2726` `_substitution_rule` keys **only** on
`tone_target` and never reads the field. Result: **84 slots (16.1%) are banned
from a first-person frame their real counterpart actually used.**

**B4 — `tone_overlay_slot` / `tone_overlay_instruction` are read in 5 places and
assigned nowhere.** Always `""`. Dead.

**B5 — `_delexicalize_tone_examples` (`backend.py:1086-1106`) matches three
literal strings that no longer exist in `TONE_DEFINITIONS`.** Permanent no-op.

**B6 — `tone_target == "constructive_polite_helpful"`
(`run_sampled_reddit_generator.py:1299-1308`) can never be true;** `TONE_CLASSES`
has four other values. Both disjuncts dead.

**B7 — `allocate_story_and_affect` is a no-op auditor.**
`generation_distribution.py:148-153` hard-wires `story_modes_after` to the same
dict object as `_before`, and `converted`/`demoted` to empty. The whole
`personal_min_share` argument chain exists only to feed it.

**B8 — template overrides are silently swallowed.**
`planner_distribution.apply_slot_distribution_schedule` accepts an `events` list;
`backend.py:503-506` calls it **without** `events`. Every tone/story/affect/opener
override is discarded unlogged. Also `planner_distribution.py:148-150`
unconditionally overwrites the planner's `story_mode` with the default *before*
reading `original`, so a disagreement can never be recorded.

**B9 — dead validations that can never fire:** `invalid_perspective`
(`planning_quality.py:389-397`, because `backend.py:1600-1621` normalizes first);
`branch_route_conflict` (`:398-414`, compares a value against itself after
`backend.py:1718` sets it); `utterance_mode` in the plan dict (never requested by
either schema, always recomputed at `run_sampled_reddit_generator.py:984`).

**B10 — `empty` is not in the core blocking set** (`:1688-1707`), so empty output
does not trigger degraded recovery; it survives only because `writer_quality.py`
short-circuits on falsy text.

**B11 — `perspective_id` repair is structurally impossible but burns budget.**
`backend.py:1738` overwrites the planner's choice unconditionally, and
`_annotate_plan_metadata` re-runs on every repaired plan, so
`perspective_concentration` can never be repaired — yet it is in `repair_issues`
and consumes an LLM call per slot per round.

**B12 — repair feedback references a block the reply prompt lacks.**
`planning_quality.py:266,272` tells the planner to vary its R# usage; reply
batches (where most repairs happen) render no R# rows at all.

**B13 — `--writer-hard-recovery-rounds` default is 2 but every run used 0**, so
`backend.py:1927`'s hard-recovery loop has never executed.

---

# 4. TASK LIST P0–P6

Full detail in `tasks/todo.md`. Only **P2 is done**.

## P0 — turn the validation layer on (highest value, no new concepts)

- [ ] **Add the 6 missing metrics to the Writer's distribution target**
      (`run_generate.py:488`). This is the single most important item: 6 of 12
      metrics currently have no writer-side control loop at all, and they are
      exactly the failing ones. Offline-verifiable (render prompts, no API cost).
- [ ] Raise `--writer-retries` above 0; re-check cost per thread.
- [ ] Move `lexical_overlap_high`, `missing_concrete_anchor`,
      `template_phrase_reused` out of `SINGLE_STAGE_DIAGNOSTIC_PROBLEMS` so they
      can force a retry. **Sequencing constraint:** `missing_concrete_anchor`
      cannot be satisfied while `prompts.py:1286` bans unlisted entities — do P4
      first or it will fight itself.
- [ ] Add `empty` to the core blocking set.
- [x] Fix B1 (`micro_reaction` collision) — done in v75.

## P1 — plan-echo guard (now cheap; echo is at 0.0%)

- [ ] New validator: longest contiguous shared word run vs `task.semantic_move`.
      **Threshold ≥12 is empirically calibrated:** flags 0.0% of v67 comments
      ≥25 words and 34.7% of v74's. Register in `HARD_REALIZATION_PROBLEMS`, keep
      it OUT of `SINGLE_STAGE_DIAGNOSTIC_PROBLEMS`, and add it to
      `REPAIRABLE_WRITER_PROBLEMS` or it will `skip: True` and drop comments.
- [ ] Remove `semantic_move` from `has_task_anchor_overlap`'s source (cleanup,
      no measurable effect — see 3.3).
- [ ] Extend the audit `evaluable`/`healthy` gate to see plan echo.

## P2 — stop the Planner writing the comment  ✅ DONE (v75)

See §5. Echo 25.8% → 0.0%.

- [ ] Remaining sub-item: constrain the planner's *register*.
      `prompts.py:694-695` addresses the planner as the participant ("what
      happened when you personally used X") and `reply_planning.py:71` defines
      `corroborating_datapoint` as "your own concrete experience". These are why
      19.3% of moves opened with "I". Deliberately left separate so it can be
      attributed on its own.

## P3 — reply-planner sibling visibility + move de-duplication

- [ ] Give `render_direct_reply_planner_prompt` a prior-plan ledger / coverage
      summary (see 3.4).
- [ ] Add a real `semantic_move` similarity check (see 3.5).
- [ ] Fix B2, the beat-budget contradiction — it is also what blinds the
      collision detector.

## P4 — persistent speaker identity + split the grounding ban  ← BIGGEST WIN LEFT

- [ ] Give each speaker a persistent kit + history across their slots.
- [ ] **Split the ban:** seed-product facts stay grounded; the speaker's own kit
      and experience become free to invent (see 3.7).
- [ ] Targets `self_bertscore` (never passed, d=0.96), concreteness (0.08 vs
      0.55), novel entities (6.6 vs 47.3), and emotion.

## P5 — polite register  (DE-PRIORITIZED by the user; do not lead with this)

Kept only for completeness. `generation_distribution.py:473-478` records a design
decision that the data refuted: `TONE_DEFINITIONS["polite"]` forbids hedges and
thank-yous on the theory that they'd collapse into `somewhat_polite`; the
measured collapse is into **impolite** (65% of polite slots, 7.4% realized). Also
B3 (`allow_first_person_frame`), B4, B5, B6.

## P5b — story instruction (allocation is correct, do not touch it)

- [ ] Scope the fact ban to the seed product, not the speaker's history.
- [ ] Allow a consequence; a story without one is not a story.

## P6 — surface realism (user rated this lowest priority)

- [ ] Convert `_low_info_writer_prompt` to the focused prompt (106/522 slots,
      mean 15,468 chars, dumps 11 internal labels, produced 9-word outputs).
- [ ] `LENGTH_BUCKET_BOUNDS["very_long"] = (120, 220)`
      (`engine/vocabulary.py:195`). Real threads exceed 220 words in **10/10**
      matched threads (max 845). This caps `length_cv`.
- [ ] Drop the single-paragraph instruction (39.1% of prompts; real
      multi-paragraph rate 32.8% vs generated 3.1%).
- [ ] Straight apostrophes / typography (79.4% curly vs 10.9% real). Verified
      model-emitted, identical pre/post cleanup — a deterministic post-step fixes
      it with no prompt work.
- [ ] Consider deleting the `gpt_cleanup` stage (0.6% of comments, whitespace
      only).

## Rejected: two writers supervising each other

The user asked about this. **Recommendation: do not build an LLM critique loop.**
Cost doubles, and critique-driven revision pushes text toward the balanced,
hedged, "on the other hand" register — which is the exact failure mode. The
stronger argument: the system already has ~20 validators and ignores 18 of them.

The useful form of "supervision" is a **deterministic discriminator** on the
checks above — free, reproducible, aimed at measured gaps. If an LLM ever enters
the loop, the right shape is **re-voicing** a draft as a different speaker with a
different kit, with no critique language in the prompt, measured on its own.

## Another idea worth trying (not yet built)

**De-topicalized real skeleton.** `real_surface_shape` is currently compressed to
a sentence ("long uneven Reddit paragraph, about 21 sentences") and everything
useful is discarded. Instead hand the writer a held-out real comment's *skeleton*
with topic content stripped: punctuation rhythm, paragraph breaks, sentence-length
sequence, opener form. Targets `self_bleu`, `length_cv`, plausibility; leaks no
content; uses the existing `real_sample_id` / `real_word_count` fields. This also
matches the user's own request for "真实评论修改之后的 example".

---

# 5. CHANGE HISTORY — WHAT WORKED AND WHAT DID NOT

## Worked

| change | evidence |
|---|---|
| **Refactor 9,290-line generator → 2,059 facade + 12 engine modules** | 165/166 surviving definitions byte-identical; 218 tests pass; all 13 files pinned |
| **`--domain-claim` ablation flag** | isolated the v71 regression (8/12 → 4/12) to `domain_claim`; v72 confirmed |
| **Focused writer prompt (v74)** | prompt 22,249 → 8,139 chars, cost $3.34 → $2.24, diversity held, 5 metrics matched |
| **Route lock `own_words` (v75)** | plan echo 25.8% → 0.0%; reviewer-voice frame 7.23% → 2.29%; word-count median 33.3 → 36.7 vs real 34.6 |
| **`micro_reaction` pool fix (v75)** | verified on the real collision: tasks 26/32/116 now give `Nah`/`Solid`/`This` |
| **Fix E (length scaling)** | `length_cv` PARTIAL d=-0.54 → PASS d=-0.40 |
| **Fix B (hard_disagree)** | reached d=0.17, best of any version |

## Did NOT work

| change | result |
|---|---|
| **Affect-instruction rewrite** (prohibition-led → affirmative, 28 entries) | never cleanly attributed; bundled into v73 |
| **Route-lock rewording in v73** ("say what the turn is about in ordinary words") | frame stayed at ~7%; and it *started* the echo regression (2% → 8%) |
| **Prompt slimming as a register fix (v74)** | prompt −57% but warm markers 6.3% → 4.8%, frame 19.6% → 19.8%. **The prompt-length hypothesis is refuted at this magnitude.** |
| **Beat budget 1/80 → 1/35 words** | long comments stayed at ~0.72× matched length; wrong cause (see `lessons.md`) |
| **`polite` defined as un-hedged warmth** | predicted collapse into `somewhat_polite`; actual collapse into **impolite**, 7.4% realized |

## The single most instructive experiment

A 30-slot A/B (same model, same slot): shipped **21,920-char** prompt vs a
**532-char** plain prompt → warm 20.0% → 26.7%, frame 26.7% → 6.7%. **The model
can produce the target register.** But the 8,139-char focused prompt scored the
same as the 22,249-char one, so the difference is **which controls are present,
not character count**. The controls present in focused but absent from the 532-char
experiment are the untested suspects: `opener_rule`, `surface_skeleton`, the tone
contrast block, `substitution_rule`, `story_rule`, branch exclusion, the two
ledgers.

---

# 6. MY ERRORS — READ THIS BEFORE REPEATING THEM

## Analysis errors

1. **Bundling changes prevents attribution.** v71 bundled 6 fixes with
   `domain_claim` → 8/12 → 4/12, cause unknown until a dedicated ablation run. I
   then **repeated it** in v73, and again in v75 (`plan-quality-repairs`).
   **One mechanism per run.**
2. **Wrong metric measured.** Declared "Fix E failed" from pooled comment-level CV
   instead of the per-thread `length_cv` metric. It had actually passed.
3. **Declared "Fix B failed"** when `hard_disagree_rate` had reached its best-ever
   d=0.17.
4. **Quoted the wrong baseline** — used the 763-thread global real average in
   prose while evaluation used the 10 matched threads. Corrected matched-real
   values: `emotion_entropy` 1.636, `length_cv` 0.948, `self_bertscore` 0.496.
5. **Over-concluded from n=10** — judged a frame "not an echo" from 2/10 first
   comments (CI 3%–56%); redid at n=40 (27.5%).
6. **Prompt-composition error** — claimed "Already used openings" was 44.4% of the
   prompt; block-boundary method absorbed adjacent text. Correct value 9%.
7. **Case-sensitive regex false alarm** — flagged A7/A9 as invented; they were in
   the seed post in lowercase.
8. **Compared cleaned/ against generated/** without noticing the different
   pipeline stage. (Re-measured; they happened to be identical, but the method was
   wrong.)
9. **A "systematic" default-diff script with a silent hole** — it matched argparse
   dest names (`plan_quality_repairs`) against config leaf keys (`repair_rounds`).
   Different names → silently skipped → the v75 confound. **When writing a
   completeness check, verify the check itself is complete.**
10. **Reported a regression that did not exist** — `self_bleu_4` PARTIAL→FAIL in
    v75 is distributional noise; the mean excess is +0.0088 vs +0.0089.

## Tooling errors

11. **Lost 38 paid completions** in `slim_prompt_thread_ab.py` by scoring before
    persisting; also used the wrong scorer field name. Now caches.
12. **Reachability analyzer too loose**, then **too narrow** — it missed
    `prompts.py` reading generator attributes via the `backend` parameter and
    wrongly deleted live functions. **The test suite caught it.**
13. **Gave the user three broken commands in a row** (`--seed-pool` doesn't exist;
    missed `--writer-hard-recovery-rounds 0`; a tag that couldn't `--resume`
    because I'd re-pinned hashes after its preflight). The user was rightly
    annoyed. **Now: always dry-run a command on a throwaway tag with
    `--prepare-only` before handing it over, and separately verify whatever
    `--prepare-only` does not cover.**

## Design errors

14. **Dropped a hard-failure rule while "simplifying"** — the first
    `_focused_writer_prompt` dropped `_story_fact_safety_rule` (factual grounding)
    and the polite register's first-person frame. **The test suite caught both.**
15. **"No metric depends on it" is not a reason to delete a control.** v74 dropped
    the semantic-difference contract on that reasoning; no metric measured plan
    echo, and echo went 10.2% → 25.8%. **Absence of a metric is not absence of a
    function.**
16. **Applied a fix to only 80% of slots.** v74's focused prompt never converted
    `_low_info_writer_prompt` (106/522 slots at ~15,468 chars), which confounded
    the whole result. In v75 the realization rule was deliberately rendered on
    **both** paths.
17. **Ignored the user's stated priorities** — kept analyzing politeness after
    being told repeatedly it was de-prioritized.

---

# 7. OPERATING RULES

## Reading

- **Read every relevant file end to end before diagnosing.** Not grep hits, not
  line ranges. "你决定要读哪个 code file，就要读它全量的 code files."
  In this codebase that means: the CLI, the backend adapter, **every** prompt
  builder (root planner / direct-reply planner / focused writer / low-info
  writer — there is more than one per role), the shared generator facade, the
  engine modules, and the policy modules.
- **When several builders exist for one role, diff their schemas.** A field
  present in one and absent in another is a likely defect. That is how the
  reply-planner `development_plan` omission and the root-vs-reply `semantic_move`
  schema contradiction were found.
- **Verify an analysis script against a ground-truth count before trusting it.**
  `generation_records.json` has one record per comment *and* nests replies —
  recursing into `replies` double-counts.
- **Trace the full consequence path before adding a check.** Adding a plan-echo
  validator before removing the cause would have dropped up to 130 slots, because
  a code outside `REPAIRABLE_WRITER_PROBLEMS` → `skip: True` at `backend.py:2022`,
  and repair exhaustion → `skip: True` at `backend.py:2205`.
- Subagent reports can be wrong or overstated. **Two of three full-file reads
  contained a materially wrong claim** (the "blocking guards" contradiction and
  "`specific_personal_story` is near-unreachable"). Re-verify load-bearing claims
  against run artifacts before acting.

## Changing

- **One mechanism per API run.** Predict the expected magnitude first so a null
  result is interpretable.
- **Every behavior change gets an ablation flag** and is recorded in
  `run_config`. Follow the existing pattern: `--domain-claim`, `--writer-prompt`,
  `--writer-route-lock` (env var + `backend.py` module attr + CLI flag + config
  record).
- **Re-pin `core_contract.py` after editing any pinned file.** Compute the hash,
  verify the drift list contains *exactly* the files you edited. Bump
  `GENERALIZED_V2_GENERATION_POLICY_VERSION` and **move the old version string
  into `HISTORICAL_GENERATION_POLICY_VERSIONS`**, or previous runs become
  un-evaluatable.
- **Run the tests.** `cd generalized_card && python3 -m pytest tests/ -q`
  (currently **224 pass**). The 3 failures in the repo-root `tests/` are
  pre-existing calibration-subsystem failures, unrelated.
- **Write a test for anything that regressed silently.** The route lock had
  **zero** coverage, which is why it regressed twice unnoticed.
- Also run the backend self-test:
  `GENERALIZED_CARD_DOMAIN=camera python3 generalized_card/scripts/run_generator_backend.py --self-test`

## Handing the user a command

- **Dry-run it first** on a throwaway tag with `--prepare-only`, then delete the
  throwaway dir.
- **Separately verify what `--prepare-only` skips** (it returns at
  `run_generate.py:715`, before the API-key check at :719).
- **Diff the full config against the previous run**, not against argparse
  defaults by name. Read `run_config.json` from both runs and compare the
  flattened trees.

---

# 8. KEY FACTS AND COMMANDS

## Environment

- API keys live in `third_party/MiroFish/.env` as **`LLM_API_KEY`** (not
  `OPENAI_API_KEY`). `LLM_BASE_URL=https://api.openai.com/v1`.
- `run_generate.py` now loads that file itself (added in v75), but still pass
  `--api-key-env LLM_API_KEY`.

## Current run command (v75 shape — remember to add `--plan-quality-repairs 0`)

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag <NEW_TAG> \
  --domain camera --model gpt-5.4-mini \
  --base-url https://api.openai.com/v1 \
  --api-key-env LLM_API_KEY \
  --pool-size 150 --max-posts 10 --posts-per-run 1 \
  --start-seed-index 0 --sampling-seed 42 \
  --context-dropout-rate 0.42 --context-jitter-rate 0.32 \
  --plan-quality-repairs 0 \
  --writer-hard-recovery-rounds 0 \
  --writer-local-repair-rounds 1 --writer-slot-retry-limit 0 \
  --domain-claim off --writer-prompt focused \
  --writer-route-lock own_words 2>&1 | tee /tmp/<TAG>_gen.log
```

Non-default flags in v74 were exactly: `domain-claim off`, `posts-per-run 1`,
`writer-hard-recovery-rounds 0`, `writer-local-repair-rounds 1`,
`plan-quality-repairs 0`. Everything else is a default or derived.

The seed pool is **not** a flag; it is derived as
`artifacts/generalized_card/seed_pools/{domain_id}_{pool_size}_seed{sampling_seed}.json`
= `camera_product_150_seed42.json`.

## Evaluation command

```bash
python3 -u generalized_card/scripts/run_evaluate.py \
  --tag <TAG> --device cpu 2>&1 | tee /tmp/<TAG>_eval.log
```

## Useful artifact paths (per run)

```
artifacts/generalized_card/runs/<TAG>/
  run_config.json                      # full config incl. ablation flags
  generated/run_NN_sampled_reddit/
    generation_records.json            # task + prompt + raw + comment per slot
    discussion.json
  cleaned/run_NN_run_NN_sampled_reddit/
    discussion.json[.pre_gpt_cleanup]  # diff these to see cleanup's (non-)effect
    politeness_results.json            # threads[].comments[].pred_label
    go_emotions_results.json, storyseeker_results.json, ...
  logs/
    writer_distribution_control.jsonl  # 522 rows: attempts, final_status, self_bleu
    planning_quality.jsonl             # 145 rows: repair_attempts, collisions, issues
    story_affect_distribution.jsonl    # 10 rows: tone/affect/story targets vs actual
    token_usage_summary.json
  matched_evaluation/
    matched_generated_thread_scores.csv
    matched_real_thread_scores.csv
    matched_seed_group_eval.json
```

## Real-comment ground truth

```
data/raw/discussions/camera_product/<product>/<product>.comments.jsonl
```
Match by `link_id` endswith `source_raw_post_id` (from the seed pool's
`seed_posts[].source_file` / `.source_raw_post_id`).

## Scratchpad analysis tools (rewrite if the session changed)

`sbs.py` (side-by-side real vs generated), `content_diag.py` (the tell table),
`copy_diag.py` / `echo_threshold.py` (plan-echo measurement). These were in the
session scratchpad and may not persist.

## Repo state

- Branch `main`. HEAD = `15a92a2` "checkpoint: generalized-v2 engine at policy
  v73, before the writer-prompt rebuild".
- **v74 and v75 changes are uncommitted working-tree modifications** in:
  `generalized_card/generalized_card/{backend,core_contract,prompts,reply_planning}.py`,
  `generalized_card/scripts/run_generate.py`,
  `generalized_card/tests/test_generalized_card.py`,
  `scripts/sampling_generator/run_sampled_reddit_generator.py`,
  `tasks/{generator_audit,todo}.md`.
- Current policy version:
  `generalized-card-v2-writer-realizes-planner-move-v75-20260814`.
- Consider committing before the next change so v75 is recoverable.

---

# 9. RECOMMENDED NEXT STEP

1. **Rerun v75's code with `--plan-quality-repairs 0`** to remove the confound and
   get a clean read on the route lock alone (~$2.2). Optional but it makes every
   later comparison honest.
2. **P0's first item: add the 6 missing metrics to the Writer's distribution
   target.** Biggest structural fix, offline-verifiable, orthogonal to P2.
3. **P4: persistent speaker identity + split the grounding ban.** The only lever
   aimed at `self_bertscore`, which has never passed in any version.

Do **not** start with politeness. Do **not** add the echo validator before
confirming `REPAIRABLE_WRITER_PROBLEMS` membership.
