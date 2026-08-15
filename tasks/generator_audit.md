# Full-read audit of the active generation path

Every entry is verified against the 522 prompts and tasks of
`generalized_card_camera_gpt54_v67_smoke10_20260813_v1` (coverage 1.00), not from
reading alone. Reading a definition is not evidence that it executes: 44 generator
functions are replaced at runtime by the domain adapter, and one replacement
(`rebalance_card_surfaces`) discards its argument entirely.

## Reading status

Complete. Every line of every file below has been read.

| file | lines | status |
|---|---:|---|
| `scripts/sampling_generator/run_sampled_reddit_generator.py` | 9291 | all read: 1-2600, 2600-3540, 3540-5112, 5112-6070, 6070-7079, 7079-7559, 7559-8028, 8028-9291 |
| `generalized_card/generalized_card/backend.py` | 2289 | all read |
| `generalized_card/generalized_card/prompts.py` | 2786 | all read |
| `planning_quality.py` | 1009 | all read |
| `persona_bridge.py` | 537 | all read — inert, `persona_conditioning.mode=none` in v70 |
| `writer_quality.py` | 513 | all read |
| `audit.py` | 484 | all read |
| `reply_planning.py`, `planner_distribution.py`, `opener_profile.py`, `entity_inventory.py`, `domain_claim.py`, `domain_profile.py`, `semantic_realization.py`, `task_distribution.py`, `generation_distribution.py`, `length_policy.py`, `long_form_planning.py`, `surface_contract.py`, `first_pass_policy.py`, `generation_diversity.py`, `lexical_quality.py`, `branch_routing.py`, `viewpoint_bank.py`, `reference_metric_calibration.py`, `actor_conditioning.py` | — | all read |

Generator line accounting: 1-3540 sampling/planning/task expansion (active), 3540-5112 CARD rebalance
(dead except three functions, see below), 5112-6070 post generation and writer validation (active),
6070-6660 signatures and guards (mixed), 6660-7108 CARD writer prompts (dead, replaced),
7108-7560 context rendering and controls (mixed), 7560-8028 persistence (active),
8028-9291 `run_self_test` (test-only, replaced by `_run_generalized_self_test`).

## Confirmed dead at runtime

- **CARD rebalance chain, ~1570 lines (3540-5112).** `rebalance_card_surfaces`
  begins `del core_rebalance, kwargs` and returns its input unchanged, so
  `rebalance_tone_shapes`, `assign_thread_tone_targets`, `make_reddit_polite_task`,
  `cap_question_like_tasks`, `rebalance_micro_short_tasks` and the rest never run.
- **`LENGTH_BUCKET_BOUNDS` 220-word ceiling.** Reached only through
  `writer_length_rule`, which appears in 0 of 522 prompts; the generalized
  `soft_length_guidance` replaces it.
- **`SYSTEM_PROMPTS["gpt54_reddit_writer"]` r/CreditCards wording.** Rewritten at
  configure time to the domain's community context; the run-time snapshot shows
  "Reddit camera and photography communities". The string survives in the
  reproducibility snapshot only because that file records all seven profiles.
- **`writer_token_cap` 260-token ceiling.** `writer_provider_token_budget`
  (backend:1867) expands it to `real_words * 1.7 + 64`, capped at 1600, so a long
  slot is not truncated by the provider budget.

## Confirmed active defects

### 1. A reply's planned `semantic_move` is overwritten and never restored
`run_sampled_reddit_generator.py:2150-2156` sets, for every reply carrying a
delta, `realized_semantic_move = reply_novelty_anchor or reply_delta`. The
generalized `restore_planner_task_contract` then restores planner-owned fields,
but `semantic_move` is absent from `PLANNER_OWNED_TASK_FIELDS`
(`task_distribution.py:55`), while `decision_boundary` is present.

Measured: **347 of 347 replies (100%)** have `semantic_move == reply_novelty_anchor`;
`semantic_move == decision_boundary` in 0%, confirming the boundary is restored and
the move is not.

Consequence: replies are 67% of comments, and each one's entire semantic content
is a short abstract anchor. This is the mechanism behind the measured
`semantic_move` vocabulary breadth of 2.69 against 7.49 in the real thread, the
four recurring abstract propositions across 197 comments, and a large part of the
self-BLEU gap.

### 2. Two hard parent-exclusion rules live in the generator, not the prompt layer
`2114-2117` writes into `must_not_do`: "Do not re-answer the OP. Do not summarize,
quote, or restate the parent." `2162-2167` appends to `avoid_repeating`: "Do not
restate the parent contribution: {parent_move}."

Measured: each appears in **349 of 522** prompts. A relaxation added to
`prompts.py` appears in 411 prompts and does not remove either, so the two
conflict inside the same prompt and the generator's harder, more specific wording
is what the reply follows. Real reply-to-parent content overlap is 0.197 and
generated is 0.129.

### 3. The opener ledger is truncated upstream of the prompt
`5289`: `recent_openings = recent_openings[-18:]`. A change in `prompts.py` to
render every prior opener cannot exceed what the caller passes.

Measured: **10 openers per prompt** in v67, against a thread of up to 197.

### 4. The reply planner prompt omits the `claim_family` whitelist
First diagnosis was wrong and is recorded here as the correction it needed. The
credit-card whitelist is **not** the cause: `backend.py:361` sets
`module.CLAIM_FAMILIES = prompts.GENERIC_CLAIM_FAMILIES`, and a module-global
lookup inside the pinned parser resolves to the replacement, so
`normalize_vocab_value` at `2007` is already checking against the 16 generic
families. Reading the whitelist definition alone would have shipped a fix for a
defect that does not exist.

The actual cause is a difference between the two planner prompts. The root prompt
interpolates the full list: `prompts.py:578` renders
`"claim_family": "{families}"` from `GENERIC_CLAIM_FAMILIES`. The reply prompt
asks for free text: `reply_planning.py:253` renders
`"claim_family": "one generic claim family"`, with no enumeration anywhere in the
request. Every unlisted string normalizes to `miscellaneous`.

Measured in v67: `miscellaneous` is **349 of 349 reply slots (100%)** and 38 of
173 root slots (22%). `generate_post_from_tasks:5160` skips `miscellaneous` when
enforcing `claim_family_max_share`, so the mechanism that stops one claim family
from dominating a thread governs 27% of the thread and none of the replies, which
are 67% of comments and the half with the worse diversity.

Fix is one interpolation in the reply prompt; no core-contract change.

### 5. The only length floor is very weak
`real_slot_min_words` (5991) demands 70 generated words for a real slot of 140 or
more. For the 413-word comment in the matched real thread that is 17% of target.
`real_slot_too_short` is blocking, so this floor is enforced, but nothing pushes a
long slot beyond 70 words. Generated `word count CV` is 0.899 against a real 1.344.

### 6. `domain_claim` had no enrich step
The shared parser `normalize_comment_move_plans` (1962-2018) keeps only declared
fields. `tone_class`, `affect_role`, and `development_plan` each have an
`enrich_*_plan_fields` step; `domain_claim` shipped without one and reached
**0 of 520** slots. Fixed by `enrich_domain_claim_fields`, guarded now by
`tests/test_planner_field_survival.py`.

### 7. The opener contract is assigned correctly and realized 43.8% of the time
Assignment is not the problem. `build_slot_distribution_schedule` writes
`opener_type` into every plan (`planner_distribution.py:105`),
`apply_slot_distribution_schedule` carries it through (`:139`), and
`prompts._opener_rule` renders it in **522 of 522** v69 prompts. The assigned mix
matches the domain profile almost exactly: content_phrase 0.423 assigned against
0.419 measured in real threads, first_person 0.188/0.203, noun_phrase
0.111/0.103, discourse_marker 0.073/0.068, polarity_token 0.054/0.054.

Realization is the problem. Classifying each generated comment's own opener
against its assignment (v69, n=520):

| assigned | realized | rate |
|---|---:|---:|
| conditional | 13/14 | 92.9% |
| quote | 14/17 | 82.4% |
| polarity_token | 23/28 | 82.1% |
| first_person | 77/98 | 78.6% |
| content_phrase | 76/221 | 34.4% |
| discourse_marker | 13/38 | 34.2% |
| address | 3/12 | 25.0% |
| noun_phrase | 9/58 | 15.5% |
| question | 0/23 | 0.0% |
| imperative | 0/10 | 0.0% |
| link | 0/1 | 0.0% |
| **all** | **228/520** | **43.8%** |

Two separate failures. `question` and `imperative` are never realized because the
assignment contradicts another assigned control on the same slot: a slot whose
payload is not a question cannot open with one, and the cooperative-tone rules
forbid the second-person instruction that an imperative opener requires.
`content_phrase` and `noun_phrase`, the two largest real categories, lose their
mass to `first_person` and `polarity_token`, which the Writer reaches for by
default; realized polarity_token is 0.123 against 0.054 in real threads.

This refines the control-type hierarchy. An assigned categorical field is not
uniformly 86% effective: `tone_target` reaches 86% because a classifier judges the
whole comment, while an opener is a hard constraint on the first token that
competes with every other instruction. The fix is to stop assigning the
incompatible types and gate the rest by payload, not to add prompt text.

### 8. Latent, not firing: the hardcoded micro-reaction substitution
`shape_writer_text_for_task:7484` replaces any output over 6 words on a
`micro_reaction` slot with a fixed six-string list, and forces a literal
`[deleted]`/`[removed]` on `deleted_removed`. The adapter blanks
`surface_texture` (`backend.py:571`), which disables the "Thanks, ", " lol",
" /s" and "..." injections, but **not** these two branches.

Measured: 20 micro_reaction slots in v67, **0 substitutions fired**, and no
`[deleted]`/`[removed]` output. It is a latent hazard, not a current cause. I am
recording it as latent rather than active because the definition alone would have
read like a live defect.

### 9. `domain_claim` is confirmed fixed end to end
v70 carries the claim rule in **127 of 137** prompts against **0 of 197** in v69.

### 10. `_generalize_instruction_text` is a 40-entry hand-written substitution table
`backend.py:1096-1126` rewrites credit-card wording into generic wording with
literal pairs such as `("issuer/card", "product/service")` and
`("hard pull vs soft pull, credit-line transfer, USBAR limit, ...", "setup
difference, compatibility constraint, ...")`. Every instruction string the Writer
and Planner see passes through it. It is the concrete thing the LLM domain adapter
has to replace: it cannot generalize to news, sports, or reddit, and a phrase it
misses silently reaches the Writer as credit-card language.

### 11. Every reply prompt carries three to four parent prohibitions against one permission
Four separate places write a parent-exclusion rule into the same prompt:
`generator:2114` into `must_not_do`, `generator:2162` into `avoid_repeating`,
`prompts.py:2147` in the route lock, and `prompts.py:1086` in the hard rules. A
fifth path, `writer_quality.writer_local_repair_task:355`, appends yet another to
`must_not_do` on every retry.

Measured in v70 over 61 reply prompts: prohibitions 1, 2 and 3 appear in
**61 of 61**; prohibition 4 and the single permission clause each appear in 47 of
61. **47 reply prompts carry four simultaneous prohibitions and 14 carry three**,
against at most one clause saying the interpersonal move is still allowed. Real
reply-to-parent content overlap is 0.197 and generated is 0.129.

### 12. Two length floors, both far below the slot's scale
`generator:5991` (`real_slot_min_words`) asks 70 words for a 140+ word real slot.
`writer_quality.substantive_length_floor_problem:219` asks
`max(8, min(32, real_words*0.5))` — hard-capped at **32 words** regardless of slot
size. For the 413-word comment in the matched real thread the two floors are 70
and 32. Nothing pushes a long slot past 70. Generated word-count CV is 0.899
against a real 1.344.

### 13. Removable, but pinned by the core contract
- **CARD rebalance block, `generator:3606-5102`, about 1,450 lines.** Of the 73
  functions defined in 3540-5112, exactly three have live callers outside the
  block: `rebalance_tasks_for_diversity` (replaced, original discarded),
  `finalize_rebalanced_task` (replaced and live), `first_choice` and
  `claim_family_budget_for_total` (live utilities). The other 69 are unreachable.
- **CARD writer prompts, `generator:6660-7108`,** plus
  `gpt_metric_guidance_block`, `gpt_tone_discourse_guidance_block`,
  `gpt_placeholder_guidance_block`, `gpt_payload_specific_guidance_block`,
  `render_gpt_thread_memory`, `render_gpt_distribution_pressure`: all reachable
  only from `build_writer_prompt`, which is replaced, and the generalized layer
  calls none of them. The most metric-explicit instruction block in the whole
  generator never reaches a Writer.
- **Reviser subsystem.** `prompts.py:1459-1934` (five reviser prompt builders plus
  the static replacement table) is reachable only from `reviser_backend.py` and
  the `run_*_reviser_backend.py` scripts, never from the generation path. The
  self-loop reviser is abandoned, so this is roughly 590 lines of `prompts.py`
  plus two modules that no current run touches.
- **`persona_bridge.py`, 537 lines,** and the actor-conditioning branches: v70
  records `persona_conditioning.mode=none` and `actor_conditioning.mode=none`.
  Note the consequence: because actor mode is off, the early return in
  `_evaluator_aligned_lexical_overlap_check` does **not** fire, so the
  lexical/semantic distribution guards are live.

All of these are hash-pinned in `core_contract.py`, so removal means a contract
bump, not a plain delete.

### 14. Hypotheses I checked and had to discard
Recording these because each looked like a defect from reading the definition and
was refuted by measuring the run. This is the reason the read was worth doing at
this length.

- **"`claim_family` is normalized against the credit-card whitelist."** False;
  `backend.py:361` replaces it. The real cause is the reply prompt (finding 4).
- **"`render_sampled_plan_block`'s reduced branches drop `semantic_move` and
  `development_plan` for 60% of slots."** False. The label
  `- semantic_move:` appears in only 155 of 522 prompts, but the generalized
  layer renders the same content elsewhere: the actual `semantic_move` **text**
  appears in **522 of 522** prompts and `development_plan` text in **272 of 272**
  slots that have one. Harmless.
- **"The r/CreditCards writer prompt is reaching the model."** False. 0 of 522
  prompts contain `r/CreditCards`, `credit-card discussion`, `credit card`, or
  `Subreddit context`.
- **"`shape_writer_text_for_task` is injecting fixed micro strings."** Not firing:
  20 micro_reaction slots, 0 substitutions (finding 8).
- **"The `_dependent_variation` collision exemption requires all three of stance,
  evidence_mode and detail_focus to differ, so it suppresses the supportive reply
  types added in v65."** False. Supportive types are 39.8% of replies in v67,
  40.1% in v69, 42.6% in v70, and stance=agree is the majority (170 of 349 in
  v69). The exemption only widens what is allowed; failing it still requires
  crossing a similarity threshold to be flagged.
- Note what this last one implies: the supportive reply types **were** present at
  39.8% in v67, the run that was worse than v64 on 8 of 12 metrics. So v65 is not
  the win I assumed, and the v64→v67 regression is still unattributed. It needs a
  controlled A/B, not another reading pass.

## Cross-cutting lesson

Defects 1, 2, 3 and 5 all live in the shared generator while the previous six
versions of changes were made in the generalized layer's `prompts.py`. Upstream
field overwrites and upstream hard rules were never removed, which is why the
metric aggregate degraded from v64 (6/12) to v67 (5/12) while individual component
measurements looked like wins.

---

# Stage 4 re-audit (after the refactor)

Every finding above was re-located in the reorganized engine and re-measured
against the **v70** run, not carried forward on trust. The filter is the one that
matters for this work: does it plausibly move how comments *sound* —
`self_bleu_4`, `self_bertscore`, `semantic_cosine`, `hard_disagree_rate`,
`emotion_entropy`, `mean_story_probability`, `length_cv`? Content fidelity here
is the way real Redditors talk, not factual accuracy.

v70 baseline: 137 comments, 76 roots and 61 replies.

## Confirmed, and worth fixing

| # | Finding | v70 measurement | New location | Metric it should move |
|---|---|---|---|---|
| 1 | reply `semantic_move` overwritten by the novelty anchor | **61/61** replies; `== decision_boundary` 0/61 | facade `:924-930`; fix in `task_distribution.py:55` | `self_bleu_4`, `self_bertscore`, `semantic_cosine` |
| 4 | reply planner prompt omits the `claim_family` enumeration | **61/61** replies are `miscellaneous`, 1 distinct family, against 14 across roots | `reply_planning.py:253` | `self_bleu_4`, `semantic_cosine` |
| 3 | openings ledger truncated upstream | ledger caps at **18** lines while **44** prior comments exist in the thread | facade, `generate_post_from_tasks` | `self_bleu_4` |
| 2/11 | parent prohibitions stack on every reply | **61/61** reply prompts carry exactly 3 | facade `:889` and `:938` | `hard_disagree_rate`, `semantic_cosine` |
| 5/12 | two weak length floors | generated CV **0.874** / max **153** against matched real CV **1.038** / max **337** | `engine/writer_validation.py:45`, `writer_quality.py:219` | `length_cv`, `mean_story_probability` |
| 7 | opener types assigned perfectly, realized 43.8% | `question` 0/23, `imperative` 0/10 | `planner_distribution.py`, `prompts._opener_rule` | `self_bleu_4` |

### What the re-measurement changed

- **Finding 1 has a clean fix that loses nothing.** I checked whether the
  novelty anchor still reaches the Writer if `semantic_move` stops being
  overwritten: `reply_delta` and `reply_novelty_anchor` are already rendered as
  **separate** prompt fields in 61 of 61 reply prompts. The overwrite is pure
  loss. It also explains the shape of the collapse — one sampled anchor was
  `"telephoto reach on a compact camera"`, a noun phrase, and that noun phrase
  became the reply's entire semantic content.
- **Finding 2/11 is now uniform**, not the mixed 3-or-4 of v67/v69: every reply
  prompt carries three prohibitions. Worth stating why this is a metric problem
  and not just clutter: disagreement in real threads restates the thing it
  disagrees with. Measured parent overlap is 0.129 generated against 0.197 real,
  and `hard_disagree_rate` is the metric whose Cliff's delta got worse
  (0.38 → 0.61). These are the same phenomenon.
- **Finding 12 understates the gap.** The generated maximum is 153 words while
  the matched real maximum is 337. No slot in the run reached the real long tail,
  which is where stories live.

## Confirmed, but not worth fixing now

- **6, 9 — `domain_claim`**: already fixed and guarded by
  `tests/test_planner_field_survival.py`. 127/137 in v70 against 0/197 in v69.
- **8 — the micro-reaction substitution**: still latent. 0 firings. It is a trap
  for a future run, not a current cause. Leave it, note it.
- **13 — removable code**: done. See `tasks/refactor_map.md`.
- **10 — `_generalize_instruction_text`'s 40-entry substitution table**: real,
  but it is a *domain-transfer* problem, not a metric one. It blocks moving to
  `cell_phone` / `headphone` / `news` cleanly. It belongs with the LLM domain
  adapter work, not with this metric pass.

## Still unattributed, and reading will not settle it

The v64 (6/12) → v67 (5/12) regression. Six of the findings above were present
in both runs, so none of them explains a change *between* them. This needs a
controlled A/B, one variable at a time, and it should be run before any further
tuning — otherwise the next improvement is measured against a baseline nobody
understands.

---

# v71 / v72 results

## v71 — the six fixes shipped together, and the release lost ground

Matched-seed evaluation against the same 10 real threads. v69 and v71 run
configs are field-for-field identical apart from `writer_local_repair_rounds`
(0 -> 1), the same seed pool, the same domain-profile sha256.

| metric | v69 | v71 |
|---|---|---|
| semantic_mean_cosine | **PASS** p=0.99 d=0.00 | **FAIL** p=0.012 d=0.58 |
| emotion_entropy | **PASS** p=0.089 d=-0.46 | **FAIL** p=0.012 d=-0.64 |
| self_bleu_4 | **PASS** p=0.089 d=0.46 | PARTIAL p=0.021 d=0.62 |
| hard_disagree_rate | **PASS** p=0.29 d=0.29 | PARTIAL p=0.026 d=0.60 |
| impolite_rate | FAIL d=0.88 | **PARTIAL** d=0.61 |
| neutral_rate | PASS d=-0.21 | PASS d=-0.11 |
| mean_story_probability | PASS d=0.12 | PASS d=-0.02 |
| length_cv | PARTIAL d=-0.54 | PARTIAL d=-0.62 |
| self_bertscore_mean_f1 | FAIL d=0.90 | FAIL d=0.94 |
| polite_rate / avg_depth / structural_virality | unchanged | unchanged |
| **total** | **8/12 PASS** | **4/12 PASS** |

Every mechanism the fixes targeted did change as predicted -- reply
`semantic_move` 7.7 -> 23.0 words and `== novelty_anchor` 61/61 -> 0/349,
`claim_family` `miscellaneous` 100% -> 22% over 14 families, the openings ledger
18 -> 183 lines, opener realization 42.6% -> 49.4% with `question` 0% -> 27% and
`imperative` 0% -> 30%. The mechanisms were fixed and the metrics still got
worse, which is the whole lesson: a mechanism defect is not the same thing as a
metric cause, and I reported the mechanism results before the metrics existed.

## v72 — one variable removed

v71 bundled a change that was not part of the six: `domain_claim` finally
worked. It reached **0 of 522 comments in v69 and 508 of 522 in v71**, because
v70 fixed its enrich step and v70 was never evaluated (it completed 6 posts).
So the release under test contained a 0 -> 97% intervention that had never been
measured.

`--domain-claim off` was added as an ablation control (recorded in run_config,
covered by a test). v72 is v71 with only that flag changed.

| | domain claims | `semantic_overlap_high` flags | parent-child Jaccard | generated length CV |
|---|---|---|---|---|
| v69 | 0/522 | 37 | 0.066 | 0.899 |
| v71 | 508/522 | 208 | 0.077 | 0.893 |
| v72 | 0/522 | **51** | 0.066 | 0.908 |

157 of the 171 extra semantic-overlap flags follow the domain claim, not the six
fixes. Injecting one concrete domain fact into 97% of comments pulls the whole
thread into a single factual space.

## v72 result: 7/12 PASS, 2 PARTIAL

`--domain-claim off` recovers almost everything v71 lost. Cliff's delta, matched
seeds throughout:

| metric | v69 | v71 | v72 |
|---|---|---|---|
| self_bleu_4 | PASS 0.46 | PART 0.62 | PART 0.64 |
| self_bertscore_mean_f1 | FAIL 0.90 | FAIL 0.94 | FAIL **0.82** |
| semantic_mean_cosine | PASS 0.00 | FAIL 0.58 | PASS 0.20 |
| hard_disagree_rate | PASS 0.29 | PART 0.60 | PASS **0.17** |
| polite_rate | FAIL -0.63 | FAIL -0.69 | PART -0.64 |
| impolite_rate | FAIL 0.88 | PART 0.61 | FAIL **0.74** |
| neutral_rate | PASS -0.21 | PASS -0.11 | PASS -0.30 |
| length_cv | PART -0.54 | PART -0.62 | **PASS -0.40** |
| avg_depth | PASS -0.07 | PASS -0.03 | PASS -0.04 |
| structural_virality | PASS -0.03 | PASS -0.02 | PASS -0.02 |
| mean_story_probability | PASS 0.12 | PASS -0.02 | PASS **0.00** |
| emotion_entropy | PASS -0.46 | FAIL -0.64 | FAIL -0.65 |
| **PASS total** | **8** | **4** | **7** |

Effect size improved on 7 metrics and worsened on 4. The PASS count is 7 rather
than 8 only because `emotion_entropy` crossed its threshold.

### Two corrections to what this file said earlier

- **Fix E worked.** I recorded it as a failure from the pooled comment-level CV
  (0.874 -> 0.893), which is not the metric. The metric is per-thread
  `length_cv`: it went **PARTIAL d=-0.54 to PASS d=-0.40**, value 0.815 against
  a real 0.812. Measuring a statistic that resembles the metric is not
  measuring the metric.
- **Fix B worked.** `hard_disagree_rate` reached d=**0.17**, the best of any
  version, value 0.125 against a real 0.091. Removing two of the three parent
  prohibitions let replies push back the way real ones do.

### emotion_entropy: concentration, not vocabulary

The pooled label mix is effectively identical between v69 and v72 -- neutral
57.1% vs 57.9%, approval 13.7% vs 13.8%, annoyance 10.4% vs 10.8% -- and v72
actually reaches more distinct labels (18 of 28 against 16). Mean per-thread
entropy still fell 1.358 -> 1.260 against a real 1.325. The affect mix is right
across the collection and too concentrated *inside* each thread. That is an
allocation problem in `_assign_labels`/`_affect_cost`, not a missing-emotion
problem, and it is worth diagnosing before another paid run.

### The metric no version has ever passed

`self_bertscore_mean_f1`: 0.515 against a real 0.463, d=0.82. It is the largest
remaining gap and the only metric that has failed in every version measured.

---

# Diagnosis after v72: why the remaining metrics fail

All measurements below are v72 against **the 10 matched real threads**, not the
763-thread global average. That distinction matters and I had it wrong earlier:

| metric | generated | matched real | (global real, what I quoted before) |
|---|---|---|---|
| emotion_entropy | 1.260 | **1.636** | 1.325 |
| length_cv | 0.815 | **0.948** | 0.812 |
| self_bertscore_mean_f1 | 0.515 | **0.496** | 0.463 |
| self_bleu_4 | 0.038 | **0.030** | 0.031 |
| semantic_mean_cosine | 0.311 | 0.305 | 0.280 |

Against the correct baseline the bertscore and cosine gaps are much smaller than
I reported, and the emotion gap is much larger.

## One root cause behind polite_rate, impolite_rate and emotion_entropy

**The tone assignment is not realized.** Assigned vs realized, 520 aligned slots:

| assigned | realized as assigned | realized impolite |
|---|---|---|
| polite (136) | **10 = 7.4%** | 88 = 65% |
| somewhat_polite (44) | 2 = 4.5% | 32 = 73% |
| neutral (78) | 10 = 12.8% | 52 = 67% |
| impolite (262) | 172 = 65.6% | — |

Everything collapses into impolite. This also explains the affect failure: the
warm affects are scheduled onto polite slots, and a warm affect on a slot whose
tone was not realized is realized **2 of 86 times**. Assigned admiration 6.5% ->
realized 0.8%; remorse 5.4% -> 0.2%; annoyance 2.3% -> **10.8%**.

The assignment itself is correct — `affect_counts` equals `affect_target_counts`
exactly, and assigned neutral (50.4%) is close to the real dominant-neutral
share. Realization is the whole problem, exactly as with openers.

### Why the Writer cannot produce "polite"

Not length. I tested my own recorded calibration (polite is length-driven in
real data) and it fails on generated text:

| words | generated polite | real polite (11,817 held-out) |
|---|---|---|
| 10-25 | 2% | 17% |
| 60-120 | 11% | **52%** |
| 120+ | 11% | **64%** |

Generated long comments are 70-89% impolite. Real long comments are majority
polite. Length is available; the register is not.

Measured against the 631 matched real comments:

| | generated | real |
|---|---|---|
| negation markers | 33.5% | **47.2%** |
| warm/appreciation markers | 8.7% | **20.6%** |
| emotional endpoint ("I love it", "so glad") | 0.2% | 2.4% |
| decision-framing nouns ("the part that matters", "value proposition") | **8.1%** | 1.1% |

Real comments contain *more* negation than generated and are still scored
polite. The difference is warmth, not disagreement. A real polite comment reads
"I owned it for a year but I didn't like the slow startup... I sold it and
bought the Ricoh and I love it" — negative content, personal narrative,
emotional endpoint. The generated equivalent ends on an analytical verdict:
"the value proposition gets a lot narrower pretty quickly."

## The register comes from the prompt

One three-word frame — "the part that / that's the bit / the annoying part" —
appears in **20% of generated comments and 0 times in 39,265 real tokens**. It is
not self-echo: it is 20% in the *first* comment of a thread, before any history
exists, and flat across all positions. The `prompts.py` comment claiming this was
fixed by rewording is wrong.

The prompt averages **22,096 characters to produce a 56-word comment** — about
395 prompt characters per output word — and carries a mean of **66 abstract
decision-analysis words** (decision, boundary, proposition, contribution,
increment, anchor, axis, claim). The slot's own proposition is restated
**verbatim 3.4 times** in its own prompt and the decision boundary 3.2 times,
across four separate blocks: the route lock, the private controls, the planner
intent, and the semantic difference contract.

A Writer asked, four times over, to "state the one new proposition this slot
owns" and "the only decision boundary you may establish" writes "that's the part
that actually matters". The output register is the prompt's register.

## Contradictory controls still shipped

180 pair-instances across 522 slots:

| pair | count |
|---|---|
| voice=annoyed + affect_role=neutral | 55 |
| real_surface_shape=full_answer + surface_skeleton=1-sentence | 49 |
| speaker_role=confused_asker + tone_target=impolite | 34 |
| tone=impolite + affect=approval | 20 |
| tone=impolite + affect=remorse | 19 |

`_affect_fits_tone` only excludes a small set. A slot told to be an annoyed,
impolite, confused asker with a neutral affect and a one-sentence full answer
has no realizable target, so the model falls back on its default: flat analytic
impolite prose.

## What this implies

The failing metrics are not five independent problems. They are one: the
Planner-Writer interface describes a comment in decision-analysis vocabulary,
and the Writer reproduces that vocabulary and register. Adding more control
fields has been the strategy for ~15 versions; the control-realization rates
(tone 7.4% polite, openers 49%, warm affect 2%) say the interface is saturated.

---

# Plan after v72 (2026-08-14)

## Correction on the baseline

No p-value, Cliff's delta or Wasserstein number ever changed. The evaluation
always compared against the 10 matched real threads. My prose quoted the
763-thread global average as "real", which understated the emotion gap and
overstated the bertscore gap. The corrected matched-real values are in the
diagnosis section above.

## The N=150 problem

Critical |Cliff's delta| = 1.96*sqrt(2N+1)/(N*sqrt(3)):
N=10 -> 0.519, N=50 -> 0.227, N=150 -> **0.131**, N=400 -> 0.080.

v72 projected to N=150: **3 of 12 pass**, and the three that pass
(`avg_depth`, `structural_virality`, `mean_story_probability`) are the ones the
matched-thread sampler constrains directly, not evidence of generation quality.

Four of the seven current N=10 passes are undetected, not close:
`semantic_mean_cosine` 0.20, `neutral_rate` -0.30, `hard_disagree_rate` 0.17,
`length_cv` -0.40.

All three dimensions that matter fail at N=150: diversity (0.82 / 0.64 / 0.20),
emotion (0.74 / -0.65 / -0.64 / -0.30 / 0.17), length (-0.40).

## Track 1 - remove what is measurably hurting (no new mechanism)

1. **Stop shipping contradictory control pairs.** 180 pair-instances over 522
   slots. `_affect_fits_tone` checks only tone-affect; extend the joint check to
   voice/affect, role/tone, and surface_shape/skeleton, and drop the label that
   loses rather than shipping both.
   *Signal*: contradictory pairs 180 -> 0; tone realization above 7.4%.
2. **Render each planned proposition once.** `semantic_move` is restated
   verbatim 3.4x and `decision_boundary` 3.2x per prompt across the route lock,
   the private controls, the planner intent, and the semantic contract.
   *Signal*: prompt 22,096 -> under 12,000 chars; no metric regression.
3. **Keep decision-analysis vocabulary out of the rendered prompt.** 66 such
   words per prompt. The Writer needs the content of the constraint, not the
   words "decision boundary", "increment", "proposition", "contract".
   *Signal*: "the part that / the bit that" frame 20% -> under 5%;
   `self_bleu_4` and `self_bertscore` down.

## Track 2 - length distribution

4. **Raise the beat ceiling and enforce delivery.** `MAX_DEVELOPMENT_BEATS=24`
   at 21 words/beat caps reachable length near 504 words; the 845-word slot got
   7 beats against 24 requested and produced 197 words. Realized output is 24.8
   words/beat, so the per-beat budget is right and the cap and the delivery rate
   are not.
   *Signal*: 200+ word slots from 0.51x to above 0.8x; `length_cv` -0.40 toward 0.
5. **Hold the short end.** Slots under 30 real words run 1.15x long. Fix E added
   a matched-slot ceiling but it is a soft diagnostic; make it blocking for
   short slots only.

## Track 3 - register learned from held-out threads (the domain adapter)

The control interface is saturated: tone realizes at 7.4%, warm affect at 2%,
openers at 49%. Adding a sixteenth control field will not reach the register.

What the register difference actually is, measured on 631 matched real comments:

| | generated | real |
|---|---|---|
| negation markers | 33.5% | **47.2%** |
| warm/appreciation markers | 8.7% | **20.6%** |
| emotional endpoint | 0.2% | 2.4% |
| decision-framing nouns | **8.1%** | 1.1% |

Real comments disagree *more* and are still scored polite. Warmth and personal
narrative, not agreement, is what the metric keys on.

Two ways to supply it, both domain-general:

- **3a (recommended): register exemplars.** Give the Writer 2-3 whole comments
  drawn from the domain's evaluation-excluded threads, selected for the
  register this slot is assigned, and labelled as style evidence, not content.
  Legitimate: the 424 held-out threads are already disjoint from the evaluation
  seed pool and already feed the Planner's 2,557 reference viewpoints. It does
  change CARD's zero-shot-style character, so it is a decision to make
  explicitly, and the run config must record it.
- **3b: register targets as more scheduled fields.** Measure warm-marker rate,
  narrative rate and negation rate per domain and schedule them like tone.
  Cheaper and preserves the current character, but the realization evidence
  predicts it fails the same way tone did.

## Validation order (cost discipline)

Track 1 is verifiable with **no API spend**: render prompts offline and check
the contradiction count, prompt size, and frame rate. Only after that is clean
should a paid run happen. A single small thread (~$0.3) can test Track 3a before
a full 10-thread run ($3.4).

At N=10 a metric needs |d| < 0.519 to pass and at N=150 it needs < 0.131, so a
run that merely restores 7-8/12 at N=10 has not answered the real question. The
target to steer by is effect size, not the pass count.

---

# v74 (`--writer-prompt focused`) content audit

Run: `generalized_card_camera_gpt54_v74_focused_20260814_v1`, 10 matched seeds,
522 slots, $2.2353 (v73: $3.3716), prompt 22,249 → 8,139 chars on the 416/522
slots that took the focused path (106/522 still take the unconverted
`_low_info_writer_prompt` at ~15,468 chars).

## Metric outcome: 7 PASS / 2 PARTIAL / 3 FAIL

Only five metrics have a p-value with room to survive N=150:

| metric | MWU | Cliff | verdict |
|---|---:|---:|---|
| `avg_depth` | 1.00 | 0.01 | matched (sampler-determined) |
| `structural_virality` | 1.00 | -0.01 | matched (sampler-determined) |
| `semantic_mean_cosine` | 0.79 | 0.08 | matched (generation win) |
| `mean_story_probability` | 0.57 | -0.16 | matched (generation win) |
| `neutral_rate` | 0.36 | -0.25 | matched — new in v74 |
| `hard_disagree_rate` | 0.054 | 0.52 | borderline, not matched |
| `length_cv` | 0.054 | -0.52 | borderline, not matched |

v73 had 4 genuinely matched, v74 has 5, at 33% lower cost. `self_bertscore`
still the worst metric in every version (MWU=0.00044, Cliff=0.94).

## Content diagnostics, real vs generated, all 10 threads

Measured with `scratchpad/content_diag.py`. Rates are % of comments.

| tell | real | gen | reading |
|---|---:|---:|---|
| curly `’ “ ” —` | 10.9 | **68.1** | LLM typography, model-emitted |
| straight `'` inside a word | 44.3 | **0.7** | ditto, inverse |
| specs/comment (`f/2.8`, `28mm`, `ISO 1600`, `$900`, `18w`) | **0.55** | 0.06 | 9× less concrete |
| distinct new brand/model tokens per thread (not in seed post) | **47.3** | 5.8 | 8× narrower referent set |
| contains a URL | **4.5** | 0.0 | generated never links |
| contains an emoji | **2.1** | 0.0 | |
| `lol/haha/lmao` | 0.6 | 0.0 | |
| ALL-CAPS word (3+) | **22.7** | 6.6 | |
| blank-line paragraph break | **32.8** | 3.1 | 10× — long generated comments are one wall |
| ends without final punctuation | **26.4** | 8.4 | |
| ellipsis `...` | 6.2 | 1.2 | |
| analytic-frame template | 0.15 | **7.2** | up to 28% on one thread |
| ends on a question | 5.8 | **12.2** | 2× too many |
| max comment words (thread mean) | **262** | 153 | real maxima 845/337/331 |
| word-count CV (pooled, per thread) | 1.01 | 0.87 | |

Curly-quote rate is identical pre- and post-`gpt_cleanup` (72/72, 79/79, 83/83),
so the typography comes from the writer model, not the cleanup pass.

## Length ceiling is structural

`scripts/sampling_generator/engine/vocabulary.py:195` —
`LENGTH_BUCKET_BOUNDS["very_long"] = (120, 220)`. Real threads exceed 220 words
in 10/10 matched threads (max 845). No planned slot can reach the real upper
tail, which caps `wc_max` and holds `length_cv` at Cliff=-0.52.

## Negative result: novel-entity deficit does NOT explain `self_bertscore`

| seed | real ents | gen ents | real bert | gen bert |
|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0.492 | 0.546 |
| 6 | 13 | **18** | 0.500 | 0.534 |
| 7 | 34 | 6 | 0.512 | 0.513 |
| 8 | 144 | 13 | 0.489 | 0.526 |

Seed 6 introduces *more* novel entities than the real thread and still
overshoots by 0.034; seed 7 has 6 vs 34 entities and matches to 0.001. The
overshoot is a near-uniform +0.033 offset on 9/10 threads (real range
0.481-0.514, gen 0.505-0.546) — a shared register signature applied evenly, not
topical narrowness. Cliff=0.94 comes from the consistent sign, not the size.

## Politeness is the deepest content miss

`polite_rate` real median 0.378 vs gen 0.080 (4.7× low); `impolite_rate` real
0.435 vs gen 0.621. Generated comments are curt-analytic where real comments are
warm and concrete, and this feeds `emotion_entropy` (real 1.76 vs gen 1.43).

---

# v75 change 1: the Writer realizes the Planner's move instead of transcribing it

Policy version `generalized-card-v2-writer-realizes-planner-move-v75-20260814`.
v74's version string is now in `HISTORICAL_GENERATION_POLICY_VERSIONS` so the v74
run stays evaluable. Six pinned files changed and were re-pinned; the drift check
reported exactly those six and nothing else.

## What was wrong

Two instructions, both already present in the v73 checkpoint, told the model to
copy rather than write:

- `prompts.py` route lock: `"- Say this, and only this: " + compact(move, 260)`
- `reply_planning.py` schema: `"a full sentence stating what this reply asserts
  ... not a bare noun phrase"`

19.3% of the 522 planner moves open with the word "I", so "say this, and only
this" sat in front of a finished first-person sentence. v74 then removed the
counterweight -- `_focused_writer_prompt` dropped the semantic-difference
contract on the reasoning that no metric depended on it. No metric measured plan
echo.

Longest contiguous shared word run between `semantic_move` and its own comment,
share at 12 words or more, ~520 slots per run:

| run | all comments | comments >= 25 words |
|---|---:|---:|
| v67 | 0.4% | **0.0%** |
| v69 | 1.0% | 1.5% |
| v73 | 10.2% | 11.7% |
| v74 | **25.8%** | **34.7%** |

Split by slot type in v74: reply slots 25.1%, root slots 6.4%. The root schema
has always said "one concrete but non-verbatim action".

## What changed

New ablation switch `--writer-route-lock own_words|say_only`
(`GENERALIZED_CARD_WRITER_ROUTE_LOCK`), recorded in `run_config`. `say_only`
reproduces v74 on both the Writer and Planner side.

1. Route lock reads `"- The point this comment makes, in your own words: "`.
2. Reply schema reads `"one concrete but non-verbatim action for this reply, at
   the same scale as a top-level slot's semantic_move - name what it asserts,
   not a bare noun phrase and not the comment's own drafted sentence"`. The scale
   clause is what stopped bare noun phrases, so it stays.
3. `_realization_rule` restores the counterweight: *the point above is a
   specification of what to say, never wording to reuse*. Rendered by **both**
   the focused and the low-info path. 106 of 522 v74 slots took the low-info
   branch, and applying a fix to only 80% of slots is what made the v74 result
   impossible to attribute.
4. `_render_reply_delta_contracts` carries the same wording change so the root
   prompt's reply contract does not contradict the reply planner's schema.

Diff of the rendered writer prompt, `say_only` -> `own_words`: two hunks, the
route-lock line and the new rule. +192 characters on a ~8,139-character prompt.

## Also fixed: `micro_reaction` was dropping comments

`shape_writer_text_for_task` overwrote any over-length micro slot with
`micro_options[local_task_id % 6]`. One v74 thread held **10** micro slots
against a pool of 6, so tasks 26, 32 and 116 all resolved to `"This"`; the first
was kept and the other two raised `exact_duplicate` on every repair round,
because local repair never changes the task id. Both were dropped after the
budget. It also poisoned `self_bleu` with a sibling similarity of 0.9999998.

The pool is now 14 strings and the index also depends on the candidate's own
text, so a fresh attempt lands elsewhere. Verified against the real collision:
tasks 26/32/116 now give `Nah` / `Solid` / `This`.

## Not done yet, deliberately

`missing_concrete_anchor` (84 hits) and `lexical_overlap_high` (79 hits) stay
advisory. Making them blocking now would fight the prompt's own
`"Name a product, model, or number only if it is visible above"` rule, so the
writer could not satisfy them; that is P4/P6, not P0.

The plan-echo **validator** is also deliberately not added yet. Tracing the
acceptance path: a problem code outside `REPAIRABLE_WRITER_PROBLEMS` reaches
`backend.py:2022` and returns `skip: True`, and repair exhaustion at
`backend.py:2205` does the same. `retry_note` is only populated inside
`generate_writer_text`'s own loop, which runs once under `--writer-retries 0`,
and the adapter's repair loop calls it with `writer_retries: 0` while modifying
only `planner_intent` and `must_not_do` -- neither of which the focused prompt
renders. So a detector added before this change would have regenerated an
identical prompt and dropped up to 130 slots. It becomes a cheap safety net once
echo is back near the v67 floor, and it needs `--writer-retries >= 1` to have a
repair channel.

## Verification

- 224 tests pass (218 before, 6 new). The route lock had **no** test coverage
  before, which is why it regressed twice unnoticed.
- New: `WriterRouteLockTest` (4), `MicroReactionShapeTest` (2), plus a low-info
  path assertion added to the existing core-path test.
- `core_contract` drift check: none. Backend self-test: PASS.
