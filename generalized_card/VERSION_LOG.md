# Generalized CARD Version Log

## Read this before quoting any pass count

Pass counts are only comparable between runs with equal **structural coverage**
(generated comments / matched real comments over the same seeds). Measured over
the 10-seed pool, whose matched real threads hold 515 comments:

| version | coverage | metrics passing |
|---|---:|---:|
| v19 | **0.45** | 11 |
| v14 | 0.60 | 9 |
| v16 | 0.67 | 9 |
| v15 | 0.68 | 9 |
| v34 | 0.67 | 8 |
| v33 | 0.71 | 8 |
| v64 | **1.01** | 6 |

Coverage and pass count move together, and the mechanism is not subtle:
`self_bleu_4`, `self_bertscore_mean_f1`, and `semantic_mean_cosine` are means over
all comment pairs, so a shorter thread has fewer pairs and lower mean pairwise
overlap. `length_cv` shifts as well. **Truncating a thread flatters exactly the
metrics this work is trying to match.**

So v19's 11/12 is a truncation artifact, not a target: it generated 231 comments
where the real threads held 515. v34 generated 76 comments against a 186-comment
real thread. v64 was the first version to generate complete threads, which is why
its 6/12 is the first honest measurement rather than a regression. **Only v64 and
later are comparable to one another.**

The historical runs therefore violated this repository's own rule in
`AGENTS.md`: "For first-pass Planner-Writer generation, preserve every matched
structural slot. Never shrink, cap, or omit a matched thread by default."
`RUN_INDEX.md` reports comment counts per run; check them against the seed pool
before drawing any comparison.


One entry per generator policy version: what changed, why, which run tested it,
and what happened. Git history, this file, `core_contract.py`'s historical policy
set, and each run's `run_config.json` together form the provenance chain.

Machine-generated companion: [`RUN_INDEX.md`](RUN_INDEX.md), rebuilt with

```bash
PYTHONPATH=generalized_card python3 generalized_card/scripts/build_version_log.py
```

## Recording a version

Before any run that changes behavior:

1. Bump `GENERALIZED_V2_GENERATION_POLICY_VERSION` in `core_contract.py` and move
   the previous value into `HISTORICAL_GENERATION_POLICY_VERSIONS`.
2. Recompute the pinned `CORE_FILES` hashes.
3. Add an entry below **before** spending the API call, stating the hypothesis
   and the predicted direction, so a null result stays interpretable.
4. Use a run tag containing the version number.
5. After evaluating, fill in the result and regenerate `RUN_INDEX.md`.

---

## v81 — joint story/affect handoff and prompt-residue removal (2026-08-17)

Policy ID: `generalized-card-v2-joint-story-affect-handoff-v81-20260817`.
The implementation commit is the git entry that adds this section; every paid
artifact additionally stores its exact source/config snapshot.

v80 showed that making the Writer instruction stronger was not enough. The
direct-reply Planner saw fixed social labels as prose but did not return them in
its schema, 61 slots used firsthand evidence against a 17-story quota, and 104
short replies copied the `development_plan` schema example into a real plan.
Post-parse normalization then hid bad plans by rewriting them to one repeated
gratitude sentence or to `soft_helpful`.

Changed:

- Story is now a bidirectional Planner invariant. `no_story` rejects firsthand
  evidence and personal-story payloads; a story slot requires firsthand,
  personal-datapoint semantics. Unresolved story/surface/long-form contracts
  stop before the Writer instead of being logged and persisted.
- Direct replies receive story, tone, affect, and opener controls as structured
  per-slot contracts. A no-story row cannot choose the explicitly narrative
  `corroborating_datapoint` route.
- Short slots deterministically clear any copied development-plan prose. Both
  Planner schemas now use literal `none` and explicitly require it below the
  long-form threshold. Root and direct-reply prompts use the same dynamic beat
  capacity function as validation; the conflicting 35-word/16-beat prose was
  removed.
- Removed semantic post-parse rewrites. Gratitude/relief and substantive-slot
  conflicts go through targeted Planner repair; no shared canned semantic move
  and no automatic `soft_helpful` conversion remain.
- Tone/affect marginals are paired jointly before planning. On the frozen v80
  seed-8 template the new schedule assigned every label while reducing
  `approval+impolite` 10→0 and `neutral-affect+polite` 27→2.
- The focused Writer renders the tone definition once, gives neutral affect a
  non-conflicting instruction, and omits known schema defaults from its
  semantic ledger. Impolite and amusement contracts explicitly permit
  non-targeted profanity and natural laughter tokens, respectively, without
  requiring a fixed phrase.
- First-pass distribution resampling is disabled at the public CLI. Repetition,
  Self-BLEU, Self-BERTScore, and semantic cosine are collection diagnostics;
  only non-persistable Writer failures retain bounded recovery.

Offline acceptance before the first run:

- v80 replay: 104 short development residues removed; 59 latent story-contract
  conflicts detected rather than passed through.
- the v80 template's 186 tone/affect assignments remain complete with zero
  unassigned labels and zero story/social-close collisions.
- expected direction: story probability down toward the frozen template;
  emotion realization and entropy stability improve; shared prompt scaffolding,
  Self-BLEU, Self-BERTScore and helpful/explainer register decrease. Structure
  is unchanged because every matched slot and parent edge is preserved.
- complete test suite: 259 passed; backend self-test passed; 72 pinned source
  files report zero missing and zero drifted entries.

Formal acceptance still requires a multi-thread matched evaluation. An n=1 run
is only a content and contract diagnostic.

## v80 — coherent Planner social contracts (2026-08-16)

Tag: pending; do not start with a paid run.

Measured diagnosis on the existing v79 seed-8 artifact:

- Only 17 comments were assigned a story mode, and they contributed about 25%
  of the thread's total StorySeeker probability. Among 167 `no_story` comments,
  25 were still classified as stories. The highest-scoring rows retained
  `payload_type=personal_story` or a temporal firsthand plan after the schedule
  overwrote only `story_mode`.
- Of 46 `polite` slots, only 6 realized as polite; 27 realized as impolite. The
  Planner prompt already requires an agreeing personal datapoint, reaction, or
  positive verdict, but mismatching roles/functions survived because the
  post-normalization quality gate did not check that contract.
- On 40 existing comments (780 unordered pairs), changing only curly apostrophes
  to ASCII moved Self-BERTScore 0.52947 -> 0.52381. Curly double quotes had
  effectively no effect. This is a real but secondary global signature, not an
  explanation of the full 0.034 v79-vs-real gap.

Changed before any API call:

- Plan-quality validation now rejects `no_story + personal_story` and incoherent
  polite role/stance/function combinations, so targeted Planner repair operates
  on the whole semantic contract instead of relabeling one field after planning.
- Every `no_story` Writer path now explicitly forbids a temporal event sequence
  while still allowing one firsthand observation.
- Polite guidance now follows the observed real-discussion cues: ordinary
  hedging and brief thanks are allowed, an emotional endpoint is required, and
  repeated abstract decision-framing is discouraged. The refuted generated-data
  length hint was removed.
- Direct-reply planning now exposes sibling coverage, including already committed
  sibling delta types and novelty anchors.
- Both interventions have explicit ablations:
  `--social-contract-coherence off` and
  `--reply-sibling-visibility off` restore the pre-v80 arms, and both fields are
  part of the recorded and resume-checked experiment identity.
- Resume/extension/upgrade checks share one experiment-field list that includes
  every behavior flag. The prior implementation wrote those flags to the run
  record but omitted them from lineage comparison.
- Removed proven-unreferenced helpers and stale tone-example rewrites; generalized
  Planner prose no longer assumes every domain is equipment/products.

Predicted direction: planned-social-contract realization above v79's 59.2%,
`no_story` StorySeeker mass down materially, polite realization above 13%, and no
change to matched tree structure. Validate plan-contract counts and prompt
snapshots before a paid run; evaluate p-values only after a comparable multi-seed
run.

## v68-v79 provenance correction (recorded 2026-08-16)

The narrative log previously stopped at v67 even though run artifacts and the
historical policy set continued through v76. The durable record is:

- v68: domain-claim/entity generalization.
- v69: scheduled opener grammar; evaluated on ten threads at 8/12, with the
  cancellation caveats described in the handoff.
- v70: domain-claim field survival; the recorded smoke was not fully evaluated.
- v71: Planner-owned reply move and single-parent exclusion; ten-thread result
  4/12.
- `v72_noclaim` was an experiment tag, not a policy version: its run config
  correctly retained the v71 policy string. It scored 7/12.
- v73: affirmative affect and uncapped anonymous slot shape; 8/12.
- v74: focused Writer prompt; 7/12.
- v75: Writer realizes the Planner move in its own words; the evaluated repeat
  scored 4/12.
- v76: own-fact-license experiment arms.
- v77, v78, and v79 changed repetition/recovery behavior but incorrectly reused
  the v76 policy string. They are retained as artifact tags, not claimed as
  reproducible policy releases, and must not be ranked from their one-thread
  12/12 p-value output.

---

## v64 — calibrated tone registers and length scale (2026-08-13)

Tag: `generalized_card_camera_gpt54_v64_tone_smoke10_20260813_v1` (10 threads, 521 comments)

Changed:
- `TONE_CLASSES` extended to the classifier's full four-way partition. The
  reported metrics stay polite/impolite/neutral, but planning over three classes
  had renormalized the missing `somewhat_polite` mass onto the reported three.
- `TONE_DEFINITIONS` rewritten from measurements on 11,817 evaluation-excluded
  camera comments rather than a generic notion of manners.
- `_tone_cost` reversed: polite now routes to longer slots, matching the observed
  distribution, instead of the shortest compatible slot.
- Writer's blanket ban on acknowledgement and first-person framing scoped so it
  no longer cancels the tone control it sits next to.
- `allow_first_person_frame` no longer forced off for a no-story polite slot.
- Beat budget moved from one beat per 80 words to one per 35.

Result: **6/12 pass**, down from v34's 8/12.
- Improved: neutral_rate PARTIAL→PASS, semantic_cosine 0.21→0.31,
  avg_depth and structural_virality to p=1.00, planner→writer tone contract
  fidelity 40.1%→54.7%, somewhat_polite rate 0.269→0.124 against a real 0.125.
- Regressed: self_bleu_4 PASS→FAIL, self_bertscore PASS→PARTIAL,
  emotion_entropy PASS→FAIL, impolite_rate worse.
- polite_rate did not move (0.068→0.048 against a real 0.297).

Diagnosis of the regression: the tone text prescribed sentence structure
("Lead with the disagreement"), which gave every same-register comment a shared
entry route and inflated within-thread lexical and semantic similarity. The beat
change had almost no effect because it was aimed at the wrong constraint.

## v65 — tone-compatible reply increments and reply development plans (2026-08-13)

Tag: `generalized_card_camera_gpt54_v65_bigthread_seed78_20260813_v1`
(1 thread, seed_index 78, 197 comments, $1.95, 24 min)

Hypothesis: polite could not be realized because the Planner's schema could not
express a warm reply at all. `REPLY_DELTA_TYPES` held seven values, six of them
inherently critical, so 92% of polite-planned slots were planned as
`speaker_role=advisor` delivering a technical adjudication — content no tone
control can turn warm.

Changed:
- Added `corroborating_datapoint`, `useful_extension`, and
  `endorsement_with_reason`, and gated the allowed set per tone register.
- Propagated the new vocabulary to every consumer: the direct-reply planner
  schema and rules, the root planner schema and rules, the reply-delta contract
  block, the Writer's `realization_by_type` route lock, and
  `planning_quality.reply_increment_problem`, which had been rejecting the new
  types as "generic agreement".
- Joint tone/affect assignment, so a polite slot can no longer receive
  disapproval, anger, or disappointment.
- `development_plan` added to the direct-reply planner, which had omitted the
  field entirely. Every long slot at depth ≥ 1 (33 of 77) was receiving no
  development guidance and was realized at ~0.72x its matched length.
- Per-slot beat requirements now stated on each row in both planners.
- All sentence-structure prescriptions removed from the tone guidance.

Predicted direction: polite fidelity up from 6%; advisor share down from 72%;
long-slot ratio up from 0.72; self_bleu_4 and emotion_entropy recovered to at
least v34 levels now that the shared entry routes are gone.

Result: every predicted plan-level change landed.

| | v64 | v65 |
|---|---:|---:|
| advisor share of slots | 72% | 9% |
| stance=agree | 14% of polite slots | 64% of all slots |
| supportive delta types | absent | 57% of replies |
| long slots with a development_plan | 30% | 95% |
| ... of those at depth >= 1 | 0 of 33 | 12 of 12 |
| long-slot length ratio (100+ words) | 0.72 | 0.87 |
| polite contract fidelity | 6% | 14% |
| generated polite_rate | 0.048 | 0.117 |

Caveat: v65 ran one thread at seed_index 78 while v64 ran seeds 0-9, so the
realization numbers are not a clean A/B. The plan-level counts are unambiguous
because they measure the fields that were changed. Seed 78 is now the fixed
iteration thread so later versions compare against this row directly.

Remaining defect: 64% of polite-planned slots are still classified impolite.
Inspecting the text separates the two groups cleanly by the *valence of the
concrete object*, not by stance, role, or length (misses average 80 words, hits
57):

- Miss: "that's a genuinely awkward spot", "the body stopped seeming so alien",
  "if the body doesn't put the buttons where your fingers expect, it never
  really settles in", "what broke for me was...". The slot agrees with its
  parent but corroborates a *friction*.
- Hit: "one thing that helped me was...", "it genuinely made the body feel less
  intimidating", "that was the bit that clicked for me", "Appreciate that".
  The concrete object is a *resolution or benefit*.

`corroborating_datapoint` accounts for 28 of the 78 misses: the Writer confirms
the parent's difficulty rather than a positive outcome. The supportive delta
definitions are valence-neutral, so a warm register attached to a
friction-shaped anchor still reads as complaint.

Deferred: making the supportive increments valence-bearing for polite slots.
Politeness was deprioritized in favour of diversity and emotion.

## v66 — held-out entity inventory, unseeded route lock, route ledger, beat rate (2026-08-13)

Tag: pending. Same seed as v65 (seed_index 78) so the comparison is a real A/B.

Priorities reset: diversity (`self_bleu_4`, `self_bertscore_mean_f1`,
`semantic_mean_cosine`) and `emotion_entropy` matter most; politeness least.
Against that ordering, the v65 thread stood at:

| metric | real | v65 | verdict |
|---|---:|---:|---|
| semantic_mean_cosine | 0.2825 | 0.2490 | already past real |
| emotion_entropy | 1.9394 | 2.1037 | already past real |
| avg_depth / structural_virality | 2.244 / 3.971 | 2.250 / 3.971 | matched |
| **self_bleu_4** | 0.0264 | 0.0338 | too repetitive |
| **self_bertscore_mean_f1** | 0.5026 | 0.5188 | too similar |
| **length_cv** | 0.9456 | 0.8515 | too narrow |

Reading the matched real thread rather than only its statistics produced the
main finding. Over the same 197 slots:

| | real | v65 |
|---|---:|---:|
| repeated 4-gram share | 0.0545 | 0.0790 |
| **distinct camera models named** | **117** | **23** |
| most frequent model's share of mentions | 0.03 | 0.29 |
| no-end-punctuation share | 0.183 | 0.091 |

The Writer's rule "named entities may appear only when visible in the discussion
or in the visible factual anchors" is correct for claims *about* the seed, but it
also means all 197 comments can only ever name the two or three products the seed
mentions. `the sony a7 iv` appeared 9 times, `sony a7` 22, `the a7` 19. Real
commenters name their own gear instead, which is what spreads entity mass.

Changed:
- **E1** New `entity_inventory` module and profile field (schema 10): equipment
  designators learned by brand adjacency over the 424 evaluation-excluded
  threads, then counted in every form. 63 clean designators for camera. Offered
  to the Writer only on slots whose plan already licenses first-person
  experience, rotated by slot so mass spreads, excluding anything already visible
  in the slot, and licensed strictly as the speaker's own gear.
- **D1** `_semantic_route_lock` said "make this the part that changes the
  parent"; the Writer echoed "that's the part that…" 18 times. Reworded so the
  scaffolding no longer contains the construction it asks for.
- **D5** `used_sentence_routes` ranked by recency and carried no counts, so the
  entrenched routes were pushed out of the ledger by recent one-offs. Now ranked
  by reuse with counts attached.
- **L1** `WORDS_PER_REALIZED_BEAT` 35 → 21 and the cap 16 → 24, from the measured
  realized rate (246/12, 179/8, 134/6 words per beat).

Also recorded: 190 of 197 v65 prompts already listed `that s the part` as a
repeated four-gram and 23 comments used it anyway, and the running self-BLEU
plateaued at 0.0335 by comment #62 while the calibrated band only contracted
enough to flag it at #182. Prompt-level exclusion lists do not work here, and the
existing guard cannot detect the problem while it is still fixable. If v66 does
not close the self-BLEU gap, those two are the next targets rather than more
prompt wording.

Predicted direction: distinct models 23 → 50+ with top-model share well under
0.29; repeated 4-gram share moving from 0.0790 toward the real 0.0545;
`self_bleu_4` gap and `self_bertscore` gap both shrinking; long-slot ratio from
0.87 toward 1.0 and `length_cv` from 0.8515 toward 0.9456; `semantic_cosine` and
`emotion_entropy` holding.

Tag: `generalized_card_camera_gpt54_v66_entity_seed78_20260813_v1`
(same seed as v65, 195 comments, $1.99, 23 min)

Result: **the changes did not land.** Every predicted magnitude missed.

| | real | v65 | v66 | predicted |
|---|---:|---:|---:|---|
| repeated 4-gram share | 0.0545 | 0.0790 | 0.0755 | toward 0.0545 |
| distinct models | 117 | 23 | **27** | 50+ |
| top model share | 0.032 | 0.289 | **0.308** | well under 0.29 |
| distinct 3-word openers | 0.888 | 0.772 | 0.810 | — |
| "that's the part/bit" | 0 | 24 | **19** | down |
| long-slot ratio | — | 0.87 | **0.88** | toward 1.0 |
| thread word CV | 0.943 | 0.856 | **0.823** | toward 0.946 |

Why E1 missed, measured the same way as the earlier exclusion-list check: 129 of
195 prompts (66%) offered the equipment shortlist, and only **21 of those 129
(16%)** named an offered item. The Writer ignores an optional affordance in this
prompt exactly as it ignores an exclusion list.

Why D1 only half worked: `that's the part` fell 18 → 9, but `that's the bit` rose
6 → 10, and a new frame `the rest of the` appeared 11 times. Removing the seeded
wording made the model reach for a synonym; the underlying rhetorical act was
untouched.

Why L1 missed: the Planner supplied 7.0 of 7.8 requested beats, so planning
complied, but doubling the beat budget produced no extra length (0.87 → 0.88) and
the CV fell. One 282-word slot collapsed to 92 words despite 12 planned beats.

## Cross-cutting conclusion after v64-v66

Compliance depends on the *kind* of control, not on its wording:

| control | kind | compliance |
|---|---|---|
| `tone_target=impolite` | planned categorical field | 86% |
| `development_plan` present vs absent | planned field, presence | 0.76 → 0.88 ratio |
| `tone_target=polite` | planned categorical field | 14% |
| equipment shortlist | prompt affordance | 16% |
| beat count doubled | planned field, magnitude | no effect |
| repeated-n-gram exclusions | prompt rule | ~0 (23 violations after being shown) |
| opener/route exclusions | prompt rule | ~0 |

The Writer follows *what kind of thing to say*. It does not follow *how much*,
*how not to*, or *which optional resource to use*. The writer prompt averages
23,000 characters and 84 bulleted rules to produce a ~270-character comment, so
nothing in the rule mass is being attended to.

**Therefore: adding or rewording prompt text cannot fix diversity or length.**
Three runs now support that. The remaining levers are structural:

1. Radical prompt reduction — cut the Writer prompt from ~23k to ~3k characters
   containing only the plan and the visible context. Untested, cheapest, and it
   attacks the common cause of every non-compliance above.
2. Act on the guard that already exists. `lexical_overlap_problem` computes the
   evaluator's exact self-BLEU per candidate against a held-out band;
   `writer_local_repair_rounds` is currently 0. v19, the historical 11/12 run,
   had it at 2. This touches the `AGENTS.md` prohibition on best-of-N for a
   distribution metric and needs an explicit decision.
3. Deterministic non-LLM surface transforms after generation. Effective but it is
   post-hoc text editing, which is what this project set out to avoid.

Also fixed for the future: the running self-BLEU band contracts with progress, so
a uniformly slightly-too-repetitive thread is only flagged at ~92% completion.
Any variant of lever 2 must compare against the final target from early on.

## v67 — bounded thread blackboard (2026-08-13)

Tag: pending. Same seed as v65 and v66 (seed_index 78).

This measures the cause of the three preceding failures rather than another
control. Section sizes in the largest v66 writer prompt (67,284 characters):

| section | chars | share |
|---|---:|---:|
| **Structured thread blackboard** | **43,728** | **65.0%** |
| Hard rules | 5,171 | 7.7% |
| Per-slot instructions | 4,583 | 6.8% |
| Planner intent | 3,023 | 4.5% |
| One-shot semantic contract | 2,802 | 4.2% |
| Semantic route lock | 777 | 1.2% |
| Visible discussion | 576 | 0.9% |
| equipment shortlist | 341 | 0.5% |

Inside the blackboard, `semantic_coverage_entries` alone was 30,148 characters
across 140 entries: 69% of the blackboard and 45% of the whole prompt. Its
purpose is to hold down semantic repetition, and `semantic_mean_cosine` is the
one diversity metric already past real — so the largest block in the prompt was
over-serving the metric that needed nothing while crowding out everything else.

Every ledger cap scaled with thread length, so the blackboard grew without bound:

| slot | prompt | blackboard share |
|---:|---:|---:|
| 0 | 10,839 | 16% |
| 20 | 29,342 | 50% |
| 60 | 35,229 | 71% |
| 140 | 53,492 | **81%** |

By comment 140 the slot's own assignment was 19% of its prompt. That is why a
341-character affordance drew 16% uptake, why 190 prompts listing a banned
four-gram produced 23 violations, and why doubling the beat budget did nothing.

Changed:
- Every ledger capped at a constant instead of scaling with thread length.
- `semantic_coverage_entries` reduced to `move` and `boundary` per entry, capped
  at 24, and ranked by lexical relevance to the current slot rather than
  recency, so what survives the cap is what this slot could actually duplicate.
- `used_sentence_routes` capped at 20 (already frequency-ranked from v66).
- Earlier-comment tail 12 → 8 entries with 5 tags instead of 11.
- The short-line exclusion ledger is kept complete only for slots that could
  reproduce a short line; a long slot cannot, and it was pure prompt mass there.
  This preserves the exact-duplicate invariant exactly where it applies.
- `_tone_discourse_guidance_block` renders the assigned register and the one it
  drifts into, not all four.

Verified offline against the v66 tasks, no API call:

| | before | after |
|---|---:|---:|
| blackboard mean / max | 31,544 / 44,230 | **8,230 / 9,667** |
| writer prompt mean / max | 46,166 / 67,284 | **22,852 / 31,903** |
| plan share at slot 60 | 29% | 54% |
| plan share at slot 140 | 19% | **53%** |

Predicted direction: this is a compliance fix, so the controls that previously
missed should move without being re-worded — equipment uptake above 16%,
`that's the part/bit` below 19, long-slot ratio above 0.88, `length_cv` above
0.823, and `self_bleu_4` below v66's 0.0338. `semantic_mean_cosine` is the
metric most at risk, since its ledger shrank the most; it had headroom
(0.249 against a real 0.283) and is expected to rise but stay under real.

Tag: `generalized_card_camera_gpt54_v67_bounded_seed78_20260813_v1`
(seed_index 78, 197 comments, $1.48, 20 min. One earlier attempt was killed
externally at planning slot 131; a single-post run persists atomically at post
completion, so that attempt's $0.30 of planning was lost and it restarted.)

Result: **the compliance hypothesis holds.** Controls that had missed for two
versions moved without a single word of their wording changing.

| | real | v65 | v66 | v67 |
|---|---:|---:|---:|---:|
| writer prompt mean | — | — | 46,269 | **22,115** |
| writer prompt max | — | — | 67,284 | **31,009** |
| equipment uptake | — | — | 16% | **26%** |
| long-slot ratio | — | 0.93 | 0.88 | **0.99** |
| distinct models named | 117 | 23 | 27 | **39** |
| most frequent model's share | 0.032 | 0.289 | 0.308 | **0.175** |
| repeated 4-gram share | 0.0545 | 0.0790 | 0.0755 | **0.0707** |
| thread word CV | 0.946 | 0.856 | 0.823 | 0.846 |
| longest generated comment | 413 | 246 | 304 | 302 |

Long-slot ratio here is the mean of per-slot `generated/real`. An earlier note
reported 0.87 for v65 using the ratio of bucket means, which is a different
statistic; per-slot means are used throughout this table.

Still open after v67:
- `that's the part/bit` is 21, against 24 and 19 — flat. Shrinking the prompt did
  not touch it, because the shared *rhetorical act* is what produces it, not
  prompt pressure. This is the deferred D4 fix: schedule a `rhetorical_form`
  across slots the way tone, affect, and story are scheduled, instead of letting
  every slot invent its own `opening_style`.
- Distinct models 39 against a real 117. Uptake tripled the entity spread but the
  affordance is still optional. Making the equipment a *planned field* the
  Planner writes into the slot contract should close more of it, since planned
  categorical fields are the only control type this Writer reliably follows.
- No-end-punctuation share 0.066 against a real 0.183. Real comments drop final
  punctuation about one time in five; generated output is too polished.
  `surface_texture=no_punct_fragment` exists and is under-scheduled.
- Thread word CV 0.846 against 0.946, with the top still truncated at 302 words
  against a real 413.

Operational note: `--posts-per-run` persists at post granularity, so an
interrupted single-post run of a 197-comment thread loses all of its spend. Worth
changing to slot-level atomicity before many more long-thread iterations.

### v67 on the comparable 10-thread pool

Tag: `generalized_card_camera_gpt54_v67_smoke10_20260813_v1`
(10 threads, 520 comments, coverage 1.01, $2.97, 36 min)

**v67 did not improve the pass count: 5/12 against v64's 6/12.** Both runs have
coverage 1.00, so this is the only valid comparison available.

| metric | v64 p | v67 p | v64 \|d\| | v67 \|d\| | real noise p90 \|d\| | v67 closer on |
|---|---:|---:|---:|---:|---:|---:|
| semantic_mean_cosine | 0.307 | **0.791** | 0.28 | **0.08** | 0.48 | — |
| emotion_entropy | 0.016 | 0.049 | 0.65 | 0.53 | 0.44 | — |
| self_bertscore_mean_f1 | 0.017 | 0.002 | 0.64 | 0.82 | 0.44 | **7/10** |
| self_bleu_4 | 0.014 | 0.011 | 0.66 | 0.68 | 0.40 | 5/10 |
| length_cv | 0.017 | 0.021 | 0.64 | 0.62 | 0.46 | 4/10, **0/2 large** |
| hard_disagree_rate | 0.162 | 0.023 | 0.38 | 0.61 | 0.44 | — |
| mean_story_probability | 0.850 | 0.345 | 0.06 | **0.26** | 0.43 | 5/10, **0/2 large** |
| avg_depth | 1.000 | 0.909 | 0.01 | 0.04 | 0.44 | — |
| structural_virality | 1.000 | 0.970 | 0.01 | 0.02 | 0.44 | — |

**Cliff's delta saturates under systematic bias, so it is the wrong progress
metric.** `self_bertscore` shows this exactly: its delta rose 0.64 to 0.82 while
the per-thread magnitude improved on 7 of 10 threads. On thread 38jlgz the gap
went from -0.0156 to +0.0015 — far smaller, but it flipped to positive, and once
every generated thread sits on the same side of real, delta approaches 1
regardless of how small the gaps are. Use delta to predict whether a test will
pass at a given N; use the mean absolute gap or Wasserstein to track progress.

Attributing the three changes:
- **Bounded blackboard: keep.** `semantic_mean_cosine` delta 0.28 to 0.08,
  `emotion_entropy` 0.65 to 0.53, `self_bertscore` magnitude better on 7/10.
- **L1, beat divisor 35 to 21: revert or soften.** It raised the long-slot length
  ratio to 0.99 but `length_cv` got worse on both large threads and on seed 78,
  because lengthening mid-size comments compresses the spread the metric measures.
- **E1, equipment plus first-person licensing: gate it.**
  `mean_story_probability` delta went 0.06 to 0.26 with gaps up to +0.19, worst on
  the large threads. Offering own-gear anecdotes on `no_story` slots produces
  content StorySeeker scores as narrative. The offer should require
  `story_mode != no_story` or `evidence_mode == firsthand_experience` rather than
  the bare `allow_first_person_frame` flag that v64 set true for polite slots.

`hard_disagree_rate` degraded across v65-v67 (0.070, then delta 0.38 to 0.61) and
has no attributed cause yet.
