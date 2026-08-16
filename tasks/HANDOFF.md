# Handoff — synthetic Reddit thread generation (generalized_card)

## 2026-08-17 v81 current-state addendum

This addendum is authoritative for the current implementation. Historical
implementations and old TODO hypotheses are **not design evidence**: use the
current active path, scorer implementations, current run artifacts, and fresh
offline checks. Git commits and each run's reproducibility/source snapshot are
the mechanism for reproducing an old version; do not keep dead runtime branches
or contradictory prompt rules merely to preserve an old arm.

v81 fixes the root defects found in the v80 large-thread run:

- story/no-story is now a joint semantic/evidence contract before the Writer;
- direct replies plan with fixed story, tone, affect, and opener controls in
  their JSON contract instead of receiving those labels after semantic design;
- copied short-slot `development_plan` schema prose is removed;
- deterministic normalization no longer rewrites bad plans into the same
  gratitude move or `soft_helpful` payload;
- tone, affect, and story marginals are assigned jointly for feasibility;
- focused, low-info, and full Writer prompts no longer repeat static metric or
  tone blocks, and neutral affect no longer conflicts with impolite tone;
- metric-guided Writer retries, candidate ranking, blocking repetition guard,
  dead CLI flags, and their old tests were deleted. Distribution metrics are
  diagnostics; only non-persistable output has bounded recovery.

Verification: complete `generalized_card/tests` suite **259 passed**; backend
self-test passed for `camera_product`; 72 source pins have 0 missing and 0
drifted entries. Exact metric definitions, v80 evidence, and the implementation
audit are in `tasks/v81-worklog.md`. Formal success still requires a new
multi-thread matched evaluation; n=1 is only the content/contract diagnostic.

Written 2026-08-16. **This supersedes the 2026-08-15 handoff**, which is preserved
at `tasks/HANDOFF-20260815.md`. Several of its load-bearing claims were measured
this session and turned out to be wrong; §6 lists every correction.

Read in this order:

1. this file, end to end
2. `tasks/todo.md` — the re-prioritised plan
3. `tasks/lessons.md` — 8 recorded mistakes, 3 of them from this session
4. `tasks/generator_audit.md` — the older evidence base, with §6 here as its errata

Every number here is measured from run artifacts. Where a claim is uncertain, or
where the measurement cannot separate a cause, it says so.

---

# 1. THE GOAL

Generate synthetic Reddit threads that are statistically indistinguishable from
real ones across **12 thread-level metrics**, using `generalized_card/`, a
domain-configured Planner–Writer implementation of CARD.

## What "real" means here — the user's own framing, authoritative

> 我们要模仿的是说话方式，而不是真正的 content …… 只要能做好就行，因为我这个指标
> 其实就是衡量人们是怎么样说话、怎么样讨论的。

Not factual accuracy. **How people talk and discuss.** The user decomposed this
into four dimensions, and they map onto the metrics:

| user's dimension | metrics | state |
|---|---|---|
| 1. semantic is dispersed | `semantic_mean_cosine` | passes at n=10, but by cancellation (§4.3) |
| 2. low lexical overlap | `self_bleu_4`, `self_bertscore_mean_f1` | bertscore has never passed in any version |
| 3. stories told in first person | `mean_story_probability` | passes at n=10 by cancellation; per-thread 1.5–2.4× too high |
| 4. tone and emotion are varied | `emotion_entropy`, `polite_rate`, `impolite_rate`, `neutral_rate`, `hard_disagree_rate` | all fail, all in the same direction |

**Dimension 4 is the largest and most coherent failure.** See §5.

## The judging standard

- A metric is matched only if **MWU p > 0.05 AND KS p > 0.05, and the p-value is
  comfortably large.** Barely above 0.05 does not count.
- The user rejected N-based extrapolation: 不希望用根据 N 的大小来测试的方法，除非
  publicly scientifically proved. Do not argue "this would pass at N=150".
- Final target: **150 threads per domain.**
- **Print findings in chat, not only in MD files.**
- The user runs the paid commands: 修改完之后，我来负责测试。

## Standing constraints

- **Domain-generalised, never domain-specific.** This bit me this session: a rule
  I shipped said "your own gear … what you shot or set it to", which is camera
  vocabulary. Every test ran on the camera domain so nothing caught it.
  `test_the_named_rule_carries_no_domain_vocabulary` now asserts against it.
- **Every behaviour change gets an ablation flag**, `off` reproduces the previous
  release byte-for-byte, and the flag is recorded in `run_config.json`.
- **Style exemplars must be modified real comments**, never verbatim. Not built.
- Politeness *as a topic* is de-prioritised, but `polite_rate` / `impolite_rate` /
  `emotion_entropy` are dimension 4 and are the biggest gap. The distinction that
  matters: the work is **"make the assigned register actually appear"**, not
  "make the comments polite".

---

# 2. WHERE THE CODE IS

Branch `generator/v75-writer-realizes-planner-move`. The v80 work starts from
`4633af7`; use `git log` for the resulting implementation commit.

```
generalized_card/generalized_card/
  backend.py              2.6k lines  adapter and current Planner/Writer lifecycle
  prompts.py              2.7k lines  root/reply Planner + focused/full/low-info Writer
  writer_quality.py        270 lines  diagnostics + hard-output recovery only
  writer_grounding.py      323 lines  fact/grounding rules in one place
  speaker_roster.py        230 lines  NEW this session: who is speaking, across their turns
  generation_distribution.py 576     TONE_DEFINITIONS, AFFECT_INSTRUCTIONS, social targets
  task_distribution.py     260 lines  which task fields survive the surface rebalancer
  core_contract.py         520 lines  72 pinned file hashes + policy versions
scripts/sampling_generator/
  run_sampled_reddit_generator.py 2.1k  the CARD facade the adapter patches
  engine/*.py                            12 modules, all pinned
generalized_card/scripts/
  run_generate.py         1.4k lines  CLI, config record, subprocess env
  run_evaluate.py                     clean → score → matched-evaluate
  repin_core_contract.py   NEW this session: walks the whole CORE_FILES table
```

**All of `generalized_card/generalized_card/*.py`, `generalized_card/scripts/
run_generate.py` and every `scripts/sampling_generator/**` file is hash-pinned.**
A change anywhere means re-pinning. Use the new script; do not hand-edit hashes:

```bash
python3 generalized_card/scripts/repin_core_contract.py          # report drift
python3 generalized_card/scripts/repin_core_contract.py --write  # re-pin
```

It exists because hand re-pinning means listing the files you *remember*
changing. That is the same shape as the config-diff script that silently skipped
`plan_quality.repair_rounds` and cost a confounded run.

---

# 3. HOW EACH METRIC IS ACTUALLY MEASURED

The user's instruction: 对于每一个 metrics，你都必须清楚它们到底是怎么衡量的才可以.
This section exists because a previous claim ("the story allocation is correct,
do not change it") was wrong precisely from not reading the scorer.

| metric | scorer | what it computes |
|---|---|---|
| `self_bleu_4` | `score_thread_self_bleu.py` | pure n-gram, **no model**. Runs in seconds offline. |
| `self_bertscore_mean_f1` | `score_thread_self_bertscore.py` | BERTScore F1 between comment pairs in a thread |
| `semantic_mean_cosine` | `score_thread_semantic_uniformity.py` | mean pairwise cosine of sentence embeddings |
| `hard_disagree_rate` | `score_thread_disagreement.py` | share of parent→child pairs classed as hard disagreement |
| `polite_rate`, `impolite_rate`, `neutral_rate` | `score_thread_politeness.py` | **Intel/polite-guard, 4-way single-label.** `pred_label = argmax` over {polite, somewhat_polite, neutral, impolite}; each rate is the share of comments with that label. `somewhat_polite` is a real class that absorbs mass but is **not reported** as a metric. |
| `length_cv` | `score_thread_structure.py` | per-thread word-count coefficient of variation |
| `avg_depth`, `structural_virality` | `score_thread_structure.py` | tree shape only — determined by the matched sampler, not by generation |
| `mean_story_probability` | `score_thread_storyseeker.py` | **mariaantoniak/storyseeker**, RoBERTa. P(story) per comment, then the **mean over every comment in the thread**. LABEL_1 = story. Not the same as `story_rate`, which thresholds at 0.5. |
| `emotion_entropy` | `score_thread_go_emotions.py` | **SamLowe/roberta-base-go_emotions**, 28 labels, sigmoid, threshold 0.5. Each comment's `dominant_emotion = argmax`. The metric is the **Shannon entropy of the histogram of dominant emotions** across the thread. To raise it you need more distinct dominant emotions, spread more evenly. |

Consequences worth holding onto:

- **`self_bleu_4` is free to compute.** Never approximate it. This session I wrote
  my own approximation, it disagreed with the real scorer by an order of
  magnitude in effect size, and I reported a win that did not exist (§6.4).
- **`emotion_entropy` is about the *variety of argmax labels*, not about
  intensity.** v79 has 13 distinct dominant emotions but `neutral` takes 48.4%
  and `approval` 20.1%. Flattening that histogram is the lever.
- **`mean_story_probability` averages over *all* comments**, so it moves when
  non-story comments start sounding narrative, not only when story slots change.

---

# 4. CURRENT STATE

## 4.1 Runs

| tag | flags that differ | comments | cost |
|---|---|---|---|
| `…v75_ownwords_20260815_v2` | 10 threads, **`plan_quality.repair_rounds=3`** (unintended) | 522 | $5.99 |
| `…v76a_baseline_seed8_20260815_v1` | seed 8 only, repairs=0, all new flags off | 186 | $0.76 |
| `…v76b_ownfacts_seed8_20260815_v1` | `--own-fact-license own` | 186 | $0.77 |
| `…v77_repguard_seed8_20260816_v1` | `--repetition-guard blocking`, retry-limit 1 | **172** | $0.86 |
| `…v78_frameguard_seed8_20260816_v1` | + whole-comment frame check, retry-limit 2 | **182** | $0.88 |
| `…v79_nodrop_seed8_20260816_v1` | + style-residual retention | **186** | $0.90 |

All five seed-8 runs have `repair_rounds=0`, so **the v75 confound is resolved**
for seed 8. It is still unresolved at 10 threads; that only matters if you want a
clean 10-thread baseline.

Seed 8 is the largest thread in the pool (185 real comments). The pool's seed
range is `[start-seed-index, start-seed-index + max-posts)`, so seed 8 alone is
`--start-seed-index 8 --max-posts 1`.

## 4.2 The five seed-8 runs against real seed 8

Real values come from `…v75…/matched_evaluation/matched_real_thread_scores.csv`.

```
metric                     REAL     v76a     v76b      v77      v78      v79
self_bleu_4              0.0283   0.0377   0.0354   0.0389   0.0354   0.0375
self_bertscore_mean_f1   0.4887   0.5241   0.5192   0.5171   0.5181   0.5227
semantic_mean_cosine     0.1865   0.2110   0.2183   0.1954   0.2009   0.2267
hard_disagree_rate       0.1697   0.3279   0.2967   0.2982   0.2809   0.3516
polite_rate              0.2324   0.0870   0.0710   0.0698   0.0889   0.0543
impolite_rate            0.4649   0.6739   0.7158   0.7151   0.6556   0.6957
neutral_rate             0.1622   0.0978   0.0820   0.1163   0.1389   0.0978
length_cv                0.8951   0.8639   0.8304   0.9147   0.8746   0.8593
avg_depth                3.6000   3.5978   3.5683   3.4128   3.5111   3.5978
structural_virality      4.5608   4.5663   4.5153   4.3326   4.4887   4.5663
mean_story_probability   0.1114   0.1944   0.2610   0.1935   0.1633   0.2152
emotion_entropy          1.9459   1.5443   1.4402   1.4986   1.4591   1.6572
```

**These five runs cannot be ranked.** Any two runs differ in ~99% of planner
fields (branch_goal 100%, semantic_move 99%), and there is **no
same-config-twice run to estimate that noise**. Mean relative error is 25.0%
(v78) to 38.8% (v76b) and that spread may be entirely noise.

What *is* readable is the **sign, which is identical in all five runs**:

- `polite_rate` 3–4× too low
- `impolite_rate` ~1.5× too high
- `hard_disagree_rate` ~1.8× too high
- `emotion_entropy` too low, i.e. too concentrated
- `mean_story_probability` 1.5–2.4× too high
- `self_bertscore` +0.03, every run, every thread

`length_cv`, `avg_depth` and `structural_virality` are within ~5% everywhere.

## 4.3 The most important structural finding: passing ≠ matched

MWU and KS are **unpaired** tests over 10 thread values. A metric can pass while
every individual thread is wrong, provided the errors cancel. Measured on v75:

```
metric                    v75    mean |gen-real|/real   threads within 20%   reading
avg_depth                PASS            0.5%                10/10           matched per thread
structural_virality      PASS            0.8%                10/10           matched per thread
semantic_mean_cosine     PASS           19.4%                 5/10           PASSES BY CANCELLATION
mean_story_probability   PASS           46.4%                 3/10           PASSES BY CANCELLATION
self_bertscore           FAIL            6.9%                10/10           close per thread, fails on consistent sign
length_cv                PART           11.7%                 8/10           close per thread
hard_disagree_rate       PART          188.9%                 0/10           wrong per thread
impolite_rate            FAIL           70.4%                 0/10           wrong per thread
polite_rate              FAIL           65.2%                 1/10           wrong per thread
neutral_rate             PART           63.4%                 0/10           wrong per thread
emotion_entropy          FAIL           39.2%                 3/10           wrong per thread
self_bleu_4              FAIL           34.9%                 2/10           wrong per thread
```

Two conclusions:

1. **Of the four metrics that "pass", only two are real**, and both are
   sampler-determined tree shape. **Zero metrics are currently won by generation
   quality.** Any claim of the form "we pass 4 of 12" should be read this way.
2. **`self_bertscore` is not a large error — it is a small, perfectly consistent
   one.** 6.9% mean error, 10/10 threads inside ±20%, and it fails only because
   all ten overshoot by ~+0.03. That is the signature of **one global constant
   offset**, not of content. This reframes the metric that has never passed.

---

# 5. THE DIAGNOSIS THAT SHOULD DRIVE THE NEXT CHANGE

## 5.1 Dimension 4 is one root cause, not five failures

Measured on v79, joining each slot's assigned `tone_target` to polite-guard's
`pred_label` (184 aligned slots):

```
assigned          n    realized impolite    realized as assigned
impolite         90         93%                    93%
polite           46         59%                    13%
neutral          35         34%                    34%
somewhat_polite  13         38%                    54%
                                overall realization 59.2%
```

The Writer can produce `impolite` and essentially nothing else. Assigned-polite
slots collapse into impolite 59% of the time. **One register per thread explains
polite ↓, impolite ↑, hard_disagree ↑, and emotion_entropy ↓ simultaneously.**

## 5.2 Two candidate causes measured and eliminated this session

**Not length.** `TONE_SCOPE_HINTS` at `generation_distribution.py:508-518`
records that polite-guard's polite class is length-driven in real data (52% of
60–120 word comments, 64% above 120). It does not transfer:

```
                 60–120 words     120+ words
real polite          52%             64%
v79 polite            6%              0%
```

Generated long comments are 73–88% impolite. **Giving polite slots more length
will not work**; that item can be struck from the plan.

**Not insufficient agreement.** Real comments carry *more* negation than
generated (41.5% vs 31.2% on seed 8) and are still scored polite. What differs:

```
surface                       REAL     v79    ratio
warm / appreciation marker   14.0%   11.8%    0.84
emotional endpoint            2.5%    1.1%    0.43
hedge                        18.0%   12.9%    0.72
decision-framing noun         0.5%    4.3%    8.60
```

`TONE_DEFINITIONS["polite"]` (`generation_distribution.py:480-489`) currently
**forbids two of the three surfaces the real data uses**:

> "Do not hedge the positive judgement into a maybe, and do not
> use customer-service phrasing or a template thank-you."

and the 8.6× on decision-framing nouns says the Writer substitutes analysis for
feeling. The comment block above `TONE_DEFINITIONS` documents the reasoning for
the hedge ban: a softener reading was predicted to collapse into
`somewhat_polite`. The measured collapse is into **impolite**. The prediction was
wrong, so the rule should go.

## 5.3 `self_bertscore`: one global signature, and a concrete candidate

The offset is uniform (§4.3), so look for something identical across every
generated comment. Typography is the strongest candidate available:

```
of comments containing any apostrophe:
   real   17.6% use only curly ’
   v78   100.0% use only curly ’
overall curly-typography rate:  real 11–13%   generated 72–74%
straight apostrophe inside a word: real 51%   generated 0%
```

**Every generated comment carries the same typographic fingerprint.** This is
model-emitted, not from `gpt_cleanup` (verified in an earlier session: identical
pre/post). A deterministic post-step fixes it with zero API cost and can be
verified offline over the whole corpus.

This item currently sits in P6, rated lowest priority. **That ranking is wrong**
— it is the only mechanism on the board whose shape matches how `self_bertscore`
actually fails.

## 5.4 `mean_story_probability` is too HIGH, not too low

Corrected from the previous handoff. Real per-thread `story_rate` ranges 0.000
(seeds 0, 3, 5) to 0.275 (seed 6), mean 0.110. Generated overshoots on seed 8 by
1.5–2.4×. Separately measured: of 559 real comments across the ten threads, 32
narrate a personal experience (5.7%), and **32 of 32 are first person**.

So the user's dimension 3 is right in form — when a real commenter tells a
story it is first person, always — but stories are a **minority** of real
comments, and the generator currently tells too many of them.

---

# 6. ERRATA — claims in the previous handoff that measurement refuted

Load-bearing corrections. Do not re-derive these.

**6.1 "P0's first item: add the 6 missing metrics to the Writer's distribution
target (`run_generate.py:488`)" — the location is a record, not a wire.**
`run_generate.py:523` writes a `metrics` list into `run_config.json` for the
reader. The real target is hard-coded in
`generation_diversity.build_thread_distribution_target:40-43`. It feeds
`joint_target_distance`, which `writer_quality.py` uses only to **rank
candidates**. Under `--writer-retries 0` there is one candidate, so the list has
no effect at all. Five of the six missing metrics also need transformer
classifiers inside the generation process; only `length_cv` is free.

**6.2 "`LENGTH_BUCKET_BOUNDS["very_long"] = (120, 220)` caps `length_cv`" — dead
at runtime.** It is read only by `backend.py:2426` inside
`_retry_note_for_problems`, i.e. on a retry. With `--writer-retries 0` retries
were ~never taken. Every prompt gets `soft_length_guidance` instead. The
generator audit's own "confirmed dead" section was right and the later section
was wrong. `length_cv` is within 3.5% of real anyway.

**6.3 "The story allocation is correct, do not change it."** Wrong. See §5.4.

**6.4 "The plan-echo route lock and the frame guard closed most of the
`self_bleu` gap."** Wrong, and the error was mine twice over. I wrote my own
self-BLEU approximation (max overlap against any other comment) instead of
running `score_thread_self_bleu.py`, which is free. The real metric moved
0.03775 → 0.03750 across the whole frame intervention: **nothing**. The frame
family "that's the part" did drop from 8.1% to 2.7% of comments, so the text
defect is real — but it and `self_bleu_4` are close to independent.

**6.5 "`allocate_story_and_affect` is a no-op auditor (B7)."** It is a
*deliberate* auditor. `generation_distribution.py:108-114` documents that a
post-Planner allocator used to force incompatible affects onto coherent plans and
was removed on purpose. Not a bug.

**6.6 "P4a: license the speaker's own kit and history."** Shipped as
`--own-fact-license own`, measured in v76b, **refuted**: specification tokens
0.05 → 0.02 per comment against a real 0.54, and 0.083 → 0.024 on the licensed
slots themselves. Two reasons, both measurable in advance: 78 of 114
spec-carrying real comments (68%) carry no first-person frame, so the gate
selected the wrong slots; and replacing a vague blanket ban with an explicit
"about the product under discussion, name only what is visible above" made the
binding constraint **sharper** on exactly the detail real comments are full of.

**6.7 "Concreteness means specifications."** Thread-dependent, so not
generalisable: spec-carrying comments are 0% of seed 1 and 64% of seed 5. What
holds on **all ten** threads is quantities (real 12.3× generated) and proper
nouns (real 1.85×). Any concreteness rule must be phrased that way.

---

# 7. WHAT THIS SESSION CHANGED IN THE CODE

One commit: `e9a9fbe`, on top of `67e4e9b`. 14 files, +2032/−75. 266 tests pass,
backend self-test passes, contract drift none.

## 7.1 New modules

**`generalized_card/generalized_card/writer_grounding.py`** — the fact/grounding
rules, which were previously smeared across eight places that disagreed with each
other:

```
prompts.py:1341   focused writer      "Name a product, model, or number only if visible above."
prompts.py:2756+  _story_fact_safety_rule, three branches
prompts.py:1503   system prompt       "Do not invent facts, specifications, numbers…"
prompts.py:113    _own_equipment_block "…do not invent a specification, price, measurement…"
prompts.py:2786   _metric_guidance_block (low-info path only)
prompts.py:~1434  low-info hard rules
prompts.py:1517   mask_specifics       $900 → [amount], 3+ digits → [number]
engine/vocabulary.py:226 (pinned core)  "…product details unless they are visible in the prompt"
```

Measured over the 522 v75 slots before the extraction: 443 slots (84.9%) carried
the blanket ban, 249 (47.7%) carried an "Equipment you may claim as your own"
permission, and **170 (32.6%) carried both — a permission and its revocation in
the same prompt.** All 249 equipment blocks closed by forbidding any
specification about that equipment.

Three modes, all reproducible:
- `off` — v75 verbatim. Verified byte-identical: fingerprint sha256
  `7257a066cf9fc05f80862d0d89ae54d597ea550777fc99baf2cbb96e4a9c32ca` over all 522
  slots, before and after the refactor.
- `own` — the refuted personal-history license. Kept as an arm, not a
  recommendation.
- `named` — the correction. Domain-neutral wording, gated on `substantive_slot`
  (≥25 real words and not micro/short) rather than on a first-person frame.
  **Never run.**

**`generalized_card/generalized_card/speaker_roster.py`** — a thread has people,
not slots. `run_sampled_reddit_generator.py:1408` built the author name as a pure
function of the slot index, so a 186-comment thread was 186 people who each spoke
once. The matched real threads are 559 comments from 265 named authors (2.11
each), with 68% of comment mass written by someone who spoke more than once; seed
8's busiest author wrote 10.

The structure is a **join, not a new sampling policy**: `real_sample_id` already
binds each slot to one matched real comment, and that comment has an author.
`selected_matched_comments` is deterministic (no rng), so the adapter recomputes
the same list. Verified on seed 8: `real_word_count` agrees for 186 of 186 slots,
and the rebuilt roster reproduces 80 named speakers at 2.11 comments each,
busiest 10 — identical to counting the raw jsonl directly.

Leakage: the real author string is used **only** as a grouping key. It is never
stored on a `Speaker`, never rendered, never written to an artifact. A test
asserts this. Equipment is keyed by speaker instead of slot index (previously the
same participant got a different kit in each of their turns), and
`_speaker_identity_block` shows a speaker their own earlier comments in this
thread. **Never run.**

**`generalized_card/scripts/repin_core_contract.py`** — see §2.

## 7.2 Flags added

All default to the previous behaviour, all recorded in `run_config.json`, all
plumbed env var → `backend.py` module attr → CLI flag.

| flag | values | state |
|---|---|---|
| `--own-fact-license` | `off` \| `own` \| `named` | `own` refuted; **`named` never run** |
| `--speaker-identity` | `off` \| `matched` | **never run** |
| `--repetition-guard` | `off` \| `blocking` | run in v77/78/79; works mechanically, no metric effect |

## 7.3 The repetition guard, and what it taught

`writer_quality.py` gained `REPETITION_DIAGNOSTIC_PROBLEMS` (promoting
`template_phrase_reused`, `opener_family_reused`, `opening_reused` out of the
advisory set under the guard) and a new whole-comment `repeated_frame:` check.

Why the new check exists: the core's `template_phrase_signature`
(`engine/writer_validation.py:154-178`) reads only `tokens[:28]`. Of the 15
comments in v76a carrying the "that's the part" family it saw **4**; the rest sat
at token 20, 52, 62, 80. The new check reads the whole comment and normalises
curly apostrophes; verified on the actual run output it catches 15 of 15. It is
kept separate from the core signature because that signature also decides
`first_person_frame_unwanted` and `uncertainty_frame_unwanted`, which genuinely
are about how a comment opens.

Result: frame 8.1% → 2.7% of comments (real 0.0%), top shared 4-gram 6.5% → 2.7%
(real 1.5%), novel entities 27 → 45 (real 96). `self_bleu_4` unchanged (§6.4).

## 7.4 Two comment-loss bugs fixed

Both cost real comments, and a shortened thread also damages `avg_depth` and
`structural_virality`, two of the four metrics that "pass".

1. **v77 lost 14 of 186.** Promoting the repetition codes made them
   non-distribution failures, so `consider_distribution_candidate`
   (`backend.py:2160-2179`) returned early and `best_distribution_candidate`
   stayed `None`; exhaustion then returned `skip: True`. The previous handoff
   listed *two* drop paths; I verified `REPAIRABLE_WRITER_PROBLEMS` and forgot
   `backend.py:2205`.
2. **v78 still lost 4.** Their residue was `missing_concrete_anchor` or
   `question_mark_unwanted` — both advisory, both silently accepted on attempt 1.
   Rejecting on attempt 5 what attempt 1 would have kept is not a stricter
   policy, only an inconsistent one.

Fix: `only_style_problems()` plus an `accepted_style_residual_after_repair` path
that retains the best candidate when every residual problem is one the run would
have tolerated at first pass. Tests assert it never fires over `exact_duplicate`,
`empty`, `parent_copy` or `placeholder_literal`. v79: **186 of 186**, and
`rejected_distribution_repair_exhausted` no longer appears.

## 7.5 Other edits

- `engine/model.py` (pinned): `CommentTask.speaker_id`, default `""`.
- `run_sampled_reddit_generator.py` (pinned): author name uses `speaker_id` when
  present, otherwise the old slot-indexed name; `speaker_id` carried onto the
  comment dict.
- `task_distribution.py`: `speaker_id` added to `PLANNER_AND_SLOT_INVARIANTS` so
  the surface rebalancer cannot rewrite it — the same omission that lost
  `semantic_move` in 347 of 347 reply slots.
- `core_contract.py`: `writer_grounding` and `speaker_roster` are now pinned and
  verified by `run_generate.py`. Policy version is
  `generalized-card-v2-own-fact-license-v76-20260815`; v74 and v75 are in
  `HISTORICAL_GENERATION_POLICY_VERSIONS`.
- ~60 new tests across `test_generalized_card.py` and
  `test_planner_field_survival.py`.

## 7.6 Known debt

**The policy version was not bumped after v76.** v77, v78 and v79 have different
Writer behaviour but the same policy string. The flags are in each
`run_config.json` so the behaviour is recoverable, but the next change should
bump the version and move `…own-fact-license-v76-20260815` into
`HISTORICAL_GENERATION_POLICY_VERSIONS`, or those three runs become
un-evaluatable. `run_evaluate.py` passes `allow_historical=True`, so evaluation
of existing runs is safe either way; `--resume` on generation is not.

## 7.7 v80 continuation — implementation complete, paid run pending

The active generation path, all policy modules, all twelve scorers, the
evaluation aggregator, and the existing v79 records were read end to end before
this change. The new policy string is
`generalized-card-v2-planner-contract-coherence-v80-20260816`.

Free replay over v79's 186 Planner records found **30 contradictions that the old
quality gate accepted**: 21 `no_story + personal_story` plans and 9 `polite`
plans attached to incompatible roles/functions. The StorySeeker join independently
showed that planned story slots contributed only about 25% of total story
probability; 25 of 167 `no_story` comments were still classified as stories.

Implemented:

- validate the whole story/tone semantic contract after the frozen schedule is
  applied, before Writer execution;
- render an explicit non-narrative rule for every `no_story` Writer path;
- replace the refuted polite length/anti-hedge theory with measured social cues;
- show direct-reply planners sibling delta/novelty coverage;
- record every behavioral field in resume/extension/upgrade comparisons;
- add `--social-contract-coherence` and `--reply-sibling-visibility`; `off`
  preserves the pre-v80 arms and both values are recorded in `run_config.json`;
- remove only functions proven unreferenced by repository-wide reference and AST
  audits. Active reviser helpers remain; the user-owned evaluation/cleanup files
  outside `generalized_card/` were not touched.

The curly-apostrophe counterfactual was also run for free on 40 comments / 780
pairs. Self-BERTScore moved 0.52947 -> 0.52381, toward real but far short of the
full ~0.034 gap. Do not ship a held-out-test-calibrated typography transform;
this is a secondary hypothesis requiring evaluation-excluded calibration.

Verification before any paid generation: 270 full generalized tests passed,
the backend self-test passed, CLI help renders, Ruff passed, all 72 core pins
matched, and scoped `git diff --check` passed. No claim is made that a metric or
p-value improved; v80 has not yet generated new text.

---

# 8. THE PLAN

Full detail in `tasks/todo.md`. Re-ordered by **which measured gap it moves**,
not by where a code defect happens to be. The old P0–P6 numbering is retained in
`tasks/todo.md` for traceability, with each item marked kept / struck / demoted.

## A — realize the assigned register  ← the largest gap, dimension 4

Targets `polite_rate`, `impolite_rate`, `neutral_rate`, `emotion_entropy`,
`hard_disagree_rate` together, because §5.1 shows they are one failure.

- [ ] Remove the hedge and thank-you prohibitions from
      `TONE_DEFINITIONS["polite"]` (`generation_distribution.py:480-489`). The
      comment above the table records the prediction that justified them; §5.2
      shows the prediction was wrong.
- [ ] License the emotional endpoint explicitly (real 2.5% vs generated 1.1%).
- [ ] Suppress decision-framing nouns in the Writer's own rules (real 0.5% vs
      generated 4.3%, an 8.6× overshoot).
- [ ] Do **not** add length to polite slots — measured and eliminated in §5.2.
- [ ] Offline acceptance check before any paid run: re-render the v79 prompts and
      confirm the banned surfaces are gone. Then a seed-8 run, judged on
      **tone realization rate (59.2% baseline) and emotion_entropy**, not on
      p-values.
- [ ] Ablation flag, `off` byte-identical.

## B — the global typographic signature  ← the only lever aimed at `self_bertscore`

- [ ] Deterministic post-generation normalisation: curly → straight quotes and
      apostrophes at a rate matched to the domain's real corpus, rather than
      100% curly on every comment. Real: 51% of comments carry a straight
      apostrophe inside a word, generated 0%.
- [ ] Free, offline-verifiable over the whole corpus, no prompt work, no API.
- [ ] Then re-score `self_bertscore` on an existing run — the scorer does not
      need regeneration, only re-cleaning. **This makes the hypothesis testable
      for $0.**

## C — bring `mean_story_probability` down

- [ ] Generated overshoots 1.5–2.4× on seed 8. Real per-thread `story_rate` is
      0.000 on three of ten threads. The per-thread target already scales from
      the matched template (`generation_distribution.py:129-134`), so check
      whether the overshoot is allocation or realization before changing
      allocation — non-story comments sounding narrative would also do it, since
      the metric averages over every comment (§3).

## D — the two built-but-never-run arms

- [ ] `--own-fact-license named` — targets quantities and proper nouns, the two
      concreteness signals that hold on all ten threads.
- [ ] `--speaker-identity matched` — targets `self_bertscore` via voice
      variation. Note §5.3 may explain that metric more cheaply; run B first.

## E — reply-planner sibling visibility (was P3)

- [ ] Every depth ≥ 1 batch takes `render_direct_reply_planner_prompt`, which
      renders no prior-plan ledger, no coverage summary, no sibling contract.
      Verified on seed 2: depths 3–8 are single-slot batches and tasks 38–45 are
      nine near-duplicate moves. Plausibly feeds dimension 1 and 4.

## Struck

- **Plan-echo validator (old P1).** Echo is at 0.0% and the route lock that fixed
  it moved no metric. Nothing to guard.
- **Length for polite slots (part of old P5).** Eliminated in §5.2.
- **`LENGTH_BUCKET_BOUNDS` 220-word ceiling (part of old P6).** Dead at runtime,
  §6.2.
- **`--own-fact-license own`.** Refuted, §6.6. Kept only as a reproducible arm.

## Rejected: two writers supervising each other

The user asked. Recommendation stands: **no LLM critique loop.** Cost doubles and
critique-driven revision pushes text toward the balanced, hedged register that is
already the failure mode. The stronger argument: the system has ~20 validators and
ignores most of them. The useful form of supervision is a deterministic
discriminator, which is what §7.3's frame check is.

---

# 9. OPERATING RULES

## Before diagnosing

- **Read every file on the active path end to end.** Not grep hits. In this
  codebase that means the CLI, the backend adapter, **every** prompt builder
  (root planner / direct-reply planner / focused writer / low-info writer — there
  is more than one per role, and their schemas have contradicted each other
  twice), the generator facade, the engine modules, and the policy modules.
- **Read the scorer before theorising about a metric.** §3 exists because two
  wrong conclusions came from not doing this.
- **Never approximate a metric that is cheap to compute.** `self_bleu_4` needs no
  model and runs in seconds.
- Subagent reports and prior handoffs can be wrong. Two of three full-file reads
  in an earlier session contained a materially wrong claim, and §6 lists six more
  from the last handoff. Re-verify load-bearing claims against run artifacts.

## Before changing

- **One mechanism per API run**, and predict the magnitude first so a null result
  is interpretable. Write the prediction down.
- **Do not use old implementations as design authority.** Reproduce them from
  git or the run's source snapshot when needed; delete disproven, unreachable
  controllers from the current path.
- **Distribution diagnostics never select a Writer candidate.** Only output
  that cannot be persisted may receive bounded hard recovery.
- **Apply the change to every path.** v74 converted only the focused writer and
  left 106 of 522 slots on the old prompt, which made that release
  unattributable.
- **No domain vocabulary in Writer-facing rule text.** Every test runs on camera,
  so nothing else will catch it.

## Before handing the user a command

- **Dry-run on a throwaway tag with `--prepare-only`**, then delete the tag. This
  caught two real errors this session: a missing `choices` entry, and the fact
  that the adapter files are pinned under different registry names than I assumed.
- **Separately verify what `--prepare-only` skips** — it returns before the
  API-key check.
- **Re-pin with the script, then confirm the drift list is exactly the files you
  edited.**
- Run `PYTHONPATH=generalized_card .venv/bin/python -m pytest -q generalized_card/tests`
  (**259 pass** at v81) and the camera-product backend self-test.

## Interpreting a single-thread run

- n=1 gives no p-value and a degenerate Cliff's delta. **Only the relative error
  against the matched real thread is readable.**
- Two runs of the same config have never been made, so **run-to-run noise is
  unknown** and any single-thread ranking of two configs is unsupported. If a
  future decision depends on ranking, spend $0.9 on a repeat of an existing
  config first.
- Content diagnostics computed directly on the text (frame rate, comment count,
  realization rate) are far more trustworthy at n=1 than metric deltas.

---

# 10. COMMANDS AND PATHS

## Environment

API keys live in `third_party/MiroFish/.env` as **`LLM_API_KEY`**;
`run_generate.py` loads that file itself but still pass `--api-key-env LLM_API_KEY`.

## Generation, seed 8 (~$2, ~45 min on the v80 observed request volume)

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag <NEW_TAG> \
  --domain camera --model gpt-5.4-mini \
  --base-url https://api.openai.com/v1 --api-key-env LLM_API_KEY \
  --pool-size 150 --max-posts 1 --posts-per-run 1 \
  --start-seed-index 8 --sampling-seed 42 \
  --context-dropout-rate 0.42 --context-jitter-rate 0.32 \
  --plan-quality-repairs 3 --writer-hard-recovery-rounds 2 \
  --domain-claim off --writer-prompt focused --writer-route-lock own_words \
  --social-contract-coherence on --reply-sibling-visibility on \
  --own-fact-license off --speaker-identity off \
  2>&1 | tee /tmp/<TAG>_gen.log
```

`--max-posts 10 --start-seed-index 0` gives the 10-thread evaluation set. Its
cost and runtime should be re-estimated from the n=1 v81 token summary before
spending. `plan-quality-repairs 3` gives joint contract conflicts
a bounded Planner repair budget. Hard Writer recovery handles only empty,
duplicate/copy, placeholder, or planner-skeleton output; it does not optimize a
metric.

The seed pool is derived, not a flag:
`artifacts/generalized_card/seed_pools/camera_product_150_seed42.json`.

## Evaluation — zero API cost, CPU only

```bash
python3 -u generalized_card/scripts/run_evaluate.py --tag <TAG> --device cpu
```

Cheap single metric, no model, seconds:

```bash
cd scripts/evaluation && python3 score_thread_self_bleu.py \
  <run>/generated/run_NN_sampled_reddit/discussion.json --output /tmp/sb.json
```

## Artifacts per run

```
artifacts/generalized_card/runs/<TAG>/
  run_config.json                        full config incl. every ablation flag
  generated/run_NN_sampled_reddit/
    generation_records.json              per slot: the 62 task fields, prompt, raw, comment
    discussion.json
  cleaned/run_NN_.../
    politeness_results.json              per comment pred_label  (no text — join by comment_id)
    go_emotions_results.json             per comment dominant_emotion
    storyseeker_results.json
  logs/writer_distribution_control.jsonl attempts, final_status, per-attempt problems
  evaluation/revised_generated_thread_scores.csv    the 12 metrics, per thread
  matched_evaluation/matched_real_thread_scores.csv real values for the matched seeds
```

Real ground truth: `data/raw/discussions/camera_product/<product>/<product>.comments.jsonl`,
filtered by `post_id == seed_pool.seed_posts[].source_raw_post_id`.

## Reconstructing tasks offline

`generation_records.json[].task` carries exactly the 62 `CommentTask` dataclass
fields, so `CommentTask(**record["task"])` round-trips. That is how every offline
prompt-rendering check in this session was built — no API, full corpus.

---

# 11. RECOMMENDED FIRST MOVE

The free checks are complete. Run the seed-8 command in §10 first, with social
contract on and sibling visibility off. Judge it on Planner repair counts,
tone-label realization (59.2% baseline), and StorySeeker mass in `no_story`
slots—not on an n=1 p-value. If the mechanism moves those diagnostics in the
predicted direction, run the comparable ten-seed arm before enabling sibling
visibility as a separate experiment.
