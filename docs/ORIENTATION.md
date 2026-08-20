# ORIENTATION — read this first

**Purpose.** This is the single entry point for anyone — human or a fresh agent
session — picking up the synthetic-Reddit-thread work. It answers four things:
what we are trying to do, how we are doing it, how to read the metrics, and how
to work on this codebase without wasting a paid run.

It is deliberately high-level and it is deliberately short. Every section ends
with a pointer to the file that holds the evidence. **This file states
conclusions; the linked files hold the measurements.**

Last verified: **2026-08-20**. See §9 for what "verified" means here and what
was actually checked to write this line.

---

## Your role

You are working on this as a **senior AI research scientist**, not as a coding
assistant taking tickets. The user's framing:

> 你是作为一个 senior AI research scientist 去做我现在这个东西的。所以你的思考要
> 全面、要够，你要去很好地理解。

Concretely, that means:

- **Own the problem, not the task.** The goal in §1 is the deliverable. A change
  that closes a code defect but moves no metric is not progress; three of four
  paid runs in one earlier stretch did exactly that.
- **Think all the way through before acting.** Comprehensively — the whole
  mechanism, the whole active code path, the alternative explanations, and the
  measurement that would distinguish them. Not the first plausible story.
- **Understand it deeply, not operationally.** Know what each metric's scorer
  actually computes (§3), why the mechanism you are proposing would move it, and
  what would happen if you were wrong.
- **Hold yourself to a research standard of evidence.** A hypothesis is not a
  finding. Try to kill your own hypothesis before you build on it (§4, step 3);
  measure on the real scorer; report effect sizes; write down what you rejected.
- **Be your own reviewer.** Self-check before, during and after — including
  before you start. Ask "would a staff engineer and a careful reviewer approve
  this?" and answer honestly.
- **Never be satisfied.** A metric that passes weakly has not passed (§2). A fix
  that was not shown to move what it claimed is not a fix.

The discipline in §7 is the operational form of this role. It is not
bureaucracy — every rule there has a wasted paid run behind it.

---

## 0. Reading order

| # | file | what it is | when to read |
|---|---|---|---|
| 1 | **this file** | goal, method, metrics, discipline | always, first |
| 2 | `tasks/todo.md` | the task list, ordered by which measured gap it moves | before choosing what to do |
| 3 | `tasks/v<N>-worklog.md` | the current version's full evidence, including rejected hypotheses | before touching that version's code |
| 3b | `tasks/v99-worklog.md` | the politeness diagnosis: four rejected hypotheses and the verified mechanism | before touching tone or register |
| 4 | `generalized_card/VERSION_LOG.md` | every released version, its arms, and its result | when comparing versions |
| 5 | `tasks/lessons.md` | 48 mistakes, each with the rule that prevents it | before diagnosing anything |
| 6 | `generalized_card/AGENTS.md` | binding engineering rules for `generalized_card/` | before writing code |
| 7 | `tasks/HANDOFF.md` | the long-form evidence archive, newest addendum first | when you need the detail behind a claim |
| 8 | `docs/thread_metric_score_reference.md` | every exported metric, its scorer, its model | when you need a scorer's exact semantics |
| 9 | `generalized_card/RUN_INDEX.md` | all runs with tag, cost, and outcome | when locating an artifact |
| 10 | `.claude/handoffs/` | session handoffs, newest last; each carries a TYPE INSTRUCTION block for the next agent | when picking the work up cold |

`tasks/HANDOFF.md` is **an archive, not a spec.** It grows by addendum and its
older numbered sections carry stale state and at least one claim that was later
retracted (see §6). Where this file and `HANDOFF.md` disagree, this file wins,
and the disagreement should be fixed in `HANDOFF.md` the same session.

---

## 1. The goal

Generate synthetic Reddit threads that are **statistically indistinguishable
from real ones across 12 thread-level metrics**, using `generalized_card/` — a
domain-configured Planner→Writer implementation of CARD — with **first-pass
generation only** (no reviser, no best-of-N, no resampling toward a metric).

### What "real" means here

The user's framing is authoritative. Verbatim:

> 我们要模仿的是说话方式，而不是真正的 content …… 我这个指标其实就是衡量人们是
> 怎么样说话、怎么样讨论的。

> 大家可能有问号、有句号、有提议、有反对，会讨论不同的话题。而不是说总是一些
> helpful 的，或者一些很不自然的、非人类的讨论样子。像 AI 生成的，很多时候可能就
> 是同一类话术，或者很容易去讨论同一个话题，又或者全都是很 helpful、thoughtful
> 这类的。

So: **not factual accuracy, and not topical similarity — the manner of
speaking.** The failure mode being designed against is the AI register: one
voice, one level of helpfulness, everything converging on one topic.

The user decomposed this into four dimensions, which map onto the metrics:

| dimension | metrics | v98 state |
|---|---|---|
| 1. semantic is dispersed | `semantic_mean_cosine` | PASS |
| 2. low lexical overlap | `self_bleu_4`, `self_bertscore_mean_f1` | bleu PASS (weak); **bertscore has never passed in any version** |
| 3. stories told in first person | `mean_story_probability` | PASS |
| 4. tone and emotion are varied | `emotion_entropy`, `polite_rate`, `impolite_rate`, `neutral_rate`, `hard_disagree_rate` | entropy + disagree PASS; **the politeness trio is the open failure** |

### The two acceptance criteria

The user restated the target in its final form:

> 只要是肉眼无法识别出 generated 和 real 并且 p value 大于 0.05 就可以，最终目的
> 是这个，无所谓 process 是什么。

1. **All 12 metrics at p > 0.05** (see §2 for the exact rule).
2. **A human reading the threads cannot tell generated from real.**

Criterion 2 is not measured by any of the 12 metrics, so it needs its own
checks. Known eye-visible tells still present in v98, each measured against the
matched real text: no generated comment contains a link (real 0.051), `check`
at ~10× its real rate, `will` at ~1% of its real rate, `their`/`we` absent,
10 distinct product designators against 40 real. These are logged in
`tasks/v98-worklog.md` and are not yet fixed.

**Process is explicitly not the goal.** Any mechanism that moves a metric
without leaking evaluation text is acceptable. The constraints in §7 exist
because they protect the *validity* of the measurement, not because they
prescribe a method.

### Final scale

**150 threads per domain.** Everything at N=1 or N=10 is a gate, not a result.

---

## 2. The judging standard

### The rule the evaluator applies

`generalized_card/scripts/run_evaluate.py:411` (`_metric_status`):

```
sample_size <= 1  →  DESCRIPTIVE   (no PASS/FAIL claim is made at all)
MWU p > 0.05 AND KS p > 0.05  →  PASS
exactly one of them > 0.05    →  PARTIAL
neither                       →  FAIL
```

The 12 metrics are the tuple `REQUIRED_THREAD_METRICS` at
`run_evaluate.py:28`. Both tests are **two-sided**: Mann–Whitney U on the
distribution shift and two-sample Kolmogorov–Smirnov on the shape. Each report
also carries **Cliff's delta** (generated − real; positive = generated is
higher) and the **Wasserstein distance**.

### How to actually read it — four traps

1. **A large p-value at N=10 is not evidence of a match.** The test is
   *unpaired* while the data is *paired by seed* — each generated thread has
   exactly one matched real thread. Unpaired tests on paired data are
   conservative, so N=10 p-values are **optimistic**. At |Cliff| = 0.25 a
   metric passes ~87% of the time at N=10 and ~4% at N=150.
2. **Therefore the real target is the effect size, not the p-value.**
   **|Cliff's delta| ≤ 0.10** is the number to steer by. `self_bleu_4` at
   MWU 0.121 with Cliff +0.42 is a *weak* pass that will not survive N=150.
3. **Barely above 0.05 does not count.** The user rejected N-based
   extrapolation ("this would pass at N=150") unless it is publicly,
   scientifically established.
4. **Multiplicity is an open decision, still owned by the user.** 12 metrics ×
   2 tests at α = 0.05 means even a *perfect* generator passes all 12
   simultaneously only ≈ 0.94¹² ≈ 52% of the time. Before the 150-thread run
   somebody has to choose: a multiplicity correction, or effect-size-led
   reporting with |Cliff| ≤ 0.10 as the bar. **Do not run N=150 before this is
   decided** — the result would not be interpretable either way.

### What passing does not mean

Two metrics — `avg_depth` and `structural_virality` — are determined by the
matched sampler (the reply tree is copied from the real thread's shape), not
won by generation. They pass structurally. Do not read them as evidence that
the generator is working.

A thread-level metric can also pass **by cancellation**: half the threads too
high, half too low, distributions overlapping. Always read the per-thread
column in `matched_evaluation/matched_generated_thread_scores.csv`, not only
the aggregate row.

---

## 3. The 12 metrics — what each one actually measures

**Rule: read the scorer before theorising about a metric.** This section exists
because two wrong conclusions in this project came from skipping that. Full
computational detail is in `docs/thread_metric_score_reference.md`; this table
is what you need to *reason* about a metric.

| metric | what the scorer computes | the lever that moves it | trap |
|---|---|---|---|
| `self_bleu_4` | Symmetric pairwise BLEU-4 over every unordered comment pair, averaged. **No model, runs in seconds.** | Across 160 real threads it is a **length metric first**: share of ≤15-word comments r = +0.783, mean words r = −0.723. Generated already matches length. | **Never approximate it** — it is free to compute. Also: `pairwise_self_bleu_for_order` takes **tokenized lists**, not strings; passing strings scores character-by-character and inflates it ~4×. |
| `self_bertscore_mean_f1` | BERTScore F1 between every comment pair, averaged. `microsoft/deberta-xlarge-mnli`. | **Unknown.** Four hypotheses measured and rejected: length spread, duplication tail, surface register, lexical breadth. | The gap is a **uniform +0.02 lift on every pair**, flat under trimming — so it is not a few duplicate pairs. Reading the highest-F1 pairs misleads: real threads reach F1 > 0.74 through **shared image URLs**, not shared content. |
| `semantic_mean_cosine` | Mean pairwise cosine of `all-mpnet-base-v2` comment embeddings. | Topical spread across the thread. | Drops comments under 2 words (`is_usable_comment`), so generated and real thread sizes can differ even under `--exact-matched-thread-size`. Known bug, §6. |
| `hard_disagree_rate` | Share of parent→reply pairs the local Stance_Rel head hard-labels `disagree`. | Planned stance actually being realized as disagreement. | The local checkpoint has **no graph-inference path**; missing graph features fall back to zeros. It is this repo's practical wrapper, not the published pipeline. |
| `polite_rate` / `impolite_rate` / `neutral_rate` | `Intel/polite-guard`, 4-way **single-label argmax** over {polite, somewhat polite, neutral, impolite}; each rate is the share of comments with that label. | **Warmth markers.** Measured over 412 real threads: warmth-marker rate ↔ `polite_rate` **r = +0.727**, ↔ `impolite_rate` **r = −0.601**, monotone across quintiles. This is the only causal claim in the politeness work that survived falsification. | `somewhat polite` is a **real fourth class that absorbs mass but is never reported**, so the three reported rates do not sum to 1. The failure is **realization, not planning**: the plan marginal (0.270 polite / 0.493 impolite) already matches real (0.288 / 0.443), but planned-polite slots realize as impolite 53.8% of the time. |
| `length_cv` | Per-thread coefficient of variation of whitespace word counts. | The **spread** of comment lengths, not the mean. Fixed in v98 by inverting the measured Writer length transfer function (`length_calibration.py`) so the cue asks for what actually gets realized. | A cue that says "do not pad" applied to a slot that is *undershooting* makes it worse. That was the v97 bug: the threshold was written as 100 words while the realized/target curve crosses 1.0 near **35**. |
| `avg_depth`, `structural_virality` | Tree shape only — mean comment depth, and mean shortest-path distance over all comment pairs. | Nothing in generation. Copied from the matched real thread. | See §2 — these pass structurally. |
| `mean_story_probability` | `mariaantoniak/storyseeker` P(story) per comment, **averaged over every comment in the thread** (not thresholded). | Narrative pacing anywhere in the thread. | Because it averages over *all* comments, it moves when **non-story** comments start sounding narrative. It is not the same as `story_rate`, which thresholds at 0.5. In earlier versions it was too **high**, not too low. |
| `emotion_entropy` | `SamLowe/roberta-base-go_emotions`, 28 labels; each comment's `dominant_emotion = argmax`; the metric is the **Shannon entropy of the histogram of dominant emotions** across the thread. | The **variety of argmax labels**, spread evenly. Raised in v98 by drawing exclamation marks and other typing habits at their measured rates — in the 24,029-comment reference corpus a comment with an `!` is 1.48× as likely to carry a non-neutral dominant emotion. | It is **about label variety, not emotional intensity.** A thread can be strongly emotional and score low if `neutral` still wins 48% of the argmaxes. |

---

## 4. The method — how a version gets built

### The generation pipeline

```
real Reddit corpus (camera: 574 threads)
  │
  ├─► seed pool  ── stratified, deterministic, seed 42 ──►  150 matched seeds
  │      each seed keeps its real post ID, so every metric is matched 1:1
  │
  └─► evaluation-EXCLUDED threads only (424 threads / 11,817 comments, 0 seed overlap)
         │
         └─► domain_profile.json  (schema 15)
                measured shares: typography, layout, tone×length joint,
                sentence rhythm, final punctuation, length transfer function
                                    │
                                    ▼
      Planner  ──► per-slot controls: semantic_move, stance, evidence, tone,
        │           story mode, payload type, reply delta, sentence route
        │           (never sees matched real comment text)
        ▼
      Writer   ──► one realization per slot. No resampling, no best-of-N.
        │           Only unusable output (empty / copied / placeholder /
        │           leaked control) gets bounded same-slot recovery.
        ▼
      artifacts/generalized_card/runs/<tag>/generated/
        │
        ▼
      run_evaluate.py  ──► byte-identical staging → 12 metrics → matched-seed
                            stats → content_profile_audit.{json,md}
```

The two commands:

```bash
# generation (paid; the user runs this)
python3 -u generalized_card/scripts/run_generate.py --tag <tag> --domain camera \
  --model gpt-5.4-mini --base-url https://api.openai.com/v1 \
  --api-key-env LLM_API_KEY --pool-size 150 --max-posts 10 --posts-per-run 5 \
  --start-seed-index 2 --sampling-seed 42 --resume

# evaluation (free, CPU only)
python3 generalized_card/scripts/run_evaluate.py --tag <tag> --metric-parallel 5 --resume
```

API keys live in `third_party/MiroFish/.env` as `LLM_API_KEY`.

### The large-thread gate

Every version gets one paid single-thread run **before** N=10, on a thread big
enough to show a distribution effect. `--start-seed-index 8` is post `i1o51h`
at **186 comments**, and it sits inside the N=10 window (seeds 2–11), so the
gate is *paired*: its row appears directly in the later N=10 evaluation and the
two numbers are comparable.

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag <tag>_seed8_large --domain camera --model gpt-5.4-mini \
  --base-url https://api.openai.com/v1 --api-key-env LLM_API_KEY \
  --pool-size 150 --max-posts 1 --posts-per-run 1 \
  --start-seed-index 8 --sampling-seed 42 --resume
python3 generalized_card/scripts/run_evaluate.py --tag <tag>_seed8_large \
  --metric-parallel 5 --resume
```

Other seeds in the 100–200 range, if a second large thread is wanted:
60 (199), 78 (197), 14 (195), 23 (193), 125 (189), 137 (188), 147 (186),
81 (183), 104 (182), 25 (170). Only seeds 2–11 are inside the N=10 window.

**Judge it on both axes.** At n=1 the evaluator prints `DESCRIPTIVE` and no
p-value is meaningful (§2), so:

- **content** — read the comments. Did the mechanism appear? At the rate it was
  drawn? Did it break anything the prompt already controlled?
- **distance** — the relative error against that thread's matched real row, plus
  the realized rate of whatever the version changed. `content_profile_audit.md`
  decomposes matched real → template → generated for every metric.

### The development loop, in the order that actually works

This ordering is not style. Three of four paid runs in one earlier stretch fixed
a real code defect and moved no metric, because the loop was run backwards.

1. **Measure the gap on the artifact.** Generated vs matched real, on the real
   scorer. Never on an approximation.
2. **Form a causal hypothesis** naming the mechanism, not the correlation.
3. **Try to falsify it on the evaluation-excluded real corpus — before writing
   code.** This is the single highest-value step in the loop and it costs
   nothing. In v98 it killed three hypotheses out of four, one of them *after*
   the module was already written, tested and pinned.
4. **Build one arm per mechanism**, with a flag whose legacy value reproduces
   the previous version byte-for-byte.
5. **Verify offline**: full test suite, Ruff, core-contract re-pin with zero
   drift, both parity scopes, backend self-test with the arm on *and* off,
   domain profile rebuild with zero seed overlap, `--prepare-only` dry run.
6. **One paid gate on a LARGE thread — 100 to 200 comments.** Write the
   predictions down first, then judge it two ways: **content** (read the
   comments) and **distance** (the metrics against that thread's matched real
   row). A small thread cannot show a register or distribution effect; v97 and
   v98 both gated on a 45-comment thread and neither gate predicted what N=10
   found. The command is under "The large-thread gate" above.
7. **N=10, paired to the previous version's seeds** (same `--start-seed-index`,
   same `--sampling-seed`) so the comparison means something. Only after the
   large-thread gate is clean — the p-value tests are the last step, not the
   first.
8. **Write down what was rejected**, not only what shipped. `tasks/v98-worklog.md`
   and `tasks/v99-worklog.md` are the models for this.

### Arms — the reproducibility mechanism

Every behaviour change is a named CLI flag. The flag is written into
`run_config.json`, listed in `RUN_EXPERIMENT_FIELDS`
(`run_generate.py:1205`), and checked on resume — a resume with changed
generation parameters is **rejected**, so a tag can never mean two configs.
Setting an arm to its legacy value must reproduce the prior release exactly.

v100 ships **18** such arms. Read from `run_generate.py:195-400`:

| flag | CLI default | other value(s) |
|---|---|---|
| `--writer-prompt` | `focused` | `full` |
| `--writer-route-lock` | `own_words` | `say_only` |
| `--social-contract-coherence` | `on` | `off` |
| `--reply-sibling-visibility` | `on` | `off` |
| `--own-fact-license` | `off` * | `own`, `named` |
| `--speaker-identity` | `matched` | `off` |
| `--domain-claim` | `selective` | `planned`, `off` |
| `--turn-frame` | `adjudicative_only` | `universal` |
| `--tone-length-fit` | `conditional` | `median` |
| `--long-form-layout` | `measured` | `beats_only` |
| `--reddit-typography` | `on` | `off` |
| `--sentence-rhythm` | `measured` | `off` |
| `--length-calibration` | `measured` | `off` |
| `--final-punctuation` | `measured` | `off` |
| `--route-ledger` | `on` | `off` |
| `--no-story-scope` | `sequence` | `tense` |
| `--register-realization` | `measured` | `off` |
| `--closing-move` | `measured` | `off` |

Plus two optional experimental conditioning modes, both default `none`:
`--actor-conditioning` and `--persona-conditioning`.

Which version introduced which arm is **not reliably recoverable from
`VERSION_LOG.md`** — `--writer-prompt`, `--writer-route-lock` and
`--speaker-identity` are not named in it at all, and v68–v79 share a single
provenance-correction entry rather than per-version sections. Use `git log -S`
on the flag string if you need the lineage, and do not quote a version number
for an arm's origin without checking.

\* **Gotcha: the CLI default is not always the run default.** The v97 and v98
N=10 runs used `--own-fact-license named` (CLI default `off`). Read
`run_config.json` for what a run actually used; never infer it from the CLI
defaults.

### Domain generalization — verified, with one study-design constraint

Every mechanism is domain-adaptive by construction: the *patterns* are
domain-neutral English surface forms (intensifiers, appraisal words, `my X`,
thanks) and the *rates* are measured from each domain's own evaluation-excluded
threads. A test asserts no cue text contains domain vocabulary, because every
test runs on camera and nothing else would catch it.

Checked on all four registered domains with the real sampler
(`_distribution_preserving_sample`, seed 42, 150-thread pool):

| domain | eligible threads | reference after the pool | reference comments | polite bands | somewhat_polite bands |
|---|---:|---:|---:|---|---|
| camera | 441 | **424** | 11,817 | 6/6 | 5/6 |
| cell_phone | 201 | 108 | 2,577 | 5/6 | 3/6 |
| headphone | 177 | 90 | 1,547 | 4/6 | **2/6** |
| laptop | 185 | 126 | 1,274 | 5/6 | **1/6** |

All three measured mechanisms (register, closing move, tone-length) report
`available` on all four. Missing bands degrade correctly — `band_row` returns
empty, the cue is withheld, and nothing is defaulted — so a sparse domain gets
less of the mechanism rather than a wrong rate.

**The constraint is the corpus, not the code.** Camera has 441 eligible threads,
so a 150-thread evaluation pool consumes 34% of them. The other three have
177–201, so the same pool consumes **75–85%** and leaves only 90–126 reference
threads. **For a non-camera domain, use a smaller evaluation pool** (100 or
fewer) or accept that the profile is measured on a fifth of the data camera has.
Decide this before building the seed pool, because the pool is what a run's
`run_config.json` pins.

Prerequisite per domain: per-comment `politeness_results.json` tables under
`data/raw/discussions/<domain>/<product>/`. All four have them, at coverage
camera 35/196 products, cell_phone 137/179, headphone 10/196, laptop 51/61 —
which is why headphone's reference comment count is low despite 240 threads.

### What may never happen

These protect the validity of the measurement, and they are not negotiable:

- The Writer **never** sees matched evaluation comment text.
- The domain profile is built **only** from threads excluded from the full seed
  pool, with zero seed overlap verified and recorded.
- Nothing is tuned against final test-set p-values. Calibrate on excluded
  reference data.
- Distribution diagnostics **never** select a Writer candidate. Only output that
  cannot be persisted gets bounded recovery.
- Every matched structural slot is preserved. No silent shrinking or capping of
  a matched thread.
- No domain vocabulary in Writer-facing rule text. Every test runs on the camera
  domain, so nothing else will catch it.

---

## 5. Where the code is

Branch `generator/v75-writer-realizes-planner-move`.

```
generalized_card/generalized_card/
  backend.py                 3170  adapter, arm wiring, Planner/Writer lifecycle
  prompts.py                 2716  root/reply Planner + focused/full/low-info Writer
  planning_quality.py        1082  plan audit, repair, diagnostics
  core_contract.py            773  103 pinned file hashes + policy versions
  planner_distribution.py     601  slot schedule, tone/story/affect allocation
  generation_distribution.py  576  TONE_DEFINITIONS, AFFECT_INSTRUCTIONS
  ── policy modules, one mechanism each ──
  sentence_rhythm.py          361  7 typing habits drawn per slot at measured rates
  semantic_realization.py     361  thread ledgers, turn frame, route ledger
  tone_length_fit.py          295  P(tone | size band) + iterative proportional fit
  surface_typography.py       286  per-speaker keyboard/typographic draw
  comment_structure.py             paragraph/list/quote layout per size band
  length_policy.py                 the soft length cue
  length_calibration.py            inverts the measured length transfer function
  story_scope.py                   the no-story instruction text
  register_realization.py          draws the assigned warm register per slot
  closing_move.py                  draws how a comment stops
  source_provenance.py             refuses a run whose sources are not committed
generalized_card/scripts/
  run_generate.py            1400  CLI, run_config record, subprocess env
  run_evaluate.py                  audit → stage → score → matched-evaluate
  repin_core_contract.py           walks the whole CORE_FILES table
scripts/sampling_generator/        the CARD facade the adapter patches (untouched by v97/v98)
scripts/evaluation/                the 12 scorers
```

**Everything under `generalized_card/generalized_card/`,
`generalized_card/scripts/run_generate.py` and `scripts/sampling_generator/**`
is hash-pinned** in `core_contract.py` (104 files). `verify_core_contract`
raises on drift, so:

> **Never edit a pinned file while a generation run is in flight.** It aborts the
> next batch. Park the change in `scratchpad/` and land it as its own version.

Re-pin with the script, never by hand, and confirm the drift list is exactly
the files you edited:

```bash
python3 generalized_card/scripts/repin_core_contract.py          # report drift
python3 generalized_card/scripts/repin_core_contract.py --write  # re-pin
```

New files must be `git add`-ed **before** re-pinning — the script refuses to pin
a source that is not recoverable from git.

---

## 6. Current state — v99 + v100 built, v98 measured (2026-08-20)

**v99 and v100 are built and offline-verified; neither has been run.** They ship
together in one gate, because each has its own flag and its own directly
measurable realized rate, which is what makes per-arm attribution possible from a
single artifact (§4).

- **v99** `--register-realization` — repairs the polite realization (problem 1).
- **v100** `--closing-move` — the root of the adjudication frame (problem 4).

Predictions for both are written down in `generalized_card/VERSION_LOG.md`. Read
them before the gate. **Both predict `polite_rate` and `impolite_rate` still
fail**; v100 in particular is justified by acceptance criterion 2, not by
p-values. Next action: the large-thread gate (§4).

The last measured result is v98. Policy
`generalized-card-v2-drawn-typing-rhythm-length-calibration-v98-20260819`. Run
`generalized_card_camera_gpt54_v98_rhythm_n10_20260820_v1`, N=10, paired to v97
(`--start-seed-index 2`, `--sampling-seed 42`).

**8 PASS / 1 PARTIAL / 3 FAIL** (v97 was 7/1/4).

| metric | v97 MWU | v97 Cliff | v98 MWU | v98 Cliff | status |
|---|---:|---:|---:|---:|---|
| `self_bleu_4` | 0.186 | +0.36 | 0.121 | +0.42 | PASS, weak, slightly worse |
| `self_bertscore_mean_f1` | 0.00033 | +0.96 | 0.00058 | +0.92 | **FAIL** |
| `semantic_mean_cosine` | 0.910 | +0.04 | 0.623 | −0.14 | PASS |
| `hard_disagree_rate` | 0.307 | +0.28 | 0.290 | +0.29 | PASS |
| `polite_rate` | 0.010 | −0.69 | 0.013 | −0.67 | **FAIL** |
| `impolite_rate` | 0.00077 | +0.90 | 0.0010 | +0.88 | **FAIL** |
| `neutral_rate` | 0.017 | −0.64 | 0.021 | −0.62 | PARTIAL |
| `length_cv` | 0.021 | −0.62 | **0.473** | **+0.20** | FAIL → **PASS** |
| `avg_depth` | 0.940 | +0.03 | 0.970 | +0.02 | PASS (structural) |
| `structural_virality` | 1.000 | 0.00 | 0.970 | +0.02 | PASS (structural) |
| `mean_story_probability` | 0.734 | −0.10 | 0.678 | −0.12 | PASS |
| `emotion_entropy` | 0.326 | −0.27 | **0.571** | **−0.16** | PASS, improved |

Two of the user's four priority metrics were fixed: `length_cv` and
`emotion_entropy`.

### The open problems, and what is known about each

**Four metrics are not matched**: `self_bertscore_mean_f1`, `polite_rate`,
`impolite_rate` (FAIL) and `neutral_rate` (PARTIAL). `self_bleu_4` passes but
weakly. Criterion 2 — eye-indistinguishability — is separate from all of them.

1. **`polite_rate` / `impolite_rate` / `neutral_rate` — one cause, three
   metrics, and it is now diagnosed.** Full evidence in `tasks/v99-worklog.md`.
   The plan is right (0.275 polite / 0.494 impolite against a real 0.288 /
   0.443); realization is the whole failure and it is asymmetric: **planned
   impolite realizes impolite 89.7% of the time, planned polite realizes polite
   19.3% and realizes impolite 50.3%.** The Writer has one register. It is worst
   where real text is most positive — 120+ word slots are 67.3% planned polite,
   76.7% polite in real text, and 14.3% in generated.

   The lexical signature is measured: **the generated positive vocabulary is
   about two words wide (`thanks` 2.96×, `nice` 1.48×) where real is about ten**
   — `very` 0.19×, `would` 0.21×, `love` 0.21×, `good` 0.33×, `great` 0.33×,
   `my` 0.51×, and `thank` / `amazing` / `awesome` / `incredible` / `https` at
   0.00×, all per 1,000 tokens so length cannot explain it. A TF-IDF logistic
   model fitted on excluded real text reproduces polite-guard (AUC 0.87–0.91)
   and decomposes the gap as a **+8.381 polite-vocabulary deficit against a
   −0.767 impolite-vocabulary "excess"** — generated text uses *less* of the
   impolite vocabulary than real, so suppressing negative markers would make the
   metric worse.

   **Four hypotheses rejected**, each because the gap stayed flat inside every
   cell of the conditioning variable: marker frequency (moving presence to the
   real level predicts 0.070 → 0.088 only), warmth-as-concession (contrastives
   *raise* P(polite) in real text), first-person lived experience (every
   experience feature lifts 1.4–2.2× against warmth's 3.56×), and a
   dismissive-adjudicative register (excluded-real P(polite) is 0.293 with it and
   0.315 without — no effect). **The per-slot warmth-marker schedule this project
   was carrying as the v99 plan would have been a near-null paid run.**
2. **`self_bertscore_mean_f1` — no verified mechanism.** Four hypotheses
   measured and rejected (§3). This is the honest state: do not build against a
   fifth hypothesis without falsifying it first.
3. **`self_bleu_4` — a weak pass, characterised, no cheap lever.** An exact ablation harness
   that reproduces the evaluator's number to 5 significant figures shows **no
   phrase drives it**: apostrophe normalisation, `check` openings, `that's the
   part`, and yeah/basically/actually all change it by ≤ 0.0005. OLS
   `self_bleu_4 = 0.04964 − 0.000288·meanWords − 0.00127·entityTypesPerComment`
   (R² = 0.527) explains ~48% of the gap; entity diversity's **partial r is only
   −0.097** and is worth about a third of it. Generated entity diversity is
   0.438× real in 10/10 threads, which is also an eye-visible tell — so it is
   worth doing for criterion 2 even though it is a weak metric lever.
4. **Eye-indistinguishability (criterion 2) — the adjudication frame is now
   diagnosed and addressed by v100.** The "that's the part that actually
   matters" family survived five phrase-level attempts since v73 because the
   phrase was never the thing: **how the comment stops** is. Real text closes on
   an abstract verdict 0.014 of the time and generated 0.265 (**19.1×**); real
   closes on a concrete fact of the speaker's own 0.152 against 0.048. Among real
   *story* comments the broad frame is at **0.003** against 0.382 generated —
   127× — because a story has the most obvious place to pivot.

   Three Planner-side explanations were measured and rejected first, so it is not
   a control being echoed: "decision intent" lifts the frame 1.08×, "decision
   boundary" **0.83×** (slots receiving it produce it *less*), and v97's gate
   leaves gated slots at 0.175 against ungated 0.210.

   Still unfixed: no generated comment contains a link (real 0.051), `check` at
   ~10× real, entity diversity 0.438× real.

### Known bugs, unfixed

- **Evaluation drops <2-word comments unevenly**, so `--exact-matched-thread-size`
  can still yield mismatched counts (24 generated vs 22 real on `post04_seed011`).
- **The slot distribution schedule is never persisted** to `discussion.json`, so
  `tone_length_fit` / `tone_length_joint` cannot be audited after the fact.
- **`--template-phrase-reuse-budget 4` is flat** and wrong at large thread sizes
  (real threads reach `uncertainty_frame` 7, 8, and 12).

### Recommended revert

`--no-story-scope sequence` produced **no metric benefit** (past tense
0.289 → 0.288, lexical breadth 15.95 → 16.20 against a real 21.02) and added new
repeated 4-grams. It did remove a genuine prompt contradiction — 247 of 532
prompts carried two mutually exclusive rules — so keep the contradiction fix,
but the default should go back to `tense`.

### Next step

**Run the v99 large-thread gate** (§4): `--start-seed-index 8`, 186 comments.
The predictions to read it against are in `generalized_card/VERSION_LOG.md`.
v99 predicts `polite_rate` 0.070 → 0.14–0.19, which **still fails** — it repairs
the polite realization only.

**Then v100: the impolite bleed.** Planned-neutral realizes impolite 0.513 and
planned-somewhat_polite 0.478 — 122 slots, the larger remaining share of
`impolite_rate`. It needs a *suppressive* mechanism, since no additive move
discriminates `neutral`. The two over-produced families are measured: `adjudge`
(0 of 15,294 excluded real, 0 of 659 matched real, **37 of 528 generated**) and
`dismiss_noun` (5.17× real).

### Retracted claims — do not reuse

- ~~"the no-story instruction cut advice from 0.090 to 0.008"~~ — probe
  artifact. Six independent probes ranged 0.08× to 1.53×. Token-level truth:
  `check` 0.019 → 0.194 (10×), `i'd` 3.1×, while `you should` / `you could` /
  `consider` / `make sure` are all 0.000. Advice is **register-narrow, not
  absent**.
- ~~"`self_bertscore` is lexical breadth"~~ (`tasks/HANDOFF.md` §13) — measured
  and rejected: r = +0.077 across 22 real threads.
- ~~"the tone gap is mostly a length effect"~~ — polite-guard's polite class
  keys on **warmth markers** (lift 2.24); length is secondary.

---

## 7. Working discipline — the standing rules

These are the user's instructions, and they are binding. They exist because each
one has a paid run behind it.

### Before diagnosing

- **Read every related file end to end — not grep hits, not "the relevant
  function".** In this codebase that means the CLI, the backend adapter, *every*
  prompt builder (there is more than one per role and their schemas have
  contradicted each other twice), the generator facade, the engine modules, and
  the policy modules.

  > 你确定问题、查找问题原因、解决方案、修改代码都要有了对全局代码正确的认识，
  > 也就是要读所有相关的 related files，不要盲目和修改。

  This is the most-violated rule in the project's history. `sentence_rhythm.py`
  was written, tested and pinned on an unverified hypothesis that a
  zero-cost falsification test then rejected.
- **Read the scorer before theorising about a metric** (§3).
- **Never approximate a metric that is cheap to compute.** `self_bleu_4` needs
  no model and runs in seconds; an approximation of it once produced a reported
  win that did not exist.
- **Prior handoffs, subagent reports, and your own earlier claims can be
  wrong.** Re-verify anything load-bearing against run artifacts. §6 lists three
  retractions.

### Self-check, continuously

- Check **before** acting, **while** acting, and **after** acting. Ask each
  time: is this actually the cause, or the first plausible story?
- **Try to falsify your own hypothesis on the excluded real corpus before
  writing code.** It costs nothing and it has the highest hit rate of anything
  in this project.
- Never be satisfied with a fix that has not been shown to move what it claimed
  to move.

### While changing code

- **Simplify as you go.** Do not let a module grow large or unmanageable; a high
  line count is itself a defect. Put a separable mechanism in its own focused
  module behind a small interface — do not add an algorithm to `backend.py`,
  `prompts.py`, or the shared generator.
- **Refactor promptly** rather than accreting.
- **Delete dead code** — but only after auditing repo-wide references, imports,
  monkey-patch assignments, subprocess entrypoints, reproducibility snapshots
  and tests. An accepted CLI argument whose value is never consumed is a
  correctness bug, not harmless cleanup.
- **Apply the change to every path.** One release converted only the focused
  Writer and left 106 of 522 slots on the old prompt, which made the whole run
  unattributable.
- **Import modules, not values.** `from .x import SOME_DICT` captures the empty
  dict at import time. This bug has occurred three times in this project. Use
  `from . import x` and read `x.SOME_DICT`.

### Before a paid run

- **Commit the version.** `run_generate.py` will refuse to start otherwise — see
  §8. Do not reach for the override to get past it; the override exists for
  throwaway probes, and it marks the run's artifact as unreproducible.
- Re-pin with the script and confirm the drift list is exactly the files you
  edited. `git add` new files first; the script will not pin an untracked source.
- Dry-run the exact command on a throwaway tag with `--prepare-only`, then delete
  the tag. Separately verify what `--prepare-only` skips — it returns before the
  API-key check.
- Write the predictions down before spending, so a null result is interpretable.

### Before claiming done

- Run `PYTHONPATH=generalized_card .venv/bin/python -m pytest -q generalized_card/tests`,
  Ruff, `repin_core_contract.py`, both parity scopes, and the backend self-test.
- **Prove the change is on the active path**, not merely importable. A unit test
  passing is not evidence that the runtime reaches the code — run the real entry
  point and watch it happen.
- **Report outcomes faithfully.** If a check was skipped, say so. If a number is
  a probe rather than the real scorer, say so.

### Document maintenance — the stop ritual

> 所有内容都要及时 update……当你只要停下来的时候，你就应该把一些笔记和这些 notes、
> logs 之类的东西都记好了。

**Every time you stop — not only at the end of a version — update:**

| file | what goes in it |
|---|---|
| `docs/ORIENTATION.md` (this file) | current state, next step, anything that changed the goal, the method, or a metric's interpretation |
| `tasks/todo.md` | the task list; check items off, add what the evidence opened |
| `tasks/v<N>-worklog.md` | the version's evidence, **including every rejected hypothesis and the measurement that rejected it** |
| `generalized_card/VERSION_LOG.md` | the released version, its arms, its offline gate, its run result |
| `generalized_card/RUN_INDEX.md` | any new run |
| `tasks/lessons.md` | after **any** correction from the user, or any mistake of your own: the pattern and the rule that prevents it |
| `tasks/HANDOFF.md` | a dated addendum when the detailed evidence matters |

Before you stop, the notes must answer three questions without the reader
needing the chat history:

1. **What is the next step, and why that one?**
2. **What checks did you run, and what did they say?**
3. **What did you reject, and what measurement rejected it?**

A rejected hypothesis is as valuable as a shipped one — it is the thing that
stops the next session repeating the work.

---

## 8. Reproducibility — the contract, and the gate that enforces it

**Every version must be reproducible.** Not "should be", and not by anyone
remembering to do something. This section is the contract, and it is enforced by
a check that runs before any API call.

### What makes a version reproducible

Five things, all recorded in the run's own `run_config.json`:

| # | mechanism | what it gives you |
|---|---|---|
| 1 | `source_provenance.commit` | the commit holding the exact sources — `git show <sha>:<path>` returns the file that produced the run |
| 2 | `generator_policy_version` + `generator_core_provenance` | the policy string and the SHA-256 of all 55 pinned generation sources, as they were at run time |
| 3 | every arm value, plus `RUN_EXPERIMENT_FIELDS` resume verification | the exact configuration; a resume with changed parameters is rejected, so a tag can never mean two configs |
| 4 | `domain_profile_sha256` + `domain_profile_schema_version`, and `domain_profile.json` copied into the run directory | the measured shares the generation was conditioned on |
| 5 | `seed_pool` path + `sampling_seed` + `start_seed_index` | which real threads were matched, in which order |

### The gate

`generalized_card/generalized_card/source_provenance.py`, called from
`run_generate.py` immediately after `verify_core_contract` and **before the seed
pool, the domain profile, and every API call**:

> A generation run **refuses to start** if any file that defines the version is
> not in `HEAD`.

It checks `version_source_paths(...)` — the 55 pinned generation sources **plus
`core_contract.py` itself**. The contract cannot carry its own hash, so
`verify_core_contract` can never check it, yet it is the file that names the
policy version and holds every other pin; a version whose contract is
uncommitted is not recoverable even when all 55 modules are.

"Not in `HEAD`" covers all three ways a file can be missing: modified in the
working tree, **staged but never committed**, and untracked.

The error names every offending file and the fix. There is one override,
`GENERALIZED_CARD_ALLOW_UNCOMMITTED_SOURCE=1` — an environment variable rather
than a CLI flag so it cannot be set by accident inside a long generation
command. Using it is recorded as `source_provenance.override: true` in
`run_config.json`, so a run made without provenance says so in its own artifact
instead of looking like every other run.

`tests/test_source_provenance.py` includes a **live guard**,
`test_this_repository_is_currently_reproducible`, which fails whenever the
working tree holds an unshipped version. If you see that test fail, the fix is
to commit, not to change the test.

### How to reproduce a past run

```bash
R=artifacts/generalized_card/runs/<tag>
python3 -c "import json;print(json.load(open('$R/run_config.json'))['source_provenance'])"
git worktree add /tmp/repro <commit-from-above>     # or: git checkout <commit>
python3 -c "import json;print(json.load(open('$R/run_config.json'))['command'])"
```

`command` is the redacted original invocation. The domain profile is already in
the run directory, so it does not need rebuilding — and rebuilding it would be
wrong, since the profile is measured and its SHA-256 is what was recorded.

### Why this exists

`HISTORICAL_GENERATION_POLICY_VERSIONS` stores version strings with **no
per-version file hashes**, so it identifies a version but cannot reconstruct it.
Reconstruction has only ever depended on git. And on 2026-08-20, checked rather
than assumed:

```
HEAD = a34abc6  → core_contract.py at HEAD named v96 as current
git log -- generalized_card/generalized_card/sentence_rhythm.py   → empty
git log -- generalized_card/generalized_card/length_calibration.py → empty
```

**v97 and v98 existed only in the working tree** — two shipped releases, one of
them the source of the N=10 result in §6, with no recoverable source tree.

The near miss is worth knowing, because it is why a written rule was not enough:
`repin_core_contract.py` already refused to pin a file `git ls-files` did not
know about, and it reported `untracked active: 0` the whole time. `git ls-files`
lists *tracked* files, and a staged-but-never-committed file is tracked. Every
check in place answered "has this drifted?"; none answered "can this be
recovered?"

Closed by:

| commit | contents |
|---|---|
| `e213f7a` | v97 + v98 sources, tests, version docs — 33 files |
| `1abdb0e` | this file, the metric reference, the README pointer |
| `aa22450` | traceability record, two corrected claims, two lessons |
| this version | `source_provenance.py`, the gate, 19 tests |

The chain was then verified, not assumed: the working tree is clean for every
pinned source against `HEAD`, and `repin_core_contract.py` reports zero drift —
so `HEAD`'s blobs hash to exactly the pinned values.

**One loss is permanent.** v97 and v98 could not be separated after the fact —
the working tree interleaved them — so they share commit `e213f7a` and v97's
standalone tree is unrecoverable. That is the cost of having found this late, and
it is the reason the gate is a gate rather than a paragraph.

### The rule

**Commit the version before the paid run, not after.** The gate now enforces it,
but the rule is what matters: the commit is what gives `run_config.json`'s policy
string something to point at. `scripts/sampling_generator/` is untouched by v97
and v98, so the CARD core is unaffected by either.

---

## 9. What was verified to write this file

Not "I believe"; these were run on 2026-08-20:

| check | result |
|---|---|
| `pytest -q generalized_card/tests` | **527 passed** (449 + 19 provenance + 28 register + 31 closing) |
| `repin_core_contract.py` (report mode) | 104 pinned, 0 missing, 0 untracked active, 0 unpinned local imports, **0 drift** |
| v99 profile rebuild | schema 16 over 424 excluded threads, **0 seed overlap**, 4,787 polite comments measured; six bands monotone |
| v99 draw fidelity | rendered draw against measured share within **0.011** in every band and every move, 4,000 slots per band |
| v99 on the real prompt path | rule in 33/40 polite prompts with the arm on, **0/40 off**, 6 distinct forms, +283 chars (+7.7%) |
| v99 backend self-test | passes with the arm on **and** off |
| v100 profile rebuild | schema 17, 6,609 comments of 25+ words over 424 excluded threads, **0 seed overlap** |
| v100 draw fidelity | within **0.008** in every band and move |
| v100 on the real prompt path | rule in 32/40 slots, **silent below the 25-word floor**, 8/8 at 45w+; self-test passes on and off |
| v98 N=10 metric table | read verbatim from `matched_evaluation/matched_seed_group_eval.md` |
| v97 N=10 metric table | read verbatim from the v97 run's same file |
| v97/v98 seed pairing | both `start_seed_index=2`, `sampling_seed=42`, `max_posts=10` — confirmed paired |
| status thresholds | read from `run_evaluate.py:411`, not from memory |
| the 12-metric list | read from `REQUIRED_THREAD_METRICS`, `run_evaluate.py:28` |
| git traceability, before | `git log` per file + `git show HEAD:core_contract.py` — v97/v98 confirmed **uncommitted** |
| git traceability, after | v97+v98 committed as `e213f7a`, docs as `1abdb0e`; `git log` per new file now resolves; v98's policy string present in `HEAD`'s `core_contract.py`; pinned sources clean against `HEAD` with 0 drift |
| the provenance gate, on the real path | `run_generate.py --prepare-only` on a throwaway tag **refused to start**, naming all three uncommitted files including `core_contract.py`, and created no run directory |
| the override, on the real path | same command with `GENERALIZED_CARD_ALLOW_UNCOMMITTED_SOURCE=1` proceeded and wrote `source_provenance: {commit, branch, uncommitted[3], checked: 56, override: true}` into `run_config.json`; throwaway tag then deleted |
| `ruff check generalized_card` | **no issues found** |
| the 16 ablation arms | read from `run_generate.py:195-400`, with each default and legacy value |
| module line counts | `wc -l` |

Anything in this file not in that table came from a linked document, and the
link is the citation.
