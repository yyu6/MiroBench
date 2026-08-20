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
| 4 | `generalized_card/VERSION_LOG.md` | every released version, its arms, and its result | when comparing versions |
| 5 | `tasks/lessons.md` | 48 mistakes, each with the rule that prevents it | before diagnosing anything |
| 6 | `generalized_card/AGENTS.md` | binding engineering rules for `generalized_card/` | before writing code |
| 7 | `tasks/HANDOFF.md` | the long-form evidence archive, newest addendum first | when you need the detail behind a claim |
| 8 | `docs/thread_metric_score_reference.md` | every exported metric, its scorer, its model | when you need a scorer's exact semantics |
| 9 | `generalized_card/RUN_INDEX.md` | all 142 runs with tag, cost, and outcome | when locating an artifact |

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
6. **One paid single-seed content gate.** Write the predictions down first, then
   read every comment against them.
7. **N=10, paired to the previous version's seeds** (same `--start-seed-index`,
   same `--sampling-seed`) so the comparison means something.
8. **Write down what was rejected**, not only what shipped. `tasks/v98-worklog.md`
   is the model for this.

### Arms — the reproducibility mechanism

Every behaviour change is a named CLI flag. The flag is written into
`run_config.json`, listed in `RUN_EXPERIMENT_FIELDS`
(`run_generate.py:1205`), and checked on resume — a resume with changed
generation parameters is **rejected**, so a tag can never mean two configs.
Setting an arm to its legacy value must reproduce the prior release exactly.

v98 ships **16** such arms. Read from `run_generate.py:195-400`:

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
  core_contract.py            747  101 pinned file hashes + policy versions
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
generalized_card/scripts/
  run_generate.py            1400  CLI, run_config record, subprocess env
  run_evaluate.py                  audit → stage → score → matched-evaluate
  repin_core_contract.py           walks the whole CORE_FILES table
scripts/sampling_generator/        the CARD facade the adapter patches (untouched by v97/v98)
scripts/evaluation/                the 12 scorers
```

**Everything under `generalized_card/generalized_card/`,
`generalized_card/scripts/run_generate.py` and `scripts/sampling_generator/**`
is hash-pinned** in `core_contract.py` (101 files). `verify_core_contract`
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

## 6. Current state — v98, 2026-08-20

Policy `generalized-card-v2-drawn-typing-rhythm-length-calibration-v98-20260819`.
Run `generalized_card_camera_gpt54_v98_rhythm_n10_20260820_v1`, N=10, paired to
v97 (`--start-seed-index 2`, `--sampling-seed 42`).

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
   metrics.** Verified: warmth-marker rate ↔ `polite_rate` r = +0.727 over 412
   real threads, monotone across quintiles. Generated sits at warmth 0.143
   (real 0.186) and negative markers 0.141 (real 0.047, **3×**). It is a
   realization failure — the plan marginal already matches. **Recorded caveat:**
   at warmth 0.143 the quintile curve predicts `polite_rate` ≈ 0.23 but the
   generator gets 0.066, so the markers are also being *used* differently, not
   only used less. Any fix must account for that residual.
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
4. **Eye-indistinguishability (criterion 2)** — the tells listed in §1 are
   unfixed, and no metric measures them.

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

**v99: schedule warmth markers, second-person advice forms, and
negative-marker suppression at their measured rates**, through the
`sentence_rhythm` mechanism (already proven, draw fidelity ≤ 0.008). It targets
3 of the 4 open metrics and it is the only remaining hypothesis with a verified
causal claim. Before writing it, read `planner_distribution.py` and the
tone/affect renderers in `prompts.py` (`_tone_shape_guidance`,
`_speaker_role_guidance`, `_utterance_mode_guidance`, `_substitution_rule`) —
that reading is **not yet done**.

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

### Before claiming done

- Run `PYTHONPATH=generalized_card .venv/bin/python -m pytest -q generalized_card/tests`,
  Ruff, `repin_core_contract.py`, both parity scopes, and the backend self-test.
- Dry-run any command on a throwaway tag with `--prepare-only`, then delete the
  tag. And separately verify what `--prepare-only` skips — it returns before the
  API-key check.
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

## 8. Traceability

### How a version is supposed to be recoverable

Four mechanisms, and all four have to hold:

1. `run_config.json` in the run directory records the policy version, every arm
   value, the domain-profile SHA-256 and schema version, the seed pool, and the
   exact command.
2. `core_contract.py` pins the SHA-256 of all 101 active source files and
   refuses to run on drift.
3. `HISTORICAL_GENERATION_POLICY_VERSIONS` in `core_contract.py` records every
   released policy string, so an old artifact can be identified.
4. **Git commits carry the actual source tree.**

### The defect that was found, and how it was closed — 2026-08-20

`HISTORICAL_GENERATION_POLICY_VERSIONS` stores **version strings only, with no
per-version file hashes.** So mechanism 3 identifies a version but cannot
reconstruct it — reconstruction depends entirely on mechanism 4, git. And
mechanism 4 was broken:

```
HEAD = a34abc6  "fix(generalized-card): ground selective facts in reply plans"
     → core_contract.py at HEAD named v96 as current
git log -- generalized_card/generalized_card/sentence_rhythm.py   → empty
git log -- generalized_card/generalized_card/length_calibration.py → empty
```

**v97 and v98 existed only in the working tree** — two shipped releases, one of
them the source of the N=10 result in §6, with no recoverable source tree. The
run directory holds `generated/`, `logs/` and `run_config.json`; there is no
source snapshot.

Fixed by two commits on `generator/v75-writer-realizes-planner-move`:

| commit | contents |
|---|---|
| `e213f7a` | v97 + v98: 33 files — 14 policy modules, `run_generate.py`, 11 test modules, `VERSION_LOG.md`, `RUN_INDEX.md`, worklogs |
| `1abdb0e` | `docs/ORIENTATION.md`, `docs/thread_metric_score_reference.md`, README pointer |

`scripts/sampling_generator/` is untouched by v97 and v98, so the CARD core is
unaffected by either.

**The chain is now closed, and this was checked rather than assumed:** the
working tree is clean for all 101 pinned sources against `HEAD`, and
`repin_core_contract.py` reports zero drift — so `HEAD`'s blobs hash to exactly
the pinned values. `run_config.json` names the policy version; that string is
present in `HEAD`'s `core_contract.py`; that commit holds the sources the pins
were computed from.

**One loss is permanent.** v97 and v98 could not be separated after the fact —
the working tree interleaved them — so they share one commit boundary and v97's
standalone tree is not recoverable.

**The rule going forward: commit at every version boundary, before the paid run,
not after.** A pinned hash proves a file has not changed since you pinned it; it
does not store the file. Any check that answers "has this drifted?" is not an
answer to "can this be recovered?" — see the 2026-08-20 lesson in
`tasks/lessons.md`.

---

## 9. What was verified to write this file

Not "I believe"; these were run on 2026-08-20:

| check | result |
|---|---|
| `pytest -q generalized_card/tests` | **449 passed** in 25.4s |
| `repin_core_contract.py` (report mode) | 101 pinned, 0 missing, 0 untracked active, 0 unpinned local imports, **0 drift** |
| v98 N=10 metric table | read verbatim from `matched_evaluation/matched_seed_group_eval.md` |
| v97 N=10 metric table | read verbatim from the v97 run's same file |
| v97/v98 seed pairing | both `start_seed_index=2`, `sampling_seed=42`, `max_posts=10` — confirmed paired |
| status thresholds | read from `run_evaluate.py:411`, not from memory |
| the 12-metric list | read from `REQUIRED_THREAD_METRICS`, `run_evaluate.py:28` |
| git traceability, before | `git log` per file + `git show HEAD:core_contract.py` — v97/v98 confirmed **uncommitted** |
| git traceability, after | v97+v98 committed as `e213f7a`, docs as `1abdb0e`; `git log` per new file now resolves; v98's policy string present in `HEAD`'s `core_contract.py`; pinned sources clean against `HEAD` with 0 drift |
| `ruff check generalized_card` | **no issues found** |
| the 16 ablation arms | read from `run_generate.py:195-400`, with each default and legacy value |
| module line counts | `wc -l` |

Anything in this file not in that table came from a linked document, and the
link is the citation.
