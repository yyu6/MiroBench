# ORIENTATION — read this first

**Purpose.** This is the single entry point for anyone — human or a fresh agent
session — picking up the synthetic-Reddit-thread work. It answers four things:
what we are trying to do, how we are doing it, how to read the metrics, and how
to work on this codebase without wasting a paid run.

It is deliberately high-level and it is deliberately short. Every section ends
with a pointer to the file that holds the evidence. **This file states
conclusions; the linked files hold the measurements.**

Last verified: **2026-08-20** (v101 result, plus the `hard_disagree_rate`
diagnosis). See §9 for what "verified" means here and what was actually checked
to write this line.

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
| 2 | **`docs/DECISIONS.md`** | every rule currently in force, each marked VERIFIED / MEASURED / ASSUMED / RETRACTED | **always, second** — an ASSUMED rule is not a rule |
| 3 | `tasks/todo.md` | the task list, ordered by which measured gap it moves | before choosing what to do |
| 4 | `tasks/v<N>-worklog.md` | the current version's full evidence, including rejected hypotheses | before touching that version's code |
| 4b | `tasks/v99-worklog.md` | the politeness diagnosis: four rejected hypotheses and the verified mechanism | before touching tone or register |
| 4c | `tasks/v102-worklog.md` | the `hard_disagree_rate` diagnosis: nine rejected hypotheses, two surviving mechanisms, one causally measured | before touching stance, openers, or reply framing |
| 4d | `tasks/v104-worklog.md` | the tone-pair diagnosis: why eight versions of marker work failed, and the carrier-sentence mechanism | before touching `polite_rate` or `impolite_rate` |
| 4e | `generalized_card/analysis/` | the scripts that reproduce every number in 4b-4d and in `DECISIONS.md` | when you want to re-measure rather than re-derive |
| 5 | `generalized_card/VERSION_LOG.md` | every released version, its arms, and its result | when comparing versions |
| 6 | `tasks/lessons.md` | every mistake made here, each with the rule that prevents it | before diagnosing anything |
| 7 | `generalized_card/AGENTS.md` | binding engineering rules for `generalized_card/` | before writing code |
| 8 | `tasks/HANDOFF.md` | the long-form evidence archive, newest addendum first | when you need the detail behind a claim |
| 9 | `docs/thread_metric_score_reference.md` | every exported metric, its scorer, its model | when you need a scorer's exact semantics |
| 10 | `generalized_card/RUN_INDEX.md` | all runs with tag, cost, and outcome | when locating an artifact |
| 11 | `.claude/handoffs/` | session handoffs, newest last; each carries a TYPE INSTRUCTION block for the next agent. **Newest: `2026-08-21-231454-geo-v104-...`. The 2026-08-20 handoff's judging section is retracted — it steers by `\|Cliff\| <= 0.10`.** | when picking the work up cold |

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
`tasks/v98-worklog.md` and are not yet fixed. Found on v103 (§6.3): two
different comments in the same thread independently restating one specific
argument in different words — read the actual examples in §6.3, not just this
pointer.

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

### How to actually read it — five traps

1. **A large p-value at N=10 is not evidence of a match.** The test is
   *unpaired* while the data is *paired by seed* — each generated thread has
   exactly one matched real thread. Unpaired tests on paired data are
   conservative, so N=10 p-values are **optimistic**. At |Cliff| = 0.25 a
   metric passes ~87% of the time at N=10 and ~4% at N=150.
2. **The real target is the effect size — but not zero, and not 0.10.**
   **RETRACTED 2026-08-21: `|Cliff| ≤ 0.10` was below the noise floor.** Measured
   on 440 real camera threads, two *real* samples reach `|Cliff| ≤ 0.10` on all
   twelve metrics **0.0% of the time at N=10** and 26% at N=150. The null 95th
   percentile of `|Cliff|` is **≈0.52 per metric at N=10** and **≈0.13 at
   N=150**. So an N=10 Cliff reading anywhere under about 0.5 is
   indistinguishable from a second sample of real threads, and every N=10 Cliff
   number in this project's history below that is noise. Steer by the distance
   to that floor, per metric and per N, or by trap 4's paired bias.
   Reproduce with `generalized_card/analysis/acceptance_standard.py`.
3. **Barely above 0.05 does not count.** The user rejected N-based
   extrapolation ("this would pass at N=150") unless it is publicly,
   scientifically established.
4. **At N=10, Cliff against matched real is not the generator's error.** The
   Planner aims at a **held-out same-size** real thread, never the matched one,
   so the per-thread target is an independent draw and carries almost no
   information about the thread it stands in for — for `hard_disagree_rate`,
   corr(template, matched real) = **−0.281**. Cliff of the *template* against
   matched real is the **ceiling a perfect generator could reach**, and on the
   v103 N=10 it ran from 0.00 to **−0.36** depending on the metric. **Measure the
   generator against its own target, paired** (`generated − reference_metric_template`,
   Wilcoxon against zero); that is the part that survives to N=150, because the
   template distribution converges on the real one as n grows while a generator
   bias does not. Steering by raw Cliff-vs-real at n=10 is partly steering by
   which ten templates were drawn.
5. **The current standard fails a perfect generator half the time.** This is
   no longer an estimate. Drawing two disjoint samples of real camera threads and
   running the evaluator's own tests on them:

   | standard | N=10 | N=150 |
   |---|---:|---:|
   | current: all 24 raw p > 0.05 | 0.63 | **0.50** |
   | **Holm–Bonferroni over the 24 tests** | 0.98 | **0.98** |
   | every \|Cliff\| ≤ 0.10 | 0.00 | 0.26 |

   A perfect generator *is* a second sample of real threads, so the row reading
   ≈0.95 is the only one that does not fail correct work. **Recommendation:
   report all 12 with Holm–Bonferroni over the 24 tests, and print the
   real-vs-real null pass rate beside it as the calibration line.** The choice is
   still the user's; what is no longer open is that the current standard is
   mis-specified. `acceptance_standard.py` produces these three rows for any
   domain from that domain's own thread tables.

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
| `hard_disagree_rate` | Share of parent→reply pairs the local Stance_Rel head hard-labels `disagree`. Root comments are pairs too — their parent is the post — and they are ~37% of all pairs. | **The opening of a reply.** The head is a reply-text classifier (surrogate AUC 0.740 on the reply, 0.579 on the parent) keyed by explicit stance tokens, *agreement ones included*. A `polarity_token` opener carries P(disagree) 0.457 against a 0.18 base. | The head is **nearly degenerate** — all three class probabilities sit inside ≈[0.26, 0.41], so the metric is an argmax on a knife edge and a ±0.005 probability shift moves it. Do not read it as semantic disagreement; planned `agree` slots are labelled disagree *more* often than planned `disagree` ones. `pair_count` in the merged CSV is **not** this metric's pair count. |
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

v102 ships **19** such arms. Read from `run_generate.py:195-400`:

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
| `--opening-move` | `measured` | `off` |

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

## 6. Current state — v103 measured 9/1/2, effect sizes worse than v101 (2026-08-21)

The last measured result is **v103**. Policy
`generalized-card-v2-stance-consistent-opening-v103-20260821`. Run
`generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1`, N=10, paired
to v101 (`--start-seed-index 2`, `--sampling-seed 42`), $3.7345.

**9 PASS / 1 PARTIAL / 2 FAIL** — nominally the best count in the project's
history (v101 9/0/3, v98 8/1/3, v96 6/0/6). **The count is not the result.**
Metrics inside the |Cliff| ≤ 0.10 bar went 6/12 → 4/12, **but that count is not a
sound version comparison** — see §2's new trap 5. Corrected for the template
ceiling, the generator improved on five metrics and got significantly worse on
none. Only **three** metrics carry a statistically real generator bias:
`polite_rate` (p = 0.002), `impolite_rate` (p = 0.002) and
`self_bertscore_mean_f1` (p = 0.014) — the three that have failed since v96.
Full tables in `generalized_card/VERSION_LOG.md`.

`hard_disagree_rate` moved Cliff +0.37 → −0.23, mean 0.1569 → 0.0920 against a
real 0.1208. **That is not an overshoot — it is convergence.** The Planner aims
at a *held-out same-size* real thread, never the matched one (§4, isolation), so
the per-thread target is an independent draw. Measured against its own target,
the generator's bias went **+0.0681 → +0.0032** (Wilcoxon p = 1.000): it now hits
the plan almost exactly. Cliff of the **template** against matched real — what a
perfect generator would score at n=10 — is **−0.36**, so v103 at −0.23 is closer
to real than a perfect generator would be with these ten templates.

Pooled over all pairs it reads 0.1198 against 0.1218, while the root conditional
fell 0.0621 → 0.0284 (real 0.0630) and the reply conditional was repaired (+0.080
over real → +0.022). The pooled agreement is partly cancellation, and the metric
is a thread mean, not a pooled rate — but the thread-level number is dominated by
template noise, not by that split.

**What that exposed is the useful part.** Realized polarity-token openers sit at
0.0847 on root comments against a real 0.0224, and 0.0507 on replies against a
real 0.0685 — **inverted**. `opener_profile` measures a pooled marginal and
nothing makes the assignment respect the root/reply conditional. In v101 this was
invisible because the opener was only 18% obeyed. **Repairing a realization is
what made the plan's own error measurable for the first time.**

The v101 table below is kept because it is the comparison baseline.

| metric | v98 MWU | v98 Cliff | v101 MWU | v101 Cliff | |
|---|---:|---:|---:|---:|---|
| `self_bleu_4` | 0.1212 | +0.42 | 0.1041 | +0.44 | PASS, weak |
| `self_bertscore_mean_f1` | 0.00058 | +0.92 | 0.00283 | **+0.80** | FAIL |
| `semantic_mean_cosine` | 0.6232 | −0.14 | 0.8501 | **−0.06** | PASS |
| `hard_disagree_rate` | 0.2897 | +0.29 | 0.1735 | +0.37 | PASS, worse |
| `polite_rate` | 0.0126 | −0.67 | 0.0210 | **−0.62** | FAIL |
| `impolite_rate` | 0.0010 | +0.88 | 0.0046 | **+0.76** | FAIL |
| `neutral_rate` | 0.0210 | −0.62 | 0.0587 | **−0.51** | PARTIAL → **PASS** |
| `length_cv` | 0.4727 | +0.20 | 0.7337 | **+0.10** | PASS |
| `avg_depth` | 0.9698 | +0.02 | 0.9095 | +0.04 | PASS (structural) |
| `structural_virality` | 0.9697 | +0.02 | 0.9697 | +0.02 | PASS (structural) |
| `mean_story_probability` | 0.6776 | −0.12 | 0.8501 | **−0.06** | PASS |
| `emotion_entropy` | 0.5708 | −0.16 | 0.8501 | **−0.06** | PASS |

### The honest N=150 projection — this is the number that matters

Pass probability at the final scale is a function of the **effect size**, not the
current p-value (§2, and the 2026-08-19 lesson):

| \|Cliff\| | metrics | P(pass) at N=150 |
|---|---|---:|
| ≤ 0.06 | `semantic_mean_cosine`, `mean_story_probability`, `emotion_entropy`, `structural_virality`, `avg_depth` | ~0.90 |
| 0.10 | `length_cv` | ~0.72 |
| 0.37–0.44 | `hard_disagree_rate`, `self_bleu_4` | ~0.01 |
| 0.51–0.80 | `neutral_rate`, `polite_rate`, `impolite_rate`, `self_bertscore_mean_f1` | ~0.00 |

**Six safe, six not** — and two of the six safe ones are copied from the real
reply tree by the matched sampler, not won by generation. 9/0/3 at N=10 is real
progress and is **not** the same thing as being close at N=150.

### The open problems, and what is known about each

1. **`hard_disagree_rate` — diagnosed 2026-08-20, two mechanisms survive.** Full
   evidence in `tasks/v102-worklog.md`; reproduce it with
   `generalized_card/analysis/disagreement_diagnosis.py`. The metric is the share
   of parent→reply pairs the local Stance_Rel head argmaxes to `disagree`, and
   the head is **nearly degenerate** — all three class probabilities sit inside
   ≈ [0.26, 0.41], so the metric is an argmax on a knife edge and the entire gap
   is a **uniform ≈ +0.017 translation of the decision margin**.

   Where the gap lives: **root pairs already match** (generated 0.0621 against a
   real 0.0630); **reply pairs are 1.56× real** (0.2235 against 0.1433) and are
   100% of the gap. A TF-IDF surrogate fitted on excluded real reaches AUC 0.740
   on the reply text alone and 0.579 on the parent alone — the head is
   essentially a **reply-text** classifier keyed by explicit stance tokens,
   *agreement ones included* (`agree`, `agreed`, `yup`, `yeah`, `exactly`).

   Two mechanisms survived falsification:
   - **The assigned opener is not realized.** `opener_type` is scheduled from the
     domain profile and the instruction reaches the Writer prompt at exactly the
     measured share, but `polarity_token` comes out at **2.42×** it (0.1274
     against 0.0526), sourced from `discourse_marker` slots (obeyed 0.184) and
     `content_phrase` slots (0.460). `polarity_token` is the
     highest-disagreement opener there is: real P(d) 0.457 against a 0.18 base.
     Causally measured on an exact ablation harness that reproduces the artifact
     label-for-label: stripping only the *unassigned* polarity openers moves the
     reply rate 0.2235 → **0.1862**, i.e. **47% of the reply-pair gap**, with
     `self_bleu_4` unharmed (0.03330 → 0.03297).
   - **Generated replies echo the parent's content words 1.4–1.6× as often**
     (0.2145 against a real 0.1367–0.1542). In real text P(disagree) rises
     monotonically with echo across all six bins, generated's within-bin
     conditionals track real's, and the counterfactual at the real echo
     distribution closes **55%** of the gap. It survives conditioning on both
     parent length and reply length — 1.27–2.04× in all ten populated cells. The
     existing `context_transform` arm does **not** fix it: echo is *highest*
     (0.259) in `parent_hidden`, where the Writer never sees the parent text.

   Nine hypotheses were rejected, including the v100 adjudication frame
   (−0.0029 on 11 slots), contrastives (removing them *raises* the rate), the
   closing sentence (removing it *raises* the rate), hedges (0.0000), a graph
   feature asymmetry (coverage 0.0000 vs 0.0039), and environment drift (real
   tables reproduce exactly today).
2. **`polite_rate` / `impolite_rate` / `neutral_rate` — one cause, three
   metrics, diagnosed.** Full evidence in `tasks/v99-worklog.md`. The plan is
   right (0.275 planned polite against a real 0.288); realization is the whole
   failure and it is asymmetric: **planned impolite realizes impolite 89.7%,
   planned polite realizes polite 19.3% and impolite 50.3%.**

   The lexical signature is measured: **the generated positive vocabulary is
   about two words wide (`thanks` 2.96×, `nice` 1.48×) where real is about ten**
   — `very` 0.19×, `would` 0.21×, `love` 0.21×, `good` 0.33×, `great` 0.33×,
   `my` 0.51×, and `thank` / `amazing` / `awesome` / `incredible` / `https` at
   0.00×, all per 1,000 tokens so length cannot explain it. A TF-IDF logistic
   model fitted on excluded real reproduces polite-guard (AUC 0.87–0.91) and
   decomposes the gap as a **+8.381 polite-vocabulary deficit against a −0.767
   impolite-vocabulary "excess"** — generated uses *less* of the impolite
   vocabulary than real, so suppressing negative markers makes the metric worse.

   Four hypotheses rejected, each because the gap stayed flat inside every cell:
   marker frequency, warmth-as-concession, first-person lived experience, and a
   dismissive-adjudicative register. v99/v101's per-register drawn realization
   improved the effect sizes for the first time in four versions but did not
   close them. **The opener defect above is not this defect** — real polite rate
   by opener runs 0.18–0.47 and generated 0.02–0.15 in *every* class, so the
   tone gap is flat across openers.

   The **largest single untapped lever** remains the possessive: generated
   carries `my X` at 0.081 against a real 0.230, and the real conditional is
   P(polite | possessive) = 0.509 against 0.254 without. It only works if the
   possessive arrives as a bare fact rather than a story, which is what v101's
   state-not-event cue rewording was for and which v101 confirmed does not raise
   story probability.
3. **`self_bertscore_mean_f1` — six hypotheses rejected; the pairwise
   decomposition (`docs/DECISIONS.md` G3) now localizes it.** Length spread,
   duplication tail, surface register, lexical breadth (r = +0.077), narrow
   shared vocabulary (r = +0.155 and −0.096, both the wrong sign; the narrowness
   is *cross-thread* while the metric is *within-thread*), and environment
   drift (real-side per-thread scores were computed under
   `transformers==5.7.0`, generated-side under `4.48.0`/`5.10.1` — every version
   since v96 — but rescoring one real thread under `4.48.0` moved its mean by
   1.6e-8, so this is not a mechanism either).

   Decomposed `--include-pairs` on the v103 N=10 artifact against its 10
   matched real threads (fidelity-checked against the shipped per-thread means
   first, `generalized_card/analysis/bertscore_pair_diagnosis.py pairs`): the
   excess is **not** the `hard_disagree_rate` parent-echo story. `same_branch`
   pairs (siblings, cousins, indirect ancestor-descendant) show no reliable
   excess at all (+0.0056, Wilcoxon p = 0.32 across the 10 thread pairs); if
   generated replies were echoing content down their own branch, this bucket
   should be elevated and it is not. Instead the gap is a **root-vs-reply role
   effect**: `root_root` pairs are clean (+0.0039, p = 0.63) while `reply_reply`
   pairs carry the largest, most significant excess (+0.0274, p = 0.002, same
   sign in all 10 threads) and `root_reply` sits between (+0.0130, p = 0.027).
   It is a **sign inversion**, not a uniform shift: real reply-reply pairs are
   *less* similar than real root-root pairs (0.4905 vs 0.4955) while generated
   reply-reply pairs are *more* similar than generated root-root pairs (0.5136
   vs 0.5089). `parent_child` pairs carry the single largest per-pair excess
   (+0.0256, p = 0.0098) but are only 1.3% of all pairs, too rare to move the
   pooled metric. **Root comments already match real on this metric, same as
   `hard_disagree_rate`** — this is now the second metric where the entire
   defect lives in reply comments and root-level generation is clean.

   No sixth hypothesis has been built for *why* generated replies read more
   similar to each other than real replies do, regardless of branch. That is
   the open question, and it is now scoped to reply-only generation rather than
   the whole metric.

   **Two follow-up checks, both offline.** First: is "real replies are more
   diverse than real root comments" a property of the 10 matched threads or of
   Reddit writing generally? Checked at corpus scale with the cheap
   `all-mpnet-base-v2` cosine proxy (`analysis/root_reply_diversity.py`) over
   247 of the 424 evaluation-excluded real camera threads: `reply_reply`
   cosine sits below `root_root` cosine in 82% of threads (202/247), mean
   difference −0.096, Wilcoxon p≈0. It generalizes — this is not the ten
   matched threads' luck.

   Second: is `microsoft/deberta-xlarge-mnli`/BERTScore even a sound choice of
   detector for this dimension, or could the whole excess be a model artifact?
   Read the actual highest/lowest-F1 pairs on the current artifact
   (`bertscore_pair_diagnosis.py inspect`) rather than trust the v98-era note.
   The real high tail: two literal same-author self-repeats (confirmed against
   the raw scrape — one author, two different parents, "I'm talking about
   video." said twice in one thread), the previously-documented shared-image-URL
   artifact (reconfirmed), and one genuine paraphrase. The **generated** high
   tail is dominated by real, visible **argument-level paraphrase duplication**
   — distinct comments in the same thread independently restating one specific
   claim in different words: "compactness doesn't matter once it's in a bag"
   said twice in seed002, "you need a real stress test" said twice in seed008,
   "test AF tracking with a moving subject" said twice in seed008, "check
   full-res files, not the compressed clip" said twice in seed011, plus a
   near-identical short pair ("Shipping email?" / "Shipping email? haha") in
   seed006. This is a genuine criterion-2 tell — add it to the list in §1 —
   but v98's trimming test already showed it is quantitatively too small to be
   the aggregate driver (trimming the top 20% of pairs barely moves the gap).
   The two readings do not conflict; they are different questions. **No
   evidence the detector choice itself is bad**: it responds correctly to
   genuine paraphrase on both sides, and its one known noise mode (shared
   URLs) cannot explain the generated-side excess, since generated text never
   contains a link.
4. **`self_bleu_4` — a weak pass, characterised, no cheap lever.** An exact
   ablation harness reproducing the evaluator to 5 significant figures shows **no
   phrase drives it**: apostrophe normalisation, `check` openings, `that's the
   part`, and yeah/basically/actually all change it by ≤ 0.0005. OLS
   `self_bleu_4 = 0.04964 − 0.000288·meanWords − 0.00127·entityTypesPerComment`
   (R² = 0.527) explains ~48% of the gap; entity diversity's partial r is only
   −0.097. Generated entity diversity is 0.438× real in 10/10 threads, which is
   also an eye-visible tell.
5. **Criterion 2 — eye-indistinguishability.** v100's measured closing move found
   the root of the adjudication frame chased since v73: **how the comment stops**
   was the thing, not the phrase. Real text closes on an abstract verdict 0.014
   of the time and v99 generated 0.265 (19.1×); v100 took the verdict-close
   vocabulary 0.271 → 0.150 on its gate. Still unfixed: **no generated comment
   contains a link** (real 0.051), `check` at ~10× real, entity diversity 0.438×
   real, `will` at ~1% of real, and generated hedges on 2.9% of replies against a
   real 17.6%.

### Known bugs, unfixed

- **Evaluation drops <2-word comments unevenly**, so `--exact-matched-thread-size`
  can still yield mismatched counts (24 generated vs 22 real on `post04_seed011`).
- **The slot distribution schedule is never persisted** to `discussion.json`, so
  `tone_length_fit` / `tone_length_joint` cannot be audited after the fact. The
  same is true of **`opener_type`**, which had to be recovered by grepping the
  saved Writer prompts to build the realization matrix above.
- **`--template-phrase-reuse-budget 4` is flat** and wrong at large thread sizes
  (real threads reach `uncertainty_frame` 7, 8, and 12).
- **`sentence_rhythm`'s digit cue produces a bare `0` or `1` where a person writes
  the word** — "0 verdict from me", "wrap 1 hand around it". 0.140 of v102 and
  0.151 of v101 comments against 0.071 in excluded real, and the real figure
  includes legitimate decimals ("0.1% of consumers") so the true ratio is worse
  than 2×. An eye-visible tell for criterion 2, unfixed.
- **`pair_count` in `matched_*_thread_scores.csv` is not the stance pair count.**
  It is `n(n-1)/2` from the pairwise metrics. Do not read it for
  `hard_disagree_rate`.
- **One Reddit post can sit under two product folders**, so a naive read of
  `data/raw/discussions/` double-counts pairs (1.24× on the matched set, 1.32×
  on the camera corpus). Dedupe by `(thread_id, reply_id)`.
  `generalized_card/analysis/politeness_diagnosis.py` does not.

### Recommended revert

`--no-story-scope sequence` produced **no metric benefit** (past tense
0.289 → 0.288, lexical breadth 15.95 → 16.20 against a real 21.02) and added new
repeated 4-grams. It did remove a genuine prompt contradiction — 247 of 532
prompts carried two mutually exclusive rules — so keep the contradiction fix, but
the default should go back to `tense`.

### v104 and v105 — executed since the text above was written

The "Next step" that used to be here (make the opener schedule respect the
root/reply conditional) was v104. It shipped, was offline-verified, and was
paid-gated on 2026-08-21: **the arms worked on their own terms and the metric
did not follow** — full result in `generalized_card/VERSION_LOG.md`'s v104
entry. Do not read the paragraph that used to occupy this spot as pending; it
is done.

**v105 (2026-08-22): a chain-scoped reply-novelty check for
`self_bertscore_mean_f1`**, diagnosed this session (§6.3 above, `docs/DECISIONS.md`
G3) and built as `--reply-novelty-scope {parent_only,chain}`. Offline-verified,
including self-test green on all four registered domains — **no paid gate has
run yet**. Full detail in `generalized_card/VERSION_LOG.md`'s v105 entry.

### Next step

**Gate v105.** Write predictions against the gate thread's own matched real
(not the pooled corpus — this project has mis-set that band twice,
`tasks/lessons.md`), then run the large-thread gate and N=10 per §4's loop.
**This needs the user to confirm which API credential to bill first** — the
prior session flagged this as unconfirmed and it still is.

**Still blocking N=150: the reporting standard.** 12 metrics × 2 tests at
α = 0.05 means a perfect generator passes all 12 simultaneously only ≈ 52% of the
time. That is the user's decision (§2, trap 5).

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
- ~~"the verdict suppression strips polite appraisal from long slots"~~ — real
  polite comments close that way at 0.010–0.029; the drop was 4/9 → 1/9, noise.
- ~~"negative markers at 3× real should be suppressed"~~ — the decomposition is a
  **+8.381 polite deficit against a −0.767 impolite excess**; generated uses
  *less* of the impolite vocabulary than real.
- ~~"`hard_disagree_rate` has never had a mechanism"~~ (the 2026-08-20 handoff) —
  superseded: two mechanisms are now measured, one of them causally.

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
| 2 | `generator_policy_version` + `generator_core_provenance` | the policy string and the SHA-256 of every pinned generation source, as they were at run time (58 at v102; read `CURRENT_GENERATION_CORE_NAMES`, do not quote a count from memory) |
| 3 | every arm value, plus `RUN_EXPERIMENT_FIELDS` resume verification | the exact configuration; a resume with changed parameters is rejected, so a tag can never mean two configs |
| 4 | `domain_profile_sha256` + `domain_profile_schema_version`, and `domain_profile.json` copied into the run directory | the measured shares the generation was conditioned on |
| 5 | `seed_pool` path + `sampling_seed` + `start_seed_index` | which real threads were matched, in which order |

### The gate

`generalized_card/generalized_card/source_provenance.py`, called from
`run_generate.py` immediately after `verify_core_contract` and **before the seed
pool, the domain profile, and every API call**:

> A generation run **refuses to start** if any file that defines the version is
> not in `HEAD`.

It checks the pinned generation sources **plus
`core_contract.py` itself**. The contract cannot carry its own hash, so
`verify_core_contract` can never check it, yet it is the file that names the
policy version and holds every other pin; a version whose contract is
uncommitted is not recoverable even when every module is.

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
| **the `hard_disagree_rate` diagnosis (2026-08-20)** | every figure in §6.1 is reproduced by `generalized_card/analysis/disagreement_diagnosis.py`; the subcommands are `structure`, `openers`, `echo`, `surrogate`, `ablate` |
| the stance scorer's semantics | read from `scripts/evaluation/score_thread_disagreement.py` end to end, not recalled |
| the ablation harness | re-scores with the evaluator's own scorer classes and reproduces the shipped v101 artifact on **526/526 pairs**, max \|Δp\| **0.000000**, asserted before any edited number is printed |
| the real stance tables | re-scored today in this environment and reproduce their stored labels exactly, so the real/generated comparison carries no environment drift |
| the opener realization matrix | recovered by matching the 11 `OPENER_INSTRUCTIONS` strings against the **532 saved Writer prompts** — `opener_type` is not persisted |
| the causal opener edit | reply-pair rate 0.2235 → **0.1862** stripping only the 36 unassigned polarity openers; `self_bleu_4` 0.03330 → 0.03297 on the same edit, exact scorer |
| **v102 offline gate (2026-08-20)** | 559 tests, Ruff clean, **105 pins with 0 drift**, backend self-test passes with `--opening-move` on **and** off |
| v102 profile rebuild | schema 18 → 19, 15,294 comments over 424 excluded threads, **0 seed overlap**; every other profile section byte-identical to v101's, so the gate is a single-arm comparison |
| v102 draw fidelity | rendered draw against measured share within **0.0108** in every register, entry type and token, 4,000 slots per cell |
| v102 on the real prompt path | the self-test block was proven to execute by injecting a temporary failure; it renders the drawn word per slot (`"ah"`, `"also"` on two slots) and the ten-token prohibition |
| v102 domain generalization | available on all four domains; cells 8/8 camera, 7/8 cell_phone, 5/8 headphone, 3/8 laptop — the sparse ones lose cells rather than receiving a wrong word |
| v102 `--prepare-only` on the real path | ran on a throwaway tag, recorded policy `...-v102-20260820`, `opening_move: measured`, `source_provenance.uncommitted: []`, 59 sources checked (v101: 58); tag then deleted |
| corpus deduplication | pairs deduped by `(thread_id, reply_id)`; a naive read inflates the matched set **1.24×** and the camera corpus **1.32×** |
| `pytest -q generalized_card/tests` | **527 passed** at the v100 boundary; **537 passed** at v101 (see `VERSION_LOG.md`) |
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
