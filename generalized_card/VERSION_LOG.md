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
4. **Commit the version.** `run_generate.py` refuses to start when any file that
   defines the version is missing from `HEAD` -- the 55 pinned generation sources
   plus `core_contract.py` itself. See "Reproducibility infrastructure" below.
5. Use a run tag containing the version number.
6. After evaluating, fill in the result and regenerate `RUN_INDEX.md`.

---

## N=10 gate — `--semantic-coverage-nonrepeat on` isolated (predictions, 2026-08-23)

No new policy version: the arm already exists (v108), verified offline
across all four domains, and confirmed on the seed-8 single thread with
the arm firing 186/186 times (`docs/DECISIONS.md` G23). This is the best
single-thread `self_bertscore_mean_f1` result of any mechanism gated this
session (+0.0183 -> +0.0139, four of five depth bins improved together).
Isolated on purpose: not stacked with `--digit-cue-guard`/
`--verdict-close-guard`, to keep attribution to this one arm clean.

**Baseline is the last true N=10 run** (`generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1`,
532 comments, `--start-seed-index 2 --sampling-seed 42 --max-posts 10`),
the correct J6 control since this arm did not exist then:

| metric | real | v103 (arm off) | gap | Cliff |
|---|---:|---:|---:|---:|
| `self_bertscore_mean_f1` | 0.4942 | 0.5097 | +0.0169 | 0.86 |
| `self_bleu_4` | 0.0278 | 0.0325 | +0.0040 | 0.40 |
| `hard_disagree_rate` | 0.1208 | 0.0920 | -0.0243 | -0.23 |
| `mean_story_probability` | 0.1266 | 0.1219 | -0.0184 | -0.14 |
| `emotion_entropy` | 1.5803 | 1.7092 | -0.0220 | 0.02 |

**Predictions, each a named way to be wrong:**

- `self_bertscore_mean_f1`: **narrower gap, plausibly not closed to PASS.**
  The single-thread result (-0.0044 on seed 8, 24% relative reduction) is
  the strongest single-thread signal this session, but G16 already showed
  a clean single-thread win on this exact seed (`--verdict-close-guard`'s
  check-variant, fully eliminated on seed 8) can fail to replicate at
  N=10 entirely. A repeat of that failure -- flat or worse at the pool
  level despite the seed-8 win holding up -- is a real, named possibility,
  not a strawman.
- `hard_disagree_rate`/`mean_story_probability`/`emotion_entropy`: **no
  confident prediction.** All three moved on the seed-8 gate in directions
  this arm has no mechanism to cause (it touches only the "already
  covered" instruction, nothing about stance, story framing, or emotion).
  Read as thread-level regeneration noise (`tasks/lessons.md`); expected
  to land close to the v103 baseline row once pooled over N=10 rather than
  repeat the single-thread swings.
- `self_bleu_4` must not measurably worsen -- the instruction asks for a
  new relation, not a stylistic change.
- Coverage-block-driven prompt-length growth should stay marginal (one
  sentence per slot past the first few), not scale with thread length.

**Commands:**

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1 \
  --domain camera --model gpt-5.4-mini \
  --base-url https://api.openai.com/v1 --api-key-env LLM_API_KEY \
  --pool-size 150 --max-posts 10 --posts-per-run 5 \
  --start-seed-index 2 --sampling-seed 42 \
  --semantic-coverage-nonrepeat on --resume

python3 generalized_card/scripts/run_evaluate.py \
  --tag generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1 \
  --metric-parallel 5 --resume
```

Expected cost: the last N=10 run at these pool parameters
(`generalized_card_camera_gpt54_v107_digit_verdict_n10_20260822_v1`, two
arms, one crashed-and-resumed attempt) cost $4.3909; this run has one
arm and no known crash risk specific to it, so plausibly somewhat lower,
in the $3.50-4.50 range.

**Before spending: confirm the arm actually fires**, the same $0 check
that would have caught the v1 bug --
`python3 -m pytest generalized_card/tests/test_generalized_card.py -k semantic_coverage_nonrepeat_reaches_both`
-- and after the run, grep the artifact's own `generation_records.json`
for the instruction string before reading any metric.

### Result — 2026-08-23. Confirms G16's shape: real pair-level effect, no pooled movement

Run `generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1`,
532 comments across 10 threads, **$3.5978**, 53.3 min. **Arm confirmed
firing 532/532 slots** (grepped `generation_records.json`'s saved
prompts for the instruction string) before reading any metric.

**Aggregate report:** `self_bertscore_mean_f1` still **FAIL** (MWU=0.0036,
KS=0.0021, Cliff=0.78, gap generated-vs-real +0.0188). `impolite_rate`
newly reads FAIL at N=10 (Cliff=0.71, MWU=0.008) where v103 read PASS —
expected under G4 (power, not a regression this arm caused; see below).
The other ten metrics PASS or are DESCRIPTIVE-adjacent noise, consistent
with G9.

**The aggregate Cliff (0.86 -> 0.78 vs v103) is not the right read** —
per `ORIENTATION.md` §2 trap 4, Cliff-vs-real is unpaired and confounded
by which ten templates were drawn. Both runs share the identical 10
matched seeds, so a **paired, same-seed comparison is available and is
the correct test**: for each of the 10 threads, `(v108 generated - matched
real) - (v103 generated - matched real)`, Wilcoxon on the 10 differences.

| metric | v103 mean gap | v108 mean gap | threads improved | Wilcoxon p |
|---|---:|---:|---:|---:|
| `self_bertscore_mean_f1` | +0.0155 | +0.0188 | 4/10 | 0.232 |
| `self_bleu_4` | +0.0046 | +0.0049 | 5/10 | 0.846 |
| `impolite_rate` | +0.1908 | +0.2004 | 5/10 | 0.734 |
| `polite_rate` | -0.1827 | -0.1306 | 2/10 | 0.164 |
| `hard_disagree_rate` | -0.0288 | -0.0098 | 5/10 | 0.510 |
| `mean_story_probability` | -0.0047 | +0.0032 | 3/10 | 0.432 |
| `emotion_entropy` | +0.1289 | +0.0011 | 8/10 | 0.131 |

**No metric moved with statistical credibility at the thread-pooled
level** (all p > 0.13). For the target metric specifically, `self_bertscore_mean_f1`'s
gap nominally *widened* (not significantly), and the split is 4
improved / 6 worsened — this is exactly the "flat or worse at the pool
level despite the seed-8 win holding up" failure mode the prediction
named before spending as a real possibility, and it happened: seed 8's
own gap did hold an improvement (v103 +0.0235 -> v108 +0.0213), just a
smaller one than the isolated single-thread gate reported (-0.0044 there
vs -0.0022 here), and the other 9 threads did not follow the same
direction.

**Pair-level depth-bin decomposition explains why** —
`bertscore_pair_diagnosis.py depth`, run against both this artifact and
the v103 N=10 artifact, fidelity-checked first (every one of the 20
recomputed thread means reproduces its shipped value to <1e-7):

| depth range | v103 excess (arm off) | v108 excess (arm on) | |
|---|---:|---:|---|
| [0,1) | +0.0040 | -0.0008 | improved (small n, root pairs) |
| [1,2) | +0.0004 | +0.0072 | worsened |
| [2,4) | +0.0174 | +0.0173 | flat |
| [4,7) | +0.0198 | +0.0196 | flat |
| [7,+) | **+0.0432** | **+0.0284** | **improved, 34% relative reduction** |

The deepest bin — exactly the population this mechanism targets, since a
long reply chain is where the "already covered" ledger accumulates the
most entries — moved by more than any single depth-bin move measured for
any mechanism gated this session, including G16's combined
`--digit-cue-guard`/`--verdict-close-guard` on the same bin
(+0.0432 -> +0.0346). But deep pairs concentrate in the one or two
largest threads, while `self_bertscore_mean_f1` is an **equal-weight
mean of 10 thread-level means**, not a pair-weighted pooled mean (G17) —
so a large real effect confined to a small thread-count share cannot
move the reported metric. This is the identical dilution G16 already
diagnosed for the digit/verdict guards, now confirmed a second time for
a structurally different mechanism (Writer-prompt instruction, not
plan-level or lexical-guard).

**Guardrails held.** `self_bleu_4` flat as required. `impolite_rate`'s
new FAIL is consistent with G4's own prediction (these metrics fail at
higher power) and shows no paired-comparison movement against v103
(p=0.734) — not an effect of this arm. `hard_disagree_rate`/
`mean_story_probability`/`emotion_entropy` all moved within thread-level
noise (p > 0.13), as predicted.

**Decision: default stays `off`.** This does not overturn `docs/DECISIONS.md`
G22 — the metric-level result is a second independent null, now for a
mechanism that reaches Writer realization directly rather than a
forbidden output-check (G20) or a Planner-side check (G21), which
strengthens rather than weakens the reading that `self_bertscore_mean_f1`
is closed at its current thread-equal-weight definition regardless of
mechanism category. Two things are worth carrying forward without being
built now: (1) `--semantic-coverage-nonrepeat on` earns the same
independent criterion-2 standing as `--digit-cue-guard`/
`--verdict-close-guard` (G12, G16) — it measurably suppresses literal
argument restatement in the deepest, longest reply chains, regardless of
whether the reported metric moves; (2) it hits a different depth bin
than G16's guards ([7,+) here vs [4,7)/[7,+) there, by a smaller amount
each) — stacking all three has never been isolated-tested together and
is a plausible next combination, not a spend decision made here.

Full analysis in `docs/DECISIONS.md` G24.

---

## N=150 — the paper's scale, never run on any version (planned 2026-08-25)

**Not run.** This section exists so the command and its rationale are on record
before any spend, per the same discipline every gate entry follows.

### Why now

The reporting standard was the stated blocker (`docs/ORIENTATION.md` §6, §2 trap
5) and the user selected **Holm–Bonferroni** on 2026-08-25 (J2, G51). Nothing
else blocks it.

More importantly, **every priority claim in this project is a projection from a
ten-thread window, and that window is not a miniature of the pool**:

| | N=10 window (seeds 2–11) | full pool (150) |
|---|---:|---:|
| comments | 532 | 5,974 |
| mean comments per thread | 53.2 | 39.8 |
| threads with ≥100 comments | 1 of 10 | **16 of 150** |
| share of all comments in those threads | — | **46%** |

Both failing pairwise metrics are thread-size sensitive — `self_bertscore_mean_f1`
is an equal-weight mean of thread means (G17) and `self_bleu_4` is length- and
size-dependent through its brevity penalty and smoothing (G27) — and
`polite_rate`'s deficit is localised to 25+ word comments (G25). So the N=150
bias can differ from the N=10 bias in either direction, and G42, G51 and G52 are
all simulations built on the N=10 estimate.

### Config

v110's arm list with **`--length-transfer v97`**. G49 did not promote `refit`
(the arm fired 532/532 and its own channel provably did not operate, G48), and
the paper's headline run should not carry a rejected arm. `--development-scope`
stays `long_only`: v111 is ungated, and §4 requires a large-thread gate before
N=10, let alone before this.

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag generalized_card_camera_gpt54_n150_20260825_v1 --domain camera \
  --model gpt-5.4-mini --base-url https://api.openai.com/v1 \
  --api-key-env LLM_API_KEY --pool-size 150 --max-posts 150 --posts-per-run 5 \
  --start-seed-index 0 --sampling-seed 42 --resume \
  --semantic-coverage-nonrepeat on \
  --evaluation-tier measured --downtoner-tag suppress \
  --partitive-reference suppress --opening-move measured \
  --closing-move measured --register-realization measured \
  --length-calibration measured --length-transfer v97 \
  --final-punctuation measured --route-ledger on \
  --sentence-rhythm measured --long-form-layout measured \
  --reddit-typography on --no-story-scope sequence \
  --tone-length-fit conditional --turn-frame adjudicative_only \
  --domain-claim selective --speaker-identity matched \
  --own-fact-license off --writer-prompt focused \
  --writer-route-lock own_words --social-contract-coherence on \
  --reply-sibling-visibility on
```

Then, free, on CPU:

```bash
python3 generalized_card/scripts/run_evaluate.py \
  --tag generalized_card_camera_gpt54_n150_20260825_v1 --metric-parallel 5 --resume
```

Cost, scaled from v110's measured 532 comments / $3.7599 / 58.5 min by the pool's
own comment count: **~$42 and ~11 hours**, resumable — `--resume` is checked
against `run_config.json`'s `RUN_EXPERIMENT_FIELDS`, so an interrupted run
continues and a changed one is refused.

### Predictions, written before spending

From `analysis/holm_state.py`, simulating each metric's current relative bias at
N=150 over the 763-thread real baseline, Holm computed with the conservative
Bonferroni bound:

| metric | P(pass) predicted | rel. bias |
|---|---:|---:|
| `polite_rate` | **0.00** | −51.5% |
| `impolite_rate` | **0.01** | +39.9% |
| `self_bleu_4` | 0.16 | +18.8% |
| `self_bertscore_mean_f1` | 0.19 | +2.6% |
| `neutral_rate` | 0.81 | −19.7% |
| the other five non-structural | 0.97–1.00 | ≤5% |

So the prediction is **7 or 8 of 12 PASS under Holm**, with the four above
failing and `neutral_rate` marginal. `avg_depth` and `structural_virality` pass
structurally (J9) and are not evidence the generator works.

**The prediction that matters is not the count.** It is whether the *measured*
N=150 biases match the N=10 ones. If they do, the simulation is validated and the
closure targets in G42/G51/G52 can be trusted for the first time. If they do not
— which the size-mix table above makes a live possibility — then every mechanism
priced against those targets, including v111's 8–26%, has to be re-priced.

### Result

**Not run yet.**

## v111 — extend the development beat plan to the band that compresses (2026-08-25)

Policy ID: `generalized-card-v2-development-scope-v111-20260825`.
Arm `--development-scope {long_only,measured}`, default `long_only`, which
reproduces v110 byte-for-byte. Module
`generalized_card/generalized_card/long_form_planning.py` (the budget) and
`length_policy.py` (the routing). Evidence: `docs/DECISIONS.md` G50; reproduce
with `analysis/length_instrument_rdd.py all` and `analysis/length_fix_pricing.py`.

### Why this and not something else

G48 killed the asked word count as a length instrument and named the structural
cues as the untested remainder, to be priced by a paid probe. They did not need
one. `length_calibration.py`'s own docstring says the calibration moves **only**
the number, so v110 held every structural cue fixed; and
`expected_development_beats` returns 0 at `real_word_count <= 100` and
`max(3, round(w/21))` above it, which is a **regression discontinuity** sitting
inside four runs already paid for.

| assigned words | realized/assigned | the cue that band receives |
|---|---:|---|
| 1–9 | 1.157 | one narrow local move |
| 10–34 | 0.994 | one narrow local move |
| **35–60** | **0.856** | one narrow local move |
| **61–100** | **0.840** | vague "two or three beats"; beat plan **deleted** |
| 101–251 | **0.956** | enumerated `round(w/21)` beats |
| **252+** | **0.785** | enumerated beats, capped at 12 |

The one band that receives an enumerated per-slot beat plan is the one band at
0.956. Across the 100/101 boundary, pooled over the four comparable N=10 runs,
assignment rises 11.9 words and realized length rises **24.2**; the ratio jumps
**0.816 → 0.953**, and the local-linear jump is +15.5 / +11.0 / +17.6 / +11.2
words in the four runs separately. **0 of 1,816** slot-instances at ≤100 assigned
words carry a beat plan, and above 100 the Writer delivers **21.3** realized
words per delivered beat against the module's own design constant of 21.0.
Against this, G48 measured the asked number's elasticity at −0.02 to 0.11.

The 61–100 prompt is self-contradictory as shipped: it tells a 90-word slot it
"contains roughly 118 words" and, in the same sentence, to "give it the two or
three connected beats this slot's scale supports" — about 42–63 realized words.
The categorical cue wins, which is E4's rule (a named concrete thing gets ~1.0
compliance, a named category 0.23).

**Priced at 8–26% of `self_bleu_4`'s gap**, with a 3× spread between estimator
families (cell reweighting at 5/10/20/40 bins gives 26.3/26.0/22.6/19.5%; the
continuous within-thread estimator gives 7.8%; full compliance everywhere would
give 22–55%). That is above the 5–10% class G42 retires and below one structural
lever. **Whether it is worth gating depends on the reporting standard** (G51):
under the shipped raw rule the N=150 bar is ~90% closure and this is not enough
alone; under J2's Holm–Bonferroni the bar is ~50–75% and a stack of length
(8–26%) + links (8.8%) + markdown emphasis (3.6%) + entity variety (≤9.4%) is a
credible route. **The user selected Holm–Bonferroni on 2026-08-25.**

**Not shipped, deliberately.** Lifting `MAX_DEVELOPMENT_BEATS` for the 252+ band
is 36.7% of the word deficit and worth only **+0.7pp** of the `self_bleu_4` gap,
because 13 of 532 comments carry few pairs — G45's arithmetic a fourth time. The
Planner is also measured to saturate near 9 beats, so its instrument would have
to be invented rather than extended.

### Predictions, written before the paid gate

Gate is seed 8 (post `i1o51h`, 186 comments), which sits inside the N=10 window,
so its row is directly comparable to v110's. Every number below is **this
thread's own**, not the pool (J6). On this thread v110 realizes 0.9132 overall;
66 of its 186 slots fall in the arm's 35–100 band at **0.812**, and the band that
already receives the cue sits at **0.966**.

**The mechanical audit comes first, and it decides whether any metric is read**
(G23, G48). It is free:

| audit | v110 | required |
|---|---:|---|
| slots in [35,100] carrying an enumerated development sequence | **0 / 66** | **66 / 66** |
| slots at ≤34 words carrying one | 0 | **0** (unchanged) |
| the string `two or three connected beats` in any prompt | 29 | **0** |

If the arm does not fire 66/66 the run is a wiring result, not a mechanism
result, and no metric from it may be quoted.

| measure | v110 | predicted | note |
|---|---:|---|---|
| realized/assigned, 35–100 band | 0.812 | **0.90 – 0.97** | the served band reaches 0.966 here |
| realized/assigned, whole thread | 0.9132 | **0.955 – 0.985** | arithmetic consequence of the above |
| realized/assigned, 1–34 band | 1.059 | **unchanged** | the arm returns 0 beats below 35 |
| realized words per delivered beat | 21.3 | **19 – 23** | if it falls, the Planner is padding beats |

Guardrails, each a named way to be wrong:

- **`self_bertscore_mean_f1` and `semantic_mean_cosine` must not worsen.** This
  is the G37 failure mode: v109's cue prescribed a shared speech act and treated
  slots converged. Measured here at matched realized length, comments with an
  enumerated beat plan score **+0.0007** self_bleu_4 above those without, 95%
  bootstrap CI **[−0.0024, +0.0040]** — not demonstrated and **not excludable**,
  and the upper bound is larger than the whole expected benefit. The cruder
  contrast against the 13 Planner-failure slots reads +0.0087, but that is a
  selected population, not a control. **This is the main risk in the release.**
- **`mean_story_probability` must not rise, read on the 35–100 subgroup and not
  only on the thread.** More beats is more narrative room, and G38 showed a
  thread-level guardrail conceals a subpopulation effect on a rate-drawn arm.
- **Plan-repair traffic must not explode.** `development_plan_problem` now fires
  for 35–100 slots, and v94 once spent 130 of 152 requests on repair. Read
  `logs/planning_quality.jsonl`; 2–5 beats is well inside the Planner's measured
  reliable range of ~9, so a large repair count means the request is malformed.
- **`length_cv` must not fall.** It passes today at 0.9624 against a real 0.9468,
  and moving realized toward assigned should raise thread spread toward
  assigned's own 1.333, not flatten it.
- **The 1–34 band must not move.** The arm is inert there by construction; a move
  means something other than the arm changed.

At n=1 the evaluator prints `DESCRIPTIVE` and no p-value is meaningful (§4), so
`self_bleu_4` is judged on direction only, against **this thread's** matched real
row.

### Offline state at the gate

693 tests (4 new, re-collected in 5 modules), ruff clean, 108 pins re-pinned with
the drift list exactly the four edited files, backend self-test **PASS on both
arm values**, `--development-scope long_only` proven to render the identical cue
on the real prompt path by unit test over 0–900 assigned words (differing set is
exactly `range(35, 101)`). Reachability audited before building, per G41: the
length cue is present in **532/532** saved v110 prompts including every
low-info-template slot, so the arm reaches **172/172** of its target band with no
template cap.

### Gate command — every arm v110 set, plus the one under test

Diffed against v110's own `run_config.json` (the G39 lesson: every arm defaults
to `off`, so naming only the new flag silently drops every previously-won arm).
`--length-transfer refit` is carried unchanged so exactly one field differs,
even though G49 measured it inert.

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag v111_development_scope_seed8_20260825_v1 --domain camera \
  --model gpt-5.4-mini --base-url https://api.openai.com/v1 \
  --api-key-env LLM_API_KEY --pool-size 150 --max-posts 1 --posts-per-run 1 \
  --start-seed-index 8 --sampling-seed 42 --resume \
  --development-scope measured \
  --length-transfer refit --semantic-coverage-nonrepeat on \
  --evaluation-tier measured --downtoner-tag suppress \
  --partitive-reference suppress --opening-move measured \
  --closing-move measured --register-realization measured \
  --length-calibration measured --final-punctuation measured \
  --route-ledger on --sentence-rhythm measured --long-form-layout measured \
  --reddit-typography on --no-story-scope sequence \
  --tone-length-fit conditional --turn-frame adjudicative_only \
  --domain-claim selective --speaker-identity matched \
  --own-fact-license off --writer-prompt focused \
  --writer-route-lock own_words --social-contract-coherence on \
  --reply-sibling-visibility on
```

### Gate result

**Not run yet.**

## v110 — refit the length transfer function (2026-08-24)

`--length-transfer {v97,refit}`, default `v97`. Also in the tree at default
`off`: `--length-fidelity {off,measured}`, built and tested but **not** in this
gate (see "Why the band gate is not shipped" below).

### Why this and not something else

The user's target is `p ~ 0.5-0.6`, not `p > 0.05`. Simulated over the 763 real
camera threads (`docs/DECISIONS.md` G42) that needs **~90% gap closure at N=150,
~75% at N=50, ~50-75% at N=10** -- which rules out the whole 5-10% class the last
four releases came from. Every candidate was therefore priced before anything was
built:

| candidate | priced at | verdict |
|---|---|---|
| length composition | **31-35% `self_bleu_4`, 14-18% `self_bertscore`** | build (G43, G46) |
| absent links / markdown emphasis | 12.4% / 14.6% | defer, higher risk |
| full Planner de-duplication | <=2.4% / <=1.8% | **killed** (G45) |
| entity variety | <=9.4%, saturating | **killed** (G40, and v109 measured no movement) |
| "generated writes fewer, longer sentences" | 15.15 vs 15.54 words/sentence | **killed, my own hypothesis** (G44) |
| `no end punctuation` as a tell | generated has MORE (54 vs 34) | **killed** (G44) |
| seven other surface features | generated matches or exceeds real | **killed** (G44) |

### The mechanism

Not new machinery -- a corrected constant. `length_calibration` already inverts a
fitted transfer function, but its constants regress realized words on the
**uncalibrated** ask, which was only true of v97. Refitting the object that
governs the current system over **1,436 slots from four runs (21 thread
instances)** gives `log(realized) = 0.5580 + 0.8276 * log(asked)`, R2 0.879,
against the shipped 0.3835 / 0.8925, and the residual the old constants leave is
stable across all four runs: **1.64x below 10 asked words, 0.68-0.80x above 80**.

Effect on the asks for this gate thread, verified offline:

| assigned | n | v97 asks | refit asks | v109 realized |
|---|---:|---:|---:|---:|
| 1-9 | 24 | 5.2 | **4.7** | 8.9 |
| 10-24 | 52 | 17.0 | 16.8 | 18.3 |
| 25-49 | 53 | 35.0 | 37.5 | 35.2 |
| 50-99 | 40 | 79.5 | **91.0** | 59.8 |
| 100+ | 17 | 164.6 | **199.5** | 123.2 |

Two dependent fixes ride with it: the ask-multiplier clamp would bind at 0.51x
and 1.61x inside the refit's own range, and `writer_provider_token_budget` raised
the provider ceiling only above 100 assigned words, which the larger asks would
overrun. The revised ceiling guard is **proven a no-op for every target 1-100
under the legacy constants**, so `--length-transfer v97` still reproduces v109
byte-for-byte.

### Why the band gate is not shipped

`length_fidelity` (require the realized count to stay in its assigned measured
decile band) was built first and is in the tree, tested, and priced. It is not in
this gate because the measured miss is **~0.5 band**: on v109, 42.5% of slots land
in the exact band, 45.2% miss by one, 8.1% by two. A +/-0 tolerance therefore
flags **57.5%** of slots -- mostly near-miss noise, roughly doubling cost through
retries and putting one prescriptive instruction into more than half the prompts,
which is the convergence failure mode G37 measured for v109's cue. A +/-1
tolerance flags 12.4% and misses the bias entirely. Band membership is the wrong
instrument for a half-band bias; a corrected ask is the right one, and it costs no
extra calls and adds no instruction.

### Gate predictions, written before spending

Gate thread: seed 8 / `i1o51h`, 186 comments, the standing gate seed. Baseline is
**v108 v2** for both metrics, not v109 -- v109 carried the `entity_spread` arm
that G37 showed caused its `self_bertscore` and cosine regressions, and per G47
v109's untreated subgroup is not a usable `self_bleu_4` baseline because the arm
shifted comment length by ~15 words and that metric is length-dominated.

| what | baseline | predicted |
|---|---:|---|
| **realized/assigned words, total** | 0.916 | **0.98-1.02.** Mechanical, free, needs no metric |
| realized/assigned, assigned 50-99 | 0.82 | **~1.00** |
| realized/assigned, assigned 1-9 | 1.44 | **~1.00** |
| `length_cv` | 0.847-0.857 | **rises toward real 0.895** |
| comments over 100 words | 0.059 | **rises toward real 0.091** |
| `self_bleu_4` gap | +0.005664 | **+0.0037 to +0.0040** (31-35% closure), J7-discounted |
| `self_bertscore_mean_f1` gap | +0.0139 | **+0.0114 to +0.0120** (14-18% closure), J7-discounted |
| `mean_story_probability` | PASS | **guardrail: must not rise.** Longer comments are the v67 risk |
| `semantic_mean_cosine` | +0.0117 | **guardrail: must not rise materially** |
| cost | $1.1785 | **~$1.25** (assigned words realized in full is +9% output) |

**Read the mechanical audit before any metric.** If realized/assigned has not
moved off 0.916, the arm did not reach the live prompt and every metric in that
run is void -- the check the wasted v108 v1 gate paid $1.19 to learn (G23).

Per G42 a 31-35% closure puts `self_bleu_4` at MWU ~0.41-0.49 at N=10 and
`self_bertscore` at ~0.30. **That is not the 0.5-0.6 target**, and this release is
not pitched as reaching it; it is the largest verified lever available and the
first one aimed at a measured cause rather than a plausible story.

### Offline state

673 tests pass (6 new: the legacy arm reproduces every shipped ask, the refit
inverts exactly and its clamp never binds in 1-250, the provider ceiling clears
the refit ask and is a legacy no-op, and three for the unshipped band gate). Ruff
clean. Core contract: 0 missing, 0 untracked, 0 unpinned, 0 drifted.
`--prepare-only` passed with no API calls. `length_fidelity` profiles build on all
four registered domains with genuinely different decile cuts (camera
[6,11,16,22,29,38,52,72,111], headphone roughly half that), every band above the
40-comment floor. **Caveat on domain adaptivity:** the transfer function is a
property of the model and the prompt, not of the domain, so it ships as a recorded
constant rather than a profile -- and per D3 no paid run has ever been done off
camera, so that claim is untested there.

### N=10 gate instead of the single thread — predictions, written before spending

Run at the user's request, for a directly readable pooled result. Seeds 2-11, the
same cohort as the **v108 N=10** run, which is therefore the baseline and differs
by exactly one arm. `--semantic-coverage-nonrepeat on` is carried purely so the
comparison stays single-armed; G39 measured that arm at ~0.0007 on seed 8 and G24
found no pooled N=10 improvement from it, so it is not expected to contribute.

**v108 N=10 baseline (seeds 2-11, 532 slots):**

| metric | real | generated | gap | MWU | KS | Cliff |
|---|---:|---:|---:|---:|---:|---:|
| `self_bleu_4` | 0.0278 | 0.0327 | **+0.0049** | 0.121 | 0.418 | +0.42 |
| `self_bertscore_mean_f1` | 0.4942 | 0.5130 | **+0.0188** | **0.004** | **0.002** | +0.78 |
| `semantic_mean_cosine` | 0.2892 | 0.2794 | -0.0099 | 0.678 | 0.994 | -0.12 |
| `impolite_rate` | 0.4041 | 0.6045 | +0.2004 | 0.008 | 0.012 | +0.71 |
| `polite_rate` | 0.3085 | 0.1778 | -0.1306 | 0.089 | 0.168 | -0.46 |
| `neutral_rate` | 0.1715 | 0.0849 | -0.0866 | 0.054 | 0.418 | -0.52 |
| `length_cv` | 0.9468 | 0.9165 | -0.0303 | 0.850 | 0.994 | +0.06 |
| `hard_disagree_rate`, `avg_depth`, `structural_virality`, `mean_story_probability`, `emotion_entropy` | | | | 0.60-1.00 | | |

**Length baseline over the same 532 slots: realized/assigned = 0.8896**, worse
than seed 8's 0.9161 — [1,10) 1.245, [10,25) 0.979, [25,50) 0.951,
**[50,100) 0.842**, [100,+) 0.870. The refit raises asked words from 1.134x
assigned to **1.318x**.

**Predictions.** The mechanism controls the gap; the p-value is a consequence, and
at N=10 it is noisy enough that only the gap is worth predicting as a number.

| what | baseline | predicted |
|---|---:|---|
| **realized/assigned words, total** | 0.8896 | **0.97-1.02.** Free, mechanical, no metric needed |
| realized/assigned, assigned 50-99 | 0.842 | **~1.00** |
| realized/assigned, assigned 1-9 | 1.245 | **~1.00** |
| `length_cv` | 0.9165 | **rises toward real 0.9468** |
| `self_bleu_4` gap | +0.0049 | **+0.0032 to +0.0034** (31-35% closure) |
| `self_bleu_4` Cliff | +0.42 | **falls**; MWU rises off 0.121 |
| `self_bertscore` gap | +0.0188 | **+0.0154 to +0.0162** (14-18% closure) |
| `self_bertscore` Cliff | +0.78 | **falls**, but this metric will still fail |
| `mean_story_probability` | +0.0032, MWU 0.970 | **guardrail: must stay passing.** Longer comments are the v67 risk |
| `semantic_mean_cosine` | -0.0099, MWU 0.678 | **guardrail: must stay passing** |
| the six currently-passing metrics | MWU 0.60-1.00 | **guardrail: none may drop below 0.05** |
| cost | $3.5978 | **~$4.0-4.2** (+12% realized output words) |

**What this release will and will not achieve**, simulated at the actual measured
biases over the 763 real threads. `self_bleu_4`'s N=10 relative bias is +17.6% and
`self_bertscore`'s is +3.8%:

| closure | `self_bleu_4` MWU at N=10 / 50 / 150 | `self_bertscore` MWU at N=10 / 50 / 150 |
|---|---|---|
| 0% (now) | 0.40 / 0.12 / 0.01 | 0.29 / 0.02 / 0.00 |
| **31-35% (predicted)** | **0.44-0.46 / 0.23-0.30 / 0.05-0.08** | **0.38-0.39 / 0.10-0.14 / 0.00-0.01** |
| 75% | 0.51 / 0.47 / 0.36 | 0.50 / 0.40 / 0.23 |
| 90% | 0.56 / 0.52 / 0.50 | 0.50 / 0.50 / 0.48 |

So this release is expected to move `self_bleu_4` into the target band **at N=10
only**, and to leave `self_bertscore` short of it at every N. Stated plainly
before spending so the result cannot be oversold afterwards. Note also that a
0%-closure N=10 draw has an *expected* MWU of 0.40 for `self_bleu_4` while the
v108 run observed 0.121 — N=10 is noisy in both directions, so the observed
improvement will contain a draw effect that only N=50+ can separate. Choosing the
reported N to improve a p-value is forbidden by `docs/ORIENTATION.md` §4; this
N=10 run is a gate, not the paper's reported scale.

### Command — N=10

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag v110_length_transfer_n10_20260824_v1 --domain camera \
  --model gpt-5.4-mini --base-url https://api.openai.com/v1 \
  --api-key-env LLM_API_KEY --pool-size 150 --max-posts 10 --posts-per-run 5 \
  --start-seed-index 2 --sampling-seed 42 \
  --length-transfer refit --semantic-coverage-nonrepeat on --resume

python3 generalized_card/scripts/run_evaluate.py \
  --tag v110_length_transfer_n10_20260824_v1 --metric-parallel 5 --resume
```

---

### Command — single thread (kept for the record)


```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag v110_length_transfer_seed8_20260824_v1 --domain camera \
  --model gpt-5.4-mini --base-url https://api.openai.com/v1 \
  --api-key-env LLM_API_KEY --pool-size 150 --max-posts 1 --posts-per-run 1 \
  --start-seed-index 8 --sampling-seed 42 \
  --length-transfer refit --resume

python3 generalized_card/scripts/run_evaluate.py \
  --tag v110_length_transfer_seed8_20260824_v1 --metric-parallel 5 --resume
```

Full arm list for this gate, per G39: every other arm at its default. The only
non-default is `--length-transfer refit`.

---

## v109 — per-slot referent spread (2026-08-24)

Policy ID: `generalized-card-v2-entity-referent-spread-v109-20260824`.
Arm `--entity-spread {off,measured}`, default `off`. Module
`generalized_card/generalized_card/entity_spread.py`, wired through
`domain_profile.py` (measured profile), `backend.py` (arm + profile install)
and `prompts.py` (one shared helper reaching both writer-prompt paths).

**The defect, measured** (`docs/DECISIONS.md` G35). Per matched thread on the
v108 N=10 artifact: real names **40.8** distinct equipment designators,
generated **7.4**; real's most frequent designator takes **0.152** of that
thread's mentions, generated's **0.485**. Pooled 302 against 67; one generated
thread names zero; and the concentration has *degraded* across releases
(top share v98 0.190 -> v103 0.214 -> v108 0.266). This is the measured form of
the goal's own framing -- a real thread wanders, a generated one trends to one
topic.

**Why the existing machinery could not reach it.** `entity_inventory.py`
already builds a held-out designator vocabulary and `slot_equipment_options`
already rotates it per slot. Its only consumer, `prompts._own_equipment_block`,
is gated on `own_fact_license in {own, named}` **and**
`_first_person_experience_slot`. Measured over **18,829 designator mentions** in
the evaluation-excluded corpus: **14.0%** possession context, **8.9%** explicit
comparison, **77.1%** bare -- so **86.0% of real entity mentions need no
first-person frame**, and that gate can only reach the smallest slice. It is why
the two paid runs that did enable `named` (v97, v98) still landed at 81 pooled
designators against a real 302.

**Why the gate was not simply widened.** Offering *owned gear* to a slot not
planned for personal experience is a measured regression: v67 moved
`mean_story_probability`'s Cliff from 0.06 to 0.26 (gaps up to +0.19) because
own-gear anecdotes on `no_story` slots produce text StorySeeker reads as
narrative, and v88 deleted invented kit/tenure biography for the same class of
reason. v109 offers a **bare comparison referent** -- the 86% case -- never a
possession, never a claim about the seed, and leaves `_own_equipment_block`
untouched.

**Priced before it was built (J7).** Exact ablation on the real scorer, in the
direction of the fix: raising generated variety from 7.4 to 13.0 distinct
designators (top share 0.485 -> 0.297) closes **5.4%** of `self_bleu_4`'s
+0.00489 gap. Collapsing *real*'s variety to a single designator costs real
16.9%, so the relationship is real but asymmetric. **This is not shipped as a
`self_bleu_4` fix.** Per G35 no single large lever exists for that metric and
the correct shape of work is several stacked fixes of about this size.

### Gate predictions, written before spending

Gate thread: seed 8 / `i1o51h`, 186 comments (the standing gate seed, `large`
band, measured `distinct_per_comment` 0.63). Baseline is the v108 artifact.

| what | v108 baseline | predicted | why not a precise number |
|---|---:|---|---|
| realized distinct designators, this thread | 21 | **rises materially**; real for this thread is 118 | the offer is drawn per slot and the Writer may decline it; compliance for a comparable in-prompt offer runs 0.33 (G31) |
| top designator share | 0.652 | **falls** toward real's 0.139 | same |
| `self_bleu_4` | +0.0062 vs real | narrows slightly, **~5% of the gap at best** | that is the measured ablation upper bound, and J7 says discount it |
| `mean_story_probability` | PASS | **guardrail: must not rise.** This is the v67 regression this design exists to avoid | the cue forbids possession and narrative explicitly; if story probability rises anyway the cue wording is wrong, not the mechanism |
| `self_bertscore_mean_f1` | +0.0139 | **no prediction** | nothing in this mechanism targets it |
| `semantic_mean_cosine` | PASS | **guardrail: must stay inside its band** | naming more outside entities could push topical spread past real |

**Before reading any metric, grep the run's own `generation_records.json` for
`"Other things in this space you may name"` and confirm it appears at roughly
the drawn rate** -- the free check the wasted v108 v1 gate paid $1.19 to learn
(G23).

**Offline state.** 638 tests pass (3 new: both writer-prompt paths through the
real dispatch with a real profile file, draw determinism and rate, and a
domain-vocabulary check on the cue text). Ruff clean on all shipped code. 5 pins
recomputed, drift exactly the 4 edited files plus the new module. Arm `off`
renders 0 characters. The measured profile builds on **all four registered
domains** with genuinely different rates (large band: camera 0.63, cell_phone
0.33, headphone 0.28, laptop 0.43), so the mechanism is domain-adaptive by
construction and a band with no data withholds the cue rather than defaulting.
**No paid run yet.**

Command:

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag v109_entity_spread_seed8_20260824_v1 --domain camera \
  --model gpt-5.4-mini --base-url https://api.openai.com/v1 \
  --api-key-env LLM_API_KEY --pool-size 150 --max-posts 1 --posts-per-run 1 \
  --start-seed-index 8 --sampling-seed 42 \
  --entity-spread measured --resume

python3 generalized_card/scripts/run_evaluate.py \
  --tag v109_entity_spread_seed8_20260824_v1 --metric-parallel 5 --resume
```

The domain profile is rebuilt automatically into `<run_root>/domain_profile.json`
on every run, so `entity_spread_profile` needs no separate step. Verified with
`--prepare-only` before spending: all four bands measured, 424 reference
threads, 182 inventory terms, `seed_reference_overlap_count: 0`.

### Gate result — 2026-08-24. Mechanism works, and it is net-negative on the two priority metrics

Ran: $1.1785, 350 requests, 25.8 min, 186 comments. Arm audited before any
metric was read: the block appears in the run's own prompts.

**Scorecard against the predictions above.** The paired seed-8 comparison is
J6-legal for the columns marked v108, with the confound in the last row.

| prediction | outcome | verdict |
|---|---|---|
| distinct designators rise materially | **21 -> 69** (real 118) | **hit**, largest move on this property of any release |
| top designator share falls | **0.6889 -> 0.5031** (real 0.1394) | **hit**, and monotone best of five releases |
| `self_bleu_4` narrows ~5% at best | **+0.0057 -> +0.0062** | **missed**: no movement, mid-range of the v104-v108 band (+0.0057..+0.0070) |
| `mean_story_probability` must not rise | thread **+0.0159 -> -0.0027**, but treated slots **0.1298 vs 0.0884** untreated | **failed at the mechanism level**; the thread-level pass concealed it (G38) |
| `semantic_mean_cosine` must stay in band | **+0.0117 -> +0.0304**, worst of five | **breached**, and causally this arm (G37) |
| `self_bertscore_mean_f1` no prediction | **+0.0139 -> +0.0241** | regression, causally this arm (G37) |

**What the arm actually bought (G36).** Mentions per distinct name
**4.286 -> 2.333** against real **2.432** -- the naming *shape* defect G35
measured is closed almost exactly. Pooled 3- and 4-gram precisions moved from
above real to **below** it (1.123x, 1.243x -> 0.951x, 0.930x) and the pooled log
excess fell 1.2018x -> 1.0771x.

**What it cost, established by randomised within-run contrast (G37).** The draw
keys on `local_task_id`, which is Planner traversal order and independent of the
slot's plan, so fired-vs-not is a treatment effect rather than the between-thread
slope G35 retracted. `self_bertscore` pair F1: neither **0.5033**, one
**0.5101**, both **0.5289**. `semantic_mean_cosine`: neither **0.1889**, one
**0.2144**, both **0.2525**. Real is 0.4887 and 0.1865, so **the untreated half
of this thread sits at real and the treated half carries the whole excess.**
Length-stratified, the cosine effect survives at +0.0443 / +0.0582 / +0.0167 /
+0.0461, so the ~15 extra words per treated comment explain about a third;
inventory partitioning would recover only 16%; the rest is the cue's own
prescribed speech act.

**A command error, recorded (G39).** The gate command names only the new arm, and
every arm defaults to off, so it also turned **`--semantic-coverage-nonrepeat`
off**. Two experiment fields differ between v108 v2 and v109, not one. The
within-run randomisation is unaffected and also prices v108's arm: v109's
untreated subgroup sits at 0.5033 against v108's whole-thread 0.5026, so
**v108's arm is worth ~0.0007 on this thread** and its "best result this
session" was noise. Every future gate command must carry every previously-won
arm explicitly.

**Decision.** `--entity-spread` stays in the tree with default `off`; it is not
promoted. The repair is a v110 that keeps the naming-shape win and removes the
convergence cost, by three evidence-derived changes: deliver the drawn referent
through the existing `concrete_anchors` list with **no cue text at all** (the
anchors block already renders per-slot-distinct names and prescribes no speech
act), forbid any added sentence, and withhold it from `no_story` slots. The
`self_bleu_4` work moves off entities entirely -- G40 prices the last entity
mechanism at <=9.4% and saturating, and localises the residual to a
clause-structure signature (`the` carries 20.9% of the 1-gram excess mass; `.`,
`to`, `i`, `of`, `with`, `for`, `and`, `but`, `be`, `have`, `are` are all
under-shared).

Artifacts and scripts:
`artifacts/generalized_card/runs/v109_entity_spread_seed8_20260824_v1`,
`generalized_card/analysis/entity_spread_gate_audit.py` (rate, naming, shape,
cosine, bertscore, labels, mediation -- each fidelity-checked against the
shipped metric first), `generalized_card/analysis/subject_rename_ablation.py`.

---

## REJECTED BEFORE BUILD — v109 / v110 (2026-08-24)

**Both mechanisms below were killed by the zero-cost falsification step and
were never built.** Full evidence in `docs/DECISIONS.md` G34. Kept in full
because the rejected hypothesis is why the diagnosis exists, and because the
proposal text below records what was predicted before the test ran.

- **v109 (seed-anchor taper)**: on the real scorer the anchoring->BERTScore
  slope is 0.0246 (r=+0.032), 20-30x weaker than on the mpnet cosine proxy
  that suggested it. Upper bound 4-7% of the excess; zeroing anchoring
  entirely buys 8%. Below N=10 resolution after J7 discounting.
- **v110 (syntactic/function-word register)**: measured on the exact scorer
  over 209 excluded real threads. Clause-form spread has partial r=-0.003
  controlling length (the raw -0.415 is a pure length confound), and the
  better-powered function-word L1 spread has partial r=**+0.137** -- the
  wrong sign. The fix would make `self_bleu_4` ~28% worse.

The underlying measurements (G31, G32, G33) stand as verified criterion-2
tells. They are not what these two metrics measure.

## The original proposal, kept for the record

Not built, not gated. Written before any code so the predictions are on
record (J7, and the "write predictions before spending" rule). Evidence in
`docs/DECISIONS.md` G31/G32.

### Why these two and not another surface patch

G26 showed both mechanisms ever built for `self_bertscore_mean_f1` acted on
`[7,+)`, which carries 11.9% of the defect; `[2,4)+[4,7)` carry 82.7% and
had never been targeted. G28 then showed input *separation* between slots
already improves with depth and does not transmit, so the remedy is not
"separate the plans more." G31 found what is actually shared instead: a
**common attractor**. Real reply chains drift off the seed post (anchoring
0.1404 -> 0.0336, a 4.2x fall by depth `[4,7)`); generated chains do not
(0.1058 -> 0.0643, then rising), and generated is more anchored in 6/6
threads at depth>=4 (p=0.0312). Pooled over depth the two are identical
(0.0765 vs 0.0775), which is why eleven versions of thread-level analysis
missed it.

### v109 — `--seed-anchor-taper {off,measured}`, Planner-side

**Mechanism.** Taper each slot's seed-derived material by depth, in the
plan the Planner writes, not in the Writer's output. Two edits:
`concrete_anchors` stops offering seed-derived designators at depth>=2
(preferring entities the branch or parent introduced, or the held-out
inventory), and the prose fields' seed vocabulary is tapered on the same
schedule. Rates measured per depth bin from each domain's own
evaluation-excluded threads, so it is domain-adaptive by construction (D1-D4).

**Why permitted.** It edits Planner-authored plan fields. G20 forbids
Writer *output* selection against a metric-shaped band; this is the
`reply_increment_problem` category, not that one.

**The defect it fixes is real regardless of the metric.** `concrete_anchors`
seed-share currently *rises* with depth -- 0.3539 root, 0.4319 `[2,4)`,
0.4379 `[4,7)` -- the opposite of what real text does. Anchors are live:
0.800 of them reach the prompt, 0.332 reach the text.

**Predictions, each a named way to be wrong.**

| what | now | predicted | why not a number |
|---|---:|---|---|
| depth>=4 text seed-anchoring | +0.0769 over real | falls toward real; 6/6 -> majority at parity | the fix removes one channel; the plan prose carries the rest, so partial closure is the honest expectation |
| `self_bertscore_mean_f1` `[2,4)`+`[4,7)` pair excess | +0.0173 / +0.0196 | **narrows; no confident magnitude** | first mechanism ever aimed here; anchor in-text compliance is only 0.237 at `[4,7)`, so the ceiling is bounded |
| thread-level `self_bertscore_mean_f1` | gap +0.0188 | may still not move at N=10 | G16/G24 both showed real pair-level effects diluted by the equal-weight-thread-mean definition (G17). **A pair-level win with a flat thread metric is the expected outcome, not a failure** |
| `semantic_mean_cosine` | PASS, Cliff -0.12 | **guardrail: must not fall further** | reducing shared vocabulary could push topical spread past real; this is the metric most at risk |
| `self_bleu_4` | +0.0049 | may improve slightly | fewer repeated seed designators removes some 1-gram mass, but G27 showed entity effects are weak |

**Required before spending:** an offline ablation on the v108 artifact --
strip seed-derived anchors from depth>=2 slots, re-measure the pair excess
-- and per J7 the paid prediction is that number **discounted**, not that
number.

### v110 — `--syntactic-register {off,measured}`, for `self_bleu_4`

**Mechanism.** A per-slot draw over *clause form* -- direct question,
sentence fragment, imperative, conditional, first-person past -- at rates
measured per length band and per speaker from excluded real threads. Same
E3 shape as `sentence_rhythm.py`: one module, one arm, measured profile,
per-slot SHA-256 draw, realized-rate audit.

**Why this and not more phrase suppression.** G27: `self_bleu_4` is the
geometric mean of 1-4 gram precisions, p1+p2 carry 63% of the gap and are
the only stable part across versions, so every phrase-level fix targeted
the terms that move least. G32: generated comments' function-word profiles
are measurably less varied than real (L1 spread 0.4646 vs 0.5342, 9/10
threads, p=0.0039), and the over-shared tokens are precisely function words
and punctuation while the under-shared are the conversational verbal core
(`to` `i` `be` `have` `will` `would`). Clause form is what moves function
words; nothing shipped controls it (`sentence_rhythm` is punctuation-only,
`surface_texture`/`real_surface_shape` are layout, `opening_style` is
free-text with 529 distinct values over 532 slots).

**Prediction.** Function-word L1 spread rises toward real's 0.5342;
`self_bleu_4`'s p1/p2 ratios (1.147/1.215) fall. Whether the thread metric
follows is **not** predicted -- it is currently a weak PASS (MWU 0.12) and
the honest framing is that this attacks the component that actually carries
the gap, for the first time.

**Guardrail.** Must not raise `mean_story_probability` (first-person past
is a story-adjacent form) and must not disturb the tone metrics.

### Also fix, as correctness not as a metric play

Three controls are silently inert: `writer_temperature` is computed and
discarded on every paid run (G29), `sentence_route` is empty on all 532
slots while `--route-ledger` defaults `on` (G32), and `REPORTED_TONE_CLASSES`
is defined and never referenced. Per ORIENTATION's rule that an accepted-but-
unconsumed control is a correctness bug, each should be wired or deleted --
and G29 records why enabling the temperature values would *reduce* diversity,
so that one is a deletion, not an activation.

---

## v108 — semantic-coverage non-repeat instruction (2026-08-23)

Policy ID: `generalized-card-v2-semantic-coverage-nonrepeat-v108-20260823`.
Arm `--semantic-coverage-nonrepeat {off,on}`, default `off` (byte-for-byte
v107 and earlier -- verified by inspection: the coverage block plus one
blank line before the next header is unchanged when off). No domain-profile
change. Module `generalized_card/generalized_card/prompts.py`
(`_thread_memory`, `SEMANTIC_COVERAGE_NONREPEAT_INSTRUCTION`), wired through
`backend.py` and `run_generate.py`.

**Why, and a different category from G20/G21/G22.** `docs/DECISIONS.md`
G22 closed `self_bertscore_mean_f1` to further *checks* -- a Writer-output
similarity gate is forbidden (G20, `docs/ORIENTATION.md` §4), and widening
the Planner-side plan-similarity check further has a low ceiling and
evidenced downside (G21). This is neither: it changes what the Writer's
*prompt* contains before the one generation call, the same category as
every existing cue-text arm (`digit-cue-guard`, `verdict-close-guard`), not
a check-and-reject loop.

The Writer prompt already carries a "thread memory" block
(`_thread_memory`) with four parts: recent comments, short utterances
already used (with "do not repeat" attached), **semantic contributions
already covered** (no instruction attached), and sentence/clause routes
already used (with "do not reuse" attached). Read against the real
seed002 chain that restates "compactness doesn't matter once it's in a
bag" six times (v103 N=10, comments 40→45): comment 45's own actual
coverage block, recomputed with the real `semantic_coverage_entries`
function, **already listed all five earlier near-paraphrases verbatim** --
the information was present, nothing told the Writer what to do with it,
and it restated the point a sixth time anyway. The coverage block is the
one of the three "already used" blocks with no instruction attached to
it, unlike its two siblings.

**The fix.** One instruction appended after the coverage block, matching
the style already used by its two siblings:

> Do not restate one of these already-covered points in different words.
> Add a genuinely new relation, consequence, caveat, or evidence type
> beyond what is listed here.

No domain vocabulary. Does not touch `semantic_coverage_entries`'s
selection logic (still lexical relevance, the same limitation G13 already
found in the sibling `used_sentence_routes` mechanism -- a real, separate,
not-yet-acted-on lead: a paraphrase that shares few literal tokens with
the current slot could still be dropped from the capped list before this
instruction ever gets a chance to apply to it).

**Offline state.** 608/609 tests pass (5 new; the one failure is the live
provenance guard while this version sits uncommitted, clears on commit),
Ruff clean, 3 pins re-computed with 0 unexpected drift (`prompts.py`,
`backend.py`, `run_generate.py`), backend self-test on and off for both
arm values across all four registered domains (8 runs, $0, all exit 0).
**No paid gate yet.**

### Large-thread gate — predictions, written before the paid run

Gate thread: seed 8 / `i1o51h`, 186 comments, the project's standing gate
seed. Baseline is the pure "everything off" artifact
(`v104_evaluative_seed8_20260821_v1`), the correct J6 control.

**What replaying the real coverage-block builder shows on this exact
baseline artifact** (measured, not assumed): of the 186 comments, every
comment past the first few already receives a non-empty coverage block:
the mechanism has something to say to it. Whether the *specific*
already-covered points shown are close enough to what a given slot is
about to write, and whether the Writer complies with the new instruction
when they are, is exactly what this gate tests -- there is no cheap offline
proxy for "does the Writer follow a one-sentence instruction it wasn't
following implicitly," the same reason `abstract_verdict_close`'s
existing cue still under-complies 15.2% of the time even when delivered
(G14).

**Predictions, each a named way to be wrong:**

| what | v104 baseline | predicted direction | why not a precise number |
|---|---:|---|---|
| `self_bertscore_mean_f1` gap vs real | +0.0183 | **no confident prediction; genuinely uncertain, unlike the last three arms** | this is the first mechanism this session that reaches Writer realization directly rather than the Planner or a forbidden output check -- it could work, or the Writer could ignore a plain-language instruction the same way it under-complies with the existing, similarly-worded closing-move cue |
| qualitative: does the seed002-style restatement pattern recur on this thread | present at baseline (this thread doesn't have that exact chain, but its own deep bins show the same shape) | fewer near-identical restatements in the deepest bins if the mechanism works at all | one thread, small n of any single restatement pattern |

**Guardrails:** `self_bleu_4` must not measurably worsen (the instruction
asks for a new relation, not a stylistic change). `hard_disagree_rate`/
`polite_rate`/`impolite_rate` should stay within this thread's own noise --
this arm touches only the coverage block's trailing instruction, nothing
about stance or tone. `avg_depth`/`structural_virality` are structural and
must not move. Prompt length should grow by roughly one sentence per slot
past the first few, not measurably more (no scaling change to the
coverage block itself).

Command, run after this entry was committed:

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag v108_semantic_coverage_nonrepeat_seed8_20260823_v1 --domain camera \
  --model gpt-5.4-mini --base-url https://api.openai.com/v1 \
  --api-key-env LLM_API_KEY --pool-size 150 --max-posts 1 --posts-per-run 1 \
  --start-seed-index 8 --sampling-seed 42 \
  --semantic-coverage-nonrepeat on --resume

python3 generalized_card/scripts/run_evaluate.py \
  --tag v108_semantic_coverage_nonrepeat_seed8_20260823_v1 --metric-parallel 5 --resume
```

### Gate ran, the arm never fired, and the bug that caused it -- 2026-08-23

Run `v108_semantic_coverage_nonrepeat_seed8_20260823_v1`, seed 8, 186
comments, **$1.1867**, 19.3 min. Before reading a single metric, checked
whether the mechanism actually fired by grepping the run's own saved
`generation_records.json` for the instruction string: **0 of 186
prompts contained it.**

**Root cause.** The shipped fix touched `_thread_memory`, the ledger
builder for `--writer-prompt full`. `writer_prompt`'s default is
`--writer-prompt focused` (`_writer_prompt_mode`'s own default, unset by
this run's command and by every gate command in this project's history),
which dispatches to a *different* function, `_focused_thread_ledger` --
which renders its own, separately-coded version of the same "already
covered" block, also missing the instruction, never touched by the
original edit. The arm was real, tested, offline-verified, and reached
zero real generation calls, because it was built against a code path this
project stopped using as the default in v82.

This is the exact failure mode `test_sentence_rhythm.WriterPromptTest`'s
"reaches both paths" convention exists to catch (its own comment: v74's
first focused-prompt cut once left 106 of 522 slots on the old path) --
the v108 tests checked the instruction rendered correctly in isolation,
never checked which of the two prompt builders a default run actually
calls. See `tasks/lessons.md`.

**Fixed the same day** (commit after this one): `_focused_thread_ledger`
now carries the same conditional instruction. Added the end-to-end
regression test that would have caught this before spending --
`FocusedWriterPromptTest.test_semantic_coverage_nonrepeat_reaches_both_writer_prompt_paths`,
built through `configure_generator_backend`/`build_writer_prompt`, the
real dispatch, not the helper function directly -- for both `focused` and
`full`. 622 tests pass, Ruff clean, `prompts.py` re-pinned, self-test
green on/off across all four domains again.

**This run's numbers are not evidence about the mechanism** (it never
ran) and are not used for anything: `self_bertscore_mean_f1` gap vs real
widened, +0.0183 (v104 baseline) -> +0.0243, and the depth-binned excess
moved worse in three of five bins -- exactly the shape of ordinary
thread-level regeneration noise this project has already documented
repeatedly on this same seed-8 thread (`tasks/lessons.md`), not a result
to read as "the idea doesn't work." The idea has not been tested yet.

### Gate result (v2, the arm actually firing this time) — 2026-08-23. The best single-thread result of any mechanism this session

Run `v108_semantic_coverage_nonrepeat_seed8_20260823_v2`, seed 8, 186
comments, **$1.2036**, 19.9 min. **Confirmed the arm fired before reading
any metric**, per the lesson the wasted v1 run wrote:
`generation_records.json`'s saved prompts contain the instruction string
in **186 of 186** slots.

**`self_bertscore_mean_f1` improved on this thread by the largest margin
of any single mechanism gated this session:**

| | real | generated | gap | vs v104 baseline (+0.0183) |
|---|---:|---:|---:|---|
| this run | 0.4887 | 0.5026 | **+0.0139** | **-0.0044** (24% relative reduction) |

For comparison: v107 isolated (G15) improved this same gap by only
-0.0010; v105+v106 combined (G16, at N=10) made the pooled metric flat to
slightly worse. This is the first mechanism this session to move this
metric by a margin that isn't plausibly just this thread's own noise
floor -- though it is still one thread, and G16 already showed a clean
single-thread win on this exact seed can fail to replicate at N=10
(`--verdict-close-guard`'s check-variant result). That lesson still
applies here; this is a promising single data point, not a result.

**Depth-decomposed, fidelity-checked against the shipped artifact first
-- improved in four of five bins, including the deepest, most heavily
affected one every prior gate has struggled with:**

| depth range | v104 excess | this run's excess | |
|---|---:|---:|---|
| [0,1) root-root | +0.0111 | +0.0036 | improved |
| [1,2) | -0.0021 | -0.0053 | ~flat (already near zero) |
| [2,4) | +0.0121 | +0.0103 | improved |
| [4,7) | +0.0209 | +0.0147 | improved |
| [7,+) | +0.0401 | **+0.0329** | improved |

Unlike every prior gate on this metric (v105+v106 worsened [2,4) and
[7,+); v107 isolated improved only [4,7)/[7,+) while [2,4) worsened
slightly), this is the first mechanism to move the excess in the same
direction across nearly the whole depth range, not trade one bin for
another.

**Guardrails, read against the v104 baseline:** `self_bleu_4` flat
(+0.0066 -> +0.0057). `avg_depth`/`structural_virality` flat, structural.
`hard_disagree_rate` moved further from real (-0.0230 -> -0.0454),
flagged not chased -- this arm touches no stance mechanism.
`polite_rate`/`impolite_rate` roughly flat, deprioritized (G4/G8).
`mean_story_probability` and `emotion_entropy` both moved *toward* real
(+0.0231->+0.0159; -0.4092->-0.3240) -- the same direction seen on
essentially every other gate on this thread regardless of which arm was
on, continuing to read as thread-level regeneration noise
(`tasks/lessons.md`), not an effect of this arm.

**Decision: default stays `off`.** One favorable, broad, mechanism-confirmed
single-thread result is real progress -- the best this session has
produced for this metric -- but per this project's own "a gate is one
thread" discipline (J6) and the specific, already-learned lesson that a
clean single-thread win here has failed to replicate before (G16), this
is not grounds to flip a default. The natural next step is an N=10 pool
to get a statistically powered read, isolated (not stacked with
`--digit-cue-guard`/`--verdict-close-guard`, to keep attribution clean) --
a spend decision, not made here.

---

## N=10 gate — `--digit-cue-guard on --verdict-close-guard on` combined (predictions, 2026-08-22)

No new policy version: both arms already exist (v106, v107), each verified
offline across all four domains and each independently gated on the seed-8
single thread. `--reply-novelty-scope` stays `parent_only` (default) —
excluded, because its only gate evidence (G11/G13) shows it *worsening* the
exact `self_bertscore_mean_f1` depth bins it targeted. This run tests whether
the two clean single-thread results replicate at a statistically powered
N=10, which a single thread's own regeneration noise (documented on both the
v106 and v107 gates — `tasks/lessons.md`, "two unrelated arms moving the same
metric the same direction") cannot settle.

**Baseline — the last true N=10 run, unchanged control per J6**
(`generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1`, 532
comments, paired seeds `--start-seed-index 2 --sampling-seed 42 --max-posts
10`; neither guard existed then):

| metric | real | v103 (arms off) | gap | Cliff | status |
|---|---:|---:|---:|---:|---|
| `self_bertscore_mean_f1` | 0.4942 | 0.5097 | +0.0169 | 0.86 | FAIL |
| `self_bleu_4` | 0.0278 | 0.0325 | +0.0040 | 0.40 | PASS |
| `hard_disagree_rate` | 0.1208 | 0.0920 | -0.0243 | -0.23 | PASS |
| `polite_rate` | 0.3085 | 0.1258 | -0.2515 | -0.60 | PARTIAL |
| `impolite_rate` | 0.4041 | 0.5948 | +0.1837 | 0.61 | FAIL |
| `mean_story_probability` | 0.1266 | 0.1219 | -0.0184 | -0.14 | PASS |
| `emotion_entropy` | 1.5803 | 1.7092 | -0.0220 | 0.02 | PASS |

**Predictions**, each stated as a way to be wrong:

- `self_bertscore_mean_f1`: **narrower gap, not closed to PASS.** Both arms'
  single-thread gates moved this metric by roughly ±0.001-0.004 on one
  thread; at N=10 pooled that is a small fraction of the +0.0169 pooled gap.
  A gap under ~0.012 (Cliff below ~0.65) would be a genuine surprise, not the
  expected case.
- The two criterion-2 tells (`digit_cue_diagnosis.py`, `verdict_close_diagnosis.py`)
  should both land near real's rate corpus-wide, replicating the single-thread
  result at scale — this is the actual bet this run is placing.
- `hard_disagree_rate`, `mean_story_probability`, `emotion_entropy`: **no
  confident prediction.** Both single-thread gates moved these by amounts
  that read as thread-level regeneration noise, not either arm's effect
  (`tasks/lessons.md`); at N=10 that noise should average out, so these
  should land close to the v103 baseline row above, not repeat the
  single-thread swings.
- `self_bleu_4` must not measurably worsen (neither arm touches lexical
  diversity by design).
- `polite_rate`/`impolite_rate` are not expected to move — out of scope for
  both arms, already flagged G4/G8 as likely not closable without gaming.

**Commands:**

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag generalized_card_camera_gpt54_v107_digit_verdict_n10_20260822_v1 \
  --domain camera --model gpt-5.4-mini \
  --base-url https://api.openai.com/v1 --api-key-env LLM_API_KEY \
  --pool-size 150 --max-posts 10 --posts-per-run 5 \
  --start-seed-index 2 --sampling-seed 42 \
  --digit-cue-guard on --verdict-close-guard on --resume

python3 generalized_card/scripts/run_evaluate.py \
  --tag generalized_card_camera_gpt54_v107_digit_verdict_n10_20260822_v1 \
  --metric-parallel 5 --resume
```

Expected cost: the last N=10 run (`v103`, same pool parameters) cost $3.7345
for 532 comments; this run should land in the same range, plausibly a little
higher from the two guards' occasional repair-loop retries.

### Gate result — 2026-08-23. A null result at the scale that actually counts: the single-thread win did not replicate

Run `generalized_card_camera_gpt54_v107_digit_verdict_n10_20260822_v1`, 532
comments, **$4.3909**, 65.1 min (includes one crashed and resumed attempt,
below). User-run, per the committed prediction entry above.

**Operational note, not a result.** The first generation attempt crashed:
one comment (seed 6/`382jsa`, task 34 of 91) exhausted the Writer's bounded
slot-local recovery, and `--post-retry-limit`'s default of 1 disables
whole-post regeneration by design (confirmed from its own `--help` text),
so the entire batch process raised rather than silently shipping a shorter
thread — the project's own "never omit a matched slot" rule, working as
intended, just triggered for the first time at N=10 scale (10× the posts of
any single-thread gate, so 10× the chances of one hard Writer miss). A
second invocation of the identical `--resume` command completed cleanly;
`--resume` correctly skipped the four already-persisted posts and
regenerated only the missing slot fresh (confirmed against
`completed_seed_slots`, which reads persisted `discussion.json`, not attempt
history — so no double-charge). Cost includes both attempts.

**`self_bertscore_mean_f1`: unchanged from the pre-fix baseline, to two
decimal places of Cliff's delta.**

| | real | generated | gap | Cliff | status |
|---|---:|---:|---:|---:|---|
| v103 (both guards off) | 0.4942 | 0.5097 | +0.0169 | **0.86** | FAIL |
| this run (both guards on) | 0.4942 | 0.5112 | +0.0170 | **0.86** | FAIL |

This directly fails to replicate v107's own isolated single-thread gate
result (G15: gap narrowed +0.01834→+0.01726 on seed 8 alone). Read per
thread, the reason is visible: **5 of the 10 threads' own gap improved, 5
got worse**, netting to no net movement at the 10-thread-average the actual
metric computes (seed 8/`i1o51h` itself did improve here too, +0.02345→
+0.02010, consistent with G15 — but seed 6/`382jsa`, the one crash-regenerated
thread, moved the other way, +0.00228→+0.01093, and the other 8 split
roughly evenly). A win on one thread, even the standing gate thread, was not
a reliable predictor of the pool's direction.

**Depth-decomposed at the pair level (pooled across all 10 threads' pairs,
fidelity-checked against both shipped artifacts first) tells a more precise
and genuinely informative story:**

| depth range | v103 excess | this run's excess | |
|---|---:|---:|---|
| [0,1) root-root | +0.0040 | +0.0039 | flat |
| [1,2) | +0.0004 | +0.0012 | flat (noise) |
| [2,4) | +0.0174 | +0.0172 | flat |
| [4,7) | +0.0198 | +0.0180 | improved |
| [7,+) | +0.0432 | **+0.0346** | improved, largest move |

The two deepest bins — where a long, winding reply chain is most likely to
close on a verdict, the exact thing v107 targets — **did** improve, by a
real, non-trivial margin, pooled over both artifacts' full pair sets. This
is not the same finding as "the fix does nothing": it is evidence the
mechanism has a genuine, reproducible effect on the specific pairs it should
affect. **It does not reach the official metric** because
`self_bertscore_mean_f1` averages 10 *thread-level* means with equal
weight, not all pairs pooled with equal weight — a real improvement
concentrated in the deep-pair population of a few large threads is
diluted, sometimes to invisibility, by the equal-weight average once other
threads move the other way for unrelated reasons. This is a real property
of the metric's own definition, not a flaw in the analysis.

**Criterion-2 tells: one partially replicated, one did not.**

`digit_cue_diagnosis.py`: any bare `0`/`1` fell from **4.6×** real's rate
(v103, pre-fix) to **2.45×** (this run) — a real, if partial, improvement,
short of the near-parity (1.05×-ish) the single-thread v106 gate showed.
plain-quantifier sub-pattern: **8.2×** → **3.17×** — same pattern, real but
partial. **A new guardrail flag**: `enum_or_fact` (genuine numbered
lists/fractions/prices) fell to **0/532** (was 0.004, ~2 instances,
pre-fix) against real's 0.00216 — the guardrail this session named in
advance ("if `enum_or_fact` falls, the guard is suppressing real
quantities") may be showing exactly that, though n is too small (0 vs an
expected ~1) to be confident it is a real effect rather than noise at an
already-low base rate. Flagged, not acted on.

`verdict_close_diagnosis.py`: the check/test variant this fix specifically
targets **did not move** — 0.0065 (2/308, pre-fix) → 0.0067 (2/298, this
run), statistically identical at n=2 either way. This directly contradicts
the isolated single-thread gate's "fully eliminated" result (G15) — that
result reads, in hindsight, as a small-sample fluke on one thread rather
than a population effect. The pre-existing `abstract_verdict_close` pattern
did fall substantially again (0.1656→0.0973), continuing the same
unattributed-to-either-guard pattern seen on every gate this session
(`tasks/lessons.md`).

**The other nine metrics stayed within the range this project has already
established as N=10 noise (G9)**; `impolite_rate` moved from FAIL to
PARTIAL, `neutral_rate` from PASS to PARTIAL — shuffling within the same
underpowered band, not a new finding.

**Decision.** Both guards ship unchanged, default `off`; nothing here
argues for flipping either default. `--digit-cue-guard`'s criterion-2
improvement is real at this scale, just partial. `--verdict-close-guard`'s
own targeted number did not hold up outside one thread. Most importantly:
this is the **third** independently-verified, well-motivated mechanism
built specifically to move `self_bertscore_mean_f1` (after v104's
evaluative register and v105's chain-scoped novelty) to fail at real
statistical power, on top of the fourth if v106 (never intended as a fix
for this metric) is counted as a control. Per this project's own
systematic-debugging discipline, three failed, well-targeted fixes is the
threshold for questioning the approach rather than attempting a fourth
narrow patch — see `docs/DECISIONS.md` G16 and `tasks/todo.md` for the
recommendation this produces.

---

## v106 — digit-cue quantifier guard (2026-08-22)

Policy ID: `generalized-card-v2-digit-cue-quantifier-guard-v106-20260822`.
Arm `--digit-cue-guard {off,on}`, default `off` (byte-for-byte v105 and
earlier). No domain-profile change. Module
`generalized_card/generalized_card/sentence_rhythm.py`
(`set_digit_cue_guard`, `_DIGIT_CUE_GUARDED`). A criterion-2 (eye-visible)
tell, not one of the 12 metrics; reproduce with
`generalized_card/analysis/digit_cue_diagnosis.py`.

**Why.** `sentence_rhythm`'s "digit" habit asks the Writer to cite a real
quantity "as a figure rather than described in words." Flagged during a
previous gate read and never designed: the Writer sometimes numeralizes an
ordinary quantifier or negation instead -- "1 thing I'd actually check",
"I found my 1 update", "that 1 folder" -- where a person writes the word.

**Measured, not assumed, including the part that complicates the obvious
story.** On the v103 artifact against 424 evaluation-excluded real camera
threads: a bare `0`/`1` appears in 0.092 of generated comments against 0.020
of real ones (4.6x). Real writers do numeralize a plain quantifier too -- it
is 55% of real's own bare-`1` occurrences, not a rare exception -- but
generated does it at 96% of its own and **8.2x** real's per-comment rate for
that specific pattern (0.083 against 0.010), against **1.7x** for
enumerated/fractional/price uses (0.004 against 0.002; a numbered list, a
fraction, a price range). The excess concentrates in the sub-pattern that
does not serve the cue's own stated purpose, not in the raw digit rate.

**The fix.** One added sentence naming the failure mode by example when the
digit habit is drawn; the underlying "cite a figure" instruction is
unchanged, so a genuine count, price, or spec is unaffected. No domain
vocabulary in the added text.

**Offline state.** 596/597 tests pass (the one failure is the live
provenance guard while this version sits uncommitted, clears on commit),
Ruff clean, 3 pins re-computed with 0 unexpected drift
(`sentence_rhythm.py`, `backend.py`, `run_generate.py`), both parity scopes
healthy, backend self-test on and off for both arm values across all four
registered domains (8 runs, $0, all exit 0).

### Combined v105+v106 large-thread gate — predictions, written before the paid run

Credential confirmed by the user: `LLM_API_KEY` from `third_party/MiroFish/.env`
(the same one every prior run in this project has used). Gate thread: seed 8 /
`i1o51h`, the project's standing large-thread gate seed, 186 comments. Both
arms on: `--reply-novelty-scope chain --digit-cue-guard on`.

**Baseline is the same thread in the immediately prior version's artifact**
(J6): `v104_evaluative_seed8_20260821_v1`, generated `self_bertscore_mean_f1`
0.50707 (184 usable comments) against matched real `i1o51h` 0.48873 (185) --
gap **+0.0183**. Neither arm existed then, so that run is the correct `off`
control for both; no second paid run is needed to establish it.

**What the pre-run offline replay already shows on this exact baseline
artifact** (not an ablation -- a replay of the real checks against plans that
were never regenerated, so this is context for the prediction, not a price
for it, per J7):
- `reply_novelty_chain_diagnosis.py --run v104_evaluative_seed8_20260821_v1`:
  **18 of 186 plans (9.7%)** would trip `reply_increment_conflict` under
  `chain` that did not trip under `parent_only` (which caught 0, as always).
- `digit_cue_diagnosis.py --run v104_evaluative_seed8_20260821_v1`: bare `0`/`1`
  in 0.086 of this thread's comments, **100% of them** (15/15) the
  plain-quantifier sub-pattern the guard targets, none the genuine
  enumerated/fractional/price kind.

**Predictions:**

| what | v104 (arms off) | predicted direction | why not a precise number |
|---|---:|---|---|
| `self_bertscore_mean_f1` gap vs real | +0.0183 | **narrower**, not closed to zero | ~10% of reply slots are directly implicated; the mechanism forces a different plan, not a specific rewritten sentence, so no text-substitution ablation exists to price this the way v104's tag/partitive edits could |
| `reply_novelty_chain_diagnosis.py` chain trips on the new artifact | 18 (replayed, not causal) | **substantially fewer**, not necessarily zero | the repair loop is budget-bounded; a slot that exhausts its repair budget still ships |
| bare-1 plain-quantifier rate | 0.081 | **down**, not to real's 0.010 | the guard adds an exclusion, it does not remove the "digit" habit draw itself |

**Guardrails, each a named way to be wrong:**
- `self_bleu_4` must not measurably worsen. Forcing a different plan should
  diversify content, not degrade text quality generically; a worse
  `self_bleu_4` would mean the repair loop is producing worse writing, not
  more distinct writing.
- `hard_disagree_rate`, `polite_rate`, `impolite_rate` should not move beyond
  this thread's own noise -- neither mechanism touches stance, tone, or
  register.
- `avg_depth` / `structural_virality` are structural (copied from the real
  tree) and must not move regardless.
- Genuine digit uses (prices, specs, counts) must not drop. If
  `digit_cue_diagnosis.py`'s `enum_or_fact` rate falls on the new artifact,
  the guard is suppressing real quantities, not just ordinary quantifiers.

Command, run after this entry was committed:

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag v106_chain_novelty_digit_guard_seed8_20260822_v1 --domain camera \
  --model gpt-5.4-mini --base-url https://api.openai.com/v1 \
  --api-key-env LLM_API_KEY --pool-size 150 --max-posts 1 --posts-per-run 1 \
  --start-seed-index 8 --sampling-seed 42 \
  --reply-novelty-scope chain --digit-cue-guard on --resume
python3 generalized_card/scripts/run_evaluate.py \
  --tag v106_chain_novelty_digit_guard_seed8_20260822_v1 --metric-parallel 5 --resume
```

### Gate result — 2026-08-22. Both mechanisms confirmed working; a different, previously-masked defect is what the metric was tracking

Run `v106_chain_novelty_digit_guard_seed8_20260822_v1`, seed 8, 185 comments,
**$1.2081**, 23.7 min generation. Credential used: `LLM_API_KEY` from
`third_party/MiroFish/.env`, confirmed by the user.

**Both mechanisms did exactly what they were built to do, verified directly,
not inferred from the metric:**
- `reply_novelty_chain_diagnosis.py` on the new artifact: **0 of 186 plans**
  trip `reply_increment_conflict` under `chain` (the same artifact's
  predecessor, replayed, had 18). The diagnosed chain-restatement defect is
  gone at the plan level.
- `digit_cue_diagnosis.py` on the new artifact: bare `0`/`1` fell from 0.086
  to **0.0215** (real: 0.020 -- essentially at parity), and the
  plain-quantifier sub-pattern fell from 8.01x real's rate to **1.60x**.

**`self_bertscore_mean_f1` did not improve, and by the pooled thread mean got
slightly worse:** gap vs real (`i1o51h`, same thread both times) **+0.0183 →
+0.0218**. This is the same shape as v104's own gate: the arms worked, the
metric did not follow. It is one thread (N=1); per this project's own
discipline a single thread cannot establish direction with confidence, but it
can and does show a *failure to improve* on the one thing being tested.

**Decomposed by depth, isolated to this one thread (before/after, not
pooled):**

| depth range | v104 excess | v106 excess | |
|---|---:|---:|---|
| [0,1) root-root | +0.0111 | **+0.0005** | improved |
| [1,2) | -0.0021 | -0.0061 | flat |
| [2,4) | +0.0121 | **+0.0214** | worse |
| [4,7) | +0.0209 | +0.0209 | unchanged |
| [7,+) | +0.0401 | **+0.0474** | worse |

Root-level pairs, which the fix never targeted, improved anyway (probably
noise on 10 pairs). The reply-chain bins the fix *did* target did not
improve -- they got worse in two of four bins.

**Why, read from the actual pairs (`bertscore_pair_diagnosis.py inspect` on
the new artifact):** the pre-fix high-F1 tail was claim-level duplication
("compactness doesn't matter in a bag," said six times). That is gone -- none
of the new artifact's 8 highest-scoring pairs restate the same claim. What
replaces it is **sentence-template reuse across different claims**:
`"@OP, watch the subject cross the EVF and see if your eye can keep up."`
is one side of three different high-scoring pairs, each time against a
*different* specific claim (eye-tracking, EVF blanking, a display-tilt test);
`"[noun]. That's the [X] check"` and `"I'd still want to see X... that's a
solid check"` each recur once; two more pairs are generic gratitude closers.
The Planner really did diversify the content (confirmed: 0 plan-level
violations) -- the Writer is falling back on a narrow set of reusable
sentence *frames* regardless of what content gets slotted into them, and
that is what the embedding metric reads as similarity.

**This explains why the existing route ledger didn't already catch it.**
`used_sentence_routes`/`reused_sentence_routes` (`semantic_realization.py`)
match on the first 3-4 literal tokens of a clause. `"@OP, watch the subject"`
and `"@OP, check whether the"` differ at the second token, so they are two
distinct "routes" to that ledger even though they are the same template to a
reader. The ledger needs a way to catch a template with a variable slot, not
just a repeated literal n-gram -- that is a new mechanism, not a parameter
change to this one.

**Guardrails, read against the same before/after:** `self_bleu_4` +0.0003
(flat, not a violation). `hard_disagree_rate` moved +0.0217, *toward* real
(0.1467 → 0.1685 against real 0.1697) -- unpredicted but favorable, plausibly
a side effect of forced re-planning changing opener distribution; not
concerning. `mean_story_probability` moved +0.0454, *away* from real (0.1345
→ 0.1799 against real 0.1114) -- unpredicted and unfavorable; flagged, not
chased, on a single thread. `emotion_entropy` moved +0.1414 toward real.
`semantic_mean_cosine`, `avg_depth`, `structural_virality` essentially flat.
No guardrail crossed a threshold that would call the run invalid, but two
moved substantially in ways neither mechanism should have touched, which is
exactly why a single thread cannot be read as a verdict on anything but the
one thing it was gated for.

**Decision: do not run N=10 on `chain`/`digit-cue-guard on` yet.** The
mechanisms are real and worth keeping (the digit-cue result alone is a clean
win for criterion 2), but spending 10x more to confirm a metric result this
gate already shows did not move would repeat the mistake this project's
process exists to prevent. The next paid step, if any, should follow a
sentence-template mechanism for `self_bertscore_mean_f1`, not a repeat of
this one at larger N. See `docs/DECISIONS.md` G13 and `tasks/todo.md`.

### Follow-up, same day: the sentence-template hypothesis rejected at scale; a different, real lead found instead

The gate result above ended on "the next paid step needs a sentence-template
mechanism." Before building one, per §4 step 3, the hypothesis was measured
at scale (`analysis/template_reuse_diagnosis.py`) rather than trusted from 8
examples: within-thread opener/closer clause embedding similarity, generated
vs real. **Rejected.** Generated's near-duplicate rate (opener 0.0016, closer
0.0005, pooled over the v103 10-thread pool and the v106 gate combined) is
barely above matched real's (0.0009/0.0003) and indistinguishable from an
80-thread evaluation-excluded real null (0.0012/0.0005). The 8 examples were
real text, not a fabrication -- they were the extreme tail of a statistic
real threads produce at a comparable rate. Same trap as v98's rejected
"duplication tail" hypothesis, on a different metric: a vivid top-of-list
read does not generalize just because it is vivid.

Reading those same 8 examples again for *what* they actually were (rather
than assuming "generic template") found something narrower and real: three
were opener-side ("@OP, ...", a separate mechanism, `opener_profile.py`, not
chased further this session), and the closer-side ones -- "Plate flush on
the mount. That's the check", "that's a solid check" -- are a lexical
variant of `closing_move.py`'s already-known `abstract_verdict_close` tic
(chased since v73, v100's fix), using "check"/"test" as the head noun where
the existing measured pattern only recognizes "matters/the real thing/the
part/...". Measured directly (`analysis/verdict_close_diagnosis.py`): this
variant is real and elevated (13-37x real depending on population), and,
more importantly, **the pattern v100's fix already targets is still
10-13x over real even where its suppression cue reaches the Writer** -- the
existing fix reduced the tic, it did not close it. See v107 below.

---

## v107 — verdict-close check-variant guard (2026-08-22)

Policy ID: `generalized-card-v2-verdict-close-check-guard-v107-20260822`.
Arm `--verdict-close-guard {off,on}`, default `off` (byte-for-byte v106 and
earlier). No domain-profile change -- this widens the Writer-facing
suppression cue only, not `closing_move.py`'s measurement pattern, so no
profile rebuild is needed. Module
`generalized_card/generalized_card/closing_move.py`
(`set_verdict_close_guard`, `_VERDICT_CLOSE_GUARDED`). Reproduce with
`generalized_card/analysis/verdict_close_diagnosis.py`.

**Why.** Found reading the v106 gate's actual pairs while diagnosing why
`self_bertscore_mean_f1` didn't move (`docs/DECISIONS.md` G13/G14). The
generic "sentence-template reuse" story that motivated looking was rejected
at scale (see the v106 follow-up above); what survived is narrower: a
"that's the check"/"a solid check" closing is a lexical variant of the
already-known, already-partially-fixed `abstract_verdict_close` tic that the
existing pattern's word list never named.

**Measured, including the more important number.** `verdict_close_diagnosis.py`
on the last sentence of every 25+-word comment:

| population | existing pattern | check/test variant (new) |
|---|---:|---:|
| v103 N=10 generated | 0.166 | 0.0065 |
| v106 gate (seed 8) | 0.130 | 0.0185 |
| evaluation-excluded real | 0.013 | 0.0005 |

The new variant alone is 13-37x real's rate. But the *existing* pattern --
the one `closing_move.py`'s "measured" arm has suppressed since v100 -- is
**still 10-13x over real**, on artifacts built well after that fix shipped.
Decomposed on the v106 gate: of the 25+-word slots whose prompt actually
carried the suppression cue, **15.2% still produced the tic anyway** (too
few cue-absent slots on one thread, n=9, to read that side confidently).
v100's fix reduced the tic; it did not close it, and this session is the
first time that gap was measured directly rather than assumed closed because
an arm exists.

**The fix is deliberately narrow.** It widens `_VERDICT_CLOSE_GUARDED`'s
wording to name the check/test variant explicitly, without touching
`abstract_verdict_close`'s detection pattern or measured rate -- extending
the *measurement* would require a profile rebuild across all four domains
with a fresh zero-seed-overlap check, which this session did not have
evidence yet to justify. The 10-13x shortfall in the *existing* pattern is
left as an open question (`docs/DECISIONS.md` G14) -- possibly a coverage
gap (not every slot receives the cue) rather than a compliance gap; the v106
gate's single thread cannot separate the two.

**Offline state.** 603 tests pass (the provenance guard clears on commit,
same as v105/v106), Ruff clean, 3 pins re-computed with 0 unexpected drift
(`closing_move.py`, `backend.py`, `run_generate.py`), both parity scopes
healthy, backend self-test on and off for both arm values across all four
registered domains (8 runs, $0, all exit 0).

### Large-thread gate — predictions, written before the paid run

Credential confirmed by the user: `LLM_API_KEY` from `third_party/MiroFish/.env`
(the same one every prior run in this project has used). Gate thread: seed 8 /
`i1o51h`, the project's standing large-thread gate seed, 186 comments.

**Isolated, not stacked on v105/v106.** v105 (`--reply-novelty-scope chain`)
was already gated on this exact thread and falsified as
`self_bertscore_mean_f1`'s driver; v106 (`--digit-cue-guard on`) is an
independent criterion-2 fix unrelated to this mechanism. That combined gate
already moved several secondary metrics unpredictably on N=1 (`hard_disagree_rate`,
`mean_story_probability`, `emotion_entropy`) -- stacking a third ungated arm
onto that same noise floor would make anything this gate finds unattributable
to v107 specifically. This run sets only `--verdict-close-guard on`; both
other arms stay at their `off`/`parent_only` defaults, so the correct control
is the pure "everything off" baseline, `v104_evaluative_seed8_20260821_v1` --
neither v105, v106, nor v107 existed when that artifact was built, and J6
requires the baseline be the same thread in the immediately prior version's
artifact, not the v106 gate artifact.

**What the offline replay shows on this exact baseline artifact** (measured
just now, not assumed -- `verdict_close_diagnosis.py --run
.../v104_evaluative_seed8_20260821_v1`):

| population (seed 8 thread) | existing pattern | check-variant (new) |
|---|---:|---:|
| v104 baseline (all arms off) | 0.1887 (20/106) | 0.0283 (3/106) |
| v106 gate (v105+v106 on, v107 off) | 0.1296 (14/108) | 0.0185 (2/108) |
| evaluation-excluded real | 0.0129 | 0.0005 |

The v106 gate's lower rate on both columns is a side effect of forced
re-planning (v105/v106 touch Planner output, which reshuffles which comments
draw which closing move) -- not this fix, which hadn't shipped yet. The
correct pre-fix number for this gate is the v104 row.

`self_bertscore_mean_f1` on this exact thread, already established (J6
baseline): generated 0.50707 vs matched real 0.48873, gap **+0.0183**.

**Predictions:**

| what | v104 baseline | predicted direction | why not a precise number |
|---|---:|---|---|
| check-variant rate | 0.0283 | **down**, toward real's 0.0005 | the guard only widens the suppression wording for slots where the move is drawn *not* to happen; it does not touch the draw probability or `abstract_verdict_close`'s own wording |
| existing-pattern rate | 0.1887 | **not expected to move** | v107 does not touch `abstract_verdict_close`'s cue text or detection pattern -- the 10-13x compliance gap there (G14) is a separate, still-open question this gate is not built to answer |
| `self_bertscore_mean_f1` gap vs real | +0.0183 | **no confident prediction; flat is more likely than closed** | the template-reuse hypothesis this fix grew out of was already rejected at scale as a population-level driver (G13) -- this gate is a criterion-2 compliance check, the same shape as v106's clean win, not a second self_bertscore bet |

**Guardrails, each a named way to be wrong:** `self_bleu_4` must not
measurably worsen. `hard_disagree_rate`/`polite_rate`/`impolite_rate` should
stay within this thread's own noise -- v107 touches only closing-move
wording, not stance or tone. `avg_depth`/`structural_virality` are
structural (copied from the real tree) and must not move regardless.

Command, run after this entry was committed:

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag v107_verdict_close_guard_seed8_20260822_v1 --domain camera \
  --model gpt-5.4-mini --base-url https://api.openai.com/v1 \
  --api-key-env LLM_API_KEY --pool-size 150 --max-posts 1 --posts-per-run 1 \
  --start-seed-index 8 --sampling-seed 42 \
  --verdict-close-guard on --resume
python3 generalized_card/scripts/run_evaluate.py \
  --tag v107_verdict_close_guard_seed8_20260822_v1 --metric-parallel 5 --resume
```

### Gate result — 2026-08-22. First arm across four gates to move `self_bertscore_mean_f1` in the right direction on its own gate thread

Run `v107_verdict_close_guard_seed8_20260822_v1`, seed 8, 184 usable comments,
**$1.1637**, 18.3 min generation. Credential used: `LLM_API_KEY` from
`third_party/MiroFish/.env`, confirmed by the user. Isolated: only
`--verdict-close-guard on`; `--reply-novelty-scope` and `--digit-cue-guard`
stayed at their `parent_only`/`off` defaults.

**The targeted defect closed exactly where predicted, and further than
predicted** (`verdict_close_diagnosis.py` on the new artifact, same thread,
against the v104 baseline row measured before spending):

| population (seed 8 thread) | existing pattern | check-variant (new) |
|---|---:|---:|
| v104 baseline (all arms off) | 0.1887 (20/106) | 0.0283 (3/106) |
| v107 (guard on, isolated) | 0.1132 (12/106) | **0.0000 (0/106)** |
| evaluation-excluded real | 0.0129 | 0.0005 |

The check-variant, the thing this fix actually targets, is fully gone on this
thread — better than the predicted "down toward real," landing below real's
own rate. The existing-pattern rate also fell, further than predicted ("not
expected to move"), but this fix touches neither `abstract_verdict_close`'s
cue text nor its detection pattern, and the unrelated v105+v106 gate showed a
same-direction drop on this identical measurement (0.1887→0.1296) that had
nothing to do with this mechanism either — read as run-to-run sampling noise
on one regenerated thread, not credited to this fix without more evidence.

**`self_bertscore_mean_f1` moved, for the first time across four gates, in
the favorable direction on its own gate thread:** gap vs real (`i1o51h`, same
thread all four times) v104 **+0.01834** → v107 **+0.01726** (Δ −0.00108).
Small, and still one thread (`inferential_status: DESCRIPTIVE`, `cliffs_delta`
computed on N=1 is not a real inferential test) — but this is the first arm
of the four gated so far (v104, v105+v106, now v107) that did not leave this
metric flat or worse.

**Decomposed by depth, isolated to this one thread, fidelity-checked against
the shipped artifact before reading it** (`bertscore_pair_diagnosis.py depth`):

| depth range | v104 excess | v107 excess | |
|---|---:|---:|---|
| [0,1) root-root | +0.0111 | +0.0061 | improved |
| [1,2) | -0.0021 | -0.0006 | ~flat |
| [2,4) | +0.0121 | +0.0157 | worse |
| [4,7) | +0.0209 | +0.0177 | improved |
| [7,+) | +0.0401 | **+0.0274** | improved, largest single move |

The deepest bin — where a long, winding-down reply is most likely to reach
for a closing verdict — improved the most (-0.0127). This is the mechanism
working exactly where it should: v105+v106's gate *worsened* these same two
deep bins ([2,4) and [7,+)); v107 improves both except [2,4), which moves the
other way here too, by a smaller margin than v105+v106 moved it. Not run
against a matched pair of the same thread with the guard off but everything
else identical — the counterfactual is the v104 baseline artifact, not a
same-artifact ablation — so this is evidence, not proof, that this specific
fix is what moved the deep bins, but the mechanism (a check/test closing
disproportionately ends long, deep-thread comments) makes the correlation a
plausible causal story rather than a coincidence.

**Guardrails, read against the v104 baseline (not the v106 gate, which is a
different arm combination):** `self_bleu_4` +0.00021 (flat: +0.00664→+0.00685,
not a violation). `avg_depth`/`structural_virality` essentially flat
(structural, copied from the real tree, as required). `hard_disagree_rate`
moved *away* from real and further than v104's own gap (-0.02296→+0.03682,
|gap| grew) — flagged, not chased, single thread; v107 touches no stance
mechanism. `mean_story_probability` moved away from real again
(+0.02313→+0.03719) — the *same direction* it moved on the unrelated v106
gate (+0.0454 there too), which is evidence this specific movement is
thread-level regeneration noise rather than something either gate's arm
caused, since the two gates share no mechanism that touches story framing.
`emotion_entropy` moved toward real again (-0.409→-0.311), also the same
direction as the v106 gate (+0.141 there) — same read: noise, not an effect
of either fix. `neutral_rate` moved closer to real (gap -0.0209→+0.0063).
`polite_rate`/`impolite_rate` moved slightly further from real, consistent
with G4/G8 (deprioritized, not gamed for).

**Decision: default stays `off`.** This is the first favorable
`self_bertscore_mean_f1` result of four gates, but it is still one thread
read descriptively, per this project's own "a gate is one thread" discipline
(J6) — not grounds to flip a default. `--digit-cue-guard on` (v106, criterion-2
clean win) and `--verdict-close-guard on` (v107, criterion-2 clean win plus
the first favorable primary-metric nudge) are now two independently gated,
non-conflicting fixes with no evidence either hurts the other (they touch
different code paths, digit habit vs. closing move). `--reply-novelty-scope
chain` (v105) stays `parent_only`: it is a real, verified plan-level fix, but
its only gate evidence so far shows it *worsening* the exact depth bins it
targeted, so it is not bundled into what should go forward. The natural next
paid step is a statistically powered comparison — the N=10 pool, not another
single thread — running `--digit-cue-guard on --verdict-close-guard on`
together, since single-thread descriptive reads cannot separate a real
2-gate effect from this thread's own noise, which the guardrails above show
is not small. Flagged to the user as a spend decision, not decided
unilaterally, given the roughly 10x cost jump from a one-thread gate.

---

## v105 — chain-scoped reply novelty (2026-08-22)

Policy ID: `generalized-card-v2-chain-scoped-reply-novelty-v105-20260822`.
Arm `--reply-novelty-scope {parent_only,chain}`, default `parent_only`
(byte-for-byte v104 and earlier — verified by inspection of the returned
message string, not just asserted). No domain-profile change; schema stays at
20. Module `generalized_card/generalized_card/planning_quality.py`
(`reply_increment_problem`, `_ancestor_chain`), wired through `backend.py` and
`run_generate.py`. Diagnosis in `docs/DECISIONS.md` row G3; reproduce with
`generalized_card/analysis/bertscore_pair_diagnosis.py depth` and
`generalized_card/analysis/reply_novelty_chain_diagnosis.py`.

**Why.** `self_bertscore_mean_f1` is the one metric failing a standard that
does not fail correct work (v103 N=10: MWU 0.001, KS 0.002, |Cliff| 0.86
against a floor of 0.50). The excess is a root-vs-reply effect: generated
reply chains restate the same argument as they deepen (measured, excess grows
+0.0004 at depth 1-2 to +0.0432 at depth 7+), while real chains diversify with
depth (checked at scale: `reply_reply` cosine below `root_root` in 82% of 247
evaluation-excluded real threads, Wilcoxon p≈0).

**A mechanism already existed and was already required
(`require_reply_novelty=True` since before this version) — it just structurally
could not fire.** `reply_increment_problem` compared a reply's narrow
`reply_novelty_anchor` phrase against its parent's full
`{semantic_move, decision_boundary, detail_focus}` — a short phrase against a
longer compound description, which suppresses cosine similarity regardless of
content. Measured on the v103 N=10 artifact: **0 trips, on all 528 comments,
at the existing 0.76 threshold.** The named qualitative chain
(`sampled_run00_post00_seed002`, comments 40→41→42→43→44→45, "compactness
doesn't matter once it's in a bag" six times in a row) scored 0.42–0.61
against its own ancestors this way — nowhere close.

**The fix compares same-shape probes and walks the whole ancestor chain.**
`novelty_scope="chain"` compares the reply's own full plan
(`semantic_move`/`decision_boundary`/`detail_focus`) against every ancestor
already in its branch (walking `parent_id` through the ledger already passed
into the function — no data-flow change, `evaluate_plan_batch`'s `seen` already
carried the whole thread). On the same artifact: **60 trips**, including the
named seed002 chain and a second qualitatively-found chain
(`sampled_run01_post01_seed008`, an AF-tracking argument repeated across many
branches of the largest thread). Every hop of the seed002 chain scores
0.73–0.92 against its own immediate parent this way, against 0.22–0.62 for
genuinely unrelated branches in the same artifact — a clean separation the old
probe never had a chance at, at any scope.

**No new threshold.** 0.76 is reused unmodified; a distance-decay function
would need its own calibration and any calibration available right now would
come from re-inspecting camera-only numbers, which the fix is designed to
avoid needing.

**Domain-adaptivity — checked, not assumed.** `reply_increment_problem`,
`_ancestor_chain`, and `PlanSemanticIndex` take no domain-profile argument
anywhere (confirmed by reading every call site); they compare embeddings of
Planner-authored `SEMANTIC_FIELDS` text, the same construction the pre-existing
mechanism already used. The backend self-test
(`generalized_card/scripts/run_generator_backend.py --self-test`, a
`return`-guarded path that never reaches seed loading or an API call) was run
directly for all four registered domains, both scope values — **8 runs, $0,
all exit 0**: `camera_product`, `cell_phone_product`, `headphone_product`,
`laptop_product`. There is nothing in this mechanism that *can* silently
degrade on a sparse domain the way a measured-profile band lookup can, because
it reads no profile — but that had to be observed by running it, not asserted
from the code.

### Offline state — no paid run yet

593 tests (592 pass; the one failure is the live provenance guard while this
version sits uncommitted, which clears on commit), Ruff clean, 3 pins
re-computed with 0 unexpected drift (`planning_quality.py`, `backend.py`,
`run_generate.py`), both parity scopes healthy, backend self-test on and off
for all four domains (8/8 exit 0, $0), default arm value reproduces v104's
prompt path byte-for-byte.

### Not done in this version, deliberately

- **No N=10 run, no default flip yet.** `--reply-novelty-scope` ships with
  `parent_only` as the default until a gate result justifies `chain`.
- **The credential question this note originally raised as unconfirmed was
  answered by the user on 2026-08-22: `LLM_API_KEY` from
  `third_party/MiroFish/.env`, the same one every prior run has used.** The
  large-thread gate (combined with v106, since both ship in the same policy
  state and neither existed in the immediately prior artifact) is recorded
  under v106's entry below, predictions written before spending.

---

## v104 — evaluative register (2026-08-21)

Policy ID: `generalized-card-v2-evaluative-register-v104-20260821`.
Arms `--evaluation-tier {measured,off}`, `--downtoner-tag {suppress,off}`,
`--partitive-reference {suppress,off}`; each `off` reproduces v103, verified on
the real prompt path (every arm off renders an empty rule, so the v103 prompt is
byte-identical). Profile schema 19 → 20, new section `evaluative_profile`.
Module `generalized_card/generalized_card/evaluative_register.py`. Full
diagnosis in `tasks/v104-worklog.md`; reproduce with
`generalized_card/analysis/polite_sentence_diagnosis.py all`.

**Why the previous eight attempts were mis-specified.** `polite_rate` and
`impolite_rate` carry the largest statistically real generator bias against the
Planner's own target (−0.1856 and +0.1529, Wilcoxon p = 0.002 each) and have
failed since v96. Polite Guard is **confident, not near-degenerate** — unlike
the Stance_Rel head behind `hard_disagree_rate`, its median margin on a
generated non-polite comment is −0.934, only 2.1% sit within 0.10 of flipping,
and the median P(impolite) among impolite-labelled generated comments is 0.981.
No sub-sentence marker edit was ever going to tip that.

**What the generator actually does wrong.** It already writes the appreciative
*forms* — `gratitude` at 1.48x the real rate, `positive_predicate` at 1.39x,
`bare_verdict` at parity — and they do not land. Same form, P(the sentence reads
polite on its own): `bare_verdict` real 0.900 / generated 0.111,
`react_to_parent` 0.780 / 0.188, `gratitude` 0.672 / 0.256. Three surface
differences account for it, over 19,386 excluded-real and 1,674 v103 sentences:

| | real | v103 | ratio |
|---|---:|---:|---:|
| hot-tier evaluative word, per 1k sentences | 64.5 | 19.7 | 0.31x |
| trailing downtoner tag, per 1k sentences | 0.98 | 41.82 | **42.7x** |
| partitive reference, share of comments | 0.018 | 0.241 | **13.6x** |
| hot share *within* a positive sentence | 0.482 | 0.128 | 0.27x |

Real writes "Wonderful camera.", "The IV is fantastic.", "Fantastic breakdown!".
This generator writes "Eye AF is good, sure.", "That part was good.", "Pretty
useful, honestly." A person reads those as grudging, so the classifier is not
wrong about them.

**Causal, on the shipped v103 artifact.** Each edit applied to the comment, the
whole comment re-scored with the evaluation's own checkpoint, after the harness
reproduced the artifact's labels flip-for-flip (0 flips, max |ΔP| 0.000000):

| edit | polite | impolite | polite gap closed |
|---|---:|---:|---:|
| strip the trailing tag | 0.1212 | 0.6042 | 8.3% |
| de-partitive | 0.1174 | 0.6269 | 6.2% |
| warm tier → hot tier | 0.1439 | 0.6155 | **20.8%** |
| **CONTROL** warm → *other warm* word | 0.1061 | 0.6250 | **0.0%** |
| all three | 0.1572 | 0.5985 | **28.1%** (impolite 13.7%) |

The control is the falsification: the same 157 comments, the same number of
substitutions, a different *tier* — and `polite_rate` moves by 0.0000.

**Checked before building.** The saved v103 prompts carry **no** rule against
any of the three; the only adjacent text runs the other way ("Ordinary hedges
and brief thanks are allowed when they fit the turn", 292 prompts). And the
reuse ledger, which echoes `- that's the bit that (used 3x)` and `- The $200
part is nice, sure, but` back to the Writer, was tested as a **priming source
and rejected**: partitive lift 0.96x, tag lift 0.49x, flat or lower where the
ledger is present once position in the thread is controlled. The tics are the
model's own register.

**The measured profile has the structure the design needs.** Hot share by
register: polite 0.572, somewhat_polite 0.340, impolite 0.361, neutral 0.210,
against a nearly flat 0.466–0.502 by band — so the register is the informative
dimension and a blunt slot is never handed the warm slot's rate. The tier rule
is conditional on the comment evaluating something at all: the Planner owns
whether a slot praises anything and this arm only sets how far it travels.

### Predictions, written before the paid gate

Realized rates, from `evaluative_register.realized_evaluative_shares` on the
artifact:

| audit field | v103 | predicted | measured real |
|---|---:|---:|---:|
| `downtoner_tag_per_1k_sentences` | 41.82 | **< 8** | 0.51 |
| `partitive_comment_rate` | 0.2405 | **< 0.08** | 0.0177 |
| `hot_share_of_positive` | 0.1284 | **0.30 – 0.45** | 0.4821 |

Metrics:

| metric | v103 | predicted | matched real |
|---|---:|---:|---:|
| `polite_rate` | 0.106 | **0.14 – 0.18** | 0.288 |
| `impolite_rate` | 0.623 | **0.57 – 0.61** | 0.443 |

Guardrails, each a named way to be wrong:

- `positive_per_1k_sentences` must **not rise** above v103's 153.5. The
  generator already evaluates *more* than real (130.3); this arm changes
  strength, not count. A rise means the cue was read as "add praise" and the
  Planner's tone marginal has been overwritten.
- The **planned** tone marginal must stay at v103's polite 0.271 / impolite
  0.494. The arm acts on realization only.
- `self_bleu_4` must not worsen. Naming concrete words is what buys compliance
  (v102: ~1.0 for a named token against 0.23 for a category) and it is also how
  a lexicon concentrates. Each slot draws its own 3-word window out of 24 hot
  words; if `self_bleu_4` rises, widen the window before widening anything else.
- `hard_disagree_rate` must not move. Nothing here touches the opener, and v103
  brought its generator bias to +0.0032 (Wilcoxon p = 1.000).

Offline state at the gate: 588 tests, ruff clean, 106 pins 0 drift, self-test
on and off, `off` proven empty on the real prompt path, profile rebuilt at
schema 20 on 424 evaluation-excluded threads.

### v104 gate result — 2026-08-21. The arms worked; the metric did not follow.

Run `v104_evaluative_seed8_20260821_v1`, seed 8, 186 comments, **$1.1288**,
19.5 min generation. Every number below is the **same thread** in every column:
seed 8 also appears in the v103 N=10 artifact as `sampled_run01_post01_seed008`
at 186 comments, and the matched real thread is `i1o51h` at 185.

**My prediction bands were set against the pooled corpus, not this thread.**
That is the v102 error repeated — recorded again in `tasks/lessons.md`. This
thread's real `hot_share_of_positive` is **0.3373**, not the pooled 0.4821, and
v103's `impolite_rate` on it was already 0.5792, inside the band I "predicted".
The check harness now compares a gate against the baseline's largest thread
rather than its ten-thread pool.

| realized rate | v102 | v103 | **v104** | this thread's real | gap closed |
|---|---:|---:|---:|---:|---:|
| downtoner tag / 1k sentences | 33.73 | 31.45 | **10.00** | 1.93 | **73%** |
| partitive, share of comments | 0.2634 | 0.2131 | **0.0815** | 0.0162 | **67%** |
| hot share of positive sentences | 0.1013 | 0.1585 | **0.2556** | 0.3373 | **54%** |
| hot words / 1k sentences | — | 27.3 | **46.0** | 54.1 | **85%** |
| positive / 1k sentences *(guard)* | 156.75 | 171.91 | **180.00** | 160.23 | **BREACHED** |

All three arms moved substantially and **none hit its band**. The guardrail
named in advance — "must not rise; a rise means the cue was read as *add
praise*" — **breached**: 171.91 → 180.00 against a real 160.23.

**And the metric barely moved.**

| | v103 | **v104** | real | |
|---|---:|---:|---:|---|
| `polite_rate` | 0.1093 | **0.1196** | 0.2324 | **8.4%** of the gap |
| `impolite_rate` | 0.5792 | **0.5924** | 0.4649 | **wrong way**, +0.0132 |

Twelve metrics against the same real thread: five closer, seven wider.
`length_cv` 0.8662 → 0.9380 (real 0.8951) and `hard_disagree_rate`
0.1758 → 0.1467 (real 0.1697) both crossed and overshot; `self_bleu_4` was
unmoved at 0.0349, so the drawn word window did not concentrate the lexicon.

**Why it did not follow — measured, not guessed.** The carrier rate is what
reconstructs the metric, and it barely moved:

| | v103 | **v104** | real |
|---|---:|---:|---:|
| carriers (comments holding a P(polite) > 0.80 sentence) | 0.0437 | **0.0598** | 0.1622 |
| P(polite \| carrier) | 0.625 | 0.727 | 0.800 |
| P(polite \| not) | 0.086 | 0.081 | 0.123 |

Reconstruction holds on all three (0.1095 / 0.1197 / 0.2328 against actuals
0.1093 / 0.1196 / 0.2324), so the framing is intact. **13.6%** of the carrier
gap closed against **85%** of the hot-word density gap.

**So hot words are not what makes a carrier.** v104's hot sentences are median
13 words against a real 18, 0.348 of them ten words or shorter against 0.321,
and 0.348 hedged against 0.357 — structurally indistinguishable from real ones.
Density is at 85% of real. And the sentences still do not read as unambiguously
appreciative. The tier hypothesis is **substantially falsified as a lever on
the metric**, despite an ablation that put it at 20.8%.

**The lesson that generalises.** An ablation edits the shipped text and gets an
upper bound; a prompt cue asks the model to write differently and does not reach
it, and the shortfall is not only compliance. v104 reached 54–85% compliance on
every arm's own statistic and delivered 8.4% of the metric. **Do not price a
mechanism off an ablation again without discounting it.**

Kept, reverted, and open:

- The two suppressions fix eye-visible tells and are cheap. `That's the missing
  bit, honestly.` still survives in the artifact, so they are not finished
  either, but 73% and 67% in one release is the shape v98's semicolon
  suppression had.
- `--evaluation-tier` breached its guardrail and bought almost nothing. It
  should go to `off` unless the cue can be rewritten to change strength without
  adding evaluations.
- The carrier gap is untouched and is still the whole thing. What makes a real
  sentence read as unambiguously appreciative is **not** its evaluative word,
  its length, or its hedging. That is now three things it is not, which is
  progress of a kind, and the next mechanism cannot be built until it is one
  thing it is.

## v103 — stance-consistent opening (2026-08-21)

Policy ID: `generalized-card-v2-stance-consistent-opening-v103-20260821`.
Same arm, `--opening-move {measured,off}`; `off` still reproduces v101.

**A correction the v102 gate exposed, found by reading the comments rather than
the metrics.** v102 draws the opening token from the register's measured
distribution. A polarity token carries a **stance**, and the Planner has already
assigned one, so the draw could contradict the plan — and did:

| planned stance | drawn | result |
|---|---|---|
| `agree` | `no` | "no, I'd just check my RAW at the 1 final delivery crop…" |
| `agree` | `no` | "no, wrap 1 hand around it and see if the fingers lock in…" |

**2 of 10 polarity slots on the gate.** Neither comment is actually a negation —
"no," is bolted onto text that agrees. That is an eye-visible tell, which is half
the acceptance criteria, and it is a defect v102 introduced.

The metric table hid it: `hard_disagree_rate` improved to 3.0% relative error on
the same run. Reading 23 comments is what found it.

**The fix: the plan vetoes, and the slot redraws inside the family.** The draw
runs over the register's full measured distribution first; only if the plan
commits to a polarity **and the drawn token disagrees** does the slot redraw
inside the family, under a separate hash namespace:

- `stance=agree` → the affirmative family, renormalised
- `stance=disagree` → the negative family, renormalised
- `mixed`, `uncertain`, `neutral`, `joking`, unset → the full measured draw
- a register whose measured table holds no token of the required family keeps the
  full draw, because withholding would cost the slot its assigned entry type and
  inventing a token would leave the measurement behind

**Why veto-and-redraw rather than simply restricting the family.** The first
implementation restricted the family and drew inside it, which renormalises the
cumulative walk and therefore moves slots that were **never in conflict** — 13.5%
of `polite`+`agree` slots, 32.3% of `impolite`+`agree`, up to 54% in the worst
cell. That would have made a v102/v103 comparison unattributable and it made this
entry's own claim false. Under veto-and-redraw, measured over 4,000 keys in all
eight cells, **every slot that changes is a slot that was contradicting its
plan**, and the within-family shares still reproduce the measurement to 0.004.
The redraw uses its own namespace: reusing the first draw's value would map the
vetoed slice of [0,1) onto the family's CDF and pile those slots onto whichever
tokens that slice happens to cover.

`discourse_marker` is untouched: `thanks / oh / well / and / so / but / lol / ah`
carry no polarity, and all 13 of the gate's discourse slots read correctly against
their stance.

**Marginal drift, checked before building.** Over v101's 28 polarity-assigned
slots the planned stance mix is agree 0.286, uncertain 0.250, neutral 0.214,
disagree 0.143, mixed 0.107 — so **28.6% is forced affirmative, 14.3% forced
negative, and 57.1% still drawn free**. Projected realized affirmative share
**0.726 against a real 0.770**, a drift of −0.044, in the direction of *more* `no`
openings, which the generator under-produces anyway.

### Prediction

| quantity | v102 gate | predicted v103 | real |
|---|---:|---:|---:|
| polarity slots contradicting their plan | 2 of 10 | **0** | — |
| polarity slots changed for any other reason | — | **0** | — |
| realized affirmative share of polarity openers | — | 0.70–0.78 | 0.770 |
| `hard_disagree_rate`, thread | 0.1749 | **no change expected** | 0.1697 |
| realized `polarity_token` share | 0.0538 | no change expected | 0.0526 |

This arm is justified by **acceptance criterion 2**, not by p-values. Under
veto-and-redraw it changes the token on **only** the slots that contradicted
their plan, so no metric is predicted to move. `hard_disagree_rate` is the one to watch for an *unintended* move:
`no` has P(disagree) 0.203 and `yes` 0.462 in real text, so forcing affirmatives
onto agree slots could push it **up**. Against that, the affirmative share only
moves 0.770 → 0.726, and the slots in question are 2 of 186.

**A second prediction, from the gate's one unresolved question.** On the v102
gate the 23 drawn slots rose 0.076 → 0.157 in `mean_story_probability`, and the
two largest movers were both stance-conflict slots — one went 0.051 → **0.963**
("No. Measure the working distance first" → "yes, my call waits on holding it in
person"). If the conflict is what produced the deliberative first-person hedging,
**v103 should take part of that story rise back.** Story rose in 15 of 23 drawn
slots, which at n=23 is p ≈ 0.21 and not resolvable; the N=10 run resolves it at
≈230 drawn slots. **Do not reword the cue before then.**

### What else was checked before shipping, and what it killed

The v102 gate was re-audited end to end after the correction. Four hypotheses
were raised and **three were killed by measurement**:

- **"The family restriction leaves degenerate cells."** True — `neutral`+`disagree`
  and `polite`+`disagree` both collapse to a single token, `no`. **Not a defect:**
  in excluded real text those cells *are* degenerate, `neutral` negatives are
  10 of 10 `no` and `polite` negatives 20 of 22. Pooling the family across
  registers to restore variety would replace a faithful within-register rate with
  a cross-register one. Left alone deliberately.
- **"v102 under-produces negative openers, so the ban should not list `no`."**
  **Killed.** That reading came from one thread's 185 real comments (0.0216).
  Corpus-wide the real rate is **0.0113** and v102 produced **0.0108 — 0.96x
  real**, against v101's 3.81x. The ban list is right as it stands. A single-thread
  reference nearly caused a wrong change; see the 2026-08-20 granularity lesson.
- **"The token ban suppresses negation inside the comment body."** Aggregate looked
  like it: body negation 0.3920 → 0.3580 against a real 0.3933, density 1.308 →
  1.118 against 1.412. **Killed by the paired tests** the pairing demands —
  McNemar on 176 paired slots is 31 lost against 25 gained, **p = 0.504**, and
  Wilcoxon on per-comment density is **p = 0.653**. Not distinguishable from
  run-to-run churn at n=1. **Watch item for N=10**, where the same two tests run
  at ~1,700 paired slots; do not reword the ban before then.
- **"Removing the `Yeah,` opener removed the adjudication frame."** **Killed.**
  Reading the 19 leak-removal slots suggested it — six of them were
  `Yeah/Yep, that's the {good,nice,missing} part`. Measured, the frame is
  **v100 0.0806, v101 0.0645, v102 0.0806** against a real **0.0000** on that
  thread. The construction moved off the opener and did not go away. No win here;
  v100's finding that the phrase was never the thing still stands.

Also verified: **`--opening-move off` reproduces v101 exactly.** Every distinct
opener-rule line the v101 run actually rendered, extracted from its 532 saved
prompts, is reproducible with the arm off; none is missing.

### N=10 result — 2026-08-21

Run `generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1`, paired to
v101 on the same seeds, **$3.7345**, 56.8 minutes, 532/532 comments.

**9 PASS / 1 PARTIAL / 2 FAIL** — nominally the best count in project history
(v101 9/0/3). **Read the effect sizes instead.**

| metric | v101 Cliff | v103 Cliff | |
|---|---:|---:|---|
| `hard_disagree_rate` | +0.37 | **−0.23** | sign flipped — overshot |
| `impolite_rate` | +0.76 | **+0.61** | better |
| `neutral_rate` | −0.51 | **−0.30** | better |
| `self_bleu_4` | +0.44 | +0.40 | better |
| `length_cv` | +0.10 | **+0.04** | better |
| `emotion_entropy` | −0.06 | +0.02 | better |
| `polite_rate` | −0.62 | −0.60 | FAIL → PARTIAL, effect unchanged |
| `avg_depth` | +0.04 | +0.05 | structural |
| `structural_virality` | +0.02 | +0.02 | structural |
| `semantic_mean_cosine` | −0.06 | **−0.22** | **worse** |
| `mean_story_probability` | −0.06 | **−0.14** | **worse** |
| `self_bertscore_mean_f1` | +0.80 | **+0.86** | **worse** |

**Metrics inside the |Cliff| ≤ 0.10 bar: v101 6/12, v103 4/12** — but see the
CORRECTION below: that count conflates generator bias with template-selection
noise and is not a sound version comparison. `semantic_mean_cosine` and
`mean_story_probability` both left the safe zone. This is exactly the trap
§2 of `ORIENTATION.md` describes, and the reason that section exists.

### Both pre-registered watch items resolved as noise

The gate raised two questions that n=1 could not settle. Both were pre-registered
with the test to run, and both are now answered at 3-6x the sample:

| watch item | gate (n=1) | N=10 | verdict |
|---|---|---|---|
| story probability on the drawn slots | 0.076 → 0.157 on 23 slots, rose 15/23 | 0.0769 → **0.0552** on 65 slots, rose 33/65, Wilcoxon **p = 0.966** | did not reproduce |
| negation suppressed in the body | 0.3920 → 0.3580 on 176 slots | 0.4048 → **0.3988** on 504 slots (real 0.3933), McNemar **p = 0.881**, Wilcoxon **p = 0.383** | did not reproduce |

Not rewording the cue or the ban on the strength of the gate was the right call
in both cases.

### The mechanism did exactly what it was built to do

| quantity | v101 | v103 | measured |
|---|---:|---:|---:|
| polarity slots contradicting their plan | — | **0 of 28** | — |
| slots prepending an unassigned polarity token | 46 of 504 | **5 of 504** | — |
| `discourse_marker` obeyed | 0.184 | **0.974** | — |
| `polarity_token` obeyed | 0.893 | **1.000** | — |
| realized `polarity_token` share | 0.1335 | **0.0620** | 0.0526 |
| realized `discourse_marker` share | 0.0244 | **0.0846** | 0.0726 |

### And that is what broke the metric

`hard_disagree_rate` did not drift — it **overshot**, mean 0.1569 → **0.0920**
against a real 0.1208. Decomposed by pair kind:

| | v101 | v103 | matched real |
|---|---:|---:|---:|
| P(disagree \| root pair) | 0.0621 | **0.0284** | 0.0630 |
| P(disagree \| reply pair) | 0.2235 | **0.1657** | 0.1433 |
| pooled over all pairs | 0.1692 | **0.1198** | 0.1218 |
| **thread mean — the metric** | **0.1569** | **0.0920** | **0.1208** |

The reply conditional was repaired: +0.080 over real down to +0.022. **The root
conditional broke**: it was matched to three decimals in v101 and is now less than
half of real. The pooled figure looks excellent at 0.1198 against 0.1218 — that is
a **pass by cancellation**, and the metric is a mean of per-thread rates, not a
pooled rate, so the cancellation does not save it.

### CORRECTION (same day) — "overshot" was the wrong reading

Everything above about *what the mechanism did* stands. The interpretation of
`hard_disagree_rate` does not, and the priority order it implied was wrong.

**The Planner does not aim at the matched real thread.** It aims at a
**held-out, same-size real thread** — `reference_metric_template`,
`raw_text_included: false` — because aiming at the matched thread would be
tuning against the test set. So the per-thread target is an independent draw
from the real population, and at n=10 that is very noisy:

| | corr(generated, its template) | corr(generated, matched real) | corr(template, matched real) |
|---|---:|---:|---:|
| `hard_disagree_rate` | **+0.572** | −0.110 | **−0.281** |
| `impolite_rate` | +0.889 | +0.188 | +0.440 |
| `polite_rate` | +0.839 | +0.471 | +0.359 |

The generator tracks its target. The target carries essentially no information
about the thread it stands in for — which is by design.

**So compute the ceiling.** Cliff of the *template* against matched real is what
a perfect generator would score at n=10:

| metric | template ceiling | v101 | v103 | |
|---|---:|---:|---:|---|
| `hard_disagree_rate` | **−0.36** | +0.37 | **−0.23** | **beats the ceiling** |
| `length_cv` | +0.36 | +0.10 | +0.04 | beats it |
| `mean_story_probability` | +0.20 | −0.06 | −0.14 | beats it |
| `neutral_rate` | −0.23 | −0.51 | −0.30 | at it |
| `semantic_mean_cosine` | −0.16 | −0.06 | −0.22 | at it |
| `emotion_entropy` | +0.00 | −0.06 | +0.02 | at it |
| `self_bleu_4` | +0.00 | +0.44 | +0.40 | generator adds +0.40 |
| `impolite_rate` | +0.19 | +0.76 | +0.61 | generator adds +0.42 |
| `polite_rate` | −0.08 | −0.62 | −0.60 | generator adds −0.52 |
| `self_bertscore_mean_f1` | −0.08 | +0.80 | +0.86 | generator adds **+0.94** |

`hard_disagree_rate` at −0.23 is **closer to real than a perfect generator would
be** with these ten templates. It did not overshoot.

**The quantity that survives to N=150 is the generator's bias against its own
target, paired — template-selection noise cancels:**

| metric | v101 bias | v103 bias | Wilcoxon p (v103 ≠ 0) |
|---|---:|---:|---:|
| `hard_disagree_rate` | +0.0681 | **+0.0032** | 1.000 |
| `semantic_mean_cosine` | +0.0202 | **+0.0028** | 0.695 |
| `emotion_entropy` | −0.0956 | **+0.0138** | 0.770 |
| `neutral_rate` | −0.0357 | **−0.0165** | 0.688 |
| `impolite_rate` | +0.1893 | **+0.1529** | **0.002** |
| `polite_rate` | −0.2002 | −0.1856 | **0.002** |
| `self_bertscore_mean_f1` | +0.0196 | +0.0174 | **0.014** |
| `self_bleu_4` | +0.0051 | +0.0042 | 0.105 |
| `length_cv` | −0.1140 | −0.1193 | 0.275 |
| `mean_story_probability` | −0.0287 | −0.0332 | 0.084 |

**`hard_disagree_rate` went from +0.068 above its own target to +0.003 — the
generator converged, it did not overshoot.** Only **three** metrics carry a
statistically real generator bias: `polite_rate`, `impolite_rate` and
`self_bertscore_mean_f1`. Those are the three that have failed since v96.

**What this means for the "6/12 → 4/12" line above:** Cliff against matched real
at n=10 conflates generator bias with template-selection noise, and for
`hard_disagree_rate` the noise term is −0.36. That count is not a reliable
version comparison and should not have been reported as one.

**What still stands:** the root/reply opener inversion below is real and worth
fixing on its own terms — real text puts bare agreement tokens on replies at 3x
the rate it puts them on roots and the schedule does the opposite — but it is a
fidelity defect, not the cause of the `hard_disagree_rate` number.

### The upstream defect this exposed

Realized polarity-token openers, split by pair kind:

| | generated v103 | excluded real |
|---|---:|---:|
| on root comments | **0.0847** | 0.0224 |
| on replies | **0.0507** | 0.0685 |

**Inverted.** Real text puts its bare agreement tokens on replies at 3x the rate
it puts them on root comments; the schedule assigns them to roots at 0.0847 and
to replies at 0.0366. `opener_profile` measures a **pooled marginal** (0.0526)
and nothing makes the assignment respect the root/reply conditional.

Fed the real thread's own comment rows with their true depths,
`build_slot_distribution_schedule` does the right thing — 0 polarity openers on
roots, 0.0556 on replies — so `opener_cost` is not the bug. The inversion is in
the slot rows the Planner actually receives. Two candidates, both unverified:
the generated tree carries **0.335 root share against a matched real 0.267**, and
the depths reaching `_slot` may not be the template's.

**This defect is older than v102.** In v101 it was invisible because obedience was
0.18 and the leak added polarity openers to replies (realized 0.1493 on replies
against 0.1017 on roots), which accidentally pointed the right way. Making the
Writer obey a schedule exposed that the schedule was wrong.

> **The lesson, and it is the useful part of this run:** repairing a realization
> does not only move the metric — it makes the plan's own errors visible for the
> first time. A control that is 18% obeyed cannot be wrong in a way anyone can
> measure.

### Next

Do **not** ship another opener change against this. The next version should make
the opener schedule respect the root/reply conditional — measure
`opener_profile` per pair kind the way `register_realization` measures per
register — and the root-share question (0.335 against 0.267) has to be answered
first, because it moves the same metric on its own.

---

### Zero-API result

**566 tests pass** (8 new), Ruff clean, **105 pins with zero drift**, backend
self-test passes with the arm on and off. No profile change: schema stays 19 and
the v102 profile is reused unaltered, so v102 → v103 is a pure Writer-side
correction and the gate artifacts remain comparable.

The self-test was **proven to catch the defect**: with the family restriction
temporarily disabled it fails with
`AssertionError: ('agree', 'no', 'Opening grammar for this turn: polarity_token.
Open with the bare token "no" …')` — the exact v102 gate defect — and passes with
it restored.

---

## v102 — drawn opening move (2026-08-20)

Policy ID: `generalized-card-v2-drawn-opening-move-v102-20260820`.
Arm `--opening-move {measured,off}`; `off` reproduces v101.

**The schedule was already right and already rendered. It was not obeyed.**
`opener_profile` assigns a grammatical entry type per slot at the domain's
measured share, and the v101 prompts carried it faithfully — `polarity_token` on
28 of 532 slots, exactly the profile's 0.0526. The realization:

| planned | n | obeyed | where the rest went |
|---|---:|---:|---|
| `quote` | 18 | 1.000 | — |
| `conditional` | 15 | 1.000 | — |
| `first_person` | 100 | 0.960 | — |
| `polarity_token` | 28 | 0.893 | — |
| `question` | 21 | 0.476 | content_phrase 9 |
| `content_phrase` | 224 | **0.460** | noun_phrase 49, first_person 26, **polarity_token 21** |
| `imperative` | 10 | 0.400 | content_phrase 6 |
| `noun_phrase` | 59 | 0.254 | content_phrase 42 |
| `discourse_marker` | 38 | **0.184** | **polarity_token 19**, content_phrase 11 |
| `address` | 13 | 0.077 | content_phrase 9 |

`polarity_token` came out at **0.1274 against a measured 0.0526** and
`discourse_marker` at **0.0247 against 0.0726**. The `content_phrase` ↔
`noun_phrase` traffic is a harmless confusion between two content-bearing
classes; the damaging leak is narrow — **36 of 349 reply slots prepend an
agreement token they were not assigned.**

**Why it matters:** a `polarity_token` opening is the highest-disagreement entry
there is. On excluded real reply pairs, against a base rate of 0.180 —
`agreed` 0.882, `exactly` 0.800, **`yep` 0.778**, `same` 0.500, `yes` 0.462,
`yeah` 0.405, `no` 0.203 — while `thanks` is 0.037 and `thank` 0.055. The
generator concentrates on the bad end: `yeah` 31 and `yep` 21 of 71, so `yep`
runs at 0.30 of the class against a real 0.047.

Measured causally on an exact ablation harness that reproduces the shipped v101
artifact on 526/526 pairs with max |Δp| 0.000000 before it edits anything:
stripping only the *unassigned* polarity openers moves the reply-pair
`hard_disagree_rate` **0.2235 → 0.1862** against a matched real 0.1433 — 47% of
the gap — and moves `self_bleu_4` 0.03330 → 0.03297, down in 10 of 10 threads.
Full diagnosis in `tasks/v102-worklog.md`, reproducible with
`generalized_card/analysis/disagreement_diagnosis.py`.

### Why this is not another prohibition

**There already is one.** `_opener_rule` has appended "Do not open with a bare
agreement or disagreement token" to every non-`polarity_token` slot since v96. On
the v101 run it reached **504 of 532 prompts and was violated on 9.1% of them.**
Naming a category does not work in either direction — the same finding
`TONE_DEFINITIONS["polite"]` produced at 19.3% realization. What has worked here
is naming a concrete surface form: v98's "Use no semicolons" took the semicolon
0.109 → 0.023 and "Do not join two clauses with a dash" took the dash clause
0.299 → 0.071.

So the arm does two concrete things: it **draws the actual opening word** for the
two entry types whose category resolves to the wrong act and names it, and it
**replaces the categorical prohibition with the token list it is about**.

The draw is per register, because the opening connective is not register-neutral
and a flat table would tell a blunt correction slot to open with `Thanks`:

| register | discourse_marker share | the words |
|---|---:|---|
| polite | 0.0999 | thanks .33 thank .30 oh .08 so .07 well .05 |
| somewhat_polite | 0.0880 | thanks .42 oh .13 ah .12 but .09 so .09 |
| neutral | 0.0219 | and .31 also .22 but .22 well .18 so .08 |
| impolite | 0.0558 | well .23 and .16 oh .15 so .09 lol .07 |

Gratitude is 63% of the polite row and absent from the blunt one. A register the
profile does not measure gets **no** rule rather than a default.

### Prediction, with the population named

The v99 prediction failed because it predicted corpus rates from a corpus
baseline for a mechanism firing on 25% of slots. The drawn word fires on the
**67 of 532 slots** the schedule assigns `discourse_marker` or `polarity_token`
(12.6%); the named-token prohibition fires on the **504 of 532** not assigned
`polarity_token` (94.7%). Baselines below are measured on exactly those
populations.

| quantity | v101 | predicted v102 | real |
|---|---:|---:|---:|
| `discourse_marker` slots realizing it | 0.184 | **0.55–0.80** | — |
| realized `polarity_token` share, all slots | 0.1274 | **0.06–0.08** | 0.0526 |
| realized `discourse_marker` share, all slots | 0.0247 | **0.045–0.065** | 0.0726 |
| `yep` within polarity openers | 0.30 | 0.04–0.10 | 0.047 |
| reply-pair `hard_disagree_rate` | 0.2235 | **0.19–0.20** | 0.1433 |
| thread `hard_disagree_rate` | 0.1692 | **0.145–0.155** | 0.1218 |
| `hard_disagree_rate` Cliff | +0.37 | **+0.15 to +0.25** | — |

**This does not reach the |Cliff| ≤ 0.10 working ceiling and is not claimed to.**
The other half of the gap is parent echo — generated replies re-use the parent's
content words 1.4–1.6× as often as real ones, monotone against P(disagree) across
all six real bins and surviving conditioning on both parent and reply length in
all ten cells — and that has no mechanism yet.

The ablation's 0.1862 is **textual surgery, not regeneration**: a Writer told to
open differently writes a different sentence, not the same sentence minus
`Yeah,`. It is the direction and the order of magnitude, not a forecast.

**Guardrails.**
- `polite_rate` and `impolite_rate` must **not** move. The tone gap is flat across
  opener classes — real polite runs 0.18–0.47 and generated 0.02–0.15 in *every*
  class — so movement means the drawn word leaked into the register.
- No blunt slot may open on gratitude. Enforced by the per-register draw and
  asserted in the backend self-test.
- `self_bleu_4` must not rise. A named word repeats across slots, which is why
  `register_realization` names acts and never phrases; here the draw spreads over
  5–12 tokens per cell where the Writer's own default concentrated 73% of its
  polarity openers on two, and the surgery moved the metric down.
- `mean_story_probability`, `emotion_entropy` and `length_cv` should not move at
  all. The rule changes one word.

### Zero-API result

**559 tests pass** (22 new), Ruff clean, **105 pins with zero drift**, backend
self-test passes with the arm **on and off**.

Schema 18 → 19. The opening profile measures **15,294 comments over 424 excluded
threads with 0 seed overlap**, split polite 4,787 / impolite 6,538 / neutral
2,514 / somewhat_polite 1,455; every register carries both entry types, and a
cell under 40 comments is absent rather than defaulted.

Draw fidelity: rendered draw against measured share within **0.0108** in every
register, entry type and token, 4,000 slots per cell.

On the real prompt path, with the arm on, the rule renders the drawn word per
slot — two self-test slots drew `"ah"` and `"also"` — and the prohibition names
ten measured tokens. With the arm off it reverts to the categorical wording, so
`off` reproduces v101.

Domain generalization, real sampler, seed 42:

| domain | eligible | pool | reference | available | cells |
|---|---:|---:|---:|---|---:|
| camera | 441 | 150 | 291 | yes | 8/8 |
| cell_phone | 201 | 100 | 101 | yes | 7/8 |
| headphone | 177 | 100 | 77 | yes | 5/8 |
| laptop | 185 | 100 | 85 | yes | 3/8 |

The sparse domains lose cells rather than receiving a wrong word, which is the
correct degradation.

### Large-thread gate result — 2026-08-20

Run `generalized_card_camera_gpt54_v102_opening_seed8_20260820_v1`, seed index 8,
post `i1o51h`, **186 of 186 comments**, 0 degraded, 0 leaks, 338 requests,
**$1.1392**, 24.4 minutes. Compared against **v101's own row for the same thread**
(the seed-8 slice of its N=10 run) and that thread's real text, not against a
ten-thread average.

**Every prediction was beaten, and the mechanism did exactly what it named.**

| quantity | v101 (seed 8) | predicted | v102 | real | |
|---|---:|---:|---:|---:|---|
| `discourse_marker` slots realizing it | 0.231 | 0.55–0.80 | **0.923** | — | beat |
| realized `polarity_token` share | 0.1559 | 0.06–0.08 | **0.0538** | 0.0526 | beat |
| realized `discourse_marker` share | 0.0323 | 0.045–0.065 | **0.0753** | 0.0726 | beat |
| slots prepending an unassigned polarity token | 19 | — | **0** | — | — |
| reply-pair `hard_disagree_rate` | 0.228 | — | **0.1835** | 0.1688 | |
| thread `hard_disagree_rate` | 0.2022 | 0.145–0.155 | **0.1749** | 0.1697 | see below |

Compliance with a **named token** ran ≈1.0 where the same instruction as a
**category** ran 0.23. That is the sharpest demonstration in the project so far of
the standing finding, and it is a wider gap than v98's semicolon or v99's register
moves produced.

**My prediction band for the thread rate was computed against the wrong
population** — 0.145–0.155 came from the N=10 pooled real (0.1218), while this
thread's real is 0.1697. Scaled to this thread the ablation implied ≈0.168 and the
run gave 0.1749. The band was wrong; the mechanism was not. That is the same
granularity error the 2026-08-20 lesson is about, made once more in the
prediction rather than in the code.

### Distance on every metric, against the same thread's real row

| metric | real | v100 | v101 | v102 | err v101 → v102 |
|---|---:|---:|---:|---:|---|
| `hard_disagree_rate` | 0.1697 | 0.2350 | 0.2022 | **0.1749** | 19.1% → **3.0%** |
| `neutral_rate` | 0.1622 | 0.0919 | 0.1148 | **0.1398** | 29.2% → **13.8%** |
| `emotion_entropy` | 1.9459 | 1.6461 | 1.5356 | **1.6867** | 21.1% → **13.3%** |
| `polite_rate` | 0.2324 | 0.0757 | 0.0820 | **0.1075** | 64.7% → **53.7%** |
| `impolite_rate` | 0.4649 | 0.6973 | 0.6503 | **0.6237** | 39.9% → **34.2%** |
| `semantic_mean_cosine` | 0.1865 | 0.1907 | 0.2241 | 0.1937 | 20.2% → 3.9% |
| `length_cv` | 0.8951 | 0.9254 | 0.8769 | 0.9046 | 2.0% → 1.1% |
| `self_bertscore_mean_f1` | 0.4887 | 0.5063 | 0.5090 | 0.5076 | 4.2% → 3.9% |
| `avg_depth` | 3.6000 | 3.5892 | 3.6120 | 3.5914 | 0.3% → 0.2% |
| `structural_virality` | 4.5608 | 4.5508 | 4.5861 | 4.5457 | 0.6% → 0.3% |
| `self_bleu_4` | 0.0283 | 0.0353 | 0.0343 | 0.0347 | 21.4% → 22.8% |
| `mean_story_probability` | 0.1114 | 0.1439 | 0.1222 | 0.1283 | 9.7% → 15.2% |

`hard_disagree_rate` at 3.0% relative error is the closest this metric has ever
come on a large thread. The reply-pair decomposition confirms the route: the
reply conditional went 0.228 → 0.1835 against a real 0.1688, and **0 of 158 reply
slots prepended an unassigned polarity token**, against 19 in v101.

### The guardrail I got wrong, and why

I wrote that `polite_rate` and `impolite_rate` **must not move**, on the grounds
that the tone gap is flat across opener classes. They moved, favourably, and the
reasoning behind the guardrail was wrong.

A natural experiment attributes it. The arm's positive cue fires on the 23 slots
the schedule assigned one of the two drawn types; the leak removal touches the
rest:

| slot group | n | polite v101 → v102 | impolite | neutral |
|---|---:|---|---|---|
| assigned a drawn type | 23 | 0.000 → **0.174** | 0.652 → 0.652 | 0.130 → 0.000 |
| every other slot | 163 | 0.094 → 0.098 | 0.650 → **0.620** | 0.113 → **0.160** |

The polite gain is **entirely on the arm's own slots** — the polite-register draw
is `thanks`/`thank` 63% of the time, and polite-guard keys on warmth markers. The
neutral gain is **entirely on the other slots**, where removing `Yeah,` moved
comments off `impolite`.

The error in my reasoning: I checked that the *conditional* P(polite | opener
class) gap was flat and concluded the opener could not move `polite_rate`. But
changing the opener changes which class a comment is **in**, and the classes have
very different real polite rates — `discourse_marker` is the most polite class at
0.466. That is the prevalence-versus-conditional distinction the v99 work is built
on, applied correctly to `hard_disagree_rate` and then forgotten one metric later.

### The two regressions, honestly

**`self_bleu_4` +0.0004 is not the drawn words.** The risk named in the module
docstring was that naming a word repeats it. Measured with the exact scorer: v101
carried `"yep , that's the"` 4× and `"yep , thanks ."` 2× as repeated opening
4-grams, and v102 has **13 comments sharing an opening 4-gram against v101's 14**.
Trimming the first word from every comment *raises* `self_bleu_4` in both runs
(v101 −0.00013, v102 −0.00024), so the opening is a diversity contributor, not a
repetition source. The arm removed the repeating openers and the metric still
ticked up 0.0004 for an unattributed reason. At n=1 that is noise; the exact
surgery on the N=10 pool gave −0.0003.

**`mean_story_probability` +0.006 is located on the arm's slots but is not
attributable.** The 23 drawn slots went 0.076 → 0.157 while the other 163 went
0.129 → 0.124. But **the 23-slot mean is carried by 3 comments** — 0.963, 0.813,
0.666 — and the **median of the 23 is 0.063**. A 3-comment swing at n=23 is
exactly the case a claim was retracted for on 2026-08-20. Recorded as unresolved,
not as an effect. The N=10 run resolves it at n≈230 drawn slots; do not reword the
cue against this.

### Content, read comment by comment

All 23 drawn slots obeyed their word and the words are varied — `also`, `ah`,
`oh`, `so`, `personally`, `lol`, `thanks`, `same`, `agreed`, `yes`, `no`, `yeah` —
against v101's 19 near-identical `Yeah,` / `Yep,` openings on the same thread.
That is the criterion-2 gain this arm was for. One slot missed its word
("Pretty much" for an assigned `also`), which is the 1 of 13.

**A pre-existing tell surfaced while reading, unrelated to this arm.** Comments
carry a bare `0` or `1` as a word — "0 verdict from me", "wrap 1 hand around it",
"Ask what 1 thing you're giving up" — at **0.140 in v102 and 0.151 in v101**
against **0.071** in excluded real. `sentence_rhythm`'s digit cue asks for "a
figure rather than described in words", and the model supplies a figure where a
person writes the word. Caveat on the probe, per the 2026-08-19 lesson: the real
0.071 includes legitimate decimals such as "0.1% of consumers", so the true real
rate of the unnatural form is lower and the ratio is worse than 2×. Recorded as a
lead for criterion 2, not acted on here.

### Next

**Go to N=10.** Nothing on this gate calls for a correction first: the metric it
targeted improved to 3.0% relative error, three more improved, and both
regressions are noise-level with one of them contradicted by its own
mechanism-level check. Command paired to v101:

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag generalized_card_camera_gpt54_v102_opening_n10_20260820_v1 \
  --domain camera --model gpt-5.4-mini \
  --base-url https://api.openai.com/v1 --api-key-env LLM_API_KEY \
  --pool-size 150 --max-posts 10 --posts-per-run 5 \
  --start-seed-index 2 --sampling-seed 42 --opening-move measured --resume
```

---

## v101 — per-register realization, and a state-not-event correction (2026-08-20)

Policy ID: `generalized-card-v2-per-register-realization-v101-20260820`.
Same arm, `--register-realization {measured,off}`; `off` reproduces v98.

**Fixes the scoping error the v100 gate exposed.** v99 restricted the register
moves to `polite` because no move *discriminates* the other labels. That was the
wrong test: discrimination and rate-matching are different questions, and reading
real is rate-matching. Real comments of every register carry these moves.

| real label | any_intensifier | plain_verdict | own_thing | love_like |
|---|---:|---:|---:|---:|
| polite | 0.485 | 0.393 | 0.375 | 0.142 |
| somewhat_polite | 0.324 | 0.202 | 0.184 | 0.027 |
| neutral | 0.130 | 0.070 | 0.108 | 0.004 |
| impolite | **0.300** | 0.128 | **0.182** | 0.026 |

On the v100 gate, three of the four moves were at **exactly zero** on every
non-polite register, and `any_intensifier` on planned-impolite slots was 0.100
against a real 0.300. Decomposed for that move: polite slots +0.059 at weight
0.25, every other slot **−0.170 at weight 0.75**.

The profile is now per-register (schema 17 → 18), measured over all 15,294
excluded real comments rather than the 4,787 polite ones. The per-register tables
are strongly differentiated — long blunt comments are heavily marked (`impolite`
essay: intensifier 0.946, verdict 0.732) while `neutral` micro is genuinely bare
at 0.016 — so this is not one table applied four times.

`plain_verdict`'s cue was reworded to hold in any register: "Name one thing here
that is plainly good … even if your overall judgement is negative." Inside a blunt
turn that is the concession real blunt comments make at 0.128; inside a warm one
it is an appraisal. The rule is now labelled "Register, realized" rather than
"Warm register", because naming a register would tell a blunt slot to soften. A
test asserts no cue names one.

### The v100 evaluation, and the second correction it forced

n=1 so every metric reads `DESCRIPTIVE`; the readable quantity is the relative
error against the same thread's matched real row.

| metric | v98 err | v100 err | |
|---|---:|---:|---|
| `emotion_entropy` | 33.1% | **15.4%** | better |
| `length_cv` | 8.4% | **3.4%** | better |
| `hard_disagree_rate` | 44.9% | 38.5% | better |
| `self_bertscore_mean_f1` | 4.4% | 3.6% | better |
| `semantic_mean_cosine` | 3.8% | 2.2% | better |
| `polite_rate` | 60.3% | **67.4%** | **worse** |
| `impolite_rate` | 42.6% | **50.0%** | **worse** |
| `mean_story_probability` | **0.8%** | **29.2%** | **worse** |
| `self_bleu_4` | 21.3% | 25.0% | worse |

**The run contains a natural experiment that attributes the regressions.** The
closing cue renders only where the matched slot is 25 words or longer, so the
shorter slots received no closing instruction in the same run, same model, same
thread:

| slot group | n | polite v98→v100 | impolite v98→v100 | story v98→v100 |
|---|---:|---|---|---|
| closing cue rendered (≥25w) | 107 | 0.112 → 0.112 | 0.710 → **0.692** | 0.145 → 0.173 |
| no closing cue (<25w) | 79 | 0.065 → **0.026** | 0.597 → **0.705** | 0.067 → **0.105** |

**The regression is worst where the closing arm never fired**, so that arm is not
the cause and on its own slots it slightly *improved* `impolite`. Short
planned-polite slots fell from 0.326 realized polite to 0.174, and reading them
shows why: at 15 words the model spends its budget satisfying the cue and returns
a clipped technical fragment — "Yep, fair. Plain flat target first",
"Front-element glare off a polarizer is a nice quick check" — instead of a warm
reaction.

**Both of my cues were written as events, and got events.**

| | with a possessive | without | possessive prevalence |
|---|---:|---:|---:|
| real, story probability | **0.279** | 0.128 | **0.230** |
| v98 | 0.618 | 0.071 | 0.075 |
| v100 | 0.510 | **0.112** | 0.081 |

Possessive prevalence barely moved (0.075 → 0.081), and the story rise is mostly
in the comments *without* one (0.071 → 0.112). Real text uses the possessive as a
bare fact — "my copy is junk" — at 0.230 with story probability 0.279; generated
text puts it inside a narrative and reaches 0.510. The two cue texts named events:
"what you ended up keeping" and "how long you have had it, what it did or did not
do". Both now name a state and rule out the recounting, with a test on each
asserting the event wording is gone.

Also recorded: generated possessive prevalence is 0.081 against a real 0.230, and
the real conditional is P(polite | possessive) = 0.509 against 0.254 without. That
is the largest single untapped lever on `polite_rate` — but only if the possessive
arrives as a fact rather than a story, which is exactly what this correction is
for.

### Prediction, with the arithmetic shown

The v99 prediction failed because I predicted corpus rates from a corpus baseline
for a mechanism firing on 25% of slots. Weighting by the gate thread's planned
tone mix (polite 0.247, somewhat 0.070, neutral 0.199, impolite 0.484) and by the
compliance measured on that gate:

| move | v100 | full rate-match | measured compliance | predicted | thread real |
|---|---:|---:|---:|---:|---:|
| `any_intensifier` | 0.204 | 0.314 | 0.70 | **0.26–0.30** | 0.373 |
| `plain_verdict` | 0.054 | 0.187 | 0.40 | **0.09–0.13** | 0.200 |
| `own_thing` | 0.081 | 0.215 | 0.33 | **0.11–0.15** | 0.168 |
| `love_like` | 0.016 | 0.050 | 0.00 | **0.02–0.04** | 0.043 |

`love_like` is the low-confidence one: 3 slots were asked for it on the gate and 0
realized it, so its cue may simply not work.

**`polite_rate` is not predicted to move much and `impolite_rate` even less.**
This arm is aimed at what the text reads like, and it deliberately raises the
positive-register vocabulary on blunt slots, which is what real blunt comments do
but is not what raises `polite_rate`. Guardrail: `impolite_rate` must not *fall*
below the real 0.443 — asking blunt slots to concede one good thing could soften
them past the target, and the plan's tone marginal is already correct.

### Zero-API result

532 tests pass, Ruff clean, 104 pins with zero drift, self-test passes with the
arm on and off. Schema-18 profile over 424 excluded threads, 0 seed overlap,
15,294 comments split polite 4,787 / impolite 6,538 / neutral 2,514 /
somewhat_polite 1,455; `neutral` has no `very_long` or `essay` band and is
correctly absent there rather than defaulted.

Draw fidelity within **0.013** in every register, band and move. Rendered prompts
carry the rule on 23/24 polite slots (9 distinct forms), 22/24 somewhat_polite
(7), 21/24 impolite (5), 17/24 neutral (4).

---

### N=10 result — 2026-08-20

Run `generalized_card_camera_gpt54_v101_register_n10_20260820_v1`, paired to v97,
v98 and v100 on the same seeds (`--start-seed-index 2`, `--sampling-seed 42`).

**9 PASS / 0 PARTIAL / 3 FAIL — the best result in the project's history**
(v98 was 8/1/3, v97 7/1/4, v96 6/0/6). Cliff's delta improved on **8 of 12**
metrics.

| metric | v98 MWU | v98 Cliff | v101 MWU | v101 Cliff | |
|---|---:|---:|---:|---:|---|
| `self_bleu_4` | 0.1212 | +0.42 | 0.1041 | +0.44 | PASS, weaker |
| `self_bertscore_mean_f1` | 0.00058 | +0.92 | 0.00283 | **+0.80** | FAIL, improved |
| `semantic_mean_cosine` | 0.6232 | −0.14 | 0.8501 | **−0.06** | PASS |
| `hard_disagree_rate` | 0.2897 | +0.29 | 0.1735 | +0.37 | PASS, worse |
| `polite_rate` | 0.0126 | −0.67 | 0.0210 | **−0.62** | FAIL, improved |
| `impolite_rate` | 0.0010 | +0.88 | 0.0046 | **+0.76** | FAIL, improved |
| `neutral_rate` | 0.0210 | −0.62 | 0.0587 | **−0.51** | PARTIAL → **PASS** |
| `length_cv` | 0.4727 | +0.20 | 0.7337 | **+0.10** | PASS |
| `avg_depth` | 0.9698 | +0.02 | 0.9095 | +0.04 | PASS |
| `structural_virality` | 0.9697 | +0.02 | 0.9697 | +0.02 | PASS |
| `mean_story_probability` | 0.6776 | −0.12 | 0.8501 | **−0.06** | PASS |
| `emotion_entropy` | 0.5708 | −0.16 | 0.8501 | **−0.06** | PASS |

**The state-not-event correction worked.** `mean_story_probability` was pushed to
29.2% relative error on the v100 gate by cue text that named events; rewritten to
name states, its Cliff came back to −0.06, better than v98's −0.12. That is the
clearest single confirmation in this version line that cue *wording* is a
measurable control, not a formality.

**`gratitude` restored per register and band was worth `neutral_rate`.** It moved
PARTIAL → PASS (MWU 0.021 → 0.059), and `polite_rate` and `impolite_rate` both
improved on effect size for the first time in four versions.

### The honest N=150 projection

At N=150 the pass probability is a function of the effect size, not the current
p-value (see the 2026-08-19 lesson). Applying that table to these Cliffs:

| |Cliff| | metrics | P(pass) at N=150 |
|---|---|---:|
| ≤0.06 | `semantic_mean_cosine`, `mean_story_probability`, `emotion_entropy`, `structural_virality`, `avg_depth` | ~0.90 |
| 0.10 | `length_cv` | ~0.72 |
| 0.37–0.44 | `hard_disagree_rate`, `self_bleu_4` | ~0.01 |
| 0.51–0.80 | `neutral_rate`, `polite_rate`, `impolite_rate`, `self_bertscore_mean_f1` | ~0.00 |

**Six metrics are safe at N=150 and six are not.** The four currently-failing or
barely-passing tone metrics plus `self_bleu_4` and `hard_disagree_rate` all sit
far above the |Cliff| ≤ 0.10 working ceiling. 9/0/3 at N=10 is real progress and
is not the same thing as being close at N=150.

`hard_disagree_rate` moved the wrong way (+0.29 → +0.37) and passes on a wide
spread, exactly like `self_bleu_4`. **It has since been diagnosed** — see
`tasks/v102-worklog.md`, reproducible with
`generalized_card/analysis/disagreement_diagnosis.py`. Root pairs already match
(0.0621 against a real 0.0630); reply pairs are 1.56× real and are the whole gap.
Two mechanisms survived falsification and nine did not; the opener-realization
one is causally measured at 47% of the reply gap on an exact ablation harness.

---

## v100 — measured closing move (2026-08-20)

Policy ID: `generalized-card-v2-measured-closing-move-v100-20260820`.
Arm `--closing-move {measured,off}`; `off` reproduces v99.

**The root of the adjudication frame, found after three Planner-side rejections.**
The "that's the part that actually matters" family has been chased since v73
through a phrase ban, a rewording, a route lock, a prompt rebuild, and v97's
adjudication gate. It survived all five because the phrase was never the thing.

Measured on the last sentence only, comments of 25 words or more:

| closing move | real | generated | ratio |
|---|---:|---:|---:|
| abstract verdict | 0.014 | **0.265** | **19.1x** |
| a concrete fact of the speaker's own | 0.152 | 0.048 | **0.32x** |
| a figure in the last sentence | 0.318 | 0.200 | 0.63x |
| a conditional about the reader | 0.095 | 0.145 | 1.53x |

The verdict close is the 19x defect. **The reader-conditional close is only
mildly over-produced and is deliberately left alone** -- real people do end that
way at 0.095, and it reads worse than it measures.

Real endings: "No issues yet about 40,000 clicks in." / "I was using a 1dxm3
myself." / "It had 1mil for it's shutter count before it needed service." /
"so yeah, I know how you feel". Generated endings: "If the current feature set
lines up with your routine, the age starts to matter a lot less" / "That's
probably the real separator" / "So yeah, my take is just to wait".

**Three Planner-side explanations measured and rejected first**, so the cause is
not a control the Writer is echoing:

- the rendered "decision intent" line -- lift on the frame 1.08x
- the rendered "decision boundary" line -- **0.83x**, slots receiving it produce
  the frame *less*
- v97's adjudication gate -- gated slots 0.175 against ungated 0.210, so the gate
  works and the frame is not coming from those lines

What does predict it is the payload the Planner assigned: `personal_story` 0.448,
`correction` 0.212, `advice` 0.210, against `low_info_reaction` 0.071 and
`bare_answer` 0.091. Among real story comments the frame is at **0.003** against
0.382 generated -- 127x. A story has the most obvious place to pivot, so it pivots
most. The Writer reaches for a verdict because it has no other way to stop.

### Also measured and NOT acted on

- Suppressing the whole restrictive register (`just`/`only`/`still`/`actually`)
  would **not** fix `impolite_rate`: out-of-sample lift 1.02-1.18x, and the
  counterfactual moves 0.697 to 0.655 against a real 0.443.
- Banning the frame's exact phrasing removes only **15-27%** of the over-used
  abstract vocabulary; after removing every frame match, `matters` is still at
  33x, `whole` 17x, `otherwise` 29x. Which is why five phrase-level attempts
  failed and why this version names the move instead.
- **`self_bertscore_mean_f1` hypothesis five rejected.** The narrow shared
  vocabulary does not explain it: per matched pair, r(bert gap, breadth ratio) =
  **+0.155** and r(bert gap, top-200 concentration gap) = **-0.096**, both the
  wrong sign, and the narrowest thread has the smallest gap. Per-thread breadth
  ratio is 0.893, not the 0.76 the pooled figure suggested -- the narrowness is a
  cross-thread phenomenon and `self_bertscore` only sees within-thread. The metric
  still has no verified mechanism.

### Prediction, written before the paid run

| quantity | v99 | predicted v100 | real |
|---|---:|---:|---:|
| verdict close | 0.265 | 0.02-0.05 | 0.014 |
| own-concrete close | 0.048 | 0.11-0.16 | 0.152 |
| broad adjudication frame, anywhere | 0.203 | 0.05-0.10 | 0.004 |
| frame on `personal_story` slots | 0.448 | 0.10-0.20 | 0.003 |
| `impolite_rate` | 0.697 | 0.62-0.68 | 0.443 |

**`impolite_rate` and `polite_rate` are still predicted to fail.** This arm is
justified by acceptance criterion 2 -- a human should not be able to tell
generated from real -- not by the p-values. The 19x verdict close and the 47x
frame are the largest remaining content tells; the metric effect is secondary and
measured to be small.

Guardrails: `mean_story_probability` must not fall (a story that ends on a
concrete own fact should read *more* like a story, and it is already slightly low
at Cliff -0.12), and the concrete-close cue must not invent a possession -- it is
bounded by the plan's fact license, which is the boundary most likely to leak.
Interaction with v99: "commit to a positive judgement" and "do not close on a
verdict" are compatible but adjacent, so the gate must confirm polite slots still
land their judgement rather than dropping it.

### Zero-API result

526 tests pass (31 new), Ruff clean, 104 pins with zero drift. Self-test passes
with the arm on and off. Schema 16 -> 17; the closing profile measures 6,609
comments of 25+ words over the 424 excluded threads with 0 seed overlap:

| band | n | own-concrete | verdict | median final words |
|---|---:|---:|---:|---:|
| medium | 3579 | 0.1492 | 0.0126 | 15 |
| long | 1999 | 0.1476 | 0.0115 | 16 |
| very_long | 840 | 0.1286 | 0.0167 | 17 |
| essay | 191 | 0.1152 | 0.0314 | 18 |

Draw fidelity within 0.008 in every band and move. In rendered prompts the rule
reaches 32 of 40 slots, is silent on every slot below the 25-word floor (a slot
whose last sentence is its whole body has no closing move), and 8 of 8 at 45
words and above.

---

### Large-thread gate result — 2026-08-20

Run `generalized_card_camera_gpt54_v100_closing_seed8_20260820_v1`, seed index 8,
post `i1o51h`, **186 of 186 comments**, 0 degraded, 0 empty, 343 requests,
`$1.1421`, 23.7 minutes. Both v99 and v100 arms at `measured`. Compared against
v98's output for the *same* thread and the same thread's real text, not against a
ten-thread average.

**Both arms underperformed their predictions, and the reason is one analytical
error of mine plus one mechanism aimed at the wrong slots.**

| quantity | v98 | predicted | v100 | real | |
|---|---:|---:|---:|---:|---|
| verdict close (25w+) | 0.271 | 0.02–0.05 | **0.150** | 0.009 | missed |
| own-concrete close (25w+) | 0.047 | 0.11–0.16 | 0.103 | 0.218 | just short |
| broad adjudication frame | 0.183 | 0.05–0.10 | **0.156** | 0.000 | missed |
| frame on story slots | 0.412 | 0.10–0.20 | **0.471** | 0.003 | **rose** |
| `any_intensifier` | 0.258 | ~0.45 | **0.204** | 0.373 | **fell** |
| `plain_verdict` | 0.043 | ~0.28 | 0.054 | 0.200 | barely moved |
| `own_thing` | 0.075 | ~0.30 | 0.081 | 0.168 | barely moved |
| `love_like` | 0.011 | ~0.12 | 0.016 | 0.043 | barely moved |

**The v99 prediction was arithmetically impossible and I should have caught it.**
The arm fires only where the plan assigned `polite`, which is 25% of slots. I
predicted corpus-wide rates from a corpus-wide v98 baseline. On the slots the cue
actually reaches, v98 was *already* at 0.565 for `any_intensifier` — above the
real corpus rate of 0.373 — so there was nothing there to fix. Isolated to its own
slots the arm did work, modestly: `plain_verdict` 0.152 → 0.217, `love_like`
0.043 → 0.065, `own_thing` 0.217 → 0.239.

**The deficit is on the slots v99 deliberately excludes.** Real move rates by the
classifier's own label show every register carries these moves — real *impolite*
comments carry `any_intensifier` at 0.300 and `own_thing` at 0.182:

| real label | any_intensifier | plain_verdict | own_thing | love_like |
|---|---:|---:|---:|---:|
| polite | 0.485 | 0.393 | 0.375 | 0.142 |
| somewhat_polite | 0.324 | 0.202 | 0.184 | 0.027 |
| neutral | 0.130 | 0.070 | 0.108 | 0.004 |
| impolite | 0.300 | 0.128 | 0.182 | 0.026 |

Generated, by planned tone: polite 0.543 / 0.217 / 0.239 / 0.065 (the intensifier
*overshoots*), somewhat_polite 0.154 / **0.000** / **0.000** / 0.000, neutral
0.054 / 0.000 / 0.081 / 0.000, impolite 0.100 / **0.000** / **0.011** / 0.000.
Decomposed for `any_intensifier`: polite slots +0.059 at weight 0.25, other slots
**−0.170 at weight 0.75**.

**The scoping error behind it.** v99 restricted itself to `polite` because no move
*discriminates* the other labels — every candidate scored a held-out lift below
0.3 for `neutral`. That was the wrong test. Discrimination and rate-matching are
different questions: a move can fail to predict a label and still be the rate real
comments of that label carry. Making text read real is rate-matching.

**v100 partially worked and did not do what its cue named.** The frame in the last
sentence is **unchanged** at 0.075, while the frame in the body fell 0.196 →
0.131 and the looser verdict-close vocabulary fell 0.271 → 0.150. So the cue
reduced `matters` / `my take` / `bottom line` in the close by 45% but left the
specific `that's the part that …` construction exactly where it was. The frame did
not displace out of the close — it was never concentrated there.

**Content did improve, which is the criterion this arm was for.** v100 story
endings: "I've been keeping mine in the bag for a few months now." / "Still got
mine." / "I ran into that myself looking at Canon stuff and it immediately stopped
being a body question." Against v98 on the same thread: "If the current feature
set lines up with your routine, the age starts to matter a lot less." One moral
closer survived: "It's a good reminder that the practical side shows up in the
shoot itself."

Unchanged and untargeted: `check` at 3.735 per 1,000 tokens against a real
**0.000** on this thread, `will` at 0.000 against 2.776, `very` at 0.05× real.

**What follows.** Extend the register moves to every tone class, drawn at that
class's own measured rate, and cap rather than push where generated already
overshoots. The profile machinery already does per-band draws; it needs per-label
bands. That is derived from this gate rather than guessed, and it addresses the
75% of the corpus that carries the deficit.

---

## v99 — drawn realization of the assigned warm register (2026-08-20)

Policy ID: `generalized-card-v2-drawn-register-realization-v99-20260820`.

**Hypothesis.** The plan already places the polite register correctly -- 0.275
planned against a real 0.288 -- and the Writer realizes it 19.3% of the time
while realizing `impolite` 89.7%. `TONE_DEFINITIONS["polite"]` describes a
register in prose; `sentence_rhythm` showed that a concrete surface act drawn per
slot reaches the output and a prose description does not. So the fix is to ask a
polite-assigned slot for the surface moves real polite comments of its size
actually carry.

Diagnosis, four rejected hypotheses, and every number below: `tasks/v99-worklog.md`.

### The measured profile (schema 15 -> 16)

Share of the 4,787 evaluation-classifier `polite` comments in the 424 excluded
threads carrying each move, by size band. Zero seed overlap.

| band | n | any_intensifier | plain_verdict | own_thing | love_like |
|---|---:|---:|---:|---:|---:|
| micro | 557 | 0.127 | 0.226 | 0.045 | 0.052 |
| short | 684 | 0.200 | 0.297 | 0.190 | 0.114 |
| medium | 1268 | 0.391 | 0.342 | 0.323 | 0.127 |
| long | 1399 | 0.615 | 0.417 | 0.486 | 0.152 |
| very_long | 704 | 0.842 | 0.580 | 0.608 | 0.227 |
| essay | 175 | 0.937 | 0.731 | 0.703 | 0.229 |

Three measured candidates were deliberately excluded, each for a stated reason:
`gratitude` (generated already runs 1.25x real, and its band curve runs backwards
to every other move), `reassure_you` (real prevalence 0.023, too rare to spend a
slot on), and `link` (generated 0.000 against a real 0.058 -- a real gap and an
eye-visible tell, but a link needs a real URL and inventing one is a hard
failure). `intensified_positive` is the conjunction of two moves already cued.

### Prediction, written before the paid run

| quantity | v98 | predicted v99 | real |
|---|---:|---:|---:|
| realization of planned polite | 0.193 | 0.45–0.60 | — |
| `polite_rate` | 0.070 | 0.14–0.19 | 0.288 |
| `impolite_rate` | 0.697 | 0.60–0.63 | 0.443 |
| `any_intensifier` prevalence | 0.288 | ~0.45 | 0.417 |
| `plain_verdict` prevalence | 0.085 | ~0.28 | 0.215 |
| `own_thing` prevalence | 0.117 | ~0.30 | 0.212 |
| `love_like` prevalence | 0.015 | ~0.12 | 0.064 |

**`polite_rate` and `impolite_rate` are predicted to still fail.** This arm
repairs the polite realization only. The remaining gap is the planned-neutral
(0.513) and planned-somewhat_polite (0.478) bleed into impolite, which is 122
slots and needs a *suppressive* mechanism -- no additive move discriminates
`neutral` at all, every candidate scoring a held-out lift below 0.3. Shipping
this separately keeps the two attributable.

Guardrails to check, not just the target metrics: `self_bleu_4` must not pass
Cliff +0.45 (a cue vocabulary can repeat -- every cue names an act, never a
phrase, and a test asserts it), and `emotion_entropy`'s dominant-emotion
histogram must not concentrate on `approval`/`admiration`.

### Arm

| flag | v99 default | reproduces v98 |
|---|---|---|
| `--register-realization` | `measured` | `off` |

Written into `run_config.json`, in `RUN_EXPERIMENT_FIELDS`, checked by the
resume-config verification.

### Zero-API result

495 tests pass (28 new), Ruff clean, 103 pins with zero drift, no untracked
active source, no unpinned local import. Self-test passes with the arm on and
off. The schema-16 profile rebuilds over 424 excluded threads with 0 seed
overlap and 4,787 polite comments measured.

**Draw fidelity**, 4,000 slots per band, rendered rule against measured share --
every move in every band within 0.011:

| band | any_intensifier | plain_verdict | own_thing | love_like |
|---|---|---|---|---|
| micro | 0.127 / 0.127 | 0.231 / 0.226 | 0.043 / 0.045 | 0.048 / 0.052 |
| medium | 0.399 / 0.391 | 0.344 / 0.342 | 0.320 / 0.323 | 0.126 / 0.127 |
| long | 0.625 / 0.615 | 0.419 / 0.417 | 0.487 / 0.486 | 0.152 / 0.152 |
| essay | 0.936 / 0.937 | 0.742 / 0.731 | 0.692 / 0.703 | 0.224 / 0.229 |

Cues per slot scale with the band as the deficit does: micro 0.45, medium 1.19,
long 1.68, essay 2.61. In rendered Writer prompts the rule appears in 33 of 40
polite slots with the arm on and 0 of 40 with it off, in 6 distinct forms, at a
cost of +283 characters (+7.7%).

### Verification gate before N=10

Per the standing protocol, one paid **large-thread** gate first --
`--start-seed-index 8`, post `i1o51h`, 186 comments, inside the N=10 window so
the row is directly comparable. Judge it on content and on distance. The specific
things to read, because each is a prediction:

1. Realized move prevalence tracking the table above, not exceeding it.
2. Planned-polite slots reading as committed positive judgement rather than a
   trade-off, especially in the `very_long` and `essay` bands.
3. No invented possession from the `own_thing` cue -- it is bounded by the plan's
   own fact license and that boundary is the thing most likely to leak.
4. No repeated cue phrasing across slots; the words should differ even where the
   drawn move is the same.
5. `mean_story_probability` still in range -- `own_thing` plus past tense raises
   it, and it is currently slightly low at Cliff -0.12.

---

## Reproducibility infrastructure (2026-08-20)

**Not a generator version.** No prompt, control, or distribution changes; the
policy version is unchanged at
`generalized-card-v2-drawn-typing-rhythm-length-calibration-v98-20260819`. It is
recorded here because it changes what shipping a version requires.

`generalized_card/generalized_card/source_provenance.py` (110 lines) is called
from `run_generate.py` immediately after `verify_core_contract` and before the
seed pool, the domain profile, and every API call. A run **refuses to start** if
any file defining the version is not in `HEAD` -- modified, staged but never
committed, or untracked. `core_contract.version_source_paths` assembles the list:
the pinned generation sources plus the contract itself, which can never carry its
own hash and so had no other check. The commit, branch, and any offending paths
are written to `run_config.json` as `source_provenance`.

Override: `GENERALIZED_CARD_ALLOW_UNCOMMITTED_SOURCE=1`, an environment variable
rather than a flag so it cannot be set by accident inside a long command. Using
it records `override: true` in the artifact, so a run made without provenance
says so rather than looking like every other run.

**Why.** v97 and v98 both shipped uncommitted; `git log` on `sentence_rhythm.py`
was empty while the N=10 result was being quoted as the project's state. The near
miss is the instructive part: `repin_core_contract.py` already refused to pin a
file `git ls-files` did not know about, and reported `untracked active: 0`
throughout -- because `git ls-files` lists tracked files and a staged file is
tracked. Every check in place answered "has this drifted?"; none answered "can
this be recovered?"

19 tests, each building a real throwaway repository rather than mocking git,
since the defect was a wrong belief about what a git command reports. One is a
live guard, `test_this_repository_is_currently_reproducible`, which fails whenever
the working tree holds an unshipped version.

Verified on the real path, not only in tests: `--prepare-only` on a throwaway tag
refused to start and named all three uncommitted files including
`core_contract.py`, creating no run directory; the same command under the
override proceeded and wrote `checked: 56, override: true`. 468 tests pass, Ruff
clean, 102 pins with zero drift.

---

## v98 — drawn typing rhythm and length calibration (2026-08-19)

Policy ID: `generalized-card-v2-drawn-typing-rhythm-length-calibration-v98-20260819`.

v97 scored 7 PASS / 1 PARTIAL / 4 FAIL over ten matched threads (532 generated
against 532 real, coverage 1.00, `$3.6664`, 55.2 minutes):

| metric | real | generated | MWU p | KS p | Cliff | status |
|---|---:|---:|---:|---:|---:|---|
| structural_virality | | | 1.000 | 1.000 | 0.00 | PASS |
| avg_depth | | | 0.940 | 0.994 | 0.03 | PASS |
| semantic_mean_cosine | | | 0.910 | 0.787 | 0.04 | PASS |
| mean_story_probability | | | 0.734 | 0.787 | -0.10 | PASS |
| emotion_entropy | 1.5803 | 1.5085 | 0.326 | 0.168 | -0.27 | PASS, large \|d\| |
| hard_disagree_rate | | | 0.307 | 0.787 | 0.28 | PASS |
| self_bleu_4 | 0.0278 | 0.0316 | 0.186 | 0.787 | 0.36 | PASS, weak |
| neutral_rate | | | 0.017 | 0.052 | -0.64 | PARTIAL |
| polite_rate | | | 0.010 | 0.012 | -0.69 | FAIL |
| length_cv | 0.9468 | 0.8567 | 0.021 | 0.012 | -0.62 | FAIL, regressed |
| impolite_rate | | | 0.0008 | 0.0002 | 0.90 | FAIL |
| self_bertscore_mean_f1 | 0.4942 | 0.5185 | 0.0003 | 0.0002 | 0.96 | FAIL |

### The `self_bertscore` cause, after two rejected hypotheses

This is the metric worth the most care, and two plausible explanations were
measured and **rejected** before the third survived. Both rejections are kept in
the source so the next version does not re-litigate them.

Every unordered comment pair of the six smallest threads was scored with the
evaluator's own BERTScore (`microsoft/deberta-xlarge-mnli`, L40) and binned:

| log(longer/shorter) | real F1 | gen F1 | delta |
|---|---:|---:|---:|
| 0.00-0.35 | 0.5116 | 0.5479 | +0.0363 |
| 0.35-0.80 | 0.5105 | 0.5346 | +0.0241 |
| 0.80-1.40 | 0.5002 | 0.5134 | +0.0133 |
| 1.40-2.20 | 0.4798 | 0.4802 | +0.0004 |
| 2.20+ | 0.4362 | 0.4126 | **-0.0236** |

**Rejected: length spread.** Reweighting the generated pairs onto the real
pairs' length-ratio mix moves the mean 0.5090 -> 0.5057, one fifth of the
0.0163 gap.

**Rejected: a duplication tail.** Trimming the top of both distributions leaves
the gap where it was: +0.0163 untrimmed, +0.0154 after dropping the top 20% of
pairs on each side. It is a uniform shift, not a tail. The Planner is clean
too -- zero exactly duplicated `semantic_move` values over 532 slots, at most
1.3% of in-thread plan pairs above 0.35 content-word Jaccard.

**Rejected: the surface register.** Same-length pairs differ most in
function-word cosine (real 0.368, generated 0.502, +0.134), which is what
`sentence_rhythm` was built for. A falsification test on real text says habits
do not cause that: real pairs that *differ* in these habits are only 0.003-0.011
lower in function-word cosine than pairs that share them, and pairs that are both
sentence-length-uneven are slightly *more* alike, not less.

**What survived: a uniform lexical narrowing, caused by one instruction.**

| | real | generated |
|---|---:|---:|
| distinct word types | 3,645 | 2,670 |
| types / sqrt(tokens) | 21.02 | 15.95 |
| hapax rate | 0.502 | 0.427 |
| top-500 type coverage | 0.783 | 0.830 |

Per-comment type-token ratio at a fixed 30 tokens is *higher* in the generated
text (0.891 against 0.866), so no comment is individually thin. The thread draws
from a smaller lexicon, which lifts every pair equally.

453 of 532 slots are planned `no_story`, and v96's instruction for them bans
tense rather than narrative -- "no past action, event, before/after change, or
then/after pacing". On those 453 slots against their 532 matched real comments:

| | real | generated (`no_story`) |
|---|---:|---:|
| past-tense verb | 0.543 | 0.181 |
| future / `'ll` | 0.226 | 0.031 |
| present perfect | 0.167 | 0.031 |

`have` at 11% of its real rate, `will` at 1%, `to` at 54%. What the model falls
back on is a timeless conditional: `the` at 147% of its real rate, `if` 225%,
`whether` 1800%, `matters` 2900%.

It was also a live contradiction. Under `--own-fact-license named` the grounding
rule for a non-story slot is "Be particular rather than general", and **247 of
the 532 rendered v97 prompts (46.4%) carried that line and the tense ban at the
same time** -- the exact defect `writer_grounding` was created to eliminate.
StorySeeker scores narrative *sequence*, and `mean_story_probability` already
passes at Cliff -0.10, below real, so the headroom runs the safe way.

### The `length_cv` cause

Realized/target over all 532 slots is a smooth monotone regression toward the
model's own preferred length, crossing 1.0 near 40 words: 1.42x at 11-15 words,
1.05x at 16-25, 0.91x at 41-90, 0.71x at 251-400. The mean survives (55.8 real
against 52.3) and the spread collapses (`length_cv` 0.947 -> 0.857, below the
matched real thread on 9 of 10 threads).

Three versions tried to talk the Writer out of it and moved 250w+ from 0.61 to
0.71. The transfer function is clean enough to invert instead:

    log(realized) = 0.3835 + 0.8925 * log(asked)     n=532, R2=0.894

`length_calibration` renders `exp((log(target) - a) / b)` in the cue and leaves
`real_word_count` as the truth everywhere else, because the layout, the beats,
the tone band, and the length floor all describe a comment of the slot's real
size. Multipliers run 0.71x at two words to 1.47x at 845, monotone, and the
clamp does not bind anywhere inside the fitted range.

### The `emotion_entropy` cause

v97 wrote **zero exclamation marks in 532 comments** against 0.079 of matched
real ones. In the 24,029-comment reference corpus a comment containing one is
1.48x as likely to carry a non-neutral dominant emotion, concentrated on
gratitude, admiration, joy, love, and amusement -- the tail labels the entropy is
made of. `sentence_rhythm` draws seven habits per slot at each size band's
measured rate; its realized rates over the 532 re-rendered v97 slots track the
measurement within sampling noise.

### Arms

| flag | v98 default | reproduces v97 |
|---|---|---|
| `--no-story-scope` | `sequence` | `tense` |
| `--length-calibration` | `measured` | `off` |
| `--sentence-rhythm` | `measured` | `off` |
| `--final-punctuation` | `measured` | `off` |
| `--route-ledger` | `on` | `off` |

All five are written into `run_config.json` and checked by the resume-config
verification. Domain profile schema 14 -> 15 (`rhythm_profile`,
`final_punctuation_profile`).

### Zero-API result

446 tests pass, Ruff clean, 100/100 pins with zero drift, no untracked active
source, no unpinned local import, both parity scopes healthy, self-test passes
with all five arms on and with all five off, and the schema-15 profile rebuilds
over 424 excluded threads with 0 seed overlap. Re-rendering all 532 v97 slots
through the v98 prompt costs +474 characters of prompt (+8.5%): rhythm +292,
route ledger +183.

### N=10 result — 2026-08-20

Run `generalized_card_camera_gpt54_v98_rhythm_n10_20260820_v1`, paired to v97
(`--start-seed-index 2`, `--sampling-seed 42`, `--max-posts 10`), all five arms
at their v98 defaults. **8 PASS / 1 PARTIAL / 3 FAIL**, against v97's 7/1/4.

| metric | v97 MWU | v97 Cliff | v98 MWU | v98 Cliff | v98 status |
|---|---:|---:|---:|---:|---|
| `self_bleu_4` | 0.1859 | +0.36 | 0.1212 | +0.42 | PASS |
| `self_bertscore_mean_f1` | 0.00033 | +0.96 | 0.00058 | +0.92 | FAIL |
| `semantic_mean_cosine` | 0.9097 | +0.04 | 0.6232 | -0.14 | PASS |
| `hard_disagree_rate` | 0.3069 | +0.28 | 0.2897 | +0.29 | PASS |
| `polite_rate` | 0.01014 | -0.69 | 0.01258 | -0.67 | FAIL |
| `impolite_rate` | 0.00077 | +0.90 | 0.00101 | +0.88 | FAIL |
| `neutral_rate` | 0.01717 | -0.64 | 0.02099 | -0.62 | PARTIAL |
| `length_cv` | 0.02113 | -0.62 | **0.4727** | **+0.20** | FAIL -> **PASS** |
| `avg_depth` | 0.9396 | +0.03 | 0.9698 | +0.02 | PASS |
| `structural_virality` | 1.000 | 0.00 | 0.9697 | +0.02 | PASS |
| `mean_story_probability` | 0.7337 | -0.10 | 0.6776 | -0.12 | PASS |
| `emotion_entropy` | 0.3256 | -0.27 | **0.5708** | **-0.16** | PASS |

Two of the four metrics the user ranked first moved: `length_cv` FAIL -> PASS,
and `emotion_entropy` improved on both p-value and effect size. `self_bleu_4`
stayed a weak pass and got slightly worse. `self_bertscore_mean_f1` did not
move.

**Which arms worked.** Realized habit rates over the run, v97 -> v98:
semicolon 0.109 -> 0.023, dash clause 0.299 -> 0.071, ellipsis 0.017 -> 0.081,
exclamation 0.000 -> 0.064, digit 0.299 -> 0.457, parenthetical 0.055 -> 0.086,
bare final punctuation 0.041 -> 0.246. The length calibration closed the tails:
realized/target 0.738 -> 0.985 in the `essay` band and 0.873 -> 0.987 in
`very_long`; `length_cv` 0.862 -> 0.981 against a real 0.959, with threads below
the real value falling from 9/10 to 6/10. The `short` band overshot downward
(1.071 -> 0.857) and is the remaining length defect.

**`--no-story-scope sequence` should be reverted to `tense`.** It produced no
metric benefit -- past-tense rate 0.289 -> 0.288, `will` 0.015 -> 0.019, lexical
breadth (types/sqrt tokens) 15.95 -> 16.20 against a real 21.02 -- and it added
new repeated 4-grams (`. before that ,` 0 -> 4, `i was wrong to` 0 -> 4). The
prompt-contradiction fix it carried is worth keeping; the loosened scope is not.
The `self_bertscore` hypothesis behind this arm is therefore rejected, and the
metric has no verified mechanism.

**`self_bleu_4` characterised, no cheap lever.** An ablation harness that
reproduces the evaluator's number to five significant figures (v97 0.18588 /
+0.36, v98 0.12122 / +0.42) shows no phrase drives the metric: normalising every
typographic apostrophe, deleting all `check ...` openings, deleting the
`that's the part` family, and deleting yeah/basically/actually each move the mean
by at most 0.0005. Dropping each comment's first sentence makes it markedly
worse. Over 160 real threads it is a length metric first (share of <=15-word
comments r = +0.783, mean words r = -0.723) and generated already matches
length. OLS `self_bleu_4 = 0.04964 - 0.000288*meanWords -
0.00127*entityTypesPerComment` (R2 = 0.527) explains about 48% of the observed
gap; entity diversity's partial r is only -0.097, worth roughly a third of it.

Current verified repo state (2026-08-20): **449 tests pass, 101 pins with zero
drift.** The 446/100 figures in the zero-API section above were the state at
that gate, before the last test module and pin were added.

## v97 — keyboard surface and measured joints (2026-08-19)

Policy ID: `generalized-card-v2-keyboard-surface-measured-joints-v97-20260819`.

v96 was the first version to produce a complete, honestly evaluable 10-thread
sample under the new content policy: 532 generated comments against 532 matched
real ones, coverage 1.00, `$3.71`, 49 minutes. **6 of 12 metrics pass.**

| metric | real | generated | MWU p | KS p | \|d\| | status |
|---|---:|---:|---:|---:|---:|---|
| semantic_mean_cosine | 0.2892 | 0.2915 | 0.970 | 0.994 | 0.02 | PASS |
| structural_virality | 2.7955 | 2.8732 | 0.909 | 1.000 | 0.04 | PASS |
| avg_depth | 2.2680 | 2.3726 | 0.850 | 1.000 | 0.06 | PASS |
| emotion_entropy | 1.5803 | 1.5951 | 0.678 | 0.418 | 0.12 | PASS |
| mean_story_probability | 0.1266 | 0.1064 | 0.521 | 0.418 | 0.18 | PASS |
| length_cv | 0.9468 | 0.8658 | 0.076 | 0.418 | 0.48 | pass, marginal |
| self_bleu_4 | 0.0278 | 0.0375 | 0.009 | 0.052 | 0.70 | FAIL |
| hard_disagree_rate | 0.1208 | 0.2249 | 0.014 | 0.052 | 0.66 | FAIL |
| neutral_rate | 0.1715 | 0.0624 | 0.007 | 0.052 | 0.72 | FAIL |
| polite_rate | 0.3085 | 0.0668 | 0.006 | 0.002 | 0.74 | FAIL |
| impolite_rate | 0.4041 | 0.6797 | 0.001 | 0.002 | 0.87 | FAIL |
| self_bertscore_mean_f1 | 0.4942 | 0.5280 | 0.0002 | 0.00001 | 1.00 | FAIL |

v96's story and emotion arms worked: `mean_story_probability` and
`emotion_entropy` both passed, and `semantic_mean_cosine` is now the strongest
pass in the set. The remaining six are the subject of this version, and four
independent causes were measured in the v96 artifact before anything was
changed.

### 1 Typography is a metric, not a cosmetic

`score_thread_self_bleu.TOKEN_PATTERN` reads `it's` as one token and `it’s` as
three. **Zero of 532 v96 comments contained an ASCII apostrophe** and 389
contained a typographic one, so every generated contraction contributed a
`<word> ’ s` trigram shared across the thread that no real comment produces. The
same holds for em dashes (187 occurrences against 3 real), curly quotes (137
against 14), and the ellipsis character.

Measured on the domain's 11,817 evaluation-excluded comments, the typographic
form appears in 27.1% of apostrophe-bearing comments, 22.5% of quote-bearing,
10.5% of dash-bearing, and 15.6% of ellipsis-bearing ones. v96 was at 100% for
all four. `surface_typography` draws that share once per **speaker** per class,
because a device either substitutes the character or it does not, and a
per-speaker draw also raises between-author surface variance inside a thread.

Replayed over the v96 output with the real scorer, not a proxy:

| | real | v96 | v96 + keyboard typography |
|---|---:|---:|---:|
| self_bleu_4 mean | 0.0280 | 0.0373 | **0.0324** |
| MWU p | — | 0.009 | **0.273** |
| KS p | — | 0.052 | **0.787** |
| self_bertscore (4 threads) | — | — | **-0.008** |
| curly-apostrophe comment share | 0.105 | 0.731 | **0.164** |

### 2 The adjudication frame was on every slot

The Writer prompt rendered "The question your turn settles: ..." on **532 of 532**
v96 slots. That frame is the documented source of the "that's the part that
actually matters" family, which survived a rewording (v73), a prompt rebuild
(v74), and a route lock (v75), and is still in 18.4% of v96 comments against
effectively nothing in 30,643 tokens of matched real text. `that s the part`
alone appears in 39 comments, `that s the bit` in 16, `that s the only` in 18.

Broken out by planned function, the frame is worst exactly where it least
belongs: personal_datapoint 29.1%, explanation_analysis 23.1%, reaction 19.0%,
correction_caveat 16.0%, verdict_evaluation 12.3%, recommendation_advice 11.8%,
question_followup 8.1%. A slot told to report an experience *and* told which
question it settles converts the experience into an adjudication. Three releases
tried to reword the frame; none tried withholding it.

v97 renders the boundary only for correction, verdict, and advice turns, and
never for a slot carrying a story. Over the v96 task set that withholds it from
362 of 532 slots (68.0%), which carried 74 of the 98 observed frame instances
(76%).

### 3 The tone-length joint was inverted

`polite_rate` and `impolite_rate` failed with a **correct marginal**: the
Planner's targets were 0.311 polite and 0.442 impolite against a real 0.308 and
0.404. The joint was backwards. `_tone_cost` ranked slots by distance from each
class's median length, so `polite` (median 53 words) took the slots nearest 53
words and the longest slots were left for whichever label was assigned last:

| planned tone | 120-250w slots | 250w+ slots |
|---|---:|---:|
| v96 impolite | 74% | **100%** |
| excluded real, same sizes | 29% | **23%** |
| excluded real polite | 64% | **72%** |

The realized output followed the plan: generated comments over 120 words came
out 87% impolite and 9% polite, against a real 27% and 71%. `neutral_rate` fails
the same way from below — real neutral comments are short bare statements of a
model, price, or spec, and generated short comments are negation-led challenges.

`tone_length_fit` measures P(tone | comment size band) over 15,294 excluded
comments and fits the template's counts onto slots by iterative proportional
fitting, so both margins stay exact. A min-cost assignment was implemented first
and rejected: it maximizes likelihood and lands in a corner, producing 100%
polite in the top band against a measured 72%. Replayed over the ten v96 threads:

| band | v96 planned polite | v97 planned polite | measured real |
|---|---:|---:|---:|
| micro | 0.000 | 0.220 | 0.251 |
| short | 0.000 | 0.136 | 0.162 |
| medium | 0.414 | 0.225 | 0.263 |
| long | 0.661 | 0.431 | 0.520 |
| very_long | 0.114 | 0.600 | 0.638 |
| essay | 0.000 | 0.538 | 0.720 |

The measured conditional also refutes the hard exclusion it replaces: 25.1% of
real comments under ten words are labelled polite, because a short thank-you is
one, and v96 could not assign that at all.

### 4 Long slots were asked for a shape that does not exist

Long slots realized 0.61x their matched length, and the largest was at 0.32:

| matched target | slots | realized ratio |
|---|---:|---:|
| 0-10 w | 59 | 1.40 |
| 10-25 w | 147 | 1.01 |
| 25-60 w | 169 | 1.01 |
| 60-120 w | 109 | 0.86 |
| 120-250 w | 35 | 0.92 |
| 250+ w | 13 | **0.61** |
| the 845-word slot | 1 | **0.32** |

Short slots overshoot and long slots undershoot, which is precisely what
compresses `length_cv` (0.866 against 0.947) and, through the pairwise means,
holds `self_bertscore` up.

The cause was not the token budget, which allows 1,500 tokens for an 845-word
slot. It was the request. The 845-word slot was asked to "develop one local
thesis through about **40** distinct, connected beats", and the Planner does not
supply beats above about nine however many are asked for: asked ~6 it returned
5.2 and the slot realized 0.95x; asked 14-40 it returned 9.5 and realized 0.60x.
The largest beat plan any slot in the run received was 26. At ~20 realized words
per beat the mechanism tops out near 250 words.

Real long comments are not one thesis at all. Measured over the excluded
threads, median paragraph count rises 1 / 1 / 1 / 2 / 3 / 6 across the size
bands, p90 reaches 14, and lists and quoted parent excerpts appear in 12.6% and
26.7% of the longest comments. Words per paragraph is nearly flat inside a band
(53.6 in the top band) while the paragraph count scales with length: 6 at
250-350 words, 10 at 500-700, 11 above 700. v96 output had a blank line in 3.4%
of comments against 33.8% of real ones, and one paragraph at every size.

`comment_structure` asks each slot for the paragraph count its size actually
has, floored at the band median and capped at its p90, and permits a paragraph
to take a related side point. The beat ceiling drops from 40 to 12 and the
minimum acceptable count is capped where the Planner still delivers, so an
unreachable request stops generating plan-repair traffic. Rendered against the
v96 targets: 90w -> 2 paragraphs, 160w -> 3, 300w -> 6, 539w -> 10, 845w -> 14.

### Also changed

The entry grammar was realized on 47.4% of v96 slots, and the drift has one
direction: 20.7% of comments opened with a bare polarity token against 6.8% of
matched real comments and 5.3% scheduled. A non-polarity opener now carries an
explicit exclusion of that one measured default rather than a general style rule.

### Arms

Every change is a named, reproducible arm recorded in `run_config.json` and
checked on resume. Each `off`/legacy value reproduces v96 exactly.

| flag | v97 default | reproduces v96 |
|---|---|---|
| `--reddit-typography` | `on` | `off` |
| `--turn-frame` | `adjudicative_only` | `universal` |
| `--tone-length-fit` | `conditional` | `median` |
| `--long-form-layout` | `measured` | `beats_only` |

Domain profile schema 11 -> 14, adding `typography_profile`,
`structure_profile`, and `tone_length_profile`. All three are measured on
evaluation-excluded threads only; seed/reference overlap is 0 and no matched
comment text is stored.

### Predicted direction

Stated before the API call so a null result stays interpretable.

- `self_bleu_4`: 0.0375 -> about 0.032 from typography alone, further down from
  the withheld frame. This is the one prediction already measured offline.
- `self_bertscore_mean_f1`: -0.008 from typography; the rest has to come from the
  withheld frame and the restored length spread. Weakest prediction of the four.
- `length_cv`: up from 0.866 toward 0.947 as the 250w+ ratio moves off 0.61.
- `polite_rate` up, `impolite_rate` down, `neutral_rate` up, from the joint fit.
- `hard_disagree_rate`: down from 0.225. Every stance probability in both
  populations sits between 0.20 and 0.40 and the mean differs by only 0.011
  (0.324 against 0.313), so the label is an argmax over near-ties and a small
  register shift should move the rate a lot.
- At risk: `mean_story_probability` and `emotion_entropy` both pass now and both
  depend on the register that four arms are changing at once. If either drops,
  the arms exist to attribute it.

### Zero-API gate

- 369 generalized-card tests pass, including new focused suites for typography,
  layout, the tone-length fit, and the turn frame rendered through the
  configured backend rather than by calling the gate directly.
- Ruff clean over the whole `generalized_card/` tree.
- Source contract: 98/98 files pinned, no untracked active source, no unpinned
  local import, zero hash drift.
- Active and active-plus-legacy parity both healthy, no unexpected overrides.
- Backend self-test passes with all four v97 arms plus `domain-claim=selective`
  and `own-fact-license=named`.
- Domain profile rebuilds at schema 14 over 424 excluded threads with all three
  new profiles available and 0 seed overlap.
- The active shaper was exercised directly: over 200 speakers the realized
  typographic apostrophe share is 0.240 against the measured 0.271, and the four
  classes draw independently.
- Exact seed-2 `--prepare-only` completed as
  `generalized_card_camera_gpt54_v97_keyboard_seed2_20260819_preflight_v1` with
  every arm recorded and no API call.

### Paid seed-2 gate result

Tag `generalized_card_camera_gpt54_v97_keyboard_seed2_20260819_v1`: 45/45
comments in one attempt, 89 requests, 343,063 input and 29,102 output tokens,
`$0.3883`, 235 seconds. No Writer retries, degraded comments, schema recoveries,
exact duplicates, or matched-text leakage.

n=1, so every p-value is descriptive. What is comparable is the same thread
generated under three consecutive policies:

| exact seed-2 property | real | v95 | v96 | v97 |
|---|---:|---:|---:|---:|
| self_bleu_4 | 0.0268 | 0.0350 | 0.0306 | **0.0273** |
| self_bertscore_mean_f1 | 0.4892 | 0.5306 | 0.5299 | **0.5074** |
| repeated 4-gram share | 0.0200 | — | 0.0795† | **0.0237** |
| distinct 3-word openers | 0.9778 | — | 0.7726† | **0.9778** |
| word-count CV | 0.8768 | — | 1.0141† | **0.8708** |
| emotion_entropy | 1.9687 | 1.6572 | — | **1.6827** |
| hard_disagree_rate | 0.0571 | — | 0.2249† | **0.0909** |
| curly-apostrophe comment share | 0.3556 | — | 0.7310† | **0.3778** |
| em-dash occurrences | 0 | — | 187† | **0** |
| polarity-token opener share | 0.0222 | — | 0.2071† | **0.0889** |

† pooled over the v96 ten-thread run rather than seed 2 alone.

Six of six predictions held.

- **self-BLEU is now at real**: 0.0273 against 0.0268. The v96 gap was +0.0038.
- **self-BERTScore closed 55% of its gap**: +0.0407 -> +0.0182. This was the
  weakest prediction and the metric with |d|=1.00 at N=10.
- **Repetition is at real**: 0.0237 against 0.0200, and no 4-gram appears in
  more than 2 of the 45 comments. `that s the part` was in 39 of 532 v96
  comments; the frame's overall share fell 0.184 -> 0.133 with the remainder on
  the adjudicative slots that keep the line by design.
- **Length spread is at real**: CV 0.8708 against 0.8768, with realized/target
  ratios 1.08, 1.11, 0.96, 0.99 across the size bands. This thread has no 250w+
  slot, so the 0.61 ratio is not yet retested.
- **The tone-length joint is now monotone**: planned polite runs 0.40 micro,
  0.30 short, 0.21 medium, 0.50 long, 0.83 very_long. v96 was 0.00 / 0.00 /
  0.41 / 0.66 / 0.11 with 100% impolite above 250 words.
- **Openers match exactly**: distinct 3-word openers 0.9778 against a real
  0.9778, and polarity-token openers 0.089 against 0.022 and a v96 0.207.

Still open on this thread, and not addressed by v97:

- `impolite_rate` 0.614 against a real 0.222 with a Planner target of 0.370, and
  `polite_rate` 0.205 against 0.489. Placement is now right and realization is
  the remaining bottleneck: tone exact realization is 0.614, up from 0.583.
- Concreteness is unchanged: domain-vocabulary comments 0.156 against 0.556,
  digit-bearing 0.356 against 0.600, 10 distinct model designators against 40.
- `no end punctuation` 0.044 against 0.244, and no comment uses `!` against a
  real 0.044. Both are typing habits of the same kind `surface_typography`
  already measures.
- `mean_story_probability` 0.080 against a template target of 0.140.

### Paid N=10 result

Tag `generalized_card_camera_gpt54_v97_keyboard_n10_20260819_v1`: 10 threads,
532 comments, coverage 1.00, 993 requests, `$3.6664`, 55.2 minutes.

**7 of 12 pass, against v96's 6.** Two metrics moved from fail to pass, one
passing metric regressed to fail, and the three tone rates did not move.

| metric | real | v96 | v97 | v96 MWU | v97 MWU | v96 \|d\| | v97 \|d\| |
|---|---:|---:|---:|---:|---:|---:|---:|
| self_bleu_4 | 0.0278 | 0.0375 | **0.0316** | 0.009 | **0.186** | 0.70 | **0.36** |
| hard_disagree_rate | 0.1208 | 0.2249 | **0.1738** | 0.014 | **0.307** | 0.66 | **0.28** |
| mean_story_probability | 0.1266 | 0.1064 | 0.1197 | 0.521 | 0.734 | 0.18 | **0.10** |
| structural_virality | 2.7955 | 2.8732 | 2.8119 | 0.909 | 1.000 | 0.04 | **0.00** |
| avg_depth | 2.2680 | 2.3726 | 2.2924 | 0.850 | 0.940 | 0.06 | **0.03** |
| semantic_mean_cosine | 0.2892 | 0.2915 | 0.2900 | 0.970 | 0.910 | 0.02 | 0.04 |
| self_bertscore_mean_f1 | 0.4942 | 0.5280 | 0.5185 | 0.0002 | 0.0003 | 1.00 | 0.96 |
| emotion_entropy | 1.5803 | 1.5951 | 1.5085 | 0.678 | 0.326 | 0.12 | 0.27 |
| length_cv | 0.9468 | 0.8658 | 0.8567 | 0.076 | **0.021** | 0.48 | **0.62** |
| neutral_rate | 0.1715 | 0.0624 | 0.0805 | 0.007 | 0.017 | 0.72 | 0.64 |
| polite_rate | 0.3085 | 0.0668 | 0.0850 | 0.006 | 0.010 | 0.74 | 0.69 |
| impolite_rate | 0.4041 | 0.6797 | 0.6849 | 0.001 | 0.001 | 0.87 | 0.90 |

**The pass count is the wrong headline.** Simulating the evaluator's own
MWU+KS pair, a metric needs \|Cliff's delta\| at or below 0.10 to have a
reasonable chance at N=150. By that standard v97 has **4 viable metrics**
(`structural_virality` 0.00, `avg_depth` 0.03, `semantic_mean_cosine` 0.04,
`mean_story_probability` 0.10) against v96's 3, and three of the seven that
"pass" at N=10 — `emotion_entropy`, `hard_disagree_rate`, `self_bleu_4` — will
not survive 150 threads at their current effect sizes.

**Wins, attributed.**

- `self_bleu_4` fail -> pass, mean gap +0.0095 -> +0.0036. Typography was the
  measured share of this; the withheld frame is the rest.
- `hard_disagree_rate` fail -> pass, 0.2249 -> 0.1738 against a real 0.1208.
  Predicted: every stance probability in both populations sits between 0.20 and
  0.40, so the label is an argmax over near-ties and a register shift moves the
  rate a lot.
- `mean_story_probability` reaches \|d\| 0.10 and `structural_virality` 0.00.
- `self_bertscore` closed 28% of its gap (+0.0338 -> +0.0243) and remains the
  worst metric in the set.

**One real regression: `length_cv`, 0.076 -> 0.021, \|d\| 0.48 -> 0.62.**
Generated CV is below real on 9 of 10 threads, up from 8. The tail fix worked
and was outweighed by an unintended middle-band inflation:

| matched target | n | v96 ratio | v97 ratio |
|---|---:|---:|---:|
| 0-10 w | 59 | 1.40 | 1.39 |
| 10-25 w | 147 | 1.01 | **1.12** |
| 25-60 w | 169 | 1.01 | **1.05** |
| 60-120 w | 109 | 0.86 | **0.94** |
| 120-250 w | 35 | 0.92 | 0.89 |
| 250+ w | 13 | **0.61** | **0.78** |

The 250w+ band moved 228.8 -> 316.1 realized words and the longest generated
comment 386 -> 638 against a real 845, exactly as intended. But 425 of 532 slots
sit in the 10-120 range, so inflating them by 5-11% compresses the spread more
than the tail expands it. The 10-25 band gets no layout cue at all, so the cause
there is one of the two other v97 changes — most likely the opener exclusion,
which now forbids the short agreement token on 87% of slots and pushes them into
a longer content opener. This is the same failure v67 recorded when it raised the
long-slot ratio to 0.99 and lost `length_cv`.

**One regression that is not attributable: `emotion_entropy`, \|d\| 0.12 ->
0.27.** Five threads rose and five fell. Within every length band v97's realized
entropy is equal to or higher than v96's, and the planned affect distribution is
identical between the two versions (21 distinct affects, same counts). The mean
fell 0.087, and the single 7-comment thread accounts for 0.079 of that on its
own: with 7 comments the metric ranges from ln(4)=1.386 to ln(7)=1.946 and one
label swap moves it 0.5. Do not attribute this to a v97 arm, and do not "fix" it.

**Unchanged: the three tone rates.** Placement is now correct — planned polite
runs monotonically 0.22 / 0.14 / 0.23 / 0.43 / 0.60 / 0.54 across the size bands
against a measured 0.25 / 0.16 / 0.26 / 0.52 / 0.64 / 0.72 — but realization is
the bottleneck. Tone exact realization is 0.61, up from 0.58. A slot planned
polite on a long turn still comes out blunt.

**What the output still does that real text does not**, pooled over 532
comments: the adjudication frame is in 16.0% of comments against 0.2% real and
supplies the five most repeated 4-grams (`that's the part that` 17 comments,
`that s the part` 14, `and that was the` 10, `the part that matters` 8,
`is the part that` 8); `no end punctuation` 0.041 against 0.173; `has a digit`
0.299 against 0.562; no comment contains `!` against a real 0.079.

Superseded by v98. See `tasks/v97-worklog.md` for the two further causes measured
during this run.

---

## v96 — selective facts and ancestor-aware reply novelty (2026-08-18)

Policy ID: `generalized-card-v2-selective-facts-ancestor-novelty-v96-20260818`.

The paid v95 seed-2 gate proved reliability but falsified its content arm. It
completed 45/45 comments in one attempt for `$0.3481`, yet generated 5 distinct
model designators against 40 real, digit-bearing comments at 0.20 against 0.60,
domain-vocabulary comments at 0.1556 against 0.5556, self-BLEU at 0.0350
against 0.0268, and self-BERTScore at 0.5306 against 0.4892. Story probability
was 0.1015 against 0.2321 and emotion entropy 1.6572 against 1.9687. The single
thread is descriptive only; its MWU/KS values are not inferential.

The active data path explained the gap. Under `domain-claim=off`, root planning
discarded excluded-reference facts, direct-reply planning received no reference
rows, and the Writer often had only one or two seed tokens. The `named` rule
also prohibited repeating any name another comment had used, suppressing the
normal product-name cohesion in real discussion. Deep replies excluded only
their immediate parent, allowing a full branch to circle one decision boundary.

v96 introduces `domain-claim=selective`, derived only from anonymous slot
capacity and evaluation-excluded reference surface roles. Only scheduled slots
can retain one Planner-restated general fact after parsing; the Writer never
sees raw reference text. Historical `planned` and `off` modes remain available.
Selective direct replies receive an excluded-reference window and compact
coverage of the full ancestor chain. Delivered claims enter Writer factual
anchors. A personal story receives either a claim or a rotating held-out
equipment shortlist, never two independent factual sources. Product names may
recur naturally, while the same fact or amount may not.

Predicted direction: more product/domain vocabulary, quantities, and concrete
story actions; lower self-BLEU/self-BERTScore and fewer deep-branch semantic
collisions; story probability and emotion entropy should move upward without
changing length or structure. The zero-API gate passed: 316 tests, full Ruff,
95/95 clean source pins, both parity scopes, backend self-test, and exact seed-2
prepare-only. Realized content remains unverified until one new paid seed-2
audit finishes. N=10 is not authorized by a successful process exit alone.

---

## v95 — compiled, non-terminal Planner content contracts (2026-08-18)

Policy ID: `generalized-card-v2-nonfatal-compiled-plan-contract-v95-20260818`.

The paid v94 seed-2 gate exhausted all three whole-post attempts without a
Writer-complete thread: 152 requests, 1,031,377 input tokens, 76,450 output
tokens, 541 seconds, and an estimated `$0.9608`. The three attempts stopped on
different slots and different rules: S20/S22 social/long-form, S43 social, and
S18 surface density. Of the 152 requests, 130 were slot-local Planner quality
repairs. More retries were therefore repeated plan sampling, not reliability.

The persisted candidates expose the ownership error. Aggregate held-out
story/affect targets were applied after the Planner, and dependent Planner
fields could contradict them. S43 planned a natural gratitude close, then the
template overlay changed only its affect to neutral and validation rejected the
contradiction created by the overlay itself. Other slots alternated between
valid development beats and a valid substantive payload without returning both
in one stochastic whole-plan candidate.

v95 compiles the frozen story/affect/capacity controls and the Planner's local
semantic move into one dependent routing contract before quality evaluation.
The compiler preserves `semantic_move`, local topic, detail, decision boundary,
and reply increment. It reconciles only evidence, payload, function, role, and
social affect required by the fixed story, social-close, micro, or substantive
slot contract, recording every change. Direct replies now receive the schedule's
default `no_story` in their Prompt instead of seeing `unassigned` and being
overwritten later.

Planner content diagnostics receive at most one slot-local repair unless a
true within-plan contract conflict remains. Missing long-form beats are
non-terminal because the Writer already has a capacity-derived fallback.
Residual content-contract and collision diagnostics are persisted as warnings
and continue to the Writer; only missing schema rows, malformed JSON,
transport/auth/safety failures, empty output, and exact-coverage failure can
stop persistence.

Zero-API replay over all 19 saved v94 batch results changes residual terminal
contract conflicts from three to zero. A cross-product stress test covers
story/no-story, neutral/gratitude, ordinary/gratitude roles, low-information and
story/advice payloads, and micro/ordinary/long capacity. Offline acceptance so
far: 307 generalized tests pass; Ruff is clean; the backend self-test passes;
active and active-plus-legacy parity are healthy with 95/95 source pins clean;
and the exact paid seed-2 configuration completed `--prepare-only` as
`generalized_card_camera_gpt54_v95_named_seed2_20260818_preflight_v1`. No paid
call had been made at release time.

The later paid gate completed all 45 slots in one attempt: 86 requests, 303,941
input tokens, 26,702 output tokens, 301 seconds, and `$0.3481`. Reliability is
confirmed. Content is not: the exact n=1 gaps above and repeated abstract
handling/fixed-lens language falsify `named + domain-claim=off` as a sufficient
content policy. Do not run v95 N=10; it is superseded by v96.

---

## v94 — state-preserving Planner repair (2026-08-18)

Policy ID: `generalized-card-v2-state-preserving-plan-repair-v94-20260818`.

The paid v93 N=10 run again completed seeds 0 and 1, then stopped on seed 2
root S9 after 100 requests, 305 seconds, and an estimated `$0.3992`. Unlike the
v92 failure, topology normalization worked: no root reply-only contract
remained. The persisted repair history exposed a separate state-loss bug.

S9 initially had two blocking fields. Its first repair supplied all five
long-form beats but retained the story/evidence conflict. Its second repair
fixed that conflict but cleared `development_plan`. Its third restored all five
beats but reintroduced the story/evidence conflict. Because targeted repair
replaced the whole plan object, each successful field repair erased the prior
one and the finite budget expired.

v94 keeps whole-plan replacement while multiple repair issues remain.
When the only remaining repair issue is `long_form_capacity`, it merges only
`development_plan` from the candidate and ignores unrelated model drift. The
Prompt states that boundary once, and the audit stores the raw candidate, the
applied candidate, and the merged field names. This is hard-contract recovery,
not metric-driven candidate selection.

Replaying the exact final v93 candidate against its selected S9 changes
blocking count `1 -> 0`, retains five beats and the already-correct
`evidence_mode=small_observation`, and leaves no S9 issue. Offline acceptance:
304 generalized tests pass, Ruff is clean, and the camera backend self-test
passes. Active and active-plus-legacy parity are healthy, with 94/94 source
pins clean. The exact seed-2 gate command also completed `--prepare-only` as
`generalized_card_camera_gpt54_v94_named_seed2_20260818_preflight_v1`. No paid
v94's paid seed-2 gate later failed all three attempts for different plan
contract combinations and produced no evaluable thread. It is superseded by
v95; do not spend or evaluate under v94.

---

## v93 — structural root/reply boundary (2026-08-18)

Policy ID: `generalized-card-v2-root-reply-boundary-v93-20260818`.

The paid v92 N=10 run completed seeds 0 and 1, then stopped in seed 2 on root
S9. Its initial 108-word slot lacked a required long-form plan. All three S9
repairs supplied the requested five connected beats. The first otherwise-valid
candidate also used `reply_delta_type=social_close` on this structurally root
slot; the social validator correctly rejected that combination. A later attempt
collapsed the long turn into a narrow question, and the third repeated the root
`social_close`, so the old empty-development plan remained selected.

v93 makes anonymous topology authoritative before quality selection. Root plans
clear `reply_delta`, `reply_delta_type`, and `reply_novelty_anchor`, with every
nonempty override recorded in `control_normalizations`. Direct replies preserve
all three fields. The root Planner schema now requests literal `none`, and 138
lines of duplicate direct-reply definitions, contrast rules, and the unreachable
parent-contract renderer were removed from `prompts.py`; direct replies already
use their dedicated compact Planner Prompt.

Replaying the actual first v92 S9 repair under v93 changes its rank from
`(1, 46)` to `(0, 41)`, removes the only blocking issue, and preserves all five
development beats. Offline acceptance: 300 generalized tests and 3 focused
Self-BERTScore tests pass; Ruff is clean; active and active-plus-legacy parity
are healthy; 93/93 pins are clean; and the exact N=10 command completed
`--prepare-only` as
`generalized_card_camera_gpt54_v93_named_n10_20260818_preflight_v1` with no API
call. Because v92 already generated two threads with the old root Prompt, the
formal N=10 evaluation must use a fresh v93 tag rather than mix policies.

---

## v92 — lossless `domain-claim=off` planning (2026-08-17)

Policy ID: `generalized-card-v2-lossless-domain-claim-off-v92-20260817`.

The post-v91 Planner→Writer audit found that `--domain-claim off` disabled only
delivery. Both root and direct-reply Planners still spent Prompt and output
tokens assigning a fact that `backend.py` then withheld from the Writer. The
Planner could consequently build `semantic_move`, `detail_focus`, or
`domain_intent` around information absent at realization—the exact handoff gap
this generator is supposed to eliminate.

v92 makes `off` mean off at both stages. Root and reply schemas require the
literal `none`, the domain-knowledge/claim instruction is absent, and a compact
rule requires the complete contribution to live in fields the Writer receives.
Normalization also clears a claim if the model ignores the rule. `planned`
mode retains the prior claim path unchanged. This removes redundant Planner
Prompt/output mass without weakening semantic planning from visible seed,
parent, branch, and evaluation-excluded discourse patterns.

The next configuration remains `--domain-claim off --own-fact-license named`:
the Planner hands over the whole semantic move, while the slot-gated Writer adds
varied local particulars instead of receiving one separately injected fact.
Expected effects are fewer abstract/incomplete moves and lower Planner cost;
12-metric movement remains unmeasured until the paid diagnostic.

Offline acceptance is complete: 299 generalized tests and 3 focused
Self-BERTScore scorer tests pass; Ruff is clean; active and
active-plus-legacy parity are healthy; all 93 source pins have zero missing,
untracked, unpinned-import, or drift findings. Rendered root and direct-reply
Prompt tests cover both flag values, normalization clears a noncompliant
off-mode claim, and the exact named/off seed-8 public command completed
`--prepare-only` as
`generalized_card_camera_gpt54_v92_named_seed8_20260817_preflight_v2` without an
API call. No v92 content or metric result is claimed yet.

---

## v91 — slot-gated concreteness permission (2026-08-17)

Policy ID: `generalized-card-v2-slot-gated-fact-license-v91-20260817`.

The pre-run completion audit found that the built-but-unrun `named`
concreteness arm was not safe to enable. Its slot resolver licensed only
substantive comments (at least 25 anonymous words and not micro/short), but its
system Prompt unconditionally told every comment to name particulars and give
amounts. A micro reaction therefore received a global pressure to invent detail
and a per-comment rule allowing names/numbers only when visible.

v91 makes the system sentence an authorization boundary only: per-comment
instructions may override the generic visibility ban for an explicitly
licensed turn. The actual name/amount instruction remains once, in the
substantive user Prompt. Unlicensed micro/short turns retain their visible-only
rule, and the legacy `own` permission receives the same slot-gated treatment.

The gate has a meaningful held-out structural scale. On the exact 186-slot
seed-8 skeleton it licenses 110 slots (59.14%); the matched real thread has a
digit in 59.68% of comments, while v80 generated only 31.35%. Real model
designators were 118 versus 29 generated. The next diagnostic should therefore
use `--own-fact-license named` with `--domain-claim off`: varied particulars are
realized locally instead of injecting one Planner fact across nearly every
comment.

Expected direction: more varied names and quantities, lower designator
concentration, and less abstract/advisory prose. This may move Self-BLEU,
Self-BERTScore, and semantic cosine through varied content, but n=1 can only
diagnose that mechanism; formal distribution claims still require sufficient N.

Offline acceptance: 297 generalized tests plus 3 focused Self-BERTScore tests,
Ruff, healthy active and active-plus-legacy parity, 93/93 clean source pins, and
a named-mode backend self-test. Full 186-slot Prompt replay produced exactly
110 licensed Prompts, each with one behavior instruction; all 76 unlicensed
Prompts had zero, the system contained one conditional authorization and no
behavior duplicate, and no invented-equipment block appeared. Exact named-mode
seed-8 `--prepare-only` passed under
`generalized_card_camera_gpt54_v91_named_seed8_20260817_preflight_v1`. No v91
API call has been made.

---

## v90 — one story-grounding boundary for both Planner paths (2026-08-17)

Policy ID: `generalized-card-v2-reply-story-grounding-v90-20260817`.

The post-v89 completion audit found that the synthetic-story repair covered the
root Comment Planner but not the specialized direct-reply Planner. Direct
replies were still required to plan an actual first-person event sequence while
also being forbidden to carry a source participant's detail or invent a fact
about the seed. That left the model to guess whether an ordinary synthetic
personal sequence was permitted—the same ambiguity that helped make the v88
root story slot unrealizable.

v90 defines that Planner boundary once and renders it on both root and direct
reply paths: an ordinary, non-verifiable first-person sequence may be
synthesized around a visible or generic local point, but product facts and other
externally checkable outcomes may not be invented. The Writer's existing
off-mode story rule already has this boundary, so v90 makes Planner and Writer
agree; it does not expand the Writer's factual license. A regression test checks
that the direct-reply Prompt contains both halves of the rule and still forbids
inventing seed facts.

Expected result: scheduled direct-reply stories should no longer consume repair
attempts or fail because the Planner interprets the factual boundary as a ban on
all personal sequence. This is a completion/reliability fix; movement on story
realization and the 12 metrics remains a paid-run question.

Offline acceptance: 295 generalized tests plus 3 focused Self-BERTScore tests,
Ruff, matched-speaker backend self-test, active and active-plus-legacy parity,
and 93/93 source pins with zero drift or closure gaps. Exact seed-8
`--prepare-only` passed under
`generalized_card_camera_gpt54_v90_preflight_seed8_20260817_v1`. No v90 API call
has been made.

---

## v89 — realizability-first Planner repair (2026-08-17)

Policy ID: `generalized-card-v2-realizability-first-planner-v89-20260817`.

The first paid v88 seed-8 attempt stopped after 24 Planner requests, before any
Writer call: 116 seconds and $0.1805. The offset-8 batch ended with blocking
contracts on S10, S13, and S15. The audit proved three distinct causes.

- Candidate selection used one scalar issue score. A repaired S15 removed its
  story conflict but introduced a semantic collision (weight 10); because the
  story conflict weighed 8, the realizable candidate was rejected and the
  impossible plan was retained. v89 ranks candidates first by the number of
  Writer-blocking contract issues, then by aggregate quality.
- The root Planner required scheduled stories and firsthand evidence while a
  blanket rule banned hidden anecdotes and all facts absent from a title-only
  seed. v89 states the same synthetic, non-verifiable personal-sequence
  boundary already enforced by the Writer, without licensing externally
  checkable product facts.
- A `polite` classifier target was treated as semantic truth and could abort a
  186-comment post unless the plan agreed and used one of three functions.
  Polite-Guard scores realized surface text, so v89 retains this pairing as
  low-weight anti-customer-support feedback but removes it from the blocking
  social contract. Story, affect/social-close, surface capacity, and long-form
  coherence remain blocking.

Planner audit rows now include JSON-safe initial, candidate, recovered, and
selected plan snapshots plus the before/candidate repair ranks. This closes the
v88 observability gap: its log recorded issue labels and scores but not the plan
whose acceptance was being decided.

Expected result for the replacement seed-8 run: no termination from the known
S10/S13 polite pairings; an S15 repair that reduces blocking contracts is kept
even if collision remains as a logged warning. This is a reliability fix, not a
claim that any of the 12 metric distributions improved. Paid content and metric
evidence remain pending.

Offline acceptance: 294 generalized tests plus 3 focused scorer tests, Ruff,
matched-speaker backend self-test, active and active-plus-legacy parity, 93/93
source pins with zero drift or closure gaps, and exact v89 seed-8
`--prepare-only`. No v89 API call has been made.

---

## v88 — structural speakers without invented biography (2026-08-17)

Policy ID: `generalized-card-v2-structural-speakers-grounding-v88-20260817`.

Completion audit before the paid v87 run found two current Prompt/structure
problems. First, `--own-fact-license off` still rendered an invented equipment
permission before the conservative fact rule revoked personal experience. A
full 186-task replay measured 78 equipment blocks, 144 personal-experience bans,
and 61 Prompts containing both. Preserving that contradiction solely as a
historical ablation violated the active Prompt rules; git already preserves it.
v88 renders an equipment shortlist only for the explicit legacy `own` license,
and the same replay now measures zero equipment blocks and zero conflicts.

Second, `speaker-identity matched` mixed a valid matched structural join with
invented kit, tenure, and use-case biography, so it could not safely be the
default. v88 deletes those semantic fields and their kit-filter helper. The
roster retains only anonymous speaker ID, OP status, slot IDs, and anonymous
account status. The Writer may see only its own earlier generated turns and an
instruction to keep factual self-claims consistent while still following the
current turn's assigned voice and affect. Real author strings never cross the
boundary. Matched recurring-speaker structure is now the default; `off` remains
the one-author-per-slot structural ablation.

Current seed-8 structural audit: 186 slots form 97 generated speaker groups,
including 80 named-source groups and 17 anonymous one-shots; named groups
average 2.112 turns, recurring groups own 66.7% of comment mass, and the busiest
group has 10 turns. The active expander integration test proves repeated source
authors receive the same anonymous `speaker_id`.

Expected directions are fewer grounding contradictions, less fake persona
boilerplate, and more realistic participant continuity. Self-BERT or other
metric movement is a hypothesis, not a result. Offline acceptance: 292
generalized tests plus 3 focused scorer tests, Ruff, matched-speaker backend
self-test, active and active-plus-legacy parity, 93/93 source pins, full Prompt
replay, and exact v88 seed-8 `--prepare-only`. No API call was made.

Paid result: the formal seed-8 attempt failed before Writer generation after 24
Planner requests (`$0.1805`, 116 seconds). No discussion artifact exists and no
content or 12-metric conclusion can be drawn. v89 supersedes v88 for rerun.

---

## v87 — payload-safe Writer routing and final-contract refresh (2026-08-17)

Policy ID: `generalized-card-v2-payload-safe-writer-routing-v87-20260817`.

Hypothesis before the paid run: a short surface shape must not send a
substantive Planner payload into the low-information Writer, whose correct hard
rules prohibit advice, explanation, and caveats. Building the focused ledger
directly from bounded records should also remove duplicated exclusions without
weakening exact-duplicate persistence checks. Expected qualitative directions
are more faithful short corrections/helpful turns, fewer contradictory social
assignments, and less repeated Prompt context. No generated content or
12-metric improvement is claimed before a paid artifact is scored.

Evidence and changes:

- Replayed all 186 recorded v80 tasks through every current Writer route. The
  old routing condition selected 32 low-information slots, including six
  `soft_helpful` payloads and one `correction`. v87 gates short utterance modes
  by payload semantics first; the replay now selects 25/186, all from the
  explicit low-information payload set and all `no_story`.
- Focused/low-information Prompts no longer build a full five-section thread
  blackboard and parse two sections back from its rendered text. They construct
  bounded semantic and short-line ledgers from source records, omit exact
  openings already shown nearby, and avoid restating the same social-close move
  as both required and already covered. Replay found no exact duplicate long
  lines and no duplicated required semantic move.
- Writer-facing tone controls are recomputed after every Planner-owned role,
  payload, voice, and surface contract is final. A stale
  `pure_acknowledgement` can no longer survive on a neutral datapoint or
  correction.
- The social-reaction validator is now bidirectional. Gratitude/relief already
  required a gratitude reaction; v87 also rejects a `gratitude_reply` or
  `social_close` paired with neutral affect, an explanatory/meta payload, a
  non-reaction function, or a story.

Offline acceptance: 290 generalized tests plus 3 focused scorer tests pass;
Ruff, camera backend self-test, active and active-plus-legacy parity pass; all
93 declared pins agree with zero missing, untracked-active, unpinned-import, or
drift entries. The exact seed-8 v87 command passed `--prepare-only` with no API
call. v87 was superseded by v88 before a paid generation.

---

## v86 — compact low-information Writer and root-relation clarity (2026-08-17)

Policy ID: `generalized-card-v2-root-relation-prompt-v86-20260817`.

Hypothesis before the paid run: low-information slots should realize their
assigned reaction, rant, question, acknowledgement, or bare answer more often
when the Writer sees one compact discourse contract rather than several
overlapping copies. Root comments should not be told that they answer a parent
that does not exist. Expected qualitative directions are fewer generic helpful
or customer-service-shaped turns, less Planner-language echo, and more faithful
short social/affective realization. No 12-metric improvement is claimed until a
new artifact is scored.

Changed:

- For a root slot, the focused Writer now receives `relation to post` and values
  such as `answers_post`; direct replies retain `reply relation` and parent
  values. The persisted Planner plan is unchanged, preserving audit evidence.
- The low-information Writer now uses the same compact discourse contract and
  bounded semantic/short-utterance ledger as the focused substantive path.
  Duplicate private-slot, semantic-contract, local-move, full-blackboard,
  placeholder, payload, tone, story, affect, and length renderings were removed.
  Its low-information and grounding hard rules remain.
- Reviser-only Prompt adaptation and Self-BLEU revision diagnostics moved from
  active `prompts.py` to `legacy_reviser_prompts.py`. AST hashes prove every
  migrated function is identical and every retained active Prompt function is
  unchanged apart from the two v86 Writer edits.

Offline acceptance: 286 generalized tests plus 3 focused scorer tests, Ruff,
camera backend self-test, active and legacy parity, and 93/93 pins with zero
active untracked or unpinned local imports. v86 was superseded by v87 before a
paid generation.

---

## v85 — auditable Planner controls and dead-path pruning (2026-08-17)

Policy ID: `generalized-card-v2-auditable-plan-controls-v85-20260817`.

This release is a current-path simplification and observability pass before the
next paid run. It does not claim a direct improvement to any of the 12 metric
values.

Changed:

- The existing slot-schedule override events are now retained in
  `planning_quality.jsonl`, both for the initial Planner response and across
  bounded repair attempts. This exposes whether the Planner originally obeyed
  each fixed story/tone/affect/opener contract; the post-override semantic
  coherence checks remain unchanged.
- `perspective_concentration` remains an audit/strict warning but no longer
  triggers a slot-local LLM repair. Structural branch ownership overwrites
  `perspective_id` before every evaluation, so such a repair could not change
  the concentration and only spent requests.
- Removed two other validations that cannot fire on the active path:
  `invalid_perspective` is deterministically canonicalized first, and
  `branch_route_conflict` compared a topology-owned branch ID with the same ID
  after normalization. The effective concentration, branch-goal, reply-delta,
  social-contract, capacity, and collision checks remain.
- Removed the retired tone-overlay control from current Writer inputs. Its two
  dataclass/persistence fields remain solely so historical records deserialize;
  current code neither assigns nor consumes them.
- Removed the unreachable `constructive_polite_helpful` finalizer branches and
  the unused scalar `projected_metric`; the live batched projection path remains.
- Replaced the old print-only content comparison with a pinned, tested matched
  audit automatically run after evaluation. The old tool matched lexical text
  correctly but compared generated emotion/story against the entire domain
  corpus. The new join uses the exact seed ID and product directory for real
  per-comment model rows, reports all 12 paired distances, Planner→Writer
  realization, repetition contributors, and explicitly weak helpful/profanity
  surface probes in machine-readable JSON and Markdown.
- Persist the exact evaluation-excluded reference metric template atomically in
  each post's `thread_plan`. The content audit now decomposes every metric into
  real → Planner target and Planner target → Writer output, with separate
  MWU/KS/Cliff/Wasserstein statistics. Legacy logs are accepted only when their
  post alignment is provably unambiguous.
- Replace pre-score cleanup with a byte-identical scoring snapshot. The output
  audit must reject bad Writer text or tree metadata; evaluation no longer
  edits, deletes, or normalizes the artifact it claims to measure.
- Move the active metric suite and formal distribution statistics behind small,
  pinned generalized modules. The matched evaluator and all scorer CLIs are now
  tracked in git; the recoverability audit checks both git tracking and the
  transitive local-import closure. Default parity excludes legacy revisers.
- Treat n=1 as descriptive at every output layer. MWU/KS numbers remain visible,
  but neither the matched evaluator nor `run_evaluate` can print a false
  `12/12 PASS` for one thread.

A zero-API audit separated target choice from realization. Across both the
10-thread diagnostic set and all 150 matched seeds, the selected excluded-real
Planner templates pass both MWU and KS on all 12 metrics. That is evidence that
the distribution sampler is working, not permission to tune against final test
p-values. On the historical v80 n=1 thread, for example, polite target/real are
0.249/0.232 but Writer output is 0.059; story target/real are 0.128/0.111 but
Writer output is 0.249. The next paid run should therefore test Writer
realization rather than rewrite the sampler.

Expected paid-run effects are bounded and falsifiable: fewer impossible Planner
repair requests, explicit counts of initial fixed-contract disagreement, and no
`tone overlay: none` Prompt noise. Content/metric success still requires the new
large-thread artifact followed by a sufficient-N matched evaluation.

Offline acceptance: all 285 generalized tests pass; the focused scorer test adds
3 more passes; Ruff passes on every active changed source; the camera-product
backend self-test passes; all 92 declared pins agree, all 67 active pins are git
tracked, and the active local-import closure has zero omissions, including
dynamically imported/launched runners and token tooling. The v80 185/186
artifact remains rejected, its content report replays under the strict legacy
join, and the exact seed-8 configuration passes `--prepare-only` without an API
call.

---

## v84 — complete Writer coverage and quote-safe recovery (2026-08-17)

Policy ID: `generalized-card-v2-complete-writer-coverage-v84-20260817`.

A full replay of the paid v80 seed-8 artifact found 186 Writer tasks but only
185 rendered comments. S99 exhausted three attempts because its scheduled quote
opener said to copy the exact parent line while `parent_copy` correctly remained
a hard failure. The generator then persisted the shortened thread under
`policy=persist_valid_comments`, and the output audit still considered its
99.46% accepted share evaluable. That silently changes comment-pair metrics and
the sampled tree, so it is not a valid matched-thread artifact.

Changed:

- Quote openers now request a short exact markdown excerpt, never the whole
  parent. A `parent_copy` finding is waived only when the Planner explicitly
  assigned `opener_type=quote`, the quoted tokens are a strict excerpt of the
  visible parent, and at least six words of independent reply remain. Ordinary
  parent copying is still a hard failure.
- Exact Writer coverage is now a pre-persistence invariant. After bounded
  same-slot hard recovery, any missing, skipped, or malformed record raises a
  recoverable post error; the incomplete post never reaches atomic persistence.
  The default still performs no hidden whole-post retry or additional API spend.
- Output audit independently rejects any recorded post whose planned slots,
  generation records, generated records, and rendered comments are not exactly
  equal, even when `accepted_share` exceeds the old threshold. This also protects
  evaluation of historical artifacts.
- Removed the unreachable `omit_without_backfill` branch and corrected run
  metadata to describe bounded Planner schema recovery followed by hard failure.

Expected metric effect before a paid run: no direct claim of better content
quality. The required effect is measurement validity: every evaluated generated
thread has exactly the matched structural slots, so Self-BLEU, Self-BERTScore,
semantic cosine, length CV, depth, virality, story, emotion, and tone metrics are
not computed on a silently shortened sample. The shorter quote instruction may
also reduce parent-line repetition, but that is a secondary hypothesis.

Offline acceptance so far: the updated audit rejects the existing v80 artifact
at 185/186 despite `accepted_share=0.9946`; focused coverage/quote/audit tests
pass; the complete suite passes 266 tests; Ruff and the camera-product backend
self-test pass; all 72 source pins agree. The exact formal seed-8 command passed
`--prepare-only` under the v84 policy with no API calls, and its temporary run
directory was moved to Trash so the formal tag remains available.

---

## v83 — matched-text semantic isolation (2026-08-17)

Policy ID: `generalized-card-v2-matched-text-semantic-isolation-v83-20260817`.

The v82 completion audit was extended from final Prompt strings back through
every expander callback that receives an anonymous matched-real body. Three
remaining paths still derived semantic controls from evaluation wording:

- lexical first-person and uncertainty markers temporarily licensed those
  frames before Planner restoration;
- a long anonymous slot was labelled `story_rant`, regardless of its plan;
- lexical prefixes such as `side note`, `unrelated`, `FWIW`, and `BTW` assigned
  a `side_tangent` real-surface shape, and `!template` assigned template meaning.

Changed:

- Matched wording can no longer license first-person or uncertainty. The two
  dead regex classifiers were replaced by one explicit false boundary; the
  Planner's story/evidence/stance contract remains the sole authority.
- Real-surface inference now uses only deleted/moderator metadata, word scale,
  question punctuation, dominant link/quote form, and identifier typography.
  Its neutral structural labels are `long_turn`, `full_answer`, and
  `compact_identifier_turn`, never story/rant/tangent labels.
- The generalized anchor builder explicitly discards `real_body`; facts still
  come only from seed, generated parent, and Planner/domain claim controls.

Expected direction before a paid run: fewer hidden Planner conflicts and fewer
comments whose story, uncertainty, gratitude, or tangent behavior mirrors the
matched evaluation comment rather than the planned slot. The tree and length
signals remain identical. Formal metric result is pending.

Offline acceptance: semantic-marker isolation tests pass, the complete suite
passes 263 tests, Ruff and backend self-test pass, and all 72 pins agree.

---

## v82 — focused Planner discourse handoff (2026-08-17)

Policy ID: `generalized-card-v2-focused-discourse-contract-v82-20260817`.

The post-v81 completion audit found one remaining Planner→Writer break in the
default path. The focused Writer received the planned proposition plus dedicated
tone/story/affect controls, but not the planned comment function, payload form,
speaker role, voice, evidence basis, content angle, stance, detail, decision intent,
reply relation, or local exclusion. A planned rant, correction, datapoint, or
bare reaction could therefore fall back to the model's generic helpful answer.

Changed:

- Add one compact, deduplicated discourse contract to the focused Writer. It
  carries those fields once without restoring the old full prompt, static metric
  guidance, overlapping surface paraphrases, or bulky payload instructions.
- Add an end-to-end contract test from raw Planner JSON through normalization,
  matched-slot expansion, finalization, and focused Prompt rendering. A valid
  `rant + ranter + hard_disagree` slot must retain each planned control exactly
  once.
- Replace the shared surface-texture classifier on the generalized path. Matched
  comment typography may shape typography, but words such as `thanks` and
  `appreciate` may no longer assign gratitude tone or a
  `pure_acknowledgement`; social meaning remains Planner-owned.

Predicted direction before a paid run: fewer generic customer-service/helpful
turns; more faithful rants, corrections, questions, datapoints, and terse social
moves; greater lexical and emotional variety, moving Self-BLEU/Self-BERTScore
and emotion-related rows toward real data. Story allocation and tree structure
are unchanged. Formal result: pending a new artifact and multi-thread evaluation.

Offline acceptance: the focused contract and matched-text isolation tests pass,
prompt size remains below the existing focused/full ratio gate, and the complete
suite passes 262 tests.

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
