# Handoff: GEO synthetic Reddit threads — v101 shipped at 9/0/3, six metrics still unsafe at N=150

## Session Metadata
- Created: 2026-08-20 18:23:25
- Project: /Users/yaoningyu/Desktop/UIUC/GEO
- Branch: `generator/v75-writer-realizes-planner-move`
- Session duration: one long session; v97 → v101 plus the reproducibility gate

### Recent Commits (for context)
  - `f56e055` docs: v101 N=10 result — 9/0/3, best in project history
  - `bb09d51` docs: domain generalization verified, and the granularity lesson
  - `c3abc42` fix: restore gratitude, and retract the long-slot claim
  - `31d9e7c` fix: the possessive cues asked for events and got narrative
  - `0395ccb` feat: v101 per-register realization
  - `21d20ff` feat: v100 measured closing move
  - `7edddd5` feat: v99 drawn realization of the assigned warm register
  - `187b438` feat: refuse a run whose sources are not committed
  - `1abdb0e` docs: add ORIENTATION.md as the single entry point
  - `e213f7a` feat: v97 keyboard surface and v98 drawn typing rhythm

## Handoff Chain

- **Continues from**: None (first handoff)
- **Supersedes**: None
- **Authoritative spec**: `docs/ORIENTATION.md` — read that first, it is maintained in place and this handoff points at it rather than duplicating it.

---

## Current State Summary

The project generates synthetic Reddit threads meant to be statistically
indistinguishable from real ones on 12 thread-level metrics, using first-pass
Planner→Writer generation only. This session took the generator from v97 to
**v101** and closed a reproducibility hole. v101's N=10 result is **9 PASS / 0
PARTIAL / 3 FAIL**, the best in the project's history (v98 was 8/1/3, v96 6/0/6),
with Cliff's delta improved on 8 of 12 metrics. **That is real progress and it is
not the same as being close.** At the final scale of 150 threads the pass
probability is a function of effect size, not the current p-value, and by that
measure **six metrics are safe and six are not**. The immediate work is the six
that are not, of which only two have a verified mechanism.

---

## THE GOAL (read this before anything else)

Two acceptance criteria, both from the user, both required:

1. **All 12 metrics at p > 0.05** — MWU *and* KS, two-sided, on 150 threads.
2. **A human reading the threads cannot tell generated from real.**

What "real" means here is the user's own framing and it is authoritative:

> 我们要模仿的是说话方式，而不是真正的 content。大家可能有问号、有句号、有提议、
> 有反对，会讨论不同的话题。而不是说总是一些 helpful 的，或者很不自然的、非人类的
> 讨论样子。

Not factual accuracy, not topical similarity — **the manner of speaking.** The
failure mode being designed against is the AI register: one voice, one level of
helpfulness, everything converging on one topic.

The user has also stated: **process does not matter.** Any mechanism that moves a
metric without leaking evaluation text is acceptable. The constraints in
`ORIENTATION.md` §4 exist to protect the *validity of the measurement*, not to
prescribe a method.

### The judging standard, and the trap in it

`run_evaluate.py:411`: PASS needs MWU p > 0.05 **and** KS p > 0.05. n=1 prints
`DESCRIPTIVE` and no p-value is meaningful.

**The trap: a high p-value at N=10 does not mean a metric is matched.** The test
is unpaired while the data is paired by seed, and the sample is small, so a
one-directional bias hides behind a wide spread. `self_bleu_4` has passed at
N=10 for five versions while being **above real in 10 of 10 threads**, and the
three N=150 runs ever done gave p = 1.4e-09, 1.0e-14 and 2.6e-17.

**Steer by |Cliff's delta| ≤ 0.10.** Use the p-value to report, never to decide.

---

## Where v101 actually stands

```
[evaluation-results] PASS/PARTIAL/FAIL: 9/0/3
self_bleu_4                  PASS   MWU=0.1041   Cliff=+0.44
self_bertscore_mean_f1       FAIL   MWU=0.0028   Cliff=+0.80
semantic_mean_cosine         PASS   MWU=0.8501   Cliff=-0.06
hard_disagree_rate           PASS   MWU=0.1735   Cliff=+0.37
polite_rate                  FAIL   MWU=0.0210   Cliff=-0.62
impolite_rate                FAIL   MWU=0.0046   Cliff=+0.76
neutral_rate                 PASS   MWU=0.0587   Cliff=-0.51
length_cv                    PASS   MWU=0.7337   Cliff=+0.10
avg_depth                    PASS   MWU=0.9095   Cliff=+0.04
structural_virality          PASS   MWU=0.9697   Cliff=+0.02
mean_story_probability       PASS   MWU=0.8501   Cliff=-0.06
emotion_entropy              PASS   MWU=0.8501   Cliff=-0.06
```

### The N=150 projection — this is the number that matters

| \|Cliff\| | metrics | P(pass) at N=150 |
|---|---|---:|
| ≤ 0.06 | `semantic_mean_cosine`, `mean_story_probability`, `emotion_entropy`, `structural_virality`, `avg_depth` | ~0.90 |
| 0.10 | `length_cv` | ~0.72 |
| 0.37–0.44 | `hard_disagree_rate`, `self_bleu_4` | ~0.01 |
| 0.51–0.80 | `neutral_rate`, `polite_rate`, `impolite_rate`, `self_bertscore_mean_f1` | ~0.00 |

**Six safe, six not.** And two of the six safe ones (`avg_depth`,
`structural_virality`) are copied from the real reply tree by the matched
sampler — they are not won by generation.

The user's own read of the remaining problems was correct and matches this:
`self_bleu_4`'s p is low, `self_bertscore` is bad, `hard_disagree_rate`'s MWU is
low, `neutral_rate` only barely passes, and the politeness pair still fails.

---

## Codebase Understanding

## Architecture Overview

```
real Reddit corpus (camera: 574 threads)
  ├─► seed pool ── stratified, deterministic, seed 42 ──► 150 matched seeds
  │      each seed keeps its real post ID, so every metric is matched 1:1
  └─► evaluation-EXCLUDED threads only ──► domain_profile.json (schema 18)
             measured shares: typography, layout, tone×length joint, sentence
             rhythm, final punctuation, length transfer function, per-register
             register moves, closing move
                          ▼
      Planner ──► per-slot controls (never sees matched real comment text)
                          ▼
      Writer  ──► ONE realization per slot. No resampling, no best-of-N.
                          ▼
      run_evaluate.py ──► 12 metrics → matched-seed stats → content audit
```

**The design law of the whole system:** a behaviour is measured on the
evaluation-excluded corpus, turned into a per-slot **draw** at the measured rate,
and given to the Writer as a **concrete surface act**. Prose descriptions of a
register do not reach the output; drawn surface acts do. That is the single most
load-bearing finding in the project — `TONE_DEFINITIONS["polite"]` describes
warmth in prose and realizes 19.3% of the time, while `sentence_rhythm` moved
seven habits to their measured rates within sampling noise.

## Critical Files

| File | Purpose | Relevance |
|---|---|---|
| `docs/ORIENTATION.md` | **The spec.** Goal, judging standard, all 12 metrics, method, current state, discipline, reproducibility | Read first. Rewritten in place; carries a "last verified" date and the checks behind it |
| `generalized_card/VERSION_LOG.md` | Every version, its arms, its predictions, its result | The evidence for every claim above |
| `tasks/todo.md` | Task list ordered by which measured gap it moves | What to do next |
| `tasks/lessons.md` | 50 mistakes, each with the rule that prevents it | Read before diagnosing anything |
| `tasks/v99-worklog.md` | The politeness diagnosis: 4 rejected hypotheses + the mechanism | Before touching tone or register |
| `generalized_card/analysis/politeness_diagnosis.py` | Reproduces every number in that worklog; `--run <dir>` retargets it | Re-measure instead of re-deriving |
| `generalized_card/generalized_card/register_realization.py` | Per-register drawn register moves (v99/v101) | The active tone mechanism |
| `generalized_card/generalized_card/closing_move.py` | Drawn closing move (v100) | The adjudication-frame fix |
| `generalized_card/generalized_card/sentence_rhythm.py` | The template every later mechanism copies | Read it to learn the pattern |
| `generalized_card/generalized_card/core_contract.py` | 104 pinned file hashes + policy versions | Not itself pinned — it cannot hold its own hash |
| `generalized_card/generalized_card/source_provenance.py` | Refuses a run whose sources are not committed | See "Potential Gotchas" |
| `generalized_card/generalized_card/prompts.py` | 2.7k lines; root/reply Planner + focused/full/low-info Writer | **Three** Writer paths — a change must reach all three |
| `generalized_card/scripts/run_generate.py` | CLI, run_config record, subprocess env | Where every arm is declared |
| `generalized_card/scripts/run_evaluate.py` | audit → stage → score → matched-evaluate | `REQUIRED_THREAD_METRICS` at line 28; status rule at 411 |

## Key Patterns Discovered

- **Every behaviour change is a named CLI flag whose legacy value reproduces the
  previous release byte-for-byte.** v101 ships 18. The flag goes into
  `run_config.json` and `RUN_EXPERIMENT_FIELDS`, and a resume with changed
  parameters is rejected, so one tag can never mean two configs.
- **A new mechanism = one focused module + one arm + a per-band measured profile
  + a per-slot hash draw + a realized-rate audit.** Copy `sentence_rhythm.py`.
  Namespace the hash (`f"register:{move}:{slot_key}"`) so drawing one habit does
  not correlate with drawing another.
- **Import modules, not values.** `from .x import SOME_DICT` captures the empty
  dict at import time. This bug has occurred **four** times here. Use
  `from . import x` and read `x.SOME_DICT`.
- **Never approximate a metric.** `self_bleu_4` needs no model and runs in
  seconds; `pairwise_self_bleu_for_order` takes **tokenized lists**, not strings.
- **The real per-comment classifier tables already exist** at
  `data/raw/discussions/<domain>/<product>/{politeness,storyseeker,go_emotions}_results.json`.
  Almost every diagnosis in this session ran on those and cost nothing.

---

## Work Completed

## Tasks Finished

- [x] **v97 + v98 committed** — both had shipped with their sources only in the working tree
- [x] **`docs/ORIENTATION.md`** created as the single spec; pointers added from README, HANDOFF, todo
- [x] **`source_provenance.py`** — a run now refuses to start unless every file defining the version is in `HEAD`
- [x] **v99** drawn realization of the assigned warm register
- [x] **v100** measured closing move — found the root of the adjudication frame chased since v73
- [x] **v101** per-register realization + the state-not-event cue correction + `gratitude` restored
- [x] **Large-thread gate protocol** adopted and documented (100–200 comments, judged on content *and* distance, before N=10)
- [x] **Domain generalization verified** on all four registered domains
- [x] **`politeness_diagnosis.py`** committed so the evidence is reproducible

## Files Modified

| File | Changes | Rationale |
|---|---|---|
| `register_realization.py` | new; then per-register; then `gratitude` restored, cue reworded to a state | The polite register realized 19.3% while impolite realized 89.7% |
| `closing_move.py` | new | Real text closes on a verdict 0.014 of the time, generated 0.265 — 19× |
| `source_provenance.py` | new | Two shipped versions had no recoverable source tree |
| `domain_profile.py` | schema 15 → 18 | Three new measured profiles |
| `prompts.py` | `_register_rule`, `_closing_rule` on all three Writer paths | v74 converted one path and made a release unattributable |
| `backend.py` | 3 new arms, profile installs, extended self-test | Arms must be switchable and self-verified |
| `core_contract.py` | 101 → 104 pins, policy v98 → v101 | Drift detection |
| `docs/ORIENTATION.md` | new, then maintained | `HANDOFF.md` had become unreadable as a spec |
| `generalized_card/analysis/politeness_diagnosis.py` | new | Seven scratchpad probes consolidated |

## Decisions Made

| Decision | Options considered | Rationale |
|---|---|---|
| Draw a concrete surface act per slot | Prose register description; drawn act | Prose realizes 19.3%; drawn acts hit their measured rate within 0.016 |
| Bundle v99+v100 in one gate | One arm per run; bundle with per-arm flags | Each arm has its own measurable realized rate, so one artifact gives per-arm attribution — the way v97's 4 and v98's 5 arms were attributed. Separate gates cost 4× for the same information |
| Enforce provenance in code | Write the rule down; gate the run | The rule was already written in `AGENTS.md` and was violated twice |
| Gate on a 100–200 comment thread | 45-comment gate; large thread | v97 and v98 both gated on 45 comments and neither gate predicted what N=10 found |
| Leave the reader-conditional close alone | Suppress it; leave it | Measured 1.53× real, not 19× — real people do end that way. It reads worse than it measures |
| Don't cue `link` | Cue it; leave it | Generated is 0.000 against a real 0.058, but a link needs a real URL and inventing one is a hard failure |

---

## Pending Work

## Immediate Next Steps

1. **`hard_disagree_rate` moved the wrong way (+0.29 → +0.37) and has never had a
   mechanism.** It passes on a wide spread, exactly like `self_bleu_4`. Start
   here: it is the cheapest unexplored metric, the scorer is
   `score_thread_disagreement.py`, and the real per-comment stance tables already
   exist at `data/raw/discussions/<domain>/<product>/stance_disagreement_results.json`.
   Diagnose it the way `politeness_diagnosis.py` diagnoses politeness.
2. **The possessive lever on `polite_rate`.** Generated carries `my X` at 0.081
   against a real 0.230, and the real conditional is P(polite | possessive) =
   0.509 against 0.254 without. This is the largest single untapped lever and its
   causal claim is measured — but it only works if the possessive arrives as a
   bare fact, which is what the v101 cue rewording is for and v101 has now
   confirmed does not raise story probability.
3. **`self_bertscore_mean_f1` — five hypotheses rejected, no mechanism.** Do not
   build a sixth without falsifying it first on the excluded corpus. Rejected:
   length spread, duplication tail, surface register, lexical breadth (r=+0.077),
   and narrow shared vocabulary (r=+0.155 and −0.096, both the wrong sign; the
   narrowness is *cross-thread* while the metric is *within-thread*).
4. **Decide the N=150 reporting standard before running it.** 12 metrics × 2
   tests at α=0.05 means a perfect generator passes all 12 together only
   0.94¹² ≈ 52% of the time. Either a multiplicity correction or effect-size-led
   reporting with |Cliff| ≤ 0.10. **This is the user's decision and it blocks
   N=150.**

## Blockers / Open Questions

- [ ] **Blocker:** the N=150 reporting standard (above). Needs: a user decision.
- [ ] **Question:** non-camera domains have only 177–201 eligible threads, so a
      150-thread evaluation pool consumes 75–85% of them and leaves 90–126
      reference threads against camera's 424. Suggested: use a smaller pool
      (≤100) for those domains, decided **before** building the seed pool.
- [ ] **Question:** `self_bleu_4` is fully characterised with an exact ablation
      harness — no phrase drives it, it is a length metric first, and generated
      already matches length. Entity diversity is worth ~1/3 of the gap at
      partial r = −0.097. There is no cheap lever. Suggested: treat it as a
      known-unsolved and spend effort on `hard_disagree_rate` instead.

## Deferred Items

- Eye-visible tells, criterion 2, none fixed: **no generated comment contains a
  link** (real 0.051), `check` at ~10× real, entity diversity 0.438× real,
  `will` at ~1% of real.
- The impolite bleed (planned-neutral realizes impolite 0.513). Suppression was
  measured and does **not** work: lift 1.02–1.18×, counterfactual 0.697 → 0.655
  against a real 0.443.
- Bug: evaluation drops <2-word comments unevenly, so `--exact-matched-thread-size`
  can still mismatch.
- Bug: the slot distribution schedule is never persisted to `discussion.json`.
- Bug: `--template-phrase-reuse-budget 4` is flat and wrong at large thread sizes.
- `--no-story-scope` should revert to `tense`: no metric benefit, and it added
  new repeated 4-grams. The prompt-contradiction fix it carried is worth keeping.

---

## Context for Resuming Agent

## Important Context

**This project's failure mode is not bad code — it is confident wrong diagnosis.**
Every version that moved a metric did so after a hypothesis survived an attempt to
kill it, and every wasted run came from one that was not tested first. The single
highest-value action available is free: **measure the candidate mechanism on the
evaluation-excluded real corpus and try to refute it before writing any code.** In
this session that step rejected **nine** hypotheses, four of them after the module
was already written.

**The error class that produced every wrong call in this version line was
reasoning at a coarser granularity than the mechanism acts at.** Four instances,
all documented in `tasks/lessons.md` (2026-08-20):

1. Predicting corpus-wide rates for a cue that fires on 25% of slots. On its own
   slots the baseline was already *above* the real corpus rate.
2. Writing a cue as an event ("what you ended up keeping") and getting narrative;
   `mean_story_probability` went from 0.8% error to 29.2%.
3. Excluding `gratitude` on a pooled 1.25× figure when, conditioned on the
   register the cue fires on, real polite micro comments thank at 0.330 against a
   generated 0.100 and real polite *short* at 0.165 against **0.000**.
4. Splitting an attribution on `real_word_count` read back from the artifact —
   **it is not persisted and is 0 on every row**, so it silently fell back to
   generated length. Redone on `length_bucket` the conclusion reversed.

**Before predicting anything, write down the population the mechanism acts on and
measure the baseline on exactly that population.**

**Claims retracted in this session — do not reuse them:**
- ~~"the no-story instruction cut advice 0.090 → 0.008"~~ — probe artifact
- ~~"`self_bertscore` is lexical breadth"~~ — r = +0.077, null
- ~~"the tone gap is mostly a length effect"~~ — warmth markers, lift 3.56×
- ~~"the verdict suppression strips polite appraisal from long slots"~~ — real
  polite comments close that way at 0.010–0.029; the drop was 4/9 → 1/9, noise
- ~~"negative markers at 3× real should be suppressed"~~ — the decomposition is a
  **+8.381 polite deficit against a −0.767 impolite excess**; generated uses
  *less* of the impolite vocabulary than real, so suppressing it hurts

## Assumptions Made

- The evaluation classifiers are ground truth for their metrics. The mechanisms
  target what the classifier responds to, not a general notion of politeness.
- A cue that names a surface act is obeyed at 0.3–0.7 on the slots that receive
  it. Measured on the v100 gate; used in every prediction since.
- `length_bucket` is a faithful proxy for the matched slot's size band in the
  artifact. Verified against `structure_bucket`; `real_word_count` is not.
- N=10 with paired seeds is enough to rank two versions on effect size. Not
  enough to resolve a per-band effect: at 5–18 slots per band a 3-comment swing
  is noise, and one claim was retracted for exactly that.

## Potential Gotchas

- **A run will refuse to start if the version is not committed.** That is
  `source_provenance.py` doing its job. Commit; do not reach for
  `GENERALIZED_CARD_ALLOW_UNCOMMITTED_SOURCE=1`, which stamps the artifact as
  unreproducible.
- **Never edit a pinned file while a generation run is in flight.** 104 files are
  hash-pinned and `verify_core_contract` aborts the next batch. Park the change
  and land it as its own version.
- **`git add` new files before `repin_core_contract.py --write`** — it refuses to
  pin a source that is not recoverable from git.
- **`prompts.py` has three Writer paths** (focused, full, low-info). The focused
  arm has been active since v82. A rule added to one path and not the others makes
  the release unattributable — that has happened.
- **`core_contract.py` is not itself pinned** (it cannot hold its own hash), which
  is why `version_source_paths()` adds it to the provenance check explicitly.
- **No domain vocabulary in Writer-facing cue text.** Every test runs on camera,
  so nothing else will catch it. There is a test asserting this over all 7 cues.
- **The 12 metrics are `REQUIRED_THREAD_METRICS` at `run_evaluate.py:28`** and the
  status rule is `_metric_status` at line 411. Read them; do not recall them.
- The user runs the paid commands. Hand over an exact command, dry-run with
  `--prepare-only` on a throwaway tag first, and delete the tag.
- A stray 170-byte file named ` --resume` sits in the repo root from a v75-era
  shell mistake. Harmless; flagged, not deleted.

---

## Environment State

- **Tests:** `PYTHONPATH=generalized_card .venv/bin/python -m pytest -q generalized_card/tests` → **537 passed**
- **Lint:** `ruff check generalized_card` → clean
- **Pins:** `python3 generalized_card/scripts/repin_core_contract.py` → 104 pinned, 0 missing, 0 untracked active, 0 unpinned local imports, **0 drift**
- **Working tree:** clean for every pinned source against `HEAD`
- **Policy:** `generalized-card-v2-per-register-realization-v101-20260820`
- **Domain profile schema:** 18
- **Arms:** 18 (see `ORIENTATION.md` §4). `--register-realization measured`, `--closing-move measured` are the two newest
- **API key:** `third_party/MiroFish/.env` as `LLM_API_KEY`; pass `--api-key-env LLM_API_KEY`
- **Latest runs:** `..._v101_register_n10_20260820_v1` (N=10, 9/0/3), `..._v100_closing_seed8_20260820_v1` (large-thread gate, 186 comments, $1.14)

### The standing workflow

```
measure the gap on the artifact (real scorer, never an approximation)
  → form a causal hypothesis naming a mechanism
  → TRY TO FALSIFY IT on the evaluation-excluded corpus, before any code
  → build one arm per mechanism, legacy value reproduces the previous release
  → verify offline: tests, ruff, repin (0 drift), self-test with the arm on AND off,
    profile rebuild with 0 seed overlap, draw fidelity, --prepare-only dry run
  → ONE paid large-thread gate (100-200 comments), judged on content AND distance
  → N=10 paired to the previous version's seeds
  → write down what was REJECTED, not only what shipped
```

### The stop ritual — do this every time you stop, not only at version end

Update, in this order: `docs/ORIENTATION.md` (current state, next step, anything
that changed the goal/method/metric interpretation) → `tasks/todo.md` →
`tasks/v<N>-worklog.md` → `generalized_card/VERSION_LOG.md` →
`generalized_card/RUN_INDEX.md` → `tasks/lessons.md` after **any** correction.

Before stopping, the notes must answer three questions without the chat history:
**what is the next step and why that one; what checks did you run and what did
they say; what did you reject and what measurement rejected it.**

---

## Related Resources

- `docs/thread_metric_score_reference.md` — every exported metric, its scorer, its model
- `generalized_card/AGENTS.md` — binding engineering rules for `generalized_card/`
- `generalized_card/RUN_INDEX.md` — all runs with tag, threads, coverage, pass count
- `tasks/HANDOFF.md` — long-form evidence archive, append-only, **not a spec**
- `generalized_card/CARD_CORE_PARITY.md` — the algorithm/domain boundary

---

## TYPE INSTRUCTION — how the next session should start

Paste this to the next agent verbatim.

> You are continuing a research project as a **senior AI research scientist**, not
> as a coding assistant taking tickets. Think comprehensively, understand the
> mechanism deeply, be your own reviewer, and never be satisfied with a fix that
> has not been shown to move what it claimed.
>
> **Start here, in this order:**
> 1. `docs/ORIENTATION.md` — the whole spec. It is authoritative over everything else.
> 2. `generalized_card/VERSION_LOG.md`, the v101 section — the current result and the honest N=150 projection.
> 3. `tasks/lessons.md`, the 2026-08-20 entries — the four instances of the one error class that caused every wrong call in this version line.
> 4. `tasks/todo.md` — the task list.
> 5. This handoff's "Immediate Next Steps".
>
> **The rules that are not negotiable:**
> - **Read every related file end to end before diagnosing.** Not grep hits, not
>   "the relevant function". A module was written, tested and pinned on an
>   unverified hypothesis in this project because that step was skipped.
> - **Try to falsify your own hypothesis on the evaluation-excluded corpus before
>   writing code.** It is free and it rejected nine hypotheses in one session.
> - **Steer by |Cliff's delta| ≤ 0.10, not by the p-value.** A metric passing at
>   N=10 while sitting above real in 10 of 10 threads is not passing.
> - **Before predicting, write down the population your mechanism acts on** and
>   measure the baseline on exactly that population.
> - **Simplify as you go.** One mechanism per focused module behind a small
>   interface. Do not grow `backend.py` or `prompts.py`.
> - **Commit at every version boundary, before the paid run.** The run will refuse
>   to start otherwise, and that gate exists because the rule was violated twice.
> - **Maintain the docs every time you stop** — see "The stop ritual" above.
> - **Report faithfully.** If a check was skipped say so; if a number is a probe
>   rather than the real scorer say so; retract your own claims when a measurement
>   contradicts them, and record the retraction where the claim was.
>
> **The first thing to do:** `hard_disagree_rate` moved the wrong way in v101
> (+0.29 → +0.37) and has never had a mechanism. Diagnose it the way
> `generalized_card/analysis/politeness_diagnosis.py` diagnoses politeness — the
> real per-comment stance tables already exist under
> `data/raw/discussions/<domain>/<product>/stance_disagreement_results.json`, so
> the diagnosis costs nothing. Do not write code until a hypothesis survives a
> falsification attempt.
>
> **What not to do:** do not run N=150 until the user has chosen the reporting
> standard (12 metrics × 2 tests at α=0.05 means a perfect generator passes all 12
> together only ~52% of the time). Do not build a sixth `self_bertscore`
> hypothesis without falsifying it first. Do not suppress negative vocabulary —
> that was measured and it makes the metric worse.
