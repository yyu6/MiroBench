# Handoff: GEO synthetic Reddit threads — two arms fired at 100%, both produced nothing on the priority metrics, and the reason is a causal-identification error that kills a whole mechanism family (v109, v110)

## Session Metadata
- Created: 2026-08-25
- Project: /Users/yaoningyu/Desktop/UIUC/GEO
- Branch: `generator/v75-writer-realizes-planner-move`
- Session duration: one long session; v108 N=10 analysis → G24–G35 → v109 built and gated → G36–G41 → the pricing sweep that killed five candidates (G42–G45) → v110 built twice and gated at N=10 → G46–G49. **$4.94 total paid spend** ($1.1785 seed-8 v109 + $3.7599 N=10 v110).
- **Net metric outcome: nothing moved.** v108 N=10 was 10 PASS / 0 PARTIAL / 2 FAIL. v110 N=10 is 9/1/2. `self_bleu_4` is byte-identical between them.

### Recent Commits (newest first, this session's work)
  - `21e793c` gate(v110, N=10): the arm fired 532/532 and did nothing — the asked word count is not a causal instrument
  - `1eba3a4` docs(v110): N=10 gate predictions and the honest closure-to-p mapping
  - `f16e8f4` docs(v110): predictions written before spending, and the ledger that killed five candidates
  - `f8f8be5` v110: repin for the refit length transfer
  - `abcf0cd` v110: refit the length transfer function the calibration inverts
  - `357c728` v110: repin the core contract for length_fidelity
  - `5cea352` v110: hold each slot's realized length inside its measured length band
  - `0c0e256` diagnose: price every candidate lever on self_bleu_4 and self_bertscore before building
  - `321af5d` gate(v109): arm works on naming shape, causally worsens both priority metrics
  - `5ecb433` docs(v109): gate predictions and guardrails, written before any spend
  - `9b4a9b7` v109: per-slot referent spread — name as many things as a real thread
  - `8367191` correct(G35): G34's generalisation retracted — generated is off-manifold
  - `e936041` reject(G34): both v109 and v110 killed by falsification before any code
  - `a18f95a` correction(G33): the probe spot-check rejected 2 of 7 clause detectors
  - `0332eb0` analysis(G33): self_bleu_4's function-word excess is a conditional-advisory register
  - `56bbf3f` analysis(G31/G32): two mechanisms with measured targets for self_bertscore
  - `94d50da` docs: G29/G30 — dead per-slot temperature, untested persona arm
  - `abca23f` analysis(G28): the Writer's whole input separates with depth, the output does not follow
  - `431d462` docs: G26/G27 — both self_bertscore mechanisms targeted 12% of the defect
  - `5a05620` docs: G25 — polite_rate is not a closed limit; correct G24's "isolated" claim
  - `00089e9` analysis: decompose self_bleu_4's gap — it is a 1/2-gram register defect
  - `12d7863` docs(v108): N=10 gate result

## Handoff Chain

- **Continues from**: [2026-08-23-114452-geo-v108-semantic-coverage-nonrepeat-and-the-writer-realization-turn.md](./2026-08-23-114452-geo-v108-semantic-coverage-nonrepeat-and-the-writer-realization-turn.md)
  - That handoff's first task — the isolated N=10 gate for `--semantic-coverage-nonrepeat` — was done and is now **G24**: no pooled improvement, nominally worse. This session then measured that same arm at **~0.0007 on seed 8** (G39), i.e. indistinguishable from nothing. Its headline claim ("best single-thread self_bertscore result yet") was **thread noise**, not the arm.
- **Supersedes**: that handoff's framing of `self_bertscore_mean_f1` as reopened. It is not closed and it is not reopened — it is *unexplained* after this session's pricing sweep, with ~60–70% of its gap attributed to nothing.

## Current State Summary

Two releases were built, gated with real money, and **neither is promoted**. Both arms fired at 100% — no wiring bugs this session — so both nulls are real mechanism failures, not plumbing failures. The two paid runs bought three things worth more than the code:

1. **A quantified target.** `p ≈ 0.5–0.6` (the user's stated standard) requires **~90% gap closure at N=150, ~75% at N=50, ~50–75% at N=10** (G42, simulated over 763 real threads). This retires the entire 5–10% mechanism class the last six releases came from.
2. **A priced ledger of every candidate**, with five killed before any code was written (G40, G44, G45).
3. **A causal elasticity that kills a family**: the asked word count has essentially **no** leverage on realized length where the compression lives (G48).

Formal metrics, v110 N=10 (seeds 2–11, 532 comments, matched real 532):

| metric | gap | MWU | KS | Cliff | status |
|---|---:|---:|---:|---:|---|
| `self_bleu_4` | +0.0049 | 0.1212 | 0.4175 | +0.42 | PASS (weak) |
| `self_bertscore_mean_f1` | +0.0148 | 0.0140 | 0.0123 | +0.66 | **FAIL** |
| `impolite_rate` | +0.1235 | 0.0101 | 0.0123 | +0.69 | **FAIL** |
| `polite_rate` | −0.1617 | 0.0452 | 0.0524 | −0.54 | PARTIAL |
| the other eight | — | 0.38–0.97 | — | — | PASS |

`self_bleu_4` did not move at all from v108 (gap +0.0049 → +0.0049, MWU 0.121 → 0.1212, Cliff +0.42 → +0.42, 5/10 threads, Wilcoxon p=0.695). `self_bertscore` moved 0.0188 → 0.0148 nominally but **7/10 threads, Wilcoxon p=0.064, sign test p=0.344**, and the channel that was supposed to produce it did not operate — so it may not be claimed.

## Architecture Overview

Unchanged from the previous handoff. `generalized_card/` wraps the CARD generator: a Planner produces one plan per matched real comment slot, a Writer realizes each slot, and every behaviour change ships as a named CLI arm whose legacy value reproduces the prior release byte-for-byte and is recorded in `run_config.json` + `RUN_EXPERIMENT_FIELDS`.

Two things about the architecture that this session learned the hard way:

- **There are THREE Writer prompt templates**, not two: `_focused_writer_prompt` (the default, 154/186 slots on seed 8), the full inline template (0 slots on seed 8), and **`_low_info_writer_prompt` (32/186)**, which renders neither the anchors block nor any equipment/referent offer. Any rate-drawn offer mechanism is capped at ~83% of slots for a reason that is not a bug (G41).
- **`--writer-retries` defaults to 0**, so `total_attempts = 1` and the Writer validation loop never gets a second attempt. On the v109 gate **65 of 186 slots failed their own validator and all were shipped unretried**, including 32 `template_phrase_reused`. The whole validation layer is currently decorative.

## Critical Files

- `docs/ORIENTATION.md` — the spec. §4 "What may never happen" is non-negotiable. Last verified 2026-08-24.
- `docs/DECISIONS.md` — **G1 through G49**. The next new row is **G50**.
- `generalized_card/VERSION_LOG.md` — v109 and v110 entries, each with predictions written *before* spending and results checked against them line by line.
- `tasks/lessons.md` — three new entries this session.
- `generalized_card/generalized_card/length_calibration.py` — carries the v97 constants (shipped) and the refit constants (G46/G48), plus the arm.
- `generalized_card/generalized_card/length_fidelity.py` — the band gate, **built, tested, priced, and deliberately shelved** at default `off`.
- `generalized_card/generalized_card/entity_spread.py` — v109's arm, default `off`, not promoted.

New analysis scripts, **all of which fidelity-check against the shipped metric before printing anything** (rule E6):
- `analysis/self_bleu_decomposition.py` — exact per-order BLEU decomposition + 1-gram token attribution.
- `analysis/composition_decomposition.py` — Oaxaca-style pair reweighting: composition vs content, both metrics.
- `analysis/feature_leverage.py` — reverse ablation on real: how much of the gap each real feature's absence explains.
- `analysis/plan_collision_leverage.py` — prices the Planner's own unresolved duplicate plans.
- `analysis/entity_spread_gate_audit.py` — the v109 randomised within-run experiment (rate, naming, shape, cosine, bertscore, labels, mediation).
- `analysis/subject_rename_ablation.py` — prices a subject re-mention cap.
- `analysis/prompt_convergence_diagnosis.py` — the rejected prompt-dilution hypothesis.

## Key Patterns Discovered

**1. The metric arithmetic that has now bitten three times.** A large pair-level effect inside a tiny pair-count share cannot move a mean over all pairs. G17 → G24 → **G45**: the Planner's own log flags 63 unresolved semantic collisions at mean similarity 0.790 (max 1.000) touching 87/186 slots, each flagged pair scoring +0.0139 `self_bleu_4` / +0.0536 `self_bertscore` above unflagged — and repairing all of them closes **0.8%**, because 63 of 16,836 pairs is 0.37%. Full-population sweep: only 1.03% of slot pairs exceed plan similarity 0.70. **Always compute the population share before valuing an effect size.**

**2. A rate-drawn arm on a content-independent key is a free randomised experiment.** `entity_spread` draws on `sha256("entity_spread:{seed_key}:{local_task_id}")` and `local_task_id` is Planner traversal order, independent of the slot's plan. That made fired-vs-not a genuine treatment contrast inside one paid run — which is what rescued the v109 analysis after the gate command accidentally dropped v108's arm. **Design future rate-drawn mechanisms this way on purpose.**

**3. …but a randomised subgroup *mean* is not a thread-metric counterfactual when the treatment moves the metric's dominant covariate** (G47). It worked for `self_bertscore` (untreated 0.5033 vs v108's 0.5026) and **failed for `self_bleu_4`** (neither-fired 0.0383 > both-fired 0.0339 > one-fired 0.0323, non-monotone) because the arm added ~15 words per treated comment and `self_bleu_4` is length-dominated through its add-one smoothing.

**4. The causal-identification error, which is the most important thing in this handoff (G48).** `length_calibration` inverts a fit of `realized ~ asked`. But `asked` is a *deterministic monotone function* of `real_word_count`, which also drives the layout guidance, the development-beat count, the surface skeleton and the token ceiling. **That regression has no identifying variation.** R² = 0.879 and stability across four runs made it look solid; it was measuring the joint scaling of every size-keyed cue and attributing it to the number. v110 broke the collinearity for the first time (ask moved, assignment fixed) and measured the true elasticity:

| assigned words | ask change | realized change | elasticity |
|---|---|---|---:|
| 50–99 | 77.7 → 88.8 (+14.3%) | 60.0 → 59.8 (−0.3%) | **−0.02** |
| 100+ | 224.7 → 275.3 (+22.5%) | 157.0 → 160.7 (+2.4%) | **0.11** |
| 25–49 | 34.4 → 36.9 | 32.7 → 33.2 | 0.21 |
| 1–9 | 5.0 → 4.4 | 7.4 → 6.9 | 0.55 |

against the **1.21** the mechanism assumed. **Every "ask for more words" mechanism is dead.** This retrospectively explains why v96/v97's prompt-wording work only moved the 250w+ ratio 0.61 → 0.71.

**5. `self_bleu_4` is not a 4-gram metric.** It is the geometric mean of add-one-smoothed 1-, 2-, 3-, 4-gram precisions × brevity penalty. After v109 the decomposition is **p1 56.3%, p2 54.9%, BP 30.1%, p3 −16.8%, p4 −24.5%** — phrase repetition is *below* real; the residual is vocabulary overlap and the brevity penalty. Exact 1-gram attribution: **`the` alone carries 20.9%** of the positive excess mass (0.196 vs real 0.130), then `canon` 9.4%, `,` 9.0%; most under-shared are `.`, `to`, `i`, `of`, `with`, `for`, `and`, `but`, `be`, `have`, `are`.

**6. Off-manifold extrapolation invalidates between-real-thread slopes** (G35, carried forward). Generated sits at the 0.0 percentile of real big threads on function-word spread, so a within-range correlational slope is not an estimator for it. This retracted G34's generalisation mid-session.

## Tasks Finished

- v108's N=10 result analysed and recorded (G24), including a correction: my own claim that the comparison was "isolated" was wrong — six arms differed.
- `self_bleu_4` decomposed exactly, per order and per token (G27, G40).
- `self_bertscore`'s excess localised by depth: [2,4)+[4,7) carry **82.7%**; [7,+) — the only population v105 and v108 acted on — carries 11.9% (G26).
- Three silently-inert controls found: `writer_temperature` dead on every paid run, `sentence_route` empty on 532/532, `REPORTED_TONE_CLASSES` unreferenced (G29, G32).
- The designed-in tidiness machine documented: `forbidden_decision_subjects` on 532/532 slots, 10.1 subjects forbidden per slot (G35).
- **v109 built, gated ($1.1785), analysed, not promoted** (G36–G41).
- **The pricing sweep** (G42–G45): the target quantified, five candidates killed.
- **v110 built twice** (band gate, then the refit), gated at N=10 ($3.7599), **not promoted** (G46–G49).
- 673 tests pass; ruff clean; core contract 0 drifted.

## Files Modified

See `git diff --stat 25e6200..HEAD`. 46 files, +9,869 / −2,912. The load-bearing ones: `length_calibration.py`, `length_fidelity.py` (new), `entity_spread.py` (new), `length_policy.py`, `writer_quality.py`, `backend.py`, `run_generate.py`, `core_contract.py`, `docs/DECISIONS.md`, `generalized_card/VERSION_LOG.md`, `tasks/lessons.md`, and seven new analysis scripts.

## Decisions Made

- **G24–G35**: v108's N=10 null and its decomposition; `self_bleu_4` is a 1/2-gram metric; `polite_rate` is not closed; three inert controls; G34 retracted by G35.
- **G36–G41**: v109's arm works on naming shape (mentions-per-name 4.286 → 2.333 against real 2.432; distinct designators 21 → 69) and **causally worsens both priority metrics** (randomised, dose-response: `self_bertscore` +0.0255, cosine +0.0635 both-fired vs neither); it raises P(story) per treated comment; the gate command accidentally dropped v108's arm; a third Writer template exists.
- **G42–G45**: the target quantified; length composition priced at 31–37% / 14–26%; links and markdown emphasis at 12.4% / 14.6%; Planner de-duplication ≤2.4% / ≤1.8% — killed.
- **G46–G49**: the transfer function refit; the collinearity error; the elasticity measurement; v110's null.

## Immediate Next Steps

1. **Decide the length instrument, cheaply.** G43's target (31–35% of `self_bleu_4`, 14–18% of `self_bertscore`) still stands; only the *ask* was disproven as its instrument. Two untested instruments remain:
   - **Validator + retry.** `length_fidelity` is already built, tested and shelved at default `off`, and `--writer-retries` is already a flag defaulting to 0. A retry note is a *different signal* from the ask number (explicit "you wrote short, lengthen" vs a target integer), and its elasticity is unmeasured. **Test this with a cheap probe, not another $3.76 N=10** — a single-thread seed-8 run at `--length-fidelity measured --writer-retries 2` is ~$1.4 and the elasticity is readable from the free assigned-vs-realized audit with no metric at all.
   - **Structural cues** (development beats, paragraph layout, surface skeleton). These are what the collinear fit was actually picking up, so they are the live candidates. **Not yet priced** — price them offline first.
2. **Do not build anything in the 5–10% class.** G42 is the gate on what is worth building at all.
3. **`self_bertscore`'s remaining ~60–70% is unexplained.** Its input side is closed (G28: the Writer's whole input separates with depth and the output does not follow), G20 forbids output-side selection, and the pricing sweep attributed only ~30%. This is the hardest open problem in the project and should probably be attacked by measurement, not by another mechanism.
4. **Turn the writer retry loop on, as correctness.** 65/186 slots failed their own validator and shipped unretried, including 32 template-phrase reuses that are visible by eye in the highest-similarity pairs. This is a bug independent of any metric.

## Blockers/Open Questions

- **N=150 has never been run.** Every result in this project is N=1 or N=10.
- **No paid run has ever been done off camera** (D3). All four domains build and prepare end to end, and `length_fidelity` profiles produce genuinely different decile cuts per domain (camera [6,11,16,22,29,38,52,72,111], headphone roughly half), but domain adaptivity is untested where it counts.
- **The length transfer function is a recorded constant, not a profile**, on the argument that it is a property of model+prompt rather than domain. Untested off camera.
- **The reporting standard (J2 Holm–Bonferroni vs raw p>0.05) is still the user's open decision.**

## Deferred Items

- Links (8.8% / 10.8%) and markdown emphasis (3.6% / 3.8%) — priced, deferred on implementation risk: generating URLs risks fabricating sources, and a prescriptive cue risks G37's convergence failure.
- `--persona-conditioning` — the one existing mechanism matching the root-cause reading, never run in a comparable arm (G30).
- The politeness trio — `polite_rate` crossed PASS → PARTIAL this session and its deficit is localised to 25+ word comments (G25).
- `length_fidelity` — built and shelved; see next steps.

## Important Context

- **The user is the spend gate.** They run every paid command themselves. Give them the command, the free pre-flight check, and predictions written down *before* spending.
- **The user has stated repeatedly that they want no wasted work.** They explicitly asked for candidates to be verified/priced before building, and this session's pricing sweep is the pattern they approved. Killing your own hypothesis in writing is what they want to see, not a liability.
- **The user's target is `p ≈ 0.5–0.6`**, not `p > 0.05`. Do not report a 5% closure as progress.
- **A paper submission is imminent and time is short.** Prefer cheap offline pricing and single-thread probes over N=10 runs.
- Credential: `LLM_API_KEY` from `third_party/MiroFish/.env`.

## Assumptions Made

- v108 N=10 is a usable baseline for v110 despite `semantic_coverage_nonrepeat` being on in both — verified single-armed by `run_config.json` diff (three of the four differing keys are new fields recorded at defaults that reproduce v108's behaviour).
- The refit transfer function's stability across four runs implied causal validity. **This assumption was wrong and is now G48.**

## Potential Gotchas

- **`--writer-retries` defaults to 0.** Any mechanism that works through the validator does nothing unless this is raised.
- **Every arm defaults to `off`**, so a gate command that names only the new arm silently drops every previously-won arm. This cost the clean single-arm reading of the $1.18 v109 run (G39). Diff the intended command's arm list against the previous paid run's `run_config.json` before spending — it is free.
- **`grep -c` on `generation_records.json` double-counts**: each record stores the prompt twice (`prompt` and `attempts[0].prompt`). The v109 audit read 180 and the true slot count was 90.
- **Comment id key is `comment_id`, not `id`**, in generation records. BERTScore pair files key on `left_comment_id`/`right_comment_id`.
- **`repin_core_contract.py --write` cannot rewrite a hash written as a Python expression** (`"0" * 64`). Write a literal 64-char placeholder.
- **argparse %-formats help strings** — escape `%` as `%%` in arm help text.
- **The self_bertscore and semantic_uniformity scorers do not save pair-level output by default.** `--include-pairs` exists for BERTScore; the semantic scorer saves per-comment embeddings, which reproduce its thread mean to 7e-09.
- Stale `__pycache__` can produce a spurious core-contract mismatch; re-run `repin_core_contract.py --write`.

## Tools/Services Used

- `gpt-5.4-mini` via `https://api.openai.com/v1`, `LLM_API_KEY`.
- Scorers: `microsoft/deberta-xlarge-mnli` (BERTScore, L40, no idf, no rescale, CPU), `sentence-transformers/all-mpnet-base-v2` (semantic uniformity and plan-collision embeddings), StorySeeker, GoEmotions, politeness, Stance_Rel, Detoxify.
- `scipy` for Wilcoxon / MWU / KS / binomial tests.

## Active Processes

None. All background jobs completed. Scratch artifacts live in the session scratchpad and are not required — every reported number is reproducible from a committed script.

## Environment Variables

- `LLM_API_KEY` (from `third_party/MiroFish/.env`)
- Arms reach the backend as `GENERALIZED_CARD_*` env vars set by `run_generate.py`; new this session: `GENERALIZED_CARD_ENTITY_SPREAD`, `GENERALIZED_CARD_LENGTH_FIDELITY`, `GENERALIZED_CARD_LENGTH_TRANSFER`.

## Related Resources

- Runs: `artifacts/generalized_card/runs/v109_entity_spread_seed8_20260824_v1`, `.../v110_length_transfer_n10_20260824_v1`, baseline `.../generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1`.
- Real baseline: `artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv` (763 threads).
- Seed pool: `artifacts/generalized_card/seed_pools/camera_product_150_seed42.json`.

## TYPE INSTRUCTION — how the next session should start

Paste this to the next agent verbatim.

> You are continuing a research project as a **senior AI research
> scientist**, not a coding assistant taking tickets. You have full
> technical and design autonomy — the user has stated this explicitly and
> repeatedly; route only spend and credential decisions back to them and
> decide everything else yourself. The user's standing instruction is
> that they do not want wasted work: **price every candidate mechanism
> before building it, try to kill your own hypothesis first, and write
> down what you rejected.** A paper submission is imminent, so prefer
> cheap offline pricing and single-thread probes over N=10 runs.
>
> **Start here, in this order:**
> 1. `docs/ORIENTATION.md` — the whole spec, including §4's non-negotiable
>    "what may never happen" list. Read it by name; don't rely on memory.
> 2. `docs/DECISIONS.md` — every rule in force, **G1 through G49**. The
>    next new row is **G50**.
> 3. `.claude/handoffs/2026-08-25-geo-v109-v110-two-live-arms-two-nulls-and-the-causal-instrument-problem.md`
>    — this handoff, especially "Key Patterns Discovered".
> 4. `generalized_card/VERSION_LOG.md` — the v109 and v110 entries show
>    the required shape: predictions written before spending, then results
>    checked against them one by one.
> 5. `tasks/lessons.md` — three new entries; the arm-list one is the most
>    likely to recur.
>
> **The state in one paragraph.** Two releases (v109 entity spread, v110
> refit length transfer) were built and gated with real money this
> session. Both arms fired at 100% — verified in the saved prompts, no
> wiring bugs — and **neither is promoted**. v110 N=10 is 9 PASS / 1
> PARTIAL / 2 FAIL against v108's 10/0/2, and `self_bleu_4` is
> byte-identical between them. `self_bertscore_mean_f1` and
> `impolite_rate` remain the two FAILs.
>
> **The three findings that should govern what you do next:**
> - **G42 — the target, quantified.** `p ≈ 0.5–0.6` needs **~90% gap
>   closure at N=150, ~75% at N=50, ~50–75% at N=10**. Do not build
>   anything in the 5–10% class; six consecutive releases came from that
>   class and none moved a metric.
> - **G48 — the asked word count is not a causal instrument.** The length
>   calibration inverts a fit of `realized ~ asked`, but `asked` is a
>   deterministic function of `real_word_count`, which also drives the
>   layout, the beat count and the token ceiling — so that regression has
>   **no identifying variation**, despite R²=0.879 across four runs. v110
>   broke the collinearity and measured the true elasticity: **−0.02 at
>   50–99 words, 0.11 above 100**, against the 1.21 assumed. Every "ask
>   for more words" mechanism is dead. **Before you invert any fit, ask
>   what varies independently in it.**
> - **G45 — population share before effect size.** The generator flags 63
>   unresolved duplicate plans at mean similarity 0.790, touching 87/186
>   slots, each worth +0.054 `self_bertscore` per pair — and fixing all of
>   them closes 0.8%, because it is 0.37% of pairs. This arithmetic has
>   now killed three separate mechanisms (G17, G24, G45).
>
> **Your first task**, which is cheap and decides the next release:
> G43's length-composition target (31–35% of `self_bleu_4`, 14–18% of
> `self_bertscore`) still stands — only the *ask* was disproven as its
> instrument. Two instruments remain untested. Price them before running
> anything:
> 1. **Validator + retry.** `generalized_card/generalized_card/length_fidelity.py`
>    is already built, tested and shelved at default `off`, and
>    `--writer-retries` already exists defaulting to **0** — which is why
>    65 of 186 slots on the v109 gate failed their own validator and
>    shipped unretried. A retry note is a different signal from the ask
>    integer and its elasticity is unmeasured. A single-thread seed-8 probe
>    at `--length-fidelity measured --writer-retries 2` costs ~$1.4 and the
>    answer is readable from the **free** assigned-vs-realized audit with
>    no metric at all.
> 2. **Structural cues** — development beats, paragraph layout, surface
>    skeleton. These are what the collinear fit was actually picking up,
>    so they are the live candidates. Price them offline first.
>
> **Before reading a single metric from any paid run**, run the free
> mechanical audit that proves the arm reached the live prompt and did
> what it claimed. For a length arm that is assigned-vs-realized words per
> band; for a prompt-text arm it is grepping the run's own
> `generation_records.json` for the rendered text. Note that each record
> stores its prompt **twice** (`prompt` and `attempts[0].prompt`), so
> `grep -c` double-counts.
>
> **The rules you must not get wrong:**
> - **Distribution diagnostics never select a Writer candidate**
>   (`ORIENTATION.md` §4, G20). Any mechanism that embeds Writer candidate
>   text and compares it to a metric-shaped band to accept/reject/resample
>   is the forbidden category, however it is dressed up.
> - **Every arm defaults to `off`**, so a gate command naming only the new
>   arm silently drops every previously-won arm. Diff your intended
>   command's arm list against the previous paid run's `run_config.json`
>   before spending — it is free, and skipping it cost a $1.18 run's clean
>   reading this session (G39).
> - **Choosing the reported N to improve a p-value is forbidden**
>   (`ORIENTATION.md` §4). N=10 gates are gates, not the paper's scale.
> - **Commit at every version boundary before any paid run**;
>   `source_provenance.py` refuses otherwise, and v97 lost its standalone
>   tree by not doing it.
>
> Credential: `LLM_API_KEY` from `third_party/MiroFish/.env`, confirmed by
> the user across many paid runs — do not re-ask. The user runs every paid
> command themselves: give them the command, the free pre-flight check,
> and predictions written into `VERSION_LOG.md` **before** spending, then
> report against those predictions one by one — including the ones you
> missed.
>
> When you verify, retract or add a rule, edit its row in
> `docs/DECISIONS.md` in the same session, with the script that proves it
> and the date. A retracted row is never deleted.
