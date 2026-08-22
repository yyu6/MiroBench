# DECISIONS — every active rule, and whether it has been verified

**Why this file exists.** On 2026-08-21 five separate decision rules in this
project turned out to be wrong, and all five failed the same way: **a number was
adopted as a rule without ever measuring the number's own null or its own
population.** `|Cliff| ≤ 0.10` steered eight releases from below the noise floor.
An ablation's 28.1% was used to price a mechanism that delivered 8.4%. A
prediction band was set against pooled real instead of the one thread being
measured — twice.

So this file is read **second, right after `ORIENTATION.md`**, before choosing
what to work on. Every row is a rule that is currently in force, and every row
says what it rests on.

**The status column is the point:**

| status | meaning |
|---|---|
| **VERIFIED** | measured against its own null, a held-out population, or a control, with the script named |
| **MEASURED** | measured once on real data, not yet null-tested or replicated |
| **ASSUMED** | inherited, plausible, never tested — **do not steer by this without testing it first** |
| **RETRACTED** | was in force, has been disproved; kept so the mistake is not remade |

**The rule that generates this file:** before a number becomes a decision rule,
run its null — the same test, the same N, with both samples drawn from real.
If that has not been done, the row is ASSUMED and the number is not a rule.

---

## Judging and measurement

| # | rule in force | status | rests on | last checked |
|---|---|---|---|---|
| J1 | Acceptance is the 12 metrics in `run_evaluate.REQUIRED_THREAD_METRICS`, plus human indistinguishability. | VERIFIED (as a definition) | `docs/ORIENTATION.md` §1, user's own framing | 2026-08-21 |
| J2 | **Report with Holm–Bonferroni over the 24 tests, not raw p > 0.05.** The raw rule passes a *perfect* generator 0.63 at N=10 and 0.50 at N=150; Holm passes it 0.98 / 0.98. | VERIFIED | `analysis/acceptance_standard.py`, 440 real camera threads | 2026-08-21 |
| J3 | ~~Steer by `\|Cliff\| ≤ 0.10`.~~ | **RETRACTED** | null `\|Cliff\|` p95 is ≈0.52 per metric at N=10 and ≈0.13 at N=150, so the target sat below the noise floor from v97 to v104 | 2026-08-21 |
| J4 | Steer by `\|Cliff\|` **distance to its own measured floor**, per metric and per N. | VERIFIED | same script; floor table in `ORIENTATION.md` §2 trap 2 | 2026-08-21 |
| J5 | At N=10, measure the generator against **its own template**, paired (Wilcoxon on `generated − reference_metric_template`), not against matched real. | VERIFIED | corr(template, matched real) = −0.281 for `hard_disagree_rate`; `ORIENTATION.md` §2 trap 4 | 2026-08-21 |
| J6 | A gate is one thread: its baseline is **the same thread** in the previous version's artifact, and its target is **that thread's** matched real. | VERIFIED | enforced in `analysis/check_v104_predictions.py`; the error it prevents cost v102 and v104 a wrong prediction band | 2026-08-21 |
| J7 | An **ablation on the artifact is an upper bound, never a price.** Discount it before it justifies a paid run, and record the discounted number as the prediction. | VERIFIED | v104: arms reached 54–85% of their own targets, metric moved 8.4% against an ablation's 28.1% | 2026-08-21 |
| J8 | N=10 p-values cannot discriminate: a true +0.20 shift is not caught 80% of the time. | VERIFIED | `analysis/acceptance_standard.py` | 2026-08-21 |
| J9 | `avg_depth` and `structural_virality` pass structurally — the reply tree is copied from the real thread. Not evidence the generator works. | VERIFIED | `ORIENTATION.md` §2 | earlier |

## Engineering and reproducibility

| # | rule in force | status | rests on | last checked |
|---|---|---|---|---|
| E1 | Every behaviour change is a named arm whose `off` reproduces the previous release exactly, recorded in `run_config.json`. | VERIFIED | v104 proved `off` renders an empty rule on the real prompt path | 2026-08-21 |
| E2 | Commit at every version boundary **before** the paid run. `source_provenance.py` enforces it. | VERIFIED | it blocked the v104 test suite until v104 was committed | 2026-08-21 |
| E3 | A mechanism = one module + one arm + a per-band/per-register **measured** profile + a per-slot SHA-256 draw + a realized-rate audit. Copy `sentence_rhythm.py`. | MEASURED | five releases built this way; v104 shows the shape works and does not guarantee the metric moves | 2026-08-21 |
| E4 | Naming the concrete token gets ≈1.0 compliance; naming the category gets 0.23. | VERIFIED | v102 gate, `discourse_marker` realization 0.231 → 0.923 | 2026-08-20 |
| E5 | Before adding any rule to the Writer, **grep the saved prompts** — it may already be there. | VERIFIED | v102 found the prohibition already reached 504 of 532 prompts and was violated on 9.1% | 2026-08-20 |
| E6 | An analysis harness must reproduce the shipped artifact **before** it prints an edited number. | VERIFIED | the first ablation harness flipped 11.2% of labels and moved the rate 0.1692 → 0.1730 | 2026-08-20 |
| E7 | Deduplicate by `(thread_id, reply_id)` — one Reddit post can sit under two product folders. | VERIFIED | 1.24× matched / 1.32× corpus over-count before the fix. Note: `politeness_diagnosis.py` still does not dedupe | 2026-08-20 |
| E8 | Every claimed number must be reproducible from a committed script, not described in a report. | VERIFIED | `analysis/` — five scripts; seven measurements were promoted into `tone_ceiling.py` on 2026-08-21 after being reported from a scratch directory | 2026-08-21 |
| E9 | A required quality gate can be silently inert. Before trusting a validator's absence-of-findings, or extending its scope, measure whether it ever fires at all. | VERIFIED | `reply_increment_problem` had `require_reply_novelty=True` since before v104 and scored **0 trips on the entire v103 N=10 artifact** — a probe-shape mismatch (short phrase vs. long compound text) suppressed cosine similarity regardless of content. Fixing the probe shape alone (same-shape comparison) surfaced 60 trips on the same artifact. `analysis/reply_novelty_chain_diagnosis.py` | 2026-08-22 |

## What is actually wrong with the generator

Ordered by evidence, not by which number looks worst. Detail in `tasks/todo.md`.

| # | claim | status | rests on |
|---|---|---|---|
| G1 | `self_bertscore_mean_f1` is **the only metric that fails a correct test**: MWU 0.001, KS 0.002, `\|Cliff\|` 0.86 against a floor of 0.50. | VERIFIED | v103 N=10 p-values read under Holm; floor from `acceptance_standard.py` |
| G2 | `self_bertscore_mean_f1` has **no mechanism**. Six hypotheses rejected, including environment drift. | VERIFIED | `tasks/todo.md`, v98 and v101 entries; environment drift (real-side JSONs computed on `transformers==5.7.0`, generated-side on `4.48.0`/`5.10.1`) tested by rescoring one real thread under `4.48.0` — delta 1.6e-8, not a mechanism (2026-08-21) |
| G3 | The excess is **not** parent-child-driven and **not** same-branch-driven; it is a **root-vs-reply role effect** that cuts across both. `same_branch` pairs (siblings/cousins/indirect ancestors) show **no reliable excess** (+0.0056, Wilcoxon p=0.32 thread-paired, n=10) — the `hard_disagree_rate` parent-echo analogy does **not** transfer. `root_root` pairs are also clean (+0.0039, p=0.63). The whole gap concentrates in pairs involving a **reply**: `reply_reply` +0.0274 (p=0.002, unanimous direction across all 10 threads), `root_reply` +0.0130 (p=0.027), and it is a **sign inversion**, not just a bigger shift — real reply-reply pairs are *less* similar than real root-root pairs (0.4905 vs 0.4955) while generated reply-reply pairs are *more* similar than generated root-root pairs (0.5136 vs 0.5089). `parent_child` pairs (1.3% of all pairs) carry the single largest per-pair excess (+0.0256, p=0.0098) but are too rare to matter to the pooled metric. Root comments already match real on this metric too, same as `hard_disagree_rate` — a second metric where the entire defect is a reply-comment phenomenon.<br><br>**The real-side direction is not a 10-thread fluke.** Checked at corpus scale with the cheap `all-mpnet-base-v2` cosine proxy (a *different* metric's model, `semantic_mean_cosine` — used here only to test the direction of a text property, not to re-measure `self_bertscore_mean_f1` itself): across 247 evaluation-excluded real camera threads with enough comments in both buckets, `reply_reply` cosine is below `root_root` cosine in 202/247 threads (82%), mean difference −0.096, Wilcoxon p≈0. Real replies genuinely are more diverse from each other than real root comments are; this is a property of Reddit writing, not noise from which 10 threads got matched.<br><br>**The detector itself was checked, not just trusted.** Read the 8 highest/lowest `bert_f1` pairs on both sides of the actual v103 data (`bertscore_pair_diagnosis.py inspect`). Real high tail: two literal same-author self-repeats within one thread (confirmed against the raw scrape — same `author`, two different parents, e.g. "I'm talking about video." — a genuine data point, not a scrape bug), the previously-documented shared-image-URL artifact (reconfirmed today), and one genuine paraphrase. Real low tail is a single huge thread's short off-topic comment paired against long ones — structural, not content. **Generated high tail is dominated by genuine argument-level paraphrase duplication**: distinct comment pairs in the same generated thread independently restating one specific claim in different words (e.g. seed002 "compactness doesn't matter once it's in a bag," seed008 "you need a real stress test," seed008 "test AF tracking with a moving subject," seed011 "check full-res files, not the compressed clip"), plus one near-identical short pair ("Shipping email?" / "Shipping email? haha"). This is a real, visible criterion-2 defect — logged in `docs/ORIENTATION.md` §1 — but v98's trimming test (rejected hypothesis 2) already showed it is quantitatively too small to be the aggregate driver; the two findings do not conflict. **No evidence the deberta-xlarge-mnli/BERTScore choice itself is a bad detector** — it responds correctly to genuine paraphrase on both sides; its one known noise mode (shared URLs) is real but rare and cannot explain the generated-side excess (generated never contains a link). | VERIFIED | `analysis/bertscore_pair_diagnosis.py pairs\|inspect`, `analysis/root_reply_diversity.py`, v103 N=10 artifact vs its 10 matched real threads plus 247 of 424 evaluation-excluded real threads at corpus scale, fidelity-checked before reading the breakdown (2026-08-21) |
| G4 | `polite_rate` / `impolite_rate` pass at N=10 only for want of power and **will fail at N=150**: gap 0.18 = 1.2 real between-thread sd, and a +0.10 shift is caught 100% of the time at N=150. | VERIFIED | `analysis/acceptance_standard.py`, `analysis/tone_ceiling.py spread` |
| G5 | The tone gap is **realization, not planning**. The plan's between-thread polite mean is 0.310 against a real 0.305; realization is 35%. | VERIFIED | `tone_ceiling.py spread`, `politeness_diagnosis.py realization` |
| G6 | The metric is reconstructed by the **carrier rate** — comments holding one sentence that reads polite alone. 0.062 generated against 0.220 real, and it reconstructs both rates to three decimals on real, generated and v104. | VERIFIED | `polite_sentence_diagnosis.py carriers` |
| G7 | Of ~20 form-only carrier predictors, only three replicate out of sample; the largest under-produced one, the **exclamation, is substantially a classifier artifact** (adding one to a sentence that evaluates nothing raises mean P(polite) 0.023 → 0.174). | VERIFIED | `tone_ceiling.py features`, `tone_ceiling.py exclamation` |
| G8 | Therefore `polite_rate` may **not be closable to N=150 tolerance without gaming**. | MEASURED | everything above; it is an inference from a closed search, not a proof. Treat as the current best reading, not a finding. |
| G9 | The nine other metrics sit **inside** their noise floor at N=10. Stop working on them. | VERIFIED | v103 N=10 Cliff against the floor table |
| G10 | Parent echo is real and un-designed: generated replies echo the parent's content words 1.4–1.6× real, and the counterfactual at the real echo distribution closes 55% of the `hard_disagree_rate` gap. | VERIFIED | `disagreement_diagnosis.py echo` |

| G11 | `--reply-novelty-scope chain` (v105) eliminates the diagnosed chain-restatement defect at the plan level: 0 of 186 plan-quality violations on the gate thread after generation, against 18 replayed on the same thread's v104 (pre-fix) plans. **It did not move `self_bertscore_mean_f1`** — gate result below (G13) — so the mechanism is confirmed correct but is not the metric's dominant driver. Ships `parent_only` (legacy) as the default; no default flip is justified by this result. | VERIFIED (mechanism); FALSIFIED (as the metric's driver) | `generalized_card/VERSION_LOG.md` v105 gate result, seed 8; `analysis/reply_novelty_chain_diagnosis.py` | 2026-08-22 |
| G12 | `--digit-cue-guard on` (v106) is a clean, confirmed win for a criterion-2 tell: bare `0`/`1` fell from 8.01× real's plain-quantifier rate (pre-fix, replayed) to **1.60×** on the actual regenerated gate thread, overall bare-digit rate landing at parity with real (0.0215 vs 0.0199). Unaffected by whether it moves any of the 12 metrics — it was never a fix for one. Ships `off` as the default pending a broader gate. | VERIFIED | `generalized_card/VERSION_LOG.md` v106 gate result, seed 8; `analysis/digit_cue_diagnosis.py` | 2026-08-22 |
| G13 | With G11's plan-level defect eliminated, `self_bertscore_mean_f1`'s gate-thread gap did not close (+0.0183 → +0.0218) and got *worse* specifically in the reply-chain depth bins the fix targeted (depth [2,4): +0.0121→+0.0214; [7,+): +0.0401→+0.0474). Reading the 8 highest-scoring pairs suggested **generic sentence-template reuse across different claims** ("@OP, [verb] X and see/check Y", "[noun]. That's the [X] check"). ~~Measured at scale with opener/closer clause embeddings (`template_reuse_diagnosis.py`), this is **REJECTED as a population-level explanation**: generated's near-duplicate rate (opener 0.0016, closer 0.0005, pooled over 25,931+26,266 pairs) is barely above matched real's (0.0009/0.0003) and indistinguishable from the broader 80-thread real null (0.0012/0.0005). The 8 examples were real but were the extreme tail of a statistic real threads produce at a comparable rate — the same trap as v98's rejected "duplication tail" hypothesis, on a different metric.~~ What *did* generalize (G14): one specific lexical variant of the already-known `abstract_verdict_close` tic. | REJECTED (as a population-level driver); the qualitative examples were real, the generalization was not | `generalized_card/VERSION_LOG.md` v106 gate result and v107 entry; `analysis/template_reuse_diagnosis.py gate\|corpus` | 2026-08-22 |
| G14 | A specific, real, elevated variant of `closing_move.py`'s already-known `abstract_verdict_close` tic (chased since v73, v100's fix): a **"that's the check"/"a solid check"** closing, which the existing measured pattern's word list (`matters/counts/settles/the real/the whole/the part/the only thing/my take/the upshot/bottom line/in the end/at the end of the day`) never named. Measured (`verdict_close_diagnosis.py`): the variant is 13–37× real's rate depending on population, and — more importantly — the *existing* pattern the v100 fix already targets is **still 10–13× over real** even where its suppression cue reaches the Writer (15.2% of cue-present slots on the v106 gate still produce the tic). `--verdict-close-guard on` (v107) widens the Writer-facing cue only — no domain-profile change, no rebuild — to also name the check/test variant. Default `off` (legacy); not yet gated. | VERIFIED (measurement); mechanism offline-only | `generalized_card/VERSION_LOG.md` v107; `analysis/verdict_close_diagnosis.py`; self-test green on all four domains, both arm values (8 runs, $0) | 2026-08-22 |

## Cross-domain

| # | claim | status | rests on |
|---|---|---|---|
| D1 | All four domains switch, plan and write end to end offline; `--prepare-only` on headphone builds the full Planner+Writer command with domain-derived rates. | VERIFIED | run 2026-08-21 |
| D2 | Off camera, roughly a third of the profile's sub-cells fall back to pooled (headphone fits on 90 reference threads against camera's 424). | VERIFIED | profile cell comparison, 2026-08-21 |
| D3 | **No paid run has ever been done on a non-camera domain.** Every tuned rate and every version result is camera-only. | VERIFIED | `RUN_INDEX.md` |
| D4 | The measured layer is domain-portable; the taxonomies, regex families and cue text are hardcoded English/Reddit. | MEASURED | code reading, not tested in another language |

---

## How to use this file

1. Read it before choosing what to work on.
2. If the rule you are about to rely on is **ASSUMED**, test it first — that test
   is the work, not a detour.
3. When you verify, retract or add a rule, edit the row **in the same session**,
   with the script that proves it and the date.
4. A retracted row is never deleted.
