# Handoff: GEO synthetic Reddit threads — three arms built and fired, the target proved reachable, and v117 hits its metric while breaking the content (v115, v116, v117)

## Session Metadata
- Created: 2026-08-26
- Project: /Users/yaoningyu/Desktop/UIUC/GEO
- Branch: `generator/v75-writer-realizes-planner-move`
- Session shape: v113/v112 N=10 gate read-out → tone re-derived from scratch → v115 built → v116 built → v117 built → the archive audited → the real-vs-real floor established → v117 calibration run gated. **$3.93 paid spend** (one 10-thread calibration run, 559 slots, 65 min).
- **Net outcome: three arms built, tested, committed, and all three verified firing on a paid run. The single most important new fact is not an arm — it is that the target is provably reachable and the generator's gap is 10x the natural noise.**

### Recent Commits (newest first, this session)
  - `f2c0cd1` calibration(v117): the cap question is answered, and neutral is the new constraint
  - (v117 blocker) analysis: v117 fires exactly and degrades the content
  - `ae21991` fix(run_generate): --prepare-only is now a real preflight, not a config printer
  - fix(self-test): the blunt-gratitude assertion passed by luck and blocks every new corpus
  - `726f87b` analysis: evidence_mode is a labelling gap, not a text gap — the labelling spend is dropped
  - `ae39457` docs(findings): record the floor, the leverage decomposition, and the archive trap
  - `7a1fd9e` audit: v115 does not cost self_bertscore — my warning was an ecological fallacy
  - `dd1c1b7` v117: draw how many reference links, and resize the claim that justified it
  - `a5a0a01` analysis: the target is reachable, and no thread-level aggregate explains the gap
  - `3d5dff0` analysis(self_bertscore): the generator has 55% of real's authorial voice separation
  - `f52c968` analysis(tone): a zero-overlap calibration pool, so the cap is not raised in-sample
  - `d66acf4` v115: add --tone-quota calibrate, the measurement value that lets the polite cap move
  - `0167f85` analysis(self_bertscore): the link arm tops out near 39% against the 42% Holm needs
  - `359b77d` docs(v115): the arm, the Lucas critique against it, and the in-sample disclosure
  - `960bd5b` v115: render the tone quota the Planner needs, not the one the metric reports
  - `f2434ae` analysis(tone): polite is a per-sentence lottery, and the lever is the assignment side
  - v116 and its VERSION_LOG entry sit between `f2434ae` and `dd1c1b7`.

## Handoff Chain
Previous: `.claude/handoffs/2026-08-25-geo-v109-v110-two-live-arms-two-nulls-and-the-causal-instrument-problem.md`

## Current State Summary

**Read `generalized_card/analysis/self_similarity/FINDINGS.md` §9–§14 and
`generalized_card/analysis/tone_carrier/FINDINGS.md` before doing anything.** Most
of what this session established is negative, and re-deriving it costs days.

Three arms exist, default `off`, each byte-identical to its predecessor when off,
each asserted by test. 720 tests pass. The repo is clean and committed.

| arm | flag | what it does | fired? |
|---|---|---|---|
| v115 | `--tone-quota {off,inverted,calibrate}` | renders the Planner's tone quota as the assignment whose *realized* mix matches the template, by inverting a measured realization matrix | `calibrate` fired (flat 137/137/140/145); `inverted` **never run** |
| v116 | `--rhythm-count {off,measured}` | draws HOW MANY parenthetical asides per slot from the band's measured distribution | fired: counts written `{1:40, 2:8, 3:3}`, compliance 0.380 → **0.585** |
| v117 | `--reference-link-count {off,measured}` | draws HOW MANY reference URLs a routed slot is offered | fired: **1.68** URLs/carrier against real's 1.67, 61 chars/URL against 61, compliance 0.950, zero markdown garbage / invented URLs / repeats |

Policy version `generalized-card-v2-drawn-link-count-v117-20260826`.
`PROFILE_SCHEMA_VERSION` 21 → 22.

## The three facts that matter most

**1. The target is reachable, and the gap is real signal.**
`analysis/self_similarity/real_vs_real_floor.py` — 150 evaluation real camera
threads against 150 **disjoint** real camera threads matched on comment count, at
coverage 0.996: **all six reported metrics PASS comfortably**, `self_bertscore`
bias **+0.24%**, `self_bleu_4` **−1.61%**. So the metric, the matching and the
sample size are sound, and the generator's +2.41% is **ten times** the natural
real-to-real spread. This is also the cheapest validation harness in the project:
it reads cached `thread_scores.csv` only and can be run on any domain before a
token is spent. **Run it first on any new domain.**

**2. `self_bertscore` has never been observed passing at full coverage, and the
archive reads as though it has.** Across every `*_controller_history.json` the
metric appears as a self-loop TARGET in **zero** observations and as a *protected*
metric in the 32 that end PASS — **there is no self_bertscore self-loop.** The four
run families carrying those PASS observations sit at coverage **0.546, 0.577, 0.603
and 0.629**, the truncation regime `VERSION_LOG.md` opens by warning about; those
four are what was checked individually. Sweeping all **284** evaluated run
directories, the coverage>=0.90 band has a median bias of **+4.28%** and exactly
**one** run under 1% — the real-comment bootstrap, which is not a generator.

**3. v117 hits its metric target and makes the content visibly worse.** The output
contains four unrelated links stacked at the end of a 46-word comment, an Apple
support URL inside a comment about a Sony A7, and a Fuji X-T5 film-simulation
recipe inside one comparing Canon compacts. URLs are drawn from an 802-entry
inventory by hash **with no relation to the comment's content**. Tolerable at one
link; an eye-visible tell at four. **This is a blocker on v117, not a success.**

## Critical Files
- `generalized_card/generalized_card/tone_realization.py` — v115. Frozen matrix + provenance + the grid solve. `POLITE_ASSIGNMENT_CAP = 0.35`.
- `generalized_card/generalized_card/sentence_rhythm.py` — v116. `habit_counts` in the band row, `slot_habit_count`, `_MAX_CUED_COUNT = 5`.
- `generalized_card/generalized_card/reference_link.py` — v117. `urls_per_carrier` in the inventory, `draw_reference_links`, `reference_links_offer`, `MAX_LINKS_PER_SLOT = 4`.
- `generalized_card/analysis/gate_audit.py` — free post-run firing audit for v112/v113/v115/v116/v117. Self-validates against the v113 artifact.
- `generalized_card/analysis/tone_carrier/fit_tone_matrix.py` — refits the matrix off any run tag.
- `generalized_card/analysis/tone_carrier/build_calibration_pool.py` — writes a zero-overlap pool. **Run it immediately before any calibration run and check it prints `overlap = 0`; `run_generate` rebuilds that path UNFILTERED if the file is deleted.**
- `generalized_card/analysis/self_similarity/real_vs_real_floor.py` — the noise floor / validation harness.

## Key Patterns Discovered

**E4 confirmed from the other direction, twice.** The v113 gate's parenthetical
count distribution was literally `{1: 48}` — every carrying comment holding
exactly one, no exceptions — because the cue read *"Put one aside in
parentheses."* The link offer read *"Include this exact URL once"* and produced
exactly one. **Naming a concrete number buys ~1.0 compliance on the number.** Both
v116 and v117 are that one insight applied.

**Thread-level aggregates cannot explain `self_bertscore`.** Ranking every cached
thread-level column by its correlation with the metric across 763 real threads and
placing the generator on each in units of real's spread: the generator sits within
**~0.3 sd of its matched real threads on essentially everything**. That is why the
nine-feature regression in FINDINGS §3 reached R²=0.60 and predicted only 40% of
the gap. The driver is per comment or per pair.

**A cross-thread correlation is not a comment-level effect.** I warned that v115
would cost `self_bertscore` because `polite_rate` correlates +0.22 with it across
real threads and the generator sits 0.5 sd low. Decomposing the metric to
per-comment leverage: generated polite comments carry **lower** leverage
(−0.0071), the opposite sign from real (+0.0042), and v115's mix change moves
`self_bertscore` **−0.00036**. Ecological fallacy; retracted.

**Self-tests can encode data facts and pass by luck.** The v102 assertion "a blunt
slot is never told to open on gratitude" rested on a docstring claim that gratitude
is "absent from the blunt row". It is 0.033 there. The assertion held only because
its twelve fixed probe keys miss a 3% cell about two thirds of the time — at 50
probes it fails on the shipped profile too. It blocked the calibration run outright.
**Any assertion about drawn values must be distributional.**

## Tasks Finished
- v115, v116, v117 built, tested, committed, VERSION_LOG entries written.
- `gate_audit.py` extended to all five arms and self-validated against the v113 artifact.
- The real-vs-real floor established on camera for the first time.
- The archive audited: no selfbert self-loop, all historical PASS numbers are truncation artifacts.
- `--prepare-only` turned into a real preflight (it used to return *before* the self-test).
- The v102 self-test assertion replaced with a distributional one, audited against four cases including a synthetic leak.
- One calibration run gated; the tone matrix refit on a balanced assignment.
- Three memory entries written to `~/.claude/projects/-Users-yaoningyu-Desktop-UIUC-GEO/memory/`.

## Decisions Made
- **The evidence_mode labelling spend is dropped.** It was the largest per-pair collision channel (+0.0228, topic controlled), but two of the three cheap cells turn out already matched *in the text* while the label sits at zero: link/quote 0.86× real, hearsay 1.08× real. The Writer produces those moves regardless of the label, so raising the labels would push the surface *above* real. Recorded in FINDINGS §13 as a decision not taken, with its reason, so it is not re-proposed.
- **The shipped `REALIZATION_MATRIX` was deliberately NOT replaced** with the calibration refit. The polite row transfers (0.3841 → 0.3942) and the neutral row does not (0.4103 → 0.2429), which says some rows are generator properties and some are corpus properties. Replacing a matrix measured on the evaluation-seed corpus with one measured on a different corpus trades a known bias for an unknown one.
- **`MAX_LINKS_PER_SLOT = 4`** costs the target 1.666 → 1.513. Real's tail runs to 9 URLs in one comment; a cue asking a Writer for nine describes nothing a person does.

## Immediate Next Steps
1. **Fix v117's content defect before anything else.** Measured on 249 real comments carrying 2+ non-media URLs: **64.3% have ALL their URLs on one host**, and the first URL sits a median **23%** into the comment rather than trailing. A v118 should draw a multi-link slot's URLs from one host at that rate and place the first link early. Until then v117 should not ship.
2. **Decide the v115 cap against the reported-metric set, not L2.** On the refit matrix, cap 0.59 lands `polite_rate` at +0.2% and `impolite_rate` at +1.3% while costing `neutral_rate` −19.7%; cap 0.35 leaves all three mid-range and closes 35%. This is a decision, not a measurement.
3. **Evaluate the calibration run** for the first full-coverage `self_bertscore` reading with all three arms on: `python3 generalized_card/scripts/run_evaluate.py --tag v117_calibration_20260826_v1`. Its **tone numbers are meaningless** (the quota was deliberately flat); `self_bertscore` and `self_bleu_4` are readable, with a ~3% confound from the tone mix that was measured separately.
4. Only then consider N=50/N=150.

## Blockers/Open Questions
- **`self_bertscore` still has no path to 42%.** Honest arithmetic on what is built: v117 ~17% of the gap, v115 ~3%, **assuming they add** — the surface channels were measured sub-additive at ~0.86. Gap 0.0119 → ~0.0095 against the 0.0069 Holm needs at N=150, and both terms are J7 upper bounds.
- **The one channel measured large enough is unbuilt.** Authorial voice separation: generated is at **0.55** of real's (stratum-weighted +0.0076 against +0.0137; in the decisive different-branch cell +0.0061 against +0.0141, 43%). Headroom **+0.0060 = 51% of the gap** — the only single channel above the 42% bar. `persona_bridge`, `speaker_roster`, `actor_conditioning`, `--speaker-identity matched` all exist and were on; **none has ever been measured against `self_bertscore`.** That is the next measurement and it is free.
- Six realization-side tone hypotheses are dead: more register cues, the omitted conjunction, hedging, length repair, the bare-assertion frame, and the polite lexicon (generated already carries real's top-45 polite tokens at **1.14×** real prevalence).
- The 9-point `self_bertscore` swing between `repro_v37` (−0.88% at 0.629 coverage) and `sample_planner_gpt4omini_writer_v37` (+8.47% at 0.603) is unexplained. Those runs differ in more than one thing, so it is an observation, not a channel.

## Potential Gotchas
- **`--prepare-only` was a config printer** until this session. If an older handoff says a command was "verified with prepare-only", that verification did not include the self-test.
- **Commit from the repo root.** `git add -A generalized_card/` run from *inside* `generalized_card/` resolves to `generalized_card/generalized_card/` and silently misses `scripts/`. It cost a debug cycle here.
- **Duplicate test-helper names are taken silently.** The v117 tests first added `_link_inventory` and `_link_task` to a class that already had both from the v113 tests; Python takes the last definition. Reversed, the v113 tests would have broken without an error.
- Never quote a historical pass count without checking coverage in `matched_generated_thread_scores.csv` against `matched_real_thread_scores.csv`.
- `str(getattr(task, "real_sample_id", "") or "")` turns `real_sample_id=0` into `""`, so slot 0 and "no real_sample_id" collide in the link draw key. Pinned since v113 and deliberately left alone.

## Tools/Services Used
- gpt-5.4-mini via `https://api.openai.com/v1`, key from `LLM_API_KEY` (`third_party/MiroFish/.env`, auto-loaded). **The user runs every paid command themselves.**
- Local scorers: `Intel/polite-guard`, `microsoft/deberta-xlarge-mnli` (BERTScore, no baseline rescaling, no idf), `sentence-transformers/all-mpnet-base-v2`.
