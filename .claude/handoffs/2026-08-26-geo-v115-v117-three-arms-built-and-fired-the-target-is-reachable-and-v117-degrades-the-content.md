# Handoff: GEO synthetic Reddit threads — the target is provably reachable, three count/quota arms are built and fired, and v117 hits its metric while breaking the content (v115, v116, v117)

## TYPE INSTRUCTION FOR THE NEXT AGENT

Read in this order and do not skip: `docs/ORIENTATION.md` §0 → `docs/DECISIONS.md`
**G54–G63, E10–E11** (and note **G2 is SUPERSEDED**) → `tasks/v115-v117-worklog.md`
→ this file's **Key Patterns** and **Potential Gotchas** → `tasks/todo.md`'s
2026-08-26 review section.

**Most of what this session produced is negative.** Six tone hypotheses, digit
runs, hapax flattening, first person, thread structural metrics and the
`evidence_mode` labelling spend are all dead **with numbers**. Re-deriving them
costs days. The "do not re-propose" list in `tasks/todo.md` is authoritative.

Then do §Immediate Next Steps in order. Step 1 is a blocker, not a preference.

## Session Metadata
- Created: 2026-08-26
- Project: `/Users/yaoningyu/Desktop/UIUC/GEO`
- Branch: `generator/v75-writer-realizes-planner-move`
- Commits: 23, `f2434ae`…`42588d3`
- **Paid spend: $3.93** — one 10-thread calibration run
  (`v117_calibration_20260826_v1`, 571 generation records / 559 scored comments,
  65.4 min). Everything else this session was free.
- Tests: **720 pass**. Repo clean. Policy
  `generalized-card-v2-drawn-link-count-v117-20260826`, `PROFILE_SCHEMA_VERSION` 22.

## Handoff Chain
Previous: `.claude/handoffs/2026-08-25-geo-v109-v110-two-live-arms-two-nulls-and-the-causal-instrument-problem.md`
(still valid for the v109/v110 nulls and the causal-instrument finding).

## Current State Summary — the one fact that outranks the arms

**The target is provably reachable at full coverage, and the generator's gap is 10x
the natural noise.** 150 evaluation real camera threads against 150 **disjoint**
real camera threads matched on comment count, coverage 0.996:

| metric | target | donor | bias | MWU | KS | |
|---|---:|---:|---:|---:|---:|---|
| self_bertscore | 0.4923 | 0.4935 | **+0.24%** | 0.810 | 0.443 | PASS |
| self_bleu_4 | 0.0330 | 0.0325 | **−1.61%** | 0.801 | 0.231 | PASS |
| semantic_mean_cosine | 0.2741 | 0.2816 | +2.72% | 0.320 | 0.443 | PASS |
| polite_rate | 0.3216 | 0.3336 | +3.75% | 0.358 | 0.628 | PASS |
| impolite_rate | 0.4079 | 0.3893 | −4.56% | 0.338 | 0.362 | PASS |
| neutral_rate | 0.1611 | 0.1773 | +10.05% | 0.384 | 0.443 | PASS |

An arbitrary real thread passes all six comfortably, so the metric, the matching
and the sample size are sound and the generator's +2.41% is real signal
(`DECISIONS.md` G54). **`analysis/self_similarity/real_vs_real_floor.py` is also a
free per-domain validation harness — run it on any new domain before spending.**

### The three arms

All default `off`, all byte-identical to their predecessor when off, all asserted
by test.

| arm | flag | what it does | fired? |
|---|---|---|---|
| v115 | `--tone-quota {off,inverted,calibrate}` | renders the Planner's tone quota as the assignment whose *realized* mix matches the template, by inverting a measured realization matrix | `calibrate` yes; **`inverted` has NEVER been run** |
| v116 | `--rhythm-count {off,measured}` | draws how many parenthetical asides a slot is cued | yes — written counts `{1:40, 2:8, 3:3}` where v113 was `{1:48}`; compliance 0.585 vs the v113 gate's 0.380 (**crosses pools**, so direction is evidence and magnitude is not) |
| v117 | `--reference-link-count {off,measured}` | draws how many reference URLs a routed slot is offered | yes — **1.68** URLs/carrier vs real 1.67, **61** chars/URL vs the inventory's 61, compliance 0.950, zero markdown garbage / invented URLs / repeats |

## Critical Files
- `generalized_card/generalized_card/tone_realization.py` — v115. Frozen matrix + provenance, grid solve, `POLITE_ASSIGNMENT_CAP = 0.35`.
- `generalized_card/generalized_card/sentence_rhythm.py` — v116. `habit_counts`, `slot_habit_count`, `_MAX_CUED_COUNT = 5`.
- `generalized_card/generalized_card/reference_link.py` — v117. `urls_per_carrier`, `draw_reference_links`, `reference_links_offer`, `MAX_LINKS_PER_SLOT = 4`.
- `generalized_card/analysis/gate_audit.py` — free firing audit for v112/v113/v115/v116/v117; self-validates against the v113 artifact.
- `generalized_card/analysis/tone_carrier/fit_tone_matrix.py` — refits the matrix off any run tag.
- `generalized_card/analysis/tone_carrier/build_calibration_pool.py` — **run immediately before any calibration run and confirm it prints `overlap = 0`; `run_generate` rebuilds that path UNFILTERED if the file is deleted.**
- `generalized_card/analysis/self_similarity/real_vs_real_floor.py` — the noise floor and the per-domain validation harness.
- `generalized_card/analysis/self_similarity/one_voice_*.py` — the unbuilt 51% channel.

## Key Patterns Discovered

**E4 confirmed from the other direction, twice.** The v113 gate's parenthetical
count distribution was literally `{1: 48}` — 48 carrying comments, every one
holding exactly one — because the cue read *"Put one aside in parentheses."* The
link offer read *"Include this exact URL once"* and produced exactly one.
**Naming a concrete number buys ~1.0 compliance on that number.** v116 and v117
are that one insight applied.

**No thread-level aggregate explains `self_bertscore` (G55).** Ranking every cached
thread-level column by its correlation with the metric across 763 real threads and
placing the generator on each **against its own matched real threads**: the
generator sits within ~0.3 sd on essentially everything. The driver is per comment
or per pair, which is why the earlier nine-feature regression reached R²=0.60 and
predicted only 40% of the gap.

**Decompose the metric before hunting for features.** Three sessions guessed a
surface feature and tested it. `self_bertscore` is a mean over pairs, so each
comment's contribution is exactly computable; one script showed real reaches its
*lowest* leverage with 23.7 words of dense content (a name, a spec list, a price, a
link) where the generator needs 46.2 words of conversational fragment.

**A cross-thread correlation is not a comment-level effect.** `polite_rate`
correlates +0.22 with `self_bertscore` across real threads and the generator sits
0.5 sd low, so I warned v115 would cost `self_bertscore`. Per-comment leverage says
generated polite comments carry **lower** leverage (−0.0071), the opposite sign
from real (+0.0042), and v115's mix change moves the metric **−0.00036**. Retracted.

**A self-test can encode a data fact and pass by luck (E10).** v102 asserted a blunt
slot is never told to open on gratitude; it is 0.033 there, and the assertion held
only because twelve fixed probe keys miss a 3% cell about two thirds of the time.
It **blocked the first calibration run outright** — a landmine aimed at exactly the
domain-adaptive work.

## Tasks Finished
- v115, v116, v117 built, tested, committed, VERSION_LOG entries written.
- `gate_audit.py` extended to all five arms and self-validated.
- The real-vs-real floor established on camera for the first time (G54).
- The archive audited: no selfbert self-loop; all historical PASS numbers are truncation artifacts (G56).
- `--prepare-only` turned into a real preflight (E11); the v102 assertion replaced with a distributional one (E10).
- One calibration run gated; the tone matrix refit on a balanced assignment (G60).
- `docs/DECISIONS.md` G54–G63 + E10–E11, `tasks/v115-v117-worklog.md`, `tasks/lessons.md` ×7, `tasks/todo.md` review, `docs/ORIENTATION.md` pointers, 3 memory entries.

## Decisions Made
- **The `evidence_mode` labelling spend is dropped (G62)** — two of three cheap cells are already matched *in the text* with the label at ~zero (link/quote 0.86x real, hearsay 1.08x). The Writer writes those moves regardless of the label.
- **The shipped `REALIZATION_MATRIX` was NOT replaced** with the calibration refit. The polite row transfers (0.3841 → 0.3942) and the neutral row does not (0.4103 → 0.2429), so some rows are generator properties and some are corpus properties; swapping a matrix measured on the evaluation-seed corpus for one measured elsewhere trades a known bias for an unknown one.
- **`MAX_LINKS_PER_SLOT = 4`** costs the target 1.666 → 1.513. Real's tail runs to 9 URLs in one comment; a cue asking a Writer for nine describes nothing a person does.

## Immediate Next Steps

1. **Fix v117's content defect (G61) — this is a blocker.** The arm's numbers are
   right and the output stacks four unrelated links at the end of a 46-word
   comment, puts an Apple support URL inside a comment about a Sony A7, and a Fuji
   X-T5 film-simulation recipe inside one comparing Canon compacts. URLs are drawn
   by hash from an 802-entry inventory with no relation to the comment's content.
   Fix is measured: of 249 real comments carrying 2+ non-media URLs, **64.3% have
   ALL their URLs on one host** and the first URL sits a median **23%** into the
   comment. **v117 must not enter a paper run before this.**
2. **Decide v115's cap against the reported-metric set, not L2 (G60).** Cap 0.59
   lands `polite_rate` +0.2% and `impolite_rate` +1.3% while costing `neutral_rate`
   **−19.7%**; cap 0.35 leaves all three mid-range at 35% closure. **A decision,
   not a measurement.**
3. **Measure the persona layer against `self_bertscore` (G57).** The only single
   channel above the 42% bar — headroom **+0.0060 = 51%** of the gap — and
   `persona_bridge` / `speaker_roster` / `actor_conditioning` /
   `--speaker-identity matched` have never been measured against it. **Free.**
4. **Evaluate the calibration run** for the first full-coverage reading with all
   three arms on:
   `python3 generalized_card/scripts/run_evaluate.py --tag v117_calibration_20260826_v1`.
   Its **tone numbers are meaningless** — the quota was deliberately flat.
   `self_bertscore` and `self_bleu_4` are readable, with a ~3% confound from the
   tone mix that was measured separately. p-values are **not** comparable to a
   paper run (different threads).
5. Only then N=50 / N=150.

## Blockers/Open Questions
- **`self_bertscore` has no path to 42% with what exists.** v117 ~17% of the gap, v115 ~3%, **assuming additivity** (the surface channels measured sub-additive at ~0.86). Gap 0.0119 → ~0.0095 against the **0.0069** Holm needs at N=150. Both terms are J7 upper bounds.
- **The one channel large enough is unbuilt (G57)** — authorial voice separation, generated at 0.55 of real's.
- Six tone realization hypotheses are dead (G53, G58, G59). The decisive one: generated already carries real's top-45 polite-discriminative tokens at **1.14x** real prevalence.
- The 9-point `self_bertscore` swing between `repro_v37` (−0.88% at 0.629 coverage) and `sample_planner_gpt4omini_writer_v37` (+8.47% at 0.603) is unexplained. Those runs differ in more than one thing, so it is an observation, not a channel.
- The user has said **domain adaptation is in scope**. E10's landmine was found because of it; expect more data-dependent assertions to surface on a new corpus.

## Potential Gotchas
- **`--prepare-only` was a config printer until this session (E11).** Any earlier handoff saying a command was "verified with prepare-only" did not include the self-test.
- **Commit from the repo root.** `git add -A generalized_card/` run from *inside* `generalized_card/` resolves to `generalized_card/generalized_card/` and silently misses `scripts/`.
- **Duplicate test-helper names are taken silently.** The v117 tests first added `_link_inventory` and `_link_task` to a class that already had both from the v113 tests. Reversed, the v113 tests would have broken with no error naming the cause.
- **Never quote a historical pass count without checking coverage** in `matched_generated_thread_scores.csv` against `matched_real_thread_scores.csv`. Across all **284** evaluated run directories the coverage>=0.90 band has a median `self_bertscore` bias of **+4.28%** and exactly **one** run under 1% — the real-comment bootstrap, which is not a generator. Every archived run that shows this metric passing is at coverage 0.546–0.629 (G56).
- **Two counts exist for a run**: generation records and scored comments. The calibration run is 571 vs 559; the matrix is fitted on 559. Say which one a figure uses.
- `str(getattr(task, "real_sample_id", "") or "")` turns `real_sample_id=0` into `""`, so slot 0 and "no real_sample_id" collide in the link draw key. Pinned since v113, deliberately left alone.
- A run tag cannot be resumed across a core-contract change (`generator_core_provenance` is resume-immutable). Use a new tag after any pin change.

## Tools/Services Used
- gpt-5.4-mini via `https://api.openai.com/v1`, key from `LLM_API_KEY` (`third_party/MiroFish/.env`, auto-loaded by `run_generate.py`). **The user runs every paid command themselves.**
- Local scorers: `Intel/polite-guard`, `microsoft/deberta-xlarge-mnli` (BERTScore, no baseline rescaling, no idf), `sentence-transformers/all-mpnet-base-v2`.
