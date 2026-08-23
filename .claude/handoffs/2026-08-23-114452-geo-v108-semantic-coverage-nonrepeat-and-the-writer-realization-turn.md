# Handoff: GEO synthetic Reddit threads — self_bertscore_mean_f1 closed as a research-design limit, then reopened by a genuinely different mechanism (v108), which is the best single-thread result yet

## Session Metadata
- Created: 2026-08-23 11:44:52
- Project: /Users/yaoningyu/Desktop/UIUC/GEO
- Branch: `generator/v75-writer-realizes-planner-move`
- Session duration: one long session; G3 → v105 → v106 → v107 → N=10 null result → G17–G22 → v108 (built, broke, fixed, gated). ~$9.15 total paid spend across five gates.

### Recent Commits (for context, newest first, this session's work)
  - `caa0d99` docs: write the isolated N=10 predictions for --semantic-coverage-nonrepeat before spending
  - `0e1d6ce` gate(v108, v2): arm fired 186/186, best single-thread self_bertscore result this session
  - `95300e1` fix(v108): the arm fired 0/186 times on the paid gate -- fixed the actually-live prompt path
  - `9aa53fc` v108: semantic-coverage non-repeat instruction -- influence the Writer's input, not its output
  - `5a84feb` decision(G22): close self_bertscore_mean_f1 as a research-design limit
  - `43b5210` analysis(G21): is this a Planner problem? Mostly no -- a second realization-gap metric
  - `f04a688` analysis: independent replication in a second embedding model, then stop before violating a non-negotiable rule
  - `278fbd1` analysis: G3's real-side direction generalizes to all four domains; audit self_bertscore for a code bug; identify the existing dropout/jitter mechanism's actual scope
  - `64cf726` gate(N=10): digit-cue + verdict-close null result on self_bertscore_mean_f1
  - `f69c55e` docs: write the N=10 combined-gate predictions before the user's paid run
  - `c973118` gate(v107): isolated verdict-close guard, self_bertscore moves favorably for the first time
  - `2fa3f82` docs(v107): write the isolated large-thread gate predictions before spending
  - `e9c3e8c` v107: verdict-close check-variant guard, and the template-reuse rejection that found it
  - `5fc3a96` gate(v105+v106): mechanisms confirmed working, self_bertscore_mean_f1 did not follow
  - `28acd53` v105/v106: write the large-thread gate predictions before spending
  - `b2718ac` v106: digit-cue quantifier guard (criterion-2 tell)
  - `ea1760f` v105: chain-scoped reply novelty check for self_bertscore_mean_f1 (G3 fix)
  - `21351d5` analysis(G3): verify the BERTScore detector and the root/reply direction at scale
  - `d089fa2` analysis(G3): decompose self_bertscore_mean_f1's excess -- root/reply role, not parent-echo

## Handoff Chain

- **Continues from**: [2026-08-21-231454-geo-v104-evaluative-register-and-the-decision-register.md](./2026-08-21-231454-geo-v104-evaluative-register-and-the-decision-register.md)
  - Previous title: GEO synthetic Reddit threads — v104 shipped and under-delivered; the acceptance standard itself was wrong
  - That handoff's "Immediate Next Steps" item 1 (turn `DECISIONS.md` G3 from ASSUMED to VERIFIED) is exactly what this session opened with, and it is now done.
- **Supersedes**: None of that handoff's content is wrong; it is superseded by volume — `docs/DECISIONS.md` now runs to G23 and this file's own "Current State Summary" is the fastest way in.
- **Authoritative spec**: `docs/ORIENTATION.md`, then `docs/DECISIONS.md`. Both maintained in place. This handoff points at them rather than duplicating them.

## Current State Summary

The project generates synthetic Reddit threads meant to be statistically
indistinguishable from real ones across 12 thread-level metrics
(criterion 1) and to a human reader (criterion 2), first-pass Planner→Writer
only. Going into this session, `self_bertscore_mean_f1` was the sole
metric failing a correct-work-passing standard (`DECISIONS.md` G1), with
`DECISIONS.md` G3 marked ASSUMED (a pairwise decomposition had never been
run).

**This session ran that decomposition, then spent the rest of its length
on `self_bertscore_mean_f1` specifically**, through a full arc: diagnose →
build a targeted fix → gate it → find the fix didn't move the metric at
scale → diagnose again → discover a fix category is forbidden by this
project's own validity rules → discover the *other* legitimate category
also doesn't reach the real driver → close the metric as a research-design
limit (`G22`, joining `polite_rate`/`impolite_rate`) → find a third,
different, legitimate category the same day (v108) → build it, break it,
catch the break, fix it → gate it → get the best single-thread result of
the whole session (+0.0183 → +0.0139, four of five depth bins improved
together). **The metric is reopened, not closed**, on the strength of one
promising but unreplicated single-thread result. Predictions for an
isolated N=10 gate are already written (`VERSION_LOG.md`); the command is
below and has not been run.

Two other arms (`--digit-cue-guard`, `--verdict-close-guard`) shipped this
session as independently real criterion-2 improvements, `off` by default,
regardless of what happens to `self_bertscore_mean_f1`.

## Architecture Overview

Unchanged: `generalized_card/` is a domain-configured Planner→Writer
implementation of CARD. The Planner emits a per-slot plan (structured
fields — `semantic_move`, `decision_boundary`, `detail_focus`, etc.); the
Writer realizes one comment per slot from a prompt built out of that plan
plus a "thread memory" ledger of what has already been said.

**What this session mapped in detail: the Writer prompt has two, and only
two, ledger-builder functions** (`generalized_card/generalized_card/prompts.py`):
`_thread_memory` (for `--writer-prompt full`) and `_focused_thread_ledger`
(for the default `--writer-prompt focused`, active since v82 — every real
run in this project's history uses this one). Both build a "semantic
contributions already covered in this thread" block from
`semantic_coverage_entries` (`semantic_realization.py`), among other
things. **This session's central, hard-won finding: a fix that only
touches one of these two functions never reaches a default run** — see
"Key Patterns Discovered" and v108's own two commits below.

There is also a **non-negotiable, previously-underappreciated rule**
(`docs/ORIENTATION.md` §4): "Distribution diagnostics never select a
Writer candidate. Only output that cannot be persisted gets bounded
recovery." A structurally near-identical mechanism to what was almost
built this session already exists (`generation_diversity.py`'s
`semantic_thread_diagnostics`/`semantic_distribution_problem`) and is
deliberately kept diagnostic-only for exactly this reason.

## Critical Files

| File | Why it matters | Read when |
|---|---|---|
| `docs/ORIENTATION.md` | the spec; §4 "What may never happen" is the rule this session almost broke | always, first |
| `docs/DECISIONS.md` | every rule in force, now G1–G23 | always, second |
| `tasks/todo.md` | task list; the `self_bertscore_mean_f1` item now documents the full G16→G23 arc | before choosing work |
| `generalized_card/VERSION_LOG.md` | v105–v108 entries, each with predictions written before spending, then the gate result | before any paid run; the v108 N=10 predictions are already written here |
| `tasks/lessons.md` | **five new entries this session**, the last two are the ones most likely to bite again — see below | before building or gating anything |
| `generalized_card/generalized_card/prompts.py` | `_thread_memory`/`_focused_thread_ledger`, both now carry the v108 instruction; `_writer_prompt_mode`'s default is `focused` | before touching any Writer-prompt content |
| `generalized_card/analysis/plan_text_realization_gap_diagnosis.py` | **NEW.** answers "is this a Planner problem?" (G21) — mostly no, r=+0.48 | before proposing any Planner-side fix for this metric |
| `generalized_card/analysis/cross_domain_reply_diversity.py` | **NEW.** G3's real-side direction replicated in all four domains (G17) | before assuming a finding is camera-specific |
| `generalized_card/analysis/reply_diversity_guard_diagnosis.py` | **NEW.** second-model (embedding) replication of G3 (G19); the guard it was calibrating for was never built (G20) | context on why v108 exists |
| `generalized_card/tests/test_generalized_card.py`'s `FocusedWriterPromptTest` | has the real end-to-end dispatch test pattern (`configure_generator_backend` + `build_writer_prompt`) that would have caught the v108 bug immediately — use it for any future prompt-text fix | before writing a test for prompt-text content |
| `generalized_card/tests/test_sentence_rhythm.py`'s `WriterPromptTest` | the original "reaches both paths" convention this project already had and v108 didn't follow at first | same as above |

## Key Patterns Discovered

- **A single-thread gate's win does not predict N=10 pool behavior, even
  on the exact same thread.** `--verdict-close-guard`'s check-variant went
  from 3/106 to 0/106 on seed 8 alone (G15), then was statistically
  unchanged (2/308 → 2/298) at N=10 (G16). This is now measured, not
  inferred, and applies exactly as much to v108's seed-8 result — treat it
  as promising, not proven.
- **Two unrelated arms moving the same untargeted metric the same
  direction, on the same thread, across independent gates, is evidence of
  thread-level regeneration noise** — not of either arm. Seen repeatedly
  for `hard_disagree_rate`/`mean_story_probability`/`emotion_entropy`
  across the v106, v107, and v108 gates.
- **Read the actual code insertion point before designing a new mechanism
  — not just a keyword search.** Grepping the repair-loop entry point
  before wiring the planned Writer-output-similarity guard surfaced that
  it was the exact category `docs/ORIENTATION.md` §4 forbids, and that a
  near-identical mechanism already exists, deliberately kept soft (G20).
- **A rejected hypothesis's own qualitative examples can still be the
  answer — read them again for what they specifically are.** The
  "generic sentence-template reuse" hypothesis was rejected at scale
  (G13), but its own 8 examples, read again individually, led straight to
  v107's real finding.
- **Verify a prompt-text fix reaches the code path a default run actually
  calls, not just that the text renders correctly where you put it.**
  v108's first paid gate ($1.19) tested nothing — the fix touched
  `_thread_memory` (the `full`-mode path); every real run, including that
  one's own command, uses the `focused`-mode default, which calls a
  *different* function. Caught only by grepping the run's own
  `generation_records.json` for the instruction string after the fact.
  **This project already had the right test pattern for this**
  (`test_sentence_rhythm.WriterPromptTest`'s "reaches both paths"
  convention, with a comment recording that v74 made this exact mistake
  once already) — it just wasn't applied here. Full lesson and the fix in
  `tasks/lessons.md` and commit `95300e1`.

## Tasks Finished

- [x] Turned `DECISIONS.md` G3 from ASSUMED to VERIFIED: the excess is a
      root-vs-reply role effect, not parent–child or uniform (`d089fa2`, `21351d5`).
- [x] Built and gated v105 (chain-scoped reply novelty, `ea1760f`) and
      v106 (digit-cue guard, `b2718ac`) together on seed 8 (`5fc3a96`,
      $1.2081): both mechanisms confirmed working on their own terms;
      `self_bertscore_mean_f1` did not follow.
- [x] Built and gated v107 (verdict-close check-variant guard, `e9c3e8c`)
      in isolation on seed 8 (`c973118`, $1.1637): the first favorable
      single-thread result, later shown not to replicate at N=10.
- [x] Ran a real N=10 pool with digit-cue + verdict-close on
      (`64cf726`, $4.3909, one crashed-and-resumed attempt): null result on
      `self_bertscore_mean_f1` (Cliff unchanged, 0.86 → 0.86).
- [x] Verified G3's real-side direction generalizes to all four domains
      (G17) and to a second embedding model (G19); audited the metric's
      own scoring code for a bug (none found); identified the existing
      context-dropout/jitter mechanism's real, narrower scope (G18) (`278fbd1`, `f04a688`).
- [x] Designed a Writer-output-similarity guard, then stopped before
      building it on discovering it would violate `ORIENTATION.md` §4 (G20)
      (`f04a688`).
- [x] Measured whether this is a Planner problem (G21): mostly not
      (r=+0.48); decided not to widen the Planner-side scope further (G22)
      (`43b5210`, `5a84feb`).
- [x] Found a different, legitimate category the same day: the Writer
      prompt's "already covered" ledger has no "do not repeat" instruction,
      unlike its siblings. Built `--semantic-coverage-nonrepeat` (v108,
      `9aa53fc`).
- [x] Gated it, found it fired 0/186 times (wrong function fixed), found
      the bug, fixed the actually-live path, added the regression test
      that would have caught it (`95300e1`).
- [x] Re-gated: fired 186/186, `self_bertscore_mean_f1` improved on seed 8
      by the largest margin this session (`0e1d6ce`).
- [x] Wrote isolated N=10 predictions for v108, not yet run (`caa0d99`).

## Files Modified

See `git log 29dba04..HEAD` (20 commits). New files this session: five
`generalized_card/analysis/*.py` scripts (see Critical Files), plus edits
to `generalized_card/generalized_card/planning_quality.py` (v105),
`sentence_rhythm.py` (v106), `closing_move.py` (v107), and `prompts.py`
(v108, the only file touched twice — once wrong, once right).

## Decisions Made

| Decision | Rationale | Evidence |
|---|---|---|
| `self_bertscore_mean_f1` closed as a research-design limit under the *checked* mechanism categories (G22) | three targeted mechanisms (v104, v105, v107) failed at real statistical power; the one category that could reach the Writer-realization majority share is forbidden (G20); the remaining Planner category has a low, evidenced ceiling (G21) | `VERSION_LOG.md` v104/v105/v106/v107 gate results; G16, G20, G21 |
| ...then reopened the same day by v108, a different category | changes the Writer's *prompt*, the same "arm" category as every cue-text fix this session, not a Writer-output-selection check | G23; not a contradiction of G22, which was scoped to the two checked categories |
| `--digit-cue-guard`/`--verdict-close-guard` ship regardless of `self_bertscore_mean_f1`'s fate | both are independently real, measured criterion-2 improvements (G12, G14/G15) | `digit_cue_diagnosis.py`, `verdict_close_diagnosis.py` |
| Do not widen `reply_increment_problem`'s scope from ancestor-chain to whole-thread | the wider population has the same healthy plan-similarity curve as the narrower one (G21); the one real test of "widen this category" (v105's own N=10 gate) made the targeted bins worse | G21, G22 |
| v108's next paid step is an isolated N=10 pool, not stacked with v106/v107 | keeps attribution to this one arm clean; predictions already written | `VERSION_LOG.md` |

## Immediate Next Steps

1. **Run the isolated N=10 gate for v108** (predictions already in
   `VERSION_LOG.md`, command below). Before spending, run
   `python3 -m pytest generalized_card/tests/test_generalized_card.py -k semantic_coverage_nonrepeat_reaches_both`
   as a final $0 sanity check. After the run, **before reading any metric**,
   grep the artifact's `generation_records.json` for
   `"Do not restate one of these already-covered points"` and confirm it
   appears in (close to) all comments — this exact check is what caught
   the v1 bug and is now standard practice for this arm.
2. **Read the N=10 result honestly against the two named risks already
   written into the prediction**: does the seed-8 win replicate, or does
   it go the way `--verdict-close-guard`'s did (G16)? Either answer is a
   real finding — write it into `VERSION_LOG.md` and `DECISIONS.md` (next
   row is G24) the same way every prior gate this session was written up.
3. **If it replicates at N=10**: this is the first mechanism with a
   credible shot at actually moving `self_bertscore_mean_f1`, and the
   decision to reconsider a default flip becomes live for the first time.
   If it doesn't: `self_bertscore_mean_f1` has now had every category this
   project's own rules permit tried and falsified, and the honest
   conclusion is to close it for good, more firmly than G22 already did.
4. Domain generalization (D1–D4) and N=150 (the actual final scale, §1)
   remain untouched this entire session and are the two largest genuinely
   open items in `docs/ORIENTATION.md`'s own "Next step" section.

## Blockers/Open Questions

- **None on credentials.** `LLM_API_KEY` from `third_party/MiroFish/.env`
  was confirmed by the user earlier and used for every paid run this
  session (~$9.15 total: $1.2081 + $1.1637 + $4.3909 + $1.1867 + $1.2036).
  Re-confirm only if the user raises it; do not re-ask by default.
- **`abstract_verdict_close`'s existing (v100-era) suppression is still
  10–13× real even where its cue reaches the Writer** (G14) — flagged as
  an open coverage-vs-compliance question, not diagnosed further this
  session; would need a larger sample than one gate thread to separate the
  two.
- **The `enum_or_fact` (genuine numbered/fractional/price digit use)
  suppression flag from the N=10 digit-cue+verdict-close gate is
  unconfirmed** (G16): fell to 0/532, but n is too small (0 vs an expected
  ~1) to be sure this is a real effect of the guard rather than noise.
- **N=150 reporting standard is still the user's call** (unchanged from
  the previous handoff).

## Deferred Items

- The opener-side "@OP, ..." tic noted while diagnosing v107 was never
  chased (a separate mechanism, `opener_profile.py`).
- Everything in the previous handoff's "Deferred Items" that this session
  did not touch: the carrier prevalence gap, v104's opener root/reply
  conditional, generated root share, several eye-visible tells.
- Domain generalization: **still zero paid runs on any non-camera
  domain**, unchanged from every prior handoff (D3).

## Important Context

- Everything from the previous handoff's "Important Context" still holds
  (the user's own stated goal in `ORIENTATION.md` §1; necessary /
  matches-real / cross-domain as standing constraints; predictions before
  metrics before comments before your own errors, when reporting a gate).
- **New this session, stated explicitly and repeatedly by the user**: full
  technical/design autonomy is delegated — "你是全权负责这个project" (you are
  fully responsible for this project). Route only genuinely non-technical
  calls (spend decisions, credential confirmation) back to the user;
  decide technical/design questions yourself. This is also recorded in
  the assistant's own persistent memory
  (`geo-full-technical-autonomy-expected.md`).
- The user pushed back, correctly, when the assistant's first framing of
  "no more options" for `self_bertscore_mean_f1` was too narrow — asking
  "isn't there a way to influence the *input*, since an LLM just maps
  input to output?" That question is what led directly to v108. When a
  metric looks closed, check whether the closure was scoped to specific
  *tried* categories before treating it as exhausted.

## Assumptions Made

- None newly marked ASSUMED in `DECISIONS.md` this session — every new row
  (G17–G23) is VERIFIED or MEASURED with a script named. `D4` (the measured
  layer is domain-portable in principle; the taxonomies/cue text are
  hardcoded English/Reddit) remains from code reading only, unchanged from
  before.

## Potential Gotchas

- **A unit test that calls a Writer-prompt helper function directly proves
  nothing about whether a default run ever calls that function.** This is
  the v108 lesson; the fix is to test through
  `configure_generator_backend` + `build_writer_prompt` for every
  `--writer-prompt` mode, per `FocusedWriterPromptTest`/`WriterPromptTest`'s
  existing convention.
- **`_writer_prompt_mode`'s default is `focused`, not `full`.** Any future
  change to Writer-prompt content must touch `_focused_thread_ledger`
  (and, for completeness, `_thread_memory`), not just whichever one is
  easiest to reach from existing tests.
- **`docs/ORIENTATION.md` §4's "What may never happen" list is short,
  explicit, and non-negotiable** — read it by name before proposing any
  mechanism that inspects Writer candidate text and could select, reject,
  or resample based on it. `semantic_thread_diagnostics`/
  `semantic_distribution_problem` in `generation_diversity.py` is the
  existing mechanism this rule already governs; a new one shaped the same
  way is not a loophole.
- Every N=1 gate result is descriptive only (`inferential_status:
  DESCRIPTIVE` in the matched-evaluation output) — this project has now
  measured, twice, that a clean N=1 win can fail to replicate at N=10.
  Read v108's seed-8 result with that calibration, not as settled.
- `source_provenance.py` still refuses to start a real run when any
  version-defining file is uncommitted. Commit at the version boundary
  first, always.

## Tools/Services Used

- OpenAI API via `--api-key-env LLM_API_KEY` (`third_party/MiroFish/.env`).
  Planner and Writer both `gpt-5.4-mini`. ~$9.15 billed this session across
  five paid runs.
- Local CPU: `sentence-transformers/all-mpnet-base-v2` (plan/text
  embeddings, `PlanSemanticIndex`, and this session's new analysis
  scripts), `microsoft/deberta-xlarge-mnli` via the local `bert_score-master`
  checkout (the real `self_bertscore_mean_f1` scorer, used in
  `bertscore_pair_diagnosis.py`). System `python3` (`transformers==4.48.0`)
  for BERTScore work, not `.venv` (`5.10.1`) — the model hash must match
  the shipped artifacts.

## Active Processes

None. All background jobs from this session (self-tests, offline
diagnostics, depth decompositions) have exited. No paid job is currently
running; the v108 N=10 gate has not been started.

## Environment Variables

No secrets recorded here. Generation reads `.env` and
`third_party/MiroFish/.env`; arms are set from `GENERALIZED_CARD_*`
variables written by `run_generate.py`, including the new
`GENERALIZED_CARD_SEMANTIC_COVERAGE_NONREPEAT`.

## Related Resources

- Latest paid artifacts this session:
  `artifacts/generalized_card/runs/v108_semantic_coverage_nonrepeat_seed8_20260823_v2`
  (the valid v108 gate; `_v1` fired 0/186 and its metrics are not evidence),
  `artifacts/generalized_card/runs/v107_verdict_close_guard_seed8_20260822_v1`,
  `artifacts/generalized_card/runs/v106_chain_novelty_digit_guard_seed8_20260822_v1`,
  `artifacts/generalized_card/runs/generalized_card_camera_gpt54_v107_digit_verdict_n10_20260822_v1` (the N=10 null result).
- Baseline for any seed-8 gate: `artifacts/generalized_card/runs/v104_evaluative_seed8_20260821_v1`
  (all arms off; the correct J6 control for every single-thread gate this session).
- Baseline for any N=10 pool: `artifacts/generalized_card/runs/generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1`.
- Run index: `generalized_card/RUN_INDEX.md` (regenerate with
  `PYTHONPATH=generalized_card python3 generalized_card/scripts/build_version_log.py`
  after any new paid run).

---

## TYPE INSTRUCTION — how the next session should start

Paste this to the next agent verbatim.

> You are continuing a research project as a **senior AI research
> scientist**, not a coding assistant taking tickets. You have full
> technical and design autonomy on this project — the user has stated this
> explicitly and repeatedly; only route spend/credential decisions back to
> them, decide everything else yourself. Think all the way through a
> mechanism before building it, try to kill your own hypothesis first,
> measure on the real scorer, report effect sizes, write down what you
> rejected, and be your own reviewer. Never present a fix that has not
> been shown to move what it claimed — and never trust that it moved
> anything until you've grepped the artifact for proof it even ran.
>
> **Start here, in this order:**
> 1. `docs/ORIENTATION.md` — the whole spec, including §4's non-negotiable
>    "what may never happen" list. Read it by name; don't rely on memory.
> 2. `docs/DECISIONS.md` — every rule now in force, G1 through G23, each
>    marked VERIFIED / MEASURED / ASSUMED / RETRACTED.
> 3. `tasks/todo.md` — the `self_bertscore_mean_f1` item has the full
>    G16→G23 arc; read it before touching this metric again.
> 4. `tasks/lessons.md` — five new entries this session; the last one (a
>    prompt-text fix that reached 0/186 real prompts because it touched
>    the wrong of two ledger-builder functions) is the one most likely to
>    recur if not read.
>
> **Your first task**: the isolated N=10 gate for `--semantic-coverage-nonrepeat`.
> Predictions are already written in `generalized_card/VERSION_LOG.md`
> (search for "N=10 gate — `--semantic-coverage-nonrepeat on` isolated").
> Before spending, run
> `python3 -m pytest generalized_card/tests/test_generalized_card.py -k semantic_coverage_nonrepeat_reaches_both`
> as a $0 sanity check that the arm actually reaches a real prompt. The
> commands:
>
> ```bash
> python3 -u generalized_card/scripts/run_generate.py \
>   --tag generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1 \
>   --domain camera --model gpt-5.4-mini \
>   --base-url https://api.openai.com/v1 --api-key-env LLM_API_KEY \
>   --pool-size 150 --max-posts 10 --posts-per-run 5 \
>   --start-seed-index 2 --sampling-seed 42 \
>   --semantic-coverage-nonrepeat on --resume
>
> python3 generalized_card/scripts/run_evaluate.py \
>   --tag generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1 \
>   --metric-parallel 5 --resume
> ```
>
> **Before reading a single metric from that run**, grep its own
> `generation_records.json` for `"Do not restate one of these
> already-covered points"` and confirm it appears in nearly every comment.
> This exact check is what caught last session's bug and is now mandatory
> practice for this arm, not optional diligence.
>
> **The three things you must not get wrong:**
> - **A single-thread gate's win is not evidence it replicates at N=10,
>   even on the same thread.** `--verdict-close-guard` proved this
>   directly (G15 → G16). Read the N=10 result on its own terms.
> - **Distribution diagnostics never select a Writer candidate — this is
>   non-negotiable** (`ORIENTATION.md` §4). Any mechanism that embeds
>   Writer candidate text and compares it to a metric-shaped band, to
>   decide accept/reject/resample, is the forbidden category, no matter
>   how it's dressed up (G20).
> - **`_writer_prompt_mode`'s default is `focused`, not `full`.** Any
>   Writer-prompt content fix must be tested through the real dispatch
>   (`configure_generator_backend` + `build_writer_prompt`) for the
>   default mode, not just asserted against the helper function you
>   edited.
>
> Credential: `LLM_API_KEY` from `third_party/MiroFish/.env`, already
> confirmed by the user across five paid runs this session — you do not
> need to re-ask. Commit at the version boundary before any run;
> `source_provenance.py` refuses otherwise. Write predictions into
> `VERSION_LOG.md` before spending, as every gate this session did, then
> report against them the same way: predictions checked one by one,
> then the metric table, then depth decomposition, then what was read in
> the actual comments, then your own errors if any.
>
> When you verify, retract, or add a rule, edit its row in
> `docs/DECISIONS.md` in the same session, with the script that proves it
> and the date. A retracted row is never deleted. The next new row is
> **G24**.
