# Plan — make the discussion read like people talking

Read [`../docs/ORIENTATION.md`](../docs/ORIENTATION.md) first — it holds the
goal, the judging standard, how to read each metric, and the working discipline.
This file is the **task list**; `tasks/v<N>-worklog.md` and `tasks/HANDOFF.md`
hold the evidence.

The previous version of this file ordered work by **where a code defect lives**.
This one orders it by **which measured gap it moves**, because three of the last
four paid runs fixed a real code defect and moved no metric. The old P0–P6 items
are all still listed, at the bottom, marked kept / struck / demoted, so nothing is
silently dropped.

## Priority, re-derived under the standard now in force — 2026-08-25

The user selected **Holm-Bonferroni** as the reporting standard on 2026-08-25
(J2, G51). Under it, `analysis/holm_state.py` gives the state and the ranking:

- **N=10 (v110) is 12/12 PASS under Holm**, 9/12 raw. This is a *gate* reading.
  N=10 p-values are optimistic by construction and **N=150 has never been run on
  any version**. Do not quote 12/12 as a result.
- **At N=150 the same generator fails four metrics**, and the order is not the
  one the last three sessions worked to:

  | metric | P(pass) @N=150 | rel. bias | closure needed |
  |---|---:|---:|---:|
  | `polite_rate` | **0.00** | −51.5% | ~75% |
  | `impolite_rate` | **0.01** | +39.9% | ~75% |
  | `self_bleu_4` | 0.16 | +18.8% | ~50% |
  | `self_bertscore_mean_f1` | 0.19 | +2.6% | ~50% |

  The other six non-structural metrics sit at 0.81–1.00 and need no work.

### 1. Run N=150 — the paper's scale, never run, ~$42 / ~11h resumable

Every priority claim above, and every one in G42/G51/G52, is a **projection**
from a 10-thread bias estimate, and the N=10 window is not a miniature of the
pool: mean 53.2 comments per thread against the pool's 39.8, and **1 of 10 big
threads against 16 of 150**, which carry 46% of all comments. Both failing
pairwise metrics are thread-size sensitive and `polite_rate`'s deficit is
localised to 25+ word comments, so the N=150 bias could differ in either
direction. The reporting standard was the stated blocker on N=150 and it is now
decided. Config: v110's arm list with `--length-transfer v97` (G49 did not
promote `refit`, and the headline run should not carry a rejected arm).

### 2. The politeness pair — the binding constraint, and still no instrument

G25 gives the headroom: the plan is right, the whole failure is realization, and
full realization of the existing plan lands `polite_rate` on real. What is
missing is the instrument. Measured this session on the evaluation's own
checkpoint (`scratchpad/subject.py`, to be promoted per E8):

- The deficit is **3.4x at the sentence level** — real 0.0817 of sentences are
  carriers (P(polite) > 0.80), generated 0.0239. Flat, not positional and not
  length-driven, which **rules out the length route** for this pair.
- **Rejected:** that a carrier is defined by predicating on the thing rather than
  narrating the speaker. `P(carrier | sentence starts with "I")` is 0.0819
  against 0.0816 otherwise — ratio 1.00 on 3,074 real sentences.
- **Rejected as measured, but the null is narrower than it looks:** no prompt
  line separates planned-polite slots that realize polite from those that
  realize impolite. A rule printed on 532/532 slots has zero variance and this
  contrast cannot see it (see `tasks/lessons.md`).
- **Open, running:** is the 3.4x a *prevalence* deficit (the generator never
  writes the shape) or a *conversion* deficit (it writes it and it does not
  land)? v104 already showed gratitude is over-produced 1.48x and converts at
  0.256 against real's 0.672, so at least one form is conversion-limited — and
  asking for more of a conversion-limited form is exactly what delivered 8.4%.

### 3. v111 — built, offline-verified, unrun

`--development-scope measured` extends the enumerated beat plan to 35 assigned
words. G50 identifies the beat plan as the causal instrument on realized length
(realized/assigned jumps 0.816 → 0.953 across the 100/101 boundary in all four
comparable runs; 21.3 realized words per delivered beat). Priced at 8–26% of
`self_bleu_4`. It targets priority **#3**, so it waits behind items 1 and 2.
Gate command and predictions: `generalized_card/VERSION_LOG.md`, v111 entry.

### 4. `self_bertscore_mean_f1` — the likeliest metric to be reported as a limit

G28 shows its convergence is produced inside the Writer, downstream of every
lever G20 permits; the largest identified lever is length composition at 14–26%
and it needs ~50%. Decide how the paper reports it rather than opening a fifth
mechanism against it.

## Priority, re-derived from the null — 2026-08-21

Reading v103's **own N=10 p-values** under Holm-Bonferroni instead of raw, and
each `|Cliff|` against its measured N=10 noise floor
(`generalized_card/analysis/acceptance_standard.py`):

| | raw | Holm | \|Cliff\| vs its N=10 floor |
|---|---|---|---|
| `self_bertscore_mean_f1` | fail | **fail** | 0.86 against 0.50 — **outside by +0.36** |
| `polite_rate` | fail | pass | 0.60 against 0.53 — outside by +0.07 |
| `impolite_rate` | fail | pass | 0.61 against 0.51 — outside by +0.10 |
| the other nine | pass | pass | all **inside** the floor |

**v103 is 11/12 under a standard that does not fail correct work** (the null
gives 12/12 98% of the time under Holm, and only 63% under the raw rule).

So the work orders itself, for the first time on evidence rather than on which
number looked worst:

- [x] **1. `self_bertscore_mean_f1` — the only metric that fails a correct test.**
      MWU 0.001, KS 0.002, `|Cliff|` 0.86 against a noise floor of 0.50. **The
      pairwise decomposition ran** (`generalized_card/analysis/bertscore_pair_diagnosis.py
      pairs`, v103 N=10 vs its 10 matched real threads, fidelity-checked
      first) and the answer is none of the three guesses cleanly:
      - It is **not** the `hard_disagree_rate` parent-echo mechanism.
        `same_branch` pairs show no reliable excess (+0.0056, p=0.32, n=10
        threads) — if replies were echoing down their own branch this bucket
        would be elevated, and it is not.
      - It is **not** uniform. `root_root` pairs are clean (+0.0039, p=0.63);
        `reply_reply` pairs carry the largest, most significant excess
        (+0.0274, p=0.002, same sign in all 10 threads); `root_reply` is
        between (+0.0130, p=0.027). It is a sign **inversion**: real
        reply-reply pairs are less similar than real root-root pairs (0.4905
        vs 0.4955); generated reply-reply pairs are more similar than
        generated root-root pairs (0.5136 vs 0.5089).
      - `parent_child` pairs do carry the single largest per-pair excess
        (+0.0256, p=0.0098) but are 1.3% of all pairs — real but too rare to
        move the pooled metric.
      - **Root comments already match real, same as `hard_disagree_rate`.**
        This is now the second metric where the whole defect is a
        reply-comment phenomenon.

      Also tested and rejected in the same pass: **environment drift**. Every
      real-side per-thread score in this project (v96-v104) was computed under
      `transformers==5.7.0`; every generated-side score under `4.48.0` or
      `5.10.1`. Rescoring one real thread under `4.48.0` moved its mean by
      1.6e-8 — not a mechanism, but worth having actually checked once.

      **No sixth hypothesis exists yet for *why* generated replies read more
      similar to each other than real replies do, independent of branch.** The
      search is now scoped to reply-only generation (Writer prompt/plan
      differences between root and reply slots), not the whole metric. Do not
      build that hypothesis without falsifying it on the excluded real corpus
      first (SS4 step 3).

      **Gated 2026-08-22 (seed 8, $1.2081): the fix works, the metric didn't
      follow.** `--reply-novelty-scope chain` eliminated the diagnosed
      chain-restatement defect completely at the plan level (0/186 plan
      violations, was 18) but `self_bertscore_mean_f1`'s gap *widened*
      (+0.0183 → +0.0218), and specifically worsened in the reply-chain depth
      bins the fix targeted. Reading the actual pairs: claim-level
      duplication is gone; what's left is **sentence-template reuse across
      different claims** ("@OP, [verb] X and see/check Y" recurring with a
      different specific object each time). `used_sentence_routes` doesn't
      catch this because it matches literal 3-4-token n-grams and these
      templates vary at the token that would need to match. See
      `docs/DECISIONS.md` G13 and `generalized_card/VERSION_LOG.md`'s v106
      gate result for the full pair evidence.

      - [x] **The template-reuse hypothesis above was measured at scale and
            REJECTED, same day.** `analysis/template_reuse_diagnosis.py`:
            generated's within-thread opener/closer near-duplicate rate
            (0.0016/0.0005) is indistinguishable from matched real's
            (0.0009/0.0003) and an 80-thread real null (0.0012/0.0005). The
            8 examples were real text, not fabricated -- they were an
            extreme tail real threads produce at a comparable rate. Same
            trap as v98's rejected "duplication tail" hypothesis. Do not
            build a route-ledger-style fix for this; there is nothing to fix.
      - [x] **What the same 8 examples actually were, read again:** three
            opener-side ("@OP, ..." -- a different mechanism,
            `opener_profile.py`, not chased further this session), and the
            closer-side ones are a lexical variant of the already-known
            `abstract_verdict_close` tic (`closing_move.py`, chased since
            v73, v100's fix) using "check"/"test" as the head noun, which
            the existing pattern's word list never named. Shipped as v107,
            `--verdict-close-guard {off,on}`, default `off`. Offline-verified
            across all four domains (self-test, 8 runs, $0). **More
            important number found along the way:** the *existing*
            `abstract_verdict_close` suppression is still 10-13x over real
            even where its cue reaches the Writer -- v100's fix reduced the
            tic, it did not close it. See `docs/DECISIONS.md` G13/G14,
            `generalized_card/VERSION_LOG.md` v107.
      - [x] **Gated 2026-08-22, isolated (seed 8, $1.1637):** decided to
            isolate rather than stack, since v105 was already falsified as
            this metric's driver and stacking a third change onto v105+v106's
            already-noisy secondary metrics would make any result
            unattributable. Result: the check-variant is fully eliminated
            (3/106→0/106, below real). `self_bertscore_mean_f1`'s gap on this
            thread **narrowed for the first time across four gates**
            (+0.0183→+0.0173), specifically in the two deepest reply-chain
            bins ([4,7) and [7,+)) — the same two bins v105+v106's gate had
            worsened. Still N=1, descriptive only; default stays `off`. See
            `docs/DECISIONS.md` G15, `generalized_card/VERSION_LOG.md` v107
            gate result.
      - [x] **N=10 gated 2026-08-23 ($4.3909, user-run): null result.**
            `self_bertscore_mean_f1` unchanged to two decimals of Cliff's
            delta (0.86→0.86, gap +0.0169→+0.0170) — the seed-8 single-thread
            win above did **not** replicate at the scale that carries real
            statistical weight (5 of 10 threads improved, 5 worsened, net
            zero). Depth-pooled pairs show a real improvement concentrated in
            the two deepest bins ([4,7), [7,+)), diluted to nothing by the
            metric's own equal-weight-per-thread averaging once other threads
            moved the other way. Criterion-2: digit-cue partially replicated
            (4.6×→2.45× real); verdict-close's own targeted number did not
            (0.0065→0.0067, statistically identical) — new flag: `enum_or_fact`
            fell to 0/532, unconfirmed, n too small. See `docs/DECISIONS.md`
            G16, `generalized_card/VERSION_LOG.md` N=10 gate result.
      - [x] **Resolved 2026-08-23: a structural fix was designed, then
            correctly stopped, then falsified as a Planner problem --
            closed as a research-design limit.** The "explicit
            depth-conditioned diversity objective" option above was
            designed as a real-time Writer-candidate-vs-thread-pool
            embedding guard, calibrated per domain
            (`analysis/reply_diversity_ceiling_calibration.py`), and
            **stopped before being wired**: reading the actual repair-loop
            insertion point first showed a structurally identical mechanism
            already exists and is deliberately diagnostic-only
            (`docs/ORIENTATION.md` §4's non-negotiable "distribution
            diagnostics never select a Writer candidate" -- G20). Before
            widening the one remaining legitimate category (Planner
            plan-structure, `reply_increment_problem`-style) instead, asked
            directly whether this is even a Planner problem
            (`analysis/plan_text_realization_gap_diagnosis.py`, G21): the
            Planner's own plan-field similarity already decreases with
            depth, including on non-ancestor pairs v105's ancestor-only
            scope can't see, and plan-vs-text similarity correlates at only
            r=+0.48 across 26,520 pairs -- the same realization-gap shape
            as `polite_rate` and `abstract_verdict_close`, now measured
            directly. Combined with v105's own N=10 gate (widening novelty
            checking to 100% ancestor-chain compliance made the targeted
            deep bins worse, not better), decided **not** to build or gate
            a whole-thread-scope version -- `docs/DECISIONS.md` G22.
            **`self_bertscore_mean_f1` now joins `polite_rate`/`impolite_rate`
            (G8) as a closed research-design limit, not an open task.** Do
            not open a fourth narrow mechanism against it without new
            evidence.
      - [x] **Reopened the same day by a genuinely different, legitimate
            category, and it's the best single-thread result yet
            (v108, `docs/DECISIONS.md` G23).** The closure above was scoped
            to two specific forbidden/low-ceiling categories (Writer-output
            checks, G20; widened Planner plan-similarity, G21/G22) -- it did
            not rule out changing what the Writer's *prompt* says before
            generation, which is a different, ordinary "arm" (same category
            as every cue-text fix this session). Found a real, concrete gap:
            the Writer prompt's "already covered" ledger lists prior
            semantic contributions but, unlike its sibling ledgers, never
            told the Writer not to repeat them; verified directly against
            the seed002 restatement chain. Shipped as
            `--semantic-coverage-nonrepeat {off,on}`. **First paid attempt
            wasted $1.19 on a real bug** (fixed the wrong of two prompt
            builders -- `_thread_memory`, not the actually-live default
            `_focused_thread_ledger` -- caught only after the gate ran, 0/186
            prompts touched). Fixed, re-verified end-to-end through the real
            dispatch this time, re-gated: arm fired 186/186, and
            `self_bertscore_mean_f1`'s gap on seed 8 narrowed +0.0183 ->
            +0.0139 -- the largest improvement of any mechanism this
            session, across four of five depth bins together rather than
            trading one for another. Still one thread (G16 already showed a
            clean single-thread win here can fail to replicate at N=10);
            default stays `off`. **Next decision, not made here:** an
            isolated N=10 pool to get a statistically powered read, given
            how much better this single-thread signal is than anything
            gated before it.
      - [x] **Isolated N=10 pool run (2026-08-23, $3.5978), and the
            single-thread win did not replicate -- `docs/DECISIONS.md`
            G24.** Arm fired 532/532. Paired, same-seed comparison against
            v103 (Wilcoxon on the 10 thread-level gap differences): no
            metric moved with statistical credibility (all p > 0.13), and
            `self_bertscore_mean_f1`'s own mean gap nominally widened
            (+0.0155 -> +0.0188, 4/10 threads improved). This is the
            named "flat or worse at the pool level" failure mode from the
            prediction, and it happened. A pair-level depth-bin
            decomposition (fidelity-checked, run against both the v103
            and v108 artifacts) shows the mechanism does have a real
            effect -- the deepest reply-chain bin improved 34% relative
            (+0.0432 -> +0.0284, the largest single depth-bin move
            measured this session) -- but it is diluted by the metric's
            own equal-weight-thread-mean definition (G17), the identical
            pattern G16 already found for the digit/verdict guards.
            **Does not overturn G22**: a second independent null result
            for `self_bertscore_mean_f1` at the metric level, now from a
            mechanism reaching Writer realization directly. Default stays
            `off`; ships as an independent criterion-2 win regardless
            (same standing as `--digit-cue-guard`/`--verdict-close-guard`).
            Untested next idea, not built: stacking all three arms, since
            each moves a different depth bin.

      Two follow-ups run, both offline: (1) the real-side direction
      generalizes -- checked on 247 of the 424 excluded threads with the cheap
      `all-mpnet-base-v2` proxy, `reply_reply` < `root_root` in 82% of them,
      p≈0 (`analysis/root_reply_diversity.py`); (2) the detector itself was
      checked by reading actual pairs (`bertscore_pair_diagnosis.py inspect`),
      not just trusted -- no sign `deberta-xlarge-mnli`/BERTScore is a bad
      choice, and the generated high tail turned out to hold a real,
      previously-undocumented criterion-2 tell: distinct comments in the same
      thread independently restating one argument in different words (see
      `docs/ORIENTATION.md` §6.3 for the examples). v98's trim test already
      showed this tail is too small to be the aggregate driver -- both
      readings are correct, they answer different questions
      (`tasks/lessons.md`).

      - [x] **A mechanism already existed for this and never fired.**
            `reply_increment_problem` (`require_reply_novelty=True` since
            before v104) scored **0 trips on the entire v103 artifact** --
            a probe-shape bug (a short anchor phrase compared against a
            longer compound ancestor description) suppressed cosine
            similarity regardless of content. Fixing the probe shape and
            walking the whole ancestor chain surfaces 60 trips on the same
            artifact, including both named qualitative chains. Shipped as
            v105, `--reply-novelty-scope {parent_only,chain}`, default
            `parent_only` (byte-for-byte legacy). Offline-verified across all
            four registered domains (self-test, 8 runs, $0). **Gated
            2026-08-22 -- fixed the plan-level defect completely (0/186
            violations) but did not move the metric; see the note above and
            `docs/DECISIONS.md` G11/G13.**
- [ ] **2. `polite_rate` / `impolite_rate` — pass at N=10 only for want of
      power, and will fail at N=150.** The gap is 0.18, which is 1.2 real
      between-thread standard deviations, and a +0.10 shift is caught 100% of the
      time at N=150. Generated sits at the **7th percentile** of the real polite
      distribution and the **90th** of the real impolite one. Everything found so
      far is worth 0.005-0.010 each: v104's three arms together moved 0.010.
      **Do not spend another paid run on a surface cue for this.** See below for
      what is left.
- [ ] **3. Everything else is inside the null. Stop working on it.** Nine of the
      twelve are indistinguishable from a second sample of real threads at N=10.
      `self_bleu_4` at Cliff 0.40 against a floor of 0.50 included -- the "no
      cheap lever exists" conclusion from v98 stands and the metric no longer
      needs one.

### What is left for the tone pair, after v104

Ruled out, each with a measurement: evaluative-word tier (paid run, 85% of the
density gap closed, 13.6% of the carrier gap), sentence length, hedging,
questions, story, possessives, marker counts, the length mix, and eleven named
semantic forms at 42% held-out recall which the generator already produces at
parity or above.

Of ~20 form-only features, only three predict a carrier at held-out lift >= 1.6:
`thanks_or_respect` (lift 7.6, generated already **1.39x** real),
`exclamation` (lift 6.5, generated **0.52x** real), and `under_8_words`
(lift 1.8, 0.82x).

**The exclamation is substantially a classifier artifact.** Adding one to a
generated sentence that evaluates nothing at all raises mean P(polite) from
0.023 to 0.174 and makes **8.8%** of them read as carriers; removing one from a
real carrier drops P(polite) 0.973 → 0.668 while a `.` → `...` control moves
0.946 → 0.955. Matching the measured real sentence-level rate is legitimate
imitation and is worth about **+0.005** on `polite_rate`. Going past it is
tuning to the metric, which this project does not do.

**The honest position: `polite_rate` may not be closable to N=150 tolerance
without gaming.** The plan's between-thread polite mean is already exactly right
(0.310 against a real 0.310); its variance is compressed (sd 0.095 against
0.155); and realization is 35%. What the classifier still keys on is not a form
property of human Reddit writing that a Planner can schedule. That is a result,
and it belongs in front of the user as a study-design question, not as more
paid runs.

## The ordering principle

The user's target is how people talk, decomposed into four dimensions. Mapping
them onto the 12 metrics and onto the per-thread evidence in `HANDOFF.md` §4.3:

| dimension | metrics | per-thread state |
|---|---|---|
| 1 semantic dispersed | `semantic_mean_cosine` | 5/10 threads within 20% — passes by cancellation |
| 2 low lexical overlap | `self_bleu_4`, `self_bertscore` | bleu 2/10; bertscore 10/10 within 20% but fails on a uniform +0.03 |
| 3 stories in first person | `mean_story_probability` | 3/10; overshoots 1.5–2.4× |
| 4 tone and emotion varied | `emotion_entropy`, polite/impolite/neutral, `hard_disagree_rate` | 0–3/10 on every one |

Only `avg_depth` and `structural_virality` are genuinely matched per thread, and
both are fixed by the matched sampler rather than won by generation.

## v98 drawn-typing-rhythm / length-calibration status — 2026-08-19

Shipped. Policy
`generalized-card-v2-drawn-typing-rhythm-length-calibration-v98-20260819`.
Evidence in `tasks/v98-worklog.md` and `generalized_card/VERSION_LOG.md`.

- [x] Render the reused mid-comment route ledger in the focused Writer prompt.
      `used_sentence_routes` had fed only the `full` arm since v66. Filtered to
      routes the thread has actually reused more than once, since a ledger
      headed "already reused" must not list ordinary phrasing. Arm
      `--route-ledger`.
- [x] Land the measured final-punctuation habit (profile schema 15). Generated
      0.041 against a real 0.173. Arm `--final-punctuation`.
- [x] Fix `length_cv`. **Not** the short-slot brake this list proposed: the
      realized/target curve is a smooth monotone regression toward the model's
      preferred length across the whole range, crossing 1.0 near 40 words, so a
      one-sided brake would have fixed one tail and left the other. Inverted the
      fitted transfer function instead (R2 0.894, n=532). Arm
      `--length-calibration`.
- [x] `emotion_entropy` was **wrong** in this list. It is not small-thread
      noise. v97 wrote zero exclamation marks in 532 comments against a real
      0.079, and in the 24,029-comment reference corpus a comment with one is
      1.48x as likely to carry a non-neutral dominant emotion. Arm
      `--sentence-rhythm`.
- [x] `self_bertscore_mean_f1`. Three hypotheses measured, two rejected. The
      cause is v96's `no_story` instruction banning tense rather than narrative
      on 453 of 532 slots: past-tense verbs 0.181 against a real 0.543, `have`
      at 11% of its real rate, `will` at 1%, and a thread lexicon of 2,670
      distinct types against a real 3,645. Arm `--no-story-scope`.

### v98 N=10 result — 2026-08-20

**8 PASS / 1 PARTIAL / 3 FAIL** (v97: 7/1/4). Full table in
`generalized_card/VERSION_LOG.md`; goal, metric interpretation and current state
in `docs/ORIENTATION.md`.

- [x] Paid seed-2 content gate, all six predictions read against every comment.
- [x] Length transfer function checked on the v98 seed-2 artifact before N=10.
- [x] N=10 under the five v98 defaults, paired to v97's seeds
      (`--start-seed-index 2`, `--sampling-seed 42`).
- [x] `length_cv` FAIL -> PASS (MWU 0.021 -> 0.473, Cliff -0.62 -> +0.20).
- [x] `emotion_entropy` improved (MWU 0.326 -> 0.571, Cliff -0.27 -> -0.16).
- [x] `self_bleu_4` characterised with an exact ablation harness. No phrase
      drives it; it is a length metric first and generated already matches
      length. Entity diversity is worth ~1/3 of the gap at partial r = -0.097.
      **No cheap lever exists. Stop looking for one.**
- [x] `self_bertscore_mean_f1` hypothesis 4 (lexical breadth via the loosened
      story scope) **rejected** — past tense 0.289 -> 0.288, metric unmoved.
      Four hypotheses now measured and rejected; the metric has **no verified
      mechanism**. Do not build a fifth without falsifying it first.

## v101 measured, `hard_disagree_rate` diagnosed — 2026-08-20

v101 shipped at **9 PASS / 0 PARTIAL / 3 FAIL**, the best in the project's
history. The honest read is in `docs/ORIENTATION.md` §6: at N=150 the pass
probability is a function of the effect size, and **six metrics are safe and six
are not**. Cliff table and projection there.

- [x] **`hard_disagree_rate` diagnosed.** Evidence: `tasks/v102-worklog.md`;
      reproduce with `generalized_card/analysis/disagreement_diagnosis.py all`.
      The handoff's claim that it "has never had a mechanism" is superseded.

      Root pairs already match (0.0621 against a real 0.0630). **Reply pairs are
      1.56× real and are 100% of the gap.** The head is nearly degenerate — all
      three class probabilities inside ≈[0.26, 0.41] — so the gap is a uniform
      ≈+0.017 translation of the decision margin, the same signature
      `self_bertscore` has.

      **Two mechanisms survived falsification.** (1) The assigned opener is not
      realized: `polarity_token` comes out at 2.42× its scheduled share, sourced
      from `discourse_marker` slots (obeyed 0.184) and `content_phrase` (0.460),
      and it is the highest-disagreement opener there is. Causally measured on an
      exact harness: obeying the plan moves the reply rate 0.2235 → **0.1862**,
      47% of the gap, with `self_bleu_4` unharmed. (2) Generated replies echo the
      parent's content words 1.4–1.6× as often; the counterfactual at the real
      echo distribution closes 55%, and it survives conditioning on both parent
      and reply length in all ten cells.

      **Nine hypotheses rejected**, including the v100 adjudication frame
      (−0.0029 on 11 slots), contrastives and the closing sentence (removing
      either *raises* the rate), hedges (0.0000), a graph-feature asymmetry, and
      environment drift.
- [x] **v102 `--opening-move` built and offline-verified.** Policy
      `generalized-card-v2-drawn-opening-move-v102-20260820`, profile schema 19,
      module `generalized_card/generalized_card/opening_move.py`. Entry in
      `generalized_card/VERSION_LOG.md` with the predictions written down first.

      **Caught during the build: a prose prohibition already existed.**
      `_opener_rule` has appended "Do not open with a bare agreement or
      disagreement token" since v96; on v101 it reached 504 of 532 prompts and
      was violated on 9.1% of them. So the arm does not add another categorical
      ban — it draws the concrete opening word per register and replaces the
      category with the ten measured tokens.

      Offline: 559 tests, ruff clean, 105 pins 0 drift, self-test on and off,
      draw fidelity within 0.0108 in every cell, rule proven on the real prompt
      path, domain generalization available on all four with correct degradation.
- [x] **Paid large-thread gate for v102 — passed.** Run
      `..._v102_opening_seed8_20260820_v1`, 186/186 comments, $1.1392, 24.4 min.
      Full result in `generalized_card/VERSION_LOG.md`.

      Every prediction beaten: `discourse_marker` realization 0.231 → **0.923**,
      realized `polarity_token` 0.1559 → **0.0538** against a measured 0.0526,
      **0 of 158 reply slots** prepended an unassigned polarity token against 19
      in v101. `hard_disagree_rate` 19.1% → **3.0%** relative error;
      `neutral_rate`, `emotion_entropy`, `polite_rate`, `impolite_rate` all
      improved too. Compliance with a **named token** ≈1.0 against 0.23 for the
      same instruction as a **category**.

      Two of my own errors recorded: the predicted thread band used the N=10
      pooled real instead of this thread's real, and the "polite_rate must not
      move" guardrail reasoned from a flat conditional while the arm acts on
      prevalence.
- [x] **v103 — stance-consistent opening.** Built 2026-08-21 after reading the
      gate's comments: v102's polarity draw contradicted the plan's stance on
      **2 of 10** polarity slots (both `stance=agree` told to open with `no`).
      The plan now picks the polarity family and the measured draw runs inside
      it. 565 tests, 105 pins 0 drift, self-test proven to catch the defect, no
      profile change. Entry in `generalized_card/VERSION_LOG.md`.
- [x] **N=10 under v103 — 9/1/2, and the effect sizes got worse.** Run
      `..._v103_stance_opening_n10_20260821_v1`, $3.7345. Full result in
      `generalized_card/VERSION_LOG.md`.

      The mechanism worked exactly as designed: **0 of 28** stance conflicts,
      leak 46/504 → **5/504**, `discourse_marker` obeyed 0.184 → **0.974**,
      realized shares now 1.17x measured (were 2.54x and 0.34x).

      **But |Cliff| ≤ 0.10 went 6/12 → 4/12** while the PASS count rose.
      `hard_disagree_rate` overshot to Cliff −0.23; `semantic_mean_cosine` and
      `mean_story_probability` left the safe zone.

      Both pre-registered watch items resolved as **noise** — story on the drawn
      slots Wilcoxon p = 0.966 at n=65, body negation McNemar p = 0.881 at n=504.
      Not acting on the gate's aggregates was right both times.
- [x] **CORRECTION to the line above: v103 did not overshoot, it converged.**
      The Planner aims at a held-out same-size thread, never the matched one, so
      the per-thread target is an independent draw — corr(template, matched real)
      is **−0.281** for `hard_disagree_rate`. Measured against its own target the
      generator's bias went **+0.0681 → +0.0032** (Wilcoxon p = 1.000). The
      template ceiling at n=10 is **−0.36**, so v103 at −0.23 is closer to real
      than a perfect generator would be. **The "6/12 → 4/12" count conflates
      generator bias with template noise and should not be used to compare
      versions.** New trap 4 in `ORIENTATION.md` §2.

      Corrected priority — only **three** metrics carry a statistically real
      generator bias, and they are the three that have failed since v96:
      `polite_rate` (bias −0.1856, p = 0.002), `impolite_rate` (+0.1529,
      p = 0.002), `self_bertscore_mean_f1` (+0.0174, p = 0.014).
- [x] **The tone pair is diagnosed.** `tasks/v104-worklog.md`; reproduce with
      `generalized_card/analysis/polite_sentence_diagnosis.py all`. The possessive
      lever this list proposed was **not** the answer and neither was any other
      marker: Polite Guard is *confident*, not near-degenerate (median margin
      −0.934, only 2.1% of generated non-polite comments within 0.10 of
      flipping, median P(impolite) among impolite-labelled 0.981), so a
      sub-sentence edit was never going to tip it. Five more hypotheses rejected
      — length mix (closes 6%), question marks (out-of-sample lift 1.08),
      `you`-modal (lift 1.37, worth +0.009), personal narrative (the gap
      persists inside every story bin), polite vocabulary (a bag of words
      *over*-predicts politeness on generated: 0.150 against an actual 0.106).

      **The mechanism is a whole sentence.** A *carrier* — a comment holding one
      sentence at P(polite) > 0.80 — reconstructs both failing rates to three
      decimals on both sides. Conversion is nearly intact (real 0.808, generated
      0.667); **prevalence is 3.6× off** (0.220 against 0.062) in every length
      band. Moving prevalence alone is worth **52%** of the `polite_rate` gap and
      **36%** of `impolite_rate`. Causally verified on the real checkpoint:
      inserting one real short appreciative sentence flips a generated
      non-polite 40w+ comment 0.29–0.50 of the time against a control of 0.121.
- [x] **Named the forms, and the answer was not a scheduling problem.** Eleven
      forms fitted on half the excluded threads and scored on the other half
      reach only **0.420** recall; three were dropped for not replicating. But
      the generator already writes the forms at or above the real rate
      (`gratitude` 1.48x, `positive_predicate` 1.39x, `bare_verdict` at parity)
      and they land at a quarter to a tenth of real precision. The cause is
      inside the sentence: hot-tier words 0.31x real, trailing downtoner tag
      **42.7x**, partitive reference **13.6x**. Ablation with a working control
      in `generalized_card/VERSION_LOG.md` under v104.
- [x] **v104 built and offline-verified.** Policy
      `generalized-card-v2-evaluative-register-v104-20260821`, profile schema 20,
      module `evaluative_register.py`, arms `--evaluation-tier`,
      `--downtoner-tag`, `--partitive-reference`. 588 tests, ruff clean, 106 pins
      0 drift, `off` proven to render an empty rule on the real prompt path,
      `--prepare-only` clean on seed 8. Predictions are in the VERSION_LOG entry;
      **read them before the gate**.
- [ ] **Paid large-thread gate for v104**, `--start-seed-index 8` (186 comments,
      about $1.14), then N=10. Read every comment: v103's stance bug was found
      by reading the gate, not by its metrics, and two of these three arms are
      suppressions whose failure mode is visible in the text before it is visible
      in a rate.
- [ ] **The carrier prevalence gap is still open and is the larger half.** Real
      comments hold an unambiguously appreciative sentence 0.220 of the time and
      generated 0.062; closing that alone is worth ~52% of the polite gap against
      v104's 28.1%. It is not addressed because 58% of those forms are still
      unnamed. It needs a better taxonomy, not a bigger regex.
- [ ] **`self_bertscore_mean_f1` — five rejected hypotheses and no mechanism.**
      Do not build a sixth without falsifying it first. The untried diagnostic is
      the one that cracked `hard_disagree_rate`: decompose the thread mean into
      its **pairwise** matrix (`--include-pairs` on the scorer) and ask whether
      the excess is uniform, parent-child, or same-branch. If it is parent-child
      it is the already-measured parent echo, which would make one mechanism
      serve `self_bertscore`, the other half of `hard_disagree_rate`, and the
      user's criterion-2 complaint at once.
- [ ] **v104 — make the opener schedule respect the root/reply conditional.**
      Demoted: this is a **fidelity** defect, not the cause of the
      `hard_disagree_rate` number.
      Realized polarity openers are **0.0847 on roots against a real 0.0224** and
      **0.0507 on replies against a real 0.0685** — inverted. `opener_profile` is
      a pooled marginal; measure it per pair kind the way `register_realization`
      measures per register. `opener_cost` is **not** the bug: fed true depths the
      scheduler behaves correctly.
- [ ] **Answer first: the generated root share is 0.335 against a matched real
      0.267.** Root pairs carry P(disagree) 0.063 against 0.143 for replies, so
      over-producing roots drags `hard_disagree_rate` down on its own, and it
      would confound any opener fix.

- [x] **v106 (2026-08-22) — digit-cue quantifier guard.** Fixed offline, not
      yet gated. Re-measured on v103 (`analysis/digit_cue_diagnosis.py`): bare
      `0`/`1` in 0.092 of generated comments against 0.020 real (4.6×). Real
      writers numeralize a plain quantifier too (55% of their own bare-`1`
      occurrences) -- the excess concentrates in that specific sub-pattern
      (8.2× generated:real) versus genuine enumerated/fractional/price uses
      (1.7×), not the raw digit rate. `--digit-cue-guard {off,on}`, default
      `off`. Offline-verified across all four domains (self-test, 8 runs,
      $0). See `generalized_card/VERSION_LOG.md` v106, `docs/DECISIONS.md`
      G12.
- [ ] **The parent-echo mechanism has no design yet.** It is the other half of
      the `hard_disagree_rate` gap and it is also the user's criterion-2
      complaint (*很容易去讨论同一个话题*). Note that `context_transform` does
      **not** address it: echo is *highest* in `parent_hidden`, where the Writer
      never sees the parent, so the echo comes from the plan's parent-local
      topic, not from copying visible text.
- [ ] Persist `opener_type` (and the rest of the slot distribution schedule) into
      `discussion.json`. The realization matrix above had to be recovered by
      grepping 532 saved Writer prompts.

### Open after v98, in priority order

- [x] **v99 — drawn realization of the assigned warm register.** Built
      2026-08-20. Policy
      `generalized-card-v2-drawn-register-realization-v99-20260820`, arm
      `--register-realization`. Diagnosis in `tasks/v99-worklog.md`, entry in
      `generalized_card/VERSION_LOG.md`.

      **The warmth-marker schedule this list proposed was rejected before it was
      built.** Moving marker presence to the real level predicts `polite_rate`
      0.070 → 0.088 only, because the gap is in the conditional
      (P(polite | marker) is 0.213 generated against 0.652 real), not the count.
      Three further hypotheses were also rejected: warmth-as-concession,
      first-person lived experience, and a dismissive-adjudicative register. And
      **negative-marker suppression would have hurt** — generated text uses
      *less* of the impolite vocabulary than real (excess −0.767 against a
      polite-vocabulary deficit of +8.381).

      What shipped instead: the plan is right (0.275 planned polite against a
      real 0.288) and realization is the failure (0.193 for polite, 0.897 for
      impolite), so polite-assigned slots are now asked for the four surface
      moves real polite comments of their size carry, drawn per slot at the
      measured rate. Draw fidelity within 0.011 in every band.
- [ ] **Paid large-thread gate for v99**, `--start-seed-index 8` (186 comments),
      then N=10. Predictions are in the VERSION_LOG entry; read them first.
- [x] **v100 — measured closing move.** Built 2026-08-20. Policy
      `generalized-card-v2-measured-closing-move-v100-20260820`, arm
      `--closing-move`. This is the **root of the adjudication frame** chased
      since v73 through five phrase-level attempts: the phrase was never the
      thing, how the comment stops is. Real text closes on an abstract verdict
      0.014 of the time and generated 0.265 (19.1x); real closes on a concrete
      fact of its own 0.152 against 0.048. On real *story* comments the frame is
      0.003 against 0.382 generated — 127x.

      Three Planner-side explanations rejected first: "decision intent" lift
      1.08x, "decision boundary" **0.83x** (slots with it produce the frame
      *less*), v97's gate 0.175 gated vs 0.210 ungated.
- [ ] ~~**v100 — the impolite bleed.**~~ **Measured and shelved.** Planned-neutral realizes impolite 0.513 and
      planned-somewhat_polite 0.478; that is 122 slots and the larger remaining
      share of `impolite_rate`. It needs a **suppressive** mechanism: no additive
      move discriminates `neutral` (every candidate below 0.3 held-out lift), and
      the two families generated over-produces are measured — `adjudge`
      (**0 of 15,294 excluded real comments, 0 of 659 matched real, 37 of 528
      generated**) and `dismiss_noun` (5.17× real). Suppression has a track
      record here: v98 took the semicolon 0.109 → 0.023 and the dash clause
      0.299 → 0.071.

      **The suppression was then measured and does not work.** Out-of-sample lift
      on P(impolite) is only 1.02–1.18x for the restrictive families, and the
      counterfactual moves `impolite_rate` 0.697 → 0.655 against a real 0.443.
      Also: banning the frame's exact phrasing removes only 15–27% of the
      over-used abstract vocabulary (`matters` still 33x after, `whole` 17x,
      `otherwise` 29x) — which is why five phrase bans failed and why v100 names
      the *move* instead. **The impolite bleed has no mechanism. Do not build
      against it without a new falsified hypothesis.**
- [ ] **Paid large-thread gate for v99 + v100 together**, `--start-seed-index 8`
      (186 comments), then N=10. They ship in one gate because each has its own
      flag and its own measurable realized rate, so one artifact gives per-arm
      attribution — the same way v97's four arms and v98's five were attributed.
      Predictions for both are in `generalized_card/VERSION_LOG.md`.
- [ ] `self_bertscore_mean_f1` — **five hypotheses now rejected.** The fifth,
      narrow shared vocabulary, failed on sign: r(bert gap, breadth ratio)
      **+0.155** and r(bert gap, top-200 gap) **−0.096**, both backwards, and the
      narrowest thread has the smallest gap. Per-thread breadth ratio is 0.893,
      so the narrowness is cross-thread while the metric is within-thread.
- [ ] Revert `--no-story-scope` default to `tense`. No metric benefit, and it
      added new repeated 4-grams. Keep the prompt-contradiction fix it carried.
- [x] **Traceability closed.** v97 and v98 had shipped uncommitted — HEAD was
      v96 and `sentence_rhythm.py` / `length_calibration.py` had zero commits.
      Committed as `e213f7a` (code) and `1abdb0e` (docs); pinned sources verified
      clean against HEAD with 0 drift. See `docs/ORIENTATION.md` §8.
      **Standing rule from here: commit at every version boundary, before the
      paid run, not after.** v97's standalone tree is permanently unrecoverable
      because the working tree interleaved the two releases.
- [x] **The rule is now enforced, not written down.** `source_provenance.py`
      makes `run_generate.py` refuse to start when any file defining the version
      is missing from HEAD -- the 55 pinned generation sources plus
      `core_contract.py`, which can never carry its own hash. The commit lands in
      `run_config.json`, so reproduction is artifact -> commit -> sources with no
      searching. Override is an env var and is recorded in the artifact. 19
      tests, verified on the real path with `--prepare-only`. See
      `docs/ORIENTATION.md` §8.
- [ ] Entity diversity: generated 0.438x real in 10/10 threads. Worth ~1/3 of
      the `self_bleu_4` gap and it also fixes an eye-visible tell (10 distinct
      product designators against 40 real).
- [ ] Eye-visible tells for acceptance criterion 2, none fixed: no generated
      comment contains a link (real 0.051); `check` at ~10x its real rate;
      `will` at ~1%; `their` and `we` absent.
- [ ] Remaining length defect: the `short` band overshot downward, realized/
      target 1.071 -> 0.857.
- [ ] Bug — evaluation drops <2-word comments unevenly, so
      `--exact-matched-thread-size` can still mismatch (24 generated vs 22 real
      on `post04_seed011`).
- [ ] Bug — `build_slot_distribution_schedule`'s `tone_length_fit` /
      `tone_length_joint` are never persisted to `discussion.json`, so the joint
      cannot be audited after the fact.
- [ ] Bug — `--template-phrase-reuse-budget 4` is flat and wrong at large thread
      sizes; real threads reach `uncertainty_frame` 7, 8 and 12. Should scale
      with comment count.
- [x] **Measured, no longer an estimate.** `acceptance_standard.py` draws two
      disjoint samples of real camera threads and runs the evaluator's own tests:
      the current standard passes **0.63 at N=10 and 0.50 at N=150**;
      Holm-Bonferroni over the same 24 tests passes **0.98 / 0.98**; the
      `|Cliff| <= 0.10` target passes **0.00 / 0.26**. A perfect generator is a
      second real sample, so the standard has to read ~0.95.
      **Recommendation to the user: Holm-Bonferroni over the 24 tests, with the
      real-vs-real null printed beside it.** Domain-portable -- same three rows
      for any domain from its own thread tables.
- [x] ~~**Drive every metric to |d| <= 0.10.**~~ **RETRACTED.** The null 95th
      percentile of `|Cliff|` is ~0.52 per metric at N=10 and ~0.13 at N=150, so
      the target sat below the noise floor from v97 to v104 and every N=10 Cliff
      reading under ~0.5 in this project's history is noise. Steer by the
      distance to the floor, or by trap 4's paired bias.

## v97 keyboard-surface / measured-joints status — 2026-08-19

The v96 N=10 run is the first complete honest sample under the new content
policy: coverage 1.00, 532/532 comments, **6 of 12 metrics pass**. Story,
emotion, and semantic dispersion now pass; the six failures are self-BERTScore,
self-BLEU, hard disagreement, and the three tone rates. Detail in
`tasks/v97-worklog.md` and `generalized_card/VERSION_LOG.md`.

- [x] Read all 12 v96 N=10 metrics from the saved matched evaluation and
      separate the passes from the failures without quoting an n=1 p-value.
- [x] Measure, in the v96 artifact, what the generated text does that the real
      text does not, at the level each failing metric measures.
- [x] Prove the typography cause with the real scorer, not a proxy: 0/532 ASCII
      apostrophes, and a per-speaker keyboard draw moves `self_bleu_4` MWU p
      0.009 -> 0.273 and KS 0.052 -> 0.787.
- [x] Prove the adjudication frame is on 532/532 slots and is worst on the least
      adjudicative functions (personal_datapoint 29.1%, reaction 19.0%).
- [x] Prove the tone marginal is correct and the joint inverted: impolite on
      100% of 250w+ slots where excluded real comments of that size are 72%
      polite.
- [x] Prove the long-slot shortfall is the request, not the token budget: 845w
      slot asked for 40 beats, Planner saturates near 9, realized 0.32x.
- [x] Add `surface_typography`, `comment_structure`, and `tone_length_fit`,
      measured on evaluation-excluded threads only; profile schema 11 -> 14.
- [x] Gate the boundary line on both Writer paths and cap the beat ceiling where
      the Planner still delivers.
- [x] Make each change a named arm recorded in `run_config.json` whose legacy
      value reproduces v96 exactly.
- [x] Complete 369 tests, Ruff, 98/98 pins, both parity scopes, self-test with
      all four arms, profile rebuild, active-shaper proof, and exact seed-2
      prepare-only.
- [x] Run one paid v97 seed-2 gate and check the six predictions. All six held;
      45/45 in one attempt for `$0.3883`.
- [x] Run v97 N=10: 532 comments, coverage 1.00, `$3.6664`, 55 min. **7/12 pass
      against v96's 6**, but by the |d| <= 0.10 standard only **4/12 are viable
      at N=150** against v96's 3.
- [x] Attribute both regressions before acting on either. `length_cv` is real
      and systematic (below real on 9/10 threads; the tail fix was outweighed by
      a 5-11% inflation of the 10-120w middle that holds 425 of 532 slots).
      `emotion_entropy` is not attributable (5 threads up, 5 down; per-band
      entropy equal or higher; the 7-comment thread alone accounts for 0.079 of
      the 0.087 mean drop).
- [x] Quantify what N=10 predicts about N=150. Simulating the evaluator's own
      MWU+KS pair: a metric needs |Cliff's delta| <= 0.10 to have a 0.72 chance
      at N=150, while |d|=0.25 still passes 87% of the time at N=10. Track |d|,
      not the p-value.
- [ ] Decide the reporting standard for 150 threads before running it. Twelve
      metrics times two tests at alpha 0.05 means even a perfect generator
      passes all twelve together only 52% of the time (0.94^12), so either a
      multiplicity correction or an effect-size-led report is needed. This is
      the user's call.
- [ ] Drive every metric to |d| <= 0.10. Of v96's six passes only three qualify:
      semantic_mean_cosine 0.02, structural_virality 0.04, avg_depth 0.06.
      emotion_entropy 0.12 and mean_story_probability 0.18 do not.

## v96 selective factual-grounding status — 2026-08-18

- [x] Complete and evaluate the paid v95 seed-2 gate; separate reliability
      success from content failure.
- [x] Compare all 12 exact n=1 metrics and direct content diagnostics without
      treating n=1 p-values as inferential.
- [x] Trace low specificity to a missing safe fact path, an incorrect ban on
      normal product-name reuse, and direct replies with no excluded reference
      knowledge.
- [x] Add `domain-claim=selective` while retaining `planned` and `off` as named,
      reproducible arms. Enforce the selected slot set after JSON parsing.
- [x] Give selective direct-reply planning evaluation-excluded reference rows
      and full ancestor semantic coverage; raw reference wording remains
      Planner-only.
- [x] Carry a delivered claim into Writer anchors and prevent a story slot from
      receiving both a claim and a second equipment-fact source.
- [x] Permit natural reuse of the thread's product name while rejecting reuse of
      the same fact or amount.
- [x] Complete 316 tests, full Ruff, 95/95 clean source pins, both parity scopes,
      selective/named backend self-test, and exact seed-2 prepare-only.
- [ ] Run one fresh v96 seed-2 paid gate, inspect every comment and the same
      content/12-metric diagnostics. N=10 remains blocked until content passes.

## v95 compiled/non-terminal Planner-contract status — 2026-08-18

- [x] Reconstruct all 19 saved v94 batch reports and all four terminal slots
      across its three whole-post attempts.
- [x] Prove 130/152 requests were Planner quality repairs and that each attempt
      stopped on a different stochastic content combination.
- [x] Identify the overlay ownership bug: aggregate target labels changed after
      semantic planning and validation rejected contradictions created by code.
- [x] Compile fixed story/social/capacity controls with dependent Planner route
      fields before quality evaluation; preserve semantic content fields.
- [x] Pass default `no_story` into the specialized direct-reply Prompt rather
      than overlaying it after a prompt that displayed `unassigned`.
- [x] Limit soft plan-quality repair to one call per slot; retain repeated repair
      only for a still-inconsistent compiled contract.
- [x] Make residual content-contract diagnostics non-terminal and auditable;
      retain hard failure for schema, transport/safety/empty output, and coverage.
- [x] Replay every saved v94 selected batch: terminal conflicts `3 -> 0`; add a
      cross-product contract stress test and complete 307 tests.
- [x] Complete Ruff, both parity scopes, 95/95 source pins, backend self-test,
      and exact v95 seed-2 `--prepare-only`; all pass without an API call.
- [x] Run seed 2 once and inspect content plus n=1 diagnostics. Reliability
      passed, but the content hypothesis failed; supersede v95 with v96 and do
      not run v95 N=10.

## v94 state-preserving Planner-repair status — 2026-08-18

- [x] Reconstruct all v93 seed-2 S9 repair candidates from the persisted audit,
      rather than extrapolating from the v92 candidate.
- [x] Identify the root cause: whole-plan replacement alternately erased the
      repaired social contract and repaired long-form beats.
- [x] Merge only `development_plan` when `long_form_capacity` is the slot's sole
      remaining repair issue; retain whole-plan repair for mixed-issue slots.
- [x] Record raw and applied candidates plus merged fields in the audit log.
- [x] Replay the exact v93 final candidate: blocking `1 -> 0`, five beats and
      `small_observation` both retained, no remaining S9 issue.
- [x] Add unit and active-wrapper regressions; complete 304 generalized tests,
      Ruff, source pins, and backend self-test.
- [x] Run the v94 seed-2 gate with three total attempts. All three failed on
      different content-contract combinations after 152 requests and `$0.9608`;
      no evaluable thread was produced.
- [x] Supersede v94 with v95. Do not retry, resume, or evaluate the v94 artifact.

## v93 root/reply-boundary status — 2026-08-18

- [x] Reconstruct the paid v92 seed-2 S9 initial plan, all three repair
      candidates, quality ranks, and structural slot contract from persisted
      audit payloads.
- [x] Prove the first repair supplied the required five beats and was rejected
      only because a root row carried `reply_delta_type=social_close`.
- [x] Make topology clear reply-only controls from roots before quality
      selection while preserving those controls on direct replies.
- [x] Remove the duplicate direct-reply rules and dead parent-contract renderer
      from the root Planner Prompt; keep the stable fields as literal `none`.
- [x] Replay the actual candidate: rank `(1,46) -> (0,41)`, blocking `1 -> 0`,
      five beats retained.
- [x] Complete full tests, scorer tests, Ruff, both parity scopes, 93/93 pins,
      backend self-test, and exact v93 N=10 prepare-only.
- [x] Attempt a fresh v93 N=10 tag. It again completed two threads, then exposed
      repair-state loss on seed 2; supersede it with v94 and do not evaluate or
      mix the partial artifact.

## v89 Planner-repair status — 2026-08-17

- [x] Preserve the failed v88 run evidence: 24 Planner requests, `$0.1805`, no
      Writer calls and no evaluable discussion.
- [x] Reconstruct the exact 186-slot schedule and offset-8 failure from current
      source plus `planning_quality.jsonl`; do not infer from the traceback.
- [x] Rank targeted repair by blocking contract count before aggregate quality,
      so a collision cannot make code retain a Writer-impossible story plan.
- [x] Keep polite/helpful role drift as low-weight repair feedback, but stop
      treating the surface classifier label as a post-blocking semantic truth.
- [x] Remove the root Planner's story/no-anecdote contradiction while retaining
      non-leakage and externally checkable fact boundaries.
- [x] Persist initial, candidate, recovered, and selected Planner snapshots and
      repair ranks in the audit log.
- [x] Complete full v89 verification, exact seed-8 prepare-only, and source-pin
      refresh before another paid run.
- [x] Supersede v89 before a paid run after the completion audit found the same
      story-grounding ambiguity on the direct-reply Planner path.

## v90 reply-story-grounding status — 2026-08-17

- [x] Read the complete direct-reply Planner and Writer grounding modules and
      compare their rendered story/fact contracts with the root Planner.
- [x] Define the conservative synthetic-story boundary once and render it on
      both Planner paths; retain the ban on seed facts and externally checkable
      outcomes.
- [x] Add a direct-reply Prompt regression for both the permission and the
      factual boundary.
- [x] Complete full offline verification, source-pin refresh, and exact v90
      seed-8 `--prepare-only` before an API call.
- [x] Supersede v90 before a paid call after auditing the pending `named`
      concreteness arm's global/per-slot instruction conflict.

## v91 slot-gated concreteness status — 2026-08-17

- [x] Trace `own-fact-license` from system Prompt through per-slot license
      resolution and every Writer Prompt path.
- [x] Prove the old `named` arm pressured unlicensed micro/short slots to add
      names and amounts.
- [x] Make the system rule a conditional authorization only; keep the concrete
      behavior instruction once in licensed substantive Prompts.
- [x] Measure the seed-8 gate: 110/186 slots (59.14%) versus 59.68% matched-real
      digit-bearing comments and 31.35% generated.
- [x] Complete full offline verification, repin, and exact named-mode v91
      prepare-only before an API call.
- [x] Supersede v91 before a paid call after proving that `domain-claim=off`
      still planned and then discarded a separate fact.

## v92 lossless-domain-claim status — 2026-08-17

- [x] Trace `domain_claim` through root/reply Prompt, normalization, registry,
      task handoff, and every Writer path under both flag values.
- [x] Make off-mode require `domain_claim=none` on both Planner routes and omit
      claim-only Prompt prose.
- [x] Require the complete move to live in Writer-visible semantic fields and
      clear a noncompliant returned claim during normalization.
- [x] Preserve planned mode and its claim-specific Prompt/delivery path.
- [x] Complete full verification, repin, rendered off/planned Prompt audit on
      both Planner routes, named backend self-test, and exact v92 prepare-only.
- [x] Start the fresh v92 N=10 tag with named concreteness. It completed two
      threads and exposed the root/reply boundary bug before seed 2 could be
      persisted; supersede it with v93 rather than mix policies.

## v88 completion-audit status — 2026-08-17

- [x] Replay the exact off-mode grounding contract over all 186 frozen tasks:
      78 equipment permissions, 144 personal-experience bans, 61 conflicts.
- [x] Stop rendering invented equipment unless the explicit legacy `own`
      license is selected. Replay now has zero equipment/ban conflicts.
- [x] Separate recurring-speaker structure from semantic persona content.
      Delete kit, tenure, use-case, display-name, and kit-filter dead weight.
- [x] Make matched anonymous participation the default and keep `off` as the
      one-author-per-slot ablation. Fail rather than silently dropping a
      requested matched roster.
- [x] Verify current seed-8 shape without source identity leakage: 186 slots,
      97 generated speaker groups, 80 named-source groups, 17 anonymous
      one-shots, 2.112 turns/named group, 66.7% recurring mass, max 10 turns.
- [x] Verify v88 offline: 292 generalized tests, 3 focused scorer tests, Ruff,
      matched backend self-test, active/legacy parity, 93/93 pins, 186-task
      Prompt replay, and exact v88 `--prepare-only`.
- [x] Attempt v88 seed 8. It failed in Planner before Writer generation; record
      the cost and failure rather than treating it as a content experiment.
- [x] Supersede the failed v88 behavior with v89; do not rerun under the same
      policy ID after changing repair semantics.

---

## v87 full-route replay status — 2026-08-17

- [x] Replay all 186 recorded v80 tasks through current root/reply and
      substantive/low-info Writer paths, preserving their long-thread order.
- [x] Gate low-information Writer routing by payload semantics before short
      utterance shape. Remove six `soft_helpful` and one `correction` false
      routes; retain 25 legitimate low-information slots.
- [x] Replace full-blackboard rendering plus text parsing with direct bounded
      focused ledgers. Dedupe nearby openings and social-close semantic moves.
- [x] Recompute Writer-facing tone controls after the final Planner contract so
      stale acknowledgement instructions cannot contradict the task.
- [x] Make gratitude/social-close metadata coherence bidirectional and blocking
      before Writer generation.
- [x] Verify v87 offline: 290 generalized tests, 3 focused scorer tests, Ruff,
      backend self-test, active and legacy parity, 93/93 pins, full Prompt
      replay, and exact v87 seed-8 `--prepare-only`.
- [x] Supersede v87 with v88 before a paid run after completion audit proved a
      grounding conflict and structural-speaker/persona coupling.
- [ ] If the v93 N=10 run completes with credible realization, interpret MWU/KS
      on that unchanged policy and use content evidence to choose any next fix.

---

## v86 Prompt audit status — 2026-08-17

- [x] Render representative focused Writer Prompts and check exact duplicate
      lines and semantic target conflicts rather than judging source strings.
- [x] Translate root-only Planner relations at the Writer boundary from
      parent language to post language; preserve direct-reply relations and the
      persisted raw plan.
- [x] Remove repeated low-information Writer blocks. Keep one route lock, one
      compact discourse contract, one per-slot guidance section, the bounded
      semantic/short-line ledger, and the low-information hard rules.
- [x] Split 214 lines of legacy reviser-only Prompt logic out of active
      `prompts.py`; prove migrated and retained functions with AST hashes.
- [x] Bump the generation policy to v86 before any paid run.
- [x] Supersede v86 with v87 before a paid run after full-route replay proved a
      payload-routing defect that representative Prompt samples did not expose.

---

## v85 implementation status — 2026-08-17

Detailed evidence and exact scorer definitions are in `tasks/v81-worklog.md`;
the completion-audit fixes are recorded in `tasks/v82-worklog.md` and
`tasks/v83-worklog.md`.
This supersedes the older idea that a repetition warning should resample one
comment: collection-level metrics are diagnostic in first-pass generation.

- [x] Remove copied short-slot `development_plan` prose before the Writer.
- [x] Make story/no-story a bidirectional Planner contract, including
      `evidence_mode`; stop unresolved hard contracts before the Writer.
- [x] Put story/tone/affect/opener controls into direct-reply planning rather
      than adding them only after its semantic plan is written.
- [x] Remove the canned gratitude semantic rewrite and automatic
      `soft_helpful` conversion; targeted Planner repair owns those choices.
- [x] Jointly pair tone, affect, and story marginals. On the frozen v80 large
      template all labels remain assignable and the measured contradictory
      pairs collapse sharply.
- [x] Remove the duplicated focused-Writer tone block and the neutral-affect
      instruction conflict.
- [x] Disable per-comment distribution retries and repetition best-of-N at the
      v81 public CLI; retain bounded recovery only for non-persistable output.
- [x] Restore the compact Planner discourse contract on the default focused
      Writer path. Function, payload, role, voice, evidence, content angle, stance,
      detail, intent, reply relation, and exclusion now survive once; a raw-plan
      end-to-end test prevents a planned rant from collapsing into generic help.
- [x] Remove matched-text tone leakage from surface inference: links, quotes,
      capitalization, emoji, and punctuation remain anonymous shape, but lexical
      gratitude no longer creates a `pure_acknowledgement` contract.
- [x] Remove the remaining indirect matched-text semantic paths: first-person,
      uncertainty, story/rant, tangent, and template labels are no longer
      inferred from evaluation wording. Delete the two dead frame regexes.
- [x] Finish source-pin/version updates and full code review: 263 tests pass,
      backend self-test passes, and all 72 source pins have zero drift.
- [x] Reject incomplete Writer coverage before persistence. The paid v80 seed-8
      artifact contained 186 records but only 185 comments; it is now also
      rejected by output audit even though its accepted share is 99.46%.
- [x] Resolve the live quote-opener/parent-copy contradiction without weakening
      the general copy guard: only a scheduled short markdown excerpt followed
      by an independent reply is allowed.
- [x] Delete the unreachable `omit_without_backfill` branch and correct run
      metadata that still described obsolete omission/persistence behavior.
- [x] Complete offline verification: 266 tests, Ruff, backend self-test, 72/72
      pins, existing-artifact audit replay, and exact v84 `--prepare-only` pass.
- [x] Audit the current dead-control candidates without treating historical
      implementations as authority. Remove retired tone-overlay Prompt reads,
      unreachable legacy tone-label branches, and the superseded scalar metric
      projection helper while retaining old-record deserialization fields.
- [x] Record initial and repair-time template-contract overrides in
      `planning_quality.jsonl`; keep post-override story/tone/affect coherence
      as a blocking pre-Writer contract.
- [x] Stop impossible perspective-concentration repair calls. Perspective
      concentration remains visible as a warning; invalid-perspective and
      branch-route checks that normalization made unreachable are deleted.
- [x] Verify v85 offline: 285 generalized tests plus 3 focused scorer tests,
      scoped Ruff, camera-product backend self-test, 92/92 declared hashes,
      67/67 active sources git-tracked, zero missing active imports, v80 audit
      replay, and exact seed-8 `--prepare-only`.
- [x] Repair the n=1 content audit before using it for decisions. Its lexical
      rows were matched, but real emotion/story rows came from the whole domain.
      Evaluation now writes exact-matched 12-metric, repetition,
      Planner→Writer, model-realization, and weak-surface JSON/Markdown reports.
- [x] Separate target selection from Writer realization for all 12 metrics.
      Persist each selected excluded-real template atomically and report
      real → target and target → generated gaps plus MWU/KS/Cliff/Wasserstein.
      The selected-target distribution passes 12/12 at N=10 and N=150; do not
      rewrite the sampler based on one high-variance n=1 draw.
- [x] Stop evaluating postprocessed text. `run_evaluate` now stages the Writer
      artifact byte-for-byte after the integrity audit; noncanonical structure
      fails rather than being normalized.
- [x] Make n=1 descriptive at the matched evaluator, console, JSON/Markdown,
      and content report. A single thread can no longer print `12/12 PASS`.
- [x] Remove active evaluation dependence on the dirty calibration candidate
      modules. The focused metric runner, formal statistics, matched evaluator,
      and scorer CLIs are pinned and git-recoverable; legacy revisers are not in
      default parity or the current workflow.
- [x] Include dynamically imported and subprocess-launched runtime sources in
      provenance. Generation/evaluation backend runners, output-audit runner,
      token tracker, and token summarizer are tracked and pinned; the closure
      audit now follows sibling-script imports as well as package-relative ones.
- [ ] Run one large n=1 content/contract diagnostic with no metric-driven
      retries, then run a multi-thread matched evaluation for formal p-values.

---

## A — realize the assigned register   [dimension 4, largest gap]

**Why.** Measured on v79, 184 aligned slots: assigned `impolite` realizes at 93%,
assigned `polite` at 13% with 59% collapsing into impolite. Overall realization
59.2%. One register per thread explains polite ↓, impolite ↑, hard_disagree ↑ and
emotion_entropy ↓ **simultaneously** — these are not four problems.

**What the data already eliminated** (do not redo):
- *Not length.* Real polite is 52% of 60–120 word comments; generated is 6%, and
  0% above 120 words. Generated long comments are 73–88% impolite.
- *Not insufficient agreement.* Real comments carry more negation than generated
  (41.5% vs 31.2%) and are still scored polite.

**What differs** (seed 8, real vs v79): warm markers 14.0% vs 11.8%; emotional
endpoint 2.5% vs 1.1%; hedge 18.0% vs 12.9%; decision-framing nouns 0.5% vs 4.3%.

**Tasks**
- [x] Delete the hedge and thank-you prohibitions from `TONE_DEFINITIONS["polite"]`
      (`generation_distribution.py:480-489`). The block above the table records
      the prediction that motivated them — collapse into `somewhat_polite`. The
      measured collapse is into **impolite**, so the prediction was wrong.
- [x] License the emotional endpoint explicitly ("I love it", "never looked
      back"): real 2.5%, generated 1.1%.
- [x] Cut decision-framing nouns from the Writer's own rule text — 8.6× overshoot,
      and the Writer is substituting analysis for feeling.
- [x] Check `_affect_instruction` rotation (`generation_distribution.py:448-470`)
      still reaches the prompt on the focused path; the affect rewrite was
      bundled into v73 and never cleanly attributed.
- [x] Ablation flag; `off` restores the pre-v80 contract and is recorded in
      `run_config` as `social_contract_coherence`.
- [x] **Offline gate before any paid run:** re-render the v79 prompts from
      `generation_records.json[].task` and confirm the banned surfaces are gone
      and the new ones present.
- [ ] **Judge the paid run on tone realization rate (59.2% baseline) and
      `emotion_entropy`**, not on p-values. n=1 has no p-value.

---

## B — the global typographic signature   [dimension 2, free]

**Why.** `self_bertscore` has never passed in any version, but §4.3 shows it is
not a large error: 6.9% mean relative error, 10/10 threads inside ±20%, failing
only because all ten overshoot by a near-uniform +0.03. That is the signature of
one global constant, not of content.

The strongest available candidate: **every generated comment carries the same
typography.** Of comments containing an apostrophe, 100% of generated use only
curly `’`, against 17.6% of real. Curly overall: generated 72–74%, real 11–13%.
Straight apostrophe inside a word: real 51%, generated 0%. Verified
model-emitted, identical before and after `gpt_cleanup`.

**Tasks**
- [x] Do not add deterministic apostrophe normalization: the actual-scorer
      counterfactual below explains only a minority of the gap, so this is not a
      justified primary fix and would add post-processing without fixing talk.
- [x] Run a no-regeneration counterfactual first: on 40 v79 comments / 780 pairs,
      curly apostrophes -> ASCII moved 0.52947 to 0.52381. This explains only a
      minority of the ~0.034 gap, so do not implement a held-out-calibrated
      normalizer as the primary fix.
- [x] Address the current rough-surface gap upstream: impolite slots explicitly
      allow ordinary non-targeted profanity and amusement slots allow a natural
      laughter token. Neither is required or hard-coded to one phrase.
- [ ] Re-measure the other surface gaps after v81, all measured on seed 8:
      paragraph breaks real 25.5% vs generated 2.8%; no final punctuation 24.0%
      vs 6.6%; URLs 4.5% vs 0%; `lol/haha` 3.0% vs 0%; ALLCAPS 19.5% vs 7.7%.
      These are prompt-level, not post-processing, so keep them separate from the
      typography change if attribution matters.

---

## C — bring `mean_story_probability` down   [dimension 3]

**Why.** Generated overshoots real by 1.5–2.4× on seed 8. Real per-thread
`story_rate` ranges 0.000 (seeds 0, 3, 5) to 0.275 (seed 6), mean 0.110. The
previous handoff said the allocation was correct and should not change; that was
wrong.

**Before changing allocation**, note how the metric is computed: StorySeeker's
P(story) averaged over **every** comment in the thread, not only story slots. So
non-story comments drifting narrative would produce the same overshoot. The
per-thread story count already scales from the matched template
(`generation_distribution.py:129-134`).

**Tasks**
- [x] Score story-mode slots and no-story slots separately in an existing run to
      see which class carries the overshoot. Offline, the per-comment
      probabilities are already in `cleaned/*/storyseeker_results.json`.
- [x] Diagnose realization as the main failure: planned story slots supplied
      only ~25% of total story probability; 25/167 `no_story` comments were
      classified as stories. Add a Writer no-sequence contract and reject
      `no_story + personal_story` plans before writing.
- [x] v81 root fix: no-story also rejects `firsthand_experience`; scheduled
      stories require a coherent personal-datapoint evidence plan in both root
      and direct-reply planners. v80 replay exposes 59 latent conflicts.
- [x] Keep first person for the slots that do tell stories: every scheduled
      story task sets `allow_first_person_frame=True`, while the joint contract
      requires firsthand personal-datapoint semantics.

---

## D — the two arms that are built but never run

- [ ] **Run `--own-fact-license named`.** Domain-neutral concreteness: name things
      and give quantities. Gated on `substantive_slot` (≥25 real words, not
      micro/short). v91 fixes its global/per-slot Prompt conflict; metric and
      content effect remain unmeasured. Targets the two signals that separate
      real from generated on all ten threads — quantities 12.3×, proper nouns
      1.85×. Note the sibling arm `own` was refuted; see HANDOFF §6.6.
- [x] **`--speaker-identity matched`.** Recover only anonymous participation
      structure; v88 removes the old invented biography/kit coupling and makes
      this the default. Current seed 8 has 80 named groups over 169 named slots,
      2.112 turns each, with 66.7% total comment mass from recurring groups.
      The prior hypothesis was:
      265 named participants over 559 real
      comments, 2.11 each, 68% of comment mass from someone who speaks more than
      once; the generator gives every comment a distinct one-shot author. Targets
      `self_bertscore` through voice variation. Metric effect remains unproven.

---

## E — reply-planner sibling visibility   [was P3, kept]

- [x] Every depth ≥ 1 batch takes `render_direct_reply_planner_prompt`
      (`prompts.py:336-381` routes there; batches never mix depths, so all of
      them qualify). It renders no prior-plan ledger, no coverage summary, no
      sibling contract, no branch goal, no R# rows. Each row sees only its
      parent. Verified on seed 2: depths 3–8 are single-slot batches and tasks
      38–45 are nine near-duplicate moves that could not see each other.
- [x] Add `--reply-sibling-visibility`; the `on` arm renders every sibling and
      already committed delta/novelty object, while `off` restores old rows.
- [x] Do not add a stricter `semantic_move` gate without new evidence. v80 seed
      8 semantic cosine is already near real, sibling visibility now exposes
      competing reply increments, and a stronger gate could over-disperse it.
      The existing whole-plan
      `semantic_collision` check cannot catch it: `plan_similarity` is a Jaccard
      over all `SEMANTIC_FIELDS` including `development_plan`, so a ~20-token
      move is ~10% of the token mass; `_dependent_variation` exempts parent–child
      pairs. Reopen only if v81 text repeats sibling moves or semantic cosine
      moves above real.
- [x] Fix the beat-budget contradiction: root and direct-reply prompts now use
      the dynamic count rendered from `expected_development_beats`; no separate
      35-word/16-beat rule remains in prompt prose.

---

## F — turn on more of the validation layer   [was P0, demoted]

Resolved in v81 by simplifying the policy instead of promoting more metrics:

- [x] Distribution and repetition findings are diagnostics and never resample,
      rank, select, or drop a persistable comment.
- [x] `missing_concrete_anchor` remains an audit signal, not a retry trigger.
- [x] Empty text is explicitly classified as a hard realization failure even
      when the shared Writer omits the problem code; bounded hard recovery
      handles it before persistence.
- [x] Delete the old repairability sets, candidate ranking, repetition arm, and
      dead CLI parameters rather than retaining an unreachable controller.

**Corrected from the previous version of this file:** "add the 6 missing metrics
to the Writer's distribution target (`run_generate.py:488`)" — that line is a
record written into `run_config.json`, not a wire. The real target is hard-coded
in `generation_diversity.build_thread_distribution_target:40-43`. In v81 the
target is audit-only and no longer ranks candidates. Five of the six also need
transformer classifiers inside the generation process. See HANDOFF §6.1.

---

## G — simplify without erasing history

- [x] Preserve v68-v80 behavioral provenance in `VERSION_LOG.md` and compare all
      behavior fields during resume/extension/policy upgrade.
- [x] Run repository-wide reference and AST audits before deletion.
- [x] Remove unreferenced reviser prompt builders and stale helpers while keeping
      every active reviser entry point.
- [x] Re-pin all changed generalized sources and commit only the scoped files;
      unrelated dirty worktree content belongs to other sessions.
- [x] Replace the 1,177-line cleanup dependency and broad calibration
      runner/scorer/stats dependencies on the active evaluation path with two
      focused modules. Remove 140 lines of reviser instructions from the current
      README while retaining legacy entry points for explicit reproduction.
- [x] Prove active provenance mechanically: every current generation/evaluation
      source is tracked, every hash agrees, and an AST import-closure audit finds
      no unpinned local dependency.

---

## Struck, with the measurement that struck them

- **Plan-echo validator (was P1).** Echo is at 0.0% since v75 and the route lock
  that achieved it moved no metric. Nothing left to guard.
- **Length for polite slots (was in P5).** Real polite is length-driven; the
  effect does not transfer. Generated 60–120 word comments are 6% polite, 120+ are
  0%. HANDOFF §5.2.
- **`LENGTH_BUCKET_BOUNDS["very_long"] = (120, 220)` (was in P6).** Read only by
  `_retry_note_for_problems`, i.e. on a retry, which under `--writer-retries 0`
  almost never happened. Dead. `length_cv` is within 3.5% of real anyway.
- **`--own-fact-license own` (was P4a).** Refuted in v76b: 0.05 → 0.02
  specification tokens per comment against a real 0.54. 68% of real
  spec-carrying comments have no first-person frame, so the gate was wrong.
  Retained only as a reproducible arm.
- **B7 "`allocate_story_and_affect` is a no-op auditor".** Not a bug; it is a
  deliberate auditor, documented at `generation_distribution.py:108-114`.

## Re-audit of old B-items against the current path

- `B2` is fixed: root and direct-reply prompts use the same dynamic beat budget.
- `B3` is false now: `allow_first_person_frame` reaches both Writer rules and
  guards.
- `B4`/`B6` are fixed in v85. The old overlay fields remain only for record
  deserialization; the unreachable old tone label is gone.
- `B5` is stale: `_delexicalize_tone_examples` no longer exists.
- `B8` was partly wrong: schedule values are applied before semantic-quality
  evaluation, so incoherent plans are repaired or blocked. v85 wires the
  previously discarded initial/repair override events into the run log.
- `B9` was partly right: invalid-perspective and branch-route validation were
  unreachable and are removed. Planner `utterance_mode` is not a requested
  semantic field; it is intentionally inferred during task construction.
- `B11` is fixed: structurally owned perspective concentration is diagnostic,
  not a slot-local repair target.
- `B12` is false now: repair feedback reaches both root and direct-reply prompts,
  and sibling/reference context is rendered when enabled.
- `B13` is false now: hard recovery is wired, audited, and covered by tests.

These corrections are based on current call order and tests, not on the old
handoff's interpretation.

## Sequencing

The free target-selection and evaluation-integrity work is complete. Next run
the fresh v93 N=10 arm and judge both Writer realization/content and formal
matched statistics. Only exact 10/10 coverage is a comparable sufficient-N
evaluation. Reopen A/C upstream only for target→generated failures that remain;
do not change the reference sampler, add a reviser, or tune against final
test-set p-values. One mechanism per later paid run, prediction written first,
control semantics versioned, and `--prepare-only` before spending.

---

# Review — 2026-08-26 session (v115, v116, v117)

Paid spend **$3.93**, one 10-thread calibration run. Full evidence:
`tasks/v115-v117-worklog.md`. Rules added: `DECISIONS.md` **G54–G63, E10–E11**;
`G2` marked SUPERSEDED. Handoff:
`.claude/handoffs/2026-08-26-geo-v115-v117-*.md`.

## What was built

| arm | flag | fired? |
|---|---|---|
| v115 | `--tone-quota {off,inverted,calibrate}` | `calibrate` yes; **`inverted` never run** |
| v116 | `--rhythm-count {off,measured}` | yes — counts written `{1:40, 2:8, 3:3}`, was `{1:48}` |
| v117 | `--reference-link-count {off,measured}` | yes — 1.68 URLs/carrier vs real 1.67, 61 chars vs 61 |

All default `off` and byte-identical when off, asserted by test. 720 tests pass.
`PROFILE_SCHEMA_VERSION` 21 → 22. Policy
`generalized-card-v2-drawn-link-count-v117-20260826`.

## The result that outranks the arms

An arbitrary **disjoint real** camera thread passes all six reported metrics at
coverage 0.996, with `self_bertscore` bias **+0.24%**. The target is reachable, the
evaluation design is sound, and the generator's +2.41% is **10x** the natural
spread (G54). This is also a free per-domain validation harness —
`analysis/self_similarity/real_vs_real_floor.py` — and it should be run on any new
domain before a token is spent.

## Ordered next steps

1. **Fix v117's content defect (G61).** Four unrelated links stacked at a comment's
   end; an Apple support URL inside a Sony A7 thread. Measured fix: of 249 real
   comments carrying 2+ non-media URLs, **64.3% have ALL their URLs on one host**
   and the first URL sits a median **23%** into the comment. **v117 must not enter
   a paper run before this.**
2. **Decide v115's `POLITE_ASSIGNMENT_CAP` against the reported-metric set, not
   L2 (G60).** Cap 0.59 lands `polite_rate` +0.2% and `impolite_rate` +1.3% while
   costing `neutral_rate` −19.7%; cap 0.35 leaves all three mid-range at 35%
   closure. This is a judgement call, not a measurement.
3. **Measure the persona layer against `self_bertscore` (G57).** The only single
   channel above the 42% bar — headroom **+0.0060 = 51%** of the gap — and
   `persona_bridge` / `speaker_roster` / `actor_conditioning` /
   `--speaker-identity matched` have never been measured against it. **Free.**
4. Evaluate the calibration run for the first full-coverage reading with all three
   arms on: `python3 generalized_card/scripts/run_evaluate.py --tag
   v117_calibration_20260826_v1`. Its **tone numbers are meaningless** (the quota
   was deliberately flat); `self_bertscore` and `self_bleu_4` are readable with a
   ~3% confound measured separately.
5. Only then N=50 / N=150.

## Closed as dead, do not re-propose

- Six tone realization hypotheses: more register cues, the omitted conjunction,
  hedging, length repair, the bare-assertion frame, the polite lexicon (G53, G58,
  G59). The lexicon one is decisive: generated already carries real's top-45
  polite tokens at **1.14x** real prevalence.
- `evidence_mode` labelling (G62) — two of three cheap cells already matched in the
  text with the label at zero.
- Digit runs (−0.0005) and hapax flattening (−0.0020) as `self_bertscore` channels
  (G63). Digits dying closes the only route needing domain vocabulary.
- First-person rate, thread structural metrics, and comment length as
  `self_bertscore` channels (G55, and the worklog's leverage section).

## Standing arithmetic

`self_bertscore` gap 0.0119. Built: v117 ~17%, v115 ~3%, **assuming additivity**
(the surface channels measured sub-additive at ~0.86). → ~0.0095 against the
**0.0069** Holm needs at N=150. Both terms are J7 upper bounds. **The gap does not
close with what exists today**; G57 is the only measured channel large enough.
