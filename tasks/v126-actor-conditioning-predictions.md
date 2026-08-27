# Predictions for v126 — thread-local actor conditioning. Written BEFORE the run.

2026-08-27. Control is **v125b on the same seeds** (the newest arm, and the one
whose `self_bertscore` Cliff is lowest at +0.90).

## The bar this run is judged against

The user fixed it this session and it is now the only standard used here:

> N=10 时 self-BERT / self-BLEU 的 p-value 至少要在 0.6–0.7；其他 metrics 即使
> 退步也不能低于 0.6–0.7。

G101 showed that is an effect-size statement. The exact Mann–Whitney null for
n=m=10 gives p=0.684 at Cliff d=0.13, and the N=150 PASS line is |d| < 0.131.
**So the bar is |Cliff d| ≤ 0.13 on all twelve metrics**, and every number below
is stated in that currency.

## What it is

`--actor-conditioning domain-derived` on top of v125b's flags. Nothing else
changes. The flag already exists (`run_generate.py:736`), is already wired into
the Planner schema, the Planner rules, the per-slot plan summary line, and the
Writer prompt's "Thread-local actor state composed by the Planner" section, and
is covered by the backend self-test. It has **never been run on the calibrated
line** — every run to date carries `actor_conditioning.mode = "none"`.

The Planner composes, per slot, from the visible thread plus the
evaluation-excluded reference rows: `knowledge_boundary`,
`participation_goal`, `evidence_access`, `attention_focus`,
`interaction_tendency`, `context_visibility`, and `realization_route` — the last
being "an abstract one-shot sentence construction and cadence", with the
standing rule to **vary it across nearby slots before Writer generation**.

## Why this, after four arms failed

G105: selection is exhausted. An oracle picking the best of three paid drafts
per slot reaches Cliff +0.34 on `self_bertscore` at best, and §1 forbids the
mechanism anyway. New material is required, not better choosing.

G103 (corrected): the gap is **not** mostly within-thread cohesion. Real threads
cohere (+0.0126 excess over their own cross-thread mean) and ours cohere only
slightly more (+0.0191). Of the +0.0248 within-thread gap, **+0.0183 (74%) is
corpus-level**: our comments are more like each other than real comments are
*even across unrelated threads*. That is one voice, not one topic.

G104: perfect outsider compliance is worth only ~36% of the gap, which the 26%
within-thread term explains arithmetically. The topical route cannot finish.

The masking experiment rejected the two lexical explanations for the 74%.
Blanking brands and model designations moved the within-thread gap 0.0248 →
0.0258; blanking comparative-hedge frames moved it 0.0248 → 0.0239. Neither
carries it. The uniformity is diffuse across the prose, so the lever has to act
on sentence architecture rather than on a token class — which is exactly what
`realization_route` is.

E14: the write-time BERTScore band control that this evidence first suggested is
barred by §1 (no resampling toward a metric) and would in any case have been
inert at the shipped `--writer-retries 0`. This arm resamples nothing:
`writer_distribution_resampling` is hard-coded `False` in both modes.

## Predictions — judged as written

**Compliance gate (checked first; if it fails the metric results are void).**
Per E12 an arm can record itself ON and reach zero prompts.

1. `actor_conditioning.mode` reads `domain-derived` in `run_config.json`.
2. **≥ 90% of Writer prompts contain "Thread-local actor state composed by the
   Planner".** v125b's rate is 0%.
3. **≥ 8 distinct `actor_realization_route` values per 10 consecutive slots**,
   measured on the shipped plans. A route repeated across a whole thread means
   the Planner accepted the schema and ignored the vary-it rule, which is the
   v125 failure mode in a new field.

**Primary.** `self_bertscore` Cliff **+0.90 → ≤ +0.60**. The mechanism attacks
the 74% term; halving that term moves the mean gap from +0.0271 to about
+0.018, which is roughly Cliff 0.6 at n=10. Reaching ≤ 0.13 in one arm is not
predicted and would be a surprise.

**Secondary.** `self_bleu_4` Cliff +0.38 → ≤ +0.30. Varied sentence architecture
should cost shared 4-grams, but the audit already shows our repeated-5-gram
share *below* real's, so most of `self_bleu_4` is not phrase reuse and the move
should be small.

**Guard — this is what kills the arm.** No metric currently at |d| ≤ 0.13 may
leave it: `impolite_rate` (+0.20 on v125b — already out, watch only),
`neutral_rate` (−0.24 — already out), `avg_depth` (+0.03), `structural_virality`
(+0.02), `mean_story_probability` (−0.12), `emotion_entropy` (+0.08),
`semantic_mean_cosine` (+0.04). **If `semantic_mean_cosine`, `avg_depth`,
`structural_virality` or `emotion_entropy` leaves |d| ≤ 0.13, the arm is
rejected regardless of what `self_bertscore` does.** `polite_rate` (−0.33) and
`hard_disagree_rate` (−0.39) are already out of band and are not guards.

**Null result.** If `self_bertscore` lands above +0.80 with the compliance gate
passed, voice conditioning is not the 74% lever and G103's corpus-level reading
needs a different mechanism — the next candidate is per-thread speaker
partitioning rather than per-slot state.

## Cost

v125b was $3.79 for 1,002 requests. The actor schema, rules, per-slot suffix and
Writer section add input tokens on every call; budget **≤ $5.00**. If the
`--prepare-only` dry run shows the Planner prompt growing by more than 30%,
stop and re-scope before spending.

## Order of operations

1. `--prepare-only` on a throwaway tag, then delete the tag. Verify separately
   what `--prepare-only` skips (it returns before the API-key check, E11).
2. Commit before the paid run — `run_generate.py` refuses otherwise, and the
   override marks the artifact unreproducible.
3. Re-pin and confirm the drift list is exactly the files edited (none expected:
   this arm adds no code).

---

## Addendum — the first firing was VOID, and it found a bug worth more than the arm

**v126 (`v126_actor_VOID_focused_path_20260827`) was killed at $0.81.** Its
compliance gate failed on the second check, exactly as that check was written to
do:

| gate | target | measured |
|---|---|---|
| actor fields on plans | — | **100%** (91/91, every batch offset) |
| actor section in Writer prompts | ≥ 90% | **26.5%** (22/83) |
| distinct `realization_route` per 10 slots | ≥ 8 | **10/10**, no repeats |

The 22 hits were exactly the low-information slots — 100% of that path.
`writer_prompt` dispatches to three builders and `_focused_writer_prompt`, the
shipped default that every substantive slot takes, never rendered the actor
state. The registry was fine: all 45 keys present for a 45-slot thread, lookup
key matching. The string had nowhere to go. Recorded as **E15**; fixed by
resolving the actor state once in `writer_prompt` and passing it to all three
builders, pinned by three tests, one of which fails on the pre-fix tree.

**Consequence beyond this arm:** every run this codebase has ever made with
`--actor-conditioning domain-derived` was ~26% applied. Those results are
unattributable, not negative.

## Addendum 2 — v126b failed on Writer coverage, and the arm is now suspect

**v126b was rejected by the generator at $0.46**, on its first post:
`generated=44/45 failed_task_ids=[22]`, policy `reject_incomplete_post`. One
unrealized slot voids the whole post, and the degraded retry returns the task
unchanged, so a slot that fails *because of* the actor block cannot recover.

The correlation is clean and it points at the fix:

| run | actor coverage | slot failures |
|---|---:|---|
| v122 | 0% | 0 |
| v125b | 0% | 0 |
| v126 (unfixed) | 26% | 0 |
| **v126b (fixed)** | **100%** | **first post** |

Three explanations were rejected for free before spending again:

- participant keys (`A3`) tripping `planner_skeleton_residue` — the regex is
  `P\d{2}|S\d+|B\d+`, so `A#` is not matched
- actor fields leaking internal control ids into the Writer prompt — **0 of 136
  plans** across both runs contain `P##`/`S#`/`B#`
- the longer prompt squeezing the completion budget —
  `writer_provider_token_budget` depends only on `real_word_count`

**The blocker was observability, not the hypothesis space.** An incomplete post
is never persisted, so the records carrying each attempt's guard problems died
with it and the only visible fact was a task id.
`_print_failed_slot_diagnosis` now prints, per unrealized slot, its payload
type, length bucket, matched word count, attempt count and every recorded guard
problem, before the post is rejected. A single-post reproduction against the
same seed is what this addendum is waiting on.

**Open question this raises regardless of the actor block:** with
`reject_incomplete_post` and 45–185 slots per post, a per-slot hard-failure
probability of even 1% voids a 185-slot post 84% of the time. Every arm to date
has run at p≈0, so the policy has never been stressed. If the diagnosis shows a
generic guard rather than an actor-specific one, the thing to fix is the
policy's brittleness, not this arm.
