# Self-Loop Reviser

A revision loop that runs **after** Planner→Writer. Each round rewrites a small
share of a cohort's comments with `gpt-5.4-mini` and keeps the round only if the
metrics it was spending on improved **and nothing else got worse**.

```bash
# recommended: survives the SIGKILLs this machine produces, resuming from the
# checkpoint instead of repeating paid rounds
SELFLOOP_OUT=artifacts/selfloop/run1 SELFLOOP_ROUNDS=14 \
  ./selfloop/run_loop.sh --tags v157_20260903_p0 v157_20260903_p1 ... --workers 12

# or directly
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python3 selfloop/controller.py \
  --tags v157_20260903_p0 ... --rounds 14 --workers 12 --domain celebrity_geo
```

Any domain: `--domain news_geo`, `--domain game_geo`, `--domain camera` — the
tags and the domain config are the only inputs that change.

`--dry-run` scores the cohort and names the target it would take, spending
nothing. `smoke.py` goes further: it prints what the model would be told about
the worst threads, and with `--live` spends one round and reports whether each
thread's own numbers moved toward its matched real thread.

---

## Targets are groups, not single metrics

Some metrics are several readings of one quantity. `similarity` —
`self_bertscore`, `semantic_mean_cosine`, `self_bleu_4` — is how much a thread's
comments repeat each other, measured over soft token alignment, embeddings and
exact word runs. A rewrite that fixes one usually fixes all three, so targeting
one and merely guarding the others throws that away.

| group | members | why together |
|---|---|---|
| `similarity` | `self_bertscore`, `semantic_mean_cosine`, `self_bleu_4` | one quantity, three readings; they agree in direction |
| `register` | `polite_rate`, `neutral_rate`, `emotion_entropy`, `mean_story_probability` | how the comments sound; on a flat cohort they move together |

The two groups **fight**, which is why they are not one round: courtesy and
narrative arrive as repeated wording, so buying `polite_rate` pushes
`self_bleu_4` and `semantic_mean_cosine` back up. `similarity` is taken first
and its members are never targeted alone. `length_cv`, `impolite_rate` and the
rest follow as singletons, worst `|d|` first.

`GROUP_STRATEGY` carries one instruction per group rather than stacking the
members', because the members' contradict each other on purpose: the
`semantic_mean_cosine` strategy says the claim has to change, the
`self_bertscore` one says do not change what it claims. Targeted together, both
have to change.

## What one round does

1. **Score** — the cohort's own published `matched_evaluation` rows are reused,
   so the baseline costs nothing; only edited threads are ever rescored.
2. **Pick the target** — the first group with a failing member, else the worst
   remaining singleton.
3. **Select** — per thread, rank comments by their own contribution to the
   target (leave-one-out on the real quantity) and take the top `max_share`,
   10–15%, never the whole thread.
4. **Explain** — each selected comment gets one call carrying its rank among the
   thread's contributors, the comments it is measurably closest to (in full, and
   only those above the thread's own mean pair cosine), the exact 4-grams it
   reuses, and the facts it already states.
5. **Propose** — 5 candidates per comment, all calls concurrent.
6. **Choose locally** — keep a candidate only if it moves the thread's summed,
   scale-normalized gap toward its own matched real thread by more than it
   pushes the guard metrics away from theirs.
7. **Rescore** — only threads whose text changed, only text-sensitive scorers,
   one model resident at a time.
8. **Gate** — accept only the largest subset of edited threads that gains
   without regressing; roll the rest back.

## The acceptance rule

The user's rule: *不能修好一个却改坏更多*, and a bad round must not be built on.

- any metric outside the group that goes **PASS → FAIL** is a regression;
- any metric outside the group whose **|d| grew by more than 0.01** is a regression;
- inside the group, no member may drift away from zero and none may fall out of
  PASS; the group's summed `|d|` must fall, or a member must newly pass.

`|d|` and not the p-value: `|d|` is quantized to `1/N²` and cannot move under
float noise, while p can. An early version compared a p-weighted score and
rejected a round on `impolite_rate: d +0.25 → +0.25` — a metric that had not
moved at all.

**A rejected round is fully undone** — text, scores, and the file on disk — so
round 4 continues from round 2 when round 3 was rejected.
`test_a_rejected_round_leaves_the_previous_text_on_disk` asserts it. Rejection
is not all-or-nothing: `choose_subset` drops threads one at a time while any
damage remains, keeping the largest subset that satisfies the rule, and a
dropped comment's rewrite is fed back into the next attempt on it so the model
re-rolls different dice.

## Why it is fast, and why it survives

The CARD-era controller rescored the whole cohort in fresh subprocesses after
every round. Timed here on 2026-09-04, **a 6-comment thread costs almost exactly
what a 42-comment thread costs** — politeness 6.1s vs 5.8s, semantic 7.1s vs
6.3s. The cost was loading transformer models, not scoring.

| change | effect |
|---|---|
| baseline reuses the cohort's published rows | 106 threads: ~20 min → 0 s, 8 GB → 420 MB |
| rescore only threads whose text changed | a round touching 3 of 30 threads scores 3 |
| rescore only text-sensitive scorers | the two structural metrics are skipped; the reply tree never changes |
| caches built once, before the calls | one embedding pass and one BLEU matrix per thread per round, not two |
| all LLM calls concurrent | one round's API time ≈ one call |

Scoring is **scorer-major**: one scorer over every thread, then that model is
freed before the next loads. Thread-major held all eight at once and the OS
killed the 106-thread run at ~8 GB, twice. The peak is now the largest single
model (deberta-xlarge-mnli, 2.6 GB) instead of the sum, at the cost of one
reload per model per round, ~15 s apiece.
`test_scorer_major_matches_thread_major` asserts the numbers are identical.

**Nothing here approximates a metric a gate reads.** Every gated number comes
from the official scorer's own `main()`, run with the arguments the official
pipeline passes; only the model constructor is wrapped in a cache.
`verify_engine.py` rescores an already-scored cohort and compares field by
field.

## Domain adaptivity

Nothing in this package names a product category. The CARD revisers said *"keep
the same card/bank/APR/fee/SUB point"* and could not be pointed at celebrity or
news without an edit; `test_no_strategy_names_a_domain` fails the build if that
creeps back. `check_domains.py` renders a real prompt against all seven domain
configs.

| channel | source |
|---|---|
| community | `configs/domains/<d>.json:community_context` |
| protected names | `configs/domains/<d>.json:protected_entity_terms`, derived from the corpus by `enable_domain.sh` |
| the facts to preserve | extracted from the comment under revision — links, numbers, quoted spans, and names |

A capitalized word opening a sentence is only treated as a name if the thread
capitalizes it away from a sentence start somewhere too. Without that test,
1112 openers across 1159 celebrity comments were listed as facts the rewrite had
to preserve — instructing the model to keep exactly what `self_bleu_4` charges
for.

The direction of every instruction is read off the measured gap, and off the
group member the thread is **furthest** from real on. Reading it off the group's
first member gave 28 of 106 celebrity threads the opposite instruction.

## Files

| file | role |
|---|---|
| `controller.py` | the loop, the objective, the gate, rollback |
| `metric_engine.py` | official scorers, models cached and evictable |
| `judge.py` | the 12-metric verdict; equals `combined_eval.py` |
| `candidate_scorer.py` | exact incremental per-thread scoring and the guard |
| `selection.py` | which comments are worth a call, and what to say about them |
| `strategies.py` | per-group and per-metric instructions, domain-neutral |
| `reviser.py` | prompt, call, parse |
| `threads.py` | read/edit/roll back `discussion.json` |
| `smoke.py` | self-test: worst threads, what the model sees, one live round |
| `verify_engine.py` | engine ≡ official scorers |
| `check_domains.py` | one prompt rendered against every domain config |
| `test_selfloop.py` | 35 tests |

## Limits, stated

- `avg_depth` and `structural_virality` have no strategy: text-only revision
  cannot move a reply tree. Protected, never targeted.
- **`self_bertscore` has no local objective.** A cheap stand-in
  (`0.5·self-BLEU + 0.5·cosine`) predicts the direction of the official metric's
  change at Spearman +0.279 (p=0.1), right on 21 of 36 single-comment swaps —
  58%, which is noise. Mean comment length does not predict it either (+0.007,
  p=0.95 across 106 threads). It is fixed as a member of `similarity`, whose
  objective is carried by the two metrics with exact per-comment forms, and
  gated on the official scorer. Checking it exactly per candidate was priced and
  rejected: 4.7 s per `bert_pair_f1` on a median 26-comment thread, ~300 edited
  comments a round.
- `hard_disagree_rate` cannot be targeted: it is pairwise over parent/child
  pairs, so a candidate has no local score. Protected.
- Its candidate ranking, and only its, falls back to comment length. Every other
  target ranks on a real per-comment contribution.
- The loop optimises the cohort it is given. A cohort of 10 has `|d|` quantized
  to 0.01 and a PASS line of `|d| < 0.52`; passing there is not passing at
  N=150, where the line is `|d| < 0.131` (G101). Cliff's `d` is a rank
  statistic, so on a small cohort every thread can move toward real and `d` not
  move at all until a generated value crosses a real one.

## Measured

On the v157 N=10 cohort, `gpt-5.4-mini`, this machine (before groups):

| | |
|---|---|
| one round | ~2.2 min (API 13 s, rescore 115 s) |
| 14 rounds | 4 accepted, every metric improved or held, 8/12 → 10/12 at the N=150 bar |

On six celebrity threads, one live round targeting `similarity`
(`smoke.py --live`): 36 of 40 rewrites applied, `semantic_mean_cosine` closer to
each thread's own real counterpart on **6 of 6**, `self_bleu_4` on 4 of 6.

## Robustness

This machine SIGKILLs the process — `Killed: 9`, no traceback, no Jetsam record.

- `TOKENIZERS_PARALLELISM=false` and single-threaded torch, set before any model
  is imported: the kills always followed a `tokenizers: the current process just
  got forked` line, and a fork at 8 GB briefly doubles the mapping.
- The guard's scorers are built through `metric_engine` with the official
  keyword arguments. Built positionally they missed the cache key entirely and
  every guard model was loaded twice — and the positional call passed
  `max_length=512` where the official default is 256, so the guard truncated at
  a different point than the gate on 49 of the cohort's 4163 comments.
- `checkpoint.json` is written after every round, holding the revised text, the
  scores, which targets have failed to move, and the per-comment feedback.
- `run_loop.sh` restarts the controller until it finishes; `--resume-from` picks
  up at the next round rather than repeating paid ones.
- A target is retired after 4 deaths, not 2: the kills seen were the OS
  reclaiming memory, a property of this machine rather than of the target, and
  at 2 the run came one kill away from retiring `similarity`.
