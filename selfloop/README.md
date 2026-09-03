# Self-Loop Reviser

A metric-targeted revision loop that runs **after** Planner→Writer. It rewrites
a small share of comments per round with `gpt-5.4-mini`, and keeps a round only
when the targeted metric improved **and nothing else got worse**.

```bash
# recommended: survives the SIGKILLs this machine produces, resuming from the
# checkpoint instead of repeating paid rounds
SELFLOOP_OUT=artifacts/selfloop/run1 SELFLOOP_ROUNDS=14 \
  ./selfloop/run_loop.sh --tags v157_20260903_p0 v157_20260903_p1 ... --workers 10

# or directly
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false python3 selfloop/controller.py \
  --tags v157_20260903_p0 ... --rounds 14 --workers 10 --domain celebrity_geo
```

Any domain: `--domain news_geo`, `--domain game_geo`, `--domain camera` — the
tags and the domain config are the only inputs that change.

`--dry-run` scores the cohort and names the metric it would target, without
spending anything.

---

## What one round does

1. **Score** the cohort with the official scorers, models held open.
2. **Pick the target**: the worst still-failing metric, else the largest `|d|`.
3. **Select**: per thread, rank comments by their own contribution to that
   metric (leave-one-out on the real quantity, not a heuristic) and take the
   top `max_share` of them — 10–15%, never the whole thread.
4. **Propose**: one `gpt-5.4-mini` call per selected comment returns 5
   candidates, all calls concurrent.
5. **Choose locally**: keep a candidate only if it moves that thread's target
   toward its own matched real thread *and* does not push the guard metrics
   away from theirs.
6. **Rescore** only the threads whose text changed, and only the scorers whose
   output can move.
7. **Gate**: accept the round if the target improved and no other metric
   regressed. Otherwise restore every thread's text and try a different round.

## The acceptance rule

The user's rule, verbatim: *修改就必须只提高 target，而不能让其他任何 metric
下降*. In `judge.regressions`:

- a metric that went **PASS → FAIL** is a regression, always;
- a metric whose **|d| grew by more than 0.01** is a regression.

`|d|` and not the p-value: `|d|` is quantized to `1/N²` and cannot move under
float noise, while p can. The first live round rejected on
`impolite_rate: d +0.25 → +0.25` — a metric that had not moved at all — because
an earlier version compared a p-weighted quality score.

A round that gains nothing is also rejected, so a stuck loop rolls back rather
than drifting.

## Why it is fast

The CARD-era controller rescored the whole cohort in fresh subprocesses after
every round. Timed here on 2026-09-04, **a 6-comment thread costs almost exactly
what a 42-comment thread costs** — politeness 6.1s vs 5.8s, semantic 7.1s vs
6.3s. The cost was loading eight transformer models eight times, not scoring.

| change | effect |
|---|---|
| models held open in one process | 8 loads per round → 8 loads per **run** |
| rescore only threads whose text changed | a round touching 3 of 30 threads scores 3 |
| rescore only text-sensitive scorers | `thread_structure` skipped; the reply tree never changes |
| candidate ranking on a fast local scorer | full suite runs once per round, not once per candidate |
| all LLM calls concurrent | one round's API time ≈ one call |

**Nothing here approximates a metric.** Every number a gate sees comes from the
official scorer's own `main()`, run with the arguments the official pipeline
passes; only the model constructor is wrapped in a cache. `verify_engine.py`
rescores an already-scored cohort and compares field by field.

## Domain adaptivity

Nothing in this package names a product category. The CARD revisers said
*"keep the same card/bank/APR/fee/SUB point"* and could not be pointed at
celebrity or news without an edit; `test_no_strategy_names_a_domain` fails the
build if that creeps back.

Domain content reaches the model through three run-time channels:

| channel | source |
|---|---|
| community | `configs/domains/<d>.json:community_context` |
| protected names | `configs/domains/<d>.json:protected_entity_terms`, derived from the corpus by `enable_domain.sh` |
| the facts to preserve | extracted from the comment under revision — links, numbers, quoted spans, capitalised names |

The direction of every instruction is read off the measured gap, so the same
strategy handles a metric failing high and one failing low.

## Files

| file | role |
|---|---|
| `controller.py` | the loop, the gate, rollback |
| `metric_engine.py` | official scorers with models held open |
| `judge.py` | the 12-metric verdict; equals `combined_eval.py` |
| `candidate_scorer.py` | fast per-thread scoring, ranking only |
| `selection.py` | which comments are worth an API call |
| `strategies.py` | per-metric instructions, domain-neutral |
| `reviser.py` | prompt, call, parse |
| `threads.py` | read/edit/roll back `discussion.json` |
| `verify_engine.py` | engine ≡ official scorers |
| `test_selfloop.py` | 13 tests |

## Measured

On the v157 N=10 cohort, `gpt-5.4-mini`, this machine:

| | |
|---|---|
| baseline scoring | 113 s for 10 threads, all 8 scorers |
| one round | ~2.2 min (API 13 s, rescore 115 s) |
| rounds 1–2 on `semantic_mean_cosine` | accepted, `d` +0.48 → +0.44 → +0.40, 12/12 held |
| round 3 on the same metric | rejected, no gain — target rotated automatically |

The CARD-era controller rescored the whole cohort in fresh subprocesses per
round; the same ten threads cost ~9 min there against 115 s here, before
counting the rounds this one does not have to repeat after a kill.

## Robustness

This machine SIGKILLs the process between rounds — `Killed: 9`, no traceback,
no Jetsam record, at 8 GB resident on a 24 GB machine. Three things follow:

- `TOKENIZERS_PARALLELISM=false` and single-threaded torch, set before any
  model is imported: the kills always followed a `tokenizers: the current
  process just got forked` line, and a fork at 8 GB briefly doubles the mapping.
- `checkpoint.json` is written after every round, holding the revised text, the
  scores, and which metrics have already failed to move.
- `run_loop.sh` restarts the controller until it finishes, and `--resume-from`
  picks up at the next round rather than repeating paid ones.

A round that raises is caught, recorded in `history.json`, and the loop
continues with the next metric.

## Limits, stated

- `avg_depth` and `structural_virality` have no strategy: text-only revision
  cannot move a reply tree. They are protected, never targeted.
- Candidate ranking for `mean_story_probability`, `emotion_entropy` and
  `hard_disagree_rate` falls back to comment length — there is no per-comment
  contribution for them cheap enough to compute at selection time. The gate is
  unaffected; only the search is weaker on those three.
- `hard_disagree_rate` cannot be targeted: it is pairwise over parent/child
  pairs, so a candidate has no local score, and a target with no local score
  applies nothing while still paying for its API calls. It stays protected.
- The loop optimises the cohort it is given. A cohort of 10 has `|d|` quantized
  to 0.01 and a PASS line of `|d| < 0.52`; passing there is not passing at
  N=150, where the line is `|d| < 0.131` (G101).
