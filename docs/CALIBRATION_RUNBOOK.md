# Calibration runbook — measuring a realization matrix for a new Writer

## Why this document exists

`tone_realization.py` holds `REALIZATION_MATRIX`, `C[i][j] = P(realize j | assign i)`,
and inverts it so the quota handed to the Planner is the assignment whose *realized*
mix lands on the reference template. The matrix is **a property of the Writer**, not
of a thread — so swapping the Writer invalidates it.

`docs/DECISIONS.md` G159 measured how far: on the same config and the same 50
threads, the two writers disagree materially on every row.

| assigned | realized `polite` (gpt) | realized `polite` (DeepSeek) |
|---|---|---|
| polite | 35.8% | **52.6%** |
| impolite retained | 86.5% | **89.9%** |

Solving the quota against the wrong matrix is not a small error. G79 records the
precedent: v120b solved against the donor-free matrix, the donor delivered 0.784
instead of 0.384, and `polite_rate` overshot real by +34.7% — **worse than the
−55.5% it started from.**

## The rule this runbook exists to obey

From `tone_realization.py`:

> Refitting it per run against a run's own output would be tuning, and refitting
> it against test-set p-values is forbidden outright (`ORIENTATION.md` §4).

So the matrix may **not** be fitted on the evaluation corpus's own output. G159's
DeepSeek matrix was measured that way and is therefore **evidence that C is
model-dependent, not a shippable table** (G161).

## The pool

The current gpt matrix's polite row rests on `v117_calibration`, which ran over
`camera_product_95_seed907` — the evaluation pool itself. That in-sample overlap is
disclosed in the module docstring rather than hidden, and it is what this pool
removes:

`run_generate.py` resolves its pool by convention and **builds it when the file is
missing**, so an exclusion recorded only inside the file would be silently undone by
the first rebuild — the pool would quietly hold evaluation threads again with no
error and nothing in `run_config.json`. `--seed-pool-exclude` therefore hashes the
held-out set into the pool's own **filename**, so the pool is self-identifying and a
rebuild reproduces the same exclusion:

```
--pool-size 30 --sampling-seed 5150 \
--seed-pool-exclude .../camera_product_150_seed42.json .../camera_product_95_seed907.json
```

resolves to `camera_product_30_seed5150_excl245x6ef9180f.json`:
**30 threads, 1,143 real comments, 0 overlap with either evaluation pool** (245 held
out). Comparable to the 1,059 slots the frozen gpt matrix rests on. `build_seed_pool.py
--exclude-pool` builds the same thing standalone.

## The run

`--tone-quota calibrate` renders a **flat** quota, which is the point: it spreads
every tone class across every stance, so each row of C gets measured instead of
only the cells today's quota happens to visit. It is recorded in `run_config.json`
and in `RUN_EXPERIMENT_FIELDS`, so a calibration artifact can never be mistaken for
a candidate.

Every other arm below is copied verbatim from `v137ds_s36_20260830_v2` so the
matrix is measured on the configuration it will be used with. Planner stays
gpt-5.4-mini; only the Writer is DeepSeek.

```bash
python3 -u generalized_card/scripts/run_generate.py \
  --tag v137ds_calib30_$(date +%Y%m%d)_v1 --domain camera \
  --model gpt-5.4-mini --base-url https://api.openai.com/v1 --api-key-env LLM_API_KEY \
  --writer-model deepseek-v4-flash --writer-base-url https://api.deepseek.com/v1 \
  --writer-api-key-env deepseek_api_key \
  --pool-size 30 --sampling-seed 5150 \
  --seed-pool-exclude artifacts/generalized_card/seed_pools/camera_product_150_seed42.json \
                      artifacts/generalized_card/seed_pools/camera_product_95_seed907.json \
  --domain-profile artifacts/generalized_card/runs/v137ds_s36_20260830_v2/domain_profile.json \
  --max-posts 30 --posts-per-run 1 --start-seed-index 0 \
  --tone-quota calibrate \
  --closing-move measured --development-scope measured --domain-claim selective \
  --downtoner-tag suppress --evaluation-tier measured --final-punctuation measured \
  --length-calibration measured --length-transfer v97 --long-form-layout measured \
  --no-story-scope sequence --opening-move measured --partitive-reference suppress \
  --reddit-typography on --reference-link measured --reference-link-count measured \
  --register-realization measured --reply-sibling-visibility on --rhythm-count measured \
  --route-ledger on --semantic-coverage-nonrepeat on --sentence-rhythm measured \
  --social-contract-coherence on --speaker-identity matched \
  --tone-length-fit conditional --turn-frame adjudicative_only \
  --writer-prompt focused --writer-route-lock own_words --writer-temperature legacy \
  --context-dropout-rate 0.42 --context-jitter-rate 0.32 \
  --post-retry-limit 3 --resume
```

Then score politeness and fit:

```bash
python3 generalized_card/scripts/run_evaluate.py --tag <tag> --metric-parallel 5 --resume
python3 generalized_card/analysis/tone_carrier/fit_tone_matrix.py --tag <tag>
```

Cost, from the DeepSeek N=50 runs at $1.82/thread: **~$55 for 30 threads**, roughly
8-10 hours serial or ~2 hours across 4 shards.

## Reuse the paper run's domain profile — do not let the calibration build its own

A run left to build its own profile excludes **only its own seed pool**. A 30-thread
calibration run would therefore measure C under a profile built from
`reference_threads=544` — 574 minus its own 30, i.e. **including every evaluation
thread** — while the paper run's profile is built from 574 minus its own 95. C would
then be measured under a different configuration from the one it is used in, and on a
corpus the paper run deliberately holds out.

Passing `--domain-profile` at the paper run's own `domain_profile.json` fixes both:
the calibration sees exactly the measured shares the paper run sees. The preflight
prints `reference_threads=` so this is checkable before spending anything.

## After the matrix lands

1. Paste the fitted rows into `REALIZATION_MATRIX_DEEPSEEK` with the same
   `_PROVENANCE` shape the gpt matrix carries — runs, slot count, row counts, fitter.
2. **`POLITE_ASSIGNMENT_CAP` must be re-derived, not reused.** Its current 0.56 is a
   maximin over the two *gpt* matrices, judged on the three reported metrics
   (`analysis/tone_carrier/cap_decision.py`). G159's DeepSeek solve wants **58.2%**
   polite, which is already outside it.
3. Re-solve the quota and check the predicted realized mix against real's
   35.5 / 8.9 / 17.5 / 38.1. G159's evaluation-fitted solve reached L2 0.0005 — a
   compliant matrix should land in the same neighbourhood, and a large disagreement
   is itself the finding.

## What NOT to do

- Do not fit the matrix on any run over `camera_product_95_seed907` or
  `camera_product_150_seed42`.
- Do not ship a `calibrate` artifact as a candidate; its quota is deliberately flat.
- Do not reuse `POLITE_ASSIGNMENT_CAP = 0.56` with a DeepSeek matrix.
