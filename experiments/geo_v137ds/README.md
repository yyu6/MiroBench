# GEO v137ds — commands

Planner `gpt-5.4-mini` + Writer `deepseek-v4-flash`. Measured on `camera_product`
at N=150, matched-pair: **PASS 8/12**, and both target metrics pass —
`self_bertscore` d +0.10 / mwu 0.143, `self_bleu_4` d −0.03 / mwu 0.610.
Ledger: `docs/DECISIONS.md` G154–G175.

Run everything from the repo root.

## 0. Confirm you are on the pinned version

```bash
./experiments/geo_v137ds/freeze.sh --verify
```

Prints `working tree matches the frozen version`, or names every file that moved.
A run from a drifted tree is not v137ds. Re-freeze deliberately with
`./experiments/geo_v137ds/freeze.sh`.

## 1. Generate one (domain, planner, writer)

`--model M` points both ends at one model. `--planner P --writer W` splits them;
the pinned v137ds arm is `--planner gpt-5.4-mini --writer deepseek-v4-flash`.

```bash
# 1. DeepSeek-v4-flash as planner AND writer -- 50 ways
HF_HUB_OFFLINE=1 ./experiments/geo_v137ds/run_geo_domain.sh camera \
  --model deepseek-v4-flash --shard-size 3 --max-parallel 50

# 2. gpt-5.4-mini as planner AND writer
HF_HUB_OFFLINE=1 ./experiments/geo_v137ds/run_geo_domain.sh camera \
  --model gpt-5.4-mini --shard-size 5 --max-parallel 10

# 3. gpt-4o-mini as planner AND writer
HF_HUB_OFFLINE=1 ./experiments/geo_v137ds/run_geo_domain.sh camera \
  --model gpt-4o-mini --shard-size 5 --max-parallel 10

# 4. gemini-2.5-flash as planner AND writer
HF_HUB_OFFLINE=1 ./experiments/geo_v137ds/run_geo_domain.sh camera \
  --model gemini-2.5-flash --shard-size 5 --max-parallel 8

# 5. THE PINNED ARM: planner gpt-5.4-mini + writer deepseek-v4-flash -- 50 ways
HF_HUB_OFFLINE=1 ./experiments/geo_v137ds/run_geo_domain.sh camera \
  --planner gpt-5.4-mini --writer deepseek-v4-flash --shard-size 3 --max-parallel 50
```

Add `--dry-run` to any of them to see the plan without spending. Swap `camera`
for `cell_phone`, `headphone` or `laptop`.

**Memory, not the API, is the binding constraint on a first run.** Every shard
reuses the profile the preflight built (`--domain-profile`); without that each
process rebuilds it, loading a sentence-transformer and embedding every
reference thread, and fifty of those at once will take a 24GB machine into swap
until it stops responding. Steady state with the shared profile is ~0.4GB per
shard, so 50 ways needs ~20GB and 20 ways ~8GB. On 24GB start at
`--max-parallel 20`.

Fan-out otherwise follows the **writer**, not the slower of the two ends: the Planner is
called once per thread while the Writer is called once per comment, so a
45-comment thread is ~45 Writer calls to 1 Planner call. The shipped camera
cohort ran arm 5 thirty-eight ways with no OpenAI throttling. Defaults are 50 for
a DeepSeek writer, 10 for OpenAI, 8 for Gemini; `--max-parallel` overrides.

Preflight runs once and alone (it builds the domain profile, and two processes
building it at the same time race), then the shards fan out. Every shard passes
`--resume`, so re-running the same command picks up where it stopped.

Run tags encode the arm: `geo137_<domain>_<arm>_<date>_pNNN`, where the arm is
`dsflash` / `g54m` / `g4om` / `gem25f` for a same-model run and `g54mxdsflash`
for a split one. `eval_geo_domain.sh` and `export_to_multidomain.sh` take the
same `--model` / `--planner` / `--writer` flags and resolve the same prefix.

**Domains this can serve: `camera`, `cell_phone`, `headphone`, `laptop`.**
The other eight `reddit_multidomain_baselines` domains (celebrity, credit_cards,
game, health_issue, movies, news, sports, tv_series) have no entry under
`generalized_card/configs/domains/` and no scored real threads — the script
refuses them by name rather than failing halfway.

Seed pools default to an existing file per domain (camera 150/seed907, headphone
150/seed42, cell_phone and laptop 100/seed42). `run_generate` *rebuilds* a
missing pool and a rebuild does not reproduce the original sample (G165), so the
script stops if the file is gone instead of silently sampling a new cohort.

### Filling a partial cohort

`--seeds "START:COUNT ..."` generates exactly those ranges instead of sweeping
the pool, and `--tag-prefix` names them. Nothing already generated is touched.

```bash
# the 25 seeds gpt-4o-mini never produced
HF_HUB_OFFLINE=1 ./experiments/geo_v137ds/run_geo_domain.sh camera \
  --planner gpt-5.4-mini --writer gpt-4o-mini \
  --seeds "6:9 23:2 55:5 75:5 138:2 143:2" \
  --tag-prefix v1374ofill2_20260901 --max-parallel 6
```

A shard generated under an older core pin cannot be `--resume`d once the pin
changes — `run_generate` refuses on `generator_core_provenance`, correctly.
Generate the gap under a NEW tag prefix instead; `--dedupe` pools old and new
into one cohort.

## 2. Score it — GEO's own matched-pair table

```bash
./experiments/geo_v137ds/eval_geo_domain.sh headphone --writer deepseek-v4-flash
```

Named cohorts live in `experiments/geo_v137ds/cohorts/*.tags`, so a cohort
assembled across many tag conventions survives a reboot:

```bash
./experiments/geo_v137ds/eval_geo_domain.sh camera --cohort camera_flash
./experiments/geo_v137ds/export_to_multidomain.sh camera --writer deepseek-v4-flash --cohort camera_flash
```

This is the test the 8/12 came from: every generated thread against the real
thread it was built from. Existing metric files are reused, so it is safe to
re-run. `--table-only` reprints the table without rescoring.

## 3. Publish into reddit_multidomain_baselines

```bash
./experiments/geo_v137ds/export_to_multidomain.sh headphone --writer deepseek-v4-flash

./experiments/reddit_multidomain_baselines/run_evaluate_domain.sh headphones \
  --models deepseek-v4-flash --baselines geo --device auto

column -s, -t < artifacts/reddit_multidomain_baselines/summary/evaluation_summary.csv
```

The export writes `generation/card/<model>/<domain>/` with a
`generation_report.json` carrying `baseline: "geo"`. That harness's evaluator
selects jobs by report, not by a fixed baseline list, so it scores GEO with the
same code it uses for oasis and synthpai — no change to that harness.

### The two tables are not interchangeable

| | pairs against | answers |
|---|---|---|
| `eval_geo_domain.sh` | the thread each output was built from, under `data/raw/discussions/` | is this cohort indistinguishable from its own source threads — **the 8/12 result** |
| multidomain `metric_comparison.csv` | `inputs/real_reference/<domain>`, a different real corpus | how does GEO compare with oasis and synthpai on a shared reference |

For camera the two real corpora share **zero** posts: the multidomain reference
is r/photography from `data/reddit_domain_posts`, GEO's is product threads from
`data/raw/discussions/camera_product`. Both numbers are valid; quoting the
multidomain one as the matched-pair result is not. The export drops GEO's own
table next to the generated threads as `geo_matched_pair.txt` so they stay
together.

## Already published

| baseline | model | domain | threads | matched-pair |
|---|---|---|---|---|
| card | deepseek-v4-flash | camera | 150 | **8/12** |
| card | gpt-5.4-mini | camera | 150 | 6/12 |

Both were generated before these scripts existed, so they were exported with an
explicit `--tags` list rather than by tag pattern.

## Files

| | |
|---|---|
| `geo_config.sh` | the pinned flag list, the per-domain seed pools, the writer endpoints. Sourced by the others; changing a flag here means a new version name. |
| `run_geo_domain.sh` | preflight, then parallel sharded generation |
| `eval_geo_domain.sh` | score every shard, print the pooled 12-metric table |
| `export_to_multidomain.sh` | publish a cohort as baseline `card`, deduped on source post |
| `freeze.sh` | snapshot / verify every module the result depends on |

The snapshot lives at `artifacts/geo_v137ds_frozen/` — `tree/` holds the
modules, `seed_pools/` the cohorts, `MANIFEST.sha256` the hashes, `VERSION.txt`
the commit and the measured result.

## Known gap

`gpt-4o-mini` on camera finished generation but never ran clean+score, so it has
no `cleaned/` and no result. Its 30 shards are `v1374o_p*_20260901_v1`.
