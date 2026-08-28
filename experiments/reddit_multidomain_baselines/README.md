# Multi-domain SynthPAI + OASIS baselines

This folder reproduces matched-root-post baseline experiments for every domain
under `data/reddit_domain_posts 2`:

`camera`, `celebrity`, `cellphone`, `game`, `headphones`, `health_issue`,
`laptop`, `movies`, `news`, `sports`, and `tv_series`.

For each requested domain, the workflow deterministically samples real posts
that have scraped comments (150 by default), uses the same roots for both
baselines, and writes a real-reference version of those threads for evaluation.
Default models are exactly:

- `gemini-2.5-flash`
- `deepseek-v4-flash`
- `gpt-4o-mini`
- `gpt-5.4-mini`

No API keys are committed. Export these once in the shell where jobs run:

```bash
export OPENAI_API_KEY='...'
export DEEPSEEK_API_KEY='...'
export GEMINI_API_KEY='...'
```

## Generate

Run all 88 generation jobs (11 domains × 2 baselines × 4 models):

```bash
./experiments/reddit_multidomain_baselines/run_generate_all.sh --continue-on-error
```

Run one small no-API smoke test first:

```bash
./experiments/reddit_multidomain_baselines/run_generate_all.sh \
  --dry-run --domains camera --models gpt-4o-mini \
  --baselines oasis synthpai --max-seeds 3 --posts-per-run 1
```

Useful scoped jobs:

```bash
# One domain at a time, across all four models and both baselines.
./experiments/reddit_multidomain_baselines/run_domain.sh laptop

# One model/baseline in one domain.
./experiments/reddit_multidomain_baselines/run_domain.sh \
  laptop --models gpt-5.4-mini --baselines oasis

# One model across every domain and both baselines.
./experiments/reddit_multidomain_baselines/run_generate_all.sh \
  --models gpt-5.4-mini --continue-on-error

# Equivalent lower-level command for one domain across all four models/baselines.
./experiments/reddit_multidomain_baselines/run_generate_all.sh \
  --domains laptop --continue-on-error

# Resume is automatic. Use --force only when intentionally regenerating output.
./experiments/reddit_multidomain_baselines/run_generate_all.sh \
  --domains laptop --models gemini-2.5-flash --baselines oasis --force
```

The defaults are 150 seed posts/domain, 5 posts/OASIS run, 50 OASIS agents,
24 simulated hours, and 12 OASIS rounds. All are exposed as CLI flags; inspect
`--help` before changing experimental settings.

OASIS is allowed to produce zero-comment seed threads by default because that
is an observed baseline outcome; those threads and their zero comment counts
remain in the artifacts and accounting instead of aborting the full domain.
Use `--oasis-min-comments-per-post 1` only when a strict non-empty quality gate
is explicitly required.

## Evaluate

After generation, run all nine existing GEO thread metrics for the real
reference and each successful generated job, then compute distributional
comparisons (KS, Mann–Whitney U, Wasserstein, and Cliff's delta):

```bash
./experiments/reddit_multidomain_baselines/run_evaluate_all.sh --device mps
```

For a single domain/model/baseline:

```bash
./experiments/reddit_multidomain_baselines/run_evaluate_domain.sh \
  laptop --models gpt-5.4-mini --baselines oasis --device mps
```

## Outputs and accounting

Everything generated goes under `artifacts/reddit_multidomain_baselines/`:

```text
inputs/seed_pools/<domain>.json                    # fixed roots and provenance
inputs/real_reference/<domain>/                    # same roots + real comments
setup/oasis/<model>/<domain>/                       # reusable OASIS personas/config
generation/<baseline>/<model>/<domain>/generated/  # generator output
generation/.../token_usage.jsonl                   # response-level token metadata
generation/.../generation_report.json              # time, cost, threads, comments
summary/generation_summary.csv                      # one row per generation job
evaluation/<baseline>/<model>/<domain>/             # metrics and comparisons
summary/evaluation_summary.csv                      # all metric comparison rows
```

`generation_report.json` and `generation_summary.csv` include wall-clock time,
request/token counts, estimated USD cost, generated run count, generated thread
count, and recursively counted generated comments. Cost is recalculated from
API response usage metadata. It is therefore an estimate and the provider
invoice is authoritative.

Prices live in [`config/models.json`](config/models.json). Gemini's standard
text rates are from the [Gemini API pricing page](https://ai.google.dev/gemini-api/docs/pricing);
DeepSeek's per-response peak/off-peak rate is selected from the published
[DeepSeek API pricing table](https://api-docs.deepseek.com/quick_start/pricing/);
and GPT-5.4 mini's input/cached-input/output rates are from the official
[OpenAI model page](https://developers.openai.com/api/docs/models/gpt-5.4-mini).
Update this config before a new study if provider pricing changes.

## Requirements

Generation uses the repository's existing `SynthPAI`, `product_reddit_sim`,
and vendored MiroFish/OASIS environments. Evaluation additionally needs the
existing local metric checkpoints/dependencies. Run the smoke test above after
pulling to validate only the generation wiring without incurring API cost.
