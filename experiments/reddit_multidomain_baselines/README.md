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

## Fresh-machine quick start

On a computer that does not already have this repository:

```bash
git clone https://github.com/yyu6/MiroBench.git
cd MiroBench
git switch experiments/reddit-multidomain-baselines
git pull --ff-only

# Full setup: pinned OASIS/MiroFish + SynthPAI, evaluation environment,
# desensitized matched inputs, and the official StanceRel checkpoint.
./experiments/reddit_multidomain_baselines/setup.sh
```

If only the OASIS baseline is needed, skip the separate SynthPAI environment:

```bash
./experiments/reddit_multidomain_baselines/setup.sh --skip-synthpai
```

Setup requires Python 3.11 or 3.12, Git, internet access, and several GB of free disk
space. It creates `.venv_reddit_baselines`, clones exact commits of MiroFish
and SynthPAI, applies the committed compatibility overrides, installs the
desensitized 150-seed input bundle, and downloads the official StanceRel model.
The generation/evaluation wrappers automatically use the project environment,
so activating it is optional.

Verify an existing installation without changing it:

```bash
./experiments/reddit_multidomain_baselines/doctor.sh --skip-synthpai
```

No API keys are committed. Export these once in the shell where jobs run:

```bash
export OPENAI_API_KEY='...'
export DEEPSEEK_API_KEY='...'
export GEMINI_API_KEY='...'
```

Only export keys for the models being run. For example, an OASIS + GPT-4o-mini
job needs only `OPENAI_API_KEY`.

## Generate

Run all 88 generation jobs (11 domains × 2 baselines × 4 models):

```bash
./experiments/reddit_multidomain_baselines/run_generate_all.sh --continue-on-error
```

Run one small no-API smoke test first. The committed portable inputs mean this
works without the original 92MB crawler directory:

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
  laptop --models gpt-4o-mini --baselines oasis

# One model across every domain and both baselines.
./experiments/reddit_multidomain_baselines/run_generate_all.sh \
  --models gpt-4o-mini --continue-on-error

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

After generation, run the eight scorers that produce the four MiroBench metric
families (Structure, Uniformity, Behavior, and Expression) for the real
reference and each successful generated job. The comparison is restricted to
the declared core metrics and reports KS, Mann–Whitney U, Wasserstein distance,
and Cliff's delta:

```bash
./experiments/reddit_multidomain_baselines/run_evaluate_all.sh --device auto
```

For a single domain/model/baseline:

```bash
./experiments/reddit_multidomain_baselines/run_evaluate_domain.sh \
  laptop --models gpt-4o-mini --baselines oasis --device auto
```

Use `--device auto` for a portable command. `mps` is Apple Silicon only;
`cuda` requires a CUDA-enabled PyTorch installation.

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

## Reproducibility and privacy

Raw crawler exports, API keys, generated outputs, caches, and model weights are
not committed. Instead, this branch commits only the fixed matched input bundle
needed by these experiments. Reddit author names are deterministically replaced
with opaque ids, Reddit `u/...` mentions are removed, and local absolute paths
are stripped. `portable_inputs_manifest.json` records SHA-256 checksums for
every bundled file, and setup verifies them before installation.

External code is pinned in `config/external_repositories.json`. Compatibility
overrides are committed under `vendor_overrides/`; setup applies them after
checking out the pinned commits. StanceRel is downloaded from the original
authors' public Google Drive and verified against the benchmark checksum.

When `SynthPAI/.venv/bin/python` exists, generation uses it automatically so
SynthPAI's legacy OpenAI SDK remains isolated from the main environment.
Override it with `--synthpai-python` only when needed.
