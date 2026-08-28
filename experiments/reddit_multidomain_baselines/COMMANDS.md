# Commands

Run these from the GEO repository root.

```bash
# 0. Confirm command options and model/domain names.
./experiments/reddit_multidomain_baselines/run_generate_all.sh --help
./experiments/reddit_multidomain_baselines/run_evaluate_all.sh --help

# 1. Configure credentials (do not put keys in a script or commit them).
export OPENAI_API_KEY='...'
export DEEPSEEK_API_KEY='...'
export GEMINI_API_KEY='...'

# 2. No-API smoke test: prepares the camera seed pool and dry-runs both generators.
./experiments/reddit_multidomain_baselines/run_domain.sh camera \
  --dry-run --models gpt-4o-mini \
  --baselines oasis synthpai --max-seeds 3 --posts-per-run 1

# 3. One domain at a time: all four models × both baselines (recommended for a sequential study).
./experiments/reddit_multidomain_baselines/run_domain.sh laptop

# 4. A single model/baseline within one domain.
./experiments/reddit_multidomain_baselines/run_domain.sh laptop \
  --models gpt-5.4-mini --baselines oasis

# 5. Full generation: 11 domains × 2 baselines × 4 models.
./experiments/reddit_multidomain_baselines/run_generate_all.sh --continue-on-error

# 6. Check generation accounting while or after it runs.
column -s, -t < artifacts/reddit_multidomain_baselines/summary/generation_summary.csv

# 7. Evaluate a single domain.
./experiments/reddit_multidomain_baselines/run_evaluate_domain.sh laptop --device mps

# 8. Run all offline metrics and distribution comparisons.
./experiments/reddit_multidomain_baselines/run_evaluate_all.sh --device mps

# 9. Check all evaluated metric rows.
column -s, -t < artifacts/reddit_multidomain_baselines/summary/evaluation_summary.csv
```

`run_domain.sh <domain>` and `run_evaluate_domain.sh <domain>` pin the job to
one domain. You may additionally supply `--models gpt-5.4-mini` and
`--baselines oasis`. Generation resumes completed jobs by default; use
`--force` only to deliberately regenerate a job.

OASIS records zero-comment seed threads instead of failing the remaining
domain by default. Add `--oasis-min-comments-per-post 1` only for a strict
non-empty-thread run.
