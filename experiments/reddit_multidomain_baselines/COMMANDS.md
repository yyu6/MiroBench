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
./experiments/reddit_multidomain_baselines/run_generate_all.sh \
  --dry-run --domains camera --models gpt-4o-mini \
  --baselines oasis synthpai --max-seeds 3 --posts-per-run 1

# 3. Full generation: 11 domains × 2 baselines × 4 models.
./experiments/reddit_multidomain_baselines/run_generate_all.sh --continue-on-error

# 4. Check generation accounting while or after it runs.
column -s, -t < artifacts/reddit_multidomain_baselines/summary/generation_summary.csv

# 5. Run all offline metrics and distribution comparisons.
./experiments/reddit_multidomain_baselines/run_evaluate_all.sh --device mps

# 6. Check all evaluated metric rows.
column -s, -t < artifacts/reddit_multidomain_baselines/summary/evaluation_summary.csv
```

To reduce scope, add any combination of `--domains laptop`,
`--models gpt-5.4-mini`, and `--baselines oasis`. Generation resumes completed
jobs by default. Use `--force` only to deliberately regenerate a job.
