# Commands

Run these from the GEO repository root.

```bash
# 0. New computer only: clone, switch branch, and install everything.
git clone https://github.com/yyu6/MiroBench.git
cd MiroBench
git switch experiments/reddit-multidomain-baselines
git pull --ff-only
./experiments/reddit_multidomain_baselines/setup.sh --skip-synthpai

# 1. Confirm the OASIS/evaluation installation and command options.
./experiments/reddit_multidomain_baselines/doctor.sh --skip-synthpai
./experiments/reddit_multidomain_baselines/run_generate_all.sh --help
./experiments/reddit_multidomain_baselines/run_evaluate_all.sh --help

# 2. Configure credentials (do not put keys in a script or commit them).
export OPENAI_API_KEY='...'
export DEEPSEEK_API_KEY='...'
export GEMINI_API_KEY='...'

# 3. No-API OASIS smoke test.
./experiments/reddit_multidomain_baselines/run_domain.sh camera \
  --dry-run --models gpt-4o-mini \
  --baselines oasis --max-seeds 3 --posts-per-run 1

# 4. One domain, one model, OASIS baseline.
./experiments/reddit_multidomain_baselines/run_domain.sh laptop \
  --models gpt-4o-mini --baselines oasis

# 5. Evaluate only that completed result. Existing metric files are reused.
./experiments/reddit_multidomain_baselines/run_evaluate_domain.sh laptop \
  --models gpt-4o-mini --baselines oasis --device auto

# 6. One domain across all configured models and both baselines.
# Run setup.sh without --skip-synthpai before using this form.
./experiments/reddit_multidomain_baselines/run_domain.sh laptop

# 7. Full generation: 11 domains × 2 baselines × 4 models.
./experiments/reddit_multidomain_baselines/run_generate_all.sh --continue-on-error

# 8. Check generation accounting while or after it runs.
column -s, -t < artifacts/reddit_multidomain_baselines/summary/generation_summary.csv

# 9. Run all offline core metrics and distribution comparisons.
./experiments/reddit_multidomain_baselines/run_evaluate_all.sh --device auto

# 10. Check all evaluated metric rows.
column -s, -t < artifacts/reddit_multidomain_baselines/summary/evaluation_summary.csv
```

`run_domain.sh <domain>` and `run_evaluate_domain.sh <domain>` pin the job to
one domain. You may additionally supply `--models gpt-4o-mini` and
`--baselines oasis`. Generation resumes completed jobs by default; use
`--force` only to deliberately regenerate a job.

OASIS records zero-comment seed threads instead of failing the remaining
domain by default. Add `--oasis-min-comments-per-post 1` only for a strict
non-empty-thread run.
