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

# 7. Full generation: 12 domains × 2 baselines × 4 models.
./experiments/reddit_multidomain_baselines/run_generate_all.sh --continue-on-error

# 8. Check generation accounting while or after it runs.
column -s, -t < artifacts/reddit_multidomain_baselines/summary/generation_summary.csv

# 9. Run all offline core metrics and distribution comparisons.
./experiments/reddit_multidomain_baselines/run_evaluate_all.sh --device auto

# 10. Check all evaluated metric rows.
column -s, -t < artifacts/reddit_multidomain_baselines/summary/evaluation_summary.csv

# 11. Score the fixed real references and run the 12-domain sanity check.
# This makes no LLM API calls. Learned metric models run locally and are cached.
./experiments/reddit_multidomain_baselines/run_real_sanity_check.sh \
  --device auto --sample-size 150 --repeats 200

# 12. Repeat only the statistical resampling after real scores already exist.
./experiments/reddit_multidomain_baselines/run_real_sanity_check.sh \
  --skip-scoring --sample-size 150 --repeats 200
```

`run_domain.sh <domain>` and `run_evaluate_domain.sh <domain>` pin the job to
one domain. You may additionally supply `--models gpt-4o-mini` and
`--baselines oasis`. Generation resumes completed jobs by default; use
`--force` only to deliberately regenerate a job.

OASIS records zero-comment seed threads instead of failing the remaining
domain by default. Add `--oasis-min-comments-per-post 1` only for a strict
non-empty-thread run.

SynthPAI follows the same default. Add
`--synthpai-min-comments-per-post 1` only when zero-comment threads should stop
the run. For `deepseek-v4-flash` and `gemini-2.5-flash`, SynthPAI disables
thinking automatically to avoid paying for reasoning that is not used in the
exported discussion.

The sanity check draws two independent bootstrap samples of 150 threads from
each domain's fixed 150-thread real reference and repeats this 200 times. Its
CSVs are written under
`artifacts/reddit_multidomain_baselines/summary/real_vs_real_sanity/`; they do
not replace `evaluation_summary.csv`.

The committed summary from the current 12-domain, 200-repeat run contains
28,800 comparisons: MWU pass rate 95.10%, KS pass rate 96.97%, and joint pass
rate 94.34%.
