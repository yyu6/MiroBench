# Submitting to the MiroBench leaderboard

The public leaderboard at <https://yyu6.github.io/MiroBench/leaderboard.html> is
**generated from this folder**. You add one folder of scored CSVs via a pull
request; on merge, CI recomputes the statistics and refreshes the site. You
never edit any HTML.

## Layout

```
experiments/
└── <model-slug>/                 # one folder per model (kebab-case)
    ├── meta.json                 # required — display info
    ├── SUMMARY.md                # optional — human-readable writeup
    └── <domain>/                 # one folder per domain you evaluated
        └── thread_scores.csv     # required — per-thread scores (mirobench score output)
```

`<domain>` must be one of: `credit_cards`, `cameras`, `cell_phones`,
`headphones`, `laptops`. You may submit a single domain or all five.

### `meta.json`

```json
{
  "display_name": "SenseNova 6.7-flash-lite",
  "engine": "OASIS",
  "submitter": "your-github-handle",
  "tier": "community",
  "date": "2026-05-23",
  "link": "https://github.com/yyu6/MiroBench/pull/1"
}
```

| field | required | meaning |
|-------|----------|---------|
| `display_name` | ✅ | model name shown on the board |
| `engine` | ✅ | simulation engine (e.g. `OASIS`) |
| `submitter` | ✅ | your GitHub handle |
| `tier` | – | `community` (default) or `paper` |
| `date`, `link` | – | optional provenance |

## What goes in `thread_scores.csv`

The exact output of `mirobench score` — one row per generated thread, one column
per metric. See [`mirobench/data/example_thread_scores.csv`](../mirobench/data/example_thread_scores.csv)
for the format. Column **order doesn't matter**; columns are read by name.

Requirements (enforced in CI by `mirobench.leaderboard.schema`):

- **≥ 50 usable threads** per domain (the `__summary_mean__` row, if present, is
  ignored). More is better — the paper uses 200 per (model × domain).
- All **16 core metrics** present with numeric values. Missing or empty core
  columns are allowed but **count as a fail** for that metric (the denominator
  stays 16 so every entry is ranked on equal footing). The most common gap is
  `hard_disagree_rate` — it needs the stance/disagreement checkpoint at score
  time; without it that Tone metric is blank and scores as a fail.

The 16 core metrics (5 families):

| Family | Metrics |
|--------|---------|
| Diversity | `self_bleu_4`, `semantic_mean_cosine`, `self_bertscore_mean_f1` |
| Tone | `hard_disagree_rate`, `polite_rate`, `impolite_rate`, `neutral_rate` |
| Structure | `length_cv`, `avg_depth`, `structural_virality` |
| Content | `mean_story_probability`, `emotion_entropy` |
| Toxicity | `toxicity_mean`, `severe_toxicity_mean`, `obscene_mean`, `threat_mean` |

## What NOT to commit

Raw generated threads (`discussion.json`, `threads/`) and per-scorer
intermediates (`*_results.json`) are **git-ignored** — they are large and not
needed to reproduce the board. Keep them locally / link them if you want others
to re-score. A `mirobench_comparison.csv` is welcome as a cross-check but is
**not used** by the board: we always recompute from your `thread_scores.csv`.

## How a cell is computed

For each (model, domain) we run `mirobench.compare` against the real reference
for that domain and count how many of the 16 core metrics have **Mann–Whitney U
`p > 0.05`** (medians indistinguishable from real Reddit). Family columns on the
summary table are the mean Wasserstein `W₁` and mean `|Cliff's δ|` across that
family's metrics, with a `×N` ratio against the real-vs-real noise floor.

## Build it locally before opening a PR

```bash
pip install -e .
python -m mirobench.leaderboard update   # build JSON + render HTML
python -m mirobench.leaderboard check     # exits non-zero if anything is stale
```

`update` regenerates `docs/leaderboard.json` and `docs/leaderboard.html`. Commit
both. CI runs `check` on your PR and will fail if they are out of date.
