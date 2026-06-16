<div align="center">

# MiroBench

**Benchmarking Realism in Agentic Simulation of Real-world Discussions**  

MiroBench is a benchmark for evaluating whether LLM-generated online discussion threads match real Reddit discussion patterns.

[Website](https://yyu6.github.io/MiroBench/) | [PDF](https://arxiv.org/pdf/2606.14715) | [Code](https://github.com/yyu6/MiroBench) | HuggingFace(coming soon)

</div>

## Overview

MiroBench provides:

- **5 product domains** with real Reddit discussion threads scored on standardized metrics
- **9 scorer families** covering 57 fine-grained metrics across lexical diversity, semantic similarity, toxicity, emotion, politeness, disagreement, narrativity, and thread structure
- **Statistical comparison tools** to measure how closely generated threads match real discussion patterns (MWU test, KS test, Cliff's delta, Wasserstein distance)
- **Product descriptions** for each domain to use as generation context
- **Iterative LLM-driven calibration system** that automatically tunes simulation parameters to close the gap between generated and real discussion distributions

## Domains

| Domain | Real Threads | Products | Description |
|--------|:---:|:---:|---|
| `credit_cards` | 2,653 | 200 | Credit card discussion threads from r/CreditCards |
| `cameras` | 738 | 200 | Digital/mirrorless camera discussions from photography subreddits |
| `cell_phones` | 358 | 200 | Smartphone discussions from phone-related subreddits |
| `headphones` | 256 | 200 | Headphone/earbuds discussions from audio subreddits |
| `laptops` | 307 | 200 | Laptop discussions from computing subreddits |

## Metrics

MiroBench evaluates generated threads across 9 scorer families:

| Scorer | Key Metrics | Description |
|--------|------------|-------------|
| **Disagreement** | `mean_disagree_probability`, `hard_disagree_rate` | Stance classification on parent-reply pairs using RoBERTa |
| **Self-BLEU** | `self_bleu_2`, `self_bleu_3`, `self_bleu_4` | Lexical diversity across comments (lower = more diverse) |
| **Self-BERTScore** | `self_bertscore_mean_f1` | Semantic similarity between comment pairs |
| **Semantic Uniformity** | `semantic_mean_cosine` | Embedding-space similarity via sentence-transformers |
| **StorySeeker** | `mean_story_probability`, `story_rate` | Narrative content detection |
| **GoEmotions** | `emotion_entropy`, `emotion_shift_rate`, `dominant_emotion_share` | Fine-grained emotion classification (28 categories) |
| **Politeness** | `polite_rate`, `impolite_rate`, `neutral_rate` | Politeness/civility classification |
| **Structure** | `max_depth`, `avg_depth`, `avg_branching_factor`, `structural_virality` | Thread tree topology |
| **Detoxify** | `toxicity_mean`, `obscene_mean`, `insult_mean`, `identity_attack_mean` | Multi-dimensional toxicity scoring |

## Installation

```bash
git clone https://github.com/yyu6/MiroBench.git
cd MiroBench
pip install -e .                  # core (scoring + comparison)
pip install -e ".[generation]"    # optional: thread generation via OASIS
```

The `generation` extra adds [`camel-oasis`](https://github.com/camel-ai/oasis)
and [`camel-ai`](https://github.com/camel-ai/camel) (the multi-agent
simulation engine used to produce the leaderboard).

### Dependencies

Core dependencies are installed automatically. Some scorers require additional model downloads (handled automatically on first use via HuggingFace):

- `sentence-transformers/all-mpnet-base-v2` (semantic uniformity)
- `microsoft/deberta-xlarge-mnli` (BERTScore)
- `SamLowe/roberta-base-go_emotions` (emotion classification)
- `Intel/polite-guard` (politeness)
- `mariaantoniak/storyseeker` (narrative detection)

For the disagreement scorer, you need the Stance_Rel model checkpoint. See [Disagreement Setup](#disagreement-scorer-setup) below.

For detoxify scoring, install the detoxify package:
```bash
pip install detoxify
```

## Quick Start

### 1. Generate Discussion Threads

You have two ways to produce the `discussion.json` files MiroBench scores:

**Option A — Use MiroBench's built-in generation pipeline (recommended).**
Install the optional `generation` extra (pulls in `camel-oasis` and
`camel-ai`), copy `.env.example` to `.env` and set `LLM_API_KEY`, then:

```bash
pip install -e ".[generation]"
cp .env.example .env  # then fill in LLM_API_KEY
mirobench generate path/to/products.json \
    --agents 30 --rounds 30 --seed-posts 3 \
    --output-dir artifacts/simulations/ \
    --discussion-backbone vanilla_oasis
```

This runs the same six-stage pipeline used to produce the leaderboard:
*(load → analyse → personas → build OASIS config → run OASIS → export
discussion.json)*. Use `--discussion-backbone geo_patched` to enable
MiroBench's runtime patches (visible-comment snapshot, reply-first guards,
anti-template logic). Add `--overlay best_overlay.json` to apply a
calibration overlay produced by `python -m calibration` (see §
[Calibration System](#calibration-system) below).

**Option B — Use any other simulator** (e.g. directly with [OASIS's
reddit_simulation cookbook](https://docs.oasis.camel-ai.org/cookbooks/reddit_simulation),
your own generator, etc.). Save each thread as a `discussion.json` file in
its own directory:

```
my_generated_threads/
  thread_001/
    discussion.json
  thread_002/
    discussion.json
  ...
```

See [`mirobench/data/example_thread_format.json`](mirobench/data/example_thread_format.json) for the expected JSON schema.

> **Minimum sample size: ≥ 50 generated threads per domain.** MiroBench's
> primary readings are the MWU and KS p-values, which are non-parametric
> rank-based tests. With fewer than ~30 threads the p-values become unreliable
> (the rank-test resolution is too coarse), so a single failed metric can be
> noise rather than signal. **50 threads** is the smallest sample size where a
> similarity ratio like `MWU p > 0.05: 11/16` is statistically meaningful;
> the paper's leaderboard uses 200 per (model × domain).

### 2. Score Your Threads

```bash
mirobench score my_generated_threads/ --device cpu
```

This runs all 9 scorers on each thread and produces `my_generated_threads/thread_scores.csv`.

The output CSV has **one row per thread** and **61 columns** (`thread_id`,
`product`, `comment_count`, then one column per metric across all 9 scorer
families — including the 16 core metrics used by `mirobench compare
--core-only`). For an exact schema reference see
[`mirobench/data/example_thread_scores.csv`](mirobench/data/example_thread_scores.csv)
(5 real threads, all 61 columns; the same file `mirobench compare` accepts
verbatim).

Options:
- `--device cpu|cuda|mps` — device for model inference
- `--force` — re-score threads that already have results
- `--output-prefix NAME` — change the output filename prefix

### 3. Compare Against Real Data

```bash
mirobench compare my_generated_threads/thread_scores.csv \
    --domains credit_cards cameras \
    --core-only \
    --model-name "my-model"
```

This computes statistical comparisons against the real reference data and outputs `mirobench_comparison.csv` with per-metric results.

Options:
- `--domains DOMAIN [DOMAIN ...]` — compare against specific domains (default: all 5)
- `--core-only` — restrict to the 16 core metrics across 5 families (Diversity / Tone / Structure / Content / Toxicity) used in the paper's leaderboard. Recommended for paper-style reporting.
- `--model-name NAME` — label for your model in the output
- `--output PATH` — custom output path

### 4. Interpret Results

The **conclusion of a MiroBench run is a similarity ratio**: of the M metrics
compared, how many have `p > 0.05` against the real Reddit reference. A
metric with `p > 0.05` cannot be statistically distinguished from real Reddit
at α = 0.05, so it counts as similar. Two p-values are reported because they
test different things and answer different questions:

| Tier | Column | Question it answers |
|------|--------|---------------------|
| **Primary** | `mwu_p_value` | Mann–Whitney U two-sided. **p > 0.05 ⇒ medians indistinguishable from real.** |
| **Primary** | `ks_p_value`  | Kolmogorov–Smirnov. **p > 0.05 ⇒ full distribution shape indistinguishable from real.** |
| Secondary | `cliffs_delta` / `abs_cliffs_delta` / `cliffs_delta_interpretation` | Signed effect size in [-1, 1]; categorised `negligible` / `small` / `medium` / `large`. |
| Secondary | `wasserstein` | Earth Mover's Distance (lower = closer to real). |
| Secondary | `quantile_error` | Mean absolute error across the 5 / 10 / … / 95 percentiles. |
| Secondary | `empirical_fail_rate` | Fraction of generated values outside the real distribution's 95% CI. |

The secondary columns only matter for the metrics that **fail** the primary
test — they tell you *how far off* a generated distribution is when it is
already known to be statistically different from real Reddit.

After writing the per-metric CSV, `mirobench compare` prints a terminal
SUMMARY block. The headline of each domain block is two similarity ratios
(MWU and KS) — e.g. *5 out of 12 metrics indistinguishable from real* ⇒ a
5 / 12 similarity at α = 0.05. The |δ| / W₁ averages are reported underneath
as the supplementary "how far off" reading.

Example terminal output (1 model · 2 domains · 50 simulated threads · core-only):

```
SUMMARY  (similarity to real Reddit = % of metrics with p > 0.05)
======================================================================

  credit_cards (16 metrics):
    Similarity (MWU p > 0.05):  11/16   (69%)   <- medians indistinguishable from real
    Similarity (KS  p > 0.05):  10/16   (62%)   <- distribution shapes indistinguishable from real
    When the test rejects, how far off (effect size):
      Avg |Cliff's δ|:  0.118   (8 neg / 5 small / 2 med / 1 large)
      Avg Wasserstein:  0.0312
```

**Reading the result.** A simulator that "passes" MiroBench at domain D is
one whose similarity ratios approach the **real-vs-real noise floor of
≈ 95%** (see §03 of the website). Anything below that floor means some
metric distributions are still statistically distinguishable from real
Reddit; the secondary |δ| / W₁ numbers then quantify how distinguishable.

### Submit to the leaderboard

The public [leaderboard](https://yyu6.github.io/MiroBench/leaderboard.html) is
**generated from the `experiments/` folder** — no HTML editing. Add one folder
of scored CSVs and open a PR:

```
experiments/<model-slug>/
├── meta.json                       # display_name, engine, submitter
└── <domain>/thread_scores.csv      # mirobench score output (≥50 threads)
```

Then regenerate and commit the build artifacts:

```bash
python -m mirobench.leaderboard update   # experiments/ → leaderboard.json → leaderboard.html
python -m mirobench.leaderboard check     # CI runs this; non-zero if stale
```

CI recomputes every cell from your `thread_scores.csv` (pure numpy/scipy, no
GPU) and refreshes the site on merge. Full spec:
[`experiments/README.md`](experiments/README.md).

## Thread Format

Each generated thread must be a JSON file named `discussion.json` with this structure:

```json
{
  "posts": [
    {
      "post_id": 1,
      "author": "username",
      "content": "Post text...",
      "comments": [
        {
          "comment_id": 1,
          "author": "commenter",
          "content": "Reply text...",
          "depth": 0,
          "replies": [
            {
              "comment_id": 2,
              "author": "another_user",
              "content": "Nested reply...",
              "depth": 1,
              "replies": []
            }
          ]
        }
      ]
    }
  ]
}
```

Required fields: `posts[].content`, `posts[].comments[].content`, `comments[].replies`. Other fields (`author`, `likes`, `timestamp`, etc.) are optional but improve scoring fidelity.

## Disagreement Scorer Setup

The disagreement scorer uses the **Stance_Rel** RoBERTa-based stance
classification model released by [Luo et al.][stancerel], which is the only
scorer that requires a manual checkpoint download — the remaining eight
scorers either need no model or pull from HuggingFace automatically.

1. Download the Stance_Rel checkpoint from the original authors' Google Drive:
   <https://drive.google.com/file/d/11YSO_BOpYCDR08FyxjpX3xi7M1O2LmRK/view?usp=sharing>
2. Unzip it into your working directory as `Stance_Rel/RoBERT_rel_1.5e-05/`
   (the folder should contain `pytorch_model.bin`, `config.json`,
   `tokenizer_config.json`, `vocab.json`, `merges.txt`, and `_rgcn.pt`).
3. The scorer will auto-detect this path. To use a different location, pass
   `--model-dir <path>` to `score_thread_disagreement.py`.

If the checkpoint is not present, the disagreement scorer is skipped and the
remaining eight scorers still run.

[stancerel]: https://github.com/LuoXiaoHeics/StanceRel

## Available Commands

```bash
mirobench generate <products.json>           # Generate Reddit threads via OASIS
mirobench score <dir>                        # Score generated threads
mirobench compare <csv>                      # Compare against real references
mirobench domains                            # List available domains with thread counts
mirobench --version                          # Show version
python -m calibration [args]                 # Run iterative calibration pipeline
python -m mirobench.leaderboard update       # Rebuild leaderboard from experiments/
python -m mirobench.leaderboard check        # CI guard: fail if generated files are stale
```

## Data Structure

```
mirobench/data/
  credit_cards/
    reference_scores/          # Real thread scores (train/val/test splits)
      thread_scores.csv
      thread_scores_train.csv
      thread_scores_val.csv
      thread_scores_test.csv
    products/                  # Product descriptions for generation
      product_descriptions.json
    example_threads/           # Example scored threads
  cameras/
    ...
  cell_phones/
    ...
  headphones/
    ...
  laptops/
    ...
  example_thread_format.json   # Reference JSON schema
```

## Calibration System

MiroBench includes an iterative LLM-driven calibration system that automatically tunes simulation parameters (prompt overlays) to minimize distributional gaps between generated and real discussion threads. The calibration loop uses an LLM reasoner to analyze metric-level discrepancies and propose parameter adjustments across iterations.

### How It Works

The calibration pipeline runs in three phases:

1. **Phase 0 — Baseline Evaluation**: Scores vanilla (uncalibrated) simulation output against real data to establish the before-calibration baseline.
2. **Phase 1 — Iterative Calibration Loop**: Over multiple iterations, an LLM reasoner examines per-metric statistical gaps (Cliff's delta, Wasserstein distance, quantile error) between generated and real threads, then proposes prompt overlay adjustments (tone, verbosity, controversy level, structural parameters, etc.). Each iteration generates candidate overlays, runs simulations, scores them, and selects the best-performing candidate.
3. **Phase 2 — Final Evaluation**: Runs multiple simulations with the best overlay found in Phase 1 and computes a comprehensive before/after statistical comparison.

### Calibration CLI

```bash
python -m calibration \
    --real-train-csv <path/to/real_train.csv> \
    --real-val-csv <path/to/real_val.csv> \
    --real-test-csv <path/to/real_test.csv> \
    --few-shot-dir <path/to/real_discussions/> \
    --products-json <path/to/products.json> \
    --output-dir <output_dir> \
    --calibration-model gpt-4o-mini \
    --iterations 12 \
    --candidates 5 \
    --seed-posts 6 \
    --final-sim-runs 9 \
    --min-sim-threads 50 \
    --device cpu
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--real-train-csv` | Real thread scores CSV (train split) — used for qualitative context | required |
| `--real-val-csv` | Real thread scores CSV (validation split) — used for candidate ranking | required |
| `--real-test-csv` | Real thread scores CSV (test split) — used only for final evaluation | required |
| `--few-shot-dir` | Directory with real discussion threads (`.comments.jsonl` files) for few-shot examples | required |
| `--products-json` | Product descriptions JSON file for simulation | required |
| `--calibration-model` | LLM model for the calibration reasoner | `gpt-4o-mini` |
| `--iterations` | Number of calibration iterations in Phase 1 | `12` |
| `--candidates` | Number of candidate overlays per iteration | `5` |
| `--seed-posts` | Number of seed posts per simulation run | `4` |
| `--final-sim-runs` | Number of simulation runs for Phase 2 final evaluation | `12` |
| `--min-sim-threads` | Minimum threads to collect in final evaluation | `50` |
| `--device` | Device for metric scoring (`cpu`, `cuda`, `mps`) | `cpu` |
| `--vanilla-scores-csv` | Pre-existing vanilla baseline scores for before/after comparison | optional |
| `--resume` | Resume a previously interrupted calibration run | `false` |

### Evaluate an Existing Overlay

To skip Phase 1 and directly evaluate a previously found overlay:

```bash
python -m calibration \
    --evaluate-overlay-json <path/to/best_overlay.json> \
    --before-group-eval-json <path/to/before_calibration_group_eval.json> \
    --real-train-csv <real_train.csv> \
    --real-val-csv <real_val.csv> \
    --real-test-csv <real_test.csv> \
    --vanilla-scores-csv <vanilla_scores.csv> \
    --few-shot-dir <real_discussions/> \
    --products-json <products.json> \
    --output-dir <output_dir> \
    --final-sim-runs 9 \
    --seed-posts 6 \
    --min-sim-threads 50
```

### Output

The calibration system produces:

```
output_dir/
  before_calibration/              # Phase 0 baseline results
    before_calibration_group_eval.json
  iterations/                      # Phase 1 per-iteration data
    iter_00/
    iter_01/
    ...
  best_overlay.json                # Best-performing prompt overlay
  after_calibration/               # Phase 2 final evaluation
    after_calibration_group_eval.json
  before_after_improvement_summary.json   # Statistical comparison
  calibration_summary.json                # Full run summary
```

The `before_after_improvement_summary.json` contains per-metric comparisons including Cliff's delta reduction, Wasserstein distance improvement, quantile error changes, and fail rate improvements.

## Acknowledgements

MiroBench is inspired by and builds on prior work in agentic LLM simulation:

- **[OASIS](https://github.com/camel-ai/oasis)** — Open Agent Social Interaction Simulation framework from CAMEL-AI. The multi-agent social-interaction engine that powers our generation pipeline.
- **[MiroFish](https://github.com/666ghj/MiroFish)** — A universal swarm-intelligence prediction engine powered by multi-agent simulation.

We thank the authors of these projects for releasing their work openly.

## Citation

```bibtex
@misc{mirobench2026,
  title={MiroBench: A Benchmark for Evaluating Synthetic Online Product Discussions},
  author={MiroBench Authors},
  year={2026},
  url={https://github.com/yyu6/MiroBench}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
