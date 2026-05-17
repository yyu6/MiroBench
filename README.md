<div align="center">

# MiroBench

**Benchmarking Realism in Agentic Simulation of Real-world Discussions**  

MiroBench is a benchmark for evaluating whether LLM-generated online discussion threads match real Reddit discussion patterns.

[PDF](https://yyu6.github.io/yaoningyu/files/MiroBench_preprint.pdf) | [Code](https://github.com/yyu6/MiroBench) | HuggingFace(coming soon)

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
pip install -e .
```

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

Generate threads using your method of choice. Each thread should be saved as a `discussion.json` file in its own directory:

```
my_generated_threads/
  thread_001/
    discussion.json
  thread_002/
    discussion.json
  ...
```

See `mirobench/data/example_thread_format.json` for the expected JSON schema.

### 2. Score Your Threads

```bash
mirobench score my_generated_threads/ --device cpu
```

This runs all 9 scorers on each thread and produces `my_generated_threads/thread_scores.csv`.

Options:
- `--device cpu|cuda|mps` — device for model inference
- `--force` — re-score threads that already have results
- `--output-prefix NAME` — change the output filename prefix

### 3. Compare Against Real Data

```bash
mirobench compare my_generated_threads/thread_scores.csv --domains credit_cards cameras
```

This computes statistical comparisons against the real reference data and outputs `mirobench_comparison.csv` with per-metric results.

Options:
- `--domains DOMAIN [DOMAIN ...]` — compare against specific domains (default: all 5)
- `--model-name NAME` — label for your model in the output
- `--output PATH` — custom output path

### 4. Interpret Results

The comparison CSV contains per-metric statistical measures:

| Measure | What It Tells You |
|---------|------------------|
| `mwu_p_value` | Mann-Whitney U test p-value (distribution difference significance) |
| `ks_p_value` | Kolmogorov-Smirnov test p-value (distribution shape difference) |
| `cliffs_delta` | Effect size (-1 to 1, how much distributions differ) |
| `cliffs_delta_interpretation` | `negligible` / `small` / `medium` / `large` |
| `wasserstein` | Earth Mover's Distance (lower = closer to real) |
| `quantile_error` | Mean absolute error across quantiles (lower = better) |
| `empirical_fail_rate` | Fraction of generated values outside the 95% CI of real data |

**Goal:** Metrics closer to the real distribution (lower Wasserstein, lower Cliff's delta, higher p-values) indicate more realistic generated threads.

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

The disagreement scorer uses a RoBERTa-based stance classification model. To set it up:

1. Download the Stance_Rel checkpoint (link TBD / contact authors)
2. Place it in your working directory as `Stance_Rel/RoBERT_rel_1.5e-05/`
3. The scorer will auto-detect and use it

If the model is not available, the disagreement scorer will be skipped and the remaining 8 scorers will still run.

## Available Commands

```bash
mirobench score <dir>                        # Score generated threads
mirobench compare <csv>                      # Compare against real references
mirobench domains                            # List available domains with thread counts
mirobench --version                          # Show version
python -m calibration [args]       # Run calibration pipeline
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

MiroBench is inspired by and builds on prior work in agentic social-media simulation:

- **[OASIS](https://github.com/camel-ai/oasis)** — Open Agent Social Interaction Simulation framework from CAMEL-AI. OASIS provides the underlying multi-agent discussion engine that powers our generation pipeline.
- **[MiroFish](https://github.com/666ghj/MiroFish)** — Provides the runtime patches and prompt scaffolding for product-discussion threads on top of OASIS.

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
