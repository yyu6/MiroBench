# MiroBench

**Benchmarking Realism in Agentic Simulation of Real-world Discussions**

A benchmark for evaluating synthetic online product discussion threads against real Reddit data across 5 consumer product domains.

<!-- Paper link (to be added after acceptance) -->

## Overview

MiroBench provides:

- **5 product domains** with real Reddit discussion threads scored on standardized metrics
- **9 scorer families** covering 57 fine-grained metrics across lexical diversity, semantic similarity, toxicity, emotion, politeness, disagreement, narrativity, and thread structure
- **Statistical comparison tools** to measure how closely generated threads match real discussion patterns (MWU test, KS test, Cliff's delta, Wasserstein distance)
- **Product descriptions** for each domain to use as generation context

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
mirobench score <dir>       # Score generated threads
mirobench compare <csv>     # Compare against real references
mirobench domains           # List available domains with thread counts
mirobench --version         # Show version
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
