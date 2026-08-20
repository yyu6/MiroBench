# Thread Metric Score Reference

This document describes the raw thread-level metrics exported by the evaluation
suite and merged into:

- `thread_metrics_summary.csv/json`
- downstream `thread_scores.csv`
- downstream `real_simulated_thread_scores_table.csv`

This document does **not** describe the later calibration-only comparison and
ranking statistics such as:

- `quantile_fail_rate`
- `mean_percentile_distance`
- `mean_abs_robust_z`
- `Cliff's delta`
- `Wasserstein distance`
- `quantile_error`
- `empirical_fail_rate`

The table is built by:

- `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/summarize_thread_metrics.py`
- `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/build_real_simulated_thread_scores_table.py`

The metrics fall into five groups:

1. disagreement
2. repetition / semantic uniformity
3. narrative and affect
4. structure
5. toxicity / politeness

## Output Table

Each row is one thread. `dataset` is either `real` or `simulated`.

In the real-discussion pipeline, one thread is effectively one post/root
discussion plus all of its comments.

Core identifiers:

- `dataset`
- `thread_id`
- `comment_count`
- `pair_count`

## 1) Disagreement

### `mean_disagree_probability`

- Unit: thread
- Input unit: parent-reply pairs inside the same thread
- How it is computed:
  - Extract every reply and its direct parent text
  - Run the local Stance_Rel-based scorer
  - Read the model's `disagree_probability`
  - Average over all parent-reply pairs in the thread
- Important implementation note:
  - The local `Stance_Rel` checkpoint in this repo does not include the full
    original graph inference path.
  - The current scorer uses the local RoBERTa stance head plus the available
    graph/user features when possible; missing graph features fall back to a
    zero-feature path.
  - So this metric is the repo's current practical Stance_Rel wrapper, not a
    byte-for-byte reproduction of the original graph-augmented pipeline.
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_disagreement.py`
- Model:
  - local `/Users/yaoningyu/Desktop/UIUC/GEO/Stance_Rel`
- Dependency type:
  - local model

### `hard_disagree_rate`

- Unit: thread
- Input unit: parent-reply pairs inside the same thread
- How it is computed:
  - Use the same Stance_Rel output as above
  - Count how many reply pairs are hard-labeled `disagree`
  - Divide by total pair count in the thread
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_disagreement.py`
- Model:
  - local `/Users/yaoningyu/Desktop/UIUC/GEO/Stance_Rel`
- Dependency type:
  - local model

## 2) Surface / Semantic Repetition

### `self_bleu_2`, `self_bleu_3`, `self_bleu_4`

- Unit: thread
- Input unit: unordered comment pairs inside the same thread
- How it is computed:
  - For every unordered comment pair `(a, b)`, compute symmetric pairwise BLEU:
    - `0.5 * BLEU(a -> b) + 0.5 * BLEU(b -> a)`
  - Aggregate all pairwise BLEU scores inside the thread
  - Export separate thread means for BLEU-2, BLEU-3, BLEU-4
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_self_bleu.py`
- Model:
  - none
- Dependency type:
  - algorithmic only

### `self_bertscore_mean_precision`, `self_bertscore_mean_recall`, `self_bertscore_mean_f1`

- Unit: thread
- Input unit: unordered comment pairs inside the same thread
- How it is computed:
  - For every unordered comment pair, compute BERTScore precision / recall / F1
  - Average the pair scores inside the thread
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_self_bertscore.py`
- Model policy:
  - preferred: `microsoft/deberta-xlarge-mnli`
  - fallback: `roberta-large`
- Dependency type:
  - Hugging Face model via local `bert_score-master`

### Current batch-scoring default

The evaluation suite now requests `microsoft/deberta-xlarge-mnli` by default
for both real and simulated runs. If an existing `self_bertscore_results.json`
was produced with a different requested backbone, the suite automatically
refreshes that file before rebuilding the combined CSV tables.

The output JSON still records both:

- `requested_model_type`
- `model_type`
- `fallback_used`
- `fallback_model_type`
- `bert_hash`

so you can see whether the run actually stayed on DeBERTa or had to fall back
to `roberta-large`.

### `self_bertscore_median_f1`

- Unit: thread
- Input unit: unordered comment pairs inside the same thread
- How it is computed:
  - Median of pairwise BERTScore F1 values within the thread
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_self_bertscore.py`

### `self_bertscore_top_k_mean_f1`

- Unit: thread
- Input unit: unordered comment pairs inside the same thread
- How it is computed:
  - Mean of the top-k largest pairwise BERTScore F1 values
  - In the current export path, `top_k=1`, so this effectively behaves as a max-style summary
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_self_bertscore.py`

### Important note on Self-BERTScore

For BERTScore, we do **not** use a sequence-classification head.

The following Hugging Face pattern is fine for a classifier:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```

But BERTScore is not based on classification logits. It uses token-level hidden
states from the encoder backbone and computes contextual token matching. In this
repo, the correct implementation is through the `bert-score` package in:

- `/Users/yaoningyu/Desktop/UIUC/GEO/bert_score-master`

That package loads the backbone for `microsoft/deberta-xlarge-mnli` internally
and computes BERTScore precision / recall / F1 between comment pairs.

The scorer also writes runtime metadata into `self_bertscore_results.json`,
including the requested/actual model, whether fallback happened, the resolved
device, and the `bert_hash` (which includes the Hugging Face transformers
version used for that run).

### `semantic_mean_cosine`, `semantic_median_cosine`, `semantic_top_k_mean_cosine`, `semantic_p90_cosine`

- Unit: thread
- Input unit: unordered comment pairs inside the same thread
- How it is computed:
  - Embed every comment separately
  - Compute all pairwise cosine similarities within the thread
  - Aggregate with mean / median / top-k mean / p90
- Backend note:
  - The default model is still `sentence-transformers/all-mpnet-base-v2`.
  - If `sentence-transformers` is installed, the scorer uses the
    `sentence-transformers` backend directly.
  - Otherwise it falls back to local `torch + transformers` mean pooling while
    keeping the same embedding model name.
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_semantic_uniformity.py`
- Model:
  - `sentence-transformers/all-mpnet-base-v2`
- Dependency type:
  - Hugging Face / sentence-transformers

## 3) Narrative / Emotion / Personal Experience

### `story_count`, `not_story_count`, `story_rate`, `mean_story_probability`

- Unit: thread
- Input unit: single comments
- How it is computed:
  - Score each comment with StorySeeker
  - Predict `story` vs `not_story`
  - Aggregate counts and rate per thread
  - Also average `story_probability`
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_storyseeker.py`
- Model:
  - `mariaantoniak/storyseeker`
- Dependency type:
  - Hugging Face classifier

### `emotion_entropy`, `emotion_entropy_normalized`

- Unit: thread
- Input unit: single comments
- How it is computed:
  - Score each comment on 28 GoEmotions labels
  - Take the dominant emotion per comment
  - Compute Shannon entropy over the dominant-emotion distribution in the thread
  - Normalize by `log(28)` for the normalized version
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_go_emotions.py`
- Model:
  - `SamLowe/roberta-base-go_emotions`
- Dependency type:
  - Hugging Face classifier

### `avg_labels_per_comment`

- Unit: thread
- Input unit: single comments
- How it is computed:
  - Count how many GoEmotions labels are above threshold `0.5` for each comment
  - Average across comments in the thread
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_go_emotions.py`

### `emotion_shift_rate`

- Unit: thread
- Input unit: single comments
- How it is computed:
  - Sort comments in thread order as exported
  - Compare each comment's dominant emotion to the previous one
  - Report the fraction of adjacent transitions where dominant emotion changes
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_go_emotions.py`

### `dominant_emotion`, `dominant_emotion_share`

- Unit: thread
- Input unit: single comments
- How it is computed:
  - Find the most frequent dominant emotion inside the thread
  - Report the label and its share of comments
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_go_emotions.py`

## 4) Politeness / Toxicity

### `polite_rate`, `somewhat_polite_rate`, `neutral_rate`, `impolite_rate`

- Unit: thread
- Input unit: single comments
- How it is computed:
  - Score each comment with Polite-Guard
  - Predict one of four labels:
    - `polite`
    - `somewhat polite`
    - `neutral`
    - `impolite`
  - Export each label's fraction of total comments in the thread
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_politeness.py`
- Model:
  - `Intel/polite-guard`
- Dependency type:
  - Hugging Face classifier

### `toxicity_mean`, `toxicity_max`, `toxicity_p90`

### `severe_toxicity_mean`, `severe_toxicity_max`, `severe_toxicity_p90`

### `obscene_mean`, `obscene_max`, `obscene_p90`

### `threat_mean`, `threat_max`, `threat_p90`

### `insult_mean`, `insult_max`, `insult_p90`

### `identity_attack_mean`, `identity_attack_max`, `identity_attack_p90`

### `aggression_score_mean`, `aggression_score_max`, `aggression_score_p90`

- Unit: thread
- Input unit: single comments
- How it is computed:
  - Score each comment with Detoxify `unbiased`
  - Keep the six Jigsaw-style dimensions:
    - `toxicity`
    - `severe_toxicity`
    - `obscene`
    - `threat`
    - `insult`
    - `identity_attack`
  - Aggregate each dimension with mean / max / p90 per thread
  - Define per-comment `aggression_score` as the unweighted mean of those six dimensions
  - Aggregate aggression with mean / max / p90 per thread
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_detoxify.py`
- Model:
  - local `/Users/yaoningyu/Desktop/UIUC/GEO/detoxify` checkout, `unbiased` model
- Dependency type:
  - local model package + checkpoint

## 5) Structure

### `length_std`, `length_iqr`, `length_cv`

- Unit: thread
- Input unit: single comments
- How it is computed:
  - Compute whitespace token length for every comment
  - Report:
    - standard deviation
    - interquartile range
    - coefficient of variation
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_structure.py`
- Model:
  - none
- Dependency type:
  - algorithmic only

### `max_depth`

- Unit: thread
- Input unit: thread tree
- How it is computed:
  - Treat top-level comments as depth 1
  - Child depth = parent depth + 1
  - Report maximum depth observed in the thread
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_structure.py`

### `avg_depth`

- Unit: thread
- Input unit: thread tree
- How it is computed:
  - Average comment depth across the thread
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_structure.py`

### `avg_branching_factor`

- Unit: thread
- Input unit: thread tree
- How it is computed:
  - For comments with at least one child, count children
  - Average those child counts across internal nodes only
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_structure.py`

### `structural_virality`

- Unit: thread
- Input unit: thread tree
- How it is computed:
  - Treat the thread as an undirected comment graph
  - Compute the shortest-path distance between every unordered comment pair
  - Average those distances inside the thread
- Implementation:
  - `/Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/score_thread_structure.py`
- Model:
  - none
- Dependency type:
  - algorithmic only

## Metric Suite Runner

To run all thread metrics for one simulated run against the real credit-card
reference set, use:

```bash
python3 /Users/yaoningyu/Desktop/UIUC/GEO/scripts/evaluation/run_full_thread_metric_suite.py \
  --real-dir /Users/yaoningyu/Desktop/UIUC/GEO/data/raw/discussions/credit_cards/american_express_platinum_card \
  --sim-dir /Users/yaoningyu/Desktop/UIUC/GEO/artifacts/simulations/credit_cards_20260420_150514
```

That script writes:

- per-metric JSON outputs
- `thread_metrics_summary.csv/json`
- `real_vs_simulated_thread_metric_comparison.csv/json`
- `simulated_thread_real_percentiles.csv`
- `real_simulated_thread_scores_table.csv/json/md`
