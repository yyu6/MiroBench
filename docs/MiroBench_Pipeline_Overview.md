#  Pipeline Overview

## 1. Purpose and scope

MiroBench is a multi-domain benchmark for evaluating how closely simulated
Reddit discussions reproduce real discussion behavior. The benchmark is built
at the **thread level**: a real Reddit root post is selected as a seed, a
simulation system generates a discussion from that seed, and the generated
thread is compared with a matched real-reference thread using the same metric
definitions.

The pipeline has three parts:

```mermaid
flowchart LR
    A[Phase 1<br/>Benchmark Construction] --> B[Phase 2A<br/>Metrics Definition & Construction]
    B --> C[Phase 2B<br/>Evaluation & Statistical Analysis]
    A --> D[Matched real threads<br/>and seed posts]
    D --> C
    C --> E[Per-domain and<br/>cross-domain findings]
```

The benchmark unit is defined as follows:

```text
1 seed = 1 selected real Reddit root post
1 generated thread = that seed post + zero or more generated comments
```

Therefore, for a completed matched-seed experiment, the number of seeds and
the number of generated threads should normally be identical. A thread with
zero comments is still a valid thread consisting only of its root post.

---

## 2. Phase 1: Benchmark Construction

### 2.1 Data collection

Reddit posts and comments are collected by domain-specific crawlers. The raw
landing zone preserves the source response for reproducibility, while all
benchmark-ready and publicly shared data must be produced as a separate,
sanitized derivative.

The collection and preparation flow is:

```text
Crawler output
  → schema normalization
  → deduplication
  → desensitization
  → content and quality filtering
  → domain validation
  → deterministic seed selection
  → matched real-reference construction
```

Each normalized thread contains:

- one root post, including title and body;
- zero or more comments;
- reply-tree structure through parent identifiers;
- non-identifying metadata required by the benchmark, such as domain and
  relative comment depth.

### 2.2 Desensitization and privacy requirements

The benchmark-ready dataset must not expose Reddit usernames or direct source
links. Desensitization should be deterministic so that repeated activity by
the same account can be represented consistently without revealing the
original username.

Required transformations include:

| Source field | Benchmark representation |
|---|---|
| Reddit username | Secret-keyed pseudonym such as `reddit_user_000001` |
| Deleted account | Preserve as `[deleted]` |
| Post/comment ID | Internal benchmark ID |
| Permalink and URL | Remove from release artifacts |
| Reddit fullname | Remove or replace with an internal ID |
| Free-text direct identifiers | Remove, mask, or exclude after privacy review |

A secret-keyed mapping, such as an HMAC-based mapping, is preferable to a plain
hash because Reddit usernames are enumerable and vulnerable to dictionary
lookup. The mapping key and lookup table must never be committed or released.

> **Current implementation status:** the raw source data intentionally retains
> provenance, and some existing real-reference comment artifacts still retain
> original Reddit usernames, IDs, and permalinks. They are **not yet suitable
> for public release**. A sanitized export and privacy audit are required before
> publishing MiroBench data. Generated OASIS/SynthPAI usernames are synthetic,
> but should still be labeled as generated identities.

### 2.3 Filtering

Filtering removes records that cannot support reliable or responsible
evaluation. The benchmark construction stage should:

- remove duplicate Reddit post IDs;
- remove missing, deleted, or unusable root-post content;
- remove deleted/removed comment bodies from metric inputs;
- remove spam, bot-generated content, and crawl failures where detectable;
- apply the benchmark's safety and sensitive-content policy;
- retain reply relationships only when their parent identifiers can be
  normalized safely;
- record zero-comment threads instead of silently discarding them, unless a
  metric-specific inclusion rule requires comments;
- record every filtering decision in a manifest.

For the 11 newly collected domains, the current seed-selection rule requires
at least one fetched real comment. The legacy credit-card seed pool was created
with a different rule and contains one zero-comment seed among the first 150.
This protocol difference must be retained in provenance and harmonized before
a final benchmark freeze if strict cross-domain identity is required.

### 2.4 Domain selection

The current benchmark contains 12 domains:

1. `credit_cards`
2. `camera`
3. `celebrity`
4. `cellphone`
5. `game`
6. `headphones`
7. `health_issue`
8. `laptop`
9. `movies`
10. `news`
11. `sports`
12. `tv_series`

Domains are retained when they have sufficient usable root posts, sufficient
comment coverage, interpretable domain boundaries, and enough data to support
the same target sample size. Domain membership is stored as metadata; it is
not an additional experimental control condition.

### 2.5 Current data scale

The table below reports the current benchmark-construction inventory. A
**comment-eligible thread** is a usable real root thread with at least one
fetched real comment. A **benchmark seed** is a selected root post; every
selected seed is expected to yield one generated thread per evaluated system.

| Domain | Crawled records | Usable root threads | Comment-eligible threads | Benchmark seeds | Target generated threads/system |
|---|---:|---:|---:|---:|---:|
| `credit_cards` | 984 | 960 | 919 | 150 | 150 |
| `camera` | 700 | 700 | 496 | 150 | 150 |
| `celebrity` | 700 | 700 | 570 | 150 | 150 |
| `cellphone` | 700 | 700 | 413 | 150 | 150 |
| `game` | 700 | 700 | 563 | 150 | 150 |
| `headphones` | 700 | 700 | 616 | 150 | 150 |
| `health_issue` | 700 | 700 | 539 | 150 | 150 |
| `laptop` | 700 | 700 | 447 | 150 | 150 |
| `movies` | 700 | 700 | 532 | 150 | 150 |
| `news` | 700 | 700 | 474 | 150 | 150 |
| `sports` | 700 | 700 | 537 | 150 | 150 |
| `tv_series` | 700 | 700 | 460 | 150 | 150 |
| **Total** | **8,684** | **8,660** | **6,566** | **1,800** | **1,800** |

Credit-card counts above use the held-out test split used by the baseline
experiments. Its stored seed pool contains 154 seeds, created as an initial 54
plus 100 additional seeds, but the standardized benchmark uses the first 150.
The complete credit-card train-plus-test source contains 2,664 deduplicated,
usable real threads; it is not all used in the matched benchmark.

### 2.6 Seed selection and matched references

Seed selection must be deterministic and reproducible:

1. apply the documented domain-specific eligibility filter;
2. deduplicate by source post ID;
3. sample with a recorded random seed;
4. freeze the ordered list of selected seed indices;
5. use the same ordered seeds for every model and baseline;
6. construct a real-reference thread set from exactly those root posts;
7. record source counts, selected counts, filtering rules, and checksums in a
   manifest.

This matched design controls for root-topic variation: differences in the
evaluation are attributable to discussion generation rather than different
sets of prompts.

---

## 3. Phase 2A: Metrics Definition & Construction

### 3.1 Metric-selection principles

MiroBench defines four core metric families rather than treating every
individual metric as a separate family:

| Core metric family | Included metrics |
|---|---|
| **Structure** | Length, coefficient of variation (CV), average depth, and structural variability |
| **Uniformity** | Soft BLEU, Soft BERTScore, and semantic cosine |
| **Behavior** | Hard disagree rate, impolite rate, neutral rate, and polite rate |
| **Expression** | Mean story probability and emotion entropy |

Metrics are computed at the thread level first. Dataset-level summaries are
then derived from the distribution of thread-level scores. A higher score is
not universally better: the goal is closeness to the matched real distribution.

The scoring code may emit auxiliary diagnostic columns in addition to these
core metrics. Such columns are retained for debugging and exploratory analysis,
but they are not additional MiroBench metric families unless explicitly added
to the benchmark specification.

### 3.2 Metric definitions

Let a thread contain comments \(C = \{c_1, \ldots, c_n\}\), and let
\(P = \{(i,j): i < j\}\) be all unordered comment pairs.

#### 3.2.1 Structure

| Metric | Measurement and calculation | Interpretation |
|---|---|---|
| **Length** (`length_std`) | Count whitespace-delimited tokens in every comment and compute the population standard deviation of comment lengths within the thread. | Measures how much comment length varies within a discussion. |
| **CV** (`length_cv`) | Divide the standard deviation of comment length by mean comment length: \(CV=\sigma_L/\bar{L}\). | Measures relative length variation while adjusting for a thread's typical comment length. |
| **Average depth** (`avg_depth`) | Construct the parent–reply tree and compute comment depth by breadth-first search, with top-level comments at depth 1. Average the depths of all comments in the thread. | Higher values indicate deeper, more multi-level reply chains. |
| **Structural variability** (`structural_virality`) | Treat parent–reply links as an undirected graph and average the shortest-path distance over connected unordered comment pairs. | Distinguishes shallow or star-shaped conversations from discussions that spread through longer interaction paths. |

The conceptual name used in MiroBench is **structural variability**; the
current implementation and output schema retain the existing field name
`structural_virality`.

#### 3.2.2 Uniformity

| Metric | Measurement and calculation | Interpretation |
|---|---|---|
| **Soft BLEU** (`self_bleu_2`, `self_bleu_3`, `self_bleu_4`) | For each unordered comment pair, compute BLEU in both directions and average them: \(s_{ij}^{(k)}=[BLEU_k(c_i,c_j)+BLEU_k(c_j,c_i)]/2\). Average \(s_{ij}^{(k)}\) over all pairs for n-gram orders 2, 3, and 4. | Higher values indicate greater surface-form uniformity or repeated phrasing. |
| **Soft BERTScore** (`self_bertscore_mean_f1`) | Compute pairwise BERTScore F1 using `microsoft/deberta-xlarge-mnli`, with `roberta-large` fallback, and average F1 over all unordered comment pairs. | Higher values indicate greater contextual or token-level semantic uniformity. |
| **Semantic cosine** (`semantic_mean_cosine`) | Encode comments with `sentence-transformers/all-mpnet-base-v2`, normalize the embeddings, compute cosine similarity for each unordered pair, and take the mean. | Higher values indicate that comments are more semantically similar to one another. |

The current code uses the historical output names **Self-BLEU** and
**Self-BERTScore** for the benchmark concepts named **Soft BLEU** and
**Soft BERTScore** in this overview.

#### 3.2.3 Behavior

| Metric | Measurement and calculation | Interpretation |
|---|---|---|
| **Hard disagree rate** (`hard_disagree_rate`) | Extract parent-comment → reply-comment pairs, classify each pair with the local Stance_Rel RoBERTa model, and divide the number predicted as disagreement by the number of valid scored pairs. | Higher values indicate more explicit disagreement between replies and their parents. |
| **Impolite rate** (`impolite_rate`) | Classify every comment with `Intel/polite-guard` and compute `impolite_count / comment_count`. | Measures the share of comments classified as impolite. |
| **Neutral rate** (`neutral_rate`) | Compute `neutral_count / comment_count` from the politeness classifier. | Measures the share of comments classified as neutral in interaction tone. |
| **Polite rate** (`polite_rate`) | Compute `polite_count / comment_count` from the politeness classifier. | Measures the share of comments classified as polite. |

`somewhat_polite_rate` may still be emitted by the classifier, but it is an
auxiliary diagnostic rather than one of the four core Behavior metrics.

#### 3.2.4 Expression

| Metric | Measurement and calculation | Interpretation |
|---|---|---|
| **Mean story probability** (`mean_story_probability`) | Apply `mariaantoniak/storyseeker` to every comment and average the predicted probability that a comment contains a personal story. | Higher values indicate a stronger presence of personal experiences or narrative expression. |
| **Emotion entropy** (`emotion_entropy`) | Use `SamLowe/roberta-base-go_emotions` to assign dominant emotion labels, form the empirical label distribution \(p_e\), and compute Shannon entropy \(H=-\sum_e p_e\log p_e\). | Higher values indicate greater emotional diversity within the thread. |

Metrics requiring at least two comments, including Soft BLEU, Soft
BERTScore, and semantic cosine, return a documented neutral or empty value for
threads with insufficient pairs. Coverage counts must therefore be reported
alongside metric values.

### 3.3 Metric output contract

Every metric implementation should provide:

- one row per thread with `thread_id`, `comment_count`, and metric values;
- the number of valid comment pairs or scored comments;
- model/checkpoint name and version when a learned metric is used;
- device and fallback information;
- an explicit direction/interpretation note;
- deterministic settings where supported;
- a machine-readable JSON/CSV output and an execution log.

The current metric suite is orchestrated by
`scripts/evaluation/score_sampled_generated_runs.py` and the individual
`scripts/evaluation/score_thread_*.py` implementations.

---

## 4. Phase 2B: Evaluation & Statistical Analysis

### 4.1 Evaluation procedure

For each `(domain, baseline, model)` result with a successful generation
report:

1. load the frozen seed manifest;
2. verify expected seed/thread coverage;
3. score the matched real-reference threads once;
4. score the generated threads using the exact same metric implementations;
5. align real and generated outputs by metric schema;
6. retain one finite numeric value per valid thread and metric;
7. compare the real and generated thread-score distributions;
8. save per-job comparisons and a benchmark-level summary.

Already computed metric files are reused by default. `--force` explicitly
recomputes them. Failed generation jobs and dry runs are excluded from
evaluation.

The primary comparison object is:

```text
(domain, baseline, model, metric)
    real thread-score distribution
        versus
    generated thread-score distribution
```

### 4.2 Statistical analysis

For every selected core numeric metric, the evaluation reports:

| Statistical output | Definition | Purpose |
|---|---|---|
| `real_mean`, `generated_mean` | Arithmetic mean of valid thread scores in each group | Descriptive location |
| Mean difference | \(\bar{x}_{gen}-\bar{x}_{real}\) | Signed direction and magnitude |
| Wasserstein distance | First Wasserstein distance between empirical distributions | Overall distributional separation in metric units; lower is closer |
| KS statistic and p-value | Two-sample Kolmogorov–Smirnov test | Tests whether the two empirical distributions differ in location or shape |
| Mann–Whitney U and p-value | Two-sided rank-based test | Tests for systematic rank/location differences without assuming normality |
| Cliff's delta | \([\#(gen>real)-\#(gen<real)]/(n_{gen}n_{real})\) | Nonparametric signed effect size |

The null hypothesis for both the KS and Mann–Whitney tests is that there is no
distributional difference under the assumptions of the respective test. A
small p-value is evidence against the null; it is not the probability that the
null hypothesis is true and does not measure practical importance.

Statistical conclusions should always combine:

- sample sizes (`real_n`, `generated_n`);
- p-values;
- effect size such as Cliff's delta;
- distributional distance such as Wasserstein distance;
- descriptive means/medians and visualization of the score distributions.

As a reporting convention, absolute Cliff's delta may be labeled approximately
as negligible (`< 0.147`), small (`< 0.33`), medium (`< 0.474`), or large
(`>= 0.474`). These labels are interpretation aids, not thresholds implemented
by the current evaluator.

### 4.3 Multiple comparisons and cross-domain reporting

The current evaluator writes raw KS and Mann–Whitney p-values. Because the
benchmark compares many metrics, models, baselines, and domains, publication
analysis should additionally apply a declared multiple-testing correction,
such as Benjamini–Hochberg false discovery rate control, within a predeclared
family of hypotheses.

Results should be reported at two levels:

1. **Per-domain:** preserve domain-specific effects and failure modes.
2. **Cross-domain:** summarize effect sizes/distances across domains, reporting
   both macro averages and between-domain variability rather than pooling all
   threads as if domains were interchangeable.

Missing metrics and zero-comment threads must be reported as coverage, not
silently dropped. A system should not appear stronger merely because difficult
or empty threads failed to produce scores.

### 4.4 Reproducibility and artifacts

Generation accounting records elapsed time, estimated API cost, request/token
counts, generated runs, generated threads, and recursively counted comments.
Evaluation outputs retain per-thread scores and distribution comparisons.

For the multi-domain runner, the primary artifact layout is:

```text
artifacts/reddit_multidomain_baselines/
├── inputs/seed_pools/<domain>.json
├── inputs/real_reference/<domain>/
├── generation/<baseline>/<model>/<domain>/
│   ├── generated/
│   ├── token_usage.jsonl
│   └── generation_report.json
├── evaluation/<baseline>/<model>/<domain>/
│   ├── generated/revised_generated_thread_scores.csv
│   ├── metric_comparison.csv
│   └── evaluation_manifest.json
└── summary/
    ├── generation_summary.csv
    └── evaluation_summary.csv
```

An exact scoped evaluation can be run as:

```bash
./experiments/reddit_multidomain_baselines/run_evaluate_domain.sh camera \
  --models gpt-4o-mini \
  --baselines oasis \
  --device auto
```

The benchmark release should include frozen manifests, code revision, model and
checkpoint identifiers, metric configuration, random seeds, filtering counts,
privacy-audit status, and an explicit record of any deviations from the common
12-domain protocol.
