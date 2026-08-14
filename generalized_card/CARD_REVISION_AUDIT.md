# CARD Revision Audit

This audit separates the frozen paper-era pipeline, the current operational
CARD scripts, and generalized extensions. It is the implementation reference
for cross-domain revision runs.

## Historical Lineage

The frozen July snapshots under `artifacts/pipeline_snapshots/` contain three
manually configured Self-BLEU passes and seven manually configured tone passes.
They do not contain one universal controller for all twelve evaluation
metrics. The current `scripts/run_metric_revision_controller.py` and
`scripts/run_tone_revision_controller.py` automate the same local-search and
accepted-collection pattern and add deterministic revision memory.

## Shared Revision Contract

1. Diagnose the generated-minus-matched-real distribution and select a repair
   profile or tone stage.
2. Rank target threads by matched-real deviation. There is no global thread
   cap; profile thresholds decide which threads are selected.
3. Rank candidate comments using the target metric's actual contribution plus
   metric-specific metadata.
4. Ask the model for a strategy-diverse best-of-N slate using full local
   context, prior accepted rewrites, prior failure feedback and preservation
   requirements.
5. Insert each candidate into the current complete thread and score it with
   the same implementation used by evaluation.
6. Keep only candidates that reduce the matched-real gap and pass factual,
   semantic, length, stance and metric-specific gates.
7. Recompute the full 150-thread collection. The metric controller accepts or
   rejects the proposed round and records its protected-metric report.
8. An accepted collection becomes the next input. A rejected collection rolls
   back to the most recent accepted input. Resume reconstructs this boundary
   from controller history.
9. In the extended seven-round chain, CARD metric order is used for ties and
   the cross-metric scheduler prioritizes stages with fewer attempted rounds.
   This prevents one difficult metric from consuming all proposals before the
   other nonpassing metrics are tried. Each stage retains its native CARD
   local-search and acceptance implementation.

## Metric Paths

| Metric group | Candidate or edit score | Supported direction | Status |
|---|---|---|---|
| Self-BLEU | Exact full-thread SacreBLEU after insertion; pair contribution; matched-real gap and undershoot gates | Decrease generated excess | CARD core |
| Self-BERT | Exact affected-pair BERTScore projection and full collection rescore | Decrease generated excess | Generalized extension built from CARD pattern |
| Semantic cosine | Exact MPNet pairwise cosine after insertion | Increase or decrease | Generalized extension |
| Hard disagreement, polite, impolite, neutral | Coupled PoliteGuard and stance vector, Self-BLEU protection, full rescore | Both directions | CARD core tone path |
| Length CV | Exact word-count coefficient of variation after insertion | Increase or decrease | Generalized extension |
| Average depth, structural virality | Exact reply-tree metrics after deterministic reattachment | Increase deficits | Later CARD extension |
| Mean story probability | Exact StorySeeker probability and matched-real thread gap | Decrease generated excess | Later CARD extension |
| Emotion entropy | Exact GoEmotions dominant-label entropy after insertion | Increase or decrease | Generalized extension |

Unsupported reverse directions are monitor-only. This is intentional: adding
stories, semantic duplication or lexical repetition solely to move a metric can
invent evidence or produce visibly artificial text; flattening an already
over-deep reply tree requires a separately validated structural operator.

## Domain Boundary

The generalized backend imports the operational CARD reviser modules. It does
not replace target selection, ranking, budgets, exact candidate scoring,
decision ranking or controller acceptance. It replaces factual entity/quantity
extraction, response salvage and static finance wording. The complete CARD
prompt is rendered before wording adaptation, so stage-specific candidate
families, contribution scores, nearby context and controller feedback remain
present. Self-BLEU additionally exposes repeated 1/2-grams while preserving the
original 3/4-gram diagnostics and exact 1--4-gram evaluator.

The domain profile is learned only from real threads outside the complete seed
set and is frozen before generation. The Planner receives matched-real
structural/surface labels plus dynamically retrieved comments from the non-test
reference bank, then abstracts those comments into domain-neutral semantic
moves. The Writer receives seed and Planner facts, never matched-real or
reference-comment text. Profile provenance stores seed/reference hashes and
source IDs, verifies zero overlap, and the output audit checks copying against
both sources.

## Controller Memory

Memory is deterministic, not learned. Each round stores the strategy,
parameters, candidate totals, accepted/rejected comment and thread outcomes,
failure reasons, target changes, protected changes and accept/rollback result.
A bounded summary is supplied to the next strategy choice and prompt. No
adaptive model training is currently performed.
