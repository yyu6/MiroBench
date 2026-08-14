# Domain-Neutral CARD Revision Playbook

This is the domain-neutral wording of the operational playbook used by the
paper CARD self-loop controller. Hard decisions remain implemented by the
shared metric rules in code.

## Self-BLEU Goal

Match the generated Self-BLEU distribution to the matched real discussion
distribution. The objective is not to lower Self-BLEU as much as possible.
Over-lowering is also wrong.

## Effective Strategies

- Use best-of-N candidates instead of a single rewrite.
- Generate candidates from multiple lexical strategies for the same comment.
- Score each candidate after inserting it back into the full thread.
- Prefer candidates that reduce the absolute gap to the matched real thread.
- Protect entities, numbers, product names, domain anchors, stance, tone,
  approximate length, and comment function.
- Use high-tail repairs only when q90 or max Self-BLEU is the main issue.
- Use middle-mass repairs when median or q75 remains above real.
- Use shape-safe repairs when p-values fail but mean and high-tail values are
  already close.

## Failure Modes To Avoid

- Fixed round counts do not guarantee success after generator stochasticity.
- Optimizing only the mean can leave KS failures.
- Optimizing only high-tail threads can leave median and q75 too high.
- Aggressive rewrites can damage semantic similarity, story probability, tone,
  and length distribution.
- Prompt-only acceptance is unsafe. The LLM proposes candidates; metric gates
  decide acceptance.

## Candidate Acceptance

Insert each candidate into the current full thread. Accept only when it moves
the target metric toward the matched real thread and all preservation checks
pass. For Self-BLEU, the candidate must reduce the absolute matched-real gap by
the configured tolerance without excessive undershoot.

## Round Acceptance

After each round, run cleanup, full scoring, and matched-seed evaluation.
Accept the round only if the target metric improves and protected metrics do
not incur a disallowed regression.

## Protected Metrics

- self_bertscore_mean_f1
- semantic_mean_cosine
- hard_disagree_rate
- polite_rate
- impolite_rate
- neutral_rate
- length_cv
- avg_depth
- structural_virality
- mean_story_probability
- emotion_entropy

## Stop Rules

- Stop when the target metric passes both MWU and KS tests.
- Roll back a rejected proposal to the most recent accepted collection.
- Stop when the configured round budget is reached.
