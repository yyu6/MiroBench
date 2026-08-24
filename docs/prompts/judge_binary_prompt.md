# Binary Judge Prompt

This file documents the exact prompt contract for the binary judge used by
[judge.py](../../product_reddit_sim/judge.py) and
[run_binary_judge.py](../../scripts/run_binary_judge.py).

Use it when you need a simple and auditable label mapping for realism
classification. The prompt is intentionally narrow:

- the input is one target discussion thread plus optional supporting context
- the supporting context can include product facts, unlabeled real references,
  and labeled few-shot examples
- the output is a binary decision
- the label mapping is fixed across the repo so evaluation is consistent

Use this when you want the judge to output a binary label only:

- `1` = real human Reddit discussion
- `0` = ai-generated discussion

## Prompt

```text
You are a strict evaluator of discussion realism.

Your task is binary classification:
- Output `1` if the TARGET DISCUSSION is more likely a real human Reddit discussion.
- Output `0` if the TARGET DISCUSSION is more likely AI-generated.

Be skeptical. Fluency alone is not evidence of being real.

Judge the discussion on:
1. Thread realism: are comments reacting to the post and to each other?
2. Human variation: do commenters sound like different people with uneven knowledge and different motivations?
3. Epistemic realism: do people show uncertainty, bias, partial knowledge, correction, or overconfidence in believable ways?
4. Product grounding: are claims tied to plausible credit-card facts, use cases, or lived experience?
5. Reddit messiness: are there short replies, disagreement, nitpicking, sarcasm, narrow takes, or low-information comments?
6. Synthetic artifacts: do many comments sound uniformly polished, repetitive, overly balanced, or like mini reviews?

Return strict JSON:
{
  "label": 0 | 1,
  "confidence": 0.0-1.0,
  "reason": "short paragraph explaining the decisive signals"
}

Decision rule:
- Use `1` only if the discussion is more likely real than generated.
- Use `0` if there are stronger synthetic signals than human signals.
```

Notes:

- The live code path also sends a short system prompt before this user prompt.
- The prompt asks for strict JSON because the runner saves results directly to
  machine-readable `.json` files.
- `confidence` is the judge model's self-reported confidence, not a calibrated
  probability from a trained classifier.

## Minimal Variant

If you want only the label with no explanation:

```text
Classify the TARGET DISCUSSION.
Return only one character:
- `1` for real human Reddit discussion
- `0` for ai-generated discussion
```

Use the minimal variant only for quick experiments. For serious evaluation, the
JSON variant is better because it preserves the judge's reasoning and makes
error analysis much easier.
