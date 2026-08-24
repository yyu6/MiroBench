# Credit-Card Simulation RAG And Judge Plan

This document is the design note for two adjacent systems:

- retrieval for simulation-time grounding
- binary judging for `real` vs `ai-generated` discussion classification

It is intentionally higher level than the implementation files. Use the files
below for code-level details:

- core judge implementation:
  [judge.py](../../product_reddit_sim/judge.py)
- judge CLI runner:
  [run_binary_judge.py](../../scripts/run_binary_judge.py)
- exact prompt contract:
  [judge_binary_prompt.md](../prompts/judge_binary_prompt.md)
- aligned credit-card splits:
  [data/processed/splits/credit_cards](../../data/processed/splits/credit_cards)

## RAG For Simulation

Do not stuff every product JSON into every agent prompt. That scales badly and also makes the agent sound more like a summarizer than a Reddit user.

Use a two-stage retrieval path instead:

1. Offline indexing
   - Chunk each product description into small factual units: fees, rewards, intro APR, travel perks, approval profile, common complaints.
   - Keep one metadata record per card: `card_name`, `issuer`, `fee_band`, `reward_type`, `credit_profile`, `chunk_id`.
   - Build a hybrid index:
     - sparse retrieval for exact card names / perk names
     - embedding retrieval for fuzzy use cases like "grocery cashback", "airport lounge", "balance transfer"

2. Per-turn retrieval
   - Build the retrieval query from:
     - the visible post text
     - the visible top comments
     - the agent persona
   - Retrieve only top `k=3..6` factual chunks.
   - Feed a short fact block into the action prompt:
     - relevant product facts
     - maybe one competing card
     - maybe one known drawback

3. Guardrails
   - Keep product facts separate from style instructions.
   - Never dump the full catalog into the prompt.
   - Cache retrieval results per post so later agents can reuse them.
   - If the post mentions no specific card, retrieve by use case and budget, not by the whole dataset.

Recommended runtime pattern:

```text
visible thread -> query builder -> retriever -> top-k fact chunks -> concise fact block -> agent action prompt
```

This is better than inserting all product JSON because:

- token cost stays bounded
- agents only see facts relevant to the thread
- responses stay more local and less "catalog summary"-like
- you can scale to hundreds of products

Implementation status:

- This section is the intended architecture for simulation-time grounding.
- The current judge already uses a lightweight retrieval path for product facts
  and actual reference threads.
- The Reddit simulation runtime does not yet have the full chunked hybrid-RAG
  stack described here.

## Judge LLM Goal

The judge should answer a binary question:

`Is this discussion more likely real human Reddit discussion (1) or AI-generated synthetic discussion (0)?`

It should be critical, not generous.

Inputs:

- target product metadata or retrieved product fact block
- target discussion thread
- optional retrieved reference discussions from the training split of actual Reddit data

Important:

- Use reference discussions only as a distributional anchor.
- Do not ask the judge to do exact similarity matching against one exemplar.
- The judge should focus on discourse realism, not only writing quality.

Judge workflow in practice:

1. Load a target thread from generated output or actual Reddit data.
2. Retrieve a few relevant product facts for that thread.
3. Optionally retrieve a few actual reference threads from the training split.
4. Ask the LLM for a binary label.
5. Save both the label and a short reason for auditing.

## Judge Prompt Template

```text
You are an expert evaluator of online discussion realism.

Your job is to determine whether the TARGET DISCUSSION should receive:
1. `1` = real human Reddit discussion
2. `0` = AI-generated synthetic discussion

Be skeptical. Do not reward a discussion just because it is fluent.

You may use the PRODUCT FACTS and OPTIONAL ACTUAL REFERENCE DISCUSSIONS as background context for what real credit-card conversations usually look like, but you must judge the TARGET DISCUSSION on its own merits.

Evaluate the target discussion on these dimensions:

1. Thread realism
- Does the discussion look like people are responding to a thread, or like isolated standalone opinions?
- Do the comments react to the actual question, prior comments, tradeoffs, or disagreements?

2. Human variation
- Do different commenters sound like distinct people with different knowledge, biases, budgets, and goals?
- Or do they sound uniformly polished, balanced, and helpful?

3. Epistemic realism
- Do commenters show partial knowledge, uncertainty, mistaken assumptions, correction, or overconfidence in believable ways?
- Or do they sound too complete and evenly informed?

4. Product grounding
- Are claims anchored in plausible credit-card facts, use cases, or lived experience?
- Or do they stay generic and interchangeable?

5. Reddit-specific messiness
- Look for normal Reddit traits: short replies, disagreement, nitpicking, sarcasm, narrow personal takes, occasional low-information comments.
- Penalize assistant-like phrasing, generic buyer-guide language, repeated structure, and over-coverage.

6. Synthetic artifact check
- Penalize if many comments:
  - restate the same idea in different words
  - are too evenly distributed in length and tone
  - sound like mini product reviews
  - avoid conflict too neatly
  - are overly on-topic in every turn

Output strict JSON with this schema:
{
  "label": 0 | 1,
  "confidence": 0.0-1.0,
  "overall_verdict": "short paragraph",
  "evidence_for_real": ["..."],
  "evidence_for_generated": ["..."],
  "critical_failures": ["..."],
  "score_breakdown": {
    "thread_realism": 1-5,
    "human_variation": 1-5,
    "epistemic_realism": 1-5,
    "product_grounding": 1-5,
    "reddit_messiness": 1-5,
    "synthetic_artifact_risk": 1-5
  }
}

Scoring rule:
- Higher scores mean more human-like, except synthetic_artifact_risk where higher means more likely generated.
- `label=1` only when the thread is more likely real than generated.
- `label=0` whenever the thread shows stronger evidence of synthetic generation than genuine Reddit interaction.
- If the discussion is mixed, choose the more likely binary label and explain the decisive signals.
```

## Best Training/Eval Setup

Use the split files in `data/processed/splits/credit_cards/` like this:

- Train:
  - actual product descriptions: `product_descriptions_train.json`
  - actual discussions: `discussion_train_manifest.json`
- Test:
  - actual product descriptions: `product_descriptions_test.json`
  - actual discussions: `discussion_test_manifest.json`

Recommended evaluation recipe:

1. Build actual discussion examples from the train split.
2. Generate synthetic discussions for the same train products.
3. Use train only for prompt refinement and few-shot reference selection.
4. Hold out the test split completely for final judge evaluation.

This is "train/test" in the prompt-engineering sense, not weight training. The
current judge is prompt-based, not a fine-tuned classifier.

## Practical Advice

- The judge should classify a whole thread, not isolated comments.
- Keep the target thread length bounded; judge on representative top comments if the thread is huge.
- For few-shot judge prompting, use 2 actual and 2 generated references max.
- Do not let the judge see file paths, run IDs, or other metadata that trivially reveals the label.
- Keep the label mapping fixed everywhere:
  - `1 = real`
  - `0 = ai-generated`
