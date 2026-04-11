# Stage 1 Scope

This repo currently covers only `1a` and `1b`, with the main implementation effort now on a stronger `1a`.

## 1a Product Descriptions

Input:

- `data/stage1/product_descriptions.csv`
- `official_product_url` for each card
- OpenAI Responses API

Process:

- loop over each non-empty `official_product_url`
- send the URL directly to the OpenAI Responses API as an `input_file.file_url`
- prompt the model to read the official page and return the best raw official product description
- write the returned text directly into `meta_description`
- checkpoint updated rows back into the CSV
- record each model output in a JSONL run log

Output:

- `data/stage1/product_descriptions.csv`
- `data/stage1/openai_meta_description_runs.jsonl`

Notes:

- Official product-page HTML fetching for Stage 1a has been removed from the repo.
- The prompt text lives in `docs/openai_meta_description_prompt.md`.

## 1b Reddit Discussion Data

Input:

- selected card universe
- target subreddits:
  - `creditcards`
  - `personalfinance`
  - `churning`

Process:

- search Reddit using card-specific aliases
- keep submissions where the card is plausibly the primary subject
- flatten comments from those submissions
- keep comment rows only when they match the card aliases
- compute per-card coverage statistics

Output:

- `data/stage1/reddit_items.jsonl`
- `data/stage1/reddit_coverage.csv`

## Selection Logic

To satisfy the user's request, the default selector aims for up to `100` cards and keeps only cards with:

- an official product page match
- at least `100` Reddit items total
- at least `25` Reddit comments

This is stricter than the PDF minimum and helps ensure that retained cards genuinely have Reddit discussion.

## Split Logic

Selected cards are split before any later modeling work:

- `70%` train
- `30%` test
- deterministic random seed

Output:

- `data/stage1/card_splits.csv`
