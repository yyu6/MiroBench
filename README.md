# GEO Reddit Simulation

> **Active research work: read [`docs/ORIENTATION.md`](docs/ORIENTATION.md) first.**
> The current line of work is synthetic Reddit thread generation under
> [`generalized_card/`](generalized_card), evaluated against 12 thread-level
> metrics. `ORIENTATION.md` states the goal, the method, how to read each
> metric, and the working discipline. The sections below describe the older
> OASIS/MiroFish simulation and judge pipeline, which is not the active path.

GEO is a product-grounded Reddit discussion simulation workspace. The active
repo now focuses on five pieces:

- product scraping scripts
- raw product and discussion datasets
- aligned train/test splits
- Reddit discussion simulation on top of OASIS/MiroFish
- an LLM judge for `real` vs `ai-generated` discussion classification

## Repository Layout

```text
.
├── product_reddit_sim/          # simulation + judge core code
├── run_discussion.py            # main simulation CLI
├── scripts/                     # utility CLIs (splits, judge)
├── scrapers/                    # source-specific scraping code
│   ├── bestbuy/
│   └── bhphoto/
├── data/
│   ├── raw/
│   │   ├── products/
│   │   └── discussions/
│   └── processed/
│       └── splits/
├── artifacts/
│   ├── simulations/
│   └── logs/
├── docs/
│   ├── audits/
│   ├── design/
│   ├── prompts/
│   └── plans/
├── tests/
└── third_party/
    └── MiroFish/
```

## Key Directories

- [`product_reddit_sim/`](product_reddit_sim): product loading, analysis,
  persona generation, config building, vanilla OASIS execution, exporting, and
  judging.
- [`scrapers/bestbuy/`](scrapers/bestbuy) and
  [`scrapers/bhphoto/`](scrapers/bhphoto): kept scraping code for the active
  product sources.
- [`data/raw/products/`](data/raw/products): source product datasets.
- [`data/raw/discussions/credit_cards/`](data/raw/discussions/credit_cards):
  real Reddit credit-card discussions.
- [`data/processed/splits/credit_cards/`](data/processed/splits/credit_cards):
  aligned train/test splits for product descriptions and discussion bundles.
- [`artifacts/simulations/`](artifacts/simulations): generated simulation runs.
- [`artifacts/logs/`](artifacts/logs): runtime logs emitted by the simulator.
- [`third_party/MiroFish/`](third_party/MiroFish): vendored OASIS/MiroFish
  runtime dependency used to execute Reddit simulations.

## Main Workflows

### 1. Run a discussion simulation

```bash
python3 run_discussion.py \
  data/raw/products/bestbuy/headphones/bestbuy_200_headphones_scrapfly_enriched.json \
  --agents 50 \
  --hours 24 \
  --rounds 24 \
  --seed-posts 5
```

Outputs go under [`artifacts/simulations/`](artifacts/simulations).

### 2. Create aligned credit-card splits

```bash
python3 scripts/create_creditcard_splits.py
```

This reads:

- [`data/raw/products/credit_cards/product_descriptions_raw_map.json`](data/raw/products/credit_cards/product_descriptions_raw_map.json)
- [`data/raw/discussions/credit_cards/`](data/raw/discussions/credit_cards)

and writes:

- [`data/processed/splits/credit_cards/`](data/processed/splits/credit_cards)

### 3. Run the binary judge

```bash
python3 scripts/run_binary_judge.py \
  artifacts/simulations/credit_cards_20260415_181234/discussion.json \
  --target-kind generated \
  --threads 5 \
  --comments 6
```

Judge documentation lives in:

- [`docs/design/creditcard_rag_and_judge.md`](docs/design/creditcard_rag_and_judge.md)
- [`docs/prompts/judge_binary_prompt.md`](docs/prompts/judge_binary_prompt.md)

## Environment

The simulation and judge CLIs load model credentials from:

- [`third_party/MiroFish/.env`](third_party/MiroFish/.env)

They also fall back to a legacy `MiroFish/.env` path if it still exists.

## Notes

- `third_party/MiroFish` is the execution backbone; GEO now calls the restored
  vanilla runner directly instead of carrying a separate repo-local runtime
  patch layer.
- The old Stage 1 code path is no longer the active repo surface. Historical
  material remains under [`archive/`](archive).
