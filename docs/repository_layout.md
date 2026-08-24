# Repository Layout

This document describes the active repository structure after the open-source
cleanup.

## Top-Level Intent

- `product_reddit_sim/`: core Python package for loading products, generating
  personas, building configs, patching OASIS/MiroFish, exporting discussions,
  and running the binary judge.
- `run_discussion.py`: main CLI for end-to-end discussion simulation.
- `scripts/`: maintenance and evaluation CLIs that are useful but not part of
  the core package API.
- `scrapers/`: source-specific data collection scripts.
- `data/`: versioned datasets and processed splits.
- `artifacts/`: generated runtime outputs and logs.
- `third_party/`: vendored runtime dependencies that GEO builds on top of.
- `docs/`: architecture, prompts, audits, and plans.
- `archive/`: old experiments and legacy material kept out of the active path.

## Data Flow

```text
scrapers/ -> data/raw/products/ -> run_discussion.py -> artifacts/simulations/
real reddit bundles -> data/raw/discussions/ -> scripts/create_creditcard_splits.py -> data/processed/splits/
artifacts/simulations/ + data/raw/discussions/ -> scripts/run_binary_judge.py
```

## Active Data Directories

- `data/raw/products/bestbuy/`
- `data/raw/products/bhphoto/`
- `data/raw/products/credit_cards/`
- `data/raw/discussions/credit_cards/`
- `data/processed/splits/credit_cards/`

## Active Runtime Directories

- `artifacts/simulations/`
- `artifacts/logs/`
- `third_party/MiroFish/`

## Historical Material

Anything not part of the active simulation/judge pipeline should go into
`archive/` instead of remaining mixed into the repo root.
