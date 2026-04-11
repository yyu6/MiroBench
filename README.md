# GEO Stage 1

This repository currently implements Stage 1 with the main focus on `1a` from [`simulation_geo_plan.pdf`](/Users/yaoningyu/Desktop/UIUC/GEO/simulation_geo_plan.pdf):

- `1a` official credit-card marketing descriptions and attributes
- `1b` Reddit discussion data from the official Reddit API

The code is intentionally small and explicit. The current Stage 1a path for official product text no longer scrapes product-page HTML directly. Instead, it sends each `official_product_url` to the OpenAI Responses API as a file URL and writes the model-returned raw description into `meta_description`.

## Layout

- [`data/reference/card_universe_seed.csv`](/Users/yaoningyu/Desktop/UIUC/GEO/data/reference/card_universe_seed.csv): curated known-card hints
- [`data/reference/product_url_overrides.csv`](/Users/yaoningyu/Desktop/UIUC/GEO/data/reference/product_url_overrides.csv): direct official product URL overrides for tricky cards
- [`data/reference/issuer_directories.csv`](/Users/yaoningyu/Desktop/UIUC/GEO/data/reference/issuer_directories.csv): broader live issuer-directory registry for Stage 1a expansion
- [`docs/stage1.md`](/Users/yaoningyu/Desktop/UIUC/GEO/docs/stage1.md): Stage 1 design notes, schemas, and command flow
- [`docs/openai_meta_description_prompt.md`](/Users/yaoningyu/Desktop/UIUC/GEO/docs/openai_meta_description_prompt.md): the exact OpenAI prompts used for `meta_description`
- [`src/geo_stage1/cli.py`](/Users/yaoningyu/Desktop/UIUC/GEO/src/geo_stage1/cli.py): command-line entry point
- [`src/geo_stage1/openai_meta_description_updater.py`](/Users/yaoningyu/Desktop/UIUC/GEO/src/geo_stage1/openai_meta_description_updater.py): OpenAI-based `meta_description` refresh over official product URLs
- [`src/geo_stage1/reddit_collector.py`](/Users/yaoningyu/Desktop/UIUC/GEO/src/geo_stage1/reddit_collector.py): Reddit API collection and coverage filtering
- [`src/geo_stage1/splitter.py`](/Users/yaoningyu/Desktop/UIUC/GEO/src/geo_stage1/splitter.py): deterministic 70/30 train/test split

## Quick Start

1. Create an environment and install dependencies from [`pyproject.toml`](/Users/yaoningyu/Desktop/UIUC/GEO/pyproject.toml).
2. Fill in `OPENAI_API_KEY` and Reddit credentials from [`.env.example`](/Users/yaoningyu/Desktop/UIUC/GEO/.env.example).
3. Install the package in editable mode, or set `PYTHONPATH=src`.
4. Run:

```bash
python3 -m pip install -e .
python3 -m geo_stage1.cli refresh-meta-descriptions-openai --input-csv data/stage1/product_descriptions.csv --output-csv data/stage1/product_descriptions.csv --print-prompts
python3 -m geo_stage1.cli collect-reddit
python3 -m geo_stage1.cli select-cards
python3 -m geo_stage1.cli split-cards
```

Outputs are written under [`data/stage1`](/Users/yaoningyu/Desktop/UIUC/GEO/data/stage1).

## Stage 1a Fields

The OpenAI updater writes:

- `card_name`
- `issuer`
- `official_product_url`
- `meta_description`

## Notes

- The OpenAI updater uses the Responses API with `input_file.file_url` and the row's `official_product_url`.
- The old direct HTML scraping path for official product pages has been removed.
- Reference source URLs are attached for NerdWallet, Bankrate, and Credit Karma so the card universe is easier to audit against the broader market.
- Reddit data uses `praw`, as specified in the plan.
- Cards are only selected if they have comments, not just matching post titles.
- The repo instruction referenced `RTK.md`, but that file is not present in this workspace, so this implementation follows the PDF and visible repo context.
