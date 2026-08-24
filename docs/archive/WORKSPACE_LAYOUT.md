# Workspace Layout

This workspace has been reduced to the active Reddit simulation pipeline plus the datasets and scraper scripts you said you still need.

## Active Paths

- Current Reddit simulation:
  - `run_discussion.py`
  - `product_reddit_sim/`
  - `outputs/smartphones_20260413_173210/`
  - `MiroFish/`

- Best Buy datasets kept:
  - `bestbuy_scraping/outputs/cell_phones/`
  - `bestbuy_scraping/outputs/headphones/`
  - `bestbuy_scraping/outputs/laptops/`

- Best Buy scraper scripts kept:
  - `bestbuy_scraping/scripts/bestbuy_category_to_json_scrapfly.py`
  - `bestbuy_scraping/scripts/bestbuy_resolve_listing_refs_scrapfly.py`
  - `bestbuy_scraping/scripts/bestbuy_enrich_scrapfly.py`
  - `bestbuy_scraping/scripts/bestbuy_enrich_scrapfly_mobile.py`
  - `bestbuy_scraping/scripts/bestbuy_enrich_chrome_live.py`

- B&H camera datasets kept:
  - `bhphoto_scraping/outputs/cameras/`

- B&H scraper scripts kept:
  - `bhphoto_scraping/scripts/bhphoto_category_to_json_chrome.py`
  - `bhphoto_scraping/scripts/bhphoto_chrome_utils.py`
  - `bhphoto_scraping/scripts/bhphoto_enrich_chrome.py`

## Archived Paths

Archived material was moved into:

- `archive/2026-04-13_workspace_cleanup/`

That archive contains:

- older Reddit simulation outputs
- debug captures, tmp state, and browser profiles
- old versioned Best Buy scripts and auxiliary one-off scripts
- legacy GEO stage1/docs/oasis reference material
- cache directories and transient artifacts

## Notes

- The active workspace is intentionally biased toward current use, not full historical completeness.
- If you want a second cleanup pass, the next logical candidates are extra tests and any dataset variants you no longer care about.
