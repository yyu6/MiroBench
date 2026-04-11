# Product Reddit Simulation System — Design Spec

**Date:** 2026-04-11
**Project:** GEO (NeurIPS research)
**Status:** Approved

---

## Overview

A general-purpose CLI system that takes any product JSON dataset, uses an LLM to automatically generate realistic Reddit-style personas appropriate for that product type, runs a multi-agent simulation using MiroFish/OASIS, and exports the resulting discussion as JSON and Markdown. Designed for research reproducibility (NeurIPS).

---

## Goals

- Accept any product JSON file (auto-detect schema) with an optional natural-language hint
- LLM determines persona archetypes appropriate for the product category; user controls total agent count
- Each agent behaves as an independent human: can post, comment, upvote, downvote, follow, search, or do nothing
- All agents share the same Reddit environment and can see all prior posts/comments
- Output: reproducibility record + agent profiles + readable discussion (JSON + Markdown) + raw SQLite for querying
- Single CLI command; designed for research methodology auditability

---

## Non-Goals

- No web UI
- No real Reddit API integration
- No streaming/live output during simulation
- No support for non-Best-Buy product schemas at launch (but architecture allows it)

---

## Architecture

```
product_reddit_sim/          ← Python package (GEO project, uses existing .venv)
├── __init__.py
├── loader.py                ← normalize any product JSON → standard schema
├── analyzer.py              ← LLM call 1: product type + persona archetypes
├── persona_gen.py           ← LLM call 2: generate N full Reddit personas
├── config_builder.py        ← LLM call 3: seed posts; write OASIS config files
├── runner.py                ← subprocess: invoke MiroFish run_reddit_simulation.py
└── exporter.py              ← SQLite → discussion.json + discussion.md

run_discussion.py            ← single CLI entry point (project root)
```

MiroFish is treated as an external dependency. The runner invokes `MiroFish/backend/scripts/run_reddit_simulation.py` as a subprocess so MiroFish's own venv stays independent.

---

## Data Flow

```
products.json + optional --hint
  │
  ▼
loader.py
  Detects schema (tries data["products"], then root array)
  Normalizes each product → {title, brand, price, description,
                              rating, review_count, features[]}
  │
  ▼
analyzer.py  [LLM Call 1]
  Input:  10 sampled products + hint
  Output: product_analysis.json
    {
      product_category,
      key_themes[],
      persona_archetypes: [{name, description, weight}],
      discussion_seed_topics[]
    }
  Saves: full prompt + raw LLM response for reproducibility
  │
  ▼
persona_gen.py  [LLM Call 2]
  Input:  archetypes + weights + N agents + product category
          + full product list (for agent context)
  Output: reddit_profiles.json
    Array of N OASIS-compatible Reddit profiles:
    {user_id, username, name, bio, persona, karma,
     age, gender, mbti, country, profession, interested_topics}
  Agents within same archetype get distinct backgrounds/opinions
  Saves: full prompt + raw LLM response for reproducibility
  │
  ▼
config_builder.py  [LLM Call 3]
  Seed post selection:
    - Count: ~10–15% of total product count (e.g. 50 products → 5–7 posts)
    - Mix: product-specific + topic-based posts
    - Products act as a pool; LLM picks the most discussion-worthy
    - Assigns appropriate persona as poster (e.g. budget post → budget archetype)
  Writes:
    - simulation_config.json (OASIS format: time config, agent configs, event config)
    - run_config.json (CLI args, model name, seed, timestamp — reproducibility record)
  │
  ▼
runner.py
  Invokes: MiroFish/backend/scripts/run_reddit_simulation.py --config ...
  Passes:  --max-rounds, --no-wait
  Output:  reddit_simulation.db (written by OASIS to output dir)
  │
  ▼
exporter.py
  Reads:   reddit_simulation.db (trace table)
  Joins:   with reddit_profiles.json (author metadata)
  Sorts:   chronologically; nests comments under their parent posts
  Writes:  discussion.json, discussion.md
```

---

## CLI Interface

```bash
python run_discussion.py \
    products.json \          # required: path to product JSON
    --agents 50 \            # optional: total number of agents (default: 30)
    --hint "commuters" \     # optional: guide LLM persona/topic generation
    --hours 48 \             # optional: simulated hours (default: 48)
    --rounds 30 \            # optional: max OASIS rounds (default: 30)
    --seed 42 \              # optional: random seed for reproducibility
    --output-dir ./outputs   # optional: output location (default: ./outputs)
```

---

## Output Directory

```
outputs/<product_category>_<timestamp>/
├── run_config.json           ← CLI args, model, seed, timestamps
├── product_analysis.json     ← LLM archetypes + raw prompt/response
├── reddit_profiles.json      ← N agent profiles (OASIS format)
├── simulation_config.json    ← full OASIS simulation config
├── reddit_simulation.db      ← raw OASIS SQLite (queryable)
├── discussion.json           ← structured discussion export
└── discussion.md             ← human-readable Reddit-style thread
```

### discussion.json Schema

```json
{
  "meta": {
    "product_category": "wireless noise-cancelling headphones",
    "hint": "commuters",
    "agent_count": 50,
    "seed": 42,
    "simulated_hours": 48,
    "rounds": 30,
    "run_id": "headphones_20260411_191705"
  },
  "posts": [
    {
      "post_id": 1,
      "author": "MetroANClife",
      "author_karma": 3560,
      "content": "Just switched from Bose QC45 to Sony XM5 for my NYC commute...",
      "timestamp": "2026-04-11T19:03:00",
      "likes": 12,
      "dislikes": 1,
      "comments": [
        {
          "comment_id": 1,
          "author": "SonyFanForever",
          "author_karma": 6540,
          "content": "LDAC codec makes such a difference on the XM5...",
          "timestamp": "2026-04-11T19:47:00",
          "likes": 5,
          "dislikes": 0
        }
      ]
    }
  ]
}
```

### discussion.md Format

```markdown
# r/headphones simulation — wireless noise-cancelling headphones
*Hint: commuters | Agents: 50 | Simulated: 48h | Run: headphones_20260411_191705*

---

## [12↑] Just switched from Bose QC45 to Sony XM5 for my NYC commute...
**u/MetroANClife** (karma: 3,560) · 7:03 PM

Just switched from Bose QC45 to Sony XM5 for my NYC commute...

> **u/SonyFanForever** (karma: 6,540) · 7:47 PM [5↑]
>
> LDAC codec makes such a difference on the XM5...
```

---

## LLM Pipeline Details

### Call 1 — Product Analyzer

- **Model:** from `.env` (`LLM_MODEL_NAME`)
- **Temperature:** 0.3 (consistent analysis)
- **Input:** up to 10 stratified product samples (by price range; fewer if dataset is small) + optional hint
- **Output format:** strict JSON, validated before proceeding
- **Saved to:** `product_analysis.json` with `_prompt` and `_raw_response` fields

### Call 2 — Persona Generator

- **Temperature:** 0.9 (maximum persona diversity)
- **Input:** archetypes + weights + product category + full product list summary
- **Distribution:** weights from Call 1 determine how many agents per archetype; within each archetype, agents get distinct names, backgrounds, opinions, MBTI types, and karma levels
- **Output:** complete JSON array of N profiles, validated against OASIS schema before saving
- **Saved to:** `reddit_profiles.json` with `_generation_prompt` and `_raw_response` appended to `product_analysis.json`

### Call 3 — Seed Post Writer

- **Temperature:** 0.8 (natural writing variety)
- **Seed count:** `min(10, max(3, round(total_products * 0.12)))` — e.g. 20 products → 3 posts, 50 products → 6 posts, capped at 10 regardless of dataset size
- **Mix:** ~60% product-specific ("I just got the [X]..."), ~40% topic-based ("Best [category] under $200?")
- **Assignment:** LLM matches poster persona to post type (budget persona → budget post; power-user persona → premium post)
- **Products as pool:** all normalized products passed as context so agents can reference any of them

---

## Simulation Configuration

| Parameter | Default | CLI Flag |
|-----------|---------|----------|
| Simulated hours | 48 | `--hours` |
| Max OASIS rounds | 30 | `--rounds` |
| Agents per hour min | 5 | hardcoded |
| Agents per hour max | 20 | hardcoded |
| Peak hours (most active) | 18–22 | hardcoded |
| Dead hours (near-zero) | 1–5 | hardcoded |
| LLM concurrency semaphore | 30 | hardcoded |

Activity schedule uses US timezone patterns. Peak multiplier 1.5×, dead zone 0.1×.

---

## Schema Detection (loader.py)

Attempts in order:
1. `data["products"]` — Best Buy scrapfly format
2. Root-level list — generic product array
3. Single object with list value — finds first list field with product-like dicts
4. Raises `ValueError` with helpful message if none match

Normalized product fields extracted:
- Required: `title`
- Optional (with fallbacks): `brand`, `price`, `description`, `rating`, `review_count`, `features`

---

## Reproducibility (NeurIPS)

Every run saves a complete `run_config.json`:

```json
{
  "run_id": "headphones_20260411_191705",
  "input_file": "bestbuy_100_headphones_scrapfly_enriched_001_100.json",
  "input_file_sha256": "abc123...",
  "hint": "commuters",
  "agents": 50,
  "hours": 48,
  "rounds": 30,
  "seed": 42,
  "llm_model": "gpt-4o-mini",
  "llm_base_url": "https://api.openai.com/v1",
  "mirofish_script": "MiroFish/backend/scripts/run_reddit_simulation.py",
  "started_at": "2026-04-11T19:00:00",
  "finished_at": "2026-04-11T19:47:23"
}
```

The `--seed` flag controls: random product sampling, agent distribution across archetypes, and active-agent selection per simulation round.

LLM calls are **not** seeded (LLM non-determinism is inherent), but full prompts and raw responses are saved so runs can be audited and qualitatively compared.

---

## Module Responsibilities

| Module | Inputs | Outputs | LLM? |
|--------|--------|---------|------|
| `loader.py` | raw JSON path | normalized product list | No |
| `analyzer.py` | product sample + hint | archetypes, themes | Yes (Call 1) |
| `persona_gen.py` | archetypes + N + products | N Reddit profiles | Yes (Call 2) |
| `config_builder.py` | profiles + products + archetypes | OASIS config files | Yes (Call 3) |
| `runner.py` | config dir path | reddit_simulation.db | No (subprocess) |
| `exporter.py` | .db + profiles | discussion.json + .md | No |

---

## Dependencies

- Python 3.11 (existing GEO `.venv`)
- `openai>=1.0.0` (already in GEO dependencies)
- `python-dotenv` (already in GEO dependencies)
- MiroFish backend (separate venv, invoked as subprocess)
  - `camel-oasis==0.2.5`, `camel-ai==0.2.78`
- Environment: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME` from `MiroFish/.env`
