# Product Reddit Simulation System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a general-purpose CLI that takes any product JSON, uses an LLM to generate contextual Reddit personas, runs a MiroFish/OASIS multi-agent simulation, and exports the discussion as JSON + Markdown.

**Architecture:** A `product_reddit_sim/` Python package with 6 focused modules (loader → analyzer → persona_gen → config_builder → runner → exporter) wired by a single `run_discussion.py` CLI. MiroFish is invoked as a subprocess; the GEO `.venv` drives the LLM pipeline; OASIS inside MiroFish drives the simulation.

**Tech Stack:** Python 3.11, openai>=1.0.0, python-dotenv (all in GEO `.venv`); MiroFish/OASIS (camel-oasis==0.2.5, camel-ai==0.2.78 in MiroFish venv, called as subprocess); SQLite (OASIS internal); pytest for tests.

---

## File Map

**Create:**
```
product_reddit_sim/__init__.py
product_reddit_sim/loader.py          ← normalize any product JSON → standard schema
product_reddit_sim/analyzer.py        ← LLM Call 1: product category + persona archetypes
product_reddit_sim/persona_gen.py     ← LLM Call 2: generate N full Reddit personas
product_reddit_sim/config_builder.py  ← LLM Call 3: seed posts + write OASIS config files
product_reddit_sim/runner.py          ← invoke MiroFish run_reddit_simulation.py subprocess
product_reddit_sim/exporter.py        ← SQLite → discussion.json + discussion.md
run_discussion.py                     ← single CLI entry point
tests/__init__.py
tests/test_loader.py
tests/test_analyzer.py
tests/test_persona_gen.py
tests/test_config_builder.py
tests/test_exporter.py
```

**Read-only references (do not modify):**
```
MiroFish/backend/scripts/run_reddit_simulation.py  ← invoked as subprocess
MiroFish/.env                                       ← LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
```

---

## Task 1: Package scaffold + `loader.py`

**Files:**
- Create: `product_reddit_sim/__init__.py`
- Create: `product_reddit_sim/loader.py`
- Create: `tests/__init__.py`
- Create: `tests/test_loader.py`

### What `loader.py` does
Reads any product JSON, detects the schema, and returns a list of `NormalizedProduct` dataclasses. Tries `data["products"]` first (Best Buy format), then root list, then first list-valued field. Extracts: `title`, `brand`, `price`, `description`, `rating`, `review_count`, `features`.

- [ ] **Step 1.1: Write failing tests for schema detection and normalization**

Create `tests/test_loader.py`:
```python
import json, os, pytest, tempfile
from product_reddit_sim.loader import load_products, NormalizedProduct

BESTBUY_FIXTURE = {
    "meta": {"scraped_count": 2},
    "products": [
        {
            "title": "Sony WH-1000XM5",
            "brand": "Sony",
            "price": {"currentPrice": 349.99},
            "rating": 4.8,
            "review_count": 1200,
            "page_description": "Best ANC headphones.",
            "feature_entries": [
                {"feature": "Industry-leading noise cancellation"},
                {"feature": "30-hour battery life"},
            ],
        },
        {
            "title": "JLab GO Air POP",
            "brand": "JLab",
            "price": {"currentPrice": 24.99},
            "rating": 4.2,
            "review_count": 5000,
            "description": "Budget earbuds.",
            "feature_entries": [],
        },
    ],
}

ROOT_LIST_FIXTURE = [
    {"title": "Product A", "price": 99.99, "brand": "BrandX"},
]

def _write(tmp, data):
    p = os.path.join(tmp, "products.json")
    with open(p, "w") as f:
        json.dump(data, f)
    return p


def test_load_bestbuy_format():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, BESTBUY_FIXTURE)
        products = load_products(path)
    assert len(products) == 2
    assert isinstance(products[0], NormalizedProduct)


def test_normalizes_nested_price():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, BESTBUY_FIXTURE)
        products = load_products(path)
    assert products[0].price == 349.99


def test_normalizes_flat_price():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, ROOT_LIST_FIXTURE)
        products = load_products(path)
    assert products[0].price == 99.99


def test_extracts_features():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, BESTBUY_FIXTURE)
        products = load_products(path)
    assert "Industry-leading noise cancellation" in products[0].features


def test_root_list_schema():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, ROOT_LIST_FIXTURE)
        products = load_products(path)
    assert len(products) == 1
    assert products[0].title == "Product A"


def test_unknown_schema_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, {"foo": "bar", "baz": 42})
        with pytest.raises(ValueError, match="Cannot detect product schema"):
            load_products(path)


def test_description_fallback_order():
    """page_description preferred over description."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, BESTBUY_FIXTURE)
        products = load_products(path)
    assert products[0].description == "Best ANC headphones."


def test_description_truncated_at_500():
    long_desc = "x" * 600
    fixture = {"products": [{"title": "T", "page_description": long_desc}]}
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, fixture)
        products = load_products(path)
    assert len(products[0].description) <= 503  # 500 chars + "..."
```

- [ ] **Step 1.2: Run tests — confirm they all fail**

```bash
cd /Users/yaoningyu/Desktop/UIUC/GEO
source .venv/bin/activate
pytest tests/test_loader.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'product_reddit_sim'`

- [ ] **Step 1.3: Create package scaffold**

Create `product_reddit_sim/__init__.py`:
```python
```
(empty file)

Create `tests/__init__.py`:
```python
```
(empty file)

- [ ] **Step 1.4: Implement `loader.py`**

Create `product_reddit_sim/loader.py`:
```python
"""Normalize any product JSON into a standard list of NormalizedProduct."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NormalizedProduct:
    title: str
    brand: Optional[str] = None
    price: Optional[float] = None
    description: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    features: list[str] = field(default_factory=list)


def load_products(path: str) -> list[NormalizedProduct]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw_list = _detect_schema(data)
    return [_normalize(p) for p in raw_list]


def _detect_schema(data) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "products" in data and isinstance(data["products"], list):
            return data["products"]
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    raise ValueError(
        "Cannot detect product schema. Expected a JSON array or an object "
        "with a 'products' key. Got: " + str(type(data))
    )


def _normalize(raw: dict) -> NormalizedProduct:
    # Price: Best Buy stores {"currentPrice": float}, others store float directly
    price_raw = raw.get("price")
    if isinstance(price_raw, dict):
        price = price_raw.get("currentPrice") or price_raw.get("regularPrice")
    elif isinstance(price_raw, (int, float)):
        price = float(price_raw)
    else:
        price = None

    # Description: prefer richer fields
    description = (
        raw.get("page_description")
        or raw.get("full_description")
        or raw.get("description")
        or ""
    )
    if len(description) > 500:
        description = description[:500] + "..."

    # Features: Best Buy uses feature_entries list of dicts; others may use "features" list
    features: list[str] = []
    for entry in raw.get("feature_entries", []):
        if isinstance(entry, dict) and entry.get("feature"):
            features.append(entry["feature"])
        elif isinstance(entry, str):
            features.append(entry)
    if not features:
        raw_features = raw.get("features", [])
        if isinstance(raw_features, list):
            features = [str(f) for f in raw_features if f]

    return NormalizedProduct(
        title=raw.get("title", "Unknown Product"),
        brand=raw.get("brand"),
        price=price,
        description=description,
        rating=raw.get("rating"),
        review_count=raw.get("review_count"),
        features=features[:6],
    )
```

- [ ] **Step 1.5: Run tests — confirm they all pass**

```bash
pytest tests/test_loader.py -v
```
Expected: 8 tests pass.

- [ ] **Step 1.6: Commit**

```bash
git add product_reddit_sim/ tests/
git commit -m "feat: add product_reddit_sim package scaffold and loader"
```

---

## Task 2: `analyzer.py` — LLM Call 1

**Files:**
- Create: `product_reddit_sim/analyzer.py`
- Create: `tests/test_analyzer.py`

### What `analyzer.py` does
Samples up to 10 products stratified by price, sends them to the LLM with an optional hint, and returns a `ProductAnalysis` with: `product_category`, `key_themes`, `persona_archetypes` (each with `name`, `description`, `weight`), and `discussion_seed_topics`. Saves the full prompt and raw LLM response for reproducibility.

- [ ] **Step 2.1: Write failing tests**

Create `tests/test_analyzer.py`:
```python
import json
from unittest.mock import MagicMock, patch
from product_reddit_sim.loader import NormalizedProduct
from product_reddit_sim.analyzer import (
    analyze_products,
    ProductAnalysis,
    PersonaArchetype,
    _stratified_sample,
)

MOCK_RESPONSE = {
    "product_category": "wireless headphones",
    "key_themes": ["ANC quality", "battery life", "value"],
    "persona_archetypes": [
        {"name": "Audiophile", "description": "Cares about sound quality.", "weight": 0.30},
        {"name": "Commuter", "description": "Needs ANC for transit.", "weight": 0.40},
        {"name": "Budget Hunter", "description": "Wants value.", "weight": 0.30},
    ],
    "discussion_seed_topics": ["Sony vs Bose ANC?", "Best under $100?"],
}

def _make_products(n=15):
    return [
        NormalizedProduct(title=f"Product {i}", price=float(i * 20), brand="BrandX")
        for i in range(1, n + 1)
    ]

def _mock_client(response_dict):
    client = MagicMock()
    completion = MagicMock()
    completion.choices[0].message.content = json.dumps(response_dict)
    client.chat.completions.create.return_value = completion
    return client


def test_returns_product_analysis():
    client = _mock_client(MOCK_RESPONSE)
    result = analyze_products(_make_products(), hint=None, client=client, model="gpt-4o-mini", seed=42)
    assert isinstance(result, ProductAnalysis)
    assert result.product_category == "wireless headphones"


def test_archetypes_parsed():
    client = _mock_client(MOCK_RESPONSE)
    result = analyze_products(_make_products(), hint=None, client=client, model="gpt-4o-mini", seed=42)
    assert len(result.persona_archetypes) == 3
    assert isinstance(result.persona_archetypes[0], PersonaArchetype)
    assert result.persona_archetypes[0].name == "Audiophile"


def test_hint_included_in_prompt():
    client = _mock_client(MOCK_RESPONSE)
    analyze_products(_make_products(), hint="for commuters", client=client, model="gpt-4o-mini", seed=42)
    call_args = client.chat.completions.create.call_args
    prompt = call_args[1]["messages"][0]["content"]
    assert "for commuters" in prompt


def test_prompt_saved_on_result():
    client = _mock_client(MOCK_RESPONSE)
    result = analyze_products(_make_products(), hint="test", client=client, model="gpt-4o-mini", seed=42)
    assert len(result._prompt) > 0
    assert len(result._raw_response) > 0


def test_stratified_sample_returns_at_most_n():
    products = _make_products(20)
    sample = _stratified_sample(products, n=10, rng=__import__("random").Random(42))
    assert len(sample) <= 10


def test_stratified_sample_spans_price_range():
    products = _make_products(20)  # prices 20..400
    sample = _stratified_sample(products, n=10, rng=__import__("random").Random(42))
    prices = [p.price for p in sample if p.price]
    assert min(prices) < 100
    assert max(prices) > 300


def test_weights_sum_validated():
    bad_response = dict(MOCK_RESPONSE)
    bad_response["persona_archetypes"] = [
        {"name": "A", "description": "x", "weight": 0.5},
        {"name": "B", "description": "y", "weight": 0.8},  # sum > 1
    ]
    client = _mock_client(bad_response)
    # Should not crash — normalize weights internally
    result = analyze_products(_make_products(), hint=None, client=client, model="gpt-4o-mini", seed=42)
    total = sum(a.weight for a in result.persona_archetypes)
    assert abs(total - 1.0) < 0.01
```

- [ ] **Step 2.2: Run tests — confirm they fail**

```bash
pytest tests/test_analyzer.py -v 2>&1 | head -20
```
Expected: `ImportError: cannot import name 'analyze_products'`

- [ ] **Step 2.3: Implement `analyzer.py`**

Create `product_reddit_sim/analyzer.py`:
```python
"""LLM Call 1: analyze product sample → product category + persona archetypes."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from .loader import NormalizedProduct


@dataclass
class PersonaArchetype:
    name: str
    description: str
    weight: float


@dataclass
class ProductAnalysis:
    product_category: str
    key_themes: list[str]
    persona_archetypes: list[PersonaArchetype]
    discussion_seed_topics: list[str]
    _prompt: str = field(default="", repr=False)
    _raw_response: str = field(default="", repr=False)


def analyze_products(
    products: list[NormalizedProduct],
    hint: Optional[str],
    client: OpenAI,
    model: str,
    seed: int,
) -> ProductAnalysis:
    rng = random.Random(seed)
    sample = _stratified_sample(products, n=10, rng=rng)
    prompt = _build_prompt(sample, hint)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)

    archetypes = [
        PersonaArchetype(
            name=a["name"],
            description=a["description"],
            weight=float(a["weight"]),
        )
        for a in data["persona_archetypes"]
    ]
    archetypes = _normalize_weights(archetypes)

    return ProductAnalysis(
        product_category=data["product_category"],
        key_themes=data.get("key_themes", []),
        persona_archetypes=archetypes,
        discussion_seed_topics=data.get("discussion_seed_topics", []),
        _prompt=prompt,
        _raw_response=raw,
    )


def _stratified_sample(
    products: list[NormalizedProduct], n: int, rng: random.Random
) -> list[NormalizedProduct]:
    priced = sorted([p for p in products if p.price is not None], key=lambda p: p.price)
    unpriced = [p for p in products if p.price is None]

    if len(priced) >= n:
        step = len(priced) / n
        return [priced[int(i * step)] for i in range(n)]

    combined = priced + unpriced
    rng.shuffle(combined)
    return combined[:n]


def _normalize_weights(archetypes: list[PersonaArchetype]) -> list[PersonaArchetype]:
    total = sum(a.weight for a in archetypes)
    if total <= 0:
        equal = 1.0 / len(archetypes)
        return [PersonaArchetype(a.name, a.description, equal) for a in archetypes]
    return [PersonaArchetype(a.name, a.description, a.weight / total) for a in archetypes]


def _build_prompt(sample: list[NormalizedProduct], hint: Optional[str]) -> str:
    product_lines = "\n".join(
        f"- {p.title} | Brand: {p.brand} | Price: ${p.price} | "
        f"Rating: {p.rating}/5 | {(p.description or '')[:120]}"
        for p in sample
    )
    hint_line = f"\nAdditional context from researcher: {hint}" if hint else ""

    return f"""You are analyzing a product dataset to design a realistic Reddit discussion simulation for NeurIPS academic research.

Products in this dataset (representative sample):
{product_lines}
{hint_line}

Analyze these products and return a JSON object with EXACTLY this structure:
{{
  "product_category": "concise category name (e.g. 'wireless noise-cancelling headphones')",
  "key_themes": ["theme1", "theme2", "theme3", "theme4"],
  "persona_archetypes": [
    {{
      "name": "archetype name",
      "description": "2-3 sentences: who this person is, their motivations, how they discuss this product category on Reddit",
      "weight": 0.25
    }}
  ],
  "discussion_seed_topics": ["Natural Reddit post title 1", "Natural Reddit post title 2"]
}}

Requirements:
- 4-8 key_themes covering the main debates in this product category
- 4-8 persona_archetypes reflecting REAL Reddit communities that discuss this category
- Weights must sum to 1.0 and reflect realistic community composition
- 5-8 discussion_seed_topics that sound like genuine Reddit post titles (not marketing copy)
- Archetypes should span different expertise levels, use cases, budgets, and attitudes"""
```

- [ ] **Step 2.4: Run tests — confirm they pass**

```bash
pytest tests/test_analyzer.py -v
```
Expected: 7 tests pass.

- [ ] **Step 2.5: Commit**

```bash
git add product_reddit_sim/analyzer.py tests/test_analyzer.py
git commit -m "feat: add analyzer LLM call 1 - product category and persona archetypes"
```

---

## Task 3: `persona_gen.py` — LLM Call 2

**Files:**
- Create: `product_reddit_sim/persona_gen.py`
- Create: `tests/test_persona_gen.py`

### What `persona_gen.py` does
Distributes N agents across archetypes by weight, sends archetypes + full product list to the LLM, and returns a list of OASIS-compatible Reddit profile dicts. Required OASIS fields: `user_id`, `username`, `name`, `bio`, `persona`, `karma`, `created_at`. Optional but included: `age`, `gender`, `mbti`, `country`, `profession`, `interested_topics`.

- [ ] **Step 3.1: Write failing tests**

Create `tests/test_persona_gen.py`:
```python
import json
from unittest.mock import MagicMock
from product_reddit_sim.loader import NormalizedProduct
from product_reddit_sim.analyzer import PersonaArchetype, ProductAnalysis
from product_reddit_sim.persona_gen import generate_personas, _distribute_agents

ARCHETYPES = [
    PersonaArchetype("Audiophile", "Sound quality obsessed.", 0.30),
    PersonaArchetype("Commuter", "Needs ANC.", 0.40),
    PersonaArchetype("Budget Hunter", "Seeks value.", 0.30),
]

ANALYSIS = ProductAnalysis(
    product_category="wireless headphones",
    key_themes=["ANC", "battery"],
    persona_archetypes=ARCHETYPES,
    discussion_seed_topics=["Best ANC?"],
)

def _mock_client(n):
    profiles = [
        {
            "user_id": i + 1,
            "username": f"user_{i}",
            "name": f"Person {i}",
            "bio": "Short bio.",
            "persona": "Detailed persona description here.",
            "karma": 1000 * (i + 1),
            "age": 25 + i,
            "gender": "male",
            "mbti": "INTJ",
            "country": "USA",
            "profession": "Engineer",
            "interested_topics": ["headphones"],
            "archetype": ARCHETYPES[i % 3].name,
        }
        for i in range(n)
    ]
    client = MagicMock()
    completion = MagicMock()
    completion.choices[0].message.content = json.dumps({"personas": profiles})
    client.chat.completions.create.return_value = completion
    return client


def test_returns_correct_count():
    client = _mock_client(10)
    profiles, _, _ = generate_personas(ANALYSIS, n_agents=10,
                                        products=[], client=client,
                                        model="gpt-4o-mini", seed=42)
    assert len(profiles) == 10


def test_user_ids_are_sequential():
    client = _mock_client(5)
    profiles, _, _ = generate_personas(ANALYSIS, n_agents=5,
                                        products=[], client=client,
                                        model="gpt-4o-mini", seed=42)
    assert [p["user_id"] for p in profiles] == [1, 2, 3, 4, 5]


def test_required_oasis_fields_present():
    client = _mock_client(3)
    profiles, _, _ = generate_personas(ANALYSIS, n_agents=3,
                                        products=[], client=client,
                                        model="gpt-4o-mini", seed=42)
    required = {"user_id", "username", "name", "bio", "persona", "karma"}
    for p in profiles:
        assert required.issubset(p.keys()), f"Missing fields in {p}"


def test_prompt_and_raw_returned():
    client = _mock_client(3)
    _, prompt, raw = generate_personas(ANALYSIS, n_agents=3,
                                        products=[], client=client,
                                        model="gpt-4o-mini", seed=42)
    assert len(prompt) > 0
    assert len(raw) > 0


def test_distribute_agents_sums_to_n():
    import random
    rng = random.Random(42)
    dist = _distribute_agents(ARCHETYPES, n=10, rng=rng)
    assert sum(dist.values()) == 10


def test_distribute_agents_all_archetypes_get_at_least_one():
    import random
    rng = random.Random(42)
    dist = _distribute_agents(ARCHETYPES, n=10, rng=rng)
    for arch in ARCHETYPES:
        assert dist[arch.name] >= 1


def test_product_list_included_in_prompt():
    products = [NormalizedProduct(title="Sony XM5", price=349.99, brand="Sony")]
    client = _mock_client(3)
    _, prompt, _ = generate_personas(ANALYSIS, n_agents=3,
                                      products=products, client=client,
                                      model="gpt-4o-mini", seed=42)
    assert "Sony XM5" in prompt
```

- [ ] **Step 3.2: Run tests — confirm they fail**

```bash
pytest tests/test_persona_gen.py -v 2>&1 | head -15
```
Expected: `ImportError: cannot import name 'generate_personas'`

- [ ] **Step 3.3: Implement `persona_gen.py`**

Create `product_reddit_sim/persona_gen.py`:
```python
"""LLM Call 2: generate N full Reddit personas distributed across archetypes."""
from __future__ import annotations

import json
import math
import random
from datetime import date
from typing import Optional

from openai import OpenAI

from .analyzer import PersonaArchetype, ProductAnalysis
from .loader import NormalizedProduct


def generate_personas(
    analysis: ProductAnalysis,
    n_agents: int,
    products: list[NormalizedProduct],
    client: OpenAI,
    model: str,
    seed: int,
) -> tuple[list[dict], str, str]:
    """Return (profiles_list, prompt, raw_llm_response)."""
    rng = random.Random(seed + 1)
    distribution = _distribute_agents(analysis.persona_archetypes, n_agents, rng)
    product_summary = _build_product_summary(products)
    prompt = _build_prompt(analysis, distribution, product_summary)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)
    profiles = data["personas"]

    # Enforce sequential user_ids starting from 1
    for i, p in enumerate(profiles):
        p["user_id"] = i + 1
        # Ensure created_at field exists (OASIS requires it)
        if "created_at" not in p:
            p["created_at"] = str(date.today())

    return profiles, prompt, raw


def _distribute_agents(
    archetypes: list[PersonaArchetype], n: int, rng: random.Random
) -> dict[str, int]:
    """Distribute n agents across archetypes proportional to weight.
    Every archetype gets at least 1 agent."""
    distribution: dict[str, int] = {}
    remaining = n
    for arch in archetypes[:-1]:
        count = max(1, round(arch.weight * n))
        count = min(count, remaining - (len(archetypes) - len(distribution) - 1))
        distribution[arch.name] = count
        remaining -= count
    distribution[archetypes[-1].name] = max(1, remaining)
    return distribution


def _build_product_summary(products: list[NormalizedProduct]) -> str:
    lines = []
    for p in products[:40]:
        price_str = f"${p.price}" if p.price else "price unknown"
        rating_str = f"{p.rating}/5" if p.rating else "unrated"
        lines.append(f"- {p.title} ({p.brand}, {price_str}, {rating_str})")
    return "\n".join(lines) if lines else "No products provided."


def _build_prompt(
    analysis: ProductAnalysis,
    distribution: dict[str, int],
    product_summary: str,
) -> str:
    total = sum(distribution.values())
    dist_lines = "\n".join(
        f"  - {name}: {count} personas" for name, count in distribution.items()
    )
    archetype_details = "\n".join(
        f"  [{a.name}]: {a.description}"
        for a in analysis.persona_archetypes
    )
    themes = ", ".join(analysis.key_themes[:5])

    return f"""You are creating {total} realistic Reddit user personas for an academic NeurIPS simulation of authentic online discussions about {analysis.product_category}.

PERSONA DISTRIBUTION (you must generate exactly these counts):
{dist_lines}

ARCHETYPE DESCRIPTIONS:
{archetype_details}

KEY DISCUSSION THEMES: {themes}

PRODUCTS POOL (agents should be aware of and able to reference these):
{product_summary}

Generate exactly {total} personas. Return JSON with this structure:
{{
  "personas": [
    {{
      "user_id": 1,
      "username": "unique_reddit_style_username",
      "name": "Full Name",
      "bio": "1-2 sentence Reddit bio in first person, casual tone",
      "persona": "3-4 paragraph detailed description covering: personality and communication style, what they care about and why, what products they own or want, how they argue or engage in discussions, their blind spots and biases — make them feel like a real human",
      "karma": 4500,
      "age": 29,
      "gender": "male",
      "mbti": "INTJ",
      "country": "USA",
      "profession": "Software Engineer",
      "interested_topics": ["topic1", "topic2"],
      "archetype": "Audiophile"
    }}
  ]
}}

RULES:
- Each persona must be MEANINGFULLY DIFFERENT even within the same archetype
- Usernames: creative Reddit-style, no spaces (e.g. "AudiophileMax_42", "budget_earbud_king")
- Karma range: casual 200–2000 | regular 2000–15000 | power user 15000–50000
- Include skeptics, enthusiasts, beginners, and experts
- Personas should reference specific products from the pool when relevant to their archetype
- Make personas feel like real humans — give them opinions, quirks, and occasional contradictions"""
```

- [ ] **Step 3.4: Run tests — confirm they pass**

```bash
pytest tests/test_persona_gen.py -v
```
Expected: 7 tests pass.

- [ ] **Step 3.5: Commit**

```bash
git add product_reddit_sim/persona_gen.py tests/test_persona_gen.py
git commit -m "feat: add persona_gen LLM call 2 - generate N Reddit personas"
```

---

## Task 4: `config_builder.py` — LLM Call 3 + OASIS config files

**Files:**
- Create: `product_reddit_sim/config_builder.py`
- Create: `tests/test_config_builder.py`

### What `config_builder.py` does
(1) Generates seed posts via LLM (scaled ~12% of product count, capped at 10, min 3; mix of product-specific and topic-based). (2) Builds OASIS `simulation_config.json` with time config, agent activity configs (activity level derived from karma), and event config. (3) Writes `reddit_profiles.json` and `simulation_config.json` to output dir.

- [ ] **Step 4.1: Write failing tests**

Create `tests/test_config_builder.py`:
```python
import json, os, tempfile
from unittest.mock import MagicMock
from product_reddit_sim.loader import NormalizedProduct
from product_reddit_sim.analyzer import PersonaArchetype, ProductAnalysis
from product_reddit_sim.config_builder import (
    build_config,
    _seed_post_count,
    _build_agent_configs,
)

ANALYSIS = ProductAnalysis(
    product_category="wireless headphones",
    key_themes=["ANC", "battery", "value"],
    persona_archetypes=[PersonaArchetype("Audiophile", "desc", 1.0)],
    discussion_seed_topics=["Best ANC under $200?"],
)

PROFILES = [
    {"user_id": 1, "username": "user1", "name": "User One", "bio": "bio",
     "persona": "persona text", "karma": 5000, "archetype": "Audiophile"},
    {"user_id": 2, "username": "user2", "name": "User Two", "bio": "bio",
     "persona": "persona text", "karma": 500, "archetype": "Audiophile"},
]

PRODUCTS = [NormalizedProduct(title=f"Product {i}", price=float(i*10), brand="B")
            for i in range(1, 21)]

SEED_POSTS_RESPONSE = {
    "seed_posts": [
        {"poster_agent_id": 1, "content": "What do you think about Product 5?", "post_type": "product_specific"},
        {"poster_agent_id": 2, "content": "Best budget headphones?", "post_type": "topic_based"},
    ]
}


def _mock_client():
    client = MagicMock()
    completion = MagicMock()
    completion.choices[0].message.content = json.dumps(SEED_POSTS_RESPONSE)
    client.chat.completions.create.return_value = completion
    return client


def test_writes_simulation_config_json():
    with tempfile.TemporaryDirectory() as tmp:
        build_config(ANALYSIS, PROFILES, PRODUCTS, tmp,
                     cli_args={"hours": 48, "rounds": 30, "model": "gpt-4o-mini", "base_url": ""},
                     client=_mock_client(), model="gpt-4o-mini", seed=42)
        assert os.path.exists(os.path.join(tmp, "simulation_config.json"))


def test_writes_reddit_profiles_json():
    with tempfile.TemporaryDirectory() as tmp:
        build_config(ANALYSIS, PROFILES, PRODUCTS, tmp,
                     cli_args={"hours": 48, "rounds": 30, "model": "gpt-4o-mini", "base_url": ""},
                     client=_mock_client(), model="gpt-4o-mini", seed=42)
        assert os.path.exists(os.path.join(tmp, "reddit_profiles.json"))


def test_config_contains_seed_posts():
    with tempfile.TemporaryDirectory() as tmp:
        build_config(ANALYSIS, PROFILES, PRODUCTS, tmp,
                     cli_args={"hours": 48, "rounds": 30, "model": "gpt-4o-mini", "base_url": ""},
                     client=_mock_client(), model="gpt-4o-mini", seed=42)
        with open(os.path.join(tmp, "simulation_config.json")) as f:
            cfg = json.load(f)
        assert len(cfg["event_config"]["initial_posts"]) >= 1


def test_agent_configs_count_matches_profiles():
    with tempfile.TemporaryDirectory() as tmp:
        build_config(ANALYSIS, PROFILES, PRODUCTS, tmp,
                     cli_args={"hours": 48, "rounds": 30, "model": "gpt-4o-mini", "base_url": ""},
                     client=_mock_client(), model="gpt-4o-mini", seed=42)
        with open(os.path.join(tmp, "simulation_config.json")) as f:
            cfg = json.load(f)
        assert len(cfg["agent_configs"]) == len(PROFILES)


def test_seed_post_count_scaling():
    assert _seed_post_count(20) == 3   # max(3, round(20*0.12))=3
    assert _seed_post_count(50) == 6   # max(3, round(50*0.12))=6
    assert _seed_post_count(200) == 10  # capped at 10


def test_agent_activity_scales_with_karma():
    configs = _build_agent_configs(PROFILES, rng=__import__("random").Random(42))
    high_karma_activity = next(c["activity_level"] for c in configs if c["agent_id"] == 1)
    low_karma_activity = next(c["activity_level"] for c in configs if c["agent_id"] == 2)
    assert high_karma_activity > low_karma_activity
```

- [ ] **Step 4.2: Run tests — confirm they fail**

```bash
pytest tests/test_config_builder.py -v 2>&1 | head -15
```
Expected: `ImportError: cannot import name 'build_config'`

- [ ] **Step 4.3: Implement `config_builder.py`**

Create `product_reddit_sim/config_builder.py`:
```python
"""LLM Call 3: generate seed posts + write OASIS config files."""
from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime
from typing import Optional

from openai import OpenAI

from .analyzer import ProductAnalysis
from .loader import NormalizedProduct


def build_config(
    analysis: ProductAnalysis,
    profiles: list[dict],
    products: list[NormalizedProduct],
    output_dir: str,
    cli_args: dict,
    client: OpenAI,
    model: str,
    seed: int,
) -> tuple[str, str]:
    """Write reddit_profiles.json + simulation_config.json. Returns (seed_prompt, seed_raw)."""
    rng = random.Random(seed + 2)

    n_seeds = _seed_post_count(len(products))
    seed_posts, seed_prompt, seed_raw = _generate_seed_posts(
        analysis, profiles, products, n_seeds, client, model
    )

    agent_configs = _build_agent_configs(profiles, rng)

    sim_id = os.path.basename(output_dir)
    config = {
        "simulation_id": sim_id,
        "project_id": "product_reddit_sim",
        "graph_id": analysis.product_category.replace(" ", "_"),
        "simulation_requirement": (
            f"Simulate authentic Reddit discussion about {analysis.product_category} "
            f"for NeurIPS research. Agents have distinct backgrounds and opinions. "
            f"Discussion should feel real — debates, anecdotes, recommendations, disagreements. "
            f"Key themes: {', '.join(analysis.key_themes[:4])}."
        ),
        "time_config": {
            "total_simulation_hours": cli_args["hours"],
            "minutes_per_round": 60,
            "agents_per_hour_min": 5,
            "agents_per_hour_max": min(20, len(profiles)),
            "peak_hours": [18, 19, 20, 21, 22],
            "peak_activity_multiplier": 1.5,
            "off_peak_hours": [1, 2, 3, 4, 5],
            "off_peak_activity_multiplier": 0.1,
            "morning_hours": [7, 8, 9],
            "morning_activity_multiplier": 0.5,
            "work_hours": list(range(10, 18)),
            "work_activity_multiplier": 0.7,
        },
        "agent_configs": agent_configs,
        "event_config": {
            "initial_posts": seed_posts,
            "scheduled_events": [],
            "hot_topics": analysis.discussion_seed_topics,
            "narrative_direction": (
                f"Authentic Reddit community discussing {analysis.product_category}."
            ),
        },
        "reddit_config": {
            "platform": "reddit",
            "recency_weight": 0.35,
            "popularity_weight": 0.35,
            "relevance_weight": 0.30,
            "viral_threshold": 8,
            "echo_chamber_strength": 0.4,
        },
        "twitter_config": None,
        "llm_model": cli_args.get("model", ""),
        "llm_base_url": cli_args.get("base_url", ""),
        "generated_at": datetime.now().isoformat(),
    }

    # Write reddit_profiles.json (strip internal archetype key — OASIS doesn't need it)
    profiles_for_oasis = [
        {k: v for k, v in p.items() if k != "archetype"} for p in profiles
    ]
    profiles_path = os.path.join(output_dir, "reddit_profiles.json")
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles_for_oasis, f, ensure_ascii=False, indent=2)

    config_path = os.path.join(output_dir, "simulation_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return seed_prompt, seed_raw


def _seed_post_count(total_products: int) -> int:
    return min(10, max(3, round(total_products * 0.12)))


def _build_agent_configs(profiles: list[dict], rng: random.Random) -> list[dict]:
    configs = []
    for p in profiles:
        karma = p.get("karma", 1000)
        # Activity scales logarithmically with karma: 1000 karma → ~0.4, 10000 → ~0.6, 50000 → ~0.8
        base_activity = min(0.9, 0.2 + (math.log10(max(karma, 100)) / 5))
        activity = round(
            max(0.1, min(0.95, base_activity + rng.uniform(-0.08, 0.08))), 2
        )
        configs.append({
            "agent_id": p["user_id"],
            "entity_uuid": f"agent_{p['user_id']}",
            "entity_name": p["username"],
            "entity_type": "person",
            "activity_level": activity,
            "posts_per_hour": round(rng.uniform(0.3, 1.5), 1),
            "comments_per_hour": round(rng.uniform(0.5, 3.0), 1),
            "active_hours": list(range(rng.randint(7, 10), rng.randint(21, 24))),
            "response_delay_min": rng.randint(5, 20),
            "response_delay_max": rng.randint(30, 90),
            "sentiment_bias": round(rng.uniform(-0.3, 0.4), 2),
            "stance": rng.choice(["supportive", "neutral", "neutral", "observer"]),
            "influence_weight": round(math.log10(max(karma, 100)) / 5, 2),
        })
    return configs


def _generate_seed_posts(
    analysis: ProductAnalysis,
    profiles: list[dict],
    products: list[NormalizedProduct],
    n_seeds: int,
    client: OpenAI,
    model: str,
) -> tuple[list[dict], str, str]:
    product_lines = "\n".join(
        f"- {p.title} | ${p.price} | Rating: {p.rating}/5 | "
        + (", ".join(p.features[:3]) if p.features else "")
        for p in products[:50]
    )
    persona_lines = "\n".join(
        f"- agent_id={p['user_id']} username={p['username']} archetype={p.get('archetype', '')}"
        for p in profiles[:25]
    )

    prompt = f"""Create {n_seeds} realistic Reddit seed posts to start a discussion about {analysis.product_category} for a NeurIPS academic simulation.

AVAILABLE PRODUCTS (agents draw from this pool):
{product_lines}

AVAILABLE PERSONAS (use their agent_id for poster_agent_id):
{persona_lines}

Generate a mix:
- ~60% product-specific (personal experience or question about a specific product)
- ~40% topic/comparison (best under $X, brand comparisons, use-case questions)

Return JSON:
{{
  "seed_posts": [
    {{
      "poster_agent_id": 3,
      "content": "Full Reddit post text written naturally in the poster's voice (2-5 sentences)",
      "post_type": "product_specific"
    }}
  ]
}}

RULES:
- Each post sounds like a DIFFERENT person with a different purpose
- Product-specific posts reference REAL products from the list above
- Topic posts use natural Reddit language ("looking for recs", "worth the upgrade?", "which is better")
- Match poster archetype to post type (budget personas ask budget questions)
- No two posts repeat the same product or topic"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)
    return data["seed_posts"], prompt, raw
```

- [ ] **Step 4.4: Run tests — confirm they pass**

```bash
pytest tests/test_config_builder.py -v
```
Expected: 6 tests pass.

- [ ] **Step 4.5: Commit**

```bash
git add product_reddit_sim/config_builder.py tests/test_config_builder.py
git commit -m "feat: add config_builder - seed posts and OASIS config generation"
```

---

## Task 5: `runner.py` — MiroFish subprocess

**Files:**
- Create: `product_reddit_sim/runner.py`

### What `runner.py` does
Locates the MiroFish `run_reddit_simulation.py` script (relative to this package), finds the right Python interpreter (MiroFish `.venv` if installed, else system Python), and runs the simulation as a subprocess with `--config`, `--max-rounds`, and `--no-wait`.

No unit tests for this module — it wraps a subprocess and the real test is integration. We add a lightweight existence-check test instead.

- [ ] **Step 5.1: Implement `runner.py`**

Create `product_reddit_sim/runner.py`:
```python
"""Invoke MiroFish run_reddit_simulation.py as a subprocess."""
from __future__ import annotations

import os
import subprocess
import sys

# Path to MiroFish script relative to this package (GEO/product_reddit_sim/../MiroFish/...)
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_GEO_ROOT = os.path.dirname(_PACKAGE_DIR)
MIROFISH_SCRIPT = os.path.join(
    _GEO_ROOT, "MiroFish", "backend", "scripts", "run_reddit_simulation.py"
)
MIROFISH_VENV_PYTHON = os.path.join(
    _GEO_ROOT, "MiroFish", "backend", ".venv", "bin", "python"
)


def run_simulation(config_path: str, max_rounds: int) -> None:
    """Run the OASIS Reddit simulation via MiroFish. Blocks until complete."""
    script = os.path.abspath(MIROFISH_SCRIPT)
    if not os.path.exists(script):
        raise FileNotFoundError(
            f"MiroFish simulation script not found at:\n  {script}\n"
            "Ensure MiroFish is cloned at GEO/MiroFish/ and backend "
            "dependencies installed (cd MiroFish/backend && uv sync)."
        )

    python = _find_python()

    cmd = [python, script, "--config", config_path,
           "--max-rounds", str(max_rounds), "--no-wait"]

    print(f"\n{'='*60}")
    print("STARTING OASIS REDDIT SIMULATION (MiroFish backbone)")
    print(f"Script:  {script}")
    print(f"Config:  {config_path}")
    print(f"Rounds:  {max_rounds}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"MiroFish simulation exited with code {result.returncode}. "
            "Check logs in the simulation output directory."
        )


def _find_python() -> str:
    """Prefer MiroFish's own venv python; fall back to current interpreter."""
    if os.path.exists(MIROFISH_VENV_PYTHON):
        return MIROFISH_VENV_PYTHON
    return sys.executable
```

- [ ] **Step 5.2: Verify module imports cleanly**

```bash
cd /Users/yaoningyu/Desktop/UIUC/GEO
source .venv/bin/activate
python3 -c "from product_reddit_sim.runner import run_simulation, MIROFISH_SCRIPT; print('runner ok'); print('script path:', MIROFISH_SCRIPT)"
```
Expected: prints `runner ok` and the MiroFish script path.

- [ ] **Step 5.3: Commit**

```bash
git add product_reddit_sim/runner.py
git commit -m "feat: add runner - MiroFish subprocess wrapper"
```

---

## Task 6: `exporter.py` — SQLite → JSON + Markdown

**Files:**
- Create: `product_reddit_sim/exporter.py`
- Create: `tests/test_exporter.py`

### What `exporter.py` does
Reads the OASIS `reddit_simulation.db` trace table (columns: `action`, `user_id`, `info` JSON, `created_at`). Filters for `create_post` and `create_comment` actions (case-insensitive). Joins with profiles for author metadata. Builds nested thread (posts with comments). Writes `discussion.json` and `discussion.md`.

**Important:** OASIS action type string values may be `"create_post"` or `"CREATE_POST"` depending on version. The exporter uses case-insensitive matching.

- [ ] **Step 6.1: Write failing tests**

Create `tests/test_exporter.py`:
```python
import json, os, sqlite3, tempfile
from product_reddit_sim.exporter import export_discussion, _render_markdown

PROFILES = [
    {"user_id": 1, "username": "AudiophileMax", "karma": 18000,
     "name": "Max", "bio": "", "persona": ""},
    {"user_id": 2, "username": "BudgetHunter99", "karma": 500,
     "name": "Dave", "bio": "", "persona": ""},
]

META = {
    "product_category": "headphones",
    "hint": "commuters",
    "agent_count": 2,
    "seed": 42,
    "simulated_hours": 48,
    "rounds": 10,
    "run_id": "test_run_001",
}


def _make_db(tmp: str) -> str:
    """Create a minimal OASIS-style trace DB."""
    db_path = os.path.join(tmp, "reddit_simulation.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE trace (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            action TEXT,
            info TEXT,
            created_at TEXT
        )
    """)
    rows = [
        (1, "create_post",
         json.dumps({"content": "What headphones for commuting?", "post_id": 101}),
         "2026-04-11T19:00:00"),
        (2, "create_comment",
         json.dumps({"content": "Sony XM5 is great!", "post_id": 101}),
         "2026-04-11T19:30:00"),
        (1, "like_post",
         json.dumps({"post_id": 101}),
         "2026-04-11T19:05:00"),
        (2, "CREATE_POST",  # test uppercase handling
         json.dumps({"content": "Budget earbuds under $50?", "post_id": 102}),
         "2026-04-11T20:00:00"),
    ]
    conn.executemany(
        "INSERT INTO trace (user_id, action, info, created_at) VALUES (?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return db_path


def _make_profiles_file(tmp: str) -> str:
    path = os.path.join(tmp, "reddit_profiles.json")
    with open(path, "w") as f:
        json.dump(PROFILES, f)
    return path


def test_creates_discussion_json():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        json_path, _ = export_discussion(db, pf, tmp, META)
    assert os.path.exists(json_path)


def test_creates_discussion_md():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        _, md_path = export_discussion(db, pf, tmp, META)
    assert os.path.exists(md_path)


def test_json_contains_posts():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        json_path, _ = export_discussion(db, pf, tmp, META)
        with open(json_path) as f:
            data = json.load(f)
    assert len(data["posts"]) >= 1


def test_author_names_resolved():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        json_path, _ = export_discussion(db, pf, tmp, META)
        with open(json_path) as f:
            data = json.load(f)
    authors = {p["author"] for p in data["posts"]}
    assert "AudiophileMax" in authors or "BudgetHunter99" in authors


def test_handles_uppercase_action_types():
    """OASIS may store CREATE_POST or create_post — both must be handled."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        json_path, _ = export_discussion(db, pf, tmp, META)
        with open(json_path) as f:
            data = json.load(f)
    # Should have 2 posts (one lowercase, one uppercase action)
    assert len(data["posts"]) == 2


def test_meta_included_in_json():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        json_path, _ = export_discussion(db, pf, tmp, META)
        with open(json_path) as f:
            data = json.load(f)
    assert data["meta"]["product_category"] == "headphones"
    assert data["meta"]["agent_count"] == 2


def test_markdown_contains_usernames():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        _, md_path = export_discussion(db, pf, tmp, META)
        with open(md_path) as f:
            md = f.read()
    assert "AudiophileMax" in md or "BudgetHunter99" in md


def test_markdown_has_header():
    with tempfile.TemporaryDirectory() as tmp:
        db = _make_db(tmp)
        pf = _make_profiles_file(tmp)
        _, md_path = export_discussion(db, pf, tmp, META)
        with open(md_path) as f:
            md = f.read()
    assert "headphones" in md
    assert "test_run_001" in md
```

- [ ] **Step 6.2: Run tests — confirm they fail**

```bash
pytest tests/test_exporter.py -v 2>&1 | head -15
```
Expected: `ImportError: cannot import name 'export_discussion'`

- [ ] **Step 6.3: Implement `exporter.py`**

Create `product_reddit_sim/exporter.py`:
```python
"""Export OASIS SQLite trace → discussion.json + discussion.md."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Optional


def export_discussion(
    db_path: str,
    profiles_path: str,
    output_dir: str,
    meta: dict,
) -> tuple[str, str]:
    """Return (json_path, md_path)."""
    profiles = _load_profiles(profiles_path)
    posts, comments = _load_from_db(db_path)
    thread = _build_thread(posts, comments, profiles)

    discussion = {"meta": meta, "posts": thread}

    json_path = os.path.join(output_dir, "discussion.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(discussion, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(output_dir, "discussion.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(discussion))

    return json_path, md_path


def _load_profiles(profiles_path: str) -> dict[int, dict]:
    with open(profiles_path, encoding="utf-8") as f:
        profiles = json.load(f)
    return {p["user_id"]: p for p in profiles}


def _load_from_db(db_path: str) -> tuple[list, list]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Case-insensitive match: OASIS versions differ (create_post vs CREATE_POST)
    cur.execute("""
        SELECT user_id, info, created_at FROM trace
        WHERE LOWER(action) = 'create_post'
        ORDER BY created_at ASC
    """)
    posts = [dict(r) for r in cur.fetchall()]

    cur.execute("""
        SELECT user_id, info, created_at FROM trace
        WHERE LOWER(action) = 'create_comment'
        ORDER BY created_at ASC
    """)
    comments = [dict(r) for r in cur.fetchall()]

    conn.close()
    return posts, comments


def _build_thread(
    posts: list[dict], comments: list[dict], profiles: dict[int, dict]
) -> list[dict]:
    post_list: list[dict] = []
    # Map from OASIS post_id → index in post_list for comment attachment
    oasis_post_id_map: dict[int, int] = {}

    for i, row in enumerate(posts):
        info = _parse_info(row["info"])
        profile = profiles.get(row["user_id"], {})
        post = {
            "post_id": i + 1,
            "author": profile.get("username", f"user_{row['user_id']}"),
            "author_karma": profile.get("karma", 0),
            "content": info.get("content") or info.get("post_content") or str(info),
            "timestamp": row["created_at"],
            "likes": 0,
            "dislikes": 0,
            "comments": [],
        }
        if "post_id" in info:
            oasis_post_id_map[info["post_id"]] = i
        post_list.append(post)

    for j, row in enumerate(comments):
        info = _parse_info(row["info"])
        profile = profiles.get(row["user_id"], {})
        comment = {
            "comment_id": j + 1,
            "author": profile.get("username", f"user_{row['user_id']}"),
            "author_karma": profile.get("karma", 0),
            "content": info.get("content") or info.get("comment") or str(info),
            "timestamp": row["created_at"],
            "likes": 0,
            "dislikes": 0,
        }
        # Attach to the post this comment references, or the latest post as fallback
        target_idx = oasis_post_id_map.get(info.get("post_id"))
        if target_idx is not None:
            post_list[target_idx]["comments"].append(comment)
        elif post_list:
            post_list[-1]["comments"].append(comment)

    return post_list


def _parse_info(info_raw) -> dict:
    if not info_raw:
        return {}
    if isinstance(info_raw, dict):
        return info_raw
    try:
        return json.loads(info_raw)
    except (json.JSONDecodeError, TypeError):
        return {"content": str(info_raw)}


def _render_markdown(discussion: dict) -> str:
    meta = discussion["meta"]
    category = meta.get("product_category", "products")
    sub = category.replace(" ", "_")
    lines = [
        f"# r/{sub} simulation — {category}",
        f"*Hint: {meta.get('hint', 'none')} | "
        f"Agents: {meta.get('agent_count')} | "
        f"Simulated: {meta.get('simulated_hours')}h | "
        f"Run: {meta.get('run_id')}*",
        "",
        "---",
        "",
    ]

    for post in discussion["posts"]:
        ts = _fmt_ts(post.get("timestamp"))
        content = post["content"]
        preview = content[:100].replace("\n", " ")
        lines += [
            f"## [{post['likes']}↑] {preview}{'...' if len(content) > 100 else ''}",
            f"**u/{post['author']}** (karma: {post['author_karma']:,}) · {ts}",
            "",
            content,
            "",
        ]
        for c in post["comments"]:
            cts = _fmt_ts(c.get("timestamp"))
            lines += [
                f"> **u/{c['author']}** (karma: {c['author_karma']:,}) · {cts} [{c['likes']}↑]",
                ">",
                *(f"> {line}" for line in c["content"].splitlines()),
                "",
            ]
        lines += ["---", ""]

    return "\n".join(lines)


def _fmt_ts(ts: Optional[str]) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(str(ts))
        return dt.strftime("%I:%M %p").lstrip("0")
    except Exception:
        return str(ts)
```

- [ ] **Step 6.4: Run tests — confirm they pass**

```bash
pytest tests/test_exporter.py -v
```
Expected: 8 tests pass.

- [ ] **Step 6.5: Commit**

```bash
git add product_reddit_sim/exporter.py tests/test_exporter.py
git commit -m "feat: add exporter - SQLite to discussion JSON and Markdown"
```

---

## Task 7: `run_discussion.py` — CLI entry point

**Files:**
- Create: `run_discussion.py`

### What `run_discussion.py` does
Wires all modules in sequence: load → analyze → generate personas → build config → run simulation → export. Saves `run_config.json` (reproducibility record including input file SHA256). Loads LLM credentials from `MiroFish/.env`. Prints progress at each step.

- [ ] **Step 7.1: Implement `run_discussion.py`**

Create `run_discussion.py` at the GEO project root:
```python
#!/usr/bin/env python3
"""
Product Reddit Simulation — Single CLI Entry Point

Usage:
    python run_discussion.py products.json --agents 50 --hint "commuters"
    python run_discussion.py products.json --agents 30 --rounds 20 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

# Load LLM credentials from MiroFish .env
_GEO_ROOT = os.path.dirname(os.path.abspath(__file__))
_MIROFISH_ENV = os.path.join(_GEO_ROOT, "MiroFish", ".env")
if os.path.exists(_MIROFISH_ENV):
    load_dotenv(_MIROFISH_ENV)
else:
    load_dotenv()  # fallback to local .env

from product_reddit_sim.loader import load_products
from product_reddit_sim.analyzer import analyze_products
from product_reddit_sim.persona_gen import generate_personas
from product_reddit_sim.config_builder import build_config
from product_reddit_sim.runner import run_simulation
from product_reddit_sim.exporter import export_discussion


def parse_args():
    p = argparse.ArgumentParser(
        description="Simulate Reddit discussions about products using LLM-generated personas (MiroFish/OASIS backbone)"
    )
    p.add_argument("products_json", help="Path to product JSON file")
    p.add_argument("--agents", type=int, default=30,
                   help="Number of agents (default: 30)")
    p.add_argument("--hint", type=str, default=None,
                   help="Optional natural-language hint to guide persona/topic generation")
    p.add_argument("--hours", type=int, default=48,
                   help="Simulated hours (default: 48)")
    p.add_argument("--rounds", type=int, default=30,
                   help="Max OASIS simulation rounds (default: 30)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--output-dir", type=str, default="outputs",
                   help="Output directory (default: ./outputs)")
    return p.parse_args()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_output_dir(base: str, category: str) -> str:
    slug = category.replace(" ", "_").replace("/", "_")[:40]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{slug}_{ts}"
    path = os.path.join(base, run_id)
    os.makedirs(path, exist_ok=True)
    return path, run_id


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        print(f"ERROR: {key} not set. Add it to MiroFish/.env")
        sys.exit(1)
    return val


def main():
    args = parse_args()
    started_at = datetime.now().isoformat()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.exists(args.products_json):
        print(f"ERROR: File not found: {args.products_json}")
        sys.exit(1)

    api_key = _require_env("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")

    client = OpenAI(api_key=api_key, base_url=base_url)
    cli_args = {
        "hours": args.hours,
        "rounds": args.rounds,
        "model": model,
        "base_url": base_url,
    }

    # ── Step 1: Load products ─────────────────────────────────────────────────
    print(f"\n[1/6] Loading products from {args.products_json}...")
    products = load_products(args.products_json)
    print(f"      → {len(products)} products loaded")

    # ── Step 2: Analyze products (LLM Call 1) ─────────────────────────────────
    print(f"\n[2/6] Analyzing product category and generating persona archetypes...")
    analysis = analyze_products(products, args.hint, client, model, args.seed)
    print(f"      → Category: {analysis.product_category}")
    print(f"      → Archetypes: {[a.name for a in analysis.persona_archetypes]}")

    # ── Create output directory (now we know the category) ────────────────────
    output_dir, run_id = _make_output_dir(args.output_dir, analysis.product_category)
    print(f"\n      Output: {output_dir}")

    # Save product analysis + LLM audit trail
    analysis_record = {
        "product_category": analysis.product_category,
        "key_themes": analysis.key_themes,
        "persona_archetypes": [
            {"name": a.name, "description": a.description, "weight": a.weight}
            for a in analysis.persona_archetypes
        ],
        "discussion_seed_topics": analysis.discussion_seed_topics,
        "_analysis_prompt": analysis._prompt,
        "_analysis_raw_response": analysis._raw_response,
    }

    # ── Step 3: Generate personas (LLM Call 2) ────────────────────────────────
    print(f"\n[3/6] Generating {args.agents} personas...")
    profiles, persona_prompt, persona_raw = generate_personas(
        analysis, args.agents, products, client, model, args.seed
    )
    print(f"      → {len(profiles)} personas generated")

    # Append persona LLM audit trail to analysis record
    analysis_record["_persona_prompt"] = persona_prompt
    analysis_record["_persona_raw_response"] = persona_raw

    analysis_path = os.path.join(output_dir, "product_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_record, f, ensure_ascii=False, indent=2)

    # ── Step 4: Build config (LLM Call 3) ─────────────────────────────────────
    print(f"\n[4/6] Generating seed posts and building OASIS config...")
    seed_prompt, seed_raw = build_config(
        analysis, profiles, products, output_dir, cli_args, client, model, args.seed
    )

    # Append seed post LLM audit trail
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis_record = json.load(f)
    analysis_record["_seed_prompt"] = seed_prompt
    analysis_record["_seed_raw_response"] = seed_raw
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_record, f, ensure_ascii=False, indent=2)

    config_path = os.path.join(output_dir, "simulation_config.json")
    print(f"      → Config written: {config_path}")

    # ── Step 5: Run simulation ────────────────────────────────────────────────
    print(f"\n[5/6] Running OASIS Reddit simulation ({args.rounds} rounds max)...")
    run_simulation(config_path, max_rounds=args.rounds)

    # ── Step 6: Export discussion ──────────────────────────────────────────────
    db_path = os.path.join(output_dir, "reddit_simulation.db")
    if not os.path.exists(db_path):
        print(f"\nWARNING: Simulation DB not found at {db_path}. Export skipped.")
        sys.exit(1)

    print(f"\n[6/6] Exporting discussion to JSON and Markdown...")
    profiles_path = os.path.join(output_dir, "reddit_profiles.json")
    meta = {
        "product_category": analysis.product_category,
        "hint": args.hint,
        "agent_count": len(profiles),
        "seed": args.seed,
        "simulated_hours": args.hours,
        "rounds": args.rounds,
        "run_id": run_id,
    }
    json_path, md_path = export_discussion(db_path, profiles_path, output_dir, meta)
    print(f"      → {json_path}")
    print(f"      → {md_path}")

    # ── Save run_config.json (reproducibility record) ─────────────────────────
    finished_at = datetime.now().isoformat()
    run_config = {
        "run_id": run_id,
        "input_file": os.path.abspath(args.products_json),
        "input_file_sha256": _sha256(args.products_json),
        "hint": args.hint,
        "agents": args.agents,
        "hours": args.hours,
        "rounds": args.rounds,
        "seed": args.seed,
        "llm_model": model,
        "llm_base_url": base_url,
        "product_category": analysis.product_category,
        "archetype_count": len(analysis.persona_archetypes),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    run_config_path = os.path.join(output_dir, "run_config.json")
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"  run_config.json       ← reproducibility record")
    print(f"  product_analysis.json ← LLM audit trail")
    print(f"  reddit_profiles.json  ← {len(profiles)} agent personas")
    print(f"  simulation_config.json")
    print(f"  reddit_simulation.db  ← raw OASIS data")
    print(f"  discussion.json       ← structured discussion")
    print(f"  discussion.md         ← human-readable thread")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Verify help text works**

```bash
cd /Users/yaoningyu/Desktop/UIUC/GEO
source .venv/bin/activate
python3 run_discussion.py --help
```
Expected: Prints usage, all flags with defaults shown.

- [ ] **Step 7.3: Commit**

```bash
git add run_discussion.py
git commit -m "feat: add run_discussion.py CLI entry point - wires full pipeline"
```

---

## Task 8: Run all tests + integration smoke test

**Files:** No new files — verifies everything works end-to-end.

- [ ] **Step 8.1: Run full test suite**

```bash
cd /Users/yaoningyu/Desktop/UIUC/GEO
source .venv/bin/activate
pytest tests/ -v
```
Expected: All tests pass (loader: 8, analyzer: 7, persona_gen: 7, config_builder: 6, exporter: 8 = 36 total).

- [ ] **Step 8.2: Install MiroFish backend dependencies**

```bash
cd /Users/yaoningyu/Desktop/UIUC/GEO/MiroFish/backend
uv sync
```
Expected: Installs `camel-oasis==0.2.5`, `camel-ai==0.2.78`, and other deps.

- [ ] **Step 8.3: Create MiroFish .env**

```bash
cp /Users/yaoningyu/Desktop/UIUC/GEO/MiroFish/.env.example \
   /Users/yaoningyu/Desktop/UIUC/GEO/MiroFish/.env
```
Then edit `MiroFish/.env` and fill in `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME`. (ZEP_API_KEY is not required for standalone simulation.)

- [ ] **Step 8.4: Run integration smoke test (headphones, 5 agents, 3 rounds)**

```bash
cd /Users/yaoningyu/Desktop/UIUC/GEO
source .venv/bin/activate
python3 run_discussion.py \
    bestbuy_scraping/outputs/headphones/bestbuy_100_headphones_scrapfly_enriched_001_100.json \
    --agents 5 \
    --hint "commuters" \
    --hours 12 \
    --rounds 3 \
    --seed 42
```
Expected:
- Progress prints for each of 6 steps
- `outputs/<category>_<timestamp>/` directory created with 7 files
- No errors

- [ ] **Step 8.5: Verify all 7 output files exist**

```bash
ls -la outputs/$(ls -t outputs/ | head -1)/
```
Expected: `run_config.json`, `product_analysis.json`, `reddit_profiles.json`, `simulation_config.json`, `reddit_simulation.db`, `discussion.json`, `discussion.md`

- [ ] **Step 8.6: Spot-check discussion.md**

```bash
head -40 outputs/$(ls -t outputs/ | head -1)/discussion.md
```
Expected: Markdown header with product category, agent count, run ID; at least one post with author username and content.

- [ ] **Step 8.7: Run laptops dataset to confirm generalizability**

```bash
python3 run_discussion.py \
    bestbuy_scraping/outputs/laptops/bestbuy_100_laptops_scrapfly_enriched_001_100.json \
    --agents 5 \
    --rounds 3 \
    --seed 42
```
Expected: Different product category detected, different archetypes generated, simulation runs successfully.

- [ ] **Step 8.8: Final commit**

```bash
git add .
git commit -m "feat: complete product reddit simulation system - generalized for any product dataset"
```

---

## Quick Reference

```bash
# Install MiroFish deps (one-time)
cd MiroFish/backend && uv sync && cd ../..

# Run a simulation
source .venv/bin/activate
python3 run_discussion.py <products.json> --agents 50 --hint "<optional>" --seed 42

# Run tests
pytest tests/ -v

# Read outputs
cat outputs/<run_id>/discussion.md
```
