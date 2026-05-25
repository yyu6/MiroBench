"""LLM Call 1: analyze product sample → product category + persona archetypes."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from .loader import NormalizedProduct
from .llm_utils import create_json_object_completion


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

    raw = create_json_object_completion(
        client=client,
        model=model,
        prompt=prompt,
        temperature=0.3,
    )
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

    return f"""You are analyzing a product dataset to design a realistic Reddit discussion simulation.

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
- Archetypes should span different expertise levels, use cases, budgets, and attitudes
- Archetype descriptions should mention how these people actually talk, argue, or lurk on Reddit
- Include some imperfect or low-expertise users, not just enthusiasts and experts
- Archetypes should also vary in motivation: some reply to help, some to complain, some to correct others, some to defend their own choices, and some mostly lurk
- Do not make every archetype sound balanced, supportive, and well-informed"""
