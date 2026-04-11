"""LLM Call 2: generate N full Reddit personas distributed across archetypes."""
from __future__ import annotations

import json
import random
from datetime import date

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
    distribution = _distribute_agents(analysis.persona_archetypes, n_agents)
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
    if "personas" not in data:
        raise ValueError(f"LLM response missing 'personas' key. Raw: {raw[:200]}")
    profiles = data["personas"]

    # Enforce sequential user_ids starting from 1
    for i, p in enumerate(profiles):
        p["user_id"] = i + 1
        # Ensure created_at field exists (OASIS requires it)
        if "created_at" not in p:
            p["created_at"] = str(date.today())

    return profiles, prompt, raw


def _distribute_agents(
    archetypes: list[PersonaArchetype], n: int
) -> dict[str, int]:
    """Distribute n agents across archetypes proportional to weight.
    Every archetype gets at least 1 agent."""
    if n < len(archetypes):
        raise ValueError(
            f"n_agents ({n}) must be >= number of archetypes ({len(archetypes)})"
        )
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
        price_str = f"${p.price}" if p.price is not None else "price unknown"
        rating_str = f"{p.rating}/5" if p.rating is not None else "unrated"
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
