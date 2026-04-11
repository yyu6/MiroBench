"""LLM Call 3: generate seed posts + write OASIS config files."""
from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime

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
    if "seed_posts" not in data:
        raise ValueError(f"LLM response missing 'seed_posts' key. Raw: {raw[:200]}")
    return data["seed_posts"], prompt, raw
