"""Build simulation config files from analyzed products and generated personas.

This module is the bridge between GEO's LLM-generated artifacts and the
OASIS/MiroFish simulator. Its responsibilities are:

1. ask the model for a small set of realistic seed posts
2. derive per-agent runtime behavior knobs from personas
3. write the two files that the downstream simulator needs:
   - `reddit_profiles.json`
   - `simulation_config.json`

In other words, analysis/persona generation decides *who* the agents are;
this module decides *how the run is initialized* and *how active each agent is*.
"""
from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from .analyzer import ProductAnalysis
from .loader import NormalizedProduct
from .llm_utils import create_json_object_completion
from .prompt_examples import (
    render_generation_few_shot_block,
    select_generation_few_shot_examples,
)


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
    """Write the OASIS-facing config files for one simulation run.

    Returns:
    - `seed_prompt`: the prompt used to generate seed posts
    - `seed_raw`: the raw model response for seed posts
    """
    rng = random.Random(seed + 2)
    few_shot_examples = _load_generation_few_shot_examples(
        analysis=analysis,
        cli_args=cli_args,
        seed=seed,
    )

    n_seeds = max(1, int(cli_args.get("seed_posts", _seed_post_count(len(products))) or 1))
    seed_posts, seed_prompt, seed_raw = _generate_seed_posts(
        analysis, profiles, products, n_seeds, client, model, few_shot_examples
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
            "agents_per_hour_min": min(5, len(profiles)),
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
            "allow_new_threads_during_simulation": False,
            "max_total_threads": n_seeds,
        },
        "prompt_config": {
            "generation_few_shot_source": str(cli_args.get("few_shot_source") or ""),
            "generation_few_shot_examples": few_shot_examples,
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

    # OASIS only needs the final profile fields, not GEO's temporary archetype
    # bookkeeping key.
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
    """Return the default number of initial seed threads.

    This is intentionally fixed at 3. We previously used more seed
    threads, but that made the subreddit feel over-seeded and pushed the
    simulator toward many disconnected top-level posts instead of a small set
    of threads that people actually pile into.
    """

    del total_products
    return 3


def _build_agent_configs(profiles: list[dict], rng: random.Random) -> list[dict]:
    """Derive OASIS runtime behavior parameters from persona metadata.

    These values are not direct copies from the persona JSON. They are sampled
    behavior controls that the patched runtime later uses for:
    - activity scheduling
    - comment/post frequency
    - response delays
    - rough stance and sentiment tendencies
    """

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
    few_shot_examples: list[dict] | None = None,
) -> tuple[list[dict], str, str]:
    """Generate the initial discussion seeds that bootstrap the simulation.

    The prompt is designed to start from a small number of plausible entry
    points rather than flooding the board with many disconnected threads.
    """

    product_lines = "\n".join(
        f"- {p.title} | ${p.price if p.price is not None else 'N/A'} | Rating: {f'{p.rating}/5' if p.rating is not None else 'unrated'} | "
        + (", ".join(p.features[:3]) if p.features else "")
        for p in products[:50]  # limit to 50 for LLM context window
    )
    persona_lines = "\n".join(
        f"- agent_id={p['user_id']} username={p['username']} archetype={p.get('archetype', '')}"
        for p in profiles[:25]
    )
    few_shot_block = render_generation_few_shot_block(
        few_shot_examples,
        heading="REAL REDDIT THREAD EXAMPLES",
    )
    rendered_examples = (
        few_shot_block if few_shot_block else "REAL REDDIT THREAD EXAMPLES:\n- [none provided]"
    )

    prompt = f"""Create {n_seeds} realistic Reddit seed posts to start a discussion about {analysis.product_category}.

AVAILABLE PRODUCTS (agents draw from this pool):
{product_lines}

AVAILABLE PERSONAS (use their agent_id for poster_agent_id):
{persona_lines}

{rendered_examples}

Generate a mix:
- ~60% product-specific (personal experience or question about a specific product)
- ~40% topic/comparison (best under $X, brand comparisons, use-case questions)

Return JSON:
{{
  "seed_posts": [
    {{
      "poster_agent_id": 0,
      "content": "Full Reddit post text written naturally in the poster's voice",
      "post_type": "product_specific"
    }}
  ]
}}

RULES:
- Each post must sound like a DIFFERENT person with a different purpose
- Product-specific posts reference REAL products from the list above
- Topic posts use natural Reddit language ("looking for recs", "worth the upgrade?", "am I overthinking this?", "anyone regret buying...")
- Match poster archetype to post type, budget, and tone
- Most posts should be 1-3 sentences; 4 sentences max unless the poster is rambling on purpose
- Include specific use-case context, uncertainty, frustration, or a concrete comparison so the thread has something to react to
- Vary the post shapes: upgrade dilemma, complaint, buyer's remorse, niche use case, comparison, budget panic, "anyone else", or a half-informed hot take
- Prefer posts that expose one concrete tension or detail people can latch onto, instead of sounding like a clean generic request for recommendations
- Real posts often mention existing setup, a specific annoyance, one fee/spec tradeoff, a recent bad experience, or one reason the OP is hesitating
- Good seed posts create room for disagreement, sarcasm, correction, or personal anecdotes in the replies. Do not make every post so reasonable that all comments would converge on the same calm answer.
- Some posters should sound underinformed, stubborn, defensive, annoyed, or a little biased so the thread has friction.
- No hashtags
- No polished review copy or market-research phrasing
- Use the examples above as style anchors only. Do not copy their products, titles, or exact wording.
- Avoid repetitive templates like "Can anyone share their experience..." on every post
- Avoid clean helper phrasing like "What are your thoughts?" or "Any advice would be appreciated" unless it strongly fits the poster
- No two posts repeat the same product or topic"""

    raw = create_json_object_completion(
        client=client,
        model=model,
        prompt=prompt,
        temperature=0.8,
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON for seed posts: {e}. Raw: {raw[:500]}") from e
    if "seed_posts" not in data:
        raise ValueError(f"LLM response missing 'seed_posts' key. Raw: {raw[:200]}")
    return data["seed_posts"], prompt, raw


def _load_generation_few_shot_examples(
    analysis: ProductAnalysis,
    cli_args: dict,
    seed: int,
) -> list[dict]:
    """Load optional prompt examples that anchor generation to real threads."""

    source = cli_args.get("few_shot_source")
    count = max(0, int(cli_args.get("few_shot_count", 0) or 0))
    if not source or count <= 0:
        return []

    query_text = "\n".join(
        [
            analysis.product_category,
            ", ".join(analysis.key_themes[:6]),
            ", ".join(analysis.discussion_seed_topics[:8]),
        ]
    )
    comments = max(1, int(cli_args.get("few_shot_comments", 2) or 2))

    # Load optional train-only thread ID filter
    allowed_thread_ids: set[str] | None = None
    ids_path = cli_args.get("few_shot_thread_ids")
    if ids_path:
        import json as _json
        ids_path = Path(str(ids_path))
        if ids_path.exists():
            allowed_thread_ids = set(_json.loads(ids_path.read_text(encoding="utf-8")))

    examples = select_generation_few_shot_examples(
        source_path=Path(str(source)),
        query_text=query_text,
        max_examples=count,
        max_comments=comments,
        seed=seed,
        allowed_thread_ids=allowed_thread_ids,
    )
    return examples
