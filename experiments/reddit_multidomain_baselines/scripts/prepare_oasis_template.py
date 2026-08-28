#!/usr/bin/env python3
"""Prepare one domain/model-specific OASIS setup without running a simulation.

``run_oasis_matched_seed_generator.py`` needs personas and a simulation config
as a template.  This script creates that reusable setup once, then the matched
seed generator replaces the LLM-generated roots with the selected real posts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common import REPO_ROOT, read_json, write_json

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-pool-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--seed-posts", type=int, default=5)
    parser.add_argument("--template-posts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    required = [
        output_dir / "product_analysis.json",
        output_dir / "reddit_profiles.json",
        output_dir / "simulation_config.json",
    ]
    if all(path.exists() for path in required) and not args.force:
        print(f"[skip] existing OASIS template -> {output_dir}")
        return
    seed_pool = read_json(args.seed_pool_json.expanduser().resolve())
    seeds = list(seed_pool.get("seed_posts") or [])[: args.template_posts]
    if not seeds:
        raise ValueError(f"No seed_posts in {args.seed_pool_json}")
    if args.dry_run:
        _write_dry_run_template(output_dir, args, seeds)
        print(f"[dry-run] OASIS template -> {output_dir}")
        return
    _write_live_template(output_dir, args, seeds)
    print(f"[done] OASIS template -> {output_dir}")


def _write_live_template(output_dir: Path, args: argparse.Namespace, seeds: list[dict[str, Any]]) -> None:
    from openai import OpenAI

    from product_reddit_sim.analyzer import analyze_products
    from product_reddit_sim.config_builder import build_config
    from product_reddit_sim.loader import NormalizedProduct
    from product_reddit_sim.persona_gen import generate_personas

    api_key = _require_api_key()
    output_dir.mkdir(parents=True, exist_ok=True)
    products = [
        NormalizedProduct(
            title=str(seed.get("title") or f"{args.domain} discussion"),
            brand=str(seed.get("subreddit") or "Reddit"),
            description=str(seed.get("content") or ""),
            features=[str(seed.get("post_type") or "reddit_post")],
        )
        for seed in seeds
    ]
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    hint = (
        f"This is the Reddit domain '{args.domain}'. Build personas and an "
        "authentic discussion setup for this community, grounded in the provided "
        "post titles and bodies."
    )
    analysis = analyze_products(products, hint, client, args.model, args.seed)
    # The upstream analyzer is product-oriented and may label this as a broad
    # category (for example, "consumer electronics").  Keep the experiment's
    # explicit Reddit domain as the stable run/category identifier.
    analysis.product_category = args.domain
    profiles, persona_prompt, persona_raw = generate_personas(
        analysis,
        args.agents,
        products,
        client,
        args.model,
        args.seed,
        overlay={},
    )
    cli_args = {
        "hours": args.hours,
        "seed_posts": args.seed_posts,
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": "",
        "overlay": {},
        "few_shot_source": "",
        "few_shot_count": 0,
        "few_shot_comments": 0,
        "few_shot_thread_ids": "",
    }
    analysis_record = {
        "product_category": args.domain,
        "key_themes": analysis.key_themes,
        "persona_archetypes": [
            {"name": item.name, "description": item.description, "weight": item.weight}
            for item in analysis.persona_archetypes
        ],
        "discussion_seed_topics": analysis.discussion_seed_topics,
        "_analysis_prompt": analysis._prompt,
        "_analysis_raw_response": analysis._raw_response,
        "_persona_prompt": persona_prompt,
        "_persona_raw_response": persona_raw,
        "_template_domain": args.domain,
        "_template_model": args.model,
    }
    write_json(output_dir / "product_analysis.json", analysis_record)
    seed_prompt, seed_raw = build_config(
        analysis,
        profiles,
        products,
        str(output_dir),
        cli_args,
        client,
        args.model,
        args.seed,
    )
    analysis_record["_seed_prompt"] = seed_prompt
    analysis_record["_seed_raw_response"] = seed_raw
    write_json(output_dir / "product_analysis.json", analysis_record)


def _write_dry_run_template(output_dir: Path, args: argparse.Namespace, seeds: list[dict[str, Any]]) -> None:
    """Write a structurally valid fixture for no-API plumbing validation only."""

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = [
        {
            "user_id": index,
            "username": f"dry_run_user_{index}",
            "karma": 1000 + index,
            "persona": "Dry-run fixture persona. Never use for a live experiment.",
        }
        for index in range(args.agents)
    ]
    config = {
        "simulation_id": output_dir.name,
        "graph_id": args.domain,
        "llm_model": args.model,
        "llm_base_url": args.base_url,
        "time_config": {"total_simulation_hours": args.hours},
        "agent_configs": [{"agent_id": profile["user_id"]} for profile in profiles],
        "event_config": {
            "initial_posts": [],
            "allow_new_threads_during_simulation": False,
            "max_total_threads": args.seed_posts,
        },
    }
    write_json(
        output_dir / "product_analysis.json",
        {
            "product_category": args.domain,
            "key_themes": [args.domain],
            "persona_archetypes": [],
            "discussion_seed_topics": [str(seed.get("title") or "") for seed in seeds[:3]],
            "_dry_run": True,
        },
    )
    write_json(output_dir / "reddit_profiles.json", profiles)
    write_json(output_dir / "simulation_config.json", config)


def _require_api_key() -> str:
    import os

    for key_name in ("LLM_API_KEY", "OPENAI_API_KEY", "OPENAI_KEY"):
        value = os.environ.get(key_name, "").strip()
        if value:
            return value
    raise SystemExit("No API key supplied. The launcher must set LLM_API_KEY for this template job.")


if __name__ == "__main__":
    main()
