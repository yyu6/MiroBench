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


def _make_output_dir(base: str, category: str) -> tuple[str, str]:
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
