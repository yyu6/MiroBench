#!/usr/bin/env python3
"""Single CLI entrypoint for the product-to-Reddit-discussion pipeline.

This script is the top-level orchestrator for the current GEO simulation flow.
It turns a product JSON file into one simulated Reddit discussion run through
six stages:

1. load normalized product records
2. analyze the product space and infer discussion archetypes
3. generate personas
4. build OASIS/MiroFish config plus seed posts
5. run the simulation
6. export the raw DB into discussion artifacts

The heavy lifting lives in `product_reddit_sim/*`; this file mainly coordinates
the order of operations, environment loading, output folders, and reproducible
run metadata.

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
from pathlib import Path

from dotenv import load_dotenv

_GEO_ROOT = os.path.dirname(os.path.abspath(__file__))
_MIROFISH_ENV_CANDIDATES = [
    os.path.join(_GEO_ROOT, "third_party", "MiroFish", ".env"),
    os.path.join(_GEO_ROOT, "MiroFish", ".env"),
]
for _env_path in _MIROFISH_ENV_CANDIDATES:
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        break
else:
    load_dotenv()


def _load_openai_client():
    """Import the OpenAI client, re-execing into `.venv` if needed.

    Users often launch this script directly from a shell where `python3` points
    to a global pyenv installation. If that environment has an incompatible
    `openai` package, the product-analysis stages fail before the simulation
    even starts. The repo venv is the canonical environment for this script, so
    we automatically hop into it when the current interpreter cannot provide
    `openai.OpenAI`.
    """

    try:
        from openai import OpenAI as _OpenAI
        return _OpenAI
    except Exception as exc:
        venv_python = os.path.join(_GEO_ROOT, ".venv", "bin", "python")
        if (
            os.path.exists(venv_python)
            and os.path.abspath(sys.executable) != os.path.abspath(venv_python)
            and os.environ.get("GEO_SKIP_VENV_REEXEC") != "1"
        ):
            print(
                "Detected an incompatible `openai` installation in "
                f"{sys.executable}. Re-running with {venv_python}...",
                file=sys.stderr,
            )
            env = os.environ.copy()
            env["GEO_SKIP_VENV_REEXEC"] = "1"
            os.execve(
                venv_python,
                [venv_python, os.path.abspath(__file__), *sys.argv[1:]],
                env,
            )
        raise ImportError(
            "Could not import `OpenAI` from the installed `openai` package. "
            "Use GEO's .venv or install `openai>=1.0.0` in the current interpreter."
        ) from exc


OpenAI = _load_openai_client()

from product_reddit_sim.loader import load_products
from product_reddit_sim.analyzer import analyze_products
from product_reddit_sim.persona_gen import generate_personas
from product_reddit_sim.config_builder import build_config
from product_reddit_sim.runner import run_simulation
from product_reddit_sim.exporter import export_discussion


def parse_args():
    """Parse CLI arguments for one simulation run."""

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
    p.add_argument("--seed-posts", type=int, default=3,
                   help="Number of initial seed threads to start with (default: 3)")
    p.add_argument("--few-shot-source", type=str, default=None,
                   help="Optional real discussion manifest/bundle used as few-shot style examples for generation")
    p.add_argument("--few-shot-count", type=int, default=0,
                   help="Number of few-shot discussion examples to inject into generation prompts (default: 0)")
    p.add_argument("--few-shot-comments", type=int, default=2,
                   help="Visible comments kept per few-shot example (default: 2)")
    p.add_argument("--few-shot-thread-ids", type=str, default=None,
                   help="JSON file listing allowed thread IDs for few-shot selection (train-only filtering)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument(
        "--discussion-backbone",
        type=str,
        default="vanilla_oasis",
        choices=["geo_patched", "vanilla_oasis"],
        help=(
            "Which discussion-generation backbone to use. "
            "`vanilla_oasis` runs the restored vendor OASIS baseline; "
            "`geo_patched` uses GEO's preserved patch wrapper."
        ),
    )
    p.add_argument("--output-dir", type=str, default="artifacts/simulations",
                   help="Output directory (default: ./artifacts/simulations)")
    p.add_argument("--overlay", type=str, default=None,
                   help="Path to calibration overlay JSON (optional)")
    return p.parse_args()


def _sha256(path: str) -> str:
    """Return a stable checksum of the input product file for reproducibility."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_output_dir(base: str, category: str) -> tuple[str, str]:
    """Create one timestamped output directory and return `(path, run_id)`."""

    slug = category.replace(" ", "_").replace("/", "_")[:40]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{slug}_{ts}"
    path = os.path.join(base, run_id)
    os.makedirs(path, exist_ok=True)
    return path, run_id


def _require_env(key: str) -> str:
    """Read a required environment variable or exit with a clear error."""

    val = os.environ.get(key)
    if not val:
        print(
            "ERROR: "
            f"{key} not set. Add it to third_party/MiroFish/.env "
            "(or the legacy MiroFish/.env path)."
        )
        sys.exit(1)
    return val


def main():
    """Run one complete simulation job from products to exported discussion."""

    args = parse_args()
    started_at = datetime.now().isoformat()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.exists(args.products_json):
        print(f"ERROR: File not found: {args.products_json}")
        sys.exit(1)

    api_key = _require_env("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")

    # One client is shared across analysis/persona/config generation so the run
    # uses a consistent model endpoint.
    client = OpenAI(api_key=api_key, base_url=base_url)
    cli_args = {
        "hours": args.hours,
        "rounds": args.rounds,
        "seed_posts": args.seed_posts,
        "few_shot_source": args.few_shot_source,
        "few_shot_count": args.few_shot_count,
        "few_shot_comments": args.few_shot_comments,
        "few_shot_thread_ids": args.few_shot_thread_ids,
        "model": model,
        "base_url": base_url,
        "discussion_backbone": args.discussion_backbone,
    }
    overlay = {}
    if args.overlay:
        overlay = json.loads(Path(args.overlay).read_text(encoding="utf-8"))
    cli_args["overlay"] = overlay

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

    # Persist the LLM-facing analysis artifact early. Later steps append their
    # prompts/raw responses so each run keeps an audit trail in one place.
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
        analysis, args.agents, products, client, model, args.seed, overlay=overlay
    )
    print(f"      → {len(profiles)} personas generated")

    # Append persona-generation audit data so the run remains inspectable.
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
    config_path = os.path.join(output_dir, "simulation_config.json")

    # Append seed-post generation audit data after config creation.
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis_record = json.load(f)
    analysis_record["_seed_prompt"] = seed_prompt
    analysis_record["_seed_raw_response"] = seed_raw
    with open(config_path, "r", encoding="utf-8") as f:
        config_record = json.load(f)
    analysis_record["_generation_few_shot_examples"] = (
        (config_record.get("prompt_config") or {}).get("generation_few_shot_examples")
        or []
    )
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_record, f, ensure_ascii=False, indent=2)
    print(f"      → Config written: {config_path}")

    # ── Step 5: Run simulation ────────────────────────────────────────────────
    print(f"\n[5/6] Running OASIS Reddit simulation ({args.rounds} rounds max)...")
    run_simulation(
        config_path,
        max_rounds=args.rounds,
        discussion_backbone=args.discussion_backbone,
    )

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

    # ── Save run_config.json (reproducibility + provenance record) ────────────
    finished_at = datetime.now().isoformat()
    relative_input = os.path.relpath(os.path.abspath(args.products_json), _GEO_ROOT)
    run_config = {
        "run_id": run_id,
        "input_file": relative_input,
        "input_file_sha256": _sha256(args.products_json),
        "hint": args.hint,
        "agents": args.agents,
        "hours": args.hours,
        "rounds": args.rounds,
        "seed_posts": args.seed_posts,
        "few_shot_source": args.few_shot_source,
        "few_shot_count": args.few_shot_count,
        "few_shot_comments": args.few_shot_comments,
        "few_shot_thread_ids": args.few_shot_thread_ids,
        "seed": args.seed,
        "discussion_backbone": args.discussion_backbone,
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
