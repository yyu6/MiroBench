"""High-level orchestrator for the product-to-Reddit-discussion pipeline.

Takes a product JSON file and runs the six-stage MiroBench generation
pipeline end-to-end:

1. Load normalized product records.
2. Analyze the product space → archetypes + seed topics (LLM call).
3. Generate concrete personas for each archetype (LLM call).
4. Build the OASIS simulation config + seed posts (LLM call).
5. Run the OASIS simulation (vanilla baseline or GEO-patched backbone).
6. Export the SQLite trace into ``discussion.json`` + ``discussion.md``.

The heavy lifting lives in the sibling modules of this package; this file
coordinates ordering, environment loading, output folders, and the
reproducible run-config audit trail.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _load_openai_client():
    """Lazily import the OpenAI client with a helpful error message."""

    try:
        from openai import OpenAI as _OpenAI
        return _OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Could not import `OpenAI`. Install with: pip install 'openai>=1.0.0'"
        ) from exc


def _sha256(path: str) -> str:
    """Stable checksum of the input product file."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_output_dir(base: str, category: str) -> tuple[str, str]:
    """Create one timestamped output directory; return ``(path, run_id)``."""

    slug = category.replace(" ", "_").replace("/", "_")[:40]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{slug}_{ts}"
    path = os.path.join(base, run_id)
    os.makedirs(path, exist_ok=True)
    return path, run_id


def _require_env(key: str) -> str:
    """Read a required env var or exit with a clear error."""

    val = os.environ.get(key)
    if not val:
        print(
            f"ERROR: {key} not set. Add it to a .env file at the repo root "
            "(or your shell environment). See .env.example."
        )
        sys.exit(1)
    return val


def generate_one_run(args: Any) -> Path:
    """Run one full product → discussion pipeline.

    Args:
        args: An ``argparse.Namespace`` with the fields populated by
            ``mirobench.cli``'s ``generate`` subparser.

    Returns:
        Absolute path to the generated ``discussion.json``.
    """

    OpenAI = _load_openai_client()

    from .loader import load_products
    from .analyzer import analyze_products
    from .persona_gen import generate_personas
    from .config_builder import build_config
    from .runner import run_simulation
    from .exporter import export_discussion

    started_at = datetime.now().isoformat()

    products_json = args.products_json
    if not os.path.exists(products_json):
        print(f"ERROR: File not found: {products_json}")
        sys.exit(1)

    api_key = _require_env("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")
    reasoning_effort = os.environ.get("LLM_REASONING_EFFORT", "")

    client = OpenAI(api_key=api_key, base_url=base_url)
    cli_args: Dict[str, Any] = {
        "hours": args.hours,
        "rounds": args.rounds,
        "seed_posts": args.seed_posts,
        "few_shot_source": args.few_shot_source,
        "few_shot_count": args.few_shot_count,
        "few_shot_comments": args.few_shot_comments,
        "few_shot_thread_ids": args.few_shot_thread_ids,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "base_url": base_url,
        "discussion_backbone": args.discussion_backbone,
    }
    overlay: Dict[str, Any] = {}
    if args.overlay:
        overlay = json.loads(Path(args.overlay).read_text(encoding="utf-8"))
    cli_args["overlay"] = overlay

    # 1. Load products
    print(f"\n[1/6] Loading products from {products_json}...")
    products = load_products(products_json)
    print(f"      → {len(products)} products loaded")

    # 2. Analyze
    print("\n[2/6] Analyzing product category and generating persona archetypes...")
    analysis = analyze_products(products, args.hint, client, model, args.seed)
    print(f"      → Category: {analysis.product_category}")
    print(f"      → Archetypes: {[a.name for a in analysis.persona_archetypes]}")

    output_dir, run_id = _make_output_dir(args.output_dir, analysis.product_category)
    print(f"\n      Output: {output_dir}")

    analysis_record: Dict[str, Any] = {
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

    # 3. Personas
    print(f"\n[3/6] Generating {args.agents} personas...")
    profiles, persona_prompt, persona_raw = generate_personas(
        analysis, args.agents, products, client, model, args.seed, overlay=overlay
    )
    print(f"      → {len(profiles)} personas generated")

    analysis_record["_persona_prompt"] = persona_prompt
    analysis_record["_persona_raw_response"] = persona_raw
    analysis_path = os.path.join(output_dir, "product_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_record, f, ensure_ascii=False, indent=2)

    # 4. Config
    print("\n[4/6] Generating seed posts and building OASIS config...")
    seed_prompt, seed_raw = build_config(
        analysis, profiles, products, output_dir, cli_args, client, model, args.seed
    )
    config_path = os.path.join(output_dir, "simulation_config.json")

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

    # 5. Simulate
    print(f"\n[5/6] Running OASIS Reddit simulation ({args.rounds} rounds max)...")
    run_simulation(
        config_path,
        max_rounds=args.rounds,
        discussion_backbone=args.discussion_backbone,
    )

    # 6. Export
    db_path = os.path.join(output_dir, "reddit_simulation.db")
    if not os.path.exists(db_path):
        print(f"\nWARNING: Simulation DB not found at {db_path}. Export skipped.")
        sys.exit(1)

    print("\n[6/6] Exporting discussion to JSON and Markdown...")
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

    finished_at = datetime.now().isoformat()
    run_config = {
        "run_id": run_id,
        "input_file": os.path.abspath(products_json),
        "input_file_sha256": _sha256(products_json),
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

    print(f"\n{'=' * 60}")
    print("SIMULATION COMPLETE")
    print(f"  Output: {output_dir}")
    print(f"  Discussion JSON: {json_path}")
    print(f"{'=' * 60}\n")

    return Path(json_path)
