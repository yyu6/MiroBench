#!/usr/bin/env python3
"""
run_discussion.py — Generate Reddit-style discussion threads using SenseNova API.

This is the simulation engine called by the calibration pipeline.
It replaces the OASIS-based backbone with direct SenseNova API calls.

Usage:
    python run_discussion.py <products_json> [options]

Options mirror the calibration runner's reference_run_config:
    --agents N          Number of agent personas to simulate (default: 50)
    --hours N           Simulated hours (default: 24)
    --rounds N          Max discussion rounds (default: 24)
    --seed-posts N      Seed posts per run (default: 4)
    --seed N            Random seed (default: 42)
    --overlay PATH      Calibration overlay JSON to apply
    --output-dir DIR    Output directory for generated threads
    --few-shot-source DIR   Directory with real threads for few-shot examples
    --few-shot-count N      Number of few-shot examples (default: 5)
    --hint TEXT         Optional topic hint
    --discussion-backbone   vanilla_oasis or geo_patched (default: geo_patched)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# SenseNova API client
# ---------------------------------------------------------------------------
def _get_client() -> OpenAI:
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://token.sensenova.cn/v1")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set. Configure .env or export it.")
    return OpenAI(api_key=api_key, base_url=base_url)


def _call_mimo(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 1.0,
    max_tokens: int = 2048,
) -> str:
    """Single chat completion call to MiMo."""
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            msg = completion.choices[0].message
            text = msg.content or getattr(msg, 'reasoning', None) or getattr(msg, 'reasoning_content', None) or ""
            return text
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"SenseNova API call failed after 3 attempts: {e}") from e


# ---------------------------------------------------------------------------
# Few-shot loading
# ---------------------------------------------------------------------------
def _load_few_shot_examples(few_shot_dir: Path, count: int) -> list[dict]:
    """Load real discussion threads as few-shot examples."""
    examples = []
    jsonl_files = list(few_shot_dir.glob("*.comments.jsonl"))
    random.shuffle(jsonl_files)
    for f in jsonl_files[:count]:
        try:
            lines = f.read_text(encoding="utf-8").strip().split("\n")
            comments = [json.loads(line) for line in lines if line.strip()]
            examples.append({"source": f.stem, "comments": comments})
        except Exception:
            continue
    return examples


# ---------------------------------------------------------------------------
# Discussion generation
# ---------------------------------------------------------------------------
PERSONA_SYSTEM_PROMPT = """You are simulating a Reddit-style product discussion community.
Generate realistic user personas with diverse backgrounds, opinions, and communication styles.
Each persona should have a username, karma range, and personality traits that affect how they post."""

POST_GENERATION_PROMPT = """You are simulating a Reddit discussion about products.
Given the product descriptions below, generate {n_posts} realistic discussion posts.

Product context:
{product_context}

{overlay_guidance}

For each post, generate:
- A post title and body (like a Reddit self-post or link post)
- 3-8 top-level comments with varying opinions
- Some comments should have nested replies (1-3 levels deep)
- Include disagreements, questions, recommendations, and casual banter
- Vary comment lengths (short quips to detailed paragraphs)
- Include realistic Reddit tropes: edits, awards mentions, TL;DRs

{few_shot_examples}

Output valid JSON matching this schema:
{{
  "posts": [
    {{
      "post_id": 1,
      "author": "username",
      "content": "Post text...",
      "likes": N,
      "comments": [
        {{
          "comment_id": 1,
          "author": "commenter",
          "content": "Reply text...",
          "depth": 0,
          "likes": N,
          "replies": [...]
        }}
      ]
    }}
  ]
}}"""


def _build_product_context(products_json: Path, seed_posts: int) -> str:
    """Build product context string from JSON file."""
    products = json.loads(products_json.read_text(encoding="utf-8"))
    if isinstance(products, list):
        selected = random.sample(products, min(seed_posts, len(products)))
    else:
        selected = [products]

    context_parts = []
    for p in selected:
        name = p.get("card_name") or p.get("name") or p.get("product_name", "Unknown")
        desc = str(p.get("description", ""))[:500]  # Truncate long descriptions
        context_parts.append(f"- {name}: {desc}")
    return "\n".join(context_parts)


def _build_few_shot_text(examples: list[dict]) -> str:
    """Format few-shot examples for the prompt."""
    if not examples:
        return ""
    parts = ["Reference real discussions for style:"]
    for i, ex in enumerate(examples[:3]):
        lines = [f"  Example {i+1} ({ex['source']}):"]
        for c in ex["comments"][:5]:
            author = c.get("author", "user")
            content = str(c.get("content", ""))[:200]
            lines.append(f"    [{author}]: {content}")
        parts.append("\n".join(lines))
    return "\n".join(parts)


def _parse_discussion_json(raw: str) -> dict[str, Any]:
    """Extract JSON from API response, handling markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # Remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise ValueError(f"Could not parse JSON from API response:\n{text[:500]}")


def _assign_depths(comments: list[dict], depth: int = 0) -> list[dict]:
    """Ensure all comments have correct depth values."""
    for c in comments:
        c["depth"] = depth
        if c.get("replies"):
            _assign_depths(c["replies"], depth + 1)
    return comments


def generate_discussion(
    client: OpenAI,
    model: str,
    products_json: Path,
    output_dir: Path,
    agents: int = 50,
    hours: int = 24,
    rounds: int = 24,
    seed_posts: int = 4,
    seed: int = 42,
    overlay: dict | None = None,
    few_shot_dir: Path | None = None,
    few_shot_count: int = 5,
    hint: str | None = None,
) -> dict[str, Any]:
    """Generate a single discussion thread."""
    random.seed(seed)

    # Build context
    product_context = _build_product_context(products_json, seed_posts)

    # Few-shot examples
    few_shot_text = ""
    if few_shot_dir and few_shot_dir.exists():
        examples = _load_few_shot_examples(few_shot_dir, few_shot_count)
        few_shot_text = _build_few_shot_text(examples)

    # Overlay guidance
    overlay_guidance = ""
    if overlay:
        persona_guidance = overlay.get("persona.generation_guidance", "")
        comment_guidance = overlay.get("prompt.comment_style_guidance", "")
        parts = []
        if persona_guidance:
            parts.append(f"Persona guidance: {persona_guidance}")
        if comment_guidance:
            parts.append(f"Comment style guidance: {comment_guidance}")
        overlay_guidance = "\n".join(parts)

    # Hint
    if hint:
        overlay_guidance += f"\nSpecial focus: {hint}"

    # Generate
    prompt = POST_GENERATION_PROMPT.format(
        n_posts=seed_posts,
        product_context=product_context,
        overlay_guidance=overlay_guidance or "(No special guidance — generate natural discussions.)",
        few_shot_examples=few_shot_text or "(No few-shot examples available.)",
    )

    messages = [
        {"role": "system", "content": PERSONA_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    raw_response = _call_mimo(client, model, messages, temperature=1.0, max_tokens=8192)
    discussion = _parse_discussion_json(raw_response)

    # Ensure meta
    if "meta" not in discussion:
        discussion["meta"] = {}
    discussion["meta"].update({
        "product_category": products_json.parent.name if products_json.parent else "unknown",
        "agent_count": agents,
        "seed": seed,
        "simulated_hours": hours,
        "rounds": rounds,
        "run_id": f"mimo_run_{uuid.uuid4().hex[:8]}",
        "model": model,
    })

    # Fix depths
    for post in discussion.get("posts", []):
        if post.get("comments"):
            _assign_depths(post["comments"])

    return discussion


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate discussion threads using SenseNova API")
    parser.add_argument("input_file", help="Product descriptions JSON file")
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--seed-posts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlay", type=str, default=None, help="Overlay JSON path")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--few-shot-source", type=str, default=None)
    parser.add_argument("--few-shot-count", type=int, default=5)
    parser.add_argument("--hint", type=str, default=None)
    parser.add_argument("--discussion-backbone", type=str, default="geo_patched")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    args = parser.parse_args()

    # Load .env
    from dotenv import load_dotenv
    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env")

    client = _get_client()
    model = args.model or os.environ.get("CALIBRATION_MODEL", "sensenova-6.7-flash-lite")

    overlay = None
    if args.overlay and Path(args.overlay).exists():
        overlay = json.loads(Path(args.overlay).read_text(encoding="utf-8"))

    few_shot_dir = Path(args.few_shot_source) if args.few_shot_source else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate multiple threads (one per round, up to --rounds)
    all_threads = []
    for round_num in range(args.rounds):
        print(f"  Round {round_num + 1}/{args.rounds}...", flush=True)
        seed = args.seed + round_num
        try:
            discussion = generate_discussion(
                client=client,
                model=model,
                products_json=Path(args.input_file),
                output_dir=output_dir,
                agents=args.agents,
                hours=args.hours,
                rounds=args.rounds,
                seed_posts=args.seed_posts,
                seed=seed,
                overlay=overlay,
                few_shot_dir=few_shot_dir,
                few_shot_count=args.few_shot_count,
                hint=args.hint,
            )

            # Save each thread in its own directory
            thread_dir = output_dir / f"thread_{round_num:03d}"
            thread_dir.mkdir(parents=True, exist_ok=True)
            (thread_dir / "discussion.json").write_text(
                json.dumps(discussion, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            all_threads.append(discussion)
            print(f"    ✓ Saved {thread_dir.name} ({len(discussion.get('posts', []))} posts)")

            # Rate limit
            time.sleep(1)

        except Exception as e:
            print(f"    ✗ Round {round_num + 1} failed: {e}", file=sys.stderr)
            continue

    print(f"\nGenerated {len(all_threads)} threads in {output_dir}")


if __name__ == "__main__":
    main()