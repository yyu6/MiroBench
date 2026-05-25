"""Retrieve and render compact few-shot discussion examples for generation.

This module provides the shared logic used by the simulation pipeline when we
want to improve generation quality with real discussion snippets.

The generation use case is different from the judge use case:
- we want short, style-oriented examples rather than full audit records
- examples should be relevant to the current category/themes when possible
- prompt size must stay bounded because action prompts are repeated every round

The helpers here therefore:
1. load threads from an actual manifest, actual bundle, or generated discussion
2. rank them against the current simulation query
3. serialize them into short prompt-ready examples
4. render a compact textual block that other prompts can embed
"""
from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .thread_loader import (
    auto_detect_target_kind,
    load_actual_threads,
    load_generated_threads,
    load_threads_from_manifest,
    sanitize_text,
)


def select_generation_few_shot_examples(
    source_path: str | Path,
    query_text: str,
    max_examples: int = 2,
    max_comments: int = 2,
    seed: int = 42,
    allowed_thread_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select compact discussion examples relevant to one simulation run.

    Args:
        source_path: A manifest, actual bundle JSON, or generated discussion
            file/directory. Real discussion sources are preferred.
        query_text: Category/themes/topics string describing the run.
        max_examples: Maximum number of examples to keep.
        max_comments: Maximum visible comments kept per example.
        seed: Deterministic tiebreak seed.
        allowed_thread_ids: If provided, only threads whose ID is in this set
            are eligible. Used to restrict few-shot examples to the train split.
    """

    if not source_path or max_examples <= 0:
        return []

    threads = _load_candidate_threads(
        source_path=source_path,
        max_examples=max_examples,
        max_comments=max_comments,
    )
    if not threads:
        return []

    # Filter to allowed thread IDs (e.g., train-only)
    if allowed_thread_ids is not None:
        threads = [t for t in threads if _thread_matches_allowed_ids(t, allowed_thread_ids)]
        if not threads:
            return []

    rng = random.Random(seed)
    query_tokens = _tokenize(query_text)
    ranked: list[tuple[float, float, Any]] = []

    for thread in threads:
        thread_text = sanitize_text(
            "\n".join(
                [
                    str(thread.topic or ""),
                    str(thread.subreddit or ""),
                    str(thread.title or ""),
                    str(thread.body or ""),
                    " ".join(comment.content for comment in thread.comments[:max_comments]),
                ]
            )
        )
        overlap = len(query_tokens & _tokenize(thread_text))
        richness = min(len(thread.comments), max_comments) * 0.5
        question_bonus = 0.5 if "?" in f"{thread.title}\n{thread.body}" else 0.0
        score = float(overlap) + richness + question_bonus
        ranked.append((score, rng.random(), thread))

    ranked.sort(key=lambda item: (-item[0], -item[1], str(item[2].thread_id)))

    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for _, _, thread in ranked:
        dedupe_key = (
            _truncate_text(str(thread.title or ""), 100).lower(),
            _truncate_text(str(thread.body or ""), 140).lower(),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        selected.append(
            {
                "thread_id": str(thread.thread_id),
                "source_kind": str(thread.source_kind),
                "topic": _truncate_text(str(thread.topic or ""), 80),
                "subreddit": _truncate_text(str(thread.subreddit or ""), 60),
                "title": _truncate_text(str(thread.title or ""), 140),
                "body": _truncate_text(str(thread.body or ""), 280),
                "comments": [
                    _truncate_text(sanitize_text(comment.content), 180)
                    for comment in thread.comments[:max_comments]
                    if sanitize_text(comment.content).strip()
                ],
            }
        )
        if len(selected) >= max_examples:
            break

    return selected


def render_generation_few_shot_block(
    examples: list[dict[str, Any]] | None,
    heading: str = "REAL REDDIT STYLE EXAMPLES",
) -> str:
    """Render compact examples into a prompt block.

    The block is intentionally short and framed as a style anchor, not a
    content template. This reduces verbatim copying and keeps token usage
    manageable for repeated action prompts.
    """

    examples = list(examples or [])
    if not examples:
        return ""

    lines = [
        f"{heading}:",
        "Use these only as style anchors for tone, specificity, disagreement, and messiness.",
        "Do not copy the products, exact wording, or specific opinions.",
        "Realistic threads allow some overlap, but most commenters should not sound like interchangeable rewrites of the same answer.",
    ]
    style_signals = summarize_generation_style_signals(examples)
    if style_signals:
        lines.append(
            "Common reply shapes in these examples: "
            + ", ".join(style_signals)
            + ". Keep that mix instead of making every reply sound the same."
        )
    for idx, example in enumerate(examples, start=1):
        lines.append(f"[EXAMPLE_{idx}]")
        if example.get("topic"):
            lines.append(f"topic: {sanitize_text(example['topic'])}")
        if example.get("subreddit"):
            lines.append(f"subreddit: {sanitize_text(example['subreddit'])}")
        if example.get("title"):
            lines.append(f"title: {sanitize_text(example['title'])}")
        post_text = sanitize_text(example.get("body") or example.get("title") or "")
        lines.append(f"post: {post_text}")
        comments = example.get("comments") or []
        if comments:
            lines.append("visible replies:")
            for comment in comments:
                lines.append(f"- {sanitize_text(comment)}")
    return "\n".join(lines)


def summarize_generation_style_signals(
    examples: list[dict[str, Any]] | None,
) -> list[str]:
    """Extract coarse reply-style signals from few-shot examples.

    The goal is not perfect linguistic classification. We only need a small,
    bounded set of human-readable hints that remind the generator that real
    threads mix calm advice, lived datapoints, short blunt replies, and sharper
    pushback, not one repeated perspective.
    """

    examples = list(examples or [])
    counter: Counter[str] = Counter()
    for example in examples:
        for raw_comment in example.get("comments") or []:
            comment = sanitize_text(raw_comment).strip()
            if not comment:
                continue
            tokens = re.findall(r"[a-z0-9']+", comment.lower())
            if re.search(r"\b(i|i'm|i’ve|i've|my|me|we|our|us)\b", comment.lower()):
                counter["personal datapoints"] += 1
            if len(tokens) <= 12:
                counter["short blunt replies"] += 1
            if "?" in comment:
                counter["skeptical or probing questions"] += 1
            if re.search(
                r"\b(lol|lmao|wtf|shit|hell|stupid|dumb|garbage|trash|rip)\b",
                comment.lower(),
            ):
                counter["snark or aggressive language"] += 1
            if re.search(
                r"\b(depends|if you|worth|keep|use|works|unless|better)\b",
                comment.lower(),
            ):
                counter["calm practical advice"] += 1

    ordered = [label for label, _ in counter.most_common(5)]
    return ordered


def _load_candidate_threads(
    source_path: str | Path,
    max_examples: int,
    max_comments: int,
):
    """Load a bounded candidate pool from the provided example source.

    Supports single files (bundle JSON, manifest, generated discussion),
    generated sim directories, and **category directories** containing
    product subdirs with ``<product>.jsonl`` bundles.
    """

    source_path = Path(source_path)
    candidate_count = max(12, max_examples * 8)

    # Category directory: walk subdirs looking for actual bundle JSONLs
    if source_path.is_dir() and not (source_path / "discussion.json").exists():
        return _load_from_category_dir(
            source_path,
            candidate_count=candidate_count,
            max_comments=max_comments,
        )

    target_kind = auto_detect_target_kind(source_path)

    if target_kind == "actual_manifest":
        return load_threads_from_manifest(
            source_path,
            max_items=max(20, candidate_count),
            threads_per_item=1,
            max_comments=max_comments,
        )
    if target_kind == "actual_bundle":
        return load_actual_threads(
            bundle_json_path=source_path,
            max_threads=candidate_count,
            max_comments=max_comments,
        )
    if target_kind == "generated":
        return load_generated_threads(
            source_path,
            max_threads=candidate_count,
            max_comments=max_comments,
        )
    return []


def _load_from_category_dir(
    category_dir: Path,
    candidate_count: int,
    max_comments: int,
):
    """Walk product subdirs in a category dir and load actual bundle threads.

    For each product subdir, find the ``<product>.jsonl`` bundle (the one
    without ``.comments.`` or ``.raw.`` in the name) and load a few threads.
    Threads from all products are pooled together.
    """
    import random as _rng

    all_threads = []
    subdirs = sorted(category_dir.iterdir())
    _rng.shuffle(subdirs)  # mix products for diversity

    per_product_limit = max(2, candidate_count // max(1, len(subdirs)))

    for sub in subdirs:
        if not sub.is_dir():
            continue
        # Find the main bundle JSONL (e.g. aeroplan_credit_card.jsonl)
        bundle_candidates = [
            f for f in sub.glob("*.jsonl")
            if ".comments." not in f.name and ".raw." not in f.name
        ]
        if not bundle_candidates:
            continue
        bundle_path = bundle_candidates[0]
        try:
            threads = load_actual_threads(
                bundle_json_path=bundle_path,
                max_threads=per_product_limit,
                max_comments=max_comments,
            )
            for thread in threads:
                setattr(thread, "_source_product", sub.name)
            all_threads.extend(threads)
        except Exception:
            continue
        if len(all_threads) >= candidate_count:
            break

    return all_threads[:candidate_count]


def _tokenize(text: str) -> set[str]:
    """Lowercase tokenizer for lightweight lexical retrieval."""

    return {token for token in re.findall(r"[a-z0-9]+", sanitize_text(text).lower()) if len(token) >= 2}


def _thread_matches_allowed_ids(thread: Any, allowed_thread_ids: set[str]) -> bool:
    """Return whether a loaded thread matches a raw or composite allowlist id."""
    thread_id = str(getattr(thread, "thread_id", "") or "")
    if thread_id in allowed_thread_ids:
        return True

    source_product = str(getattr(thread, "_source_product", "") or "").strip()
    if source_product and thread_id:
        composite_id = f"{source_product}::{thread_id}"
        if composite_id in allowed_thread_ids:
            return True

    return False


def _truncate_text(text: str, limit: int) -> str:
    """Trim prompt snippets to a bounded length."""

    compact = " ".join(sanitize_text(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
