#!/usr/bin/env python3
"""Reproduce the v107 `--verdict-close-guard` numbers.

`closing_move.py`'s `abstract_verdict_close` habit already measures and
suppresses one family of closing tic ("what matters", "the real thing",
"the part", ...) -- v100's fix for the "that's the part that actually
matters" tic chased since v73. Two things this session found, reading the
v106 gate's actual pairs (`docs/DECISIONS.md` G13):

1. Even where the existing suppression cue reaches the Writer, the move it
   targets is still wildly over-produced -- the fix reduced it, it did not
   close it.
2. A "that's the check" / "a solid check" variant recurs that the existing
   pattern's word list never named (`matters?|counts?|settles?|the real|
   the whole|the part|the only thing|my take|the upshot|bottom line|in the
   end|at the end of the day` -- no "check" or "test").

This measures both, on the last sentence of every 25+-word comment, exactly
the way `closing_move.py` itself measures the existing pattern. No API call,
no model -- pure regex over already-generated/already-scraped text.

    python3 generalized_card/analysis/verdict_close_diagnosis.py
    python3 generalized_card/analysis/verdict_close_diagnosis.py --run <other run tag>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCORER_DIR = REPO / "scripts" / "evaluation"
if str(SCORER_DIR) not in sys.path:
    sys.path.insert(0, str(SCORER_DIR))

from score_thread_semantic_uniformity import load_real_comments  # noqa: E402

DEFAULT_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1"
)
GATE_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "v106_chain_novelty_digit_guard_seed8_20260822_v1"
)
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
REAL_DIR = REPO / "data/raw/discussions/camera_product"

# `abstract_verdict_close`'s pattern, verbatim from closing_move.py.
EXISTING_PATTERN = re.compile(
    r"\b(?:matters?|counts?|settles?|the real|the whole|the part|"
    r"the only thing|my take|the upshot|bottom line|in the end|"
    r"at the end of the day)\b",
    re.I,
)
# The variant this session found reading the v106 gate's actual pairs, which
# the pattern above never named.
CHECK_VARIANT_PATTERN = re.compile(
    r"\b(?:the|a|that's the|only)\s+(?:solid|real|only|actual)?\s*(?:check|test)\b",
    re.I,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_WORDS = 25


def last_sentence(text: str) -> str:
    parts = [part for part in SENTENCE_SPLIT.split(text.strip()) if part.strip()]
    return parts[-1] if parts else text


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _generated_texts(run: Path) -> list[str]:
    texts: list[str] = []

    def walk(comments: list[dict[str, Any]]) -> None:
        for comment in comments:
            texts.append(str(comment.get("content") or ""))
            walk(comment.get("replies") or [])

    for sim_dir in sorted(run.glob("cleaned/run_*_sampled_reddit")):
        discussion = _load_json(sim_dir / "discussion.json")
        for post in discussion.get("posts") or []:
            walk(post.get("comments") or [])
    return texts


def _excluded_real_texts() -> list[str]:
    pool = _load_json(SEED_POOL)
    seed_ids = {str(row["source_raw_post_id"]) for row in pool["seed_posts"]}
    texts: list[str] = []
    for product_dir in sorted(p for p in REAL_DIR.iterdir() if p.is_dir()):
        try:
            comments_by_thread, _ = load_real_comments(product_dir)
        except FileNotFoundError:
            continue
        for thread_id, comments in comments_by_thread.items():
            if thread_id not in seed_ids:
                texts.extend(comment.text for comment in comments)
    return texts


def report(label: str, texts: list[str]) -> None:
    long_ = [text for text in texts if len(text.split()) >= MIN_WORDS]
    existing_hits = sum(1 for text in long_ if EXISTING_PATTERN.search(last_sentence(text)))
    check_hits = sum(1 for text in long_ if CHECK_VARIANT_PATTERN.search(last_sentence(text)))
    union_hits = sum(
        1
        for text in long_
        if EXISTING_PATTERN.search(last_sentence(text)) or CHECK_VARIANT_PATTERN.search(last_sentence(text))
    )
    n = len(long_) or 1
    print(
        f"{label:36s} n={len(long_):5d} existing={existing_hits:4d} ({existing_hits/n:.4f}) "
        f"check_variant={check_hits:4d} ({check_hits/n:.4f}) union={union_hits:4d} ({union_hits/n:.4f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    args = parser.parse_args()

    report("v103 N=10 generated", _generated_texts(Path(args.run)))
    report("v106 gate (seed 8)", _generated_texts(GATE_RUN))
    print()
    report("evaluation-excluded real (camera)", _excluded_real_texts())


if __name__ == "__main__":
    main()
