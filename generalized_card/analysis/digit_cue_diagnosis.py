#!/usr/bin/env python3
"""Reproduce the digit-cue quantifier-guard numbers (`--digit-cue-guard`).

`sentence_rhythm.py`'s "digit" habit asks the Writer to cite a real quantity
"as a figure rather than described in words". Measured on the v103 N=10
artifact against 424 evaluation-excluded real camera threads: a comment
containing a bare `1` is 8.2x more likely in generated text when that `1` is
an ordinary quantifier with no enumeration/fraction/price context ("1 thing
I'd check", "that 1 folder") than when it is one of those genuinely
numeric uses -- 1.7x there. Real writers do write a bare "1" as a plain
quantifier too (55% of their own bare-1 comments), so this is not "real never
does this"; it is "generated does the same thing roughly 8x as often per
comment". The excess concentrates in one sub-pattern, not the raw digit rate.

No API call, no model. Pure regex over already-generated/already-scraped
text.

    python3 generalized_card/analysis/digit_cue_diagnosis.py
    python3 generalized_card/analysis/digit_cue_diagnosis.py --run <other run tag>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
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
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
REAL_DIR = REPO / "data/raw/discussions/camera_product"

# A "1" not already glued to a digit, decimal point, currency sign, or ordinal
# suffix -- i.e. a genuinely bare token, not part of "$100", "1.5", "1st".
BARE_ONE = re.compile(r"(?<![0-9./$#-])\b1\b(?![0-9./$%)-])")
BARE_ZERO_OR_ONE = re.compile(r"(?<![0-9./$#-])\b[01]\b(?![0-9./$%)-])")
# "1 thing", "1 folder" -- a bare 1 immediately followed by an ordinary word,
# with no numeric/enumeration marker anywhere in a small window around it.
PLAIN_QUANTIFIER_WINDOW = re.compile(r"\b1\s+[a-z]+\b", re.I)
ENUM_OR_FACT_WINDOW = re.compile(r"1\)|1/|1-2|1st|#1|\$1|1x|1:")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def classify(text: str) -> str | None:
    """Which sub-pattern the first bare `1` in this comment belongs to."""

    match = BARE_ONE.search(text)
    if not match:
        return None
    window = text[max(0, match.start() - 10) : match.end() + 15]
    if ENUM_OR_FACT_WINDOW.search(window):
        return "enum_or_fact"
    if PLAIN_QUANTIFIER_WINDOW.search(window):
        return "plain_quantifier"
    return "other"


def generated_texts(run: Path) -> list[str]:
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


def real_texts() -> list[str]:
    pool = _load_json(SEED_POOL)
    seed_ids = {str(row["source_raw_post_id"]) for row in pool["seed_posts"]}
    texts: list[str] = []
    for product_dir in sorted(p for p in REAL_DIR.iterdir() if p.is_dir()):
        try:
            comments_by_thread, _ = load_real_comments(product_dir)
        except FileNotFoundError:
            continue
        for thread_id, comments in comments_by_thread.items():
            if thread_id in seed_ids:
                continue
            texts.extend(comment.text for comment in comments)
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    args = parser.parse_args()

    gen = generated_texts(Path(args.run))
    real = real_texts()

    bare01_gen = sum(1 for t in gen if BARE_ZERO_OR_ONE.search(t))
    bare01_real = sum(1 for t in real if BARE_ZERO_OR_ONE.search(t))
    print("== any bare 0/1 ==")
    print(f"generated: {bare01_gen}/{len(gen)} = {bare01_gen/len(gen):.4f}")
    print(f"real:      {bare01_real}/{len(real)} = {bare01_real/len(real):.4f}")
    print(f"ratio: {(bare01_gen/len(gen)) / (bare01_real/len(real)):.2f}x\n")

    gen_counts = Counter(filter(None, (classify(t) for t in gen)))
    real_counts = Counter(filter(None, (classify(t) for t in real)))
    print("== bare-1 sub-pattern breakdown ==")
    print(f"generated: {dict(gen_counts)} of {len(gen)} comments")
    print(f"real:      {dict(real_counts)} of {len(real)} comments\n")

    for key in ("plain_quantifier", "enum_or_fact"):
        g_rate = gen_counts.get(key, 0) / len(gen)
        r_rate = real_counts.get(key, 0) / len(real)
        ratio = g_rate / r_rate if r_rate else float("inf")
        print(f"{key:18s} generated={g_rate:.5f} real={r_rate:.5f} ratio={ratio:.2f}x")

    gen_bare1_total = sum(gen_counts.values())
    real_bare1_total = sum(real_counts.values())
    if gen_bare1_total and real_bare1_total:
        print(
            f"\nshare of own bare-1 that is plain_quantifier: "
            f"generated={gen_counts.get('plain_quantifier', 0)/gen_bare1_total:.3f} "
            f"real={real_counts.get('plain_quantifier', 0)/real_bare1_total:.3f}"
        )


if __name__ == "__main__":
    main()
