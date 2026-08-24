#!/usr/bin/env python3
"""Which real features, absent from generated, actually explain each gap?

The user's target is `p ~ 0.5-0.6` on `self_bleu_4` and `self_bertscore_mean_f1`,
which needs the gap essentially closed rather than shrunk. Four consecutive
mechanisms failed because they were chosen from a plausible story and priced
afterwards. This script inverts that: it measures, exactly and for free, **how
much of each gap a real feature accounts for**, before anything is built.

The design is a reverse ablation on the *real* side. Both metrics are means over
unordered within-thread comment pairs, so dropping a set of comments means
dropping their pairs and re-averaging -- exact, no model, no re-scoring. For a
feature the generated thread does not produce at all, or produces at a much lower
rate, the quantity

    real_mean(with the feature's comments dropped) - real_mean

is the amount of the real/generated gap that the feature's absence explains: it
is how much closer real moves to generated once the feature is taken away. Doing
the same on the generated side gives the symmetric check, so a feature is only
credited when the two sides agree about its sign.

Everything is computed from pair files that reproduce the shipped metric first
(rule E6); `self_bleu_4` pairs are computed here with the project's own scorer
functions.

    python3 generalized_card/analysis/feature_leverage.py bleu
    python3 generalized_card/analysis/feature_leverage.py bertscore \\
      --generated-pairs <gen_pairs.json> --real-pairs <real_pairs.json>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
SCORER_DIR = REPO / "scripts" / "evaluation"
if str(SCORER_DIR) not in sys.path:
    sys.path.insert(0, str(SCORER_DIR))

from score_thread_self_bleu import symmetric_pair_bleu, tokenize  # noqa: E402
from score_thread_semantic_uniformity import (  # noqa: E402
    load_generated_comments,
    load_real_comments,
)

RUNS = REPO / "artifacts/generalized_card/runs"
TREATED = "v109_entity_spread_seed8_20260824_v1"
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
REAL_DIR = REPO / "data/raw/discussions/camera_product"
SEED_SUFFIX = re.compile(r"seed(\d+)$")

URL = re.compile(r"https?://|www\.|\[[^\]]+\]\(", re.IGNORECASE)
QUOTE = re.compile(r"^\s*&?gt;|^\s*>", re.MULTILINE)
CAPS = re.compile(r"\b[A-Z]{3,}\b")
LIST = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)", re.MULTILINE)
PARAGRAPHS = re.compile(r"\n\s*\n")
ELLIPSIS = re.compile(r"\.\.\.|…")
EMPH = re.compile(r"\*[^*]+\*|_[^_]+_")

FEATURES: dict[str, Callable[[str], bool]] = {
    "contains a link": lambda t: bool(URL.search(t)),
    "quotes the parent": lambda t: bool(QUOTE.search(t)),
    "has an ALL-CAPS word": lambda t: bool(CAPS.search(t)),
    "has a list shape": lambda t: bool(LIST.search(t)),
    "multi-paragraph": lambda t: bool(PARAGRAPHS.search(t)),
    "has an ellipsis": lambda t: bool(ELLIPSIS.search(t)),
    "has markdown emphasis": lambda t: bool(EMPH.search(t)),
    "under 8 words": lambda t: len(t.split()) < 8,
    "over 120 words": lambda t: len(t.split()) > 120,
    "no terminal punctuation": lambda t: bool(t.strip()) and t.strip()[-1] not in ".!?",
    "opens with 'I'": lambda t: t.strip()[:2].lower() in {"i ", "i'"},
}


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def drop_mean(pairs: list[tuple[float, int, int]], dropped: set[int]) -> float:
    """Thread mean with every pair touching a dropped comment removed."""

    kept = [v for v, a, b in pairs if a not in dropped and b not in dropped]
    return mean(kept)


def report(
    label: str,
    gen_pairs: list[tuple[float, int, int]],
    gen_texts: list[str],
    real_pairs: list[tuple[float, int, int]],
    real_texts: list[str],
) -> None:
    gen_mean, real_mean = mean([v for v, _, _ in gen_pairs]), mean([v for v, _, _ in real_pairs])
    gap = gen_mean - real_mean
    print(f"\n== {label}: how much of the gap each real feature explains ==\n")
    print(f"  generated {gen_mean:.6f}   real {real_mean:.6f}   gap {gap:+.6f}\n")
    print(f"  {'feature':>26s} {'real n':>7s} {'gen n':>6s} {'drop from real':>15s} "
          f"{'share of gap':>13s} {'drop from gen':>14s}")
    rows = []
    for name, test in FEATURES.items():
        real_hits = {i for i, t in enumerate(real_texts) if test(t)}
        gen_hits = {i for i, t in enumerate(gen_texts) if test(t)}
        real_delta = (drop_mean(real_pairs, real_hits) - real_mean) if real_hits else 0.0
        gen_delta = (drop_mean(gen_pairs, gen_hits) - gen_mean) if gen_hits else 0.0
        rows.append((name, len(real_hits), len(gen_hits), real_delta, real_delta / gap, gen_delta))
    for name, rn, gn, rd, share, gd in sorted(rows, key=lambda r: -abs(r[4])):
        print(f"  {name:>26s} {rn:7d} {gn:6d} {rd:+15.6f} {share:12.1%} {gd:+14.6f}")
    print("\n  `drop from real` > 0 means those real comments were *below*-average")
    print("  similarity, so removing them moves real toward generated: the feature's")
    print("  absence in generated explains that much of the gap. Only credit a")
    print("  feature when the generated side has materially fewer of them.")


def load_texts() -> tuple[list[str], list[str], float]:
    sim_dir = next(iter(sorted((RUNS / TREATED).glob("cleaned/run_*_sampled_reddit"))))
    generated_by_thread, _ = load_generated_comments(sim_dir)
    thread_id, comments = next(iter(generated_by_thread.items()))
    shipped = json.loads((sim_dir / "self_bleu_results.json").read_text(encoding="utf-8"))
    shipped_value = next(
        row["self_bleu_4"] for row in shipped["threads"] if row["thread_id"] == thread_id
    )
    pool = json.loads(SEED_POOL.read_text(encoding="utf-8"))
    seed = next(
        row
        for row in pool["seed_posts"]
        if int(row["seed_index"]) == int(SEED_SUFFIX.search(thread_id).group(1))
    )
    real_by_thread, _ = load_real_comments(REAL_DIR / str(seed["source_product_dir"]))
    real = real_by_thread[str(seed["source_raw_post_id"])]
    return [c.text for c in comments], [c.text for c in real], float(shipped_value)


def bleu_pairs(texts: list[str]) -> list[tuple[float, int, int]]:
    toks = [tokenize(t) for t in texts]
    return [
        (symmetric_pair_bleu(toks[i], toks[j], 4), i, j)
        for i in range(len(toks))
        for j in range(i + 1, len(toks))
    ]


def cmd_bleu(_: Any) -> None:
    gen_texts, real_texts, shipped = load_texts()
    gen_pairs = bleu_pairs(gen_texts)
    recomputed = mean([v for v, _, _ in gen_pairs])
    print("\n== fidelity ==\n")
    print(f"  self_bleu_4 shipped={shipped:.12f} recomputed={recomputed:.12f} "
          f"delta={abs(shipped - recomputed):.2e}")
    if abs(shipped - recomputed) > 1e-9:
        raise SystemExit("recomputation does not reproduce the shipped metric")
    report("self_bleu_4", gen_pairs, gen_texts, bleu_pairs(real_texts), real_texts)


def load_pair_file(path: Path) -> tuple[list[tuple[float, int, int]], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    thread = data["threads"][0] if "threads" in data else data
    order = {str(row["comment_id"]): i for i, row in enumerate(thread["comments"])}
    texts = [str(row.get("text", "")) for row in thread["comments"]]
    pairs = [
        (
            float(p["bert_f1"]),
            order[str(p["left_comment_id"])],
            order[str(p["right_comment_id"])],
        )
        for p in thread["pairs"]
    ]
    return pairs, texts


def cmd_bertscore(args: Any) -> None:
    if not args.generated_pairs or not args.real_pairs:
        raise SystemExit("bertscore needs --generated-pairs and --real-pairs")
    gen_pairs, gen_texts = load_pair_file(Path(args.generated_pairs))
    real_pairs, real_texts = load_pair_file(Path(args.real_pairs))
    shipped = json.loads(
        (RUNS / TREATED / "cleaned/run_00_sampled_reddit/self_bertscore_results.json").read_text(
            encoding="utf-8"
        )
    )["threads"][0]["mean_bert_f1"]
    recomputed = mean([v for v, _, _ in gen_pairs])
    print("\n== fidelity ==\n")
    print(f"  self_bertscore shipped={shipped:.12f} recomputed={recomputed:.12f} "
          f"delta={abs(shipped - recomputed):.2e}")
    if abs(shipped - recomputed) > 1e-9:
        raise SystemExit("supplied pairs do not reproduce the shipped metric")
    if not any(t.strip() for t in real_texts):
        raise SystemExit("real pair file carries no comment text; rebuild it with text included")
    report("self_bertscore_mean_f1", gen_pairs, gen_texts, real_pairs, real_texts)


COMMANDS = {"bleu": cmd_bleu, "bertscore": cmd_bertscore}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--generated-pairs", default="")
    parser.add_argument("--real-pairs", default="")
    args = parser.parse_args()
    for name in (list(COMMANDS) if args.command == "all" else [args.command]):
        COMMANDS[name](args)


if __name__ == "__main__":
    main()
