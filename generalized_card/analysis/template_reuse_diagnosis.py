#!/usr/bin/env python3
"""Does a comment's opener/closer *frame* recur across a thread even when the
content slotted into it differs? (`docs/DECISIONS.md` G13.)

The v106 gate (seed 8, `i1o51h`, 2026-08-22) eliminated the diagnosed
claim-level restatement in `self_bertscore_mean_f1` at the plan level (0 of
186 plan violations, was 18) and the metric got slightly worse anyway.
Reading the actual highest-scoring pairs on the new artifact: the reused
material is no longer the same *claim* -- it is the same sentence *frame*
with a different object dropped in ("@OP, watch the subject cross the EVF
and see if your eye can keep up." / "@OP, check whether the EVF blanks out
or lags through a long burst."). `semantic_realization.used_sentence_routes`
does not catch this because it matches literal 3-4-token n-grams, and these
two openers differ at the second token.

This measures the phenomenon directly with the same tool the rest of this
project already uses for exactly this kind of thing -- a general-purpose
sentence embedding (`all-mpnet-base-v2`, `semantic_mean_cosine`'s model, not
BERTScore) -- applied to just the opening clause and just the closing clause
of each comment, rather than the whole comment. If a template recurs with a
different object, the two whole comments may not be very similar overall,
but their opener (or closer) clauses will be.

No API call, no domain vocabulary, seconds per thread (a much smaller model
than BERTScore, and only two short spans per comment, not the whole text).

    python3 generalized_card/analysis/template_reuse_diagnosis.py gate
    python3 generalized_card/analysis/template_reuse_diagnosis.py corpus
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCORER_DIR = REPO / "scripts" / "evaluation"
if str(SCORER_DIR) not in sys.path:
    sys.path.insert(0, str(SCORER_DIR))

from score_thread_semantic_uniformity import (  # noqa: E402
    CommentEmbedder,
    DEFAULT_MODEL,
    ThreadComment,
    load_generated_comments,
    load_real_comments,
)

GATE_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "v106_chain_novelty_digit_guard_seed8_20260822_v1"
)
BASELINE_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "v104_evaluative_seed8_20260821_v1"
)
V103_N10_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1"
)
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
REAL_DIR = REPO / "data/raw/discussions/camera_product"

# Same clause boundary `semantic_realization.used_sentence_routes` already
# uses, for consistency with the mechanism this diagnoses.
CLAUSE_SPLIT = re.compile(r"[.!?;:,\n]+|\s+[—–]\s+")
MIN_CLAUSE_WORDS = 3
NEAR_DUPLICATE_THRESHOLD = 0.75  # near the 0.76 reply-novelty threshold elsewhere


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def opener_and_closer(text: str) -> tuple[str, str]:
    """The first and last clause of a comment with >= MIN_CLAUSE_WORDS words.

    Falls back to the whole text if no clause boundary carries enough words
    (a short comment's opener and closer are the same span, which is exactly
    right: a one-clause "Same here" should compare against other one-clause
    comments as a whole, not against a truncated fragment of one).
    """

    clauses = [c.strip() for c in CLAUSE_SPLIT.split(text) if c.strip()]
    usable = [c for c in clauses if len(c.split()) >= MIN_CLAUSE_WORDS]
    if not usable:
        return text.strip(), text.strip()
    return usable[0], usable[-1]


def _pairwise_mean_and_near_dup(
    vectors: list[Any],
) -> tuple[float, int, int]:
    """(mean pairwise cosine, near-duplicate pairs, total pairs)."""

    import numpy as np

    sims = []
    near = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            score = float(np.dot(vectors[i], vectors[j]))
            sims.append(score)
            if score >= NEAR_DUPLICATE_THRESHOLD:
                near += 1
    return (mean(sims) if sims else float("nan")), near, len(sims)


def thread_frame_reuse(
    embedder: CommentEmbedder,
    comments_by_thread: dict[str, list[ThreadComment]],
    *,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Per-thread opener/closer pairwise-similarity stats."""

    thread_ids = list(comments_by_thread)
    openers: list[str] = []
    closers: list[str] = []
    owner: list[str] = []
    for thread_id in thread_ids:
        for comment in comments_by_thread[thread_id]:
            opener, closer = opener_and_closer(comment.text)
            openers.append(opener)
            closers.append(closer)
            owner.append(thread_id)
    if not openers:
        return []

    opener_vectors = embedder.encode(openers, batch_size=batch_size)
    closer_vectors = embedder.encode(closers, batch_size=batch_size)

    by_thread_openers: dict[str, list[Any]] = {tid: [] for tid in thread_ids}
    by_thread_closers: dict[str, list[Any]] = {tid: [] for tid in thread_ids}
    for tid, ov, cv in zip(owner, opener_vectors, closer_vectors):
        by_thread_openers[tid].append(ov)
        by_thread_closers[tid].append(cv)

    rows = []
    for thread_id in thread_ids:
        opener_mean, opener_near, opener_pairs = _pairwise_mean_and_near_dup(
            by_thread_openers[thread_id]
        )
        closer_mean, closer_near, closer_pairs = _pairwise_mean_and_near_dup(
            by_thread_closers[thread_id]
        )
        rows.append(
            {
                "thread_id": thread_id,
                "comment_count": len(by_thread_openers[thread_id]),
                "opener_mean_cos": opener_mean,
                "opener_near_dup_pairs": opener_near,
                "opener_pairs": opener_pairs,
                "closer_mean_cos": closer_mean,
                "closer_near_dup_pairs": closer_near,
                "closer_pairs": closer_pairs,
            }
        )
    return rows


def _generated_comments(run: Path) -> dict[str, list[ThreadComment]]:
    out: dict[str, list[ThreadComment]] = {}
    for sim_dir in sorted(run.glob("cleaned/run_*_sampled_reddit")):
        comments, _ = load_generated_comments(sim_dir)
        out.update(comments)
    return out


def _matched_real_comments(run: Path) -> dict[str, list[ThreadComment]]:
    """Real comments for exactly the threads matched to `run`'s seeds."""

    pool = _load_json(SEED_POOL)
    pool_by_seed = {int(row["seed_index"]): row for row in pool["seed_posts"]}
    seed_suffix = re.compile(r"seed(\d+)$")
    out: dict[str, list[ThreadComment]] = {}
    cache: dict[Path, dict[str, list[ThreadComment]]] = {}
    for thread_id in _generated_comments(run):
        match = seed_suffix.search(thread_id)
        if not match:
            continue
        seed = pool_by_seed[int(match.group(1))]
        raw_id = str(seed["source_raw_post_id"])
        product_dir = REAL_DIR / str(seed["source_product_dir"])
        if product_dir not in cache:
            all_comments, _ = load_real_comments(product_dir)
            cache[product_dir] = all_comments
        if raw_id in cache[product_dir]:
            out[raw_id] = cache[product_dir][raw_id]
    return out


def _print_rows(label: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n== {label} ==\n")
    print(
        f"{'thread':36s} {'n':>4s} | {'opener mean':>11s} {'near-dup':>9s} | "
        f"{'closer mean':>11s} {'near-dup':>9s}"
    )
    for row in rows:
        print(
            f"{row['thread_id']:36s} {row['comment_count']:4d} | "
            f"{row['opener_mean_cos']:11.4f} "
            f"{row['opener_near_dup_pairs']:4d}/{row['opener_pairs']:<4d} | "
            f"{row['closer_mean_cos']:11.4f} "
            f"{row['closer_near_dup_pairs']:4d}/{row['closer_pairs']:<4d}"
        )
    if rows:
        print(
            f"\nmean across threads: opener={mean([r['opener_mean_cos'] for r in rows]):.4f} "
            f"closer={mean([r['closer_mean_cos'] for r in rows]):.4f}"
        )
        total_opener_pairs = sum(r["opener_pairs"] for r in rows)
        total_opener_near = sum(r["opener_near_dup_pairs"] for r in rows)
        total_closer_pairs = sum(r["closer_pairs"] for r in rows)
        total_closer_near = sum(r["closer_near_dup_pairs"] for r in rows)
        print(
            f"pooled near-dup rate (>= {NEAR_DUPLICATE_THRESHOLD}): "
            f"opener={total_opener_near}/{total_opener_pairs} "
            f"({total_opener_near / total_opener_pairs if total_opener_pairs else 0:.4f}) "
            f"closer={total_closer_near}/{total_closer_pairs} "
            f"({total_closer_near / total_closer_pairs if total_closer_pairs else 0:.4f})"
        )


def cmd_gate(embedder: CommentEmbedder) -> None:
    """The one thread with direct qualitative evidence: gate vs real, and the
    pre-fix baseline for the same thread, side by side."""

    gate_gen = _generated_comments(GATE_RUN)
    gate_real = _matched_real_comments(GATE_RUN)
    baseline_gen = _generated_comments(BASELINE_RUN)

    _print_rows("v106 gate (chain + digit-cue-guard on)", thread_frame_reuse(embedder, gate_gen))
    _print_rows("v104 baseline (same thread, neither arm existed)", thread_frame_reuse(embedder, baseline_gen))
    _print_rows("matched real (i1o51h)", thread_frame_reuse(embedder, gate_real))


def cmd_corpus(embedder: CommentEmbedder) -> None:
    """Generality check: the v103 N=10 generated pool vs its matched real
    threads, vs the null (two disjoint real samples), same as this session's
    other corpus-scale checks."""

    gen = _generated_comments(V103_N10_RUN)
    real = _matched_real_comments(V103_N10_RUN)
    _print_rows("v103 N=10 generated", thread_frame_reuse(embedder, gen))
    _print_rows("matched real (10 threads)", thread_frame_reuse(embedder, real))

    pool = _load_json(SEED_POOL)
    seed_ids = {str(row["source_raw_post_id"]) for row in pool["seed_posts"]}
    excluded: dict[str, list[ThreadComment]] = {}
    for product_dir in sorted(p for p in REAL_DIR.iterdir() if p.is_dir()):
        try:
            comments_by_thread, _ = load_real_comments(product_dir)
        except FileNotFoundError:
            continue
        for thread_id, comments in comments_by_thread.items():
            if thread_id not in seed_ids and thread_id not in excluded and len(comments) >= 15:
                excluded[thread_id] = comments
    # Cap for a quick corpus-scale null read; not the full 424.
    capped = dict(list(excluded.items())[:80])
    print(f"\n(corpus-scale null over {len(capped)} evaluation-excluded real threads, capped for speed)")
    _print_rows("evaluation-excluded real (null population)", thread_frame_reuse(embedder, capped))


COMMANDS = {"gate": cmd_gate, "corpus": cmd_corpus}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    embedder = CommentEmbedder(model_name=DEFAULT_MODEL, device=args.device, max_length=64)
    names = list(COMMANDS) if args.command == "all" else [args.command]
    for name in names:
        print(f"\n{'#' * 76}\n# {name}\n{'#' * 76}")
        COMMANDS[name](embedder)


if __name__ == "__main__":
    main()
