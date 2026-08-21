#!/usr/bin/env python3
"""G3: decompose `self_bertscore_mean_f1`'s excess into its pairwise matrix.

`self_bertscore_mean_f1` is the one metric that fails a standard which does not
fail correct work (v103 N=10: MWU 0.001, KS 0.002, |Cliff| 0.86 against a floor
of 0.50 -- see `docs/ORIENTATION.md` §2/§6). Five hypotheses about *why* have
been rejected with no mechanism surviving (length spread, duplication tail,
surface register, lexical breadth r=+0.077, narrow shared vocabulary wrong-sign).
The one untried diagnostic, named in `docs/DECISIONS.md` row G3, is the same one
that cracked `hard_disagree_rate`: decompose the thread mean into its pairwise
matrix and ask whether the excess concentrates on parent-child pairs, spreads
across a reply branch, or is genuinely uniform across the whole pair population.

This never shipped with `--include-pairs`, so every existing
`self_bertscore_results.json` on disk has thread means only. This script
re-scores -- with the evaluator's own scorer classes, same model, same config --
only the ten v103 N=10 generated threads and their ten matched real threads (not
the 424-thread corpus), classifies every pair by tree relation using the
comment/parent ids the scorer already carries, and reports the excess per
relation bucket. No API call; CPU, ~10-20 min on `microsoft/deberta-xlarge-mnli`.

Fidelity is checked before anything else, per `tasks/lessons.md` E6: recomputed
generated thread means must reproduce the shipped
`cleaned/run_*_sampled_reddit/self_bertscore_results.json` values, and recomputed
real thread means must reproduce the shipped per-product
`self_bertscore_results.json` entries -- both computed at v103 run time from the
exact same comments, so agreement should be exact, not approximate.

Run with system `python3` (transformers 4.48.0), not `.venv` (transformers
5.10.1 there drifts the model hash away from the shipped artifact's):

    python3 generalized_card/analysis/bertscore_pair_diagnosis.py fidelity
    python3 generalized_card/analysis/bertscore_pair_diagnosis.py pairs
    python3 generalized_card/analysis/bertscore_pair_diagnosis.py all
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
SCORER_DIR = REPO / "scripts" / "evaluation"
if str(SCORER_DIR) not in sys.path:
    sys.path.insert(0, str(SCORER_DIR))

from score_thread_self_bertscore import (  # noqa: E402
    build_pair_specs,
    load_bert_scorer,
    score_pairs_with_device_fallback,
    DEFAULT_BERT_SCORE_PATH,
    DEFAULT_MODEL,
)
from score_thread_semantic_uniformity import (  # noqa: E402
    ThreadComment,
    load_generated_comments,
    load_real_comments,
)

DEFAULT_RUN = (
    REPO
    / "artifacts/generalized_card/runs"
    / "generalized_card_camera_gpt54_v103_stance_opening_n10_20260821_v1"
)
SEED_POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
REAL_DIR = REPO / "data/raw/discussions/camera_product"
SEED_SUFFIX = re.compile(r"seed(\d+)$")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


# --------------------------------------------------------------------------- #
# loading -- ten generated threads, their ten matched real threads
# --------------------------------------------------------------------------- #


class Threads:
    """Generated and matched-real comments for the v103 N=10 artifact."""

    def __init__(self, run: Path) -> None:
        self.run = run
        pool = _load_json(SEED_POOL)
        self.pool_by_seed = {int(row["seed_index"]): row for row in pool["seed_posts"]}
        self.generated: dict[str, list[ThreadComment]] = {}
        self.generated_shipped_mean: dict[str, float] = {}
        for sim_dir in sorted(run.glob("cleaned/run_*_sampled_reddit")):
            comments, _ = load_generated_comments(sim_dir)
            self.generated.update(comments)
            shipped = _load_json(sim_dir / "self_bertscore_results.json")
            for row in shipped["threads"]:
                self.generated_shipped_mean[row["thread_id"]] = row["mean_bert_f1"]

        self.real: dict[str, list[ThreadComment]] = {}
        self.real_shipped_mean: dict[str, float] = {}
        self.pairing: list[tuple[str, str]] = []  # (generated_thread_id, real_thread_id)
        real_cache: dict[str, tuple[dict[str, list[ThreadComment]], dict[str, float]]] = {}
        for thread_id in self.generated:
            match = SEED_SUFFIX.search(thread_id)
            if not match:
                raise SystemExit(f"cannot recover seed index from {thread_id!r}")
            seed = self.pool_by_seed[int(match.group(1))]
            raw_id = str(seed["source_raw_post_id"])
            product_dir = REAL_DIR / str(seed["source_product_dir"])
            if product_dir not in real_cache:
                all_comments, _ = load_real_comments(product_dir)
                shipped = _load_json(product_dir / "self_bertscore_results.json")
                shipped_mean = {row["thread_id"]: row["mean_bert_f1"] for row in shipped["threads"]}
                real_cache[product_dir] = (all_comments, shipped_mean)
            all_comments, shipped_mean = real_cache[product_dir]
            if raw_id not in all_comments:
                raise SystemExit(f"real thread {raw_id!r} not found in {product_dir}")
            self.real[raw_id] = all_comments[raw_id]
            self.real_shipped_mean[raw_id] = shipped_mean.get(raw_id, float("nan"))
            self.pairing.append((thread_id, raw_id))


# --------------------------------------------------------------------------- #
# scoring -- exactly the production scorer, restricted to these 20 threads
# --------------------------------------------------------------------------- #


def score(comments_by_thread: dict[str, list[ThreadComment]], device: str, batch_size: int) -> list[dict[str, Any]]:
    if not comments_by_thread:
        return []
    pair_specs = build_pair_specs(comments_by_thread)
    idf_sents = [c.text for cs in comments_by_thread.values() for c in cs]
    (scorer, *_rest, fallback_used) = load_bert_scorer(
        bert_score_path=DEFAULT_BERT_SCORE_PATH,
        model_type=DEFAULT_MODEL,
        num_layers=None,
        batch_size=batch_size,
        device=device,
        idf=False,
        idf_sents=idf_sents,
        rescale_with_baseline=False,
        local_files_only=False,
    )
    if fallback_used:
        raise SystemExit(
            "microsoft/deberta-xlarge-mnli failed to load; refusing to score with "
            "roberta-large, which would not reproduce the shipped artifact."
        )
    (pair_scores, *_rest2) = score_pairs_with_device_fallback(
        scorer=scorer,
        pair_specs=pair_specs,
        batch_size=batch_size,
        bert_score_path=DEFAULT_BERT_SCORE_PATH,
        model_type=DEFAULT_MODEL,
        num_layers=None,
        requested_device=device,
        idf=False,
        idf_sents=idf_sents,
        rescale_with_baseline=False,
        local_files_only=False,
        fallback_used=False,
    )
    return pair_scores


# --------------------------------------------------------------------------- #
# relation classification -- parent-child, same-branch, cross-branch
# --------------------------------------------------------------------------- #


def _parent_map(comments: list[ThreadComment]) -> dict[str, str | None]:
    ids = {c.comment_id for c in comments}
    out: dict[str, str | None] = {}
    for c in comments:
        out[c.comment_id] = c.parent_id if c.parent_id in ids else None
    return out


def _path_to_root(comment_id: str, parent: dict[str, str | None]) -> list[str]:
    """[post, ..., grandparent, parent, comment_id]."""

    path = [comment_id]
    current = comment_id
    seen = {comment_id}
    while parent.get(current) is not None:
        current = parent[current]
        if current in seen:  # malformed cycle guard; should not happen
            break
        seen.add(current)
        path.append(current)
    path.reverse()
    return path


def classify_relation(a: str, b: str, parent: dict[str, str | None]) -> tuple[str, str]:
    """`(relation, relation_detail)`.

    `relation` is the 3-way cut the diagnosis asks for: `parent_child` (direct
    edge), `same_branch` (share some comment ancestor -- a top-level comment or
    deeper -- that is not a direct edge), `cross_branch` (share no comment
    ancestor; different top-level chains under the post). `relation_detail`
    additionally splits `same_branch` into `ancestor_descendant` (one is a
    non-immediate ancestor of the other) and `sibling_or_cousin` (neither is,
    but they share one).
    """

    if parent.get(a) == b or parent.get(b) == a:
        return "parent_child", "parent_child"
    path_a = _path_to_root(a, parent)
    path_b = _path_to_root(b, parent)
    set_a, set_b = set(path_a), set(path_b)
    if not (set_a & set_b):
        return "cross_branch", "cross_branch"
    if a in set_b or b in set_a:
        return "same_branch", "ancestor_descendant"
    return "same_branch", "sibling_or_cousin"


def _is_root(comment_id: str, parent: dict[str, str | None]) -> bool:
    return parent.get(comment_id) is None


def classify_pairs(
    comments_by_thread: dict[str, list[ThreadComment]],
    pair_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    parent_maps = {tid: _parent_map(cs) for tid, cs in comments_by_thread.items()}
    out = []
    for pair in pair_scores:
        parent = parent_maps[pair["thread_id"]]
        relation, relation_detail = classify_relation(
            pair["left_comment_id"], pair["right_comment_id"], parent
        )
        a_root = _is_root(pair["left_comment_id"], parent)
        b_root = _is_root(pair["right_comment_id"], parent)
        depth_kind = (
            "root_root" if a_root and b_root
            else "root_reply" if a_root != b_root
            else "reply_reply"
        )
        out.append(
            {
                **pair,
                "relation": relation,
                "relation_detail": relation_detail,
                "depth_kind": depth_kind,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_fidelity(threads: Threads, device: str, batch_size: int) -> tuple[list[dict], list[dict]]:
    print("== fidelity: recomputed thread means must reproduce the shipped artifact ==\n")
    gen_pairs = score(threads.generated, device, batch_size)
    real_pairs = score(threads.real, device, batch_size)

    def check(name: str, comments: dict[str, list[ThreadComment]], pairs: list[dict], shipped: dict[str, float]) -> bool:
        by_thread: dict[str, list[float]] = defaultdict(list)
        for pair in pairs:
            by_thread[pair["thread_id"]].append(pair["bert_f1"])
        ok = True
        print(f"-- {name} --")
        for thread_id in comments:
            recomputed = mean(by_thread.get(thread_id, []))
            ship = shipped.get(thread_id, float("nan"))
            delta = recomputed - ship
            flag = "" if abs(delta) < 1e-6 else "  <-- MISMATCH"
            print(f"  {thread_id:36s} shipped={ship:.6f} recomputed={recomputed:.6f} delta={delta:+.2e}{flag}")
            ok = ok and abs(delta) < 1e-6
        return ok

    gen_ok = check("generated", threads.generated, gen_pairs, threads.generated_shipped_mean)
    real_ok = check("real", threads.real, real_pairs, threads.real_shipped_mean)
    print(f"\nfidelity: generated {'OK' if gen_ok else 'FAILED'}, real {'OK' if real_ok else 'FAILED'}")
    if not (gen_ok and real_ok):
        print("FIDELITY FAILED -- do not read the pairwise breakdown below.")
    return gen_pairs, real_pairs


def _bucket_report(label: str, key: Callable[[dict], str], gen_classified: list[dict], real_classified: list[dict]) -> None:
    print(f"\n== excess by {label}, pooled over all pairs ==\n")
    gen_by = defaultdict(list)
    real_by = defaultdict(list)
    for row in gen_classified:
        gen_by[key(row)].append(row["bert_f1"])
    for row in real_classified:
        real_by[key(row)].append(row["bert_f1"])
    order = sorted(set(gen_by) | set(real_by), key=lambda k: -len(gen_by.get(k, [])))
    print(f"{'bucket':14s} {'gen n':>7s} {'gen mean':>9s} | {'real n':>7s} {'real mean':>9s} | {'excess':>7s}")
    for name in order:
        g, r = gen_by.get(name, []), real_by.get(name, [])
        gm, rm = mean(g), mean(r)
        excess = gm - rm if g and r else float("nan")
        print(f"{name:14s} {len(g):7d} {gm:9.4f} | {len(r):7d} {rm:9.4f} | {excess:+7.4f}")


def _paired_bucket_report(
    label: str,
    key: Callable[[dict], str],
    threads: Threads,
    gen_classified: list[dict],
    real_classified: list[dict],
) -> None:
    """Per-thread bucket means, averaged with equal weight across the ten pairs
    -- the way the metric itself aggregates (thread mean, not pooled pair mean;
    see `tasks/lessons.md` 2026-08-21 "A pooled rate can match perfectly")."""

    print(f"\n== excess by {label}, thread-paired (equal weight per thread) ==\n")
    gen_by_thread: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    real_by_thread: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in gen_classified:
        gen_by_thread[row["thread_id"]][key(row)].append(row["bert_f1"])
    for row in real_classified:
        real_by_thread[row["thread_id"]][key(row)].append(row["bert_f1"])

    buckets = sorted({key(r) for r in gen_classified} | {key(r) for r in real_classified})
    for bucket in buckets:
        diffs = []
        for gen_id, real_id in threads.pairing:
            g = gen_by_thread[gen_id].get(bucket, [])
            r = real_by_thread[real_id].get(bucket, [])
            if g and r:
                diffs.append(mean(g) - mean(r))
        if diffs:
            try:
                from scipy import stats as ss

                w = ss.wilcoxon(diffs) if len(diffs) >= 6 else None
            except ImportError:
                w = None
            p = f"{w.pvalue:.4f}" if w is not None else "n/a"
            print(
                f"  {bucket:14s} n_threads={len(diffs):2d} mean_excess={mean(diffs):+.4f} "
                f"min={min(diffs):+.4f} max={max(diffs):+.4f} wilcoxon_p={p}"
            )
        else:
            print(f"  {bucket:14s} n_threads=0 (no thread has this bucket on both sides)")


def cmd_pairs(threads: Threads, device: str, batch_size: int) -> None:
    gen_pairs, real_pairs = cmd_fidelity(threads, device, batch_size)
    gen_classified = classify_pairs(threads.generated, gen_pairs)
    real_classified = classify_pairs(threads.real, real_pairs)

    print(f"\ntotal pairs: generated={len(gen_classified)} real={len(real_classified)}")
    _bucket_report("tree relation", lambda r: r["relation"], gen_classified, real_classified)
    _bucket_report("tree relation (detail)", lambda r: r["relation_detail"], gen_classified, real_classified)
    _bucket_report("depth kind", lambda r: r["depth_kind"], gen_classified, real_classified)
    _paired_bucket_report("tree relation", lambda r: r["relation"], threads, gen_classified, real_classified)
    _paired_bucket_report(
        "tree relation (detail)", lambda r: r["relation_detail"], threads, gen_classified, real_classified
    )
    _paired_bucket_report("depth kind", lambda r: r["depth_kind"], threads, gen_classified, real_classified)

    overall_gen = mean([r["bert_f1"] for r in gen_classified])
    overall_real = mean([r["bert_f1"] for r in real_classified])
    print(f"\noverall pooled excess (for reference, not the metric itself): {overall_gen - overall_real:+.4f}")


COMMANDS: dict[str, Callable[..., Any]] = {
    "fidelity": cmd_fidelity,
    "pairs": cmd_pairs,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    threads = Threads(Path(args.run))
    # "pairs" already runs the fidelity check before reporting the breakdown,
    # so "all" is just an alias for it -- not fidelity-then-pairs, which would
    # re-run BERTScore over the same 20 threads twice.
    names = ["pairs"] if args.command == "all" else [args.command]
    for name in names:
        print(f"\n{'#' * 76}\n# {name}\n{'#' * 76}\n")
        COMMANDS[name](threads, args.device, args.batch_size)


if __name__ == "__main__":
    main()
