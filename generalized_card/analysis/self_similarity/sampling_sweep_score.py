#!/usr/bin/env python3
"""Within-thread pairwise BERTScore for a sampling sweep, on the eval pipeline's own scorer."""
from __future__ import annotations
import argparse, json, math, statistics, sys
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import ThreadComment, load_real_comments  # noqa: E402
from score_thread_self_bertscore import (  # noqa: E402
    DEFAULT_BERT_SCORE_PATH, DEFAULT_MODEL, aggregate_threads,
    build_pair_specs, load_bert_scorer, score_pairs,
)

def comments(tid, texts):
    return [ThreadComment(thread_id=tid, thread_title="", comment_id=str(i),
                          parent_id="", author="", text=t, depth=0)
            for i, t in enumerate(texts) if t and t.strip()]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=48)
    args = ap.parse_args()

    corpora: dict[str, dict[str, list[str]]] = {}
    first = json.loads(Path(args.replays[0]).read_text(encoding="utf-8"))
    shipped: dict[str, list[str]] = {}
    for it in first["items"]:
        shipped.setdefault(it["thread_id"], []).append(it["original"])
    for path in args.replays:
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        s = p.get("sampling") or {}
        label = f"T={s.get('temperature')}"
        d: dict[str, list[str]] = {}
        for it in p["items"]:
            d.setdefault(it["thread_id"], []).append(it["replayed"])
        corpora[label] = d

    pool = json.loads((REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
    by_seed = {int(x["seed_index"]): x for x in pool}
    real: dict[str, list[str]] = {}
    cache: dict = {}
    for tid in shipped:
        sp = by_seed[int(tid.replace("seed", ""))]
        d = REPO / "data/raw/discussions/camera_product" / sp["source_product_dir"]
        if d not in cache:
            cache[d] = load_real_comments(d)[0]
        real[tid] = [c.text for c in (cache[d].get(sp["source_raw_post_id"]) or [])]

    allc = {"REAL": real, "SHIPPED": shipped, **corpora}
    scorer, *_ = load_bert_scorer(
        bert_score_path=DEFAULT_BERT_SCORE_PATH, model_type=DEFAULT_MODEL, num_layers=None,
        batch_size=args.batch_size, device=args.device, idf=False, idf_sents=[],
        rescale_with_baseline=False, local_files_only=False)

    results: dict[str, dict[str, float]] = {}
    for label, byt in allc.items():
        cs = {t: comments(t, v) for t, v in byt.items() if len(v) >= 4}
        specs = build_pair_specs(cs)
        print(f"scoring {label}: {len(cs)} threads, {len(specs)} pairs", flush=True)
        rows = aggregate_threads(cs, score_pairs(scorer, specs, args.batch_size), top_k=10, include_pairs=False)
        results[label] = {r["thread_id"]: float(r["mean_bert_f1"]) for r in rows if r["pair_count"]}

    threads = sorted(set.intersection(*(set(v) for v in results.values())))
    print(f"\n{'thread':<12} " + " ".join(f"{k:>10}" for k in results))
    for t in threads:
        print(f"{t:<12} " + " ".join(f"{results[k][t]:10.4f}" for k in results))
    base = statistics.mean(results["REAL"][t] for t in threads)
    ship = statistics.mean(results["SHIPPED"][t] for t in threads)
    ctrl = results.get("T=1.0")
    print(f"\n{'corpus':<12} {'mean':>9} {'vs REAL':>9} {'rel%':>8}   note")
    for k, v in results.items():
        m = statistics.mean(v[t] for t in threads)
        note = ""
        if ctrl and k.startswith("T=") and k != "T=1.0":
            c = statistics.mean(ctrl[t] for t in threads)
            g0, g = c - base, m - base
            wins = sum(v[t] < ctrl[t] for t in threads)
            note = f"vs T=1.0 control: closes {100*(g0-g)/g0:5.1f}% of its gap, better in {wins}/{len(threads)}"
        print(f"{k:<12} {m:9.4f} {m-base:+9.4f} {100*(m-base)/base:+7.2f}%   {note}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
