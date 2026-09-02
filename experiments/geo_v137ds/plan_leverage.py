#!/usr/bin/env python3
"""Leave-one-out on the Writer-visible plan: which field to change, and by how much.

  python3 experiments/geo_v137ds/plan_leverage.py a1gpt_20260902

The realization is text = a*plan + b, fitted per arm.  Plan cosine is therefore
a budget: this prints the plan cosine and the text cosine it implies for the
current plan, for every single-field drop, and for repairing the `domain_intent`
the Planner omits for whole batches, so an intervention is chosen on a measured
number instead of on which field reads worst.
"""
import collections, itertools, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL = "sentence-transformers/all-mpnet-base-v2"
KEYS = ("semantic_move", "local_topic", "detail_focus", "domain_intent")
FB = "one seed-grounded local move"


def threads(prefix):
    for d in sorted((REPO / "artifacts/generalized_card/runs").glob(f"{prefix}_p*")):
        f = d / "generated/run_00_sampled_reddit/generation_records.json"
        if not f.exists(): continue
        by = collections.defaultdict(list)
        for r in json.load(open(f)): by[int(r["seed_index"])].append(r)
        for si, rs in by.items():
            rs = [r for r in rs if str(r.get("raw") or "").strip()]
            if len(rs) >= 6: yield si, rs


def offd(v):
    iu = np.triu_indices(len(v), 1)
    return (v @ v.T)[iu]


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "a1gpt_20260902"
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cpu")
    variants = {"当前 (4 个字段)": KEYS}
    for k in KEYS:
        variants[f"去掉 {k}"] = tuple(x for x in KEYS if x != k)
    for k in KEYS:
        variants[f"只有 {k}"] = (k,)
    variants["semantic_move+detail_focus"] = ("semantic_move", "detail_focus")

    acc = collections.defaultdict(list)
    text_all, plan_all = [], []
    for si, rs in threads(prefix):
        tasks = [r["task"] for r in rs]
        texts = [str(r["raw"]).strip() for r in rs]
        strings, order = [], []
        for name, ks in variants.items():
            order.append((name, len(strings), len(tasks)))
            strings += [" ".join(str(t.get(k) or "") for k in ks).strip() or "(empty)" for t in tasks]
        # domain_intent 修好：把兜底串换成空（模拟"字段没被漏掉时不会有这串共享文本"）
        order.append(("兜底 domain_intent 清空", len(strings), len(tasks)))
        strings += [" ".join([str(t.get(k) or "") for k in KEYS[:3]] +
                             ([] if str(t.get("domain_intent") or "").strip() == FB
                              else [str(t.get("domain_intent") or "")])).strip() or "(empty)"
                    for t in tasks]
        ti = len(strings); strings += texts
        E = m.encode(strings, normalize_embeddings=True, batch_size=128,
                     convert_to_numpy=True, show_progress_bar=False)
        for name, o, k in order:
            acc[name].append(offd(E[o:o+k]))
        acc["__text__"].append(offd(E[ti:ti+len(texts)]))
    text = np.concatenate(acc["__text__"])
    base = np.concatenate(acc["当前 (4 个字段)"])
    A = np.vstack([base, np.ones_like(base)]).T
    (a, b), *_ = np.linalg.lstsq(A, text, rcond=None)
    print(f"{prefix}: realization 拟合 成文 = {a:.3f} x plan + {b:.4f}   (现成文 {text.mean():.4f})")
    print(f"目标 plan 余弦 = {(0.1555 - b)/a:.4f}  (真人 per-thread 总体 0.1555)\n")
    print(f"{'plan 组成':<30}{'plan 余弦':>10}{'预测成文':>10}{'相对现在':>10}")
    print("-"*62)
    rows = [(n, np.concatenate(v).mean()) for n, v in acc.items() if n != "__text__"]
    cur = dict(rows)["当前 (4 个字段)"]
    for n, p in sorted(rows, key=lambda r: -r[1]):
        print(f"{n:<30}{p:>10.4f}{a*p+b:>10.4f}{(a*p+b)/(a*cur+b)-1:>+9.1%}")


if __name__ == "__main__":
    main()
