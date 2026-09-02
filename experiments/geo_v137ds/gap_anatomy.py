#!/usr/bin/env python3
"""Decompose the within-thread similarity gap into plan and realization parts.

  python3 experiments/geo_v137ds/gap_anatomy.py a1gpt_20260902 [v150_20260902 ...]

For every generated thread and its own matched real thread this reports, on the
same all-mpnet-base-v2 scale the `semantic_mean_cosine` metric uses:

  1. realized text cosine, generated vs its matched real thread
  2. the PLAN's own cosine -- how far apart the Planner placed the slots
  3. the transfer function realized(plan): grouped by plan-cosine decile, so a
     Writer that floors out is distinguishable from a Planner that never asks
     for a distant pair
  4. contrasts for the three fields the Planner leaves empty (`unclear_mixed`
     content_angle, `seed_local` perspective, and the `domain_intent` the model
     drops whole batches of), each stratified by plan cosine so the contrast is
     not just re-reading the plan distance
"""
import collections, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL = "sentence-transformers/all-mpnet-base-v2"
INTENT_FALLBACK = "one seed-grounded local move"


def load_arm(prefix):
    """-> list of (seed_index, [texts], [plans])"""
    out = []
    root = REPO / "artifacts/generalized_card/runs"
    for d in sorted(root.glob(f"{prefix}_p*")):
        f = d / "generated/run_00_sampled_reddit/generation_records.json"
        if not f.exists():
            continue
        recs = json.load(open(f))
        by_seed = collections.defaultdict(list)
        for r in recs:
            by_seed[int(r["seed_index"])].append(r)
        for si, rs in by_seed.items():
            texts = [str(r.get("raw") or "").strip() for r in rs]
            plans = [r.get("task") or {} for r in rs]
            keep = [i for i, t in enumerate(texts) if t]
            if len(keep) >= 4:
                out.append((si, [texts[i] for i in keep], [plans[i] for i in keep]))
    return out


def real_by_seed():
    by_post = collections.defaultdict(list)
    src = REPO / "data/raw/discussions/celebrity_geo/celebrity/celebrity.comments.jsonl"
    for line in src.open():
        c = json.loads(line)
        t = (c.get("body") or "").strip()
        if t and t not in ("[deleted]", "[removed]"):
            by_post[str(c.get("post_id"))].append(t)
    pool = json.loads((REPO / "artifacts/generalized_card/seed_pools/celebrity_geo_150_seed907.json").read_text())
    return {int(r["seed_index"]): by_post.get(str(r["source_raw_post_id"]), [])[:120]
            for r in pool["seed_posts"]}


def offdiag(v):
    s = v @ v.T
    iu = np.triu_indices(len(v), 1)
    return s[iu]


def main():
    prefixes = sys.argv[1:] or ["a1gpt_20260902"]
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cpu")
    reals = real_by_seed()

    for prefix in prefixes:
        arm = load_arm(prefix)
        print(f"\n{'='*78}\n{prefix}   {len(arm)} 个 thread\n{'='*78}")
        rows = []
        pooled = collections.defaultdict(list)   # 收集所有 pair 供后面分组
        for si, texts, plans in arm:
            real = reals.get(si) or []
            if len(real) < 4:
                continue
            # 三种嵌入：成文、含 intent 的 plan、不含 intent 的 plan
            plan_full = [" ".join(str(p.get(k) or "") for k in
                                  ("semantic_move", "local_topic", "detail_focus", "domain_intent"))
                         for p in plans]
            plan_noint = [" ".join(str(p.get(k) or "") for k in
                                   ("semantic_move", "local_topic", "detail_focus"))
                          for p in plans]
            E = m.encode(texts + plan_full + plan_noint + real, normalize_embeddings=True,
                         batch_size=128, convert_to_numpy=True, show_progress_bar=False)
            n, k = len(texts), len(real)
            et, ep, epn, er = E[:n], E[n:2*n], E[2*n:3*n], E[3*n:]
            ct, cp, cpn, cr = offdiag(et), offdiag(ep), offdiag(epn), offdiag(er)
            rows.append((si, ct.mean(), cr.mean(), cp.mean(), cpn.mean(), n, k))
            iu = np.triu_indices(n, 1)
            for a, b, t, pf, pn in zip(iu[0], iu[1], ct, cp, cpn):
                pa, pb = plans[a], plans[b]
                pooled["text"].append(t); pooled["plan"].append(pf); pooled["plan_noint"].append(pn)
                pooled["ang"].append(int(str(pa.get("content_angle")) == "unclear_mixed")
                                     + int(str(pb.get("content_angle")) == "unclear_mixed"))
                pooled["per"].append(int(str(pa.get("perspective_id")) == "seed_local")
                                     + int(str(pb.get("perspective_id")) == "seed_local"))
                pooled["intent"].append(int(str(pa.get("domain_intent") or "").strip() == INTENT_FALLBACK)
                                        + int(str(pb.get("domain_intent") or "").strip() == INTENT_FALLBACK))
                pooled["story"].append(int(str(pa.get("story_mode")) == "no_story")
                                       + int(str(pb.get("story_mode")) == "no_story"))

        print(f"\n{'seed':>5}{'生成':>9}{'真实':>9}{'差':>8}{'plan余弦':>10}{'plan(无intent)':>15}{'生成/真实条数':>13}")
        for si, t, r, p, pn, n, k in rows:
            print(f"{si:>5}{t:>9.4f}{r:>9.4f}{(t-r)/r*100:>+7.1f}%{p:>10.4f}{pn:>15.4f}   {n:>3}/{k:<3}")
        T = np.array([r[1] for r in rows]); R = np.array([r[2] for r in rows])
        P = np.array([r[3] for r in rows]); PN = np.array([r[4] for r in rows])
        print(f"{'均值':>5}{T.mean():>9.4f}{R.mean():>9.4f}{(T.mean()-R.mean())/R.mean()*100:>+7.1f}%{P.mean():>10.4f}{PN.mean():>15.4f}")

        text = np.array(pooled["text"]); plan = np.array(pooled["plan"])
        plann = np.array(pooled["plan_noint"])
        print(f"\n--- 转移函数：按 plan 余弦十分位分组，看实际成文余弦 (共 {len(text)} 个 pair) ---")
        q = np.quantile(plan, np.linspace(0, 1, 11))
        print(f"{'plan 十分位':>12}{'plan 余弦':>10}{'成文余弦':>10}{'pair 数':>8}")
        for i in range(10):
            sel = (plan >= q[i]) & (plan <= q[i+1] if i == 9 else plan < q[i+1])
            if sel.sum() == 0: continue
            print(f"{i+1:>12}{plan[sel].mean():>10.4f}{text[sel].mean():>10.4f}{sel.sum():>8}")
        # 真实 thread 的 pair 分布，用来判断我们是否有能力产出"极不相似"的一对
        allreal = []
        for si, *_ in rows:
            real = reals[si]
            er = m.encode(real, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            allreal.append(offdiag(er))
        allreal = np.concatenate(allreal)
        print(f"\n--- pair 层面的分布对比 (生成 {len(text)} pair vs 真实 {len(allreal)} pair) ---")
        print(f"{'分位':>6}{'生成':>9}{'真实':>9}")
        for p_ in (5, 10, 25, 50, 75, 90, 95):
            print(f"{p_:>5}%{np.percentile(text,p_):>9.4f}{np.percentile(allreal,p_):>9.4f}")

        print("\n--- 空字段对照（每格：成文余弦均值 / pair 数），按 plan(无intent) 余弦三分位分层 ---")
        strata = np.quantile(plann, [0, 1/3, 2/3, 1.0])
        for name, key, label in (("content_angle=unclear_mixed", "ang", "两个都空/一个空/都不空"),
                                 ("perspective_id=seed_local", "per", "两个都空/一个空/都不空"),
                                 ("domain_intent 被兜底", "intent", "两个都兜底/一个/都没有"),
                                 ("story_mode=no_story", "story", "两个都是/一个/都不是")):
            g = np.array(pooled[key])
            print(f"\n  {name}   ({label})")
            print(f"    {'plan 层':>10}{'  2 个':>12}{'  1 个':>12}{'  0 个':>12}")
            for s in range(3):
                lo, hi = strata[s], strata[s+1]
                sel0 = (plann >= lo) & (plann <= hi if s == 2 else plann < hi)
                cells = []
                for v in (2, 1, 0):
                    ss = sel0 & (g == v)
                    cells.append(f"{text[ss].mean():.4f}/{ss.sum()}" if ss.sum() >= 15 else f"—/{ss.sum()}")
                print(f"    {s+1:>10}{cells[0]:>12}{cells[1]:>12}{cells[2]:>12}")


if __name__ == "__main__":
    main()
