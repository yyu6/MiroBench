#!/usr/bin/env python3
"""Where does the cross-branch excess come from -- the plan, or the realization?

  python3 experiments/geo_v137ds/cross_branch_residual.py a1gpt_20260902 [...]

89% of the within-thread cosine excess sits in pairs with NO reply relationship.
For those pairs this fits realized(plan) on the generated data and reports the
residual: a residual near zero means the Planner placed the slots that close and
the Writer transmitted it, a positive residual means the realization adds
similarity the plan did not ask for.  Length is reported alongside because a
shorter comment carries less to differentiate and moves cosine on its own.
"""
import collections, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL = "sentence-transformers/all-mpnet-base-v2"
PLAN_KEYS = ("semantic_move", "local_topic", "detail_focus", "domain_intent")


def real_threads():
    rows = collections.defaultdict(list)
    src = REPO / "data/raw/discussions/celebrity_geo/celebrity/celebrity.comments.jsonl"
    for line in src.open():
        c = json.loads(line); rows[str(c["post_id"])].append(c)
    out = {}
    for pid, cs in rows.items():
        by = {str(c["id"]): c for c in cs}
        def root(c, d=0):
            p = c.get("parent_comment_id")
            return str(c["id"]) if (not p or str(p) not in by or d > 40) else root(by[str(p)], d + 1)
        T, B = [], []
        for c in cs:
            t = (c.get("body") or "").strip()
            if t and t not in ("[deleted]", "[removed]"): T.append(t); B.append(root(c))
        out[pid] = (T, B)
    return out


def main():
    prefixes = sys.argv[1:] or ["a1gpt_20260902"]
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cpu")
    reals = real_threads()
    pool = json.loads((REPO / "artifacts/generalized_card/seed_pools/celebrity_geo_150_seed907.json").read_text())
    pid_of = {int(r["seed_index"]): str(r["source_raw_post_id"]) for r in pool["seed_posts"]}

    for prefix in prefixes:
        P, T, X, GW, RW = [], [], [], [], []      # plan cos, text cos, same-branch flag
        rtxt = []
        for d in sorted((REPO / "artifacts/generalized_card/runs").glob(f"{prefix}_p*")):
            f = d / "generated/run_00_sampled_reddit/generation_records.json"
            if not f.exists(): continue
            by = collections.defaultdict(list)
            for r in json.load(open(f)): by[int(r["seed_index"])].append(r)
            for si, rs in by.items():
                rs = [r for r in rs if str(r.get("raw") or "").strip()]
                if len(rs) < 6: continue
                par = {int(r["task"]["local_task_id"]):
                       (None if r["task"].get("local_parent_task_id") in (None, "") else int(r["task"]["local_parent_task_id"]))
                       for r in rs}
                def root(i, dd=0):
                    p_ = par.get(i)
                    return i if (p_ is None or p_ not in par or dd > 40) else root(p_, dd + 1)
                br = [root(int(r["task"]["local_task_id"])) for r in rs]
                texts = [str(r["raw"]).strip() for r in rs]
                plans = [" ".join(str(r["task"].get(k) or "") for k in PLAN_KEYS) for r in rs]
                E = m.encode(texts + plans, normalize_embeddings=True, batch_size=128,
                             convert_to_numpy=True, show_progress_bar=False)
                n = len(texts); et, ep = E[:n], E[n:]
                iu = np.triu_indices(n, 1)
                tc, pc = (et @ et.T)[iu], (ep @ ep.T)[iu]
                for a, b, t_, p_ in zip(iu[0], iu[1], tc, pc):
                    T.append(t_); P.append(p_); X.append(br[a] == br[b])
                GW += [len(t.split()) for t in texts]
                rt = reals.get(pid_of.get(si, ""))
                if rt: RW += [len(t.split()) for t in rt[0][:n]]; rtxt.append((si, rt, n))
        T, P, X = np.array(T), np.array(P), np.array(X)
        print(f"\n{'='*70}\n{prefix}\n{'='*70}")
        print(f"生成词数  中位 {np.median(GW):.0f}  均值 {np.mean(GW):.1f}  <=10词 {np.mean(np.array(GW)<=10):.1%}")
        print(f"真人词数  中位 {np.median(RW):.0f}  均值 {np.mean(RW):.1f}  <=10词 {np.mean(np.array(RW)<=10):.1%}")

        for name, sel in (("同 branch", X), ("跨 branch", ~X), ("全部", np.ones_like(X, bool))):
            p_, t_ = P[sel], T[sel]
            A = np.vstack([p_, np.ones_like(p_)]).T
            coef, *_ = np.linalg.lstsq(A, t_, rcond=None)
            print(f"\n{name}  ({sel.sum()} pair)")
            print(f"  plan 余弦 {p_.mean():.4f}   成文余弦 {t_.mean():.4f}")
            print(f"  拟合 成文 = {coef[0]:.3f} x plan + {coef[1]:.4f}   r = {np.corrcoef(p_, t_)[0,1]:.3f}")
        # 用同 branch 的 pair 拟合，去预测跨 branch 的 pair：realization 是不是同一套？
        A = np.vstack([P[X], np.ones(X.sum())]).T
        c1, *_ = np.linalg.lstsq(A, T[X], rcond=None)
        pred = c1[0] * P[~X] + c1[1]
        print(f"\n用【同 branch】拟合的 realization 去预测【跨 branch】:")
        print(f"  预测 {pred.mean():.4f}   实际 {T[~X].mean():.4f}   残差 {T[~X].mean()-pred.mean():+.4f}")
        print("  (残差近 0 => 两类 pair 的实现方式一样，差异全在 plan 距离上)")
        # 需要多低的 plan 余弦才能达到真人水平
        A = np.vstack([P, np.ones_like(P)]).T
        c, *_ = np.linalg.lstsq(A, T, rcond=None)
        print(f"\n全局 realization: 成文 = {c[0]:.3f} x plan + {c[1]:.4f}")
        for target, lbl in ((0.1274, "真人跨 branch"), (0.2753, "真人同 branch"), (0.1555, "真人总体")):
            need = (target - c[1]) / c[0]
            print(f"  要让成文余弦 = {target:.4f} ({lbl})，plan 余弦需 = {need:.4f}"
                  + ("   << 不可能，截距已高于目标" if need < 0 else ""))


if __name__ == "__main__":
    main()
