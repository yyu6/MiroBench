#!/usr/bin/env python3
"""Same-branch vs cross-branch comment cosine, generated against real.

Both sides define a branch the SAME way -- one top-level comment and its whole
subtree, walked from the reply tree.  An earlier version read the generated
side's branch off the Planner's `branch_id` grid, which is a different object
from a reply-tree root, and the 9-vs-18 branch count it produced was that
mismatch and not a finding.

  python3 experiments/geo_v137ds/branch_cohesion.py a1gpt_20260902 [more...]

A branch is one top-level comment and its whole subtree.  Every generated thread
is built on its matched real thread's structure, so both sides have comparable
branch partitions and the two numbers are directly comparable: if the real
thread's own branches are as cohesive as ours, branch clustering is not the
defect; if they are not, the branch grid is binding harder than the humans did.
"""
import collections, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL = "sentence-transformers/all-mpnet-base-v2"


def real_threads():
    """post_id -> (texts, branch_id per text)"""
    rows = collections.defaultdict(list)
    src = REPO / "data/raw/discussions/celebrity_geo/celebrity/celebrity.comments.jsonl"
    for line in src.open():
        c = json.loads(line)
        rows[str(c["post_id"])].append(c)
    out = {}
    for pid, cs in rows.items():
        by_id = {str(c["id"]): c for c in cs}
        def root(c, depth=0):
            p = c.get("parent_comment_id")
            if not p or str(p) not in by_id or depth > 40:
                return str(c["id"])
            return root(by_id[str(p)], depth + 1)
        def depth(c, d=0):
            p = c.get("parent_comment_id")
            return d if (not p or str(p) not in by_id or d > 40) else depth(by_id[str(p)], d + 1)
        texts, br, dep = [], [], []
        for c in cs:
            t = (c.get("body") or "").strip()
            if not t or t in ("[deleted]", "[removed]"): continue
            texts.append(t); br.append(root(c)); dep.append(depth(c))
        out[pid] = (texts, br, (sum(dep) / len(dep)) if dep else 0.0)
    return out


def gen_threads(prefix):
    out = []
    for d in sorted((REPO / "artifacts/generalized_card/runs").glob(f"{prefix}_p*")):
        f = d / "generated/run_00_sampled_reddit/generation_records.json"
        if not f.exists(): continue
        by = collections.defaultdict(list)
        for r in json.load(open(f)): by[int(r["seed_index"])].append(r)
        for si, rs in by.items():
            rs = [r for r in rs if str(r.get("raw") or "").strip()]
            if len(rs) < 6: continue
            # branch = 回复树的顶层祖先，和真人侧同一个构造
            par = {int(r["task"]["local_task_id"]):
                   (None if r["task"].get("local_parent_task_id") in (None, "")
                    else int(r["task"]["local_parent_task_id"])) for r in rs}
            def root(i, d=0):
                p_ = par.get(i)
                return i if (p_ is None or p_ not in par or d > 40) else root(p_, d + 1)
            def depth(i, d=0):
                p_ = par.get(i)
                return d if (p_ is None or p_ not in par or d > 40) else depth(p_, d + 1)
            ids = [int(r["task"]["local_task_id"]) for r in rs]
            out.append((si, [str(r["raw"]).strip() for r in rs],
                        [str(root(i)) for i in ids],
                        float(sum(depth(i) for i in ids) / len(ids)),
                        [str(r["task"].get("branch_id")) for r in rs]))
    return out


def split(m, texts, br):
    E = m.encode(texts, normalize_embeddings=True, batch_size=128,
                 convert_to_numpy=True, show_progress_bar=False)
    s = E @ E.T
    iu = np.triu_indices(len(texts), 1)
    same = np.array([br[a] == br[b] for a, b in zip(*iu)])
    v = s[iu]
    return v, same


def main():
    prefixes = sys.argv[1:] or ["a1gpt_20260902"]
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cpu")
    reals = real_threads()
    pool = json.loads((REPO / "artifacts/generalized_card/seed_pools/celebrity_geo_150_seed907.json").read_text())
    pid_of = {int(r["seed_index"]): str(r["source_raw_post_id"]) for r in pool["seed_posts"]}

    for prefix in prefixes:
        G = gen_threads(prefix)
        print(f"\n{'='*72}\n{prefix}  ({len(G)} thread)\n{'='*72}")
        print(f"{'seed':>5}  {'生成:同branch':>13}{'跨branch':>10}{'branch数':>9}"
              f" | {'真人:同branch':>13}{'跨branch':>10}{'branch数':>9}")
        acc = collections.defaultdict(list)
        for si, texts, br, gdep, pbr in G:
            gv, gs = split(m, texts, br)
            rt = reals.get(pid_of.get(si, ""))
            if not rt or len(rt[0]) < 6: continue
            rv, rs = split(m, rt[0][:len(texts)], rt[1][:len(texts)])
            g1 = gv[gs].mean() if gs.any() else float("nan")
            g0 = gv[~gs].mean() if (~gs).any() else float("nan")
            r1 = rv[rs].mean() if rs.any() else float("nan")
            r0 = rv[~rs].mean() if (~rs).any() else float("nan")
            print(f"{si:>5}  {g1:>13.4f}{g0:>10.4f}{len(set(br)):>9}"
                  f" | {r1:>13.4f}{r0:>10.4f}{len(set(rt[1][:len(texts)])):>9}"
                  f"   深度 {gdep:.2f}/{rt[2]:.2f}  Planner网格 {len(set(pbr))}")
            acc["g_same"].append(gv[gs]); acc["g_diff"].append(gv[~gs])
            acc["r_same"].append(rv[rs]); acc["r_diff"].append(rv[~rs])
            acc["g_nb"].append(len(set(br))); acc["r_nb"].append(len(set(rt[1][:len(texts)])))
            acc["per_g"].append((gv[gs].mean() if gs.any() else np.nan,
                                 gv[~gs].mean() if (~gs).any() else np.nan, gs.mean()))
            acc["per_r"].append((rv[rs].mean() if rs.any() else np.nan,
                                 rv[~rs].mean() if (~rs).any() else np.nan, rs.mean()))
        gS = np.concatenate(acc["g_same"]); gD = np.concatenate(acc["g_diff"])
        rS = np.concatenate(acc["r_same"]); rD = np.concatenate(acc["r_diff"])
        print(f"\n{'合并':>5}  {gS.mean():>13.4f}{gD.mean():>10.4f}{np.mean(acc['g_nb']):>9.1f}"
              f" | {rS.mean():>13.4f}{rD.mean():>10.4f}{np.mean(acc['r_nb']):>9.1f}")
        print(f"\n  同 branch pair 占比   生成 {len(gS)/(len(gS)+len(gD)):.1%}   真人 {len(rS)/(len(rS)+len(rD)):.1%}")
        print(f"  同 branch 超标        {(gS.mean()-rS.mean())/rS.mean():+.1%}")
        print(f"  跨 branch 超标        {(gD.mean()-rD.mean())/rD.mean():+.1%}")
        gm = (gS.sum()+gD.sum())/(len(gS)+len(gD)); rm = (rS.sum()+rD.sum())/(len(rS)+len(rD))
        exc = gm - rm
        # 把总超标拆成"同branch 更粘" 与 "跨branch 更粘" 两块
        w = len(gS)/(len(gS)+len(gD))
        print(f"\n  总超标 {gm:.4f} - {rm:.4f} = {exc:+.4f}")
        print(f"    其中同 branch 贡献 {w*(gS.mean()-rS.mean()):+.4f} ({w*(gS.mean()-rS.mean())/exc:.0%})")
        print(f"    跨 branch 贡献     {(1-w)*(gD.mean()-rD.mean()):+.4f} ({(1-w)*(gD.mean()-rD.mean())/exc:.0%})")
        # 指标是 per-thread 的，所以按 thread 取平均再拆一次
        pg = np.array(acc["per_g"]); pr = np.array(acc["per_r"])
        ok = ~np.isnan(pg[:, 0]) & ~np.isnan(pr[:, 0])
        gS_, gD_, gw = np.nanmean(pg[ok, 0]), np.nanmean(pg[ok, 1]), np.nanmean(pg[ok, 2])
        rS_, rD_, rw = np.nanmean(pr[ok, 0]), np.nanmean(pr[ok, 1]), np.nanmean(pr[ok, 2])
        print(f"\n  --- 按 thread 平均（和指标同一种加权，{ok.sum()} 个 thread）---")
        print(f"    生成 同 branch {gS_:.4f}  跨 branch {gD_:.4f}  同 branch 占 {gw:.1%}")
        print(f"    真人 同 branch {rS_:.4f}  跨 branch {rD_:.4f}  同 branch 占 {rw:.1%}")
        gm2 = gw * gS_ + (1 - gw) * gD_; rm2 = rw * rS_ + (1 - rw) * rD_
        print(f"    总 {gm2:.4f} vs {rm2:.4f} = {gm2 - rm2:+.4f}")
        print(f"      同 branch 更粘贡献        {gw * (gS_ - rS_):+.4f}")
        print(f"      跨 branch 更粘贡献        {(1 - gw) * (gD_ - rD_):+.4f}")
        print(f"      同 branch pair 更多贡献   {(gw - rw) * (rS_ - rD_):+.4f}")


if __name__ == "__main__":
    main()
