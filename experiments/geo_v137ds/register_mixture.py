#!/usr/bin/env python3
"""Price a mixed-writer arm before paying for it, and check the persona lever.

  python3 experiments/geo_v137ds/register_mixture.py

The register excess is the same size across unrelated threads as inside one, so
it belongs to the writer and its fixed prompt scaffold.  Two things follow that
can be measured on runs already paid for:

  1. how far apart two DIFFERENT writers' registers are.  If a gpt comment and a
     deepseek comment are as far apart as two human comments, then drawing each
     slot's writer from a pool lands between them and the intervention is worth
     an arm; if the two models share one register, it is not.
  2. how many personas the `best - 1` near-best band actually admits.  A band
     that keeps most of the eligible population makes `matraix-projected` a
     near-random draw, and neither persona arm was then a test of persona.
"""
import collections, itertools, json, random, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surface_vs_content import profiles, real_by_seed, gen_by_seed  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def pooled(threads, cap=40):
    random.seed(11)
    out = []
    for t in threads:
        V = profiles(t)[0]
        if len(V) > cap: V = V[random.sample(range(len(V)), cap)]
        out.append(V)
    return out


def mean_cross(A, B):
    return float(np.mean([float(a @ b.T) for a, b in itertools.product([x for x in A], [y for y in B])])) \
        if False else float(np.mean([float((a @ b.T).mean()) for a in A for b in B]))


def main():
    reals = real_by_seed()
    rt = pooled([reals[s][:120] for s in sorted(reals) if len(reals.get(s, [])) >= 6][:20])
    arms = {}
    for p in ("a1gpt_20260902", "v150_20260902", "a4fit_20260902", "a3both_20260902"):
        G = gen_by_seed(p)
        if G: arms[p] = (pooled([G[s] for s in sorted(G)]), G)

    def wi(P):
        v = []
        for V in P:
            iu = np.triu_indices(len(V), 1)
            v.append((V @ V.T)[iu].mean())
        return float(np.mean(v))

    print("功能词画像余弦。真人 thread 内 = 基准\n")
    print(f"真人 thread 内                     {wi(rt):.4f}")
    print(f"真人 跨 thread                     {mean_cross(rt, rt):.4f}\n")
    names = list(arms)
    for n in names:
        print(f"{n:<32} thread 内 {wi(arms[n][0]):.4f}")
    print()
    print("两支之间（不同 writer / 不同 persona 设置的评论互相比）:")
    for a, b in itertools.combinations(names, 2):
        c = mean_cross(arms[a][0], arms[b][0])
        print(f"  {a.split('_')[0]:>7} x {b.split('_')[0]:<8} {c:.4f}"
              f"   (对真人跨 thread {mean_cross(rt,rt):.4f} 超 {(c-mean_cross(rt,rt))/mean_cross(rt,rt):+.0%})")
    # 把两支的评论混成一个"thread"，看混合能把 register 余弦压到多少
    print("\n把两支的评论按 1:1 混进同一个 thread（模拟每个 slot 随机抽 writer）:")
    base = mean_cross(rt, rt)
    for a, b in itertools.combinations(names, 2):
        Ga, Gb = arms[a][1], arms[b][1]
        shared = sorted(set(Ga) & set(Gb))
        vals = []
        for s in shared:
            ta, tb = Ga[s], Gb[s]
            k = min(len(ta), len(tb))
            mix = [ta[i] if i % 2 == 0 else tb[i] for i in range(k)]
            if len(mix) < 6: continue
            V = profiles(mix)[0]
            iu = np.triu_indices(len(V), 1)
            vals.append((V @ V.T)[iu].mean())
        if vals:
            v = float(np.mean(vals))
            print(f"  {a.split('_')[0]:>7} + {b.split('_')[0]:<8} {v:.4f}"
                  f"   vs 真人 thread 内 {wi(rt):.4f}  超 {(v-wi(rt))/wi(rt):+.0%}   ({len(vals)} thread)")


if __name__ == "__main__":
    main()
