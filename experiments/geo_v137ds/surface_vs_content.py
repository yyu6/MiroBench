#!/usr/bin/env python3
"""Split the self-similarity excess into a content part and a wording part.

  python3 experiments/geo_v137ds/surface_vs_content.py a1gpt_20260902 v150_20260902

`semantic_mean_cosine` and `self_bertscore_mean_f1` fail together but move in
opposite directions across writers, so they are not one quantity.  BERTScore
aligns tokens greedily and is therefore lifted by shared wording, including
function words, while the mpnet cosine responds to content.  This reports, over
the same pairs and against each thread's own matched real thread:

  content  -- Jaccard over content-word types (stopwords removed)
  wording  -- Jaccard over function words only, and over 4-grams
  register -- cosine between per-comment function-word frequency profiles

Any of these being in line with the real corpus while another is not localises
the failure to that level, which the two aggregate metrics cannot do.
"""
import collections, json, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
TOK = re.compile(r"[a-z0-9']+")
# 只用高频功能词，和内容无关，能代表"说话方式"
FUNC = set("""a an the and or but so because if then than that this these those there here
it its it's he she they them him her his their our your my me we you i is are was were be been
being am do does did doing done have has had having will would can could should shall may might must
of in on at to for with from by about into over after before under above between out up down off
not no nor only just even also too very really quite pretty much more most less least same other
what which who whom whose when where why how all any both each few many some such own s t don didn
doesn isn aren wasn weren won wouldn couldn shouldn ll re ve m d ain got get like well yeah oh
""".split())


def toks(t): return TOK.findall(t.lower())


def real_by_seed():
    by = collections.defaultdict(list)
    src = REPO / "data/raw/discussions/celebrity_geo/celebrity/celebrity.comments.jsonl"
    for line in src.open():
        c = json.loads(line)
        t = (c.get("body") or "").strip()
        if t and t not in ("[deleted]", "[removed]"): by[str(c["post_id"])].append(t)
    pool = json.loads((REPO / "artifacts/generalized_card/seed_pools/celebrity_geo_150_seed907.json").read_text())
    return {int(r["seed_index"]): by.get(str(r["source_raw_post_id"]), []) for r in pool["seed_posts"]}


def jac(a, b): return len(a & b) / len(a | b) if (a | b) else 0.0


def profiles(texts):
    """每条评论的功能词频率向量 + 内容词集合 + 功能词集合 + 4-gram 集合"""
    keys = sorted(FUNC)
    idx = {k: i for i, k in enumerate(keys)}
    V, C, F, G = [], [], [], []
    for t in texts:
        w = toks(t)
        v = np.zeros(len(keys))
        for x in w:
            if x in idx: v[idx[x]] += 1
        n = v.sum()
        V.append(v / n if n else v)
        C.append({x for x in w if x not in FUNC})
        F.append({x for x in w if x in FUNC})
        G.append({tuple(w[i:i+4]) for i in range(max(0, len(w) - 3))})
    V = np.array(V)
    nz = np.linalg.norm(V, axis=1, keepdims=True); nz[nz == 0] = 1
    return V / nz, C, F, G


def stats(texts):
    if len(texts) < 4: return None
    V, C, F, G = profiles(texts)
    iu = np.triu_indices(len(texts), 1)
    reg = (V @ V.T)[iu]
    cj = np.array([jac(C[a], C[b]) for a, b in zip(*iu)])
    fj = np.array([jac(F[a], F[b]) for a, b in zip(*iu)])
    gj = np.array([jac(G[a], G[b]) for a, b in zip(*iu)])
    return reg.mean(), cj.mean(), fj.mean(), gj.mean()


def gen_by_seed(prefix):
    out = {}
    for d in sorted((REPO / "artifacts/generalized_card/runs").glob(f"{prefix}_p*")):
        f = d / "generated/run_00_sampled_reddit/generation_records.json"
        if not f.exists(): continue
        by = collections.defaultdict(list)
        for r in json.load(open(f)): by[int(r["seed_index"])].append(r)
        for si, rs in by.items():
            t = [str(r.get("raw") or "").strip() for r in rs if str(r.get("raw") or "").strip()]
            if len(t) >= 4: out[si] = t
    return out


def main():
    prefixes = sys.argv[1:] or ["a1gpt_20260902"]
    reals = real_by_seed()
    for prefix in prefixes:
        G = gen_by_seed(prefix)
        seeds = [s for s in sorted(G) if len(reals.get(s, [])) >= 4]
        rowsg, rowsr = [], []
        for s in seeds:
            g = stats(G[s]); r = stats(reals[s][:len(G[s])])
            if g and r: rowsg.append(g); rowsr.append(r)
        g = np.array(rowsg); r = np.array(rowsr)
        print(f"\n{prefix}   {len(rowsg)} 个 thread（每个都对自己的 matched real thread 比）")
        print(f"{'层面':<28}{'生成':>9}{'真人':>9}{'超标':>9}")
        print("-"*56)
        for i, name in enumerate(("功能词频率画像 余弦 (说话方式)", "内容词 Jaccard (说什么)",
                                  "功能词 Jaccard", "4-gram Jaccard (成句)")):
            a, b = g[:, i].mean(), r[:, i].mean()
            print(f"{name:<28}{a:>9.4f}{b:>9.4f}{(a-b)/b:>+8.1%}")





def cross_thread(prefixes):
    """Within-thread vs across-thread register similarity.

    A register excess that is the same size across unrelated threads is a
    property of the writer, not of anything the thread's plan says; one that
    only appears inside a thread is being driven by that thread's prompt.
    """
    import itertools, random
    reals = real_by_seed()
    print(f"\n{'='*74}\n功能词画像余弦：thread 内 vs 跨 thread（不同帖子、不同 plan）\n{'='*74}")
    print(f"{'':<22}{'thread 内':>10}{'跨 thread':>11}{'内/跨':>8}")
    random.seed(7)

    def measure(threads):
        wi, ac = [], []
        prof = [profiles(t)[0] for t in threads]
        for V in prof:
            iu = np.triu_indices(len(V), 1)
            wi.append((V @ V.T)[iu].mean())
        for a, b in itertools.combinations(range(len(prof)), 2):
            A, B = prof[a], prof[b]
            if len(A) > 40: A = A[random.sample(range(len(A)), 40)]
            if len(B) > 40: B = B[random.sample(range(len(B)), 40)]
            ac.append(float((A @ B.T).mean()))
        return np.mean(wi), np.mean(ac)

    rt = [reals[s][:120] for s in sorted(reals) if len(reals.get(s, [])) >= 6][:20]
    rw, ra = measure(rt)
    print(f"{'真人':<22}{rw:>10.4f}{ra:>11.4f}{rw/ra:>8.2f}")
    for prefix in prefixes:
        G = gen_by_seed(prefix)
        gt = [G[s] for s in sorted(G)]
        gw, ga = measure(gt)
        print(f"{prefix:<22}{gw:>10.4f}{ga:>11.4f}{gw/ga:>8.2f}"
              f"   超标 内 {(gw-rw)/rw:+.0%}  跨 {(ga-ra)/ra:+.0%}")


if __name__ == "__main__":
    main()
    cross_thread(sys.argv[1:] or ["a1gpt_20260902"])
