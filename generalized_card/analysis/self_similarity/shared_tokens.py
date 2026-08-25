"""Which tokens are over-shared between comment pairs in generated threads."""
from __future__ import annotations
import json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO/"scripts"/"evaluation"))
from score_thread_self_bleu import tokenize
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments
RUN = REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
pool = json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(p["seed_index"]): p for p in pool}
rng = random.Random(0)

def pair_share(texts):
    """For each token: share of comment pairs in this thread that both contain it."""
    sets = [set(tokenize(t)) for t in texts]
    sets = [s for s in sets if s]
    n = len(sets)
    if n < 2: return {}, 0
    idx = [(i,j) for i in range(n) for j in range(i+1,n)]
    if len(idx) > 6000: idx = rng.sample(idx, 6000)
    c = Counter()
    for i,j in idx:
        for w in sets[i] & sets[j]:
            c[w] += 1
    return {w: v/len(idx) for w,v in c.items()}, n

gen_share, real_share = defaultdict(list), defaultdict(list)
gen_texts_all, real_texts_all = [], []
cache = {}
n_threads = 0
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt,_ = load_generated_comments(d)
    for tid, comments in cbt.items():
        seed = int(tid.split("seed")[-1])
        p = by_seed[seed]
        dd = REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
        if dd not in cache: cache[dd] = load_real_comments(dd)[0]
        rcs = cache[dd].get(p["source_raw_post_id"]) or []
        gt = [c.text for c in comments]; rt = [c.text for c in rcs]
        gs,_ = pair_share(gt); rs,_ = pair_share(rt)
        if not gs or not rs: continue
        n_threads += 1
        gen_texts_all += gt; real_texts_all += rt
        for w,v in gs.items(): gen_share[w].append(v)
        for w,v in rs.items(): real_share[w].append(v)

print(f"threads compared: {n_threads}")
vocab = set(gen_share) | set(real_share)
rows = []
for w in vocab:
    g = sum(gen_share.get(w,[]))/n_threads
    r = sum(real_share.get(w,[]))/n_threads
    rows.append((g-r, g, r, w))
rows.sort(reverse=True)
tot_g = sum(x[1] for x in rows); tot_r = sum(x[2] for x in rows)
print(f"mean shared-token mass per pair: real {tot_r:.3f}  generated {tot_g:.3f}  ({100*(tot_g-tot_r)/tot_r:+.1f}%)")
pos = sum(x[0] for x in rows if x[0] > 0)
print(f"total positive excess: {pos:.3f}\n")
print(f"{'token':<16}{'gen pair-share':>16}{'real':>9}{'excess':>9}{'cum % of excess':>17}")
cum = 0.0
for d_,g,r,w in rows[:40]:
    cum += d_
    print(f"{w:<16}{g:>16.4f}{r:>9.4f}{d_:>+9.4f}{100*cum/pos:>16.1f}%")
print()
STOPish = set("the a an and or but of to in for on with is are was were be been it its this that i you it's i'd i'm that's if so as at by from my your not no do does did have has had can could would should will just like about there here what when which who".split())
content_rows = [x for x in rows if x[3] not in STOPish and x[3].isalpha()]
func_excess = sum(x[0] for x in rows if x[3] in STOPish and x[0] > 0)
print(f"excess carried by ~70 function words: {func_excess:.3f}  = {100*func_excess/pos:.1f}% of all positive excess")
print(f"\ntop non-function-word excess:")
for d_,g,r,w in content_rows[:25]:
    print(f"  {w:<18} gen {g:.4f}  real {r:.4f}  {d_:+.4f}")
