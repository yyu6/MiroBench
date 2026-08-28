"""Marginal novelty: how many NEW words does each comment bring to its thread?"""
from __future__ import annotations
import json, sys, re, statistics
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
TOK=re.compile(r"[a-z0-9']+")
STOP=set("the a an and or but if of to in on at for with is are was were be been it its this that these those i you he she we they my your his her our their as by from not no yes do does did so just have has had will would can could should there here what which who when how".split())

pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed={int(p["seed_index"]):p for p in pool}

def profile(texts):
    seen=set(); rows=[]
    for t in texts:
        toks=TOK.findall(t.lower())
        content=[w for w in toks if w not in STOP and len(w)>2]
        cs=set(content)
        new=cs-seen
        seen|=cs
        rows.append((len(toks), len(cs), len(new)))
    return rows

for RUN,label in [("generalized_card_camera_gpt54_paper_20260825_v1","paper50"),
                  ("v128_interaction_n10_20260828_v1","v128")]:
    base=REPO/"artifacts/generalized_card/runs"/RUN/"cleaned"
    cache={}; G=[]; R=[]; per_thread=[]
    for x in sorted(base.glob("run_*_sampled_reddit")):
        cbt,_=load_generated_comments(x)
        for tid,cs in cbt.items():
            seed=int(tid.split("seed")[-1]); p=by_seed.get(seed)
            if not p: continue
            dd=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
            if dd not in cache:
                try: cache[dd]=load_real_comments(dd)[0]
                except Exception: cache[dd]={}
            rcs=cache[dd].get(p["source_raw_post_id"]) or []
            if len(rcs)<15 or len(cs)<15: continue
            g=profile([c.text for c in cs]); r=profile([c.text for c in rcs])
            G+=g; R+=r
            gn=statistics.mean([a[2]/max(1,a[0]) for a in g]); rn=statistics.mean([a[2]/max(1,a[0]) for a in r])
            per_thread.append((gn,rn))
    if not G: print(f"{label}: none"); continue
    def s(rows,k): return statistics.mean([x[k] for x in rows])
    print(f"=== {label}: {len(per_thread)} matched threads, {len(G)} gen / {len(R)} real comments")
    print(f"  {'':<22} {'real':>9} {'gen':>9} {'gen/real':>9}")
    print(f"  {'words per comment':<22} {s(R,0):9.2f} {s(G,0):9.2f} {s(G,0)/s(R,0):9.3f}")
    print(f"  {'content types':<22} {s(R,1):9.2f} {s(G,1):9.2f} {s(G,1)/s(R,1):9.3f}")
    print(f"  {'NEW types to thread':<22} {s(R,2):9.2f} {s(G,2):9.2f} {s(G,2)/s(R,2):9.3f}")
    nr=statistics.mean([x[2]/max(1,x[0]) for x in R]); ng=statistics.mean([x[2]/max(1,x[0]) for x in G])
    print(f"  {'NEW types per word':<22} {nr:9.4f} {ng:9.4f} {ng/nr:9.3f}")
    nr2=statistics.mean([x[2]/max(1,x[1]) for x in R]); ng2=statistics.mean([x[2]/max(1,x[1]) for x in G])
    print(f"  {'NEW / own types':<22} {nr2:9.4f} {ng2:9.4f} {ng2/nr2:9.3f}")
    w=sum(1 for g,r in per_thread if g<r)
    print(f"  gen novelty-per-word below real in {w}/{len(per_thread)} threads")
    print()
