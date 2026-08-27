"""Self-BLEU-4, computed EXACTLY as the evaluator does, on the matched pairs.

The first attempt sampled 12 comments per thread and got the opposite sign from
the matched evaluation, because Self-BLEU-4 is a whole-thread statistic and
subsampling changes it. This uses the full matched threads and reproduces the
evaluator's number before decomposing anything.
"""
import sys,json,re,random,statistics as st
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_self_bleu import symmetric_pair_bleu, tokenize
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
RUNS=REPO/"artifacts/generalized_card/runs"
POOL=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))
BY={int(r["seed_index"]):r for r in POOL["seed_posts"]}
SEED=re.compile(r"seed(\d+)$")
def thread_mean(texts):
    tok=[tokenize(t) for t in texts]; n=len(tok)
    if n<2: return float("nan")
    return st.mean([symmetric_pair_bleu(tok[i],tok[j],4) for i in range(n) for j in range(i+1,n)])
def load_gen(tag):
    gen={}
    for sd in sorted((RUNS/tag).glob("cleaned/run_*_sampled_reddit")):
        c,_=load_generated_comments(sd); gen.update(c)
    return gen
real_cache={}
for tag in ("v122_writer_retries_n10_20260828_v1","v125b_outsider_n10_20260828_v1"):
    gen=load_gen(tag)
    rows=[]
    gen_all=[]; real_all=[]
    for tid in sorted(gen):
        seed=BY[int(SEED.search(tid).group(1))]
        pd=REPO/"data/raw/discussions/camera_product"/str(seed["source_product_dir"])
        if pd not in real_cache: real_cache[pd]=load_real_comments(pd)[0]
        rid=str(seed["source_raw_post_id"])
        if rid not in real_cache[pd]: continue
        g=[c.text for c in gen[tid]]; r=[c.text for c in real_cache[pd][rid]]
        gm,rm=thread_mean(g),thread_mean(r)
        rows.append((tid,len(g),len(r),rm,gm))
        gen_all.append(g); real_all.append(r)
    print(f"=== {tag}")
    print(f"{'thread':<32}{'n_gen':>6}{'n_real':>7}{'real':>9}{'gen':>9}{'rel':>9}")
    for tid,ng,nr,rm,gm in rows:
        print(f"{tid:<32}{ng:>6}{nr:>7}{rm:>9.4f}{gm:>9.4f}{100*(gm-rm)/rm:>+8.1f}%")
    mr=st.mean([r for *_ ,r,_ in rows]); mg=st.mean([g for *_,g in rows])
    print(f"{'MEAN':<32}{'':>13}{mr:>9.4f}{mg:>9.4f}{100*(mg-mr)/mr:>+8.1f}%")
    # cross-thread: same tokenised pool, pairs across different threads
    def crossmean(groups,cap=6000):
        picked=[];origin=[]
        r=random.Random(20260827)
        for i,g in enumerate(groups):
            for x in r.sample(g,min(20,len(g))): picked.append(x); origin.append(i)
        tok=[tokenize(t) for t in picked]
        pairs=[(i,j) for i in range(len(tok)) for j in range(i+1,len(tok)) if origin[i]!=origin[j]]
        r.shuffle(pairs); pairs=pairs[:cap]
        return st.mean([symmetric_pair_bleu(tok[i],tok[j],4) for i,j in pairs])
    cg,cr=crossmean(gen_all),crossmean(real_all)
    print(f"cross-thread (20/thread, same pairs both sides): real {cr:.4f}  gen {cg:.4f}  "
          f"corpus term {cg-cr:+.4f}   whole gap {mg-mr:+.4f}")
    if mg-mr: print(f"   corpus share of the gap: {100*(cg-cr)/(mg-mr):.0f}%\n")
