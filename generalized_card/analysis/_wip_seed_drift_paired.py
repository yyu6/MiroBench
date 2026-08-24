# Per-thread paired test of the drift finding + TEST B: function-word register variance
import sys, re, json, statistics
from pathlib import Path
from collections import defaultdict, Counter
REPO=Path('/Users/yaoningyu/Desktop/UIUC/GEO')
sys.path.insert(0,str(REPO/'scripts/evaluation'))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
from score_thread_self_bleu import tokenize
STOP=set('a an the and or but if of to in on for with at by from as is are was were be been being it its this that these those i you he she we they my your our their me him her us them do does did not no so too very can could will would should may might must have has had there here what which who when where how than then also just still really pretty s t re ve ll d m'.split())
def content(t): return {w for w in tokenize(t) if w.isalpha() and w not in STOP and len(w)>2}
pool=json.load(open(REPO/'artifacts/generalized_card/seed_pools/camera_product_150_seed42.json'))
by_seed={int(r['seed_index']):r for r in pool['seed_posts']}
SEED=re.compile(r'seed(\d+)$')
run=REPO/'artifacts/generalized_card/runs/generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1'
gen_seed={}
for f in sorted(run.glob('generated/run_*/discussion.json')):
    for p in json.load(open(f))['posts']: gen_seed[p['post_id']]=p.get('title','')+' '+p.get('content','')
gen={}
for d in sorted(run.glob('cleaned/run_*_sampled_reddit')):
    c,_=load_generated_comments(d); gen.update(c)
pairs=[]; cache={}
for tid in gen:
    s=by_seed[int(SEED.search(tid).group(1))]
    pd=REPO/'data/raw/discussions/camera_product'/str(s['source_product_dir'])
    raw=str(s['source_raw_post_id'])
    if pd not in cache: cache[pd]=load_real_comments(pd)[0]
    pairs.append((tid,raw,cache[pd][raw],s.get('title','')+' '+str(s.get('selftext','') or '')))

# --- per-thread paired: deep-comment (depth>=4) anchoring ---
print("TEST A2 -- per-thread paired, comments at depth>=4 only\n")
print(f"{'seed':>5s} {'gen anchor':>11s} {'real anchor':>12s} {'gen-real':>9s}")
diffs=[]
for tid,raw,rcs,stext in pairs:
    sg=content(gen_seed.get(tid,'')); sr=content(stext)
    g=[len(content(c.text)&sg)/len(content(c.text)) for c in gen[tid] if int(getattr(c,'depth',0) or 0)>=4 and content(c.text)]
    r=[len(content(c.text)&sr)/len(content(c.text)) for c in rcs if int(getattr(c,'depth',0) or 0)>=4 and content(c.text)]
    if len(g)<3 or len(r)<3: continue
    gm,rm=statistics.fmean(g),statistics.fmean(r); diffs.append(gm-rm)
    print(f"{SEED.search(tid).group(1):>5s} {gm:11.4f} {rm:12.4f} {gm-rm:+9.4f}")
from scipy.stats import wilcoxon
print(f"\nthreads where generated is MORE seed-anchored at depth>=4: {sum(1 for d in diffs if d>0)}/{len(diffs)}")
print(f"mean difference {statistics.fmean(diffs):+.4f}   Wilcoxon p = {wilcoxon(diffs).pvalue:.4f}")

# --- TEST B: function-word register spread between comments in a thread ---
FUNC=['the',',','.','i','you','to','a','and','it','that','is','of','my','but','for','?','!','not','have','would','will','so','just','if','with','they','we','this','do','get','be','are','was','on','at','or','no','yes','me','your','what']
def fvec(toks):
    n=len(toks) or 1
    c=Counter(toks)
    return [c[f]/n for f in FUNC]
def spread(threads):
    out=[]
    for cs in threads:
        vs=[fvec(tokenize(c.text)) for c in cs if tokenize(c.text)]
        if len(vs)<5: continue
        # mean pairwise L1 distance between comments' function-word profiles
        ds=[]
        for i in range(len(vs)):
            for j in range(i+1,len(vs)):
                ds.append(sum(abs(a-b) for a,b in zip(vs[i],vs[j])))
        out.append(statistics.fmean(ds))
    return out
G=spread([gen[t] for t,_,_,_ in pairs]); R=spread([r for _,_,r,_ in pairs])
print(f"\nTEST B -- mean pairwise L1 distance between comments' FUNCTION-WORD profiles")
print(f"  generated {statistics.fmean(G):.4f}   real {statistics.fmean(R):.4f}   ratio {statistics.fmean(G)/statistics.fmean(R):.3f}")
print(f"  threads where generated is LESS varied: {sum(1 for a,b in zip(G,R) if a<b)}/{len(G)}")
print(f"  Wilcoxon p = {wilcoxon([a-b for a,b in zip(G,R)]).pvalue:.4f}")
