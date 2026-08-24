# TEST A: do real reply chains drift away from the seed post, while generated ones stay anchored?
import sys, re, json, statistics
from pathlib import Path
from collections import defaultdict
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

# generated: seed text from discussion.json
gen_seed={}
for f in sorted(run.glob('generated/run_*/discussion.json')):
    for p in json.load(open(f))['posts']:
        gen_seed[p['post_id']]=(p.get('title','')+' '+p.get('content',''))
gen={}
for d in sorted(run.glob('cleaned/run_*_sampled_reddit')):
    c,_=load_generated_comments(d); gen.update(c)

real={}; real_seed={}; cache={}
for tid in gen:
    s=by_seed[int(SEED.search(tid).group(1))]
    pd=REPO/'data/raw/discussions/camera_product'/str(s['source_product_dir'])
    raw=str(s['source_raw_post_id'])
    if pd not in cache: cache[pd]=load_real_comments(pd)
    allc,meta=cache[pd]
    real[raw]=allc[raw]
    st=''
    for m in (meta or {}).get(raw,{}) if isinstance(meta,dict) else []:
        pass
    real_seed[raw]=s.get('title','')+' '+str(s.get('selftext','') or s.get('body','') or '')

BINS=((0,1),(1,2),(2,4),(4,7),(7,999))
def db(d):
    for lo,hi in BINS:
        if lo<=d<hi: return (lo,hi)
def profile(comments_by_thread, seedtext):
    acc=defaultdict(list)
    for tid,cs in comments_by_thread.items():
        sc=content(seedtext.get(tid,''))
        if not sc: continue
        for c in cs:
            cc=content(c.text)
            if not cc: continue
            k=db(int(getattr(c,'depth',0) or 0))
            if k is None: continue
            acc[k].append(len(cc&sc)/len(cc))   # share of a comment's content words that come from the seed
    return acc
G=profile(gen,gen_seed); R=profile(real,real_seed)
print("TEST A -- share of a comment's content words that also appear in the SEED POST\n")
print(f"{'depth':>8s} {'gen n':>6s} {'gen anchor':>11s} {'real n':>7s} {'real anchor':>12s} {'gen-real':>9s}")
for lo,hi in BINS:
    g,r=G.get((lo,hi),[]),R.get((lo,hi),[])
    if not g or not r: continue
    lbl=f"[{lo},{hi})" if hi<999 else f"[{lo},+)"
    gm,rm=statistics.fmean(g),statistics.fmean(r)
    print(f"{lbl:>8s} {len(g):6d} {gm:11.4f} {len(r):7d} {rm:12.4f} {gm-rm:+9.4f}")
allg=[x for v in G.values() for x in v]; allr=[x for v in R.values() for x in v]
print(f"\n{'POOLED':>8s} {len(allg):6d} {statistics.fmean(allg):11.4f} {len(allr):7d} {statistics.fmean(allr):12.4f} {statistics.fmean(allg)-statistics.fmean(allr):+9.4f}")
