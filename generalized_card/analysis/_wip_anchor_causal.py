"""Does seed-anchoring CAUSE pairwise similarity, or merely correlate?
Falsification-first: if the within-depth-bin slope on REAL data is ~0, the v109
hypothesis dies here for $0."""
import sys, re, json, statistics
from pathlib import Path
from collections import defaultdict
REPO=Path('/Users/yaoningyu/Desktop/UIUC/GEO')
sys.path.insert(0,str(REPO/'scripts/evaluation'))
from score_thread_semantic_uniformity import (CommentEmbedder, DEFAULT_MODEL,
    load_generated_comments, load_real_comments)
from score_thread_self_bleu import tokenize
import numpy as np

STOP=set('a an the and or but if of to in on for with at by from as is are was were be been being it its this that these those i you he she we they my your our their me him her us them do does did not no so too very can could will would should may might must have has had there here what which who when where how than then also just still really pretty s t re ve ll d m one two more most other same own'.split())
def cw(t): return {w for w in tokenize(t or '') if w.isalpha() and w not in STOP and len(w)>2}
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
real={}; real_seed={}; cache={}
for tid in gen:
    s=by_seed[int(SEED.search(tid).group(1))]
    pd=REPO/'data/raw/discussions/camera_product'/str(s['source_product_dir'])
    raw=str(s['source_raw_post_id'])
    if pd not in cache: cache[pd]=load_real_comments(pd)[0]
    real[raw]=cache[pd][raw]
    real_seed[raw]=s.get('title','')+' '+str(s.get('selftext','') or '')

emb=CommentEmbedder(model_name=DEFAULT_MODEL, device='cpu', max_length=256)
BINS=((0,1),(1,2),(2,4),(4,7),(7,999))
def db(d):
    for lo,hi in BINS:
        if lo<=d<hi: return "[%d,%s)"%(lo,hi if hi<999 else "+")

def collect(threads, seeds, label):
    rows=[]
    for tid,cs in threads.items():
        sc=cw(seeds.get(tid,''))
        if not sc or len(cs)<4: continue
        txt=[c.text for c in cs]
        V=emb.encode(txt, batch_size=32)
        V=V/np.linalg.norm(V,axis=1,keepdims=True)
        anc=[]; dep=[]
        for c in cs:
            w=cw(c.text); anc.append(len(w&sc)/len(w) if w else 0.0)
            dep.append(int(getattr(c,'depth',0) or 0))
        for i in range(len(cs)):
            for j in range(i+1,len(cs)):
                rows.append((db(max(dep[i],dep[j])), (anc[i]+anc[j])/2, float(V[i]@V[j])))
    print(f"  {label}: {len(rows)} pairs",flush=True)
    return rows

print("embedding...",flush=True)
GR=collect(gen,gen_seed,'generated'); RR=collect(real,real_seed,'real')

def slope(rows):
    x=[r[1] for r in rows]; y=[r[2] for r in rows]
    mx,my=statistics.fmean(x),statistics.fmean(y)
    d=sum((a-mx)**2 for a in x)
    if d==0: return float('nan'),float('nan')
    s=sum((a-mx)*(b-my) for a,b in zip(x,y))/d
    r=sum((a-mx)*(b-my) for a,b in zip(x,y))/((d*sum((b-my)**2 for b in y))**.5)
    return s,r

print("\n== WITHIN each depth bin: does joint seed-anchoring predict pair cosine? ==\n")
print("%-9s %18s %18s"%("depth","REAL slope (r)","GEN slope (r)"))
by=defaultdict(list)
for lo,hi in BINS:
    b=db(lo)
    rr=[x for x in RR if x[0]==b]; gg=[x for x in GR if x[0]==b]
    if len(rr)<50 or len(gg)<50: continue
    sr,rr_=slope(rr); sg,rg=slope(gg)
    by[b]=(sr,sg,len(rr),len(gg))
    print("%-9s %10.4f (%+.3f) %10.4f (%+.3f)"%(b,sr,rr_,sg,rg))

print("\n== the honest ablation estimate ==\n")
EXC={'[2,4)':(0.0552,0.0605,0.0173),'[4,7)':(0.0643,0.0336,0.0196),'[7,+)':(0.0776,0.0608,0.0284)}
tot=0.0
for b,(ga,ra,bex) in EXC.items():
    if b not in by: continue
    sr,sg,_,_=by[b]
    move=sg*(ra-ga)              # bring generated anchoring to real, at generated's own slope
    print("%-8s anchor %.4f -> %.4f (%+.4f) ; slope %.4f -> cosine change %+.5f  (bert excess %+.4f)"%(
        b,ga,ra,ra-ga,sg,move,bex))
    tot+=move
print("\n  NOTE cosine != BERTScore; this is a direction+magnitude sanity check on the")
print("  established mpnet proxy (G19), not a BERTScore prediction.")

print("\n== self-contained: cosine excess per bin, and what the taper predicts ==\n")
print("%-9s %10s %10s %10s %14s %10s"%("depth","gen cos","real cos","excess","predicted move","closes"))
for lo,hi in BINS:
    b=db(lo)
    if b not in by: continue
    gg=[x[2] for x in GR if x[0]==b]; rr=[x[2] for x in RR if x[0]==b]
    ga=statistics.fmean([x[1] for x in GR if x[0]==b]); ra=statistics.fmean([x[1] for x in RR if x[0]==b])
    gm,rm=statistics.fmean(gg),statistics.fmean(rr); exc=gm-rm
    sg=by[b][1]; move=sg*(ra-ga)
    pct=(100*(-move)/exc) if exc>0 else float('nan')
    print("%-9s %10.4f %10.4f %+10.4f %+14.5f %9.0f%%"%(b,gm,rm,exc,move,pct))
print("\n  'closes' = share of THAT BIN's cosine excess the taper would remove, at")
print("  generated's own within-bin slope. Upper bound (J7): assumes plan-level")
print("  tapering moves realized text anchoring 1:1, which in-text anchor")
print("  compliance (0.237 at [4,7)) says it will not.")
