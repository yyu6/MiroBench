"""The decisive test on the REAL scorer: does seed-anchoring predict pairwise
BERTScore F1, within depth bin, on both sides? Determines which depths the
v109 taper may safely touch."""
import sys, re, json, statistics
from pathlib import Path
from collections import defaultdict
REPO=Path('/Users/yaoningyu/Desktop/UIUC/GEO')
sys.path.insert(0,str(REPO/'scripts/evaluation'))
sys.path.insert(0,str(REPO/'generalized_card/analysis'))
from bertscore_pair_diagnosis import Threads, cmd_fidelity
from score_thread_self_bleu import tokenize

STOP=set('a an the and or but if of to in on for with at by from as is are was were be been being it its this that these those i you he she we they my your our their me him her us them do does did not no so too very can could will would should may might must have has had there here what which who when where how than then also just still really pretty s t re ve ll d m one two more most other same own'.split())
def cw(t): return {w for w in tokenize(t or '') if w.isalpha() and w not in STOP and len(w)>2}
pool=json.load(open(REPO/'artifacts/generalized_card/seed_pools/camera_product_150_seed42.json'))
by_seed={int(r['seed_index']):r for r in pool['seed_posts']}
SEED=re.compile(r'seed(\d+)$')
run=REPO/'artifacts/generalized_card/runs/generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1'

th=Threads(run)
gen_pairs, real_pairs = cmd_fidelity(th, 'cpu', 16)

gen_seed={}
for f in sorted(run.glob('generated/run_*/discussion.json')):
    for p in json.load(open(f))['posts']: gen_seed[p['post_id']]=p.get('title','')+' '+p.get('content','')
real_seed={}
for tid,_ in th.pairing:
    s=by_seed[int(SEED.search(tid).group(1))]
    real_seed[str(s['source_raw_post_id'])]=s.get('title','')+' '+str(s.get('selftext','') or '')

# anchoring per comment id, per thread
anc={}
for tid,cs in th.generated.items():
    sc=cw(gen_seed.get(tid,''))
    for c in cs:
        w=cw(c.text); anc[(tid,c.comment_id)]=len(w&sc)/len(w) if (w and sc) else 0.0
for tid,cs in th.real.items():
    sc=cw(real_seed.get(tid,''))
    for c in cs:
        w=cw(c.text); anc[(tid,c.comment_id)]=len(w&sc)/len(w) if (w and sc) else 0.0

BINS=((0,1),(1,2),(2,4),(4,7),(7,999))
def db(d):
    for lo,hi in BINS:
        if lo<=d<hi: return "[%d,%s)"%(lo,hi if hi<999 else "+")
def rows_for(pairs):
    out=defaultdict(list)
    for p in pairs:
        t=p.get('thread_id')
        a=anc.get((t,p.get('left_comment_id'))); b=anc.get((t,p.get('right_comment_id')))
        if a is None or b is None: continue
        k=db(max(int(p['left_depth']),int(p['right_depth'])))
        if k: out[k].append(((a+b)/2, p['bert_f1']))
    return out
G=rows_for(gen_pairs); R=rows_for(real_pairs)
def slope(rs):
    x=[r[0] for r in rs]; y=[r[1] for r in rs]
    mx,my=statistics.fmean(x),statistics.fmean(y); d=sum((a-mx)**2 for a in x)
    if not d: return float('nan'),float('nan')
    s=sum((a-mx)*(b-my) for a,b in zip(x,y))/d
    r=s*(d/sum((b-my)**2 for b in y))**.5
    return s,r
print("\n== BERTScore F1 vs joint seed-anchoring, WITHIN depth bin (real scorer) ==\n")
print("%-9s %8s %20s %20s"%("depth","pairs","REAL slope (r)","GEN slope (r)"))
res={}
for lo,hi in BINS:
    b=db(lo); rr,gg=R.get(b,[]),G.get(b,[])
    if len(rr)<50 or len(gg)<50: continue
    sr,cr=slope(rr); sg,cg=slope(gg); res[b]=(sr,sg)
    print("%-9s %8d %12.4f (%+.3f) %12.4f (%+.3f)"%(b,len(gg),sr,cr,sg,cg))
EXC={'[2,4)':(0.0552,0.0605,0.0173),'[4,7)':(0.0643,0.0336,0.0196),'[7,+)':(0.0776,0.0608,0.0284)}
print("\n== predicted BERTScore move from tapering anchoring to real, per bin ==\n")
print("%-9s %12s %10s %14s %10s"%("depth","anchor delta","gen slope","predicted","closes"))
for b,(ga,ra,bex) in EXC.items():
    if b not in res: continue
    sg=res[b][1]; mv=sg*(ra-ga)
    print("%-9s %+12.4f %10.4f %+14.5f %9.0f%%"%(b,ra-ga,sg,mv,100*(-mv)/bex))
print("\nUPPER BOUND (J7): assumes plan tapering moves realized text anchoring 1:1.")
