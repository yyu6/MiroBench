"""Multi-objective selection on the cached matrices -- free, so try variants.

Single-objective selection (G100) fixed self_bertscore and cost story/tone.
The question this answers is whether ANY subset of the drafts already paid for
satisfies all nine movable metrics at once. That is the ceiling for every
first-pass mechanism that works by choosing among things the Writer can write.
"""
import json, math, statistics as st, collections, sys
from pathlib import Path
import numpy as np
D = json.load(open(Path(__file__).with_name("_cache") / "selection_ceiling_cache.json"))["threads"]
KEYS = ["bert","bleu4","cos","lcv","polite","impolite","neutral","story","emoH"]

def measure(th, idx):
    B=np.array(th["B"]); L=np.array(th["L4"]); Cm=np.array(th["C"])
    ix=np.array(idx); n=len(ix)
    sub=lambda M: (M[np.ix_(ix,ix)].sum()/2.0)/(n*(n-1)/2.0)
    f=[th["feat"][i] for i in idx]
    L_=[x["w"] for x in f]; a=sum(L_)/len(L_)
    lcv=math.sqrt(sum((v-a)**2 for v in L_)/len(L_))/a if a>0 else 0.0
    tn=[x["tone"] for x in f if x["tone"]]; c=collections.Counter(tn); nt=max(1,len(tn))
    em=[x["emo"] for x in f if x["emo"]]; ec=collections.Counter(em); ne=max(1,len(em))
    H=-sum((v/ne)*math.log(v/ne) for v in ec.values()) if ec else 0.0
    so=[x["story"] for x in f if x["story"] is not None]
    return {"bert":sub(B),"bleu4":sub(L),"cos":sub(Cm),"lcv":lcv,
            "polite":c["polite"]/nt,"impolite":c["impolite"]/nt,"neutral":c["neutral"]/nt,
            "story":st.mean(so) if so else float("nan"),"emoH":H}

SCALE={k: (st.pstdev([t["target"][k] for t in D]) or 1e-6) for k in KEYS}

def cost(m, tgt, weights):
    s=0.0
    for k in KEYS:
        w=weights.get(k,0.0)
        if not w or m[k]!=m[k]: continue
        s += w*((m[k]-tgt[k])/SCALE[k])**2
    return s

def optimize(th, weights, rounds=4):
    pos={}
    for i,k in enumerate(th["owner"]): pos.setdefault(k,[]).append(i)
    slots=th["slots"]
    cur=[pos[k][0] for k in slots]
    tgt=th["target"]
    best=cost(measure(th,cur),tgt,weights)
    for _ in range(rounds):
        moved=False
        for x,k in enumerate(slots):
            for c in pos[k]:
                if c==cur[x]: continue
                t=list(cur); t[x]=c
                v=cost(measure(th,t),tgt,weights)
                if v<best-1e-12: best=v; cur=t; moved=True
        if not moved: break
    return cur

VARIANTS={
 "base":            None,
 "selfbert only":   {"bert":1.0},
 "all 9 equal":     {k:1.0 for k in KEYS},
 "2x on bert+bleu": {**{k:1.0 for k in KEYS},"bert":2.0,"bleu4":2.0},
}
res={}
for name,w in VARIANTS.items():
    rows=[]
    for th in D:
        pos={}
        for i,k in enumerate(th["owner"]): pos.setdefault(k,[]).append(i)
        idx=[pos[k][0] for k in th["slots"]] if w is None else optimize(th,w)
        rows.append((measure(th,idx), th["target"], idx, th))
    res[name]=rows

print(f"{'metric':<10}"+"".join(f"{n:>18}" for n in VARIANTS))
print("-"*(10+18*len(VARIANTS)))
for k in KEYS:
    line=f"{k:<10}"
    for n in VARIANTS:
        v=[100*(m[k]-t[k])/t[k] for m,t,_,_ in res[n] if t.get(k) and m[k]==m[k] and t[k]]
        line+=f"{st.mean(v):>+17.2f}%"
    print(line)
print()
# arm attribution for the single-objective winner
for n in ("selfbert only","all 9 equal","2x on bert+bleu"):
    cnt=collections.Counter()
    for m,t,idx,th in res[n]:
        for i in idx: cnt[th["arm"][i]]+=1
    tot=sum(cnt.values())
    print(f"{n:<18} arm share: "+"  ".join(f"{a}={cnt[a]/tot:.1%}" for a in sorted(cnt)))
json.dump({n:[{"m":m,"t":t,"idx":idx,"tid":th["tid"]} for m,t,idx,th in rows]
           for n,rows in res.items()}, open(Path(__file__).with_name("mo_opt.json"),"w"))

# ---- the bar itself: Cliff's delta against the matched real threads ---------
import functools
@functools.lru_cache(None)
def _cnt(n,m,u):
    if u<0: return 0
    if n==0 or m==0: return 1 if u==0 else 0
    return _cnt(n-1,m,u-m)+_cnt(n,m-1,u)
def n10_p(d,n):
    tot=sum(_cnt(n,n,u) for u in range(n*n+1))
    dist=[_cnt(n,n,u)/tot for u in range(n*n+1)]
    U=int(round((1-d)/2*n*n))
    U=max(0,min(n*n,U))
    return min(1.0,2*min(sum(dist[:U+1]),sum(dist[U:])))
def cliff(a,b):
    gt=sum(1 for x in a for y in b if x>y); lt=sum(1 for x in a for y in b if x<y)
    return (gt-lt)/(len(a)*len(b))
print("\nCliff's delta vs matched real  (bar: |d| <= 0.13)")
print(f"{'metric':<10}"+"".join(f"{n:>18}" for n in VARIANTS))
print("-"*(10+18*len(VARIANTS)))
nth=len(D)
for k in KEYS:
    line=f"{k:<10}"
    for n in VARIANTS:
        g=[m[k] for m,t,_,_ in res[n] if m[k]==m[k]]
        r=[t[k] for m,t,_,_ in res[n] if m[k]==m[k]]
        d=cliff(g,r)
        line+=f"{d:>+11.2f}{'  ok' if abs(d)<=0.13 else '    '}"
    print(line)
print(f"\n(n={nth} threads per side; implied N=10 two-sided p at |d|=0.13 is {n10_p(0.13,10):.2f})")
