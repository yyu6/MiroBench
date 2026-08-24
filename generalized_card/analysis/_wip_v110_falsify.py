"""v110 falsification on the EXACT scorer (self_bleu is free to compute).
Do real threads whose comments vary more in clause register have lower
self_bleu_4? If the within-real slope is ~0, v110 dies here for $0 -- the same
test that just killed v109."""
import sys
import re
import statistics
from pathlib import Path
REPO=Path('/Users/yaoningyu/Desktop/UIUC/GEO')
sys.path.insert(0,str(REPO/'scripts/evaluation'))
sys.path.insert(0,str(REPO/'generalized_card/analysis'))
from root_reply_diversity import load_excluded_threads
from score_thread_self_bleu import tokenize, pairwise_self_bleu_for_order

SENT=re.compile(r'(?<=[.!?])\s+|\n+')
# only the three probes the spot-check verified as clean
FORMS={
 'conditional': lambda s: bool(re.match(r'^\s*(if|unless|when|once|whenever)\b',s.lower())),
 'concessive':  lambda s: bool(re.match(r'^\s*(but|though|although|still|that said|then again|admittedly)\b',s.lower())),
 'question':    lambda s: s.rstrip().endswith('?'),
}
def formvec(t):
    sents=[s for s in SENT.split(t) if s.strip()]
    return tuple(1 if any(fn(s) for s in sents) else 0 for fn in FORMS.values())
def spread(texts):
    """mean pairwise Hamming distance over the 3-bit clause-form signature"""
    vs=[formvec(t) for t in texts]
    ds=[]
    for i in range(len(vs)):
        for j in range(i+1,len(vs)):
            ds.append(sum(a!=b for a,b in zip(vs[i],vs[j]))/len(FORMS))
    return statistics.fmean(ds) if ds else float('nan')

th=load_excluded_threads()
rows=[]
for tid,cs in sorted(th.items()):
    texts=[c.text for c in cs if c.text.strip()]
    toks=[t for t in (tokenize(x) for x in texts) if t]
    if not (8<=len(toks)<=80): continue
    rows.append(dict(bleu=pairwise_self_bleu_for_order(toks,4),
                     spread=spread(texts),
                     meanw=statistics.fmean(len(t) for t in toks)))
    if len(rows)%50==0: print(f"  {len(rows)} threads",flush=True)
print(f"real threads: {len(rows)}\n")

def pear(a,b):
    ma,mb=statistics.fmean(a),statistics.fmean(b)
    n=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    d=(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))**.5
    return n/d if d else float('nan')
def sl(a,b):
    ma=statistics.fmean(a); d=sum((x-ma)**2 for x in a)
    mb=statistics.fmean(b)
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/d
def resid(y,x):
    mx,my=statistics.fmean(x),statistics.fmean(y); s=sl(x,y)
    return [c-(my+s*(a-mx)) for a,c in zip(x,y)]

B=[r['bleu'] for r in rows]; S=[r['spread'] for r in rows]; M=[r['meanw'] for r in rows]
print("On REAL excluded threads:")
print(f"  r(clause-form spread, self_bleu_4)              = {pear(S,B):+.3f}")
print(f"  partial r controlling mean words                = {pear(resid(S,M),resid(B,M)):+.3f}")
print(f"  slope                                          = {sl(S,B):+.5f} per unit spread")
q=sorted(rows,key=lambda r:r['spread']); k=len(q)//4
print("\n  quartiles of clause-form spread:")
for i in range(4):
    g=q[i*k:(i+1)*k] if i<3 else q[3*k:]
    print(f"    Q{i+1} spread={statistics.fmean(r['spread'] for r in g):.4f}  self_bleu_4={statistics.fmean(r['bleu'] for r in g):.5f}  meanw={statistics.fmean(r['meanw'] for r in g):5.1f}")
