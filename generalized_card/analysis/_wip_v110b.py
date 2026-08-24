"""Same falsification with the better-powered measure: the 41-dim function-word
L1 spread from TEST B (G32), which is what G27 says actually carries the gap."""
import sys
import statistics
from pathlib import Path
from collections import Counter
REPO=Path('/Users/yaoningyu/Desktop/UIUC/GEO')
sys.path.insert(0,str(REPO/'scripts/evaluation'))
sys.path.insert(0,str(REPO/'generalized_card/analysis'))
from root_reply_diversity import load_excluded_threads
from score_thread_self_bleu import tokenize, pairwise_self_bleu_for_order
FUNC=['the',',','.','i','you','to','a','and','it','that','is','of','my','but','for','?','!','not','have','would','will','so','just','if','with','they','we','this','do','get','be','are','was','on','at','or','no','yes','me','your','what']
def fvec(toks):
    n=len(toks) or 1; c=Counter(toks)
    return [c[f]/n for f in FUNC]
def fspread(toks_list):
    vs=[fvec(t) for t in toks_list if t]
    ds=[]
    for i in range(len(vs)):
        for j in range(i+1,len(vs)):
            ds.append(sum(abs(a-b) for a,b in zip(vs[i],vs[j])))
    return statistics.fmean(ds) if ds else float('nan')
th=load_excluded_threads()
rows=[]
for tid,cs in sorted(th.items()):
    toks=[t for t in (tokenize(c.text) for c in cs) if t]
    if not (8<=len(toks)<=80): continue
    rows.append(dict(bleu=pairwise_self_bleu_for_order(toks,4), fs=fspread(toks),
                     meanw=statistics.fmean(len(t) for t in toks)))
print(f"real threads: {len(rows)}")
def pear(a,b):
    ma,mb=statistics.fmean(a),statistics.fmean(b)
    n=sum((x-ma)*(y-mb) for x,y in zip(a,b)); d=(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))**.5
    return n/d if d else float('nan')
def sl(a,b):
    ma=statistics.fmean(a); mb=statistics.fmean(b)
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/sum((x-ma)**2 for x in a)
def resid(y,x):
    mx,my=statistics.fmean(x),statistics.fmean(y); s=sl(x,y)
    return [c-(my+s*(a-mx)) for a,c in zip(x,y)]
B=[r['bleu'] for r in rows]; F=[r['fs'] for r in rows]; M=[r['meanw'] for r in rows]
print(f"\n  r(function-word L1 spread, self_bleu_4)   = {pear(F,B):+.3f}")
print(f"  partial r controlling mean words          = {pear(resid(F,M),resid(B,M)):+.3f}")
sp=sl(resid(F,M),resid(B,M))
print(f"  partial slope                             = {sp:+.5f} per unit L1 spread")
print("\n  generated L1 spread 0.4646 -> real 0.5342 is +0.0696")
print(f"  predicted self_bleu_4 move at the partial slope: {sp*0.0696:+.6f}")
print(f"  gap to close: +0.00489  ->  {100*(-sp*0.0696)/0.00489:.1f}% of it")
q=sorted(rows,key=lambda r:r['fs']); k=len(q)//4
print("\n  quartiles of function-word spread:")
for i in range(4):
    g=q[i*k:(i+1)*k] if i<3 else q[3*k:]
    print(f"    Q{i+1} spread={statistics.fmean(r['fs'] for r in g):.4f} self_bleu_4={statistics.fmean(r['bleu'] for r in g):.5f} meanw={statistics.fmean(r['meanw'] for r in g):5.1f}")
