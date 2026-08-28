"""Cliff's d for self_bertscore / self_bleu_4 across the ENTIRE artifact history."""
from __future__ import annotations
import csv, glob, os, math
from pathlib import Path

def col(rows, k):
    out=[]
    for x in rows:
        v=x.get(k)
        if v not in (None,""):
            try: out.append(float(v))
            except ValueError: pass
    return out

def cliff(a,b):
    if not a or not b: return float('nan')
    g=sum(1 for x in a for y in b if x>y); l=sum(1 for x in a for y in b if x<y)
    return (g-l)/(len(a)*len(b))

def mwu_p(a,b):
    """Normal-approx two-sided Mann-Whitney U p-value with tie correction skipped."""
    n1,n2=len(a),len(b)
    if n1<3 or n2<3: return float('nan')
    allv=sorted([(v,0) for v in a]+[(v,1) for v in b])
    ranks={}; i=0
    r=[0.0]*(n1+n2)
    while i<len(allv):
        j=i
        while j+1<len(allv) and allv[j+1][0]==allv[i][0]: j+=1
        avg=(i+j)/2+1
        for k in range(i,j+1): r[k]=avg
        i=j+1
    R1=sum(r[k] for k in range(len(allv)) if allv[k][1]==0)
    U1=R1-n1*(n1+1)/2
    mu=n1*n2/2; sd=math.sqrt(n1*n2*(n1+n2+1)/12)
    if sd==0: return float('nan')
    z=(U1-mu)/sd
    return 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))

rows=[]
seen=set()
pats = ["artifacts/*/matched_real_thread_scores.csv",
        "artifacts/*/matched_evaluation/matched_real_thread_scores.csv",
        "artifacts/generalized_card/runs/*/matched_evaluation/matched_real_thread_scores.csv"]
for pat in pats:
    for rp in glob.glob(pat):
        d=os.path.dirname(rp)
        gp=os.path.join(d,"matched_generated_thread_scores.csv")
        if not os.path.exists(gp): continue
        key=os.path.abspath(d)
        if key in seen: continue
        seen.add(key)
        try:
            rr=list(csv.DictReader(open(rp))); gg=list(csv.DictReader(open(gp)))
        except Exception: continue
        rsb,gsb=col(rr,"self_bertscore_mean_f1"),col(gg,"self_bertscore_mean_f1")
        rb4,gb4=col(rr,"self_bleu_4"),col(gg,"self_bleu_4")
        if len(rsb)<5 or len(gsb)<5: continue
        name=d.replace("artifacts/","").replace("generalized_card/runs/","")
        name=name.replace("/matched_evaluation","")
        rows.append({
            "name":name, "n":len(gsb),
            "d_sb":cliff(gsb,rsb), "p_sb":mwu_p(gsb,rsb),
            "d_b4":cliff(gb4,rb4) if rb4 and gb4 else float('nan'),
            "p_b4":mwu_p(gb4,rb4) if rb4 and gb4 else float('nan'),
            "sb":sum(gsb)/len(gsb), "rsb":sum(rsb)/len(rsb),
        })

print(f"evaluated runs with matched real: {len(rows)}\n")
ok=[r for r in rows if r["d_sb"]==r["d_sb"] and r["d_b4"]==r["d_b4"]]
ok.sort(key=lambda r: max(abs(r["d_sb"]),abs(r["d_b4"])))
print("=== best 30 by max(|d_selfbert|, |d_selfbleu4|) ===")
print(f"{'run':<62} {'n':>4} {'d_sb':>6} {'p_sb':>6} {'d_b4':>6} {'p_b4':>6}")
for r in ok[:30]:
    print(f"{r['name'][:62]:<62} {r['n']:>4} {r['d_sb']:+6.2f} {r['p_sb']:6.3f} {r['d_b4']:+6.2f} {r['p_b4']:6.3f}")
print(f"\n=== distribution of d_sb over {len(ok)} runs ===")
import statistics
ds=[r["d_sb"] for r in ok]
print(f"  min {min(ds):+.2f}  p10 {sorted(ds)[len(ds)//10]:+.2f}  median {statistics.median(ds):+.2f}  max {max(ds):+.2f}")
print(f"  runs with |d_sb|<=0.30: {sum(abs(x)<=0.30 for x in ds)}   <=0.13: {sum(abs(x)<=0.13 for x in ds)}")
db=[r["d_b4"] for r in ok]
print(f"  d_b4: min {min(db):+.2f} median {statistics.median(db):+.2f} max {max(db):+.2f}  |d|<=0.30: {sum(abs(x)<=0.30 for x in db)}")
