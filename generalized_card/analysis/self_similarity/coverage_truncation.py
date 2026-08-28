"""Among FULL-coverage historical evaluations, what is the best |d| ever achieved?"""
import csv, glob, os, statistics
def rows_of(p):
    try: return list(csv.DictReader(open(p)))
    except Exception: return []
def col(rows,k):
    o=[]
    for x in rows:
        v=x.get(k)
        if v not in (None,""):
            try: o.append(float(v))
            except ValueError: pass
    return o
def cliff(a,b):
    g=sum(1 for x in a for y in b if x>y); l=sum(1 for x in a for y in b if x<y)
    return (g-l)/(len(a)*len(b))
out=[];seen=set()
for pat in ["artifacts/*/matched_real_thread_scores.csv",
            "artifacts/*/matched_evaluation/matched_real_thread_scores.csv",
            "artifacts/generalized_card/runs/*/matched_evaluation/matched_real_thread_scores.csv"]:
    for rp in glob.glob(pat):
        d=os.path.dirname(rp); gp=os.path.join(d,"matched_generated_thread_scores.csv")
        if not os.path.exists(gp) or os.path.abspath(d) in seen: continue
        seen.add(os.path.abspath(d))
        rr,gg=rows_of(rp),rows_of(gp)
        rs,gs=col(rr,"self_bertscore_mean_f1"),col(gg,"self_bertscore_mean_f1")
        rb,gb=col(rr,"self_bleu_4"),col(gg,"self_bleu_4")
        rc,gc=col(rr,"comment_count"),col(gg,"comment_count")
        if len(rs)<5 or len(gs)<5 or not rc or not gc: continue
        cov=statistics.mean(gc)/statistics.mean(rc)
        out.append((cov,cliff(gs,rs),cliff(gb,rb) if rb and gb else float('nan'),len(gs),
                    d.replace("artifacts/","").replace("generalized_card/runs/","").replace("/matched_evaluation","")))
full=[o for o in out if o[0]>=0.90]
full.sort(key=lambda o:max(abs(o[1]),abs(o[2]) if o[2]==o[2] else 0))
print(f"evaluations at coverage >= 0.90: {len(full)}  (of {len(out)} total)")
print(f"{'cov':>5} {'d_sb':>6} {'d_b4':>6} {'n':>4}  run")
for o in full[:15]:
    print(f"{o[0]:5.2f} {o[1]:+6.2f} {o[2]:+6.2f} {o[3]:>4}  {o[4][:58]}")
print(f"\nbest |d_sb| at full coverage: {min(abs(o[1]) for o in full):+.3f}")
print(f"count at full coverage with |d_sb|<=0.13: {sum(abs(o[1])<=0.13 for o in full)}")
print(f"count at full coverage with |d_sb|<=0.30: {sum(abs(o[1])<=0.30 for o in full)}")
# and 0.8 coverage
mid=[o for o in out if 0.80<=o[0]<0.90]
if mid: print(f"coverage [0.80,0.90): n={len(mid)} best |d_sb| {min(abs(o[1]) for o in mid):+.3f}")
