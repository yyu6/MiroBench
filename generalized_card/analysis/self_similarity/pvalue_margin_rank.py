#!/usr/bin/env python3
"""The user's criterion: not the pass COUNT but the MARGIN.

v117 passed 12/12 at N=10 and 7/12 at N=50. Test the proposed explanation --
the metrics that died had thin p-values already -- then rank every run by how
many metrics clear a comfortable bar, which is what survives more data.
"""
import csv, glob, statistics
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
M12=["self_bertscore_mean_f1","self_bleu_4","semantic_mean_cosine","hard_disagree_rate",
     "polite_rate","impolite_rate","neutral_rate","length_cv","avg_depth",
     "structural_virality","mean_story_probability","emotion_entropy"]
def stats(tag):
    d=REPO/"artifacts/generalized_card/runs"/tag/"matched_evaluation"
    g=list(csv.DictReader(open(d/"matched_generated_thread_scores.csv")))
    r=list(csv.DictReader(open(d/"matched_real_thread_scores.csv")))
    o={}
    for k in M12:
        ga=[float(x[k]) for x in g if x.get(k) not in (None,"","nan")]
        ra=[float(x[k]) for x in r if x.get(k) not in (None,"","nan")]
        if len(ga)<3: continue
        o[k]=min(mannwhitneyu(ga,ra,alternative="two-sided").pvalue, ks_2samp(ga,ra).pvalue)
    return o, len(g)

a,_=stats("v117_calibration_20260826_v1")
b,_=stats("v117cfg_40more_20260829_v2")
print("v117: did the N=50 failures already have thin p-values at N=10?")
print(f"  {'metric':28}{'N=10 p':>9}{'N=40 p':>9}   verdict at N=40")
for k in M12:
    if k not in a or k not in b: continue
    print(f"  {k:28}{a[k]:>9.3f}{b[k]:>9.3f}   {'FAIL' if b[k]<=0.05 else 'pass'}")
died=[k for k in M12 if k in a and k in b and b[k]<=0.05]
liv=[k for k in M12 if k in a and k in b and b[k]>0.05]
print(f"\n  mean N=10 p of the ones that DIED : {statistics.mean(a[k] for k in died):.3f}  ({len(died)})")
print(f"  mean N=10 p of the ones that LIVED: {statistics.mean(a[k] for k in liv):.3f}  ({len(liv)})")

print("\n\nEvery full-coverage N=10 run, ranked by MARGIN not by pass count:")
rows=[]
for d in sorted(glob.glob(str(REPO/"artifacts/generalized_card/runs/*/matched_evaluation"))):
    tag=Path(d).parent.name
    try:
        o,n=stats(tag)
        g=list(csv.DictReader(open(Path(d)/"matched_generated_thread_scores.csv")))
        r=list(csv.DictReader(open(Path(d)/"matched_real_thread_scores.csv")))
        gc=statistics.mean(float(x["comment_count"]) for x in g)
        rc=statistics.mean(float(x["comment_count"]) for x in r)
    except Exception: continue
    if len(o)<12 or n<8 or gc/rc<0.9: continue
    rows.append((tag,n,sum(1 for v in o.values() if v>0.4),sum(1 for v in o.values() if v>0.2),
                 min(o.values()),statistics.median(o.values()),o))
rows.sort(key=lambda x:(-x[2],-x[3],-x[4]))
print(f"{'run':38}{'N':>3}{'p>0.4':>7}{'p>0.2':>7}{'min p':>8}{'median p':>10}")
print("-"*76)
for tag,n,g4,g2,mn,md,o in rows[:12]:
    print(f"{tag[:37]:38}{n:>3}{g4:>5}/12{g2:>5}/12{mn:>8.3f}{md:>10.3f}")
