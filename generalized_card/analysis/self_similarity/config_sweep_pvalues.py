#!/usr/bin/env python3
import csv, sys, statistics
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
M=["self_bertscore_mean_f1","self_bleu_4","semantic_mean_cosine","hard_disagree_rate",
   "polite_rate","impolite_rate","neutral_rate","length_cv","avg_depth",
   "structural_virality","mean_story_probability","emotion_entropy"]
def cliff(a,b):
    n=len(a)*len(b)
    return (sum(1 for x in a for y in b if x>y)-sum(1 for x in a for y in b if x<y))/n if n else float("nan")
def get(tag):
    d=REPO/"artifacts/generalized_card/runs"/tag/"matched_evaluation"
    g=list(csv.DictReader(open(d/"matched_generated_thread_scores.csv")))
    r=list(csv.DictReader(open(d/"matched_real_thread_scores.csv")))
    o={}
    for k in M:
        ga=[float(x[k]) for x in g if x.get(k) not in (None,"","nan")]
        ra=[float(x[k]) for x in r if x.get(k) not in (None,"","nan")]
        if len(ga)<3: continue
        o[k]=(min(mannwhitneyu(ga,ra,alternative="two-sided").pvalue, ks_2samp(ga,ra).pvalue), cliff(ga,ra))
    return o
CAND=[("v143_shaping3","v143_shaping3_n10_20260829_v1"),("v145_semcov_rhythm","v145_semcov_rhythm_n10_20260829_v1"),
      ("v138_semcov","v138_semcov_n10_20260829_v1"),("v144_v117shape","v144_v117shape_n10_20260829_v1"),
      ("v139_rhythm","v139_rhythm_n10_20260829_v1"),("v140_devscope","v140_devscope_n10_20260829_v1"),
      ("v141_noninteract","v141_noninteract_n10_20260829_v1"),("v142_tonecal","v142_tonecal_n10_20260829_v1"),
      ("v137_v117+tone","v137_v117cfg_tonefix_n10_20260829_v1"),
      ("v128 (baseline)","v128_interaction_n10_20260828_v1"),("v117 (old best)","v117_calibration_20260826_v1")]
data={}
for lbl,t in CAND:
    try: data[lbl]=get(t)
    except Exception: pass
rank=sorted(data.items(), key=lambda kv:(-sum(1 for v in kv[1].values() if v[0]>0.4), -min(v[0] for v in kv[1].values())))
print(f"{'run':22}{'p>0.4':>7}{'selfbert':>11}{'selfbleu4':>12}{'min p':>8}{'median':>9}")
print("-"*70)
for lbl,o in rank:
    g4=sum(1 for v in o.values() if v[0]>0.4)
    print(f"{lbl:22}{g4:>5}/12{o['self_bertscore_mean_f1'][0]:>11.3f}{o['self_bleu_4'][0]:>12.3f}"
          f"{min(v[0] for v in o.values()):>8.3f}{statistics.median(v[0] for v in o.values()):>9.3f}")
print("\n\n=== FULL 12-METRIC p-VALUES, top 3 + baselines ===\n")
show=[lbl for lbl,_ in rank[:3]]+["v128 (baseline)","v117 (old best)"]
show=[s for i,s in enumerate(show) if s in data and s not in show[:i]]
hdr=f"{'metric':26}"+"".join(f"{s[:14]:>16}" for s in show)
print(hdr); print("-"*len(hdr))
for k in M:
    line=f"{k:26}"
    for s in show:
        p,d=data[s][k]; line+=f"{p:>9.3f} {d:+.2f}{'' if p>0.4 else ('~' if p>0.05 else '*')}"
    print(line)
print("\n  * = fail (p<=0.05)   ~ = passes but thin (0.05<p<=0.4)")
