#!/usr/bin/env python3
"""Every numeric metric the suite records, generated vs its matched real."""
import csv, sys, math
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
SKIP={"thread_id","thread_title","_metric_thread_id","_run_id","_source_sim_dir","source_raw_post_id",
      "source_product_dir","source_file","matched_source_raw_post_id","matched_source_product",
      "matched_desired_product_dir","dominant_emotion","post_slot","seed_index","matched_seed_idx"}

def rows(p):
    with open(p) as f: return list(csv.DictReader(f))

def cliff(a,b):
    n=len(a)*len(b)
    if not n: return float("nan")
    g=sum(1 for x in a for y in b if x>y); l=sum(1 for x in a for y in b if x<y)
    return (g-l)/n

def table(tag):
    d=REPO/"artifacts/generalized_card/runs"/tag/"matched_evaluation"
    g=rows(d/"matched_generated_thread_scores.csv"); r=rows(d/"matched_real_thread_scores.csv")
    out={}
    for k in g[0]:
        if k in SKIP: continue
        try:
            ga=[float(x[k]) for x in g if x.get(k) not in (None,"","nan")]
            ra=[float(x[k]) for x in r if x.get(k) not in (None,"","nan")]
        except ValueError: continue
        if len(ga)<3 or len(ra)<3: continue
        if len(set(ga))==1 and len(set(ra))==1 and ga[0]==ra[0]: continue
        try:
            mw=mannwhitneyu(ga,ra,alternative="two-sided").pvalue
            ks=ks_2samp(ga,ra).pvalue
        except Exception: continue
        out[k]=(mw,ks,cliff(ga,ra),sum(ga)/len(ga),sum(ra)/len(ra))
    return out

A=table("v128_interaction_n10_20260828_v1")
B=table("v134_phraseledger_n10_20260828_v1")
keys=[k for k in A if k in B]
# order: failures first, then by |d|
keys.sort(key=lambda k: (min(A[k][0],A[k][1])>0.05 and min(B[k][0],B[k][1])>0.05, -abs(B[k][2])))
print(f"{'metric':34}{'real':>9}{'v128':>9}{'v134':>9} |{'v128 mwu':>9}{'ks':>7}{'d':>7} |{'v134 mwu':>9}{'ks':>7}{'d':>7}  verdict")
print("-"*126)
nf=0
for k in keys:
    a,b=A[k],B[k]
    okA=a[0]>0.05 and a[1]>0.05; okB=b[0]>0.05 and b[1]>0.05
    if not okB: nf+=1
    v=("PASS" if okB else "FAIL")+("" if okA==okB else ("  <- broke" if okA else "  <- fixed"))
    print(f"{k:34}{b[4]:>9.4f}{a[3]:>9.4f}{b[3]:>9.4f} |{a[0]:>9.3f}{a[1]:>7.3f}{a[2]:>+7.2f} |{b[0]:>9.3f}{b[1]:>7.3f}{b[2]:>+7.2f}  {v}")
print(f"\n{len(keys)-nf}/{len(keys)} metrics pass for v134 (MWU>0.05 AND KS>0.05, N=10)")
