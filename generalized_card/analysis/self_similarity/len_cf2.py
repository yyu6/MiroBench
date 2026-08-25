"""Rebuild of len_cf.py on ONE consistent comment set.

The previous version mixed two comment sets -- the scored set (which drops
comments under 2 words) for the baseline and the generation-record set for the
counterfactual -- so every MWU/KS in that table was contaminated.  Here the
scored set is the only set, and each scored comment is joined to its record by
comment_id to recover the assigned length.
"""
from __future__ import annotations
import json, math, sys, statistics as st
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_self_bleu import tokenize, sentence_bleu
from score_thread_semantic_uniformity import load_generated_comments
SP=(Path(__file__).resolve().parent / "_cache")
RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"

def floor_pair(la,lb):
    logs=sum(math.log(1.0/(max(0,la-k+1)+1.0)) for k in range(1,5))
    bp=1.0 if la>lb else math.exp(1.0-lb/max(1,la))
    return bp*math.exp(logs/4)
def sym_floor(a,b): return (floor_pair(a,b)+floor_pair(b,a))/2
def mean_pair(lens, fn):
    n=len(lens); s=0.0; c=0
    for i in range(n):
        for j in range(i+1,n): s+=fn(lens[i],lens[j]); c+=1
    return s/c if c else 0.0

assigned={}
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    disc=json.load(open(d/"discussion.json"))
    for post in disc["posts"]:
        for rec in post.get("generation_records") or []:
            cid=str((rec.get("comment") or {}).get("comment_id",""))
            if cid: assigned[cid]=int((rec.get("task") or {}).get("real_word_count") or 0)

threads={}
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(d)
    for tid,cs in cbt.items():
        rows=[]
        for c in cs:
            a=assigned.get(str(c.comment_id))
            if a is None: rows=None; break
            rows.append({"tok":len(tokenize(c.text)),"words":len(c.text.split()),"assigned":a})
        if rows and len(rows)>=2: threads[tid]=rows
print(f"threads with a complete comment->record join: {len(threads)} / 50")
miss=[t for t in threads if any(r["assigned"]<=0 for r in threads[t])]
print(f"threads containing a slot with no assigned length: {len(miss)}")

gen={g["thread"]:g for g in json.load(open(SP/"gen_sb4.json"))}
real={r["seed"]:r for r in json.load(open(SP/"real_sb4.json"))}
tids=sorted(threads)
# exact per-thread actual and floor on the SAME set
act,flo={}, {}
for tid in tids:
    toks_len=[r["tok"] for r in threads[tid]]
    flo[tid]=mean_pair(toks_len, sym_floor)
    act[tid]=gen[tid]["sb4"]
print(f"[consistency] shipped self_bleu_4 mean {st.mean([act[t] for t in tids]):.5f}, "
      f"floor on the same set {st.mean([flo[t] for t in tids]):.5f}, "
      f"excess {st.mean([act[t]-flo[t] for t in tids]):.5f}")

R=[real[int(t.split('seed')[-1])]["sb4"] for t in tids]
def report(label, factor):
    vals=[]
    for tid in tids:
        lens=[max(1,int(round(r["tok"]*factor(r)))) for r in threads[tid]]
        nf=mean_pair(lens, sym_floor)
        vals.append(act[tid]-flo[tid]+nf)
    print(f"{label:<50}{st.mean(vals):.5f}{100*(st.mean(vals)-st.mean(R))/st.mean(R):>+8.1f}%"
          f"{mannwhitneyu(R,vals,alternative='two-sided').pvalue:>9.4f}{ks_2samp(R,vals).pvalue:>9.4f}")

print(f"\n{'scenario':<50}{'self_bleu_4':>11}{'bias':>8}{'MWU':>9}{'KS':>9}")
report("today", lambda r: 1.0)
def ratio(r): return (r["assigned"]/r["words"]) if (r["words"] and r["assigned"]>0) else 1.0
report("all bands -> realized = assigned", ratio)
report("assigned 35-100 -> 1.00 (v111 ideal)", lambda r: ratio(r) if 35<=r["assigned"]<=100 else 1.0)
report("assigned 35-100 -> 0.910 (v111 realistic)",
       lambda r: (0.910/ (r["words"]/r["assigned"])) if (35<=r["assigned"]<=100 and r["words"] and r["assigned"]) else 1.0)
report("assigned >150 -> 1.00", lambda r: ratio(r) if r["assigned"]>150 else 1.0)
report("assigned >=35 -> 1.00 (35-100 and up)", lambda r: ratio(r) if r["assigned"]>=35 else 1.0)
report("assigned <10 -> 1.00 only (stop over-writing)", lambda r: ratio(r) if r["assigned"]<10 else 1.0)
report("generated given real's per-thread token lengths",
       lambda r: 1.0)  # placeholder, replaced below
