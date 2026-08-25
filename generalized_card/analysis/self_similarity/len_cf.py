"""Exact self_bleu_4 counterfactual under length-fidelity repair.

The floor term is a closed-form function of token counts only, so a length
counterfactual can be evaluated exactly.  The excess term is held fixed, which
is the conservative direction to state (J7: an ablation is an upper bound).
"""
from __future__ import annotations
import json, math, statistics as st, sys
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_self_bleu import tokenize, sentence_bleu, ngram_counts, closest_reference_length
RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
SP=(Path(__file__).resolve().parent / "_cache")

def floor_from_lengths(la, lb):
    logs=0.0
    for k in range(1,5):
        logs+=math.log(1.0/(max(0,la-k+1)+1.0))
    crl=lb
    bp=1.0 if la>crl else math.exp(1.0-crl/max(1,la))
    return bp*math.exp(logs/4)

def thread_floor(lengths):
    n=len(lengths)
    if n<2: return 0.0
    s=0.0; c=0
    for i in range(n):
        for j in range(i+1,n):
            a,b=lengths[i],lengths[j]
            s+=(floor_from_lengths(a,b)+floor_from_lengths(b,a))/2.0; c+=1
    return s/c

# per-comment: token count + assigned words, grouped by thread
threads={}
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    disc=json.load(open(d/"discussion.json"))
    for post in disc["posts"]:
        tid=post["post_id"]; rows=[]
        for rec in post.get("generation_records") or []:
            t=rec.get("task") or {}; c=rec.get("comment") or {}
            txt=str(c.get("content") or "")
            if not txt.strip(): continue
            rows.append({"tok":len(tokenize(txt)),"words":len(txt.split()),
                         "assigned":int(t.get("real_word_count") or 0)})
        if len(rows)>=2: threads[tid]=rows

gen=json.load(open(SP/"gen_sb4.json")); realj=json.load(open(SP/"real_sb4.json"))
gen={g["thread"]:g for g in gen}; realj={r["seed"]:r for r in realj}
print(f"threads with records: {len(threads)}   scored threads: {len(gen)}")

def scenario(name, factor):
    out=[]
    for tid, rows in threads.items():
        if tid not in gen: continue
        lens=[]
        for r in rows:
            f=factor(r)
            lens.append(max(1,int(round(r["tok"]*f))))
        nf=thread_floor(lens)
        out.append((tid, gen[tid]["excess"]+nf, gen[tid]["sb4"], nf, gen[tid]["floor"]))
    seeds=[int(t.split("seed")[-1]) for t,_,_,_,_ in out]
    r=[realj[s]["sb4"] for s in seeds]
    b=[x[1] for x in out]
    base=[x[2] for x in out]
    print(f"\n{name}")
    print(f"   mean floor {st.mean([x[4] for x in out]):.5f} -> {st.mean([x[3] for x in out]):.5f}")
    print(f"   self_bleu_4 {st.mean(base):.5f} -> {st.mean(b):.5f}   (real {st.mean(r):.5f}, bias {100*(st.mean(b)-st.mean(r))/st.mean(r):+.1f}%)")
    print(f"   MWU {mannwhitneyu(r,base,alternative='two-sided').pvalue:.4f} -> {mannwhitneyu(r,b,alternative='two-sided').pvalue:.4f}"
          f"   KS {ks_2samp(r,base).pvalue:.4f} -> {ks_2samp(r,b).pvalue:.4f}")

scenario("A. perfect length fidelity everywhere (realized = assigned)",
         lambda r: (r["assigned"]/r["words"]) if r["words"] and r["assigned"] else 1.0)
scenario("B. v111 scope only: assigned 35-100 reaches realized = assigned",
         lambda r: (r["assigned"]/r["words"]) if (r["words"] and 35<=r["assigned"]<=100) else 1.0)
scenario("C. long slots only: assigned > 150 reaches realized = assigned",
         lambda r: (r["assigned"]/r["words"]) if (r["words"] and r["assigned"]>150) else 1.0)
scenario("D. every band at or above 35 reaches realized = assigned",
         lambda r: (r["assigned"]/r["words"]) if (r["words"] and r["assigned"]>=35) else 1.0)
