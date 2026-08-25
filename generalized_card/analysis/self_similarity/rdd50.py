"""Does the beat-plan discontinuity at real_word_count=100 replicate at N=50?

expected_development_beats returns 0 at w<=100 and max(3, round(w/21)) above it,
so the Planner's beat plan is deleted at 100 and kept at 101.  If the beat plan
is the causal instrument for realized length, realized/assigned must jump there.
"""
from __future__ import annotations
import json, re, statistics as st
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
rows=[]
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    disc=json.load(open(d/"discussion.json"))
    for post in disc["posts"]:
        for rec in post.get("generation_records") or []:
            t=rec.get("task") or {}; c=rec.get("comment") or {}
            w=int(t.get("real_word_count") or 0); txt=str(c.get("content") or "")
            if w>0 and txt.strip():
                rows.append({"w":w,"r":len(txt.split()),"prompt":str(rec.get("prompt") or ""),
                             "plan":t.get("development_plan")})
print(f"slots: {len(rows)}   total realized/assigned: {sum(x['r'] for x in rows)/sum(x['w'] for x in rows):.4f}")

def band(lo,hi):
    s=[x for x in rows if lo<=x["w"]<=hi]
    return s, (sum(x["r"] for x in s)/sum(x["w"] for x in s) if s else float('nan'))

print("\n--- RDD at w=100 (the beat-plan threshold) ---")
for bw in (10,15,20,25):
    L,rl=band(100-bw+1,100); R,rr=band(101,100+bw)
    lr=[x["r"]/x["w"] for x in L]; rrr=[x["r"]/x["w"] for x in R]
    p=mannwhitneyu(lr,rrr,alternative="two-sided").pvalue if lr and rrr else float('nan')
    print(f"  bw={bw:>2}  left n={len(L):>3} ratio {rl:.3f}   right n={len(R):>3} ratio {rr:.3f}   jump {rr-rl:+.3f}   MWU {p:.3f}")

print("\n--- placebo cut points (no rule changes at these) ---")
for cut in (60,80,120,140):
    L,rl=band(cut-14,cut); R,rr=band(cut+1,cut+15)
    print(f"  w={cut:>3}  left {rl:.3f}  right {rr:.3f}  jump {rr-rl:+.3f}   (n {len(L)}/{len(R)})")

print("\n--- does the beat plan actually reach the prompt? (E5: grep the saved prompts) ---")
CUES=[("development beat text","beat"),("'connected beats' cue","connected beats"),
      ("'one narrow local move' cue","narrow local move")]
for lo,hi,lab in ((1,34,"w 1-34"),(35,60,"w 35-60"),(61,100,"w 61-100"),(101,150,"w 101-150"),(151,10**9,"w 151+")):
    s=[x for x in rows if lo<=x["w"]<=hi]
    if not s: continue
    line=f"  {lab:<10} n={len(s):>4}"
    for name,cue in CUES:
        line+=f"   {name}: {100*np.mean([1.0 if cue in x['prompt'].lower() else 0.0 for x in s]):5.1f}%"
    line+=f"   planner gave a development_plan: {100*np.mean([1.0 if x['plan'] else 0.0 for x in s]):5.1f}%"
    print(line)
