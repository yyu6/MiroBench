"""Per slot: does a register marker in the PLAN predict the same marker in the OUTPUT?"""
import json, glob, re, statistics
from collections import defaultdict
PLANF=("semantic_move","local_topic","detail_focus","domain_intent","decision_boundary",
       "reply_delta","reply_novelty_anchor","development_plan","branch_goal","comment_job")
MARK=["whether","the key","the real","the only","the whole","the actual","matters",
      "rather than","tradeoff","actually","depends on","the point"]
rows=[]
for p in sorted(glob.glob("/Users/yaoningyu/Desktop/UIUC/GEO/artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned/run_*_sampled_reddit/discussion.json")):
    d=json.load(open(p))
    for post in d.get("posts") or []:
        for r in post.get("generation_records") or []:
            c=r.get("comment")
            if not isinstance(c,dict): continue
            text=str(c.get("content") or "")
            plan=" ".join(str(c.get(f) or "") for f in PLANF)
            if not text: continue
            rows.append((plan.lower(), text.lower()))
print(f"slots with plan+output: {len(rows)}\n")
print(f"{'marker':<12} {'in plan':>8} {'P(out|plan)':>12} {'P(out|no plan)':>15} {'lift':>7} {'ratio':>7}")
for mk in MARK:
    A=[t for pl,t in rows if mk in pl]; B=[t for pl,t in rows if mk not in pl]
    if len(A)<15 or len(B)<15:
        print(f"{mk:<12} {len(A):>8}  (too few)"); continue
    pa=sum(mk in t for t in A)/len(A); pb=sum(mk in t for t in B)/len(B)
    print(f"{mk:<12} {len(A):>8} {pa:12.3f} {pb:15.3f} {pa-pb:+7.3f} {(pa/pb if pb else float('inf')):7.1f}")
