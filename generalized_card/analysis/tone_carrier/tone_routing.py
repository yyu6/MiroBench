"""Can Planner-side fields alone pick out the polite-assigned slots that read
impolite -- while avoiding the ones that read neutral?

Routing on the evaluation classifier's own label is forbidden (ORIENTATION s4).
This asks whether the slot's own planned fields carry the same signal.
"""
from __future__ import annotations
import json, statistics as st, sys
from collections import Counter, defaultdict
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments
from score_thread_politeness import PolitenessScorer
RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
scorer=PolitenessScorer("Intel/polite-guard","auto",256)
class T:
    def __init__(self,t):
        self.text=t; self.thread_id=""; self.thread_title=""; self.comment_id=""
        self.parent_id=""; self.author=""; self.depth=0

task_by_cid={}
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    disc=json.load(open(d/"discussion.json"))
    for post in disc["posts"]:
        for rec in post.get("generation_records") or []:
            cid=str((rec.get("comment") or {}).get("comment_id",""))
            if cid: task_by_cid[cid]=rec.get("task") or {}

rows=[]
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(d)
    for tid,cs in cbt.items():
        got=scorer.score_comments([T(c.text) for c in cs],batch_size=32,include_text=False)
        for c,g in zip(cs,got):
            t=task_by_cid.get(str(c.comment_id))
            if t: rows.append((t,g["pred_label"]))
print(f"joined slots: {len(rows)}")
pol=[(t,l) for t,l in rows if t.get("tone_target")=="polite"]
print(f"assigned tone_target=polite: {len(pol)}")
print("  realized:", dict(Counter(l for _,l in pol).most_common()))

FIELDS=["stance","comment_function","payload_type","speaker_role","evidence_mode",
        "utterance_mode","story_mode","opening_style","tone_shape","affect_role",
        "real_tone_slot","surface_texture","voice"]
base_imp=sum(1 for _,l in pol if l=="impolite")/len(pol)
base_neu=sum(1 for _,l in pol if l=="neutral")/len(pol)
print(f"\nbase within polite-assigned: impolite {base_imp:.3f}   neutral {base_neu:.3f}")
print(f"\n{'field = value':<52}{'n':>5}{'impolite':>10}{'neutral':>9}{'lift':>7}")
cands=[]
for f in FIELDS:
    by=defaultdict(list)
    for t,l in pol: by[str(t.get(f))].append(l)
    for v,ls in sorted(by.items(), key=lambda kv:-len(kv[1])):
        if len(ls)<25: continue
        imp=sum(1 for l in ls if l=="impolite")/len(ls)
        neu=sum(1 for l in ls if l=="neutral")/len(ls)
        cands.append((imp,neu,len(ls),f,v))
for imp,neu,n,f,v in sorted(cands, reverse=True)[:16]:
    print(f"{f+' = '+v[:34]:<52}{n:>5}{imp:>10.3f}{neu:>9.3f}{imp/base_imp:>7.2f}")
