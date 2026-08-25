"""Check 2: why did 'fix the >150 slots' make the floor go UP?"""
import json, math, sys, statistics as st
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_self_bleu import tokenize
RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"

def floor_pair(la,lb):
    logs=sum(math.log(1.0/(max(0,la-k+1)+1.0)) for k in range(1,5))
    bp=1.0 if la>lb else math.exp(1.0-lb/max(1,la))
    return bp*math.exp(logs/4)
def sym(a,b): return (floor_pair(a,b)+floor_pair(b,a))/2

# sanity: does lengthening one side of a pair always lower the pair floor?
print("pair floor as one side grows (other side fixed at 50 tokens):")
for L in (20,50,100,150,200,300):
    print(f"   len {L:>4}  sym floor vs 50-token comment = {sym(L,50):.5f}")
print("\npair floor as BOTH sides grow together:")
for L in (10,20,50,100,200):
    print(f"   both {L:>4}  = {sym(L,L):.5f}")

threads={}
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    disc=json.load(open(d/"discussion.json"))
    for post in disc["posts"]:
        rows=[]
        for rec in post.get("generation_records") or []:
            t=rec.get("task") or {}; c=rec.get("comment") or {}
            txt=str(c.get("content") or "")
            if txt.strip():
                rows.append({"tok":len(tokenize(txt)),"words":len(txt.split()),
                             "assigned":int(t.get("real_word_count") or 0)})
        if len(rows)>=2: threads[post["post_id"]]=rows

def thread_floor(lens):
    n=len(lens); s=0.0; c=0
    for i in range(n):
        for j in range(i+1,n): s+=sym(lens[i],lens[j]); c+=1
    return s/c

def run(label, factor):
    tot=[]; changed=0; up=0; down=0
    for tid,rows in threads.items():
        old=[r["tok"] for r in rows]
        new=[max(1,int(round(r["tok"]*factor(r)))) for r in rows]
        if new!=old: changed+=1
        fo,fn=thread_floor(old),thread_floor(new)
        if fn>fo: up+=1
        elif fn<fo: down+=1
        tot.append((fo,fn))
    print(f"{label:<44} floor {st.mean([x[0] for x in tot]):.5f} -> {st.mean([x[1] for x in tot]):.5f}"
          f"   threads changed {changed}, floor up in {up}, down in {down}")

run("only assigned>150 lengthened to assigned", lambda r: (r["assigned"]/r["words"]) if (r["words"] and r["assigned"]>150) else 1.0)
run("only assigned 35-100 lengthened to assigned", lambda r: (r["assigned"]/r["words"]) if (r["words"] and 35<=r["assigned"]<=100) else 1.0)
run("only assigned<10 shortened to assigned", lambda r: (r["assigned"]/r["words"]) if (r["words"] and r["assigned"]<10) else 1.0)
