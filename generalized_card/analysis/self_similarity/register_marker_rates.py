"""Real's rate of the registers we schedule at ~0."""
from __future__ import annotations
import json, sys, re, statistics
from pathlib import Path
REPO=Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0,str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
pool=json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed={int(p["seed_index"]):p for p in pool}
P={
 "laugh/joke (lol|haha|lmao|jk|/s)": re.compile(r"\b(lol|lmao|lmfao|haha+|hehe|jk|/s)\b|😂|🤣|😅",re.I),
 "swear/intensifier":  re.compile(r"\b(damn|shit|fuck\w*|hell|crap|wtf|af|hella)\b",re.I),
 "internet shorthand": re.compile(r"\b(imo|imho|iirc|tbh|ngl|fwiw|afaik|ymmv|btw|edit:|tl;?dr)\b",re.I),
 "pure link only":     re.compile(r"^\s*(https?://\S+\s*)+$"),
 "all-caps word":      re.compile(r"\b[A-Z]{3,}\b"),
 "ellipsis/trailing":  re.compile(r"\.\.\."),
 "exclamation":        re.compile(r"!"),
 "quote reply (>)":    re.compile(r"^\s*&?gt;|^\s*>",re.M),
 "thanks":             re.compile(r"\b(thanks|thank you|thx|appreciate)\b",re.I),
 "one-liner <=6 words":None,
}
def rate(texts,k,rx):
    n=len(texts)
    if k=="one-liner <=6 words": return sum(len(t.split())<=6 for t in texts)/n
    return sum(bool(rx.search(t)) for t in texts)/n

RUN=REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1/cleaned"
cache={}; G=[]; R=[]
for x in sorted(RUN.glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(x)
    for tid,cs in cbt.items():
        seed=int(tid.split("seed")[-1]); p=by_seed.get(seed)
        if not p: continue
        dd=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
        if dd not in cache:
            try: cache[dd]=load_real_comments(dd)[0]
            except Exception: cache[dd]={}
        rcs=cache[dd].get(p["source_raw_post_id"]) or []
        if len(rcs)<10 or len(cs)<10: continue
        G+=[c.text for c in cs]; R+=[c.text for c in rcs]
print(f"gen {len(G)}  real {len(R)}\n")
print(f"{'register marker':<36} {'real':>8} {'gen':>8} {'gen/real':>9}")
for k,rx in P.items():
    r=rate(R,k,rx); g=rate(G,k,rx)
    print(f"{k:<36} {100*r:7.2f}% {100*g:7.2f}% {g/r if r else float('nan'):9.2f}")
