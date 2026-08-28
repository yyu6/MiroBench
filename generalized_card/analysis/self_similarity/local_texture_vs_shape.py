#!/usr/bin/env python3
"""Is it the WORDS or the SENTENCE SHAPES?

Strip every content word away and keep only the grammatical skeleton, then ask
the same pairwise-overlap question. If our word overlap is 1.2x real but our
SHAPE overlap is 2x real, the defect is construction, not vocabulary -- and
that is why six vocabulary interventions priced at zero.
"""
import json, sys, re, statistics
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
TOK = re.compile(r"[a-z0-9']+")

# closed-class words = the skeleton. Everything else becomes a single slot 'X'.
FUNC = set("""a an the and or but if so then than that this these those there here
is are was were be been being am s re ve d ll t of to in on at for with from by about
into over after before as like just only very really pretty quite kind sort bit lot much
many more most less least i you he she it we they me him her us them my your his its our
their what which who whom whose when where why how all any both each few other some such
no nor not too own same can could would should may might must will shall do does did have
has had get got go goes went one two up out off down again still even also because while
since though although unless until whether where after""".split())

def skeleton(text):
    """Keep function words verbatim, replace every content word with X."""
    return [w if w in FUNC else "X" for w in TOK.findall(text.lower())]
def words(text):
    return TOK.findall(text.lower())

def overlap(seqs, n):
    sets=[set(zip(*[s[i:] for i in range(n)])) for s in seqs]
    v=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            u=sets[i]|sets[j]
            if u: v.append(len(sets[i]&sets[j])/len(u))
    return statistics.mean(v) if v else float("nan")

pool=json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by={int(p["seed_index"]):p for p in pool}
G={}
for x in sorted((REPO/"artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned").glob("run_*_sampled_reddit")):
    cbt,_=load_generated_comments(x)
    for tid,cs in cbt.items(): G[int(tid.split("seed")[-1])]=[c.text for c in cs]
cache={}; rows=[]
for s in sorted(G):
    p=by[s]; d=REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache:
        try: cache[d]=load_real_comments(d)[0]
        except Exception: cache[d]={}
    r=cache[d].get(p["source_raw_post_id"]) or []
    if len(r)<12: continue
    rows.append(([c.text for c in r], G[s]))
print(f"{len(rows)} matched threads\n")
print(f"{'':38}{'real':>10}{'ours':>10}{'ratio':>8}")
print("-"*66)
for label, fn, n in (("WORDS  4-gram overlap", words, 4),
                     ("WORDS  3-gram overlap", words, 3),
                     ("WORDS  2-gram overlap", words, 2),
                     ("SHAPE  4-gram overlap", skeleton, 4),
                     ("SHAPE  6-gram overlap", skeleton, 6),
                     ("SHAPE  8-gram overlap", skeleton, 8)):
    a=statistics.mean(overlap([fn(t) for t in r], n) for r,_ in rows)
    b=statistics.mean(overlap([fn(t) for t in o], n) for _,o in rows)
    print(f"{label:38}{a:>10.5f}{b:>10.5f}{b/a:>8.2f}")
