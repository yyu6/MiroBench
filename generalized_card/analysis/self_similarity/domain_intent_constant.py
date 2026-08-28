"""Redo the domain_intent check using the DETECTOR'S OWN tokenizer, not mine."""
import json, sys, itertools, statistics
sys.path.insert(0,"/Users/yaoningyu/Desktop/UIUC/GEO/generalized_card")
from generalized_card.planning_quality import semantic_tokens, plan_similarity, SEMANTIC_FIELDS, STOPWORDS, _stem, TOKEN_RE
from generalized_card.semantic_realization import NON_SEMANTIC_DEFAULTS
from collections import defaultdict

CONST="one seed-grounded local move"
print("tokens the detector actually keeps from the constant:",
      sorted({_stem(t) for t in TOKEN_RE.findall(CONST.lower()) if len(_stem(t))>=3 and _stem(t) not in STOPWORDS}))
print("NON_SEMANTIC_DEFAULTS:", sorted(NON_SEMANTIC_DEFAULTS))
print()
P="/Users/yaoningyu/Desktop/UIUC/GEO/artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/logs/planning_quality.jsonl"
threads=defaultdict(list)
for line in open(P):
    if not line.strip(): continue
    x=json.loads(line); ip=x.get("initial_plans")
    if isinstance(ip,dict):
        for _s,p in ip.items():
            if isinstance(p,dict): threads[str(x.get("seed_key"))].append(p)
n=sum(len(v) for v in threads.values())
hits=sum(1 for v in threads.values() for p in v if str(p.get("domain_intent") or "").casefold() in NON_SEMANTIC_DEFAULTS)
print(f"plans {n}; domain_intent is a NON_SEMANTIC_DEFAULT in {hits} ({100*hits/n:.1f}%)")
# per-field: how many plans carry a placeholder in ANY semantic field
byfield={}
for f in SEMANTIC_FIELDS:
    byfield[f]=sum(1 for v in threads.values() for p in v if str(p.get(f) or "").casefold() in NON_SEMANTIC_DEFAULTS)
print("placeholder count per semantic field:", {k:v for k,v in byfield.items() if v})
print()
def blank(p, fields):
    q=dict(p)
    for f in fields:
        if str(q.get(f) or "").casefold() in NON_SEMANTIC_DEFAULTS: q[f]=""
    return q
for label, prep in (("AS SHIPPED", lambda p: p), ("placeholders blanked", lambda p: blank(p, SEMANTIC_FIELDS))):
    lex=[]; full=[]
    for sk,ps in threads.items():
        if len(ps)<10: continue
        Q=[prep(p) for p in ps]
        T=[semantic_tokens(q) for q in Q]
        for i,j in itertools.combinations(range(len(Q)),2):
            u=T[i]|T[j]
            lex.append(len(T[i]&T[j])/max(1,len(u)))
            full.append(plan_similarity(Q[i],Q[j]))
    print(f"{label:<24} lexical Jaccard {statistics.mean(lex):.5f}   plan_similarity {statistics.mean(full):.5f}   (n={len(lex)})")
