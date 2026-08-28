"""Does the Writer actually realize the dimensions the Planner assigns?"""
import json, glob, re, statistics
from collections import defaultdict, Counter
ROOT="/Users/yaoningyu/Desktop/UIUC/GEO/artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned"
rows=[]
for p in sorted(glob.glob(ROOT+"/run_*_sampled_reddit/discussion.json")):
    d=json.load(open(p))
    for post in d.get("posts") or []:
        for r in post.get("generation_records") or []:
            c=r.get("comment")
            if isinstance(c,dict) and c.get("content"): rows.append(c)
print(f"comments with plan metadata: {len(rows)}\n")

def has(pat): 
    rx=re.compile(pat,re.I)
    return lambda t: bool(rx.search(t))

PROBES={
 "hedge (maybe/not sure/i think/probably/might)": has(r"\b(maybe|not sure|i think|probably|might|guess|dunno|idk|possibly|i'd say)\b"),
 "thanks":            has(r"\b(thanks|thank you|thx|appreciate)\b"),
 "question mark":     has(r"\?"),
 "first person past": has(r"\b(i|we)\s+(had|got|used|bought|went|tried|shot|owned|found)\b"),
 "laugh/joke marker": has(r"\b(lol|lmao|haha|ha|jk|/s|😂|🤣)\b|\!\?"),
 "negation/complaint":has(r"\b(annoying|frustrating|hate|awful|terrible|useless|garbage|sucks|disappointed)\b"),
 "agreement opener":  has(r"^\s*(yeah|yep|yes|agreed|exactly|this|same|true|right)\b"),
 "disagree marker":   has(r"\b(disagree|not really|nah|i'd push back|that's not|actually no|hard disagree)\b"),
}

def table(field, probe_name, expect_map):
    probe=PROBES[probe_name]
    g=defaultdict(list)
    for c in rows:
        v=str(c.get(field) or "")
        if v: g[v].append(probe(str(c["content"])))
    g={k:v for k,v in g.items() if len(v)>=10}
    if len(g)<2: return
    items=sorted(g.items(), key=lambda kv:-sum(kv[1])/len(kv[1]))
    print(f"--- {field}  ->  '{probe_name}'")
    for k,v in items:
        star=" <-- EXPECTED" if k in expect_map else ""
        print(f"      {k[:32]:<32} n={len(v):4d}   rate {100*sum(v)/len(v):5.1f}%{star}")
    top=items[0][0]; hit=set(expect_map)&{top}
    print()

table("stance","hedge (maybe/not sure/i think/probably/might)",{"uncertain"})
table("stance","disagree marker",{"disagree"})
table("stance","agreement opener",{"agree"})
table("affect_role","thanks",{"gratitude"})
table("affect_role","negation/complaint",{"annoyance","disapproval"})
table("affect_role","laugh/joke marker",{"amusement","excitement"})
table("payload_type","first person past",{"personal_story"})
table("comment_function","question mark",{"clarification_question","narrow_question"})
table("claim_family","question mark",{"clarification_question"})

# what role/voice taxonomies exist at all
for f in ("speaker_role","voice","payload_type","comment_function","affect_role"):
    c=Counter(str(x.get(f) or "") for x in rows)
    if len(c)>1:
        print(f"{f}: " + ", ".join(f"{k or '(empty)'}={v}" for k,v in c.most_common(12)))
