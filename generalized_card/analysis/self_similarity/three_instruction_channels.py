"""Three instruction channels, three compliance rates -- measured on the rendered prompts."""
import json, glob, re
from collections import Counter
ROOT="/Users/yaoningyu/Desktop/UIUC/GEO/artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned"
recs=[]
for p in sorted(glob.glob(ROOT+"/run_*_sampled_reddit/discussion.json")):
    d=json.load(open(p))
    for post in d.get("posts") or []:
        for r in post.get("generation_records") or []:
            c=r.get("comment"); pr=r.get("prompt")
            if isinstance(c,dict) and c.get("content"):
                recs.append((pr if isinstance(pr,str) else json.dumps(pr), str(c["content"]), c))
print(f"records with prompt + output: {len(recs)}\n")

# --- channel 3: explicit prohibitions rendered in the prompt ---
BANS=[(", honestly", r",\s*honestly"),
      ("that part",  r"\bthat part\b"),
      ("honestly",   r"\bhonestly\b")]
print("=== CHANNEL 3: a prohibition sentence in the prompt ===")
for label,rx in BANS:
    # find prompts that literally contain the banned string as a ban
    inprompt=sum(1 for pr,_o,_c in recs if label in pr.lower())
    viol=sum(1 for pr,o,_c in recs if label in pr.lower() and re.search(rx,o,re.I))
    tot_viol=sum(1 for _p,o,_c in recs if re.search(rx,o,re.I))
    print(f"  '{label}': named in {inprompt}/{len(recs)} prompts; "
          f"produced anyway in {viol} of those ({100*viol/max(1,inprompt):.1f}%); {tot_viol} overall")

# --- channel 2: the Planner's own wording ---
print("\n=== CHANNEL 2: the Planner's free-text WORDING ===")
PLANF=("semantic_move","local_topic","detail_focus","domain_intent","decision_boundary",
       "reply_delta","reply_novelty_anchor","development_plan","branch_goal","comment_job")
for w in ["whether","rather than","the key question"]:
    A=[(o) for _p,o,c in recs if w in " ".join(str(c.get(f) or "") for f in PLANF).lower()]
    B=[(o) for _p,o,c in recs if w not in " ".join(str(c.get(f) or "") for f in PLANF).lower()]
    if len(A)<10 or len(B)<10: continue
    pa=sum(w in o.lower() for o in A)/len(A); pb=sum(w in o.lower() for o in B)/len(B)
    print(f"  '{w}': in {len(A)}/{len(recs)} plans -> output {100*pa:.1f}%  |  absent from plan -> output {100*pb:.1f}%  (lift {pa-pb:+.3f})")

# --- channel 1: scheduled categorical fields ---
print("\n=== CHANNEL 1: a scheduled categorical field ===")
tests=[("payload_type","personal_story",r"\b(i|we)\s+(had|got|used|bought|went|tried|shot|owned|found)\b"),
       ("claim_family","clarification_question",r"\?"),
       ("affect_role","annoyance",r"\b(annoying|frustrating|hate|awful|terrible|useless|sucks|disappointed)\b"),
       ("comment_function","question_followup",r"\?")]
for f,val,rx in tests:
    A=[o for _p,o,c in recs if str(c.get(f) or "")==val]
    B=[o for _p,o,c in recs if str(c.get(f) or "")!=val]
    if not A: continue
    pa=sum(bool(re.search(rx,o,re.I)) for o in A)/len(A)
    pb=sum(bool(re.search(rx,o,re.I)) for o in B)/len(B)
    print(f"  {f}={val:<24} n={len(A):3d} -> {100*pa:5.1f}%   |  all others -> {100*pb:5.1f}%   (lift {pa-pb:+.3f})")
