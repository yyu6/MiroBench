import json,glob,statistics
from collections import defaultdict
import sys
from pathlib import Path
REPO=Path('/Users/yaoningyu/Desktop/UIUC/GEO')
sys.path.insert(0,str(REPO/'scripts/evaluation'))
from score_thread_self_bleu import tokenize
STOP=set('a an the and or but if of to in on for with at by from as is are was were be been being it its this that these those i you he she we they my your our their me him her us them do does did did not no so too very can could will would should may might must have has had there here what which who when where how than then also just still really pretty s t re ve ll d m one two more most other same own'.split())
def cw(t): return {w for w in tokenize(t or '') if w.isalpha() and w not in STOP and len(w)>2}
def walk(cs):
    for c in cs:
        yield c
        yield from walk(c.get('replies',[]))
PLAN=('semantic_move','local_anchor','local_topic','detail_focus','decision_boundary',
      'branch_goal','owned_decision_subject','planner_intent','reply_delta','claim_key')
BINS=((0,1),(1,2),(2,4),(4,7),(7,999))
def db(d):
    for lo,hi in BINS:
        if lo<=d<hi: return "[%d,%s)"%(lo,hi if hi<999 else "+")
acc=defaultdict(list); accA=defaultdict(list)
for f in glob.glob(str(REPO/'artifacts/generalized_card/runs/generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1/generated/run_*/discussion.json')):
    for p in json.load(open(f))['posts']:
        seed=cw(p.get('title','')+' '+p.get('content',''))
        if not seed: continue
        for c in walk(p['comments']):
            b=db(int(c.get('depth',0) or 0))
            plan=cw(' '.join(str(c.get(k,'')) for k in PLAN))
            if plan: acc[b].append(len(plan&seed)/len(plan))
            anch=cw(' '.join(str(x) for x in (c.get('concrete_anchors') or [])))
            if anch: accA[b].append(len(anch&seed)/len(anch))
print("Share of the PLAN's own content words that come from the SEED POST\n")
print("%-9s %6s %14s %14s"%("depth","slots","plan fields","concrete_anchors"))
for lo,hi in BINS:
    b=db(lo)
    if b not in acc: continue
    a2=accA.get(b,[])
    print("%-9s %6d %14.4f %14s"%(b,len(acc[b]),statistics.fmean(acc[b]),
          ("%.4f"%statistics.fmean(a2)) if a2 else "-"))
allv=[x for v in acc.values() for x in v]
print("\nPOOLED plan->seed anchoring: %.4f over %d slots"%(statistics.fmean(allv),len(allv)))
print("\n(compare: REAL text->seed anchoring falls 0.1404 -> 0.0336 from depth 0 to [4,7);")
print(" generated TEXT stays 0.1058 -> 0.0643, i.e. +0.0308 above real in that bin)")
