"""v110 groundwork: measure CLAUSE FORM rates, real excluded corpus vs generated.
Distinct from sentence_rhythm's punctuation habits, which were already rejected
as the cause of the function-word gap."""
import sys
import re
from pathlib import Path
from collections import defaultdict
REPO=Path('/Users/yaoningyu/Desktop/UIUC/GEO')
sys.path.insert(0,str(REPO/'scripts/evaluation'))
sys.path.insert(0,str(REPO/'generalized_card/analysis'))
from root_reply_diversity import load_excluded_threads
from score_thread_semantic_uniformity import load_generated_comments

SENT=re.compile(r'(?<=[.!?])\s+|\n+')
FORMS={
 'direct_question':   lambda s: s.rstrip().endswith('?'),
 'fragment':          lambda s: len(s.split())<=6 and not re.search(r'\b(is|are|was|were|am|be|do|does|did|have|has|had|will|would|can|could|should)\b',s.lower()),
 'imperative':        lambda s: bool(re.match(r'^(check|try|get|go|look|use|buy|keep|take|make|put|see|read|skip|avoid|grab|stick|shoot|test|wait|don\'?t|just)\b',s.strip().lower())),
 'conditional':       lambda s: bool(re.match(r'^\s*(if|unless|when|once|whenever)\b',s.lower())),
 'first_person_past': lambda s: bool(re.search(r"\bi\s+(\w+ed|had|was|went|got|bought|used|tried|found|shot|took|ran|kept|ended)\b",s.lower())),
 'concessive':        lambda s: bool(re.match(r'^\s*(but|though|although|still|that said|then again|admittedly)\b',s.lower())),
 'first_person_modal':lambda s: bool(re.search(r"\bi\s*('d|'ll|would|will|might|could|may)\b",s.lower())),
}
def bucket(n):
    if n<=10: return 'micro'
    if n<=25: return 'short'
    if n<=60: return 'medium'
    if n<=120: return 'long'
    return 'very_long'
def profile(texts):
    per=defaultdict(lambda: defaultdict(int)); n=defaultdict(int)
    for t in texts:
        w=len(t.split())
        if not w: continue
        b=bucket(w); n[b]+=1
        sents=[s for s in SENT.split(t) if s.strip()]
        for name,fn in FORMS.items():
            if any(fn(s) for s in sents): per[b][name]+=1
    return per,n

print("loading excluded real corpus...",flush=True)
th=load_excluded_threads()
real_texts=[c.text for cs in th.values() for c in cs]
print(f"real excluded comments: {len(real_texts)}",flush=True)
run=REPO/'artifacts/generalized_card/runs/generalized_card_camera_gpt54_v108_coverage_nonrepeat_n10_20260823_v1'
gen={}
for d in sorted(run.glob('cleaned/run_*_sampled_reddit')):
    c,_=load_generated_comments(d); gen.update(c)
gen_texts=[c.text for cs in gen.values() for c in cs]
print(f"generated comments: {len(gen_texts)}\n",flush=True)
RP,RN=profile(real_texts); GP,GN=profile(gen_texts)
BANDS=['micro','short','medium','long','very_long']
print("share of comments containing at least one clause of each form\n")
for name in FORMS:
    print(f"-- {name} --")
    print("  %-11s %8s %8s %7s   %s"%("band","real","gen","ratio","n real/gen"))
    for b in BANDS:
        if not RN[b] or not GN[b]: continue
        r=RP[b][name]/RN[b]; g=GP[b][name]/GN[b]
        print("  %-11s %8.3f %8.3f %7s   %d/%d"%(b,r,g,("%.2f"%(g/r) if r else "-"),RN[b],GN[b]))
    tr=sum(RP[b][name] for b in BANDS)/sum(RN.values()); tg=sum(GP[b][name] for b in BANDS)/sum(GN.values())
    print("  %-11s %8.3f %8.3f %7s"%("ALL",tr,tg,("%.2f"%(tg/tr) if tr else "-")))
