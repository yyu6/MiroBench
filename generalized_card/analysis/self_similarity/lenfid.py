"""Realized/assigned length by assigned band on the N=50 artifact."""
import json, statistics as st
from pathlib import Path
RUN=Path("/Users/yaoningyu/Desktop/UIUC/GEO/artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1")
rows=[]
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    disc=json.load(open(d/"discussion.json"))
    for post in disc["posts"]:
        for rec in post.get("generation_records") or []:
            t=rec.get("task") or {}
            a=int(t.get("real_word_count") or 0)
            c=rec.get("comment") or {}
            txt=str(c.get("content") or "")
            if a>0 and txt:
                rows.append((a,len(txt.split())))
print(f"slots with an assigned length and persisted text: {len(rows)}")
bands=[(1,9),(10,19),(20,34),(35,49),(50,69),(70,100),(101,150),(151,300),(301,10000)]
print(f"\n{'assigned band':<16}{'slots':>7}{'assigned':>10}{'realized':>10}{'ratio':>8}{'share of words':>16}")
tot_a=sum(a for a,_ in rows)
for lo,hi in bands:
    sub=[(a,r) for a,r in rows if lo<=a<=hi]
    if not sub: continue
    A=sum(a for a,_ in sub); R=sum(r for _,r in sub)
    print(f"{f'{lo}-{hi}':<16}{len(sub):>7}{A/len(sub):>10.1f}{R/len(sub):>10.1f}{R/A:>8.3f}{100*A/tot_a:>15.1f}%")
A=sum(a for a,_ in rows); R=sum(r for _,r in rows)
print(f"\n{'TOTAL':<16}{len(rows):>7}{A/len(rows):>10.1f}{R/len(rows):>10.1f}{R/A:>8.3f}")
sub=[(a,r) for a,r in rows if 35<=a<=100]
print(f"\nv111's target band (assigned 35-100): {len(sub)} slots = {100*len(sub)/len(rows):.1f}% of slots, "
      f"{100*sum(a for a,_ in sub)/A:.1f}% of assigned words, ratio {sum(r for _,r in sub)/sum(a for a,_ in sub):.3f}")
above=[(a,r) for a,r in rows if a>100]
print(f"above 100 (already has a beat plan):  {len(above)} slots, ratio {sum(r for _,r in above)/sum(a for a,_ in above):.3f}")
