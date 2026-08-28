#!/usr/bin/env python3
"""Read the vocabulary->self_bertscore slope off the constructed threads and
price the 137-type deficit against the gap we actually have to close."""
import json, statistics, sys
from pathlib import Path
P = Path("/private/tmp/claude-501/-Users-yaoningyu-Desktop-UIUC-GEO/d8816651-1679-43a5-8d4b-21a1a35e5936/scratchpad/vocab_dose_scored.json")
rows = json.loads(P.read_text())
rows.sort(key=lambda r: r["types"])

def ols(xs, ys):
    n=len(xs); mx=statistics.mean(xs); my=statistics.mean(ys)
    sxx=sum((x-mx)**2 for x in xs); sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    b=sxy/sxx; a=my-b*mx
    yh=[a+b*x for x in xs]
    ss=sum((y-h)**2 for y,h in zip(ys,yh)); tt=sum((y-my)**2 for y in ys)
    se=(ss/(n-2)/sxx)**0.5
    return a,b,(1-ss/tt if tt else float("nan")),se

print(f"{'types':>7}{'tokens':>8}{'bert_mean_f1':>14}{'cos_mean':>10}")
for r in rows: print(f"{r['types']:>7}{r['tokens']:>8}{r['bert_mean_f1']:>14.4f}{r['cos_mean']:>10.4f}")

xs=[r["types"] for r in rows]
print("\n--- OLS on constructed real-comment threads (tokens held at ~2610) ---")
for key,label in (("bert_mean_f1","self_bertscore mean F1"),("cos_mean","semantic mean cosine (CONTROL)")):
    ys=[r[key] for r in rows]
    a,b,r2,se=ols(xs,ys)
    print(f"{label:34} slope {b*100:+.5f} per 100 types  (SE {se*100:.5f})  R2 {r2:.3f}")
    print(f"{'':34}  t = {b/se:+.2f}")

a,b,r2,se=ols(xs,[r["bert_mean_f1"] for r in rows])
DEF=136.9                    # our type deficit at 2610 tokens (real 819.5, ours 682.6)
GAP=0.5117-0.4942            # v128 self_bertscore mean F1 minus its matched real
pred=b*DEF
lo,hi=(b-1.96*se)*DEF,(b+1.96*se)*DEF
print(f"\nour deficit: {DEF:.1f} types.  predicted self_bertscore effect of closing it: {pred:+.5f}")
print(f"  95% CI [{min(lo,hi):+.5f}, {max(lo,hi):+.5f}]")
print(f"gap that must be closed (v128 0.5117 vs real 0.4942): {GAP:+.5f}")
print(f"\n  => closing the vocabulary deficit explains {100*(-pred)/GAP:.0f}% of the gap"
      f"   [CI {100*(-max(lo,hi))/GAP:.0f}% .. {100*(-min(lo,hi))/GAP:.0f}%]")
