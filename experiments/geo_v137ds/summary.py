#!/usr/bin/env python3
"""One view of every generator on every domain, from the shared summary CSV."""
import csv, sys
from collections import defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "artifacts/reddit_multidomain_baselines/summary/evaluation_summary.csv"
TARGETS = ("self_bertscore_mean_f1", "self_bleu_4")

rows = list(csv.DictReader(open(CSV)))
g = defaultdict(list)
for r in rows:
    g[(r.get("test") or "two_sample", r["baseline"], r["model"], r["domain"])].append(r)

def passed(r):
    v = (r.get("verdict") or "").upper()
    if v: return v == "PASS"
    try: return float(r["mwu_p_value"]) > .05 and float(r["ks_p_value"]) > .05
    except (ValueError, KeyError, TypeError): return False

print(f"{CSV.relative_to(REPO)}  --  {len(rows)} rows\n")
print(f"{'test':14}{'baseline':10}{'model':20}{'domain':12}{'N':>6}{'PASS':>8}   目标指标")
print("-" * 92)
for k in sorted(g, key=lambda x: (x[0], x[3], x[1], x[2])):
    test, base, model, dom = k
    rs = g[k]
    n = max((int(r["generated_n"]) for r in rs if r.get("generated_n","").isdigit()), default=0)
    ok = sum(passed(r) for r in rs)
    tgt = []
    for m in TARGETS:
        hit = next((r for r in rs if r["metric"] == m), None)
        if hit:
            try: tgt.append(f"{m.split('_')[1][:4]} p={float(hit['mwu_p_value']):.3f}{'✓' if passed(hit) else '✗'}")
            except Exception: pass
    print(f"{test:14}{base:10}{model:20}{dom:12}{n:>6}{ok:>5}/{len(rs):<3}  {'  '.join(tgt)}")
