#!/usr/bin/env python3
"""One view of every generator on every domain, from the shared summary CSV.

Jobs do not all score the same metric set -- the laptop OASIS job scored 57
metrics including detoxify, the camera jobs 15 -- so a raw PASS/total is not
comparable across rows. Everything is reported over the twelve canonical thread
metrics, with the job's own metric count shown separately.
"""
import csv, sys
from collections import defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "artifacts/reddit_multidomain_baselines/summary/evaluation_summary.csv"
M12 = ["self_bertscore_mean_f1", "self_bleu_4", "semantic_mean_cosine",
       "hard_disagree_rate", "polite_rate", "impolite_rate", "neutral_rate",
       "length_cv", "avg_depth", "structural_virality",
       "mean_story_probability", "emotion_entropy"]
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

print(f"{CSV.relative_to(REPO)}  --  {len(rows)} rows")
print("PASS 一律按 12 个核心指标算；'scored' 是该 job 自己算了多少指标\n")
print(f"{'test':13}{'baseline':10}{'model':19}{'domain':11}{'N':>5}{'PASS/12':>9}{'scored':>8}   目标指标")
print("-" * 104)
for k in sorted(g, key=lambda x: (x[0], x[3], -sum(passed(r) for r in g[x] if r["metric"] in M12))):
    test, base, model, dom = k
    rs = g[k]
    by = {r["metric"]: r for r in rs}
    n = max((int(r["generated_n"]) for r in rs if str(r.get("generated_n","")).isdigit()), default=0)
    have = [m for m in M12 if m in by]
    ok = sum(passed(by[m]) for m in have)
    tgt = []
    for m in TARGETS:
        r = by.get(m)
        if not r: continue
        try: tgt.append(f"{'bert' if 'bert' in m else 'bleu'} p={float(r['mwu_p_value']):.3f}{'✓' if passed(r) else '✗'}")
        except Exception: pass
    miss = "" if len(have) == 12 else f" ({12-len(have)} 个核心指标缺)"
    print(f"{test:13}{base:10}{model:19}{dom:11}{n:>5}{ok:>6}/{len(have):<3}{len(rs):>8}   {'  '.join(tgt)}{miss}")
