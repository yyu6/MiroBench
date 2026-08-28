"""Are seeds 2-11 an unrepresentative (hard) subset? Same run, split by seed."""
import csv, math, statistics
P="artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1/matched_evaluation"
g=list(csv.DictReader(open(P+"/matched_generated_thread_scores.csv")))
r=list(csv.DictReader(open(P+"/matched_real_thread_scores.csv")))
def key(x): return int(x.get("matched_seed_idx") or x.get("seed_index"))
G={key(x):x for x in g}; R={key(x):x for x in r}
common=sorted(set(G)&set(R))
print(f"paper50 matched seeds: {len(common)}  range {min(common)}..{max(common)}")
def vals(d,ss,k): return [float(d[s][k]) for s in ss if d[s].get(k) not in (None,"")]
def cliff(a,b):
    gt=sum(1 for x in a for y in b if x>y); lt=sum(1 for x in a for y in b if x<y)
    return (gt-lt)/(len(a)*len(b))
def sd_cliff(n): return (2/n)*math.sqrt((2*n+1)/12)

for label, ss in [("ALL 50", common),
                  ("seeds 2-11 (the n10 dev set)", [s for s in common if 2<=s<=11]),
                  ("seeds NOT in 2-11", [s for s in common if not (2<=s<=11)])]:
    if len(ss)<5: print(f"{label}: only {len(ss)}"); continue
    for m in ("self_bertscore_mean_f1","self_bleu_4"):
        a=vals(G,ss,m); b=vals(R,ss,m)
        d=cliff(a,b)
        print(f"  {label:<30} {m:<26} n={len(ss):3d}  gen={statistics.mean(a):.4f} real={statistics.mean(b):.4f} "
              f"d={d:+.3f}  (sd at this n = {sd_cliff(len(ss)):.3f}, |d|/sd = {abs(d)/sd_cliff(len(ss)):.1f})")
    print()

# per-seed gap on the dev seeds vs the rest
dev=[s for s in common if 2<=s<=11]; rest=[s for s in common if not (2<=s<=11)]
gd=[float(G[s]["self_bertscore_mean_f1"])-float(R[s]["self_bertscore_mean_f1"]) for s in dev]
gr=[float(G[s]["self_bertscore_mean_f1"])-float(R[s]["self_bertscore_mean_f1"]) for s in rest]
print(f"per-thread selfbert gap:  dev seeds mean {statistics.mean(gd):+.4f} (n={len(gd)})   "
      f"other seeds mean {statistics.mean(gr):+.4f} (n={len(gr)})")
print(f"  real selfbert level:    dev {statistics.mean([float(R[s]['self_bertscore_mean_f1']) for s in dev]):.4f}   "
      f"other {statistics.mean([float(R[s]['self_bertscore_mean_f1']) for s in rest]):.4f}")
print(f"  gen  selfbert level:    dev {statistics.mean([float(G[s]['self_bertscore_mean_f1']) for s in dev]):.4f}   "
      f"other {statistics.mean([float(G[s]['self_bertscore_mean_f1']) for s in rest]):.4f}")
