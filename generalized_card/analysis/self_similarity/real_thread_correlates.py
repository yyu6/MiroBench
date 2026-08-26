"""Real threads span a range of self_bertscore. What moves it, and where does the generator sit?

real_vs_real_floor.py established the target is reachable: an arbitrary disjoint
real camera thread matches on all six metrics, with a self_bertscore bias of
+0.24% against the generator's +2.41%. So the question is what puts a real thread
high or low on this metric, measured across the whole cached baseline rather than
the ten seeds every ablation has used.

FINDINGS s3 tried a nine-feature regression on 536 threads, reached R^2=0.60, and
predicted only 40% of the gap with contradictory signs -- "regression is not
identification". This is deliberately not that. It ranks every cached thread-level
column by its correlation with self_bertscore inside REAL data, then reports where
the generator's own threads sit on each, in units of real's own spread. A column
where the generator is many real standard deviations away AND which correlates
with self_bertscore is a candidate; a column where it sits inside real's spread is
not, whatever the regression says.
"""
from __future__ import annotations
import csv, json, statistics as st, sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
BASE = REPO / "artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"
EVAL = REPO / "artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1/matched_evaluation"
GEN = EVAL / "matched_generated_thread_scores.csv"
# The generator copies its matched real thread's STRUCTURE (ORIENTATION.md s4:
# every matched structural slot is preserved), and the 50 evaluation threads are
# larger than the real population (39.1 comments against 32.4). Comparing the
# generator to the population therefore invents a structural gap that is really a
# selection effect. Every z below is against the generator's OWN matched real
# threads; real's spread still comes from the full baseline.
MATCHED_REAL = EVAL / "matched_real_thread_scores.csv"
TARGET = "self_bertscore_mean_f1"

real = [r for r in csv.DictReader(open(BASE))]
gen = [r for r in csv.DictReader(open(GEN))]
matched = [r for r in csv.DictReader(open(MATCHED_REAL))]
print(f"real baseline threads {len(real)}   generated {len(gen)}   their matched real {len(matched)}")


def col(rows, k):
    out = []
    for r in rows:
        v = r.get(k)
        if v in (None, "", "nan", "NaN"):
            out.append(None); continue
        try:
            out.append(float(v))
        except ValueError:
            out.append(None)
    return out


def pearson(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 30:
        return None
    xs, ys = zip(*pairs)
    mx, my = st.mean(xs), st.mean(ys)
    sx, sy = st.pstdev(xs), st.pstdev(ys)
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / (len(pairs) * sx * sy)


y = col(real, TARGET)
numeric = [k for k in real[0] if k not in ("product", "thread_id") and pearson(col(real, k), y) is not None]
rows = []
for k in numeric:
    if k.startswith("self_bertscore"):
        continue
    r = pearson(col(real, k), y)
    rv = [v for v in col(real, k) if v is not None]
    gv = [v for v in col(gen, k) if v is not None] if gen and k in gen[0] else []
    mv = [v for v in col(matched, k) if v is not None] if matched and k in matched[0] else []
    if not rv or not gv or not mv:
        continue
    sd = st.pstdev(rv)
    z = (st.mean(gv) - st.mean(mv)) / sd if sd else None
    rows.append((k, r, st.mean(mv), st.mean(gv), z, sd))

print(f"\n{'column':<34}{'corr w/ sbert':>14}{'matched real':>14}{'gen mean':>12}{'gen z':>9}{'|r*z|':>9}")
for k, r, rm, gm, z, sd in sorted(rows, key=lambda t: -abs((t[1] or 0) * (t[4] or 0)))[:24]:
    print(f"{k[:32]:<34}{r:>14.3f}{rm:>14.4f}{gm:>12.4f}{z:>9.2f}{abs(r*z):>9.2f}")

print("\n== reading it ==")
print("  corr  : how strongly this moves self_bertscore INSIDE real threads")
print("  gen z : how far the generator sits from real, in real's own standard deviations")
print("  |r*z| : a column only matters if BOTH are large -- it must move the metric")
print("          and the generator must actually differ on it.")
top = sorted(rows, key=lambda t: -abs((t[1] or 0) * (t[4] or 0)))[:6]
print("\n  top candidates:")
for k, r, rm, gm, z, sd in top:
    d = "higher" if z > 0 else "lower"
    push = "up" if (r > 0) == (z > 0) else "down"
    print(f"    {k:<32} generator is {abs(z):.1f} sd {d}; corr {r:+.2f} -> pushes self_bertscore {push}")
