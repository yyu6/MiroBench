"""What is the natural spread of self_bertscore BETWEEN real threads?

`scripts/bootstrap_real_comment_discussions.py` states the logic this measurement
follows: attach a DIFFERENT real thread's comment tree to each seed and score it.
"if this bootstrap cannot match the matched real distribution, the issue is likely
seed/eval/matching/sample-size rather than the generator; if it does match, the
target distribution is reachable in principle." Its `similar_bucket` mode copies
the donor's cached metric row, so no rescoring is needed -- the comparison is
between two sets of real thread scores.

That 2026-06 run was on **credit_cards** and reached a self_bertscore bias of
-0.34% at 0.988 coverage. It has never been done on camera, which is the domain
every current number comes from, so it is done here from the cached real baseline.

This answers a question three sessions of ablations never asked: **how big is
+2.4% compared to how much real camera threads differ from each other?** If real
matched against real produces a bias of the same order, the remaining gap is
inside the metric's own noise and the paper should say so. If it produces ~0, the
+2.4% is real signal and the target is reachable in principle.

Donor matching mirrors `similar_bucket`: nearest comment_count, no self-donation,
each donor used at most once, disjoint from the evaluation set.
"""
from __future__ import annotations
import csv, json, random, statistics as st, sys
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
BASE = REPO / "artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"
POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
METRICS = ("self_bertscore_mean_f1", "self_bleu_4", "semantic_mean_cosine",
           "polite_rate", "impolite_rate", "neutral_rate")

rows = [r for r in csv.DictReader(open(BASE))]
pool = json.load(open(POOL))["seed_posts"]
eval_ids = [str(p["source_raw_post_id"]) for p in pool]
by_id = {str(r["thread_id"]): r for r in rows}


def ok(r):
    try:
        return int(float(r["comment_count"])) >= 5 and r["self_bertscore_mean_f1"] not in ("", "nan")
    except Exception:
        return False


targets = [by_id[i] for i in eval_ids if i in by_id and ok(by_id[i])]
donors_all = [r for r in rows if ok(r) and str(r["thread_id"]) not in set(eval_ids)]
print(f"real baseline rows {len(rows)}   evaluation targets found {len(targets)}"
      f"   disjoint donor pool {len(donors_all)}")

rng = random.Random(42)
pairs, used = [], set()
for t in sorted(targets, key=lambda r: -int(float(r["comment_count"]))):
    n = int(float(t["comment_count"]))
    cand = sorted((d for d in donors_all if str(d["thread_id"]) not in used),
                  key=lambda d: (abs(int(float(d["comment_count"])) - n), str(d["thread_id"])))
    if not cand:
        break
    d = cand[0]
    used.add(str(d["thread_id"]))
    pairs.append((t, d))
print(f"donor pairs formed {len(pairs)}")
tn = [int(float(t["comment_count"])) for t, _ in pairs]
dn = [int(float(d["comment_count"])) for _, d in pairs]
print(f"comment_count  targets mean {st.mean(tn):.1f}  donors mean {st.mean(dn):.1f}"
      f"  coverage {sum(dn)/sum(tn):.3f}")

print(f"\n{'metric':<26}{'target':>10}{'donor':>10}{'bias':>9}{'MWU':>10}{'KS':>10}{'verdict':>9}")
for m in METRICS:
    a = [float(t[m]) for t, _ in pairs if t.get(m) not in ("", "nan", None)]
    b = [float(d[m]) for t, d in pairs if t.get(m) not in ("", "nan", None)
         and d.get(m) not in ("", "nan", None)]
    if len(a) != len(b) or not a:
        a = [float(t[m]) for t, d in pairs if t.get(m) not in ("", "nan", None)
             and d.get(m) not in ("", "nan", None)]
    if not a or not b:
        print(f"{m:<26} unavailable")
        continue
    bias = 100 * (st.mean(b) - st.mean(a)) / st.mean(a)
    p1 = mannwhitneyu(a, b, alternative="two-sided").pvalue
    p2 = ks_2samp(a, b).pvalue
    holm = 0.05 / 24
    print(f"{m:<26}{st.mean(a):>10.4f}{st.mean(b):>10.4f}{bias:>8.2f}%{p1:>10.4f}{p2:>10.4f}"
          f"{('PASS' if min(p1,p2) > holm else 'FAIL'):>9}")

print(f"\nGenerated, for comparison (v113 gate, coverage 1.004): "
      f"self_bertscore +2.41%, self_bleu_4 +13.0%")
print("A real donor thread is what the generator is being asked to be indistinguishable from.")
