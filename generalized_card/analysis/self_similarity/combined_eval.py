#!/usr/bin/env python3
"""Evaluate several runs as ONE sample.

Each run's matched_evaluation CSVs already hold per-thread scores against that
thread's own matched real thread, so pooling the rows is exactly equivalent to
having evaluated them together -- the MWU/KS tests consume per-thread values.

  combined_eval.py --tags v117_calibration_20260826_v1 v117_40_more_...
"""
import argparse, csv, statistics, sys
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
M12 = ["self_bertscore_mean_f1","self_bleu_4","semantic_mean_cosine","hard_disagree_rate",
       "polite_rate","impolite_rate","neutral_rate","length_cv","avg_depth",
       "structural_virality","mean_story_probability","emotion_entropy"]

def cliff(a,b):
    n=len(a)*len(b)
    if not n: return float("nan")
    g=sum(1 for x in a for y in b if x>y); l=sum(1 for x in a for y in b if x<y)
    return (g-l)/n

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--dedupe", action="store_true",
                help="drop rows whose source real thread already appeared in an "
                     "earlier tag; use when shards overlap on seed index")
a = ap.parse_args()


def key(row):
    """Identify the REAL thread a row was scored against.

    Two shards that share a seed index score the same real thread twice, which
    would double-weight it in the MWU/KS sample. Seed index alone is not the
    key -- different seed pools reuse the same indices for different posts.
    """
    return (row.get("source_product_dir", ""), row.get("source_raw_post_id", ""))


gen, real, seen = [], [], set()
for t in a.tags:
    d = REPO/"artifacts/generalized_card/runs"/t/"matched_evaluation"
    if not d.exists(): sys.exit(f"missing matched_evaluation for {t}")
    g = list(csv.DictReader(open(d/"matched_generated_thread_scores.csv")))
    r = list(csv.DictReader(open(d/"matched_real_thread_scores.csv")))
    dropped = []
    if a.dedupe:
        kept_g, kept_r = [], []
        for gi, ri in zip(g, r):
            k = key(gi)
            if k in seen:
                dropped.append(gi.get("seed_index", "?"))
                continue
            seen.add(k); kept_g.append(gi); kept_r.append(ri)
        g, r = kept_g, kept_r
    note = f"   (dropped duplicate seeds {','.join(dropped)})" if dropped else ""
    print(f"  {t}: {len(g)} generated / {len(r)} real threads{note}")
    gen += g; real += r
n = len(gen)
sd = (2/n)*((2*n+1)/12)**0.5
print(f"\npooled N = {n}   sd of Cliff d at this N = {sd:.3f}\n")
print(f"{'metric':30}{'gen':>10}{'real':>10}{'rel%':>9}{'mwu':>9}{'ks':>8}{'d':>7}  verdict")
print("-"*92)
npass=0
for k in M12:
    try:
        ga=[float(x[k]) for x in gen if x.get(k) not in (None,"","nan")]
        ra=[float(x[k]) for x in real if x.get(k) not in (None,"","nan")]
    except (ValueError,KeyError): continue
    if len(ga)<3 or len(ra)<3: continue
    mw=mannwhitneyu(ga,ra,alternative="two-sided").pvalue
    ks=ks_2samp(ga,ra).pvalue
    ok = mw>0.05 and ks>0.05; npass+=ok
    mg,mr=statistics.mean(ga),statistics.mean(ra)
    print(f"{k:30}{mg:>10.4f}{mr:>10.4f}{100*(mg-mr)/mr:>+8.1f}%{mw:>9.3f}{ks:>8.3f}"
          f"{cliff(ga,ra):>+7.2f}  {'PASS' if ok else 'FAIL'}")
print(f"\nPASS {npass}/12 at N={n}")
