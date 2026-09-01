#!/usr/bin/env python3
"""One CSV holding every GEO v137ds matched-pair result.

The matched-pair test is GEO's own: each generated thread is scored against the
real thread it was built from.  That is a different test from the one
reddit_multidomain_baselines runs (a two-sample comparison against a shared real
reference), so the two live in different files on purpose.

  python3 experiments/geo_v137ds/matched_pair_table.py \
      --cohort camera deepseek-v4-flash <tag> <tag> ... \
      --cohort camera gpt-5.4-mini      <tag> <tag> ... \
      --out artifacts/reddit_multidomain_baselines/summary/geo_matched_pair_summary.csv
"""
import argparse, csv, statistics as st, sys
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp

REPO = Path(__file__).resolve().parents[2]
M12 = ["self_bertscore_mean_f1", "self_bleu_4", "semantic_mean_cosine",
       "hard_disagree_rate", "polite_rate", "impolite_rate", "neutral_rate",
       "length_cv", "avg_depth", "structural_virality",
       "mean_story_probability", "emotion_entropy"]
TARGETS = {"self_bertscore_mean_f1", "self_bleu_4"}

def cliff(a, b):
    n = len(a) * len(b)
    if not n: return float("nan")
    g = sum(1 for x in a for y in b if x > y)
    l = sum(1 for x in a for y in b if x < y)
    return (g - l) / n

def cohort(domain, model, tags):
    gen, real, seen, missing = [], [], set(), []
    for t in tags:
        d = REPO / "artifacts/generalized_card/runs" / t / "matched_evaluation"
        if not d.exists():
            missing.append(t); continue
        g = list(csv.DictReader(open(d / "matched_generated_thread_scores.csv")))
        r = list(csv.DictReader(open(d / "matched_real_thread_scores.csv")))
        for gi, ri in zip(g, r):
            k = gi.get("source_raw_post_id", "")
            if k in seen: continue
            seen.add(k); gen.append(gi); real.append(ri)
    return gen, real, missing

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", nargs="+", action="append", required=True,
                metavar="DOMAIN MODEL TAG...",
                help="one per cohort: domain, model, then its run tags")
ap.add_argument("--out", required=True)
ap.add_argument("--generator", default="geo_v137ds")
a = ap.parse_args()

rows, summary = [], []
for spec in a.cohort:
    if len(spec) < 3:
        sys.exit(f"--cohort needs DOMAIN MODEL TAG...: got {spec}")
    domain, model, tags = spec[0], spec[1], spec[2:]
    gen, real, missing = cohort(domain, model, tags)
    n = len(gen)
    if n < 3:
        print(f"  {domain}/{model}: only {n} scored threads -- skipped"
              + (f" ({len(missing)} tags unscored)" if missing else ""))
        continue
    npass = 0
    for metric in M12:
        try:
            ga = [float(x[metric]) for x in gen if x.get(metric) not in (None, "", "nan")]
            ra = [float(x[metric]) for x in real if x.get(metric) not in (None, "", "nan")]
        except (ValueError, KeyError):
            continue
        if len(ga) < 3 or len(ra) < 3: continue
        mw = mannwhitneyu(ga, ra, alternative="two-sided").pvalue
        ks = ks_2samp(ga, ra).pvalue
        ok = mw > 0.05 and ks > 0.05
        npass += ok
        mg, mr = st.mean(ga), st.mean(ra)
        rows.append({
            "generator": a.generator, "model": model, "domain": domain,
            "test": "matched_pair", "n_threads": n, "metric": metric,
            "is_target_metric": metric in TARGETS,
            "generated_mean": f"{mg:.6f}", "real_mean": f"{mr:.6f}",
            "relative_difference_pct": f"{100*(mg-mr)/mr:.4f}" if mr else "",
            "mwu_p_value": f"{mw:.6g}", "ks_p_value": f"{ks:.6g}",
            "cliffs_delta": f"{cliff(ga, ra):+.4f}",
            "verdict": "PASS" if ok else "FAIL",
            "shards": len(tags), "shards_unscored": len(missing),
        })
    summary.append((domain, model, n, npass, len(missing)))
    print(f"  {domain}/{model}: PASS {npass}/12 at N={n}"
          + (f"   [{len(missing)} of {len(tags)} shards unscored]" if missing else ""))

out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"\nwrote {out}  ({len(rows)} rows, {len(summary)} cohorts)")
print(f"\n{'domain':12}{'model':20}{'N':>6}{'PASS':>7}")
for d_, m_, n_, p_, _ in summary:
    print(f"{d_:12}{m_:20}{n_:>6}{p_:>5}/12")
