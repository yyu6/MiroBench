#!/usr/bin/env python3
"""GEO's matched-pair results, in the schema the multidomain summary uses.

Two different tests exist per cohort and they are NOT interchangeable:

  * matched_pair -- each generated thread against the real thread it was built
    from.  This is where the camera 8/12 comes from.
  * two_sample -- the harness's own comparison against a shared real reference
    (`inputs/real_reference/<domain>`), written by run_evaluate_domain.sh.

Both now land in artifacts/reddit_multidomain_baselines/summary/, keyed the same
way and distinguished by the `test` column, so one file answers "how did every
generator do on every domain".  Rows are MERGED on
(baseline, model, domain, test, metric): re-running one cohort replaces only its
own rows and leaves every other result in place.

  python3 experiments/geo_v137ds/matched_pair_table.py \
      --cohort camera deepseek-v4-flash --tags <tag> <tag> ...
"""
import argparse, csv, statistics as st, sys
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "artifacts/reddit_multidomain_baselines/summary"
M12 = ["self_bertscore_mean_f1", "self_bleu_4", "semantic_mean_cosine",
       "hard_disagree_rate", "polite_rate", "impolite_rate", "neutral_rate",
       "length_cv", "avg_depth", "structural_virality",
       "mean_story_probability", "emotion_entropy"]
TARGETS = {"self_bertscore_mean_f1", "self_bleu_4"}
KEY = ("baseline", "model", "domain", "test", "metric")


def cliff(a, b):
    n = len(a) * len(b)
    if not n:
        return float("nan")
    g = sum(1 for x in a for y in b if x > y)
    l = sum(1 for x in a for y in b if x < y)
    return (g - l) / n


def load_cohort(tags):
    """Pool the shards, dropping a thread whose real counterpart already appeared."""
    gen, real, seen, missing = [], [], set(), []
    for t in tags:
        d = REPO / "artifacts/generalized_card/runs" / t / "matched_evaluation"
        if not d.exists():
            missing.append(t)
            continue
        g = list(csv.DictReader(open(d / "matched_generated_thread_scores.csv")))
        r = list(csv.DictReader(open(d / "matched_real_thread_scores.csv")))
        for gi, ri in zip(g, r):
            k = gi.get("source_raw_post_id", "")
            if k in seen:
                continue
            seen.add(k)
            gen.append(gi)
            real.append(ri)
    return gen, real, missing


def rows_for(domain, model, tags, baseline):
    gen, real, missing = load_cohort(tags)
    n = len(gen)
    if n < 3:
        return [], n, 0, missing
    out, npass = [], 0
    for metric in M12:
        try:
            ga = [float(x[metric]) for x in gen if x.get(metric) not in (None, "", "nan")]
            ra = [float(x[metric]) for x in real if x.get(metric) not in (None, "", "nan")]
        except (ValueError, KeyError):
            continue
        if len(ga) < 3 or len(ra) < 3:
            continue
        mw = mannwhitneyu(ga, ra, alternative="two-sided").pvalue
        ks = ks_2samp(ga, ra).pvalue
        ok = mw > 0.05 and ks > 0.05
        npass += ok
        mg, mr = st.mean(ga), st.mean(ra)
        out.append({
            "metric": metric,
            "real_n": len(ra), "generated_n": len(ga),
            "real_mean": f"{mr:.10g}", "generated_mean": f"{mg:.10g}",
            "mean_difference_generated_minus_real": f"{mg - mr:.10g}",
            "wasserstein_distance": "",
            "ks_statistic": "", "ks_p_value": f"{ks:.10g}",
            "mwu_statistic": "", "mwu_p_value": f"{mw:.10g}",
            "cliffs_delta": f"{cliff(ga, ra):.10g}",
            "baseline": baseline, "model": model, "domain": domain,
            "generation_report": "",
            "test": "matched_pair",
            "is_target_metric": str(metric in TARGETS).lower(),
            "verdict": "PASS" if ok else "FAIL",
            "shards": len(tags), "shards_unscored": len(missing),
        })
    return out, n, npass, missing


def merge(path, rows, key):
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
    incoming = {tuple(str(r.get(k, "")) for k in key) for r in rows}
    kept = [r for r in existing if tuple(str(r.get(k, "")) for k in key) not in incoming]
    allrows = kept + rows
    fields = []
    for r in allrows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in allrows)
    return len(kept)


ap = argparse.ArgumentParser()
ap.add_argument("--cohort", nargs=2, metavar=("DOMAIN", "MODEL"), required=True)
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--baseline", default="geo")
ap.add_argument("--out", default=str(SUMMARY / "evaluation_summary.csv"))
a = ap.parse_args()

domain, model = a.cohort
rows, n, npass, missing = rows_for(domain, model, a.tags, a.baseline)
if not rows:
    sys.exit(f"{domain}/{model}: only {n} scored threads -- nothing written"
             + (f" ({len(missing)} of {len(a.tags)} shards unscored)" if missing else ""))

kept = merge(Path(a.out), rows, KEY)
print(f"{a.baseline}/{model}/{domain}  matched_pair  PASS {npass}/12 at N={n}"
      + (f"   [{len(missing)} of {len(a.tags)} shards unscored]" if missing else ""))
print(f"  {len(rows)} rows merged into {Path(a.out).relative_to(REPO)}, {kept} earlier rows kept")
