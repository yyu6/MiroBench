#!/usr/bin/env python3
"""Rebuild summary/evaluation_summary.csv from every per-job result on disk.

evaluate.py used to truncate the summary with only the rows of the current
invocation, so a narrow run deleted every other result. Each job's own
evaluation/<baseline>/<model>/<domain>/metric_comparison.csv survived, so the
summary can be reconstructed from them. Merges rather than replaces: GEO's
matched_pair rows, which have no per-job file, are preserved.
"""
import csv, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
MD = REPO / "artifacts/reddit_multidomain_baselines"
OUT = MD / "summary/evaluation_summary.csv"
KEY = ("baseline", "model", "domain", "test", "metric")

recovered = []
for f in sorted(MD.glob("evaluation/*/*/*/metric_comparison.csv")):
    baseline, model, domain = f.parts[-4], f.parts[-3], f.parts[-2]
    rows = list(csv.DictReader(open(f)))
    for r in rows:
        r.setdefault("baseline", baseline)
        r.setdefault("model", model)
        r.setdefault("domain", domain)
        r["test"] = "two_sample"
    recovered += rows
    print(f"  {baseline}/{model}/{domain}: {len(rows)} metrics")

existing = list(csv.DictReader(open(OUT))) if OUT.exists() else []
# Before ``test`` became part of the summary identity, two-sample rows left
# this column blank.  Normalize them so rebuilding does not preserve a second,
# legacy copy beside the recovered ``test=two_sample`` row.
for row in existing:
    row["test"] = row.get("test") or "two_sample"
incoming = {tuple(str(r.get(k, "")) for k in KEY) for r in recovered}
kept = [r for r in existing if tuple(str(r.get(k, "")) for k in KEY) not in incoming]
allrows = kept + recovered
fields = []
for r in allrows:
    for k in r:
        if k not in fields:
            fields.append(k)
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows({k: r.get(k, "") for k in fields} for r in allrows)
print(f"\n{len(recovered)} rows recovered, {len(kept)} kept -> {OUT.relative_to(REPO)}")
