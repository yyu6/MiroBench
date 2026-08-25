"""Within-real elasticity of the two failing metrics w.r.t. comment length.

Uses only the shipped real baseline CSV plus the real corpus text, so nothing
here depends on the generator at all.
"""
from __future__ import annotations
import csv, math, sys
from pathlib import Path
import numpy as np
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO/"scripts"/"evaluation"))
from score_thread_self_bleu import tokenize
from score_thread_semantic_uniformity import load_real_comments

rows = list(csv.DictReader(open(REPO/"artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv")))
print("real baseline threads:", len(rows))

# mean tokens per comment, per real thread
lens: dict[str, float] = {}
root = REPO/"data/raw/discussions/camera_product"
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    try:
        cbt, _ = load_real_comments(d)
    except Exception:
        continue
    for tid, comments in cbt.items():
        if comments:
            lens[tid] = sum(len(tokenize(c.text)) for c in comments)/len(comments)
print("threads with text:", len(lens))

data = []
for r in rows:
    t = r["thread_id"]
    if t in lens and float(r["comment_count"]) >= 2:
        data.append((lens[t], float(r["comment_count"]),
                     float(r["self_bleu_4"]), float(r["self_bertscore_mean_f1"])))
print("usable:", len(data))
L = np.array([math.log(d[0]) for d in data])
N = np.array([math.log(d[1]) for d in data])
X = np.column_stack([np.ones_like(L), L, N])

for name, y in (("self_bleu_4", np.array([d[2] for d in data])),
                ("self_bertscore_mean_f1", np.array([d[3] for d in data]))):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y-pred)**2).sum()); ss_tot = float(((y-y.mean())**2).sum())
    # simple correlation too
    rho = float(np.corrcoef(L, y)[0,1])
    print(f"\n{name}")
    print(f"  mean over real threads          {y.mean():.5f}")
    print(f"  corr(log mean_tokens, metric)   {rho:+.3f}")
    print(f"  d(metric)/d(log mean_tokens)    {beta[1]:+.5f}   (controlling for log comment_count)")
    print(f"  d(metric)/d(log comment_count)  {beta[2]:+.5f}")
    print(f"  R^2                             {1-ss_res/ss_tot:.3f}")
    dlog = math.log(48.79/57.76)   # the generated length deficit, in logs
    print(f"  predicted shift from the generator's -15.5% length gap: {beta[1]*dlog:+.5f}")
