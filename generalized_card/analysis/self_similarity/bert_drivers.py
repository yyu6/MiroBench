"""What actually predicts self_bertscore inside real threads, and where the
generator is biased on that feature."""
from __future__ import annotations
import csv, math, random, re, sys
from pathlib import Path
import numpy as np
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO/"scripts"/"evaluation"))
from score_thread_self_bleu import tokenize
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments

DESIGNATOR = re.compile(r"\b(?=[A-Za-z0-9-]{2,8}\b)(?=[^\s]*\d)(?=[^\s]*[A-Za-z])[A-Za-z0-9-]+\b")
STOP = set("the a an and or but of to in for on with is are was were be been it its this that i you he she they we my your his her their our as at by from if so not no yes do does did have has had can could would should will just like about".split())
rng = random.Random(0)

def feats(texts: list[str]) -> dict[str, float]:
    toks = [tokenize(t) for t in texts]
    toks = [t for t in toks if t]
    if len(toks) < 2:
        return {}
    n = len(toks)
    pooled = [w for t in toks for w in t]
    sets = [set(t) for t in toks]
    csets = [set(w for w in t if w not in STOP and not w.isdigit() and len(w) > 2) for t in toks]
    idx = list(range(n))
    pairs = [(i, j) for i in idx for j in idx if i < j]
    if len(pairs) > 4000:
        pairs = rng.sample(pairs, 4000)
    def jac(a, b):
        u = len(a | b)
        return len(a & b)/u if u else 0.0
    d = DESIGNATOR
    dcounts: dict[str, int] = {}
    for t in texts:
        for m in d.finditer(t):
            k = m.group().lower(); dcounts[k] = dcounts.get(k, 0)+1
    dtot = sum(dcounts.values())
    lens = [len(t) for t in toks]
    return {
        "log_tokens": math.log(sum(lens)/n),
        "log_n": math.log(n),
        "ttr": len(set(pooled))/len(pooled),
        "jaccard_all": sum(jac(sets[i], sets[j]) for i, j in pairs)/len(pairs),
        "jaccard_content": sum(jac(csets[i], csets[j]) for i, j in pairs)/len(pairs),
        "desig_per_comment": len(dcounts)/n,
        "top_desig_share": (max(dcounts.values())/dtot) if dtot else 0.0,
        "share_digit": sum(1 for t in texts if any(c.isdigit() for c in t))/len(texts),
        "share_q": sum(1 for t in texts if "?" in t)/len(texts),
        "opener3": len({tuple(t[:3]) for t in toks})/n,
        "len_cv": (np.std(lens)/np.mean(lens)) if np.mean(lens) else 0.0,
    }

rows = {r["thread_id"]: r for r in csv.DictReader(open(REPO/"artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"))}
real_feats: dict[str, dict[str, float]] = {}
for dd in sorted((REPO/"data/raw/discussions/camera_product").iterdir()):
    if not dd.is_dir():
        continue
    try:
        cbt, _ = load_real_comments(dd)
    except Exception:
        continue
    for tid, comments in cbt.items():
        f = feats([c.text for c in comments])
        if f:
            real_feats[tid] = f

keys = ["log_tokens","log_n","ttr","jaccard_all","jaccard_content","desig_per_comment","top_desig_share","share_digit","share_q","opener3","len_cv"]
ids = [t for t in real_feats if t in rows and float(rows[t]["comment_count"]) >= 2]
X = np.array([[real_feats[t][k] for k in keys] for t in ids])
y = np.array([float(rows[t]["self_bertscore_mean_f1"]) for t in ids])
yb = np.array([float(rows[t]["self_bleu_4"]) for t in ids])
print(f"real threads used: {len(ids)}")

Z = (X - X.mean(0))/X.std(0)
for name, target in (("self_bertscore_mean_f1", y), ("self_bleu_4", yb)):
    A = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(A, target, rcond=None)
    pred = A@beta
    r2 = 1 - ((target-pred)**2).sum()/((target-target.mean())**2).sum()
    print(f"\n{name}: R^2={r2:.3f}  (standardized betas, per 1 SD of the feature)")
    order = sorted(range(len(keys)), key=lambda i: -abs(beta[i+1]))
    for i in order:
        rho = float(np.corrcoef(X[:,i], target)[0,1])
        print(f"   {keys[i]:<20} beta={beta[i+1]:+.5f}   simple r={rho:+.3f}")

# generated feature values on the 50 matched threads
import json
RUN = REPO/"artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"
gen_feats = {}
for d in sorted((RUN/"cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    for tid, comments in cbt.items():
        f = feats([c.text for c in comments])
        if f: gen_feats[tid] = f
pool = json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(p["seed_index"]): p for p in pool}
pairs = []
for tid, f in gen_feats.items():
    seed = int(tid.split("seed")[-1])
    rid = by_seed[seed]["source_raw_post_id"]
    if rid in real_feats:
        pairs.append((f, real_feats[rid]))
print(f"\nmatched pairs for the bias table: {len(pairs)}")
print(f"{'feature':<20}{'real':>10}{'generated':>11}{'gap':>10}{'gap in real SD':>16}")
for i, k in enumerate(keys):
    rv = np.mean([p[1][k] for p in pairs]); gv = np.mean([p[0][k] for p in pairs])
    sd = X[:,i].std()
    print(f"{k:<20}{rv:>10.4f}{gv:>11.4f}{gv-rv:>+10.4f}{(gv-rv)/sd:>+16.2f}")
