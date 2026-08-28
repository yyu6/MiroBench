#!/usr/bin/env python3
"""Second independent lens: let a classifier tell us what separates real from
generated, instead of me proposing properties one at a time.

Trained on held-out real threads (zero seed overlap with the generated run) so
'real' is not just 'the matched threads'. Reports the features it leans on.
"""
import json, sys, re, random, statistics
from pathlib import Path
import numpy as np
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO/"scripts"/"evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
rng = random.Random(5)

pool = json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
used = {p["source_raw_post_id"] for p in pool if 2 <= int(p["seed_index"]) <= 11}

gen = []
for x in sorted((REPO/"artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(x)
    for cs in cbt.values(): gen += [" ".join(c.text.split()) for c in cs]

real = []
for d in sorted((REPO/"data/raw/discussions/camera_product").iterdir()):
    if not d.is_dir(): continue
    try: bt, _ = load_real_comments(d)
    except Exception: continue
    for pid, cs in bt.items():
        if pid in used: continue                       # held out
        real += [" ".join(c.text.split()) for c in cs]
rng.shuffle(real); real = real[:len(gen)*3]
print(f"generated {len(gen):,}   held-out real {len(real):,}", flush=True)

X = gen + real; y = [1]*len(gen) + [0]*len(real)
vec = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=60000, sublinear_tf=True)
M = vec.fit_transform(X)
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
auc = cross_val_score(clf, M, y, cv=5, scoring="roc_auc")
print(f"5-fold ROC-AUC separating a single comment: {auc.mean():.4f} (+/- {auc.std():.4f})")
clf.fit(M, y)
names = np.array(vec.get_feature_names_out()); w = clf.coef_[0]
o = np.argsort(w)
print("\nMOST GENERATED-ish features (the tells a human discriminator sees):")
print("  " + ", ".join(f"{names[i]}({w[i]:+.2f})" for i in o[::-1][:40]))
print("\nMOST REAL-ish features (what we never say):")
print("  " + ", ".join(f"{names[i]}({w[i]:+.2f})" for i in o[:40]))
