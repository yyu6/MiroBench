"""The 19% with no channel: is it that every generated comment has the same author?

All identified channels together return 81% of the +0.0119 gap, and implementing
every one of them reduces it by 31% against the 42% Holm needs. So the question
that decides whether `self_bertscore` is reachable at all is whether the missing
19% -- and more of the delivered part than the surface classes explain -- is one
thing nobody has ablated: a real thread is written by dozens of different people
and a generated one by a single model behind a single prompt scaffold.

FINDINGS s0 already showed the excess is a **uniform floor**, not repetition:
generated has FEWER near-duplicate pairs and MORE topical spread than real, with
top-k F1 lower and the median higher. A uniform floor is exactly what one shared
voice under many topics looks like.

Author identity cannot be ablated out of text. But it can be BOUNDED: inside real
threads, compare mean pairwise F1 for pairs written by the SAME author against
pairs written by different ones. If same-author pairs score materially higher,
that difference is the size of the one-voice floor, and generated sits at the
same-author end of it by construction.

Reports the bound, the share of real pairs that are same-author (so the bound can
be scaled), and generated's own author structure for comparison.
"""
from __future__ import annotations
import json, statistics as st, sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_self_bertscore import load_bert_scorer  # noqa: E402

GATE = "v113_v112_gate_n10_20260826_v1"
SEEDS = range(2, 12)
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}

real, cache = {}, {}
for s in SEEDS:
    p = by_seed[s]
    d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if d not in cache:
        cache[d] = load_real_comments(d)[0]
    real[s] = [(c.text, str(getattr(c, "author", "") or "")) for c in
               (cache[d].get(p["source_raw_post_id"]) or [])]
gen = {}
for d in sorted((REPO / "artifacts/generalized_card/runs" / GATE / "cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    for tid, cs in cbt.items():
        gen[int(tid.split("seed")[-1])] = [(c.text, str(getattr(c, "author", "") or "")) for c in cs]

print(f"{'side':<7}{'comments':>10}{'distinct authors':>18}{'comments/author':>17}"
      f"{'same-author pair share':>24}")
for label, thr in (("real", real), ("gate", gen)):
    n = sum(len(v) for v in thr.values())
    auth = {a for v in thr.values() for _, a in v if a}
    same = tot = 0
    for v in thr.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                tot += 1
                if v[i][1] and v[i][1] == v[j][1]:
                    same += 1
    print(f"{label:<7}{n:>10}{len(auth):>18}{n/max(len(auth),1):>17.2f}{same/max(tot,1):>24.4f}")

scorer, _, _, _, _, _ = load_bert_scorer(
    bert_score_path=REPO / "bert_score-master", model_type="microsoft/deberta-xlarge-mnli",
    num_layers=None, batch_size=8, device="auto", idf=False, idf_sents=[],
    rescale_with_baseline=False, local_files_only=True)


def pair_f1(rows):
    """-> (same_author_scores, diff_author_scores) for one thread."""
    c, r, flag = [], [], []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            c.append(rows[i][0]); r.append(rows[j][0])
            flag.append(bool(rows[i][1]) and rows[i][1] == rows[j][1])
    if not c:
        return [], []
    _, _, f1 = scorer.score(c, r, batch_size=8)
    vals = [float(x) for x in f1]
    return ([v for v, f in zip(vals, flag) if f],
            [v for v, f in zip(vals, flag) if not f])


print("\n== mean pairwise F1 inside REAL threads, by author identity ==")
same_all, diff_all, per_thread = [], [], []
for s in SEEDS:
    same, diff = pair_f1(real[s])
    same_all += same; diff_all += diff
    if len(same) >= 3:
        per_thread.append((s, len(same), st.mean(same), len(diff), st.mean(diff)))
        print(f"  seed {s:<3} same-author n={len(same):<5} {st.mean(same):.4f}   "
              f"different n={len(diff):<6} {st.mean(diff):.4f}   "
              f"delta {st.mean(same)-st.mean(diff):+.4f}")
if same_all:
    delta = st.mean(same_all) - st.mean(diff_all)
    print(f"\n  pooled: same-author {st.mean(same_all):.4f} (n={len(same_all)})   "
          f"different {st.mean(diff_all):.4f} (n={len(diff_all)})   delta {delta:+.4f}")
    share = len(same_all) / (len(same_all) + len(diff_all))
    print(f"  same-author share of real pairs: {share:.4f}")
    print(f"\n  BOUND: if every real pair were same-author, real's mean would rise by")
    print(f"         about {(1-share)*delta:+.4f}, against a +0.0119 generated gap")
    print(f"         = {100*(1-share)*delta/0.0119:.0f}% of it.")
else:
    print("  no same-author pairs found -- the loader may not carry authors")
