#!/usr/bin/env python3
"""Price the link arm on self_bertscore using an artifact that ACTUALLY RAN IT.

G80's rule: splicing text into a finished artifact cannot predict what a cue
does to the Writer. v113 and v117 ran --reference-link measured for real, so
their own link-carrying comments are the only honest evidence. Ablate them
from those runs with a random-comment control: that is what the arm bought.
"""
import json, sys, re, itertools, statistics, random
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "bert_score-master"))
from score_thread_semantic_uniformity import load_generated_comments
rng = random.Random(3)
URL = re.compile(r"https?://|www\.|\[.*?\]\(.*?\)")

def threads(tag):
    out = {}
    for sub in ("cleaned", "generated"):
        root = REPO/"artifacts/generalized_card/runs"/tag/sub
        if not root.exists(): continue
        for x in sorted(root.glob("run_*_sampled_reddit")):
            try: cbt, _ = load_generated_comments(x)
            except Exception: continue
            for tid, cs in cbt.items(): out[tid] = [" ".join(c.text.split()) for c in cs]
        if out: break
    return out

from bert_score import BERTScorer
sc = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", num_layers=40, batch_size=32,
                idf=False, device="cpu", lang="en", rescale_with_baseline=False)
def mean_f1(t):
    if len(t) < 6: return None
    pr = list(itertools.combinations(range(len(t)), 2))
    _, _, F = sc.score([t[i] for i,_ in pr], [t[j] for _,j in pr], batch_size=64)
    return statistics.mean(F.tolist())

for TAG in ("v117_calibration_20260826_v1", "v113_v112_gate_n10_20260826_v1"):
    T = threads(TAG)
    if not T: print(f"{TAG}: no threads found"); continue
    tot_c = sum(len(v) for v in T.values())
    tot_l = sum(sum(1 for t in v if URL.search(t)) for v in T.values())
    print(f"\n### {TAG}: {len(T)} threads, {tot_c} comments, {tot_l} carry a URL "
          f"({100*tot_l/tot_c:.2f}%)   [real 5.14%]", flush=True)
    rows = []
    for tid, cs in sorted(T.items()):
        cs = cs[:44]
        link = [t for t in cs if URL.search(t)]
        keep = [t for t in cs if not URL.search(t)]
        k = len(link)
        if k == 0 or len(keep) < 8: continue
        ctl = list(cs)
        for i in sorted(rng.sample(range(len(ctl)), k), reverse=True): ctl.pop(i)
        b, a, c = mean_f1(cs), mean_f1(keep), mean_f1(ctl)
        rows.append((tid, k, b, a, c))
        print(f"  {tid[-22:]:24} links={k:<2} as-is {b:.4f} -> minus-links {a:.4f} "
              f"(random ctl {c:.4f})", flush=True)
    if not rows: print("  no thread carries a link"); continue
    B = statistics.mean(r[2] for r in rows); A = statistics.mean(r[3] for r in rows)
    C = statistics.mean(r[4] for r in rows)
    net = (A - B) - (C - B)
    print(f"  ---- {len(rows)} threads, {statistics.mean(r[1] for r in rows):.1f} links/thread")
    print(f"  as-is {B:.4f}   minus-links {A:.4f} ({A-B:+.4f})   random ctl {C:.4f} ({C-B:+.4f})")
    print(f"  NET the arm's OWN link comments are worth: {net:+.5f} on self_bertscore")
    print(f"  (the gap to close is +0.0175; beats control in "
          f"{sum(1 for r in rows if r[3] > r[4])}/{len(rows)} threads)")
