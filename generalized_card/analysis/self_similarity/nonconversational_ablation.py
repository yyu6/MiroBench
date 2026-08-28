#!/usr/bin/env python3
"""Controlled ablation, priced on self_bertscore (G136's rule).

Reading the floor pairs says real threads contain comments that are not
conversational turns at all -- link dumps, quoted spec blocks, terse factual
corrections. We emit none of those. So: delete them from REAL and see whether
real's self_bertscore rises toward ours. Control: delete the same NUMBER of
comments at random.
"""
import json, sys, re, itertools, statistics, random
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "bert_score-master"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
rng = random.Random(11)
TOK = re.compile(r"[a-z0-9']+")
URL = re.compile(r"https?://|www\.|\[.*?\]\(.*?\)")

def is_nonconversational(t: str) -> bool:
    s = t.strip()
    w = TOK.findall(s.lower())
    if URL.search(s): return True                       # link dump
    q = sum(1 for ln in s.splitlines() if ln.strip().startswith(">"))
    if q and q >= max(1, len(s.splitlines()) // 2): return True   # mostly quoted block
    if len(w) <= 12 and any(c.isdigit() for c in s): return True  # terse spec/number reply
    return False

pool = json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by = {int(p["seed_index"]): p for p in pool}
G = {}
for x in sorted((REPO/"artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(x)
    for tid, cs in cbt.items(): G[int(tid.split("seed")[-1])] = [" ".join(c.text.split()) for c in cs]

from bert_score import BERTScorer
sc = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", num_layers=40,
                batch_size=32, idf=False, device="cpu", lang="en", rescale_with_baseline=False)
def mean_f1(texts):
    if len(texts) < 6: return None
    pr = list(itertools.combinations(range(len(texts)), 2))
    _, _, F = sc.score([texts[i] for i,_ in pr], [texts[j] for _,j in pr], batch_size=64)
    return statistics.mean(F.tolist())

rows = []
for S in sorted(G):
    p = by[S]
    real = [" ".join(c.text.split()) for c in (load_real_comments(REPO/"data/raw/discussions/camera_product"/p["source_product_dir"])[0].get(p["source_raw_post_id"]) or [])]
    if len(real) < 14: continue
    real = real[:44]
    nc = [t for t in real if is_nonconversational(t)]
    keep = [t for t in real if not is_nonconversational(t)]
    k = len(nc)
    if k == 0 or len(keep) < 8: 
        print(f"seed{S}: {k} non-conversational of {len(real)} -- skipped", flush=True); continue
    ctl = list(real); 
    for i in sorted(rng.sample(range(len(ctl)), k), reverse=True): ctl.pop(i)
    base, abl, con = mean_f1(real), mean_f1(keep), mean_f1(ctl)
    ours = mean_f1(G[S][:44])
    rows.append((S, len(real), k, base, abl, con, ours))
    print(f"seed{S:<3} n={len(real):<3} nonconv={k:<3} real {base:.4f} -> minus-nonconv {abl:.4f} "
          f"(random ctl {con:.4f})   ours {ours:.4f}", flush=True)

print("\n" + "="*78)
b = statistics.mean(r[3] for r in rows); a = statistics.mean(r[4] for r in rows)
c = statistics.mean(r[5] for r in rows); o = statistics.mean(r[6] for r in rows)
print(f"{len(rows)} threads,  {statistics.mean(r[2] for r in rows):.1f} non-conversational comments removed per thread "
      f"({100*statistics.mean(r[2]/r[1] for r in rows):.1f}%)")
print(f"  real as-is                 {b:.4f}")
print(f"  real minus non-conv        {a:.4f}   ({a-b:+.4f})")
print(f"  real minus same n random   {c:.4f}   ({c-b:+.4f})   <- control")
print(f"  OURS                       {o:.4f}   (gap to real {o-b:+.4f})")
net = (a - b) - (c - b)
print(f"\n  NET effect of the non-conversational comments: {net:+.5f}")
print(f"  = {100*net/(o-b):.0f}% of the gap")
print(f"  beats its random control in {sum(1 for r in rows if r[4] > r[5])}/{len(rows)} threads")
