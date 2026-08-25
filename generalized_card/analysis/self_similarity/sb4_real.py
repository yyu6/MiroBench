from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
SP = (Path(__file__).resolve().parent / "_cache")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_self_bleu import tokenize, sentence_bleu, ngram_counts, closest_reference_length
from score_thread_semantic_uniformity import load_real_comments

def floor_bleu(hyp, ref, max_order=4):
    if not hyp or not ref: return 0.0
    logs = 0.0
    for order in range(1, max_order + 1):
        total = sum(ngram_counts(hyp, order).values())
        logs += math.log(max(1.0 / (total + 1.0), 1e-12))
    crl = closest_reference_length(len(hyp), [len(ref)])
    bp = 1.0 if len(hyp) > crl else math.exp(1.0 - crl / max(1, len(hyp)))
    return float(bp * math.exp(logs / max_order))

def components(toks):
    if len(toks) < 2: return 0.0, 0.0
    act, flo = [], []
    for i in range(len(toks)):
        for j in range(i+1, len(toks)):
            a, b = toks[i], toks[j]
            act.append((sentence_bleu(a,[b],4) + sentence_bleu(b,[a],4))/2.0)
            flo.append((floor_bleu(a,b) + floor_bleu(b,a))/2.0)
    return sum(act)/len(act), sum(flo)/len(flo)

pool = json.load(open(REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(p["seed_index"]): p for p in pool}
real_csv = {r["thread_id"]: r for r in csv.DictReader(open(REPO/"artifacts/baselines/camera_product_gpt5mini/real/thread_scores.csv"))}
gen = json.load(open(SP/"gen_sb4.json"))

cache = {}
out, maxdiff = [], 0.0
for g in gen:
    p = by_seed[g["seed"]]
    d = REPO/"data/raw/discussions/camera_product"/p["source_product_dir"]
    if d not in cache:
        cache[d] = load_real_comments(d)[0]
    comments = cache[d].get(p["source_raw_post_id"]) or []
    toks = [tokenize(c.text) for c in comments]
    act, flo = components(toks)
    ref = real_csv.get(p["source_raw_post_id"])
    if ref:
        maxdiff = max(maxdiff, abs(act - float(ref["self_bleu_4"])))
    out.append({"seed": g["seed"], "rid": p["source_raw_post_id"], "n": len(toks),
                "sb4": act, "floor": flo, "excess": act - flo,
                "mean_tok": (sum(len(t) for t in toks)/len(toks)) if toks else 0.0,
                "csv_sb4": float(ref["self_bleu_4"]) if ref else None,
                "csv_n": int(float(ref["comment_count"])) if ref else None})
print(f"[E6] real self_bleu_4 max |reproduced - baseline csv| = {maxdiff:.3e}")
print("comment-count agreement:", all(o["n"]==o["csv_n"] for o in out if o["csv_n"] is not None))
json.dump(out, open(SP/"real_sb4.json","w"), indent=1)
