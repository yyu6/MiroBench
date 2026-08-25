"""Additive decomposition of self_bleu_4 into a length-only floor and a
content-overlap excess.  Reproduces the shipped artifact first (E6)."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))

from score_thread_self_bleu import (  # noqa: E402
    tokenize, pairwise_self_bleu_for_order, sentence_bleu,
    clipped_ngram_overlap, ngram_counts, closest_reference_length,
)
from score_thread_semantic_uniformity import (  # noqa: E402
    load_generated_comments, load_real_comments,
)

RUN = REPO / "artifacts/generalized_card/runs/generalized_card_camera_gpt54_paper_20260825_v1"

def floor_bleu(hyp, ref, max_order=4):
    """BLEU the pair would score if clipped overlap were zero at every order.

    Mirrors sentence_bleu exactly, except overlap is forced to 0.  Orders where
    the hypothesis is shorter than n keep the scorer's (0,0) -> p=1.0 branch.
    """
    if not hyp or not ref:
        return 0.0
    logs = 0.0
    for order in range(1, max_order + 1):
        total = len(ngram_counts(hyp, order))
        total = sum(ngram_counts(hyp, order).values())
        p = (0.0 + 1.0) / (total + 1.0)
        logs += math.log(max(p, 1e-12))
    crl = closest_reference_length(len(hyp), [len(ref)])
    bp = 1.0 if len(hyp) > crl else math.exp(1.0 - crl / max(1, len(hyp)))
    return float(bp * math.exp(logs / max_order))

def sym(fn, a, b):
    return (fn(a, b) + fn(b, a)) / 2.0

def thread_components(tokenized):
    if len(tokenized) < 2:
        return 0.0, 0.0
    act, flo = [], []
    for i in range(len(tokenized)):
        for j in range(i + 1, len(tokenized)):
            a, b = tokenized[i], tokenized[j]
            act.append((sentence_bleu(a, [b], 4) + sentence_bleu(b, [a], 4)) / 2.0)
            flo.append(sym(floor_bleu, a, b))
    return sum(act) / len(act), sum(flo) / len(flo)

# ---- generated ----
gen_rows = list(csv.DictReader(open(RUN / "evaluation/revised_generated_thread_scores.csv")))
seed_pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(p["seed_index"]): p for p in seed_pool}

gen_by_dir = {}
for d in sorted((RUN / "cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    gen_by_dir[d.name] = cbt

gen_text = {}
for name, cbt in gen_by_dir.items():
    for tid, comments in cbt.items():
        gen_text[tid] = comments

rows = []
for r in gen_rows:
    seed = int(float(r["seed_index"]))
    sim_dir = Path(r["_source_sim_dir"]).name
    tid = None
    for cand, comments in gen_by_dir[sim_dir].items():
        if cand.endswith(f"seed{seed:03d}"):
            tid = cand
            break
    assert tid, (sim_dir, seed)
    rows.append((seed, tid, gen_by_dir[sim_dir][tid], float(r["self_bleu_4"]), int(float(r["comment_count"]))))

print(f"generated threads matched: {len(rows)}")
maxdiff = 0.0
gen_out = []
for seed, tid, comments, csv_sb4, ccount in rows:
    toks = [tokenize(c.text) for c in comments]
    assert len(toks) == ccount, (tid, len(toks), ccount)
    act, flo = thread_components(toks)
    maxdiff = max(maxdiff, abs(act - csv_sb4))
    gen_out.append({"seed": seed, "thread": tid, "n": ccount, "sb4": act,
                    "floor": flo, "excess": act - flo,
                    "mean_tok": sum(len(t) for t in toks) / len(toks)})
print(f"[E6] generated self_bleu_4 max |reproduced - shipped| = {maxdiff:.3e}")
json.dump(gen_out, open("/private/tmp/claude-501/-Users-yaoningyu-Desktop-UIUC-GEO/1f41c5a0-3c0b-415c-9b23-9fb381e5c727/scratchpad/gen_sb4.json", "w"), indent=1)
