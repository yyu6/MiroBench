#!/usr/bin/env python3
"""Open the black box. BERTScore is greedy token alignment, so ask it directly:
WHICH TOKENS carry the similarity, in ours versus in real?

Everything so far has been aggregate scalars. This extracts, for every pair,
the per-token max-similarity that BERTScore actually sums, and aggregates by
token type. The tokens whose contribution is inflated relative to real ARE the
defect, by construction -- no hypothesis of mine involved.
"""
import json, sys, re, itertools, statistics
from collections import defaultdict
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO/"scripts"/"evaluation"))
sys.path.insert(0, str(REPO/"bert_score-master"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments
import torch
from bert_score.utils import get_bert_embedding, get_model, get_tokenizer, sent_encode
from collections import defaultdict as dd

MODEL = "microsoft/deberta-xlarge-mnli"
tok = get_tokenizer(MODEL, use_fast=False)
model = get_model(MODEL, 40, False); model.eval()
idf = dd(lambda: 1.0); idf[tok.sep_token_id] = 0.0; idf[tok.cls_token_id] = 0.0

def embed(texts):
    embs, masks, pidf = get_bert_embedding(texts, model, tok, idf, batch_size=16, device="cpu")
    out = []
    for i, t in enumerate(texts):
        L = int(masks[i].sum().item())
        e = embs[i, :L]
        e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        ids = sent_encode(tok, t)[:L]
        out.append((e, [tok.decode([j]).strip().lower() for j in ids]))
    return out

def contributions(texts, cap_pairs=400):
    """For each pair, every token's greedy max-similarity -- what BERTScore sums."""
    E = embed(texts)
    pairs = list(itertools.combinations(range(len(texts)), 2))
    if len(pairs) > cap_pairs:
        step = len(pairs)/cap_pairs
        pairs = [pairs[int(i*step)] for i in range(cap_pairs)]
    acc, n = defaultdict(list), 0
    tot = []
    for i, j in pairs:
        (ea, ta), (eb, tb) = E[i], E[j]
        S = ea @ eb.T                       # [len_a, len_b]
        ma = S.max(dim=1).values            # each a-token's best partner
        mb = S.max(dim=0).values
        f = (ma.mean().item() + mb.mean().item())/2
        tot.append(f); n += 1
        for t, v in zip(ta, ma.tolist()):
            if t not in ("[cls]","[sep]",""): acc[t].append(v)
        for t, v in zip(tb, mb.tolist()):
            if t not in ("[cls]","[sep]",""): acc[t].append(v)
    return acc, statistics.mean(tot), n

pool = json.loads((REPO/"artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by = {int(p["seed_index"]): p for p in pool}
G = {}
for x in sorted((REPO/"artifacts/generalized_card/runs/v128_interaction_n10_20260828_v1/cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(x)
    for tid, cs in cbt.items(): G[int(tid.split("seed")[-1])] = [" ".join(c.text.split()) for c in cs]

R_acc, O_acc = defaultdict(list), defaultdict(list)
Rf, Of = [], []
for S in [2, 4, 7, 8]:
    p = by[S]
    real = [" ".join(c.text.split()) for c in (load_real_comments(REPO/"data/raw/discussions/camera_product"/p["source_product_dir"])[0].get(p["source_raw_post_id"]) or [])][:40]
    ours = G[S][:40]
    if len(real) < 12: continue
    a, fa, _ = contributions(real); b, fb, _ = contributions(ours)
    Rf.append(fa); Of.append(fb)
    for k, v in a.items(): R_acc[k].extend(v)
    for k, v in b.items(): O_acc[k].extend(v)
    print(f"seed{S}: real F1~{fa:.4f}  ours F1~{fb:.4f}", flush=True)

RT = sum(len(v) for v in R_acc.values()); OT = sum(len(v) for v in O_acc.values())
print(f"\nreal mean {statistics.mean(Rf):.4f}   ours mean {statistics.mean(Of):.4f}   "
      f"excess {statistics.mean(Of)-statistics.mean(Rf):+.4f}")
print(f"token slots: real {RT:,}  ours {OT:,}\n")

# EXCESS MASS: how much of ours' total score does each token type carry,
# minus how much it carries in real. Positive = this token is over-contributing.
rows = []
for t in set(list(R_acc)+list(O_acc)):
    ro = sum(O_acc.get(t, []))/OT if OT else 0
    rr = sum(R_acc.get(t, []))/RT if RT else 0
    rows.append((ro-rr, t, len(O_acc.get(t,[]))/OT*100 if OT else 0,
                 len(R_acc.get(t,[]))/RT*100 if RT else 0,
                 statistics.mean(O_acc[t]) if O_acc.get(t) else float('nan'),
                 statistics.mean(R_acc[t]) if R_acc.get(t) else float('nan')))
rows.sort(reverse=True)
print("tokens carrying the MOST EXCESS score share (ours minus real):")
print(f"  {'token':16}{'excess':>10}{'freq ours':>11}{'freq real':>11}{'mean sim ours':>15}{'real':>8}")
for d,t,fo,fr,so,sr in rows[:25]:
    print(f"  {t[:15]:16}{d*100:>+10.4f}{fo:>10.3f}%{fr:>10.3f}%{so:>15.3f}{sr:>8.3f}")
print(f"\n  top 25 tokens carry {sum(r[0] for r in rows[:25])*100:+.3f} of the "
      f"{(statistics.mean(Of)-statistics.mean(Rf))*100:+.3f} excess "
      f"({100*sum(r[0] for r in rows[:25])/(statistics.mean(Of)-statistics.mean(Rf)):.0f}%)")
