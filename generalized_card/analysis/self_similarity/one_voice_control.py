"""Is the same-author effect authorial voice, or is it just the same subthread?

one_voice_floor.py: inside real threads, pairs written by the SAME author score
0.5089 against 0.4912 for pairs by different authors, delta +0.0177, positive in
7 of 8 threads. Scaled to a whole thread that bound is +0.0172 -- 144% of the
+0.0119 generated gap -- which would make it far larger than URLs (67%) or
parentheticals (26%) and would explain both the 19% with no identified channel and
why the link arm delivered only 0.34 of its real-side value.

**That number is uncontrolled and must not be used as it stands.** On Reddit a
person's two comments usually sit in the same subthread, answering the same
question. The effect could be entirely topical proximity with authorship as a
proxy for it.

This conditions on conversational structure. Every pair is labelled by its
relation -- same parent, ancestor/descendant, same root branch but neither, or
different root branches entirely -- and same-author is compared with
different-author INSIDE each stratum. The decisive cell is `different branch`:
one person writing in two separate parts of the thread is the same voice on
different topics, which is what a generated thread is by construction.
"""
from __future__ import annotations
import json, statistics as st, sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments  # noqa: E402
from score_thread_self_bertscore import load_bert_scorer  # noqa: E402

SEEDS = range(2, 12)
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}

threads, cache = {}, {}
for s in SEEDS:
    p = by_seed[s]
    d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if d not in cache:
        cache[d] = load_real_comments(d)[0]
    threads[s] = cache[d].get(p["source_raw_post_id"]) or []

scorer, _, _, _, _, _ = load_bert_scorer(
    bert_score_path=REPO / "bert_score-master", model_type="microsoft/deberta-xlarge-mnli",
    num_layers=None, batch_size=8, device="auto", idf=False, idf_sents=[],
    rescale_with_baseline=False, local_files_only=True)

STRATA = ("same parent", "ancestor/descendant", "same root branch", "different branch")
buckets = {name: {"same": [], "diff": []} for name in STRATA}


def structure(comments):
    by_id = {str(c.comment_id): c for c in comments}
    parent = {str(c.comment_id): str(c.parent_id or "") for c in comments}

    def chain(cid):
        seen, out = set(), []
        while cid in by_id and cid not in seen:
            seen.add(cid)
            out.append(cid)
            cid = parent.get(cid, "")
        return out

    root = {str(c.comment_id): chain(str(c.comment_id))[-1] for c in comments}
    anc = {str(c.comment_id): set(chain(str(c.comment_id))) for c in comments}
    return parent, root, anc


for s in SEEDS:
    cs = threads[s]
    if len(cs) < 6:
        continue
    parent, root, anc = structure(cs)
    cand, ref, meta = [], [], []
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            a, b = cs[i], cs[j]
            ida, idb = str(a.comment_id), str(b.comment_id)
            if parent.get(ida) and parent.get(ida) == parent.get(idb):
                rel = "same parent"
            elif idb in anc.get(ida, ()) or ida in anc.get(idb, ()):
                rel = "ancestor/descendant"
            elif root.get(ida) == root.get(idb):
                rel = "same root branch"
            else:
                rel = "different branch"
            cand.append(a.text); ref.append(b.text)
            meta.append((rel, bool(a.author) and a.author == b.author))
    _, _, f1 = scorer.score(cand, ref, batch_size=8)
    for (rel, same), value in zip(meta, f1):
        buckets[rel]["same" if same else "diff"].append(float(value))
    print(f"seed {s}: {len(cs)} comments, {len(meta)} pairs")

print(f"\n{'relation':<24}{'same-author':>22}{'different':>22}{'delta':>10}")
for name in STRATA:
    same, diff = buckets[name]["same"], buckets[name]["diff"]
    if len(same) < 10 or len(diff) < 10:
        print(f"{name:<24}{f'n={len(same)} (thin)':>22}{f'n={len(diff)}':>22}{'-':>10}")
        continue
    d = st.mean(same) - st.mean(diff)
    print(f"{name:<24}{f'{st.mean(same):.4f} (n={len(same)})':>22}"
          f"{f'{st.mean(diff):.4f} (n={len(diff)})':>22}{d:>+10.4f}")

allsame = [v for n in STRATA for v in buckets[n]["same"]]
alldiff = [v for n in STRATA for v in buckets[n]["diff"]]
print(f"\n{'pooled (uncontrolled)':<24}{f'{st.mean(allsame):.4f} (n={len(allsame)})':>22}"
      f"{f'{st.mean(alldiff):.4f} (n={len(alldiff)})':>22}"
      f"{st.mean(allsame)-st.mean(alldiff):>+10.4f}")

# stratum-weighted delta: the same comparison, holding conversational structure fixed
weights, deltas = [], []
for name in STRATA:
    same, diff = buckets[name]["same"], buckets[name]["diff"]
    if len(same) < 10 or len(diff) < 10:
        continue
    weights.append(len(same) + len(diff))
    deltas.append(st.mean(same) - st.mean(diff))
if weights:
    adj = sum(w * d for w, d in zip(weights, deltas)) / sum(weights)
    print(f"{'stratum-weighted':<24}{'':>22}{'':>22}{adj:>+10.4f}")
    share = len(allsame) / (len(allsame) + len(alldiff))
    print(f"\nsame-author share of pairs: {share:.4f}")
    print(f"BOUND on the one-voice floor, controlled: {(1-share)*adj:+.4f}"
          f"  = {100*(1-share)*adj/0.0119:.0f}% of the +0.0119 gap")
    print(f"  (uncontrolled it was +0.0172 = 144%)")
