"""Does the generator produce distinct voices, or 326 labels on one voice?

one_voice_control.py: inside real threads, same-author pairs outscore
different-author pairs by +0.0137 after conditioning on conversational structure,
and the effect is LARGEST in the `different branch` stratum (+0.0141, n=476) and
near zero where the two comments sit closest (`same parent` -0.0008,
`ancestor/descendant` +0.0021). Topic proximity predicts the opposite ordering, so
the effect is authorial voice. Scaled to a whole thread the bound is +0.0133 --
112% of the +0.0119 generated gap.

The generator assigns 326 distinct author labels across 530 comments, MORE than
real's 260, and its same-author pair share is LOWER (0.0201 against 0.0299). So
the label structure is right. This asks whether the labels carry a voice.

The decisive comparison is generated's own same-vs-different delta against real's
+0.0137:
  - near real's       -> the personas are distinct and the floor is elsewhere
  - near zero         -> 326 labels on one voice, and the headroom is the whole
                         gap, reachable through the persona layer rather than
                         through any further surface symbol
"""
from __future__ import annotations
import json, statistics as st, sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_generated_comments  # noqa: E402
from score_thread_self_bertscore import load_bert_scorer  # noqa: E402

GATE = REPO / "artifacts/generalized_card/runs/v113_v112_gate_n10_20260826_v1"
STRATA = ("same parent", "ancestor/descendant", "same root branch", "different branch")

threads = {}
for d in sorted((GATE / "cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    for tid, cs in cbt.items():
        threads[int(tid.split("seed")[-1])] = cs

scorer, _, _, _, _, _ = load_bert_scorer(
    bert_score_path=REPO / "bert_score-master", model_type="microsoft/deberta-xlarge-mnli",
    num_layers=None, batch_size=8, device="auto", idf=False, idf_sents=[],
    rescale_with_baseline=False, local_files_only=True)

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

    return parent, {str(c.comment_id): chain(str(c.comment_id))[-1] for c in comments}, \
        {str(c.comment_id): set(chain(str(c.comment_id))) for c in comments}


for s in sorted(threads):
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

REAL = {"same parent": -0.0008, "ancestor/descendant": +0.0021,
        "same root branch": +0.0233, "different branch": +0.0141}
print(f"\n{'relation':<24}{'same-author':>22}{'different':>22}{'delta':>10}{'real delta':>12}")
weights, deltas = [], []
for name in STRATA:
    same, diff = buckets[name]["same"], buckets[name]["diff"]
    if len(same) < 10 or len(diff) < 10:
        print(f"{name:<24}{f'n={len(same)} (thin)':>22}{f'n={len(diff)}':>22}"
              f"{'-':>10}{REAL[name]:>+12.4f}")
        continue
    d = st.mean(same) - st.mean(diff)
    weights.append(len(same) + len(diff)); deltas.append(d)
    print(f"{name:<24}{f'{st.mean(same):.4f} (n={len(same)})':>22}"
          f"{f'{st.mean(diff):.4f} (n={len(diff)})':>22}{d:>+10.4f}{REAL[name]:>+12.4f}")

allsame = [v for n in STRATA for v in buckets[n]["same"]]
alldiff = [v for n in STRATA for v in buckets[n]["diff"]]
print(f"\n{'pooled (uncontrolled)':<24}{f'{st.mean(allsame):.4f} (n={len(allsame)})':>22}"
      f"{f'{st.mean(alldiff):.4f} (n={len(alldiff)})':>22}"
      f"{st.mean(allsame)-st.mean(alldiff):>+10.4f}{0.0177:>+12.4f}")
if weights:
    adj = sum(w * d for w, d in zip(weights, deltas)) / sum(weights)
    print(f"{'stratum-weighted':<24}{'':>22}{'':>22}{adj:>+10.4f}{0.0137:>+12.4f}")
    print(f"\ngenerated author voice strength as a fraction of real's: {adj/0.0137:.2f}")
    print(f"headroom if the personas carried real's voice separation: "
          f"{(0.0137-adj)*(1-len(allsame)/(len(allsame)+len(alldiff))):+.4f}"
          f"  = {100*(0.0137-adj)*(1-len(allsame)/(len(allsame)+len(alldiff)))/0.0119:.0f}%"
          f" of the +0.0119 gap")
