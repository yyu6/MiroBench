"""Does the evidence_mode floor survive holding the topic labels fixed too?

discourse_function_floor.py: over cross-branch pairs, pairs sharing an
`evidence_mode` score 0.5193 against 0.4965 for pairs that differ, delta +0.0228,
and 41.4% of cross-branch pairs currently share one because
`technical_or_policy_reasoning` alone is 50% of all slots. Floor = 0.414 x 0.0228
= +0.00941 = 79.1% of the +0.0119 gap.

Branch membership was controlled, but two comments can sit in different branches
and still be about the same sub-topic, and the Planner may pick the same evidence
mode for them BECAUSE of that. This holds the Planner's own topic labels fixed --
`claim_family` and `local_topic` -- and re-measures inside the cell where both
differ. If the effect survives there, the shared label is doing the work.

Also reports the ACHIEVABLE floor rather than the raw one: `evidence_mode` has six
values, so the same-label share cannot go to zero. Under a uniform assignment it
bottoms out at sum(p^2) = 1/6, and against the measured distribution of a target
it bottoms out higher.
"""
from __future__ import annotations
import json, statistics as st, sys
from collections import Counter
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_generated_comments  # noqa: E402
from score_thread_self_bertscore import load_bert_scorer  # noqa: E402

GATE = REPO / "artifacts/generalized_card/runs/v113_v112_gate_n10_20260826_v1"
GAP = 0.0119

meta = {}
for d in sorted((GATE / "cleaned").glob("run_*_sampled_reddit")):
    for post in json.loads((d / "discussion.json").read_text())["posts"]:
        for rec in post.get("generation_records") or []:
            cid = str((rec.get("comment") or {}).get("comment_id", ""))
            t = rec.get("task") or {}
            if cid:
                meta[cid] = {k: str(t.get(k) or "") for k in
                             ("evidence_mode", "claim_family", "local_topic", "claim_key")}

threads = {}
for d in sorted((GATE / "cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    for tid, cs in cbt.items():
        threads[int(tid.split("seed")[-1])] = [c for c in cs if str(c.comment_id) in meta]

dist = Counter(meta[str(c.comment_id)]["evidence_mode"] for v in threads.values() for c in v)
total = sum(dist.values())
print("evidence_mode distribution over all slots:")
for k, n in dist.most_common():
    print(f"   {k:<32}{n:>4}  {n/total:.3f}")
print(f"\ncollision share sum(p^2) today          : {sum((n/total)**2 for n in dist.values()):.4f}")
print(f"collision share if uniform over {len(dist)} values: {1/len(dist):.4f}")

scorer, _, _, _, _, _ = load_bert_scorer(
    bert_score_path=REPO / "bert_score-master", model_type="microsoft/deberta-xlarge-mnli",
    num_layers=None, batch_size=8, device="auto", idf=False, idf_sents=[],
    rescale_with_baseline=False, local_files_only=True)


def structure(comments):
    by_id = {str(c.comment_id): c for c in comments}
    parent = {str(c.comment_id): str(c.parent_id or "") for c in comments}

    def chain(cid):
        seen, out = set(), []
        while cid in by_id and cid not in seen:
            seen.add(cid); out.append(cid); cid = parent.get(cid, "")
        return out
    return {str(c.comment_id): chain(str(c.comment_id))[-1] for c in comments}


cells = {}
for s in sorted(threads):
    cs = threads[s]
    if len(cs) < 8:
        continue
    root = structure(cs)
    cand, ref, tags = [], [], []
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            a, b = cs[i], cs[j]
            ma, mb = meta[str(a.comment_id)], meta[str(b.comment_id)]
            if root.get(str(a.comment_id)) == root.get(str(b.comment_id)):
                continue
            same_topic = (ma["claim_family"] == mb["claim_family"]
                          or ma["local_topic"] == mb["local_topic"])
            cand.append(a.text); ref.append(b.text)
            tags.append((ma["evidence_mode"] == mb["evidence_mode"], same_topic))
    if not cand:
        continue
    _, _, f1 = scorer.score(cand, ref, batch_size=8)
    for (same_ev, same_topic), value in zip(tags, f1):
        cells.setdefault((same_ev, same_topic), []).append(float(value))
    print(f"seed {s}: {len(cand)} cross-branch pairs")

print(f"\n{'topic labels':<22}{'same evidence_mode':>26}{'different':>26}{'delta':>10}")
for same_topic, label in ((True, "shared"), (False, "both differ")):
    a = cells.get((True, same_topic), [])
    b = cells.get((False, same_topic), [])
    if len(a) < 30 or len(b) < 30:
        print(f"{label:<22}{f'n={len(a)} (thin)':>26}{f'n={len(b)}':>26}{'-':>10}")
        continue
    print(f"{label:<22}{f'{st.mean(a):.4f} (n={len(a)})':>26}"
          f"{f'{st.mean(b):.4f} (n={len(b)})':>26}{st.mean(a)-st.mean(b):>+10.4f}")

a = cells.get((True, False), []); b = cells.get((False, False), [])
if len(a) >= 30 and len(b) >= 30:
    delta = st.mean(a) - st.mean(b)
    today = sum((n / total) ** 2 for n in dist.values())
    floor_now = today * delta
    for name, target in (("uniform over 6", 1 / len(dist)),
                         ("no value above 0.30", 0.30**2 + 5 * (0.14**2))):
        gain = (today - target) * delta
        print(f"\n  ACHIEVABLE, topic held apart, {name}:")
        print(f"    collision {today:.4f} -> {target:.4f}, delta {delta:+.4f}")
        print(f"    floor removed {gain:+.5f} = {100*gain/GAP:.1f}% of the +{GAP:.4f} gap")
    print(f"\n  (raw, un-achievable 'collisions to zero' bound was {100*floor_now/GAP:.1f}%)")
