"""The reviser's own diagnosis, tested on the Planner/Writer path.

`scripts/postprocess_selfbert_tail_repair.py` is the reviser stage that moved
`self_bertscore` on the v6 revision runs. Its prompt states the mechanism outright:

    "lower this ONE thread's Self-BERTScore by changing discourse function, not by
     random synonym rewriting"
    "Choose comments that make the thread feel like a set of same-shaped generated
     replies: polished advice, generic thanks, repeated narrow questions, or
     complete helpful explanations."
    "Change the comment job/evidence source. Do not merely paraphrase."

That is not a lexical claim, and it matches what `one_voice_generated.py` found
from the other direction: the generator's floor is not vocabulary, it is that
every comment is the same KIND of contribution.

Unlike the reviser, this needs nothing the Planner does not already own.
`comment_function`, `payload_type`, `evidence_mode`, `utterance_mode` and
`speaker_role` are fields on every task. (The reviser also puts matched real
comments in its prompt, which ORIENTATION.md s4 forbids the Writer to see -- so
the reviser cannot be used for the paper even if it works.)

Measured here, with the same branch-controlled design the author test used:
  1. how concentrated the generated discourse-function distribution is,
  2. whether pairs sharing a function score higher than pairs that do not,
  3. the resulting floor, and how much of the +0.0119 gap it bounds.
"""
from __future__ import annotations
import json, math, statistics as st, sys
from collections import Counter
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_generated_comments  # noqa: E402
from score_thread_self_bertscore import load_bert_scorer  # noqa: E402

GATE = REPO / "artifacts/generalized_card/runs/v113_v112_gate_n10_20260826_v1"
FIELDS = ("comment_function", "payload_type", "evidence_mode", "utterance_mode",
          "speaker_role", "surface_texture")

meta = {}
for d in sorted((GATE / "cleaned").glob("run_*_sampled_reddit")):
    for post in json.loads((d / "discussion.json").read_text())["posts"]:
        for rec in post.get("generation_records") or []:
            cid = str((rec.get("comment") or {}).get("comment_id", ""))
            t = rec.get("task") or {}
            if cid:
                meta[cid] = {f: str(t.get(f) or "") for f in FIELDS}

threads = {}
for d in sorted((GATE / "cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    for tid, cs in cbt.items():
        threads[int(tid.split("seed")[-1])] = [c for c in cs if str(c.comment_id) in meta]

print("== how concentrated is each Planner field? ==")
print(f"{'field':<20}{'values':>8}{'entropy':>10}{'max H':>8}{'normalised':>12}   top three")
for f in FIELDS:
    counts = Counter(meta[str(c.comment_id)][f] for v in threads.values() for c in v)
    counts.pop("", None)
    total = sum(counts.values())
    h = -sum((n / total) * math.log2(n / total) for n in counts.values())
    hmax = math.log2(len(counts)) if len(counts) > 1 else 1.0
    top = ", ".join(f"{k}={n/total:.2f}" for k, n in counts.most_common(3))
    print(f"{f:<20}{len(counts):>8}{h:>10.2f}{hmax:>8.2f}{h/hmax:>12.3f}   {top}")

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
    return parent, {str(c.comment_id): chain(str(c.comment_id))[-1] for c in comments}


rows = {f: {"same": [], "diff": []} for f in FIELDS}
for s in sorted(threads):
    cs = threads[s]
    if len(cs) < 8:
        continue
    parent, root = structure(cs)
    cand, ref, tags = [], [], []
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            a, b = cs[i], cs[j]
            ida, idb = str(a.comment_id), str(b.comment_id)
            # branch control, matching one_voice_control.py's decisive stratum
            if root.get(ida) == root.get(idb):
                continue
            cand.append(a.text); ref.append(b.text)
            tags.append(tuple(meta[ida][f] == meta[idb][f] and meta[ida][f] != ""
                              for f in FIELDS))
    if not cand:
        continue
    _, _, f1 = scorer.score(cand, ref, batch_size=8)
    for tag, value in zip(tags, f1):
        for f, same in zip(FIELDS, tag):
            rows[f]["same" if same else "diff"].append(float(value))
    print(f"seed {s}: {len(cand)} cross-branch pairs")

print(f"\n== cross-branch pairs only (topic held apart), by shared Planner field ==")
print(f"{'field':<20}{'shares it':>22}{'differs':>22}{'delta':>10}{'share':>9}")
best = []
for f in FIELDS:
    same, diff = rows[f]["same"], rows[f]["diff"]
    if len(same) < 30 or len(diff) < 30:
        print(f"{f:<20}{f'n={len(same)} (thin)':>22}{f'n={len(diff)}':>22}{'-':>10}{'-':>9}")
        continue
    d = st.mean(same) - st.mean(diff)
    share = len(same) / (len(same) + len(diff))
    best.append((f, d, share))
    print(f"{f:<20}{f'{st.mean(same):.4f} (n={len(same)})':>22}"
          f"{f'{st.mean(diff):.4f} (n={len(diff)})':>22}{d:>+10.4f}{share:>9.3f}")

print("\n== what the reviser's fix is worth, as a bound ==")
print("The reviser rewrites a comment into a DIFFERENT discourse function. For each")
print("field, moving a same-function pair to different-function is worth `delta`,")
print("and `share` of cross-branch pairs currently share that field.")
for f, d, share in sorted(best, key=lambda x: -x[1] * x[2]):
    print(f"  {f:<20} floor = share x delta = {share * d:+.5f}"
          f"   = {100 * share * d / 0.0119:>5.1f}% of the +0.0119 gap")
