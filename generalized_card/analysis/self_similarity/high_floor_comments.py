"""Which comments RAISE the thread's self_bertscore, and what do they have in common?

Three sessions of work guessed a feature and then tested it: URLs, parentheticals,
digits, hapax, comma-joining, markdown, author identity, evidence_mode. Some hit,
most missed. real_thread_correlates.py then showed why the guessing was hard --
the generator sits within 0.3 sd of its matched real threads on EVERY cached
thread-level metric, so the driver is not an aggregate. It is per comment.

So stop guessing. self_bertscore is the mean over pairs, and it decomposes exactly:
each comment's leverage is the mean F1 of the pairs it appears in. Rank comments by
that, take the extreme deciles, and read what separates them -- on BOTH sides,
because the question is not "what raises it" but "what does the generator have more
of than real does".

Free. Uses the gate run and its matched real threads.
"""
from __future__ import annotations
import json, re, statistics as st, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_self_bertscore import load_bert_scorer  # noqa: E402

GATE = REPO / "artifacts/generalized_card/runs/v113_v112_gate_n10_20260826_v1"
SEEDS = range(2, 12)
FIELDS = ("evidence_mode", "comment_function", "payload_type", "utterance_mode",
          "speaker_role", "surface_texture", "tone_target", "stance")
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}

meta = {}
for d in sorted((GATE / "cleaned").glob("run_*_sampled_reddit")):
    for post in json.loads((d / "discussion.json").read_text())["posts"]:
        for rec in post.get("generation_records") or []:
            cid = str((rec.get("comment") or {}).get("comment_id", ""))
            t = rec.get("task") or {}
            if cid:
                meta[cid] = {f: str(t.get(f) or "") for f in FIELDS}

gen, real, cache = {}, {}, {}
for d in sorted((GATE / "cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    for tid, cs in cbt.items():
        gen[int(tid.split("seed")[-1])] = cs
for s in SEEDS:
    p = by_seed[s]
    dd = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if dd not in cache:
        cache[dd] = load_real_comments(dd)[0]
    real[s] = cache[dd].get(p["source_raw_post_id"]) or []

scorer, _, _, _, _, _ = load_bert_scorer(
    bert_score_path=REPO / "bert_score-master", model_type="microsoft/deberta-xlarge-mnli",
    num_layers=None, batch_size=8, device="auto", idf=False, idf_sents=[],
    rescale_with_baseline=False, local_files_only=True)


def leverage(comments):
    """-> [(index, mean F1 over the pairs this comment appears in)], plus the thread mean."""
    texts = [c.text for c in comments]
    n = len(texts)
    if n < 4:
        return [], None
    c, r, idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            c.append(texts[i]); r.append(texts[j]); idx.append((i, j))
    _, _, f1 = scorer.score(c, r, batch_size=8)
    vals = [float(x) for x in f1]
    per = [[] for _ in range(n)]
    for (i, j), v in zip(idx, vals):
        per[i].append(v); per[j].append(v)
    return [(i, st.mean(p)) for i, p in enumerate(per) if p], st.mean(vals)


WORD = re.compile(r"[A-Za-z']+")
out = {}
for label, threads in (("real", real), ("gate", gen)):
    ranked = []
    for s in sorted(threads):
        lev, mean = leverage(threads[s])
        if not lev:
            continue
        # centre within the thread so threads of different levels are comparable
        mu = st.mean(v for _, v in lev)
        for i, v in lev:
            ranked.append((v - mu, threads[s][i]))
    ranked.sort(key=lambda t: t[0])
    out[label] = ranked
    print(f"{label}: {len(ranked)} comments ranked by within-thread leverage")

print(f"\n{'side':<7}{'decile':<10}{'leverage':>11}{'words':>8}{'sentences':>11}{'1st-person':>12}{'question':>10}")
for label in ("real", "gate"):
    ranked = out[label]
    k = max(1, len(ranked) // 10)
    for name, grp in (("bottom 10%", ranked[:k]), ("top 10%", ranked[-k:])):
        w = [len(c.text.split()) for _, c in grp]
        sent = [len(re.split(r"(?<=[.!?])\s+|\n+", c.text.strip())) for _, c in grp]
        fp = sum(1 for _, c in grp if re.search(r"\b(I|my|I'm|I've)\b", c.text)) / len(grp)
        q = sum(1 for _, c in grp if "?" in c.text) / len(grp)
        print(f"{label:<7}{name:<10}{st.mean(v for v, _ in grp):>+11.4f}{st.mean(w):>8.1f}"
              f"{st.mean(sent):>11.2f}{fp:>12.2f}{q:>10.2f}")

print("\n== generated top-decile vs bottom-decile, by Planner field ==")
ranked = out["gate"]
k = max(1, len(ranked) // 10)
lo = [c for _, c in ranked[:k]]
hi = [c for _, c in ranked[-k:]]
for f in FIELDS:
    ch = Counter(meta[str(c.comment_id)][f] for c in hi if str(c.comment_id) in meta)
    cl = Counter(meta[str(c.comment_id)][f] for c in lo if str(c.comment_id) in meta)
    th, tl = sum(ch.values()), sum(cl.values())
    if not th or not tl:
        continue
    keys = sorted(set(ch) | set(cl), key=lambda x: -(ch[x] / th - cl[x] / tl))
    line = ", ".join(f"{x}:{ch[x]/th:.2f}v{cl[x]/tl:.2f}" for x in keys[:3] if ch[x] or cl[x])
    print(f"  {f:<20} {line}")

print("\n-- generated HIGH-leverage comments (these raise self_bertscore most) --")
for v, c in out["gate"][-8:]:
    m = meta.get(str(c.comment_id), {})
    print(f"   [{v:+.4f}] ev={m.get('evidence_mode','?')[:26]:<26} {c.text[:105]}")
print("\n-- generated LOW-leverage comments --")
for v, c in out["gate"][:8]:
    m = meta.get(str(c.comment_id), {})
    print(f"   [{v:+.4f}] ev={m.get('evidence_mode','?')[:26]:<26} {c.text[:105]}")
print("\n-- REAL low-leverage comments (what the generator does not produce) --")
for v, c in out["real"][:10]:
    print(f"   [{v:+.4f}] {c.text[:120]}")
