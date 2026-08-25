"""Is 'cut the link comment to its link sentence' a URL effect or a length effect?

url_shape_gap.py: cutting the gate's 23 link-carrying comments down to the sentence
holding the link moves self_bertscore 0.5061 -> 0.5013 and takes closure of the
no-link gap from 24.4% to 54.7%, past the 42% Holm needs at N=150. But that
ablation deletes ~60 words from 23 comments, and a short comment lowers pairwise
F1 against every long one regardless of what it contains. Without a control the
length artifact reads as a URL finding -- the same failure mode the self_bleu floor
decomposition exists to prevent.

Three controls, all on the gate's own output:
  1. cut 23 RANDOM non-link comments to one sentence (length effect alone)
  2. cut the link comments to one sentence and DELETE the url (shape without url)
  3. real's own length spread, since real sd is 74.8 words against generated 58.2

If (1) reproduces most of the move, the finding is 'threads need short comments',
which is a length-variance arm and not a link arm at all.
"""
from __future__ import annotations
import json, random, re, statistics as st, sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "generalized_card"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_self_bertscore import load_bert_scorer  # noqa: E402
from generalized_card.reference_link import extract_urls, URL_RE  # noqa: E402

GATE = "v113_v112_gate_n10_20260826_v1"
SEEDS = range(2, 12)
SENT = re.compile(r"(?<=[.!?])\s+|\n+")
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}

real_threads, cache = {}, {}
for s in SEEDS:
    p = by_seed[s]
    d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if d not in cache:
        cache[d] = load_real_comments(d)[0]
    real_threads[s] = [c.text for c in (cache[d].get(p["source_raw_post_id"]) or [])]
gen_threads = {}
for d in sorted((REPO / "artifacts/generalized_card/runs" / GATE / "cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    for tid, cs in cbt.items():
        gen_threads[int(tid.split("seed")[-1])] = [c.text for c in cs]

scorer, _, _, _, _, _ = load_bert_scorer(
    bert_score_path=REPO / "bert_score-master", model_type="microsoft/deberta-xlarge-mnli",
    num_layers=None, batch_size=8, device="auto", idf=False, idf_sents=[],
    rescale_with_baseline=False, local_files_only=True)


def tf1(texts):
    c, r = [], []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            c.append(texts[i]); r.append(texts[j])
    _, _, f1 = scorer.score(c, r, batch_size=8)
    return float(np.mean([float(x) for x in f1]))


def parts(t):
    return [s.strip() for s in SENT.split(t) if s.strip()]


def cut_to_link(t):
    if not extract_urls(t):
        return t
    keep = [s for s in parts(t) if extract_urls(s)]
    return " ".join(keep) if keep else t


def cut_to_link_nourl(t):
    return re.sub(r"\s+", " ", URL_RE.sub(" ", cut_to_link(t))).strip() if extract_urls(t) else t


def strip_url(t):
    return re.sub(r"\s+", " ", URL_RE.sub(" ", t)).strip()


def cut_random(texts, rng):
    """Cut as many random NON-link comments to one sentence as this thread has link ones."""
    idx = [i for i, t in enumerate(texts) if extract_urls(t)]
    pool_i = [i for i, t in enumerate(texts) if not extract_urls(t) and len(parts(t)) > 1]
    pick = set(rng.sample(pool_i, min(len(idx), len(pool_i))))
    return [parts(t)[0] if i in pick else t for i, t in enumerate(texts)]


real_base = st.mean(tf1(real_threads[s]) for s in SEEDS)
variants = {
    "gate, as shipped": lambda ts, rng: ts,
    "gate, all links stripped": lambda ts, rng: [strip_url(t) for t in ts],
    "gate, link comment -> its link sentence": lambda ts, rng: [cut_to_link(t) for t in ts],
    "CONTROL: same count of random non-link cuts": cut_random,
    "gate, link sentence but url deleted": lambda ts, rng: [cut_to_link_nourl(t) for t in ts],
}
print(f"{'real baseline':<46}{real_base:>10.4f}\n")
print(f"{'variant':<46}{'score':>10}{'bias':>9}{'vs shipped':>12}{'closure':>10}")
res = {}
for name, fn in variants.items():
    rng = random.Random(19)
    res[name] = st.mean(tf1(fn(gen_threads[s], rng)) for s in SEEDS)
nolink = res["gate, all links stripped"] - real_base
for name, v in res.items():
    d = v - real_base
    print(f"{name:<46}{v:>10.4f}{100*d/real_base:>8.2f}%"
          f"{v-res['gate, as shipped']:>+12.4f}{100*(nolink-d)/nolink:>9.1f}%")

print("\n== the length-variance channel, measured separately ==")
for label, thr in (("real", real_threads), ("gate", gen_threads)):
    ws = [len(t.split()) for cs in thr.values() for t in cs]
    print(f"  {label:<6} mean {st.mean(ws):>6.1f}  sd {st.pstdev(ws):>6.1f}  "
          f"max {max(ws):>4}  p99 {sorted(ws)[int(0.99*len(ws))]:>4}  "
          f"within-thread sd {st.mean(st.pstdev([len(t.split()) for t in cs]) for cs in thr.values()):>6.1f}")

print("\n  ablation: truncate REAL comments to the gate's max observed length")
cap = max(len(t.split()) for cs in gen_threads.values() for t in cs)
capped = st.mean(tf1([" ".join(t.split()[:cap]) for t in real_threads[s]]) for s in SEEDS)
print(f"  real capped at {cap} words: {capped:.4f}  ({capped-real_base:+.4f}, "
      f"{100*(capped-real_base)/(res['gate, as shipped']-real_base):.0f}% of the shipped gap)")
