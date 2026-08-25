"""Is the self_bertscore channel URLs specifically, or once-only tokens generally?

surface_class_prevalence.py: on the same ten seeds, generated carries 2802 types
and 1174 hapax against real's 3993 and 2040 -- 43% fewer once-only tokens on a
comparable token count. BERTScore is greedy alignment with no idf, so a token
with no counterpart anywhere else in the thread drags every pair it appears in.
URLs are the most visible instance of that class; parentheticals (+3.51% of
tokens) and digit runs (+2.49%) are larger by token share and were never tested.

Ablates each class out of REAL text and rescores with the shipped scorer, so a
rise is that class's contribution to real's lower score (its share of the gap).

CONTROL, because removing text also shortens it: every ablation is paired with a
random-token-removal control matched on the exact number of tokens removed per
comment. The reported "net" subtracts the control. Without it a length artifact
reads as a semantic one -- the same mistake the self_bleu floor decomposition
was built to avoid.
"""
from __future__ import annotations
import json, random, re, statistics as st, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "generalized_card"))
from score_thread_semantic_uniformity import load_real_comments  # noqa: E402
from score_thread_self_bertscore import load_bert_scorer  # noqa: E402
from score_thread_self_bleu import tokenize  # noqa: E402
from generalized_card.reference_link import URL_RE  # noqa: E402

SEEDS = range(2, 12)
GEN_BIAS = None  # filled from the gate run's shipped scores below
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}

threads, cache = {}, {}
for s in SEEDS:
    p = by_seed[s]
    d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if d not in cache:
        cache[d] = load_real_comments(d)[0]
    threads[s] = [c.text for c in (cache[d].get(p["source_raw_post_id"]) or [])]

PAREN = re.compile(r"\s*\([^)]{3,}\)")
DIGIT = re.compile(r"\b\d[\d,.]*\b")
WORD = re.compile(r"\S+")


def drop_url(t):
    return re.sub(r"\s+", " ", URL_RE.sub(" ", t)).strip()


def drop_paren(t):
    return re.sub(r"\s+", " ", PAREN.sub(" ", t)).strip()


def drop_digit(t):
    return re.sub(r"\s+", " ", DIGIT.sub(" ", t)).strip()


def drop_all_three(t):
    return drop_digit(drop_paren(drop_url(t)))


def hapax_flattener(texts):
    """Replace tokens occurring once in the whole thread with a frequent thread word."""
    counts = Counter(w.lower() for t in texts for w in tokenize(t))
    common = [w for w, n in counts.most_common(40) if w.isalpha() and len(w) > 2] or ["it"]

    def fn(t, _rng=random.Random(7)):
        return " ".join(_rng.choice(common) if counts[w.lower()] == 1 else w for w in WORD.findall(t))
    return fn


def random_control(text, n_removed, rng):
    """Remove n_removed whitespace tokens at random -- the length-matched null."""
    ws = WORD.findall(text)
    if n_removed <= 0 or n_removed >= len(ws):
        return text if n_removed <= 0 else ""
    keep = set(range(len(ws))) - set(rng.sample(range(len(ws)), n_removed))
    return " ".join(w for i, w in enumerate(ws) if i in keep)


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


base = {s: tf1(threads[s]) for s in SEEDS}
print(f"real baseline over the ten seeds: {st.mean(base.values()):.4f}")

GAP = 0.0138  # the gate run's shipped self_bertscore bias on these same ten seeds
print(f"gap to close on these seeds: {GAP:+.4f}\n")
print(f"{'ablation on REAL':<34}{'score':>9}{'move':>9}{'control':>9}{'net':>9}{'share of gap':>14}")

ABL = [("- URLs", drop_url), ("- parentheticals", drop_paren),
       ("- digit runs", drop_digit), ("- all three", drop_all_three)]
for name, fn in ABL:
    rng = random.Random(11)
    vals, ctrl = [], []
    for s in SEEDS:
        cut = [fn(t) for t in threads[s]]
        vals.append(tf1(cut))
        ctrl.append(tf1([random_control(t, len(WORD.findall(t)) - len(WORD.findall(c)), rng)
                         for t, c in zip(threads[s], cut)]))
    mv = st.mean(vals) - st.mean(base.values())
    cv = st.mean(ctrl) - st.mean(base.values())
    print(f"{name:<34}{st.mean(vals):>9.4f}{mv:>+9.4f}{cv:>+9.4f}{mv-cv:>+9.4f}"
          f"{100*(mv-cv)/GAP:>13.0f}%")

fn = hapax_flattener  # per-thread, so built inside the loop
vals = []
for s in SEEDS:
    vals.append(tf1([hapax_flattener(threads[s])(t) for t in threads[s]]))
mv = st.mean(vals) - st.mean(base.values())
print(f"{'hapax -> frequent thread word':<34}{st.mean(vals):>9.4f}{mv:>+9.4f}"
      f"{'(len-neutral)':>9}{mv:>+9.4f}{100*mv/GAP:>13.0f}%")
