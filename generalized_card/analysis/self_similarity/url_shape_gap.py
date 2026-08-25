"""Why does ADDING a URL to generated buy 2.5x less than REMOVING it from real?

FINDINGS.md s3: stripping every URL from real moves self_bertscore +0.0094 = 76%
of the gap at a 4.41% comment prevalence. The v113 gate wrote links at 4.32%
prevalence -- matched -- yet the same ablation applied to the gate's own output
moved only 0.00383, 22% closure. Same prevalence, 2.5x less effect, so the
difference is in the SHAPE of the carrying comment, not how many there are.

BERTScore F1 is a greedy token alignment with no idf, so what a URL is worth is
its share of its comment's tokens: a bare link-drop is almost entirely
unalignable, a link inside a 60-word paragraph is a rounding error.

Measures the shape on both sides over the same ten seeds, then prices the
ceiling of matching it (J7: an ablation is an upper bound).
"""
from __future__ import annotations
import json, re, statistics as st, sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "generalized_card"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_self_bertscore import load_bert_scorer  # noqa: E402
from score_thread_self_bleu import tokenize  # noqa: E402
from generalized_card.reference_link import extract_urls  # noqa: E402

GATE = "v113_v112_gate_n10_20260826_v1"
SEEDS = range(2, 12)
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}

# ---- load the same ten threads on both sides -------------------------------
real_threads = {}
cache = {}
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

sides = {"real": real_threads, "gate": gen_threads}

print("== shape of a URL-carrying comment ==")
print(f"{'side':<8}{'comments':>10}{'with URL':>10}{'prev':>8}{'words':>8}{'tok':>7}"
      f"{'URLtok':>8}{'URL share':>11}{'thread frac':>12}")
for label, thr in sides.items():
    allc = [t for cs in thr.values() for t in cs]
    carr = [t for t in allc if extract_urls(t)]
    utok = [sum(len(tokenize(u)) for u in extract_urls(t)) for t in carr]
    ctok = [len(tokenize(t)) for t in carr]
    thrtok = st.mean(len(tokenize(t)) for t in allc)
    print(f"{label:<8}{len(allc):>10}{len(carr):>10}{len(carr)/len(allc):>8.4f}"
          f"{st.mean(len(t.split()) for t in carr):>8.1f}{st.mean(ctok):>7.1f}"
          f"{st.mean(utok):>8.1f}{st.mean(u/c for u, c in zip(utok, ctok)):>11.4f}"
          f"{st.mean(ctok)/thrtok:>12.2f}")

print("\n== word-count distribution of URL-carrying comments ==")
for label, thr in sides.items():
    ws = sorted(len(t.split()) for cs in thr.values() for t in cs if extract_urls(t))
    q = lambda p: ws[min(len(ws) - 1, int(p * len(ws)))]  # noqa: E731
    print(f"  {label:<6} n={len(ws):<4} min={ws[0]:<4} p25={q(.25):<4} med={q(.5):<4} "
          f"p75={q(.75):<4} max={ws[-1]:<4}   share <=25 words {sum(1 for w in ws if w<=25)/len(ws):.2f}")

# ---- price it ---------------------------------------------------------------
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


SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def keep_link_sentence(text):
    """Reduce a link-carrying comment to the sentence holding the link -- real's shape."""
    if not extract_urls(text):
        return text
    parts = [s.strip() for s in SENT.split(text) if s.strip()]
    keep = [s for s in parts if extract_urls(s)]
    return " ".join(keep) if keep else text


def strip_all(text):
    return re.sub(r"\s+", " ", " ".join(w for w in text.split() if not extract_urls(w))).strip()


real_base = [tf1(real_threads[s]) for s in SEEDS]
gen_base = [tf1(gen_threads[s]) for s in SEEDS]
gen_nolink = [tf1([strip_all(t) for t in gen_threads[s]]) for s in SEEDS]
gen_short = [tf1([keep_link_sentence(t) for t in gen_threads[s]]) for s in SEEDS]

print(f"\n{'variant':<44}{'self_bertscore':>16}{'bias vs real':>14}")
print(f"{'real':<44}{st.mean(real_base):>16.4f}{'':>14}")
for name, v in (("gate, as shipped", gen_base),
                ("gate, links stripped (the counterfactual)", gen_nolink),
                ("gate, link comment cut to its link sentence", gen_short)):
    print(f"{name:<44}{st.mean(v):>16.4f}{100*(st.mean(v)-st.mean(real_base))/st.mean(real_base):>13.2f}%")

gap_nolink = st.mean(gen_nolink) - st.mean(real_base)
for name, v in (("as shipped", gen_base), ("cut to link sentence", gen_short)):
    print(f"  closure of the no-link gap, {name:<24} "
          f"{100*(gap_nolink-(st.mean(v)-st.mean(real_base)))/gap_nolink:>6.1f}%")
