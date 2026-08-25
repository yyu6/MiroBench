"""Does self_bertscore closure scale with URL token mass? The v115 draw question.

url_shape_gap.py measured the gate's link arm at 24.4% closure of the no-link gap
against the 42% Holm needs at N=150. The shape table says why: a real link-carrying
comment holds 32.5 URL tokens (1.42 URLs at 22.9 tokens) and a generated one holds
18.1 (1.22 at 14.8). The inventory itself averages 23.3 tokens per URL -- it is not
short. v113's inventory reader ran `https?://\S+` straight through Reddit's
`[url](url)` markdown, so 166/690 entries were malformed and the drawn strings were
truncated; v114 fixed the reader.

So the buildable question is whether URL mass is the dial. This adds inventory URLs
to the gate's own link-carrying comments to reach real's mass and rescores. Nothing
about comment length or any other slot changes, so the move is attributable.

J7: an ablation is an upper bound. The Writer would have to comply with a two-link
offer at some rate below 1.0, and that rate is unmeasured.
"""
from __future__ import annotations
import hashlib, json, random, re, statistics as st, sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "generalized_card"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_self_bertscore import load_bert_scorer  # noqa: E402
from score_thread_self_bleu import tokenize  # noqa: E402
from generalized_card.reference_link import extract_urls, URL_RE  # noqa: E402

GATE = "v113_v112_gate_n10_20260826_v1"
SEEDS = range(2, 12)
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}
root = REPO / "artifacts/generalized_card/runs" / GATE

# v114's reader, applied to v113's stored inventory so the strings are clean
raw_inv = (json.load(open(root / "domain_profile.json")).get("reference_link_inventory") or {}).get("urls") or []
INV = sorted({u for entry in raw_inv for u in extract_urls(entry)})
print(f"inventory: {len(raw_inv)} stored entries -> {len(INV)} clean URLs, "
      f"{st.mean(len(tokenize(u)) for u in INV):.1f} tokens each")

real_threads, cache = {}, {}
for s in SEEDS:
    p = by_seed[s]
    d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if d not in cache:
        cache[d] = load_real_comments(d)[0]
    real_threads[s] = [c.text for c in (cache[d].get(p["source_raw_post_id"]) or [])]
gen_threads = {}
for d in sorted((root / "cleaned").glob("run_*_sampled_reddit")):
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


def clean(t):
    """Rewrite the comment's own links to their v114-clean form."""
    if not extract_urls(t):
        return t
    return re.sub(r"\S*https?://\S*|\S*\bwww\.\S*",
                  lambda m: " ".join(extract_urls(m.group(0))) or "", t).strip()


def draw(seed_key, k):
    out, i = [], 0
    while len(out) < k and i < 40:
        h = int(hashlib.sha256(f"{seed_key}|{i}".encode()).hexdigest(), 16)
        u = INV[h % len(INV)]
        if u not in out:
            out.append(u)
        i += 1
    return out


def add_links(texts, extra_rate, rng):
    """Give `extra_rate` of the link-carrying comments one additional inventory URL."""
    out = []
    for i, t in enumerate(texts):
        if extract_urls(t) and rng.random() < extra_rate:
            u = draw(f"{i}|{t[:40]}", 1)[0]
            out.append(clean(t) + " " + u)
        else:
            out.append(clean(t))
    return out


def strip_url(t):
    return re.sub(r"\s+", " ", URL_RE.sub(" ", t)).strip()


real_base = st.mean(tf1(real_threads[s]) for s in SEEDS)
shipped = st.mean(tf1(gen_threads[s]) for s in SEEDS)
nolink = st.mean(tf1([strip_url(t) for t in gen_threads[s]]) for s in SEEDS)
gap_nolink = nolink - real_base
print(f"\nreal {real_base:.4f}   gate shipped {shipped:.4f}   gate no-link {nolink:.4f}"
      f"   no-link gap {gap_nolink:+.4f}\n")
print(f"{'variant':<44}{'urltok/cmt':>12}{'score':>9}{'bias':>9}{'closure':>10}")


def mass(threads):
    carr = [t for cs in threads.values() for t in cs if extract_urls(t)]
    return st.mean(sum(len(tokenize(u)) for u in extract_urls(t)) for t in carr) if carr else 0.0


print(f"{'real':<44}{mass(real_threads):>12.1f}{real_base:>9.4f}{'':>9}{'':>10}")
for name, rate in (("gate, links cleaned to v114 form", 0.0),
                   ("+ 1 extra URL on 20% of link comments", 0.20),
                   ("+ 1 extra URL on 42% of link comments", 0.42),
                   ("+ 1 extra URL on 100% of link comments", 1.00)):
    rng = random.Random(23)
    built = {s: add_links(gen_threads[s], rate, rng) for s in SEEDS}
    v = st.mean(tf1(built[s]) for s in SEEDS)
    print(f"{name:<44}{mass(built):>12.1f}{v:>9.4f}"
          f"{100*(v-real_base)/real_base:>8.2f}%{100*(gap_nolink-(v-real_base))/gap_nolink:>9.1f}%")
print(f"\n{'gate, as shipped (v113 truncated links)':<44}{mass(gen_threads):>12.1f}{shipped:>9.4f}"
      f"{100*(shipped-real_base)/real_base:>8.2f}%{100*(gap_nolink-(shipped-real_base))/gap_nolink:>9.1f}%")
print("\nHolm needs 42% closure on self_bertscore at N=150 (FINDINGS.md s4).")
