"""What ELSE does real carry that generated does not? A prevalence battery.

FINDINGS.md s3 found URLs at 76% of the self_bertscore gap by ablating five
transforms of real text. Only five were tried, and four were near-zero. URLs work
because BERTScore runs greedy token alignment with no idf, so a token with no
plausible counterpart anywhere else in the thread drags every pair it appears in.
Any token class real carries and generated does not is a candidate for the same
mechanism.

This is the free half: prevalence and token share over the same ten seeds, both
sides, no model. Classes with a large gap earn a paid ablation; classes already
matched are ruled out without spending anything.
"""
from __future__ import annotations
import json, re, statistics as st, sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "generalized_card"))
from score_thread_semantic_uniformity import load_real_comments, load_generated_comments  # noqa: E402
from score_thread_self_bleu import tokenize  # noqa: E402
from generalized_card.reference_link import extract_urls  # noqa: E402

GATE = "v113_v112_gate_n10_20260826_v1"
SEEDS = range(2, 12)
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"))["seed_posts"]
by_seed = {int(x["seed_index"]): x for x in pool}

real, cache = [], {}
for s in SEEDS:
    p = by_seed[s]
    d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if d not in cache:
        cache[d] = load_real_comments(d)[0]
    real.extend(c.text for c in (cache[d].get(p["source_raw_post_id"]) or []))

gen = []
for d in sorted((REPO / "artifacts/generalized_card/runs" / GATE / "cleaned").glob("run_*_sampled_reddit")):
    cbt, _ = load_generated_comments(d)
    for cs in cbt.values():
        gen.extend(c.text for c in cs)

CLASSES = {
    "url": lambda t: extract_urls(t),
    "u/ or r/ mention": lambda t: re.findall(r"\b[ur]/\w+", t),
    "ALLCAPS word (3+)": lambda t: re.findall(r"\b[A-Z]{3,}\b", t),
    "digit run": lambda t: re.findall(r"\d+", t),
    "price": lambda t: re.findall(r"[$£€]\s?\d[\d,.]*", t),
    "measure (mm/f/iso/mp)": lambda t: re.findall(r"\b\d+(?:\.\d+)?\s?(?:mm|MP|fps|ISO)\b|\bf/\d", t, re.I),
    "alnum model code": lambda t: re.findall(r"\b(?:[A-Za-z]+\d+[A-Za-z\d-]*|\d+[A-Za-z]+[A-Za-z\d-]*)\b", t),
    "non-ascii char": lambda t: re.findall(r"[^\x00-\x7f]", t),
    "emoji-ish": lambda t: re.findall(r"[\U0001F000-\U0001FAFF☀-➿]", t),
    "ellipsis": lambda t: re.findall(r"\.\.\.|…", t),
    "repeated punct": lambda t: re.findall(r"[!?]{2,}", t),
    "parenthetical": lambda t: re.findall(r"\([^)]{3,}\)", t),
    "quote line": lambda t: re.findall(r"(?m)^\s*>", t),
    "bullet line": lambda t: re.findall(r"(?m)^\s*(?:[-*+]\s|\d+[.)]\s)", t),
    "edit marker": lambda t: re.findall(r"(?mi)^\s*(?:edit|update)\s*\d*\s*[:\-]", t),
    "markdown emphasis": lambda t: re.findall(r"\*\*?[^*\n]+\*\*?|~~[^~\n]+~~", t),
    "em/en dash": lambda t: re.findall(r"[—–]", t),
    "semicolon": lambda t: re.findall(r";", t),
    "question mark": lambda t: re.findall(r"\?", t),
    "standalone 'lol/lmao/haha'": lambda t: re.findall(r"\b(?:lol|lmao|haha+|hah)\b", t, re.I),
    "profanity-ish": lambda t: re.findall(r"\b(?:fuck\w*|shit\w*|damn|crap|ass|hell)\b", t, re.I),
}

print(f"real comments {len(real)}   generated comments {len(gen)}")
print(f"\n{'class':<28}{'real prev':>11}{'gen prev':>10}{'ratio':>8}"
      f"{'real tok%':>11}{'gen tok%':>10}{'tok gap':>10}")
rows = []
for name, fn in CLASSES.items():
    def stats(corpus):
        prev = sum(1 for t in corpus if fn(t)) / len(corpus)
        tot = sum(len(tokenize(t)) for t in corpus)
        hit = sum(len(tokenize(" ".join(fn(t)))) for t in corpus)
        return prev, hit / tot
    rp, rt = stats(real)
    gp, gt = stats(gen)
    rows.append((name, rp, gp, rt, gt, rt - gt))
for name, rp, gp, rt, gt, d in sorted(rows, key=lambda r: -abs(r[5])):
    ratio = f"{gp/rp:.2f}" if rp else "-"
    print(f"{name:<28}{rp:>11.4f}{gp:>10.4f}{ratio:>8}{100*rt:>10.2f}%{100*gt:>9.2f}%{100*d:>+9.2f}%")

print("\n== very short comments (a bare 'this.' aligns with nothing) ==")
for label, corpus in (("real", real), ("gate", gen)):
    ws = sorted(len(t.split()) for t in corpus)
    print(f"  {label:<6} <=3 words {sum(1 for w in ws if w<=3)/len(ws):.4f}   "
          f"<=6 {sum(1 for w in ws if w<=6)/len(ws):.4f}   <=10 {sum(1 for w in ws if w<=10)/len(ws):.4f}   "
          f"p05={ws[len(ws)//20]}  med={ws[len(ws)//2]}  p95={ws[19*len(ws)//20]}  sd={st.pstdev(ws):.1f}")

print("\n== vocabulary spread (a narrower vocabulary raises every pair) ==")
for label, corpus in (("real", real), ("gate", gen)):
    toks = [w.lower() for t in corpus for w in tokenize(t)]
    types = set(toks)
    once = sum(1 for w in types if toks.count(w) == 1) if len(types) < 20000 else -1
    print(f"  {label:<6} tokens {len(toks):<7} types {len(types):<6} ttr {len(types)/len(toks):.4f}  hapax {once}")
