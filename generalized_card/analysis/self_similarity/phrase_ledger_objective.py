#!/usr/bin/env python3
"""Judge the v134 arm on its OWN objective first (G88/G98 discipline).

The ledger names function-word bigrams the thread has already leaned on and
tells the Writer to say it differently. So the question is not "did overlap
fall" -- it is: did the NAMED pairs get suppressed, and did OTHER function
pairs move in to take their place? A reshuffle that leaves the high-DF
function band the same size is a relocation, not a repair.
"""
from __future__ import annotations
import json, re, statistics, sys
from collections import Counter
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments

TOK = re.compile(r"[a-z0-9']+")
FUNCTION = set("""a an the and or but if so then than that this these those there here
is are was were be been being am s re ve d ll t don't doesn't isn't aren't wasn't
of to in on at for with from by about into over after before as like just only
very really pretty quite kind sort bit lot much many more most less least
i you he she it we they me him her us them my your his its our their mine yours
what which who whom whose when where why how all any both each few other some such
no nor not too own same can could would should may might must will shall do does did
have has had get got go goes went one two 1 2 up out off down again still even also
""".split())

def bigrams(w): return list(zip(w, w[1:]))
def fn(g): return g[0] in FUNCTION and g[1] in FUNCTION

def overlap(word_lists):
    sets = [set(bigrams(w)) for w in word_lists]
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            if u: vals.append(len(sets[i] & sets[j]) / len(u))
    return statistics.mean(vals) if vals else float("nan")

def fn_overlap(word_lists):
    sets = [{g for g in bigrams(w) if fn(g)} for w in word_lists]
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            if u: vals.append(len(sets[i] & sets[j]) / len(u))
    return statistics.mean(vals) if vals else float("nan")

def hot_band(word_lists, frac=0.10):
    n = len(word_lists); df = Counter()
    for w in word_lists:
        for g in set(bigrams(w)): df[g] += 1
    return {g: df[g] / n for g, v in df.items() if v / n > frac and fn(g)}

def load(tag, sub):
    root = REPO / "artifacts/generalized_card/runs" / tag / sub
    out = {}
    for x in sorted(root.glob("run_*_sampled_reddit")):
        cbt, _ = load_generated_comments(x)
        for tid, cs in cbt.items():
            out[int(tid.split("seed")[-1])] = [TOK.findall(c.text.lower()) for c in cs]
    return out

pool = json.loads((REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json").read_text())["seed_posts"]
by = {int(p["seed_index"]): p for p in pool}

a = load("v134_phraseledger_n10_20260828_v1", "generated")
b = load("v128_interaction_n10_20260828_v1", "cleaned")
shared = sorted(set(a) & set(b))

cache = {}
rows = []
for sidx in shared:
    p = by[sidx]
    d = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
    if d not in cache:
        try: cache[d] = load_real_comments(d)[0]
        except Exception: cache[d] = {}
    rcs = cache[d].get(p["source_raw_post_id"]) or []
    if len(rcs) < 12: continue
    r = [TOK.findall(c.text.lower()) for c in rcs]
    rows.append((sidx, r, b[sidx], a[sidx]))

print(f"threads compared: {len(rows)}  (seeds {[r[0] for r in rows]})\n")
hdr = f"{'':34}{'real':>10}{'v128':>10}{'v134':>10}"
print(hdr); print("-" * len(hdr))
def line(label, f):
    vals = [statistics.mean([f(x[k]) for x in rows]) for k in (1, 2, 3)]
    print(f"{label:34}" + "".join(f"{v:>10.5f}" for v in vals))
    return vals
ov  = line("2-gram overlap (all)", overlap)
fo  = line("2-gram overlap (function only)", fn_overlap)
nb  = line("high-DF function bigrams (count)", lambda w: float(len(hot_band(w))))
print()

# --- the relocation test -------------------------------------------------
print("per-thread: what happened to v128's hot band under v134")
print(f"{'seed':>6}{'v128 band':>11}{'v134 band':>11}{'kept':>7}{'new':>6}{'ov v128':>10}{'ov v134':>10}")
kept_t = new_t = 0
for sidx, r, g128, g134 in rows:
    h128, h134 = hot_band(g128), hot_band(g134)
    kept = len(set(h128) & set(h134)); new = len(set(h134) - set(h128))
    kept_t += kept; new_t += new
    print(f"{sidx:>6}{len(h128):>11}{len(h134):>11}{kept:>7}{new:>6}{overlap(g128):>10.5f}{overlap(g134):>10.5f}")
print(f"\n  of v128's hot function bigrams, {kept_t} survive into v134; {new_t} are NEW in v134")

# --- what real actually repeats vs what we repeat -------------------------
def top_hot(idx, k=12):
    c = Counter()
    for row in rows:
        for g in hot_band(row[idx]): c[g] += 1
    return c.most_common(k)
for name, idx in (("real", 1), ("v128", 2), ("v134", 3)):
    print(f"\n{name} -- function bigrams hot in the most threads:")
    print("   " + ", ".join(f"{' '.join(g)}({n})" for g, n in top_hot(idx)))
