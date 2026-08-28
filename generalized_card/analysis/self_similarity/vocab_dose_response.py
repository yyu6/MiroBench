#!/usr/bin/env python3
"""Price the vocabulary-breadth hypothesis ON self_bertscore ITSELF (G136's rule).

Design: build threads out of GENUINE real comments, selected so the thread hits
a target distinct-type count at a held-constant token budget. Nothing is
synthesised or substituted -- the only manipulated variable is how lexically
diverse the selected set is. Then read self_bertscore off the dose curve at
real's breadth (753 types) and at ours (622) and compare the predicted
difference to the gap we actually observe (+0.0175).

The confound is that selecting for a narrow vocabulary also selects for a
narrow topic. semantic_mean_cosine is the control: our runs have NORMAL cosine
and elevated bertscore, so if narrowing drags cosine up too, it is topic and
not vocabulary, and the hypothesis fails.
"""
from __future__ import annotations
import json, re, sys, random, statistics, argparse
from pathlib import Path
REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "bert_score-master"))
from score_thread_semantic_uniformity import load_real_comments

TOK = re.compile(r"[a-z0-9']+")
rng = random.Random(7)

def types(texts):
    return len({w for t in texts for w in TOK.findall(t.lower())})
def toks(texts):
    return sum(len(TOK.findall(t.lower())) for t in texts)

def build(pool, target_types, n_comments=50, tok_lo=2600, tok_hi=3400, iters=4000):
    """Hill-climb a set of real comments toward a target type count at fixed tokens."""
    cur = rng.sample(pool, n_comments)
    def ok(s): return tok_lo <= toks(s) <= tok_hi
    for _ in range(600):
        if ok(cur): break
        cur = rng.sample(pool, n_comments)
    best = abs(types(cur) - target_types)
    for _ in range(iters):
        i = rng.randrange(n_comments)
        cand = rng.choice(pool)
        if cand in cur: continue
        trial = list(cur); trial[i] = cand
        if not ok(trial): continue
        d = abs(types(trial) - target_types)
        if d <= best:
            best, cur = d, trial
        if best == 0: break
    return cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="600,650,700,753,800")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    # ---- pool of genuine real comments -----------------------------------
    pool = []
    root = REPO / "data/raw/discussions/camera_product"
    for d in sorted(root.iterdir()):
        if not d.is_dir(): continue
        try: bythread, _ = load_real_comments(d)
        except Exception: continue
        for cs in bythread.values():
            for c in cs:
                n = len(TOK.findall(c.text.lower()))
                if 20 <= n <= 150: pool.append(" ".join(c.text.split()))
    pool = sorted(set(pool))
    print(f"real comment pool: {len(pool):,} comments (20-150 words)", flush=True)

    targets = [int(x) for x in a.targets.split(",")]
    specs = []
    for t in targets:
        for r in range(a.reps):
            th = build(pool, t)
            specs.append({"target": t, "rep": r, "texts": th,
                          "types": types(th), "tokens": toks(th)})
            print(f"  built target={t} rep={r}: types={specs[-1]['types']} tokens={specs[-1]['tokens']}", flush=True)

    out = a.out or "/private/tmp/claude-501/-Users-yaoningyu-Desktop-UIUC-GEO/d8816651-1679-43a5-8d4b-21a1a35e5936/scratchpad/vocab_dose_threads.json"
    Path(out).write_text(json.dumps(specs))
    print(f"\nwrote {len(specs)} constructed threads -> {out}")

if __name__ == "__main__":
    sys.exit(main())
