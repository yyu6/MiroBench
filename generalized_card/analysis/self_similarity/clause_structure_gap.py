#!/usr/bin/env python3
"""The clause-structure signature G40 named, measured on the current artifact.

G27/G35/G40 priced out every entity and length mechanism for `self_bleu_4`
(<=10% each, several wrong-signed). What G40 left standing was a *profile*, not a
lever: after v109 the pooled log excess is p1 56.3% + p2 54.9% + BP 30.1% with p3
and p4 NEGATIVE, `the` alone carries 20.9% of the positive excess overlap mass,
and the most under-shared token is the full stop itself (-0.0461), followed by
`to`, `i`, `of`, `with`, `for`, `and`, `but`, `be`, `have`.

One reading fits all of it: **generated writes fewer, longer, determiner-dense,
less verbal sentences than real.** That is a clause-structure difference, and it
was never measured directly -- G40 inferred it from a token attribution.

This measures it directly on both sides of the same matched threads, and asks the
second question too: the same property would raise `self_bertscore`, because
BERTScore is a greedy alignment over contextual embeddings and a shared syntactic
register makes every pair more alignable -- which is the shape of the residual
G68 left behind (a uniform ~+0.02 lift on every pair, with the author structure
now correct).

Prints a profile only. Pricing an ablation is the next step and is NOT done here.

Usage:  python3 generalized_card/analysis/self_similarity/clause_structure_gap.py [tag ...]
"""
from __future__ import annotations

import re
import statistics as st
import sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO / "generalized_card"))
from score_thread_semantic_uniformity import (  # noqa: E402
    load_generated_comments,
    load_real_comments,
)

import json  # noqa: E402

POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_150_seed42.json"
RAW = REPO / "data/raw/discussions/camera_product"

SENT_END = re.compile(r"[.!?]+(?:\s|$)")
WORD = re.compile(r"[A-Za-z']+")
DET = {"the", "a", "an", "this", "that", "these", "those"}
# The verbal/conversational side of G40's under-shared list.
VERBAL = {
    "to", "be", "have", "has", "had", "is", "are", "was", "were", "do", "does",
    "did", "get", "got", "go", "went", "will", "would", "can", "could", "should",
    "want", "need", "think", "know", "make", "made", "take", "use", "used",
}
PRONOUN = {"i", "you", "we", "they", "he", "she", "it", "me", "us", "them", "my", "your"}


def profile(texts: list[str]) -> dict[str, float]:
    sent_counts, sent_lens, words = [], [], []
    det = verbal = pron = tokens = 0
    ends = 0
    for text in texts:
        body = str(text or "").strip()
        if not body:
            continue
        pieces = [p for p in SENT_END.split(body) if p.strip()]
        toks = [w.lower() for w in WORD.findall(body)]
        if not toks:
            continue
        sent_counts.append(max(1, len(pieces)))
        sent_lens.append(len(toks) / max(1, len(pieces)))
        words.append(len(toks))
        tokens += len(toks)
        det += sum(1 for w in toks if w in DET)
        verbal += sum(1 for w in toks if w in VERBAL)
        pron += sum(1 for w in toks if w in PRONOUN)
        ends += 1 if SENT_END.search(body[-3:] + " ") else 0
    n = len(words)
    return {
        "comments": n,
        "words/comment": st.mean(words),
        "sentences/comment": st.mean(sent_counts),
        "words/sentence": st.mean(sent_lens),
        "words/sentence p90": sorted(sent_lens)[int(0.9 * (n - 1))],
        "determiner rate": det / tokens,
        "verbal rate": verbal / tokens,
        "pronoun rate": pron / tokens,
        "ends with . ! ?": ends / n,
        "types/sqrt(tokens)": 0.0,
    }


def breadth(texts: list[str]) -> float:
    toks = [w.lower() for t in texts for w in WORD.findall(str(t or ""))]
    return len(set(toks)) / (len(toks) ** 0.5) if toks else 0.0


def main() -> None:
    tags = sys.argv[1:] or ["v117_calibration_20260826_v1"]
    seeds = {int(x["seed_index"]): x for x in json.loads(POOL.read_text())["seed_posts"]}
    for tag in tags:
        root = REPO / "artifacts/generalized_card/runs" / tag
        source = root / "cleaned" if (root / "cleaned").exists() else root / "generated"
        gen, gen_by_seed = [], {}
        for d in sorted(source.glob("run_*_sampled_reddit")):
            cbt, _ = load_generated_comments(d)
            for tid, cs in cbt.items():
                gen_by_seed[tid] = [c.text for c in cs]
                gen.extend(c.text for c in cs)
        # matched real: the seed each generated thread stands in for
        real, cache = [], {}
        for tid in gen_by_seed:
            try:
                idx = int(str(tid).split("seed")[-1])
            except ValueError:
                continue
            post = seeds.get(idx)
            if not post:
                continue
            folder = RAW / post["source_product_dir"]
            if folder not in cache:
                cache[folder] = load_real_comments(folder)[0]
            real.extend(
                c.text for c in (cache[folder].get(post["source_raw_post_id"]) or [])
            )
        if not real:
            print(f"{tag}: no matched real threads resolved"); continue
        gp, rp = profile(gen), profile(real)
        gp["types/sqrt(tokens)"] = breadth(gen)
        rp["types/sqrt(tokens)"] = breadth(real)
        print(f"\n=== {tag} ===")
        print(f"{'property':<24}{'real':>12}{'generated':>12}{'gen/real':>11}")
        for key in rp:
            r, g = rp[key], gp[key]
            ratio = (g / r) if r else float("nan")
            flag = "  <<" if abs(ratio - 1) > 0.12 and key != "comments" else ""
            print(f"{key:<24}{r:>12.4f}{g:>12.4f}{ratio:>11.3f}{flag}")


if __name__ == "__main__":
    main()
