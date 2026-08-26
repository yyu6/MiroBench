"""Does turning the persona layer ON change the generator's authorial voice?

G57 states that `persona_bridge`, `speaker_roster`, `actor_conditioning` and
`--speaker-identity matched` "all exist and were on for every run". **Two of the
four were not.** Every modern run carries `persona_conditioning.mode = "none"` and
`actor_conditioning.mode = "none"` in its `run_config.json` -- both CLI defaults --
so the +0.0076 voice separation s8 measured comes from `--speaker-identity matched`
alone, and the MatrAIx persona system prompt has never been in a Writer prompt in
any run this project has evaluated.

Sweeping all 163 `run_config.json` files: 7 runs carry `matraix-projected`, all
from 2026-08-08/09, and exactly one of them has a usable `cleaned/` tree.

This runs s8's instrument on both sides. **It is an observation, not an ablation:**
the persona-on run is ~100 versions older than the persona-off one and differs in
far more than the persona layer, which is the same trap the 9-point `repro_v37`
swing sits in. Read the direction, not the magnitude.

Usage:  python3 generalized_card/analysis/self_similarity/one_voice_persona.py <tag> [<tag> ...]
"""
from __future__ import annotations
import statistics as st, sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_generated_comments  # noqa: E402
from score_thread_self_bertscore import load_bert_scorer  # noqa: E402

STRATA = ("same parent", "ancestor/descendant", "same root branch", "different branch")
REAL = {"same parent": -0.0008, "ancestor/descendant": +0.0021,
        "same root branch": +0.0233, "different branch": +0.0141}

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
            seen.add(cid)
            out.append(cid)
            cid = parent.get(cid, "")
        return out

    return parent, {str(c.comment_id): chain(str(c.comment_id))[-1] for c in comments}, \
        {str(c.comment_id): set(chain(str(c.comment_id))) for c in comments}


def measure(tag: str) -> None:
    root = REPO / "artifacts/generalized_card/runs" / tag
    source = root / "cleaned" if (root / "cleaned").exists() else root / "generated"
    threads = {}
    for d in sorted(source.glob("run_*_sampled_reddit")):
        cbt, _ = load_generated_comments(d)
        for tid, cs in cbt.items():
            threads[tid] = cs
    buckets = {name: {"same": [], "diff": []} for name in STRATA}
    authors, comments = set(), 0
    for key in sorted(threads):
        cs = threads[key]
        if len(cs) < 6:
            continue
        comments += len(cs)
        authors |= {c.author for c in cs if c.author}
        parent, rootmap, anc = structure(cs)
        cand, ref, meta = [], [], []
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                ida, idb = str(a.comment_id), str(b.comment_id)
                if parent.get(ida) and parent.get(ida) == parent.get(idb):
                    rel = "same parent"
                elif idb in anc.get(ida, ()) or ida in anc.get(idb, ()):
                    rel = "ancestor/descendant"
                elif rootmap.get(ida) == rootmap.get(idb):
                    rel = "same root branch"
                else:
                    rel = "different branch"
                cand.append(a.text); ref.append(b.text)
                meta.append((rel, bool(a.author) and a.author == b.author))
        if not meta:
            continue
        _, _, f1 = scorer.score(cand, ref, batch_size=8)
        for (rel, same), value in zip(meta, f1):
            buckets[rel]["same" if same else "diff"].append(float(value))

    print(f"\n=== {tag} ===")
    print(f"comments {comments}   distinct authors {len(authors)}")
    print(f"{'relation':<24}{'same-author':>22}{'different':>22}{'delta':>10}{'real':>10}")
    weights, deltas = [], []
    for name in STRATA:
        same, diff = buckets[name]["same"], buckets[name]["diff"]
        if len(same) < 10 or len(diff) < 10:
            print(f"{name:<24}{f'n={len(same)} (thin)':>22}{f'n={len(diff)}':>22}{'-':>10}{REAL[name]:>+10.4f}")
            continue
        d = st.mean(same) - st.mean(diff)
        weights.append(len(same) + len(diff)); deltas.append(d)
        print(f"{name:<24}{f'{st.mean(same):.4f} (n={len(same)})':>22}"
              f"{f'{st.mean(diff):.4f} (n={len(diff)})':>22}{d:>+10.4f}{REAL[name]:>+10.4f}")
    allsame = [v for n in STRATA for v in buckets[n]["same"]]
    alldiff = [v for n in STRATA for v in buckets[n]["diff"]]
    if allsame and alldiff:
        print(f"{'pooled (uncontrolled)':<24}{f'{st.mean(allsame):.4f} (n={len(allsame)})':>22}"
              f"{f'{st.mean(alldiff):.4f} (n={len(alldiff)})':>22}"
              f"{st.mean(allsame)-st.mean(alldiff):>+10.4f}{0.0177:>+10.4f}")
    if weights:
        adj = sum(w * d for w, d in zip(weights, deltas)) / sum(weights)
        print(f"{'stratum-weighted':<24}{'':>22}{'':>22}{adj:>+10.4f}{0.0137:>+10.4f}")
        print(f"   voice strength as a fraction of real's: {adj/0.0137:.2f}"
              f"   (persona-off v113 gate measured 0.55)")


if __name__ == "__main__":
    tags = sys.argv[1:] or ["v117_calibration_20260826_v1"]
    for tag in tags:
        measure(tag)
