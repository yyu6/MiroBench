#!/usr/bin/env python3
"""Does mixing two writer models inside ONE thread lower self_bertscore?

G154 proved the writer swap itself is what moved the metric (paired p=0.0012),
and G145 proved the residual is corpus-level rather than thread-topic. Together
those say the excess is a *style signature*, which predicts something testable:
a gpt comment and a DeepSeek comment should be less alike than two comments
from the same model.

That is a question about PAIRS, not threads, so it does not need a mixed thread
to exist. Sample four pair populations under the exact scorer configuration --
gpt-gpt, ds-ds, gpt-ds (all within the same real thread's topic), and real-real
as the target -- and the mean F1 of a mixed thread at DeepSeek fraction f is
then algebraic:

    E[F1] = f^2 * (ds,ds) + (1-f)^2 * (gpt,gpt) + 2f(1-f) * (gpt,ds)

Honest limit: this prices the ceiling of mixing, because a real mixed thread
would also have each model *replying* to the other, which this cannot capture.
"""
from __future__ import annotations
import argparse, glob, json, random, statistics, sys
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")
sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
from score_thread_semantic_uniformity import load_generated_comments, load_real_comments  # noqa: E402

DS = ["v137ds_10_20260829_v1", "v137ds_40more_20260829_v1",
      "v137ds_s21_20260830_v2", "v137ds_s36_20260830_v2"]
GPT = ["v137_v117cfg_tonefix_n10_20260829_v1", "v137_gpt_40more_20260829_v1"]
POOL = REPO / "artifacts/generalized_card/seed_pools/camera_product_95_seed907.json"


def collect(tags):
    """seed_index -> (list of comment texts, source key)."""
    out = {}
    for t in tags:
        for run in sorted(glob.glob(str(REPO / "artifacts/generalized_card/runs" / t / "cleaned/run_*_sampled_reddit"))):
            cbt, _ = load_generated_comments(Path(run))
            for tid, cs in cbt.items():
                si = int(tid.split("seed")[-1])
                out.setdefault(si, [c.text for c in cs if c.text.strip()])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-per-population", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    g, d = collect(GPT), collect(DS)
    by = {int(p["seed_index"]): p for p in json.loads(POOL.read_text())["seed_posts"]}
    real, cache = {}, {}
    for si in set(g) & set(d):
        p = by.get(si)
        if not p:
            continue
        rd = REPO / "data/raw/discussions/camera_product" / p["source_product_dir"]
        if rd not in cache:
            try:
                cache[rd] = load_real_comments(rd)[0]
            except Exception:
                cache[rd] = {}
        rcs = cache[rd].get(p["source_raw_post_id"]) or []
        if len(rcs) >= 3:
            real[si] = [c.text for c in rcs if c.text.strip()]
    shared = sorted(set(g) & set(d) & set(real))
    print(f"threads usable in all three corpora: {len(shared)}")

    def draw(pick, n):
        """pick(seed) -> (left_pool, right_pool); same pool means an unordered within-pool pair."""
        got = []
        while len(got) < n:
            si = rng.choice(shared)
            L, R = pick(si)
            if len(L) < 2 or len(R) < 2:
                continue
            if L is R:
                i, j = rng.sample(range(len(L)), 2)
                got.append((L[i], L[j]))
            else:
                got.append((rng.choice(L), rng.choice(R)))
        return got

    pops = {
        "gpt x gpt":  draw(lambda s: (g[s], g[s]), a.pairs_per_population),
        "ds  x ds":   draw(lambda s: (d[s], d[s]), a.pairs_per_population),
        "gpt x ds":   draw(lambda s: (g[s], d[s]), a.pairs_per_population),
        "real x real": draw(lambda s: (real[s], real[s]), a.pairs_per_population),
    }

    sys.path.insert(0, str(REPO / "bert_score-master"))
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en",
                        idf=False, rescale_with_baseline=False,
                        device="cpu", batch_size=a.batch_size)
    tok = scorer._tokenizer
    if getattr(tok, "model_max_length", 0) > 100000:
        tok.model_max_length = 512

    res = {}
    for name, prs in pops.items():
        _, _, f1 = scorer.score([x for x, _ in prs], [y for _, y in prs], batch_size=a.batch_size)
        v = [float(x) for x in f1]
        res[name] = statistics.mean(v)
        print(f"  {name:12} n={len(v):>5}  mean F1 = {res[name]:.5f}  sd={statistics.pstdev(v):.4f}")

    gg, dd, gd, rr = res["gpt x gpt"], res["ds  x ds"], res["gpt x ds"], res["real x real"]
    print(f"\n  cross-model penalty vs the gpt-only baseline : {gd - gg:+.5f}")
    print(f"  cross-model penalty vs the ds-only baseline  : {gd - dd:+.5f}")
    print(f"\n  {'ds fraction f':>14}{'predicted mean F1':>20}{'vs real':>10}")
    for f in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]:
        e = f * f * dd + (1 - f) ** 2 * gg + 2 * f * (1 - f) * gd
        print(f"  {f:>14.2f}{e:>20.5f}{e - rr:>+10.5f}")
    print(f"\n  real x real = {rr:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
