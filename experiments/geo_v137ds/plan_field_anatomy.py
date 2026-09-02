#!/usr/bin/env python3
"""Which part of the plan makes the slots close to each other?

  python3 experiments/geo_v137ds/plan_field_anatomy.py a1gpt_20260902

The transfer function realized(plan) is steep and reaches below the real mean,
so the binding constraint is the plan's own within-thread cosine.  This reports
that cosine per plan field, splits pairs by whether they share a branch, and
measures how much of a slot's `local_topic` is its branch's goal restated --
the branch grid is built before the Planner sees the slots, so a topic inherited
from it is an architecture decision, not a Planner judgement.
"""
import collections, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL = "sentence-transformers/all-mpnet-base-v2"
FIELDS = ("semantic_move", "local_topic", "detail_focus", "domain_intent",
          "decision_boundary", "branch_goal", "local_anchor", "owned_decision_subject")


def load(prefix):
    out = []
    for d in sorted((REPO / "artifacts/generalized_card/runs").glob(f"{prefix}_p*")):
        f = d / "generated/run_00_sampled_reddit/generation_records.json"
        if not f.exists(): continue
        by = collections.defaultdict(list)
        for r in json.load(open(f)): by[int(r["seed_index"])].append(r)
        for si, rs in by.items():
            keep = [r for r in rs if str(r.get("raw") or "").strip()]
            if len(keep) >= 6: out.append((si, keep))
    return out


def offdiag(v):
    s = v @ v.T
    iu = np.triu_indices(len(v), 1)
    return s[iu], iu


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "a1gpt_20260902"
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cpu")
    arm = load(prefix)
    print(f"{prefix}: {len(arm)} thread\n")

    per_field = collections.defaultdict(list)
    same_br, diff_br = [], []
    topic_vs_goal, topic_vs_goal_same = [], []
    nbranch, nslot = [], []
    text_same, text_diff = [], []
    for si, rs in arm:
        plans = [r["task"] for r in rs]
        texts = [str(r["raw"]).strip() for r in rs]
        n = len(plans)
        br = [str(p.get("branch_id")) for p in plans]
        nbranch.append(len(set(br))); nslot.append(n)
        blocks, index = [], {}
        for f in FIELDS:
            index[f] = (len(blocks), n)
            blocks += [str(p.get(f) or "") or "(empty)" for p in plans]
        ti = (len(blocks), n); blocks += texts
        E = m.encode(blocks, normalize_embeddings=True, batch_size=128,
                     convert_to_numpy=True, show_progress_bar=False)
        for f in FIELDS:
            o, k = index[f]
            vals, iu = offdiag(E[o:o+k])
            per_field[f].append(vals.mean())
            if f == "local_topic":
                for a, b, v in zip(iu[0], iu[1], vals):
                    (same_br if br[a] == br[b] else diff_br).append(v)
        o, k = ti
        vals, iu = offdiag(E[o:o+k])
        for a, b, v in zip(iu[0], iu[1], vals):
            (text_same if br[a] == br[b] else text_diff).append(v)
        # slot 的 local_topic 与它自己 branch 的 goal 有多近
        lo, _ = index["local_topic"]; go, _ = index["branch_goal"]
        for i in range(n):
            topic_vs_goal.append(float(E[lo+i] @ E[go+i]))

    print(f"每个 thread 平均 {np.mean(nslot):.0f} 个 slot，分布在 {np.mean(nbranch):.1f} 个 branch 上\n")
    print(f"{'plan 字段':<26}{'thread 内余弦':>13}")
    print("-"*40)
    for f in sorted(FIELDS, key=lambda f: -np.mean(per_field[f])):
        print(f"{f:<26}{np.mean(per_field[f]):>13.4f}")
    print(f"\nlocal_topic  同 branch 的 pair {np.mean(same_br):.4f} ({len(same_br)} 个)"
          f"   跨 branch {np.mean(diff_br):.4f} ({len(diff_br)} 个)")
    print(f"成文        同 branch 的 pair {np.mean(text_same):.4f} ({len(text_same)} 个)"
          f"   跨 branch {np.mean(text_diff):.4f} ({len(text_diff)} 个)")
    print(f"\n一个 slot 的 local_topic 与它自己 branch_goal 的余弦 = {np.mean(topic_vs_goal):.4f}"
          f"   (中位 {np.median(topic_vs_goal):.4f}, >0.7 的占 {np.mean(np.array(topic_vs_goal)>0.7):.1%})")


if __name__ == "__main__":
    main()
