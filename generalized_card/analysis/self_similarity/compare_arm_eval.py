#!/usr/bin/env python3
"""Side-by-side p-values for two runs on the same seeds. PASS = MWU>0.05 AND KS>0.05."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
RUNS = Path("/Users/yaoningyu/Desktop/UIUC/GEO/artifacts/generalized_card/runs")
ORDER = ["self_bertscore_mean_f1","self_bleu_4","semantic_mean_cosine","hard_disagree_rate",
         "polite_rate","impolite_rate","neutral_rate","length_cv","avg_depth",
         "structural_virality","mean_story_probability","emotion_entropy"]

def load(tag):
    p = RUNS/tag/"matched_evaluation"/"matched_seed_group_eval.json"
    return json.loads(p.read_text()) if p.exists() else None

def verdict(m):
    return "PASS" if (m.get("mwu_p_value",0) > 0.05 and m.get("ks_p_value",0) > 0.05) else "fail"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v128_interaction_n10_20260828_v1")
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    B, A = load(a.base), load(a.arm)
    if A is None:
        raise SystemExit(f"no eval for {a.arm}")
    n = 10
    sd = (2/n)*math.sqrt((2*n+1)/12)
    print(f"BASE={a.base}\nARM ={a.arm}\nsd of Cliff d at N={n}: {sd:.3f}  (a move smaller than this is noise)\n")
    hdr = f"{'metric':<28} {'BASE mwu':>9} {'ks':>7} {'d':>6} {'':>5} | {'ARM mwu':>9} {'ks':>7} {'d':>6} {'':>5} | {'d move':>7}"
    print(hdr); print("-"*len(hdr))
    bp = ap_ = 0
    for k in ORDER:
        if k not in A: continue
        am = A[k]; bm = (B or {}).get(k, {})
        bv, av = verdict(bm) if bm else "-", verdict(am)
        bp += bv == "PASS"; ap_ += av == "PASS"
        move = am["cliffs_delta"] - bm["cliffs_delta"] if bm else float("nan")
        flag = "" if abs(move) < sd else ("  <<" if abs(am["cliffs_delta"]) < abs(bm.get("cliffs_delta",9)) else "  >>")
        print(f"{k:<28} {bm.get('mwu_p_value',float('nan')):9.3f} {bm.get('ks_p_value',float('nan')):7.3f} "
              f"{bm.get('cliffs_delta',float('nan')):+6.2f} {bv:>5} | {am['mwu_p_value']:9.3f} {am['ks_p_value']:7.3f} "
              f"{am['cliffs_delta']:+6.2f} {av:>5} | {move:+7.2f}{flag}")
    print(f"\nPASS count: BASE {bp}/12   ARM {ap_}/12")
    for k in ("self_bertscore_mean_f1","self_bleu_4"):
        if k in A:
            print(f"  {k}: mwu {(B or {}).get(k,{}).get('mwu_p_value',float('nan')):.3f} -> {A[k]['mwu_p_value']:.3f}   "
                  f"d {(B or {}).get(k,{}).get('cliffs_delta',float('nan')):+.2f} -> {A[k]['cliffs_delta']:+.2f}")

if __name__ == "__main__":
    main()
