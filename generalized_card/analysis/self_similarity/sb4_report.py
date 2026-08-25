import json, statistics as st
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp, wilcoxon
SP = Path("/private/tmp/claude-501/-Users-yaoningyu-Desktop-UIUC-GEO/1f41c5a0-3c0b-415c-9b23-9fb381e5c727/scratchpad")
g = json.load(open(SP/"gen_sb4.json")); r = json.load(open(SP/"real_sb4.json"))
g = {x["seed"]: x for x in g}; r = {x["seed"]: x for x in r}
seeds = sorted(g)

def col(d, k): return [d[s][k] for s in seeds]
def mwu(a,b): return mannwhitneyu(a,b,alternative="two-sided").pvalue
def ks(a,b): return ks_2samp(a,b).pvalue

print(f"{'component':<12}{'real mean':>11}{'gen mean':>11}{'gap':>10}{'gap %':>9}{'MWU p':>10}{'KS p':>9}{'paired p':>10}")
for k in ("sb4","floor","excess"):
    a, b = col(r,k), col(g,k)
    gap = st.mean(b)-st.mean(a)
    pw = wilcoxon([bb-aa for aa,bb in zip(a,b)]).pvalue
    print(f"{k:<12}{st.mean(a):>11.5f}{st.mean(b):>11.5f}{gap:>+10.5f}{100*gap/st.mean(a):>8.1f}%{mwu(a,b):>10.4f}{ks(a,b):>9.4f}{pw:>10.2e}")

a,b = col(r,"mean_tok"), col(g,"mean_tok")
print(f"\n{'mean tokens/comment':<22} real {st.mean(a):8.2f}   gen {st.mean(b):8.2f}   gap {st.mean(b)-st.mean(a):+8.2f} ({100*(st.mean(b)-st.mean(a))/st.mean(a):+.1f}%)")
print(f"{'floor share of sb4':<22} real {100*st.mean(col(r,'floor'))/st.mean(col(r,'sb4')):7.1f}%  gen {100*st.mean(col(g,'floor'))/st.mean(col(g,'sb4')):7.1f}%")

gap_total = st.mean(col(g,"sb4")) - st.mean(col(r,"sb4"))
gap_floor = st.mean(col(g,"floor")) - st.mean(col(r,"floor"))
gap_exc   = st.mean(col(g,"excess")) - st.mean(col(r,"excess"))
print(f"\nGAP DECOMPOSITION of self_bleu_4  (total {gap_total:+.5f})")
print(f"  length/smoothing floor : {gap_floor:+.5f}   {100*gap_floor/gap_total:6.1f}% of the gap")
print(f"  content overlap excess : {gap_exc:+.5f}   {100*gap_exc/gap_total:6.1f}% of the gap")

# what if the floor gap were fully removed?
cf = [g[s]["sb4"] - (g[s]["floor"] - r[s]["floor"]) for s in seeds]
print(f"\nCounterfactual: generated keeps its content excess but gets real's per-thread floor")
print(f"  gen sb4 {st.mean(col(g,'sb4')):.5f} -> {st.mean(cf):.5f}   (real {st.mean(col(r,'sb4')):.5f})")
print(f"  MWU {mwu(col(r,'sb4'),col(g,'sb4')):.4f} -> {mwu(col(r,'sb4'),cf):.4f}    KS {ks(col(r,'sb4'),col(g,'sb4')):.4f} -> {ks(col(r,'sb4'),cf):.4f}")

cf2 = [g[s]["floor"] + r[s]["excess"] for s in seeds]
print(f"\nCounterfactual: generated keeps its own floor but gets real's content excess")
print(f"  gen sb4 -> {st.mean(cf2):.5f}   MWU {mwu(col(r,'sb4'),cf2):.4f}   KS {ks(col(r,'sb4'),cf2):.4f}")
