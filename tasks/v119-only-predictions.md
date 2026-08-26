# Predictions for `v119_tonequota_only_n10_20260827_v1`, written BEFORE the run

`ORIENTATION.md` §4 step 6: write the predictions down first, then judge. Recorded
2026-08-27, before any token is spent. The comparison is **v113 on the same ten
evaluation seeds** — not the v117 calibration run, which is a different thread set.

## What this run isolates

v120's donor arm is OFF. The only change from v113 is `--tone-quota inverted` at
`POLITE_ASSIGNMENT_CAP = 0.56`. It answers one question: **is v119 a clean win, or
does it carry the same register damage v120 did (G79/G80)?**

## Predicted, with the reasoning

| metric | v113 | prediction | why |
|---|---:|---|---|
| `impolite_rate` | +49.7% | **−5% to −15%** | the arm's whole purpose; the seed-8 gate landed −6.5% |
| `polite_rate` | −47.2% | **−15% to −25%** | the solver asks a_polite 0.560 and the shipped matrix projects realized 0.244 against a 0.313 target, i.e. ~−22%. **v119 alone does NOT fix polite_rate** |
| `neutral_rate` | −33.8% | −5% to −20% | projected 0.168 against 0.164 |
| `self_bertscore` | +2.4% | **+1.5% to +4%** — no detectable change | the only paired evidence is seed 8, +4.0% → +5.7%, which is 0.58 sd of the run-to-run noise floor (G76). If it lands above +5% the assignment shift itself is harmful and v119 dies with v120 |
| `self_bleu_4` | +13.0% | +10% to +17% | seed 8 moved +19.7% → +20.0%, 0.03 sd |
| `hard_disagree_rate` | −9.9% | −5% to −15% | v120's +21.4% came from forced appreciative openers, and this run has none |
| `emotion_entropy` | +5.5% | 0% to +10% | v120's +18.3% was donor-driven (the `!` rate tripled) |

## The decision rule, fixed in advance

- **v119 ships** if `impolite_rate` improves materially AND `self_bertscore` stays
  inside +1.5% to +4%.
- **v119 dies** if `self_bertscore` exceeds ~+5%. That would mean the tone
  assignment shift is itself the register problem, not the donor cue, and the
  whole inverted-quota approach is net negative.
- `polite_rate` is expected to remain unfixed either way. The only mechanism that
  moves it is the donor, and the donor is net harmful (G79). **If v119 ships,
  `polite_rate` is still open and has no candidate.**

## What this run cannot answer

`self_bertscore` and `self_bleu_4` at N=10 carry a noise floor of sd 2.94% and
13.7% per thread (G76). Ten threads shrink that but do not remove it, so a move
inside the predicted bands is "no detectable change", not "no change".
