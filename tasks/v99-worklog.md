# v99 worklog — the polite register is a realization failure with a measured lexical signature

Date: 2026-08-20

Status: **diagnosis complete, nothing built yet.** Four hypotheses rejected, one
mechanism verified. Read this before writing any code for the politeness trio.

Target metrics: `polite_rate` (FAIL, MWU 0.013, Cliff −0.67), `impolite_rate`
(FAIL, 0.001, +0.88), `neutral_rate` (PARTIAL, 0.021, −0.62).

All measurements below are on the v98 N=10 artifact
(`generalized_card_camera_gpt54_v98_rhythm_n10_20260820_v1`, 528 comments) against
the 10 matched real threads (659 comments) and the evaluation-excluded real corpus
(15,294 comments, 150-thread seed pool removed). Real per-comment polite-guard
labels already existed at
`data/raw/discussions/camera_product/<product>/politeness_results.json`, so no
model was re-run and no metric was approximated.

## The finding, in one table

The plan is correct. Realization is the entire failure, and it is asymmetric.

| planned | n | → polite | → somewhat | → neutral | → impolite |
|---|---:|---:|---:|---:|---:|
| polite | 145 | **0.193** | 0.221 | 0.083 | **0.503** |
| somewhat_polite | 46 | 0.022 | 0.457 | 0.043 | 0.478 |
| neutral | 76 | 0.053 | 0.184 | 0.250 | 0.513 |
| impolite | 261 | 0.015 | 0.027 | 0.061 | **0.897** |

Marginals: planned 0.275 polite / 0.494 impolite, against a real 0.288 / 0.443 —
the plan is right to within 2 points. Realized: 0.070 / 0.697.

**The Writer has one register and it is the impolite one.** Told to be blunt and
dismissive it complies 89.7% of the time. Told to be "warm and personally engaged,
commit to a positive evaluation" it complies 19.3% of the time and produces the
blunt register instead in half the cases.

It is worst exactly where real text is most positive:

| realized length | n | planned polite | of those, landed polite | real polite_rate | gen polite_rate |
|---|---:|---:|---:|---:|---:|
| 0–15 w | 125 | 0.232 | 0.207 | 0.121 | 0.072 |
| 15–30 w | 128 | 0.133 | **0.000** | 0.085 | **0.000** |
| 30–60 w | 128 | 0.211 | 0.074 | 0.274 | 0.031 |
| 60–120 w | 98 | 0.398 | 0.333 | 0.442 | 0.173 |
| 120+ w | 49 | **0.673** | 0.212 | **0.767** | 0.143 |

Real comments over 120 words are 76.7% polite — enthusiastic, detailed,
first-person sharing. Generated ones are 14.3%, and 69.4% impolite, despite 67.3%
of them being planned polite. This is also why v98's `length_cv` fix did not help
politeness: it got the lengths right and filled them with the wrong register.

## Four hypotheses measured and rejected

Each was rejected the same way — condition on the feature and the generated/real
gap is unchanged inside every cell. A flat gap under conditioning means the
conditioning variable is not the cause.

**Rejected: marker frequency.** The polite register was derived from the excluded
corpus by token log-odds (document frequency, fitted on half the excluded threads,
scored on the other half) rather than hand-listed, so it carries no domain
vocabulary by construction. Out-of-sample lift on P(polite) is 3.56×.

| | marker presence | P(polite \| marker) | P(polite \| none) |
|---|---:|---:|---:|
| excluded real (15,294) | 0.308 | 0.627 | 0.173 |
| matched real (659) | 0.284 | 0.652 | 0.144 |
| generated (528) | 0.178 | **0.213** | 0.039 |

Moving presence to the real level while holding the generated conditional predicts
`polite_rate` 0.070 → **0.088**, against a real 0.288. The count is not the lever;
the conditional is. **A per-slot warmth-marker schedule — the v99 design this
project had been carrying since the v98 worklog — would have been a near-null paid
run.**

**Rejected: warmth used as a concession.** The theory was that generated warmth
appears as a concession before a verdict ("great sensor, but…"). Contrastive
connectives do not lower P(polite) in real text at all: 0.711 without against
0.738 with, and 62% of real marker-bearing comments carry one. In generated text
P(polite) is also *higher* with a contrastive (0.348 vs 0.250).

**Rejected: first-person lived experience.** The log-odds list returned
`portraits`, `night`, `sharing`, `daily`, `picked`, `taken`, `went`, `recently`,
`here's` next to `thank` and `love`, which reads as "I own this and here is what
happened". Eight surface-syntactic features were tested for out-of-sample lift:
`i_own` 2.15×, `my_thing` 2.11×, `i_past` 1.90×, `showing` 1.95×, `i_have_done`
1.92×, `time_ref` 1.78×, `any_past` 1.71×, `future` 1.43× — all well below the
warmth set's 3.56×. Those tokens are topical correlates of polite threads, not the
driver. The interaction is real but modest (warmth+past 0.811 against warmth-only
0.639 in real text).

The cell table is what rejected it, and it is the important one:

| cell | real P(polite) | gen P(polite) | real share | gen share |
|---|---:|---:|---:|---:|
| warmth + first-person past | 0.750 | 0.235 | 0.055 | 0.032 |
| warmth only | 0.627 | 0.277 | 0.167 | 0.089 |
| first-person past only | 0.339 | 0.075 | 0.094 | 0.100 |
| neither | 0.162 | 0.039 | 0.684 | 0.778 |

A uniform 3–4× gap in every cell. Same signature as the `self_bertscore` gap.

**Rejected: a dismissive-adjudicative register.** Reading the `neither` cell showed
real polite comments offering information ("it's absolutely possible if you
understand light and exposure", "EF glass works very well… still really
impressive", "you can opt to not use a grip extender") against generated ones
negating and dismissing ("Speculation is noise", "that page is junk", "Spec talk
is useless", "it's dead weight… spec-sheet fluff", "then what exactly is the
point?"). Three families were measured. Out-of-sample lift on P(impolite):
`negate_premise` 1.43×, `adjudge` 1.53× (on almost no real occurrences, so not
estimable), `dismiss_noun` 1.14× — and `dismiss_noun`'s lift on *polite* is 1.36×,
i.e. it appears slightly more in polite real comments.

The counterfactual settles it: excluded-real P(polite | any dismissive family) =
0.293 against P(polite | none) = 0.315. **No effect.** Removing the dismissive
register entirely predicts `polite_rate` 0.070 → 0.077. And generated comments
carrying no dismissive family still sit at P(polite) 0.082 against a real 0.281.

(An earlier draft of this line said 0.057, from a standalone probe that had the
generated conditionals hardcoded. `politeness_diagnosis.py dismissive` derives
them and reports 0.077. The conclusion is unchanged; the script is the
authority.)

## Two hard prevalence findings that survive as eye-visible tells

Neither drives the label, and both are criterion-2 problems worth their own fix:

- **`adjudge`: 0 of 15,294 excluded real comments, 0 of 659 matched real, 37 of
  528 generated (0.070).** The frame this project has chased since v73 — "the only
  thing that matters", "that's the part that", "the real question", "what exactly
  is the point" — appears in 7% of generated comments and *never* in real text.
- **`dismiss_noun` at 5.17× real** (0.165 against 0.032): `junk`, `noise`,
  `useless`, `fluff`, `dead weight`, `gimmick`, `spec-sheet`.

## The verified mechanism

A word-level TF-IDF logistic model (uni+bigrams, min_df 5, fitted on half the
excluded threads to predict polite-guard's own label) reproduces the classifier on
real text and predicts low on generated:

| corpus | polite-guard | model | AUC |
|---|---:|---:|---:|
| held-out excluded real | 0.314 | 0.319 | 0.874 |
| matched real (10) | 0.288 | 0.294 | 0.906 |
| generated v98 | 0.070 | 0.106 | 0.895 |

So **the gap is lexical and distributed** — a vocabulary-level fix can work — and
it decomposes almost entirely one way:

```
polite-feature deficit in generated : +8.381
impolite-feature excess in generated: -0.767      <- negative
```

Generated text uses *less* of the impolite vocabulary than real text.
**Suppressing negative markers would make the metric worse.** That reverses the
priority recorded in the v98 worklog ("negative markers 0.141 against a real
0.047, 3×"), which was measuring a different, narrower marker set.

Per 1,000 tokens, so length cannot explain it (matched real averages 62.6 words,
generated 53.3):

| token | real /1k | gen /1k | ratio |
|---|---:|---:|---:|
| `very` | 2.034 | 0.385 | **0.19×** |
| `hope` | 0.378 | 0.070 | 0.19× |
| `would` | 3.241 | 0.665 | 0.21× |
| `love` | 0.852 | 0.175 | 0.21× |
| `good` | 3.075 | 1.016 | 0.33× |
| `great` | 1.608 | 0.525 | 0.33× |
| `my` | 5.748 | 2.942 | 0.51× |
| `for` | 13.837 | 10.261 | 0.74× |
| `and` | 23.180 | 19.752 | 0.85× |
| `thank` | 0.189 | 0.000 | **0.00×** |
| `amazing` | 0.189 | 0.000 | 0.00× |
| `awesome` | 0.118 | 0.000 | 0.00× |
| `incredible` | 0.024 | 0.000 | 0.00× |
| `https` | 1.348 | 0.000 | 0.00× |
| `thanks` | 0.189 | 0.560 | **2.96×** |
| `nice` | 0.473 | 0.700 | 1.48× |

**The generated positive vocabulary is about two words wide (`thanks`, `nice`)
where real is about ten.** That is the register-narrowness defect this project has
seen in `self_bleu_4`, `self_bertscore`, and `check` at 10× real, localized to the
positive-evaluative axis.

Controlled within matched length bands, the deficit scales with length, and so
does the metric gap:

| band | polite-vocab deficit | real polite_rate | gen polite_rate |
|---|---:|---:|---:|
| 0–15 w | +0.229 | 0.121 | 0.072 |
| 15–30 w | +2.809 | 0.085 | 0.000 |
| 30–60 w | +4.968 | 0.274 | 0.031 |
| 60–120 w | +6.826 | 0.442 | 0.173 |
| 120+ w | **+13.136** | 0.767 | 0.143 |

And the link between the thin vocabulary and the low conditional holds: a
generated marker-bearing comment carries surrounding positive-feature mass 2.444
against a real 3.459, with model probability 0.520 against 0.693. **A generated
warmth marker lands in a comment otherwise empty of the positive register**, which
is why raising marker presence alone would not move the conditional.

## What v99 should be

Not a warmth-marker schedule (rejected above). Not negative-marker suppression
(would hurt). The design the evidence supports:

**A drawn positive-register realization for slots the plan already assigned
`polite`, conditioned on size band, using the per-slot hash draw whose realization
fidelity is already proven.** `sentence_rhythm` moved seven habits from
0.000–0.299 to their measured rates within sampling noise, because it names a
concrete surface act rather than describing a register. `TONE_DEFINITIONS["polite"]`
describes a register in prose and realizes at 19.3%.

Cues should target the measured deficits, and their count should scale with the
band, because the deficit does:

- an ordinary intensifier on a positive judgement (`very`, `really`) — 0.19×
- one plain committed positive verdict word (`good`, `great`, `love`) — 0.21–0.33×
- something of the speaker's own named with `my` — 0.51×
- on long slots, more of them, since the 120+ band carries a +13.1 deficit

## Risk register for the build

| metric | current | risk |
|---|---|---|
| `self_bleu_4` | weak PASS, Cliff +0.42 | **the main risk.** A fixed cue vocabulary repeats. The draw must name the *act* and let the word vary per slot, as `sentence_rhythm` does; never put a literal phrase in the cue. |
| `emotion_entropy` | PASS, Cliff −0.16 | positive affect words could concentrate dominant emotions on `approval`/`admiration` rather than spreading them. Check the histogram, not just the entropy. |
| `mean_story_probability` | PASS, Cliff −0.12 | `my` plus past tense raises it; currently slightly low, so this should help. |
| `hard_disagree_rate` | PASS, Cliff +0.29 | making planned-polite slots actually polite lowers disagreement, and Cliff is too high, so this should help. |

## Reproducing every number above

`generalized_card/analysis/politeness_diagnosis.py`, committed with v99 so this
file's evidence is reproducible rather than described. No API calls, no model
loading; the real per-comment labels come from the evaluation classifier's own
`politeness_results.json` tables already under `data/raw/discussions/`.

```bash
python3 generalized_card/analysis/politeness_diagnosis.py all
```

| subcommand | what it establishes |
|---|---|
| `markers` | REJECTED — marker frequency; the log-odds derivation and the 0.088 counterfactual |
| `experience` | REJECTED — first-person experience; the eight features and the flat cell table |
| `dismissive` | REJECTED as a cause; the three families and the counterfactual, plus the two surviving tells |
| `lexical` | CONFIRMED — the TF-IDF model, the +8.381 / −0.767 decomposition, per-1k rates (needs scikit-learn) |
| `bands` | CONFIRMED — the gap grows monotonically with length |
| `realization` | CONFIRMED — the plan/label confusion matrix by band |
| `moves` | the shipped profile, and each excluded candidate with its reason |

`--run <dir>` points it at another run, so the same diagnosis can be re-measured
against the v99 gate output without editing anything.
