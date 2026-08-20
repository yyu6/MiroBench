# Analysis scripts

Zero-cost, offline analyses that produced the numbers quoted in
`tasks/v*-worklog.md`. They exist so those numbers are reproducible rather than
described -- the same reason a generator version has to be committed before it
runs (`docs/ORIENTATION.md` §8).

`disagreement_diagnosis.py ablate` loads the local Stance_Rel checkpoint on CPU
(about six minutes for the v101 run); every other subcommand is seconds.

None of these makes an API call. Each reads run artifacts and the per-comment
classifier tables already under `data/raw/discussions/`, and each keeps the
150-thread evaluation seed pool out of anything it fits.

| script | what it establishes |
|---|---|
| `politeness_diagnosis.py` | the whole v99 diagnosis: four rejected hypotheses, the lexical decomposition, the plan/realization confusion matrix, and the per-band move profile |
| `disagreement_diagnosis.py` | the whole `hard_disagree_rate` diagnosis: where the gap lives (reply pairs, not roots), what the stance head actually reads, the opener plan/realization matrix, the parent-echo measurement, nine rejected hypotheses, and an exact ablation harness that reproduces the shipped artifact before it edits anything |

Run one with `--help` for its subcommands.
