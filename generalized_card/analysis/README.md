# Analysis scripts

Zero-cost, offline analyses that produced the numbers quoted in
`tasks/v*-worklog.md`. They exist so those numbers are reproducible rather than
described -- the same reason a generator version has to be committed before it
runs (`docs/ORIENTATION.md` §8).

None of these makes an API call. Each reads run artifacts and the per-comment
classifier tables already under `data/raw/discussions/`, and each keeps the
150-thread evaluation seed pool out of anything it fits.

| script | what it establishes |
|---|---|
| `politeness_diagnosis.py` | the whole v99 diagnosis: four rejected hypotheses, the lexical decomposition, the plan/realization confusion matrix, and the per-band move profile |

Run one with `--help` for its subcommands.
