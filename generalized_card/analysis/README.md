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
| `bertscore_pair_diagnosis.py` | the `self_bertscore_mean_f1` pairwise decomposition (`docs/DECISIONS.md` G3): fidelity-checks against the shipped per-thread means, then classifies every within-thread pair by tree relation and by root/reply role. `inspect` reads the actual highest/lowest-`bert_f1` pairs with their text on both sides -- this is what found the shared-image-URL artifact reconfirmed on real, and genuine argument-paraphrase duplication on generated. Runs the real BERTScore model (`microsoft/deberta-xlarge-mnli`) on CPU over the v103 N=10 artifact's 10 threads plus their 10 matched real threads only -- minutes, not seconds, but still no API call. Use system `python3` (`transformers==4.48.0`), not `.venv` (`5.10.1`), to match the shipped artifact's model hash. |
| `root_reply_diversity.py` | checks whether "real replies are more diverse than real root comments" (the G3 sign-inversion finding) generalizes past the ten v103-matched threads. Uses the cheap `all-mpnet-base-v2` cosine proxy (`semantic_mean_cosine`'s model, not BERTScore) over all 424 evaluation-excluded camera threads -- seconds, not minutes, and the right tool for testing a text property's direction at scale rather than reproducing `self_bertscore_mean_f1` itself. |
| `reply_novelty_chain_diagnosis.py` | falsifies the v105 `--reply-novelty-scope chain` fix before (and after) it shipped: replays the real `reply_increment_problem`/`PlanSemanticIndex` over the v103 artifact's actual Planner plans, both scopes, and reports how many replies trip the novelty contract. Found the pre-fix probe scored 0 trips everywhere; the probe-shape fix plus chain walk finds 60. `generalized_card/VERSION_LOG.md` v105, `docs/DECISIONS.md` G11/E9. |

Run one with `--help` for its subcommands.
