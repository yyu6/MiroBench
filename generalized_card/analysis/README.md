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
| `digit_cue_diagnosis.py` | the v106 `--digit-cue-guard` numbers: bare `0`/`1` rate, generated vs 424 evaluation-excluded real camera threads, split into "plain quantifier" ("1 thing") vs "enumerated/fractional/price" sub-patterns. The excess is 8.2× on the sub-pattern that doesn't serve the digit cue's own purpose, 1.7× on the genuine one -- not a uniform digit-rate gap. `generalized_card/VERSION_LOG.md` v106, `docs/DECISIONS.md` G12. |
| `template_reuse_diagnosis.py` | tests, and rejects, "generic sentence-template reuse" as an explanation for `self_bertscore_mean_f1`'s gate result: within-thread opener/closer clause embedding near-duplicate rate, generated vs matched real vs an 80-thread real null. Generated is not elevated over either. `docs/DECISIONS.md` G13. |
| `verdict_close_diagnosis.py` | the v107 `--verdict-close-guard` numbers: `closing_move.py`'s already-known `abstract_verdict_close` tic is still 10-13× real even where its v100-era suppression cue reaches the Writer, and a "that's the check"/"a solid check" variant the existing pattern never named adds 13-37× on top. `generalized_card/VERSION_LOG.md` v107, `docs/DECISIONS.md` G14. |
| `cross_domain_reply_diversity.py` | checks whether "real replies diversify with depth more than root comments do" (G3) generalizes past camera. Runs the cheap `all-mpnet-base-v2` cosine proxy over the real, evaluation-excluded corpus of all four registered domains (camera, cell_phone, headphone, laptop) -- no domain has ever had a paid generation run, so this is real-only. Confirms the direction in all four, each strongly significant, plus a depth-binned cosine curve per domain. `docs/DECISIONS.md` G17. |

Run one with `--help` for its subcommands.
