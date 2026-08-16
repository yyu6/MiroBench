# Generalized CARD v81 worklog — 2026-08-17

This is a live engineering record. It distinguishes measured behavior from a
proposed change so an interrupted session can be resumed without guessing.

## Acceptance target

The final target is distributional agreement between generated and matched-real
threads on all 12 published thread metrics. A metric passes only when both the
two-sided Mann–Whitney U and two-sample KS p-values exceed 0.05 on a sufficiently
large group. One generated thread is useful for content diagnosis, but its two
p-values are mechanically 1.0 and are not evidence of agreement. Politeness is
the lowest-priority metric; it must not be improved by sacrificing the other 9
content metrics or the 2 structural metrics.

## What the 12 rows actually measure

1. `self_bleu_4`: mean over all unordered within-thread comment pairs of
   symmetric BLEU-4, with add-one modified n-gram precision. Lower means less
   shared wording and sentence scaffolding.
2. `self_bertscore_mean_f1`: mean BERTScore F1 over all unordered comment pairs.
   Lower means less pairwise contextual/semantic similarity; formatting also
   affects it.
3. `semantic_mean_cosine`: mean all-pairs cosine between normalized MPNet
   comment embeddings. Lower means broader semantic coverage.
4. `hard_disagree_rate`: fraction of usable parent→reply pairs whose stance
   classifier argmax is `disagree`. Direction is not intrinsically good; match
   the real distribution.
5–7. `polite_rate`, `impolite_rate`, `neutral_rate`: fractions of comments whose
   four-way polite-guard argmax is the named class. `somewhat_polite` is a real
   fourth class even though it is not one of the 12 reported rows.
8. `length_cv`: population standard deviation of whitespace word counts divided
   by their mean, per thread.
9. `avg_depth`: mean BFS depth of comments, with top-level comments at depth 1.
10. `structural_virality`: mean shortest-path distance over connected unordered
    comment pairs in the undirected reply graph.
11. `mean_story_probability`: mean StorySeeker story probability over comments;
    this is not the hard story rate.
12. `emotion_entropy`: natural-log Shannon entropy of the histogram of each
    comment's GoEmotions argmax label. It is not entropy of threshold labels or
    mean probabilities.

The group evaluator applies unpaired MWU and KS tests to the per-thread rows and
also reports Cliff's delta and 1-Wasserstein distance. Matching a frozen
evaluation-excluded template in one run can differ from that run's particular
matched seed; only the multi-thread distribution decides the formal result.

## Current v80 evidence

- Frozen template for seed 8: story mean `0.1281`, emotion entropy `1.5359`.
  Generated: story mean `0.2488`, entropy `1.5358`. Emotion entropy hit this
  run's template almost exactly; story did not.
- Of 169 `no_story` outputs, 33 were hard stories and their mean story
  probability was `0.215`. `allow_first_person_frame=True` had mean story
  probability `0.468`, versus `0.066` when false.
- The causal concentration is Planner metadata: 61 slots used
  `firsthand_experience` although only 17 story slots were scheduled. The
  no-story contract rejected only `payload_type=personal_story`; it did not
  reject firsthand evidence or require story slots to use narrative evidence.
- Direct-reply planning displays story/affect/opener assignments but omits those
  fields from its JSON schema and gives no story compatibility rule. The fixed
  labels are added only after parsing, so the Planner often designs an
  incompatible semantic move.
- 104 short direct replies carried a copied schema placeholder as a real
  `development_plan`: 94 had `none for a short slot`; 10 also copied the rest of
  the example. The Writer interpreted it as a content beat. This is a direct
  source of shared prompt scaffolding, excess explanation, and repetition.
- Deterministic post-parse normalization rewrote every gratitude/relief move to
  the same sentence and converted incompatible substantive payloads to
  `soft_helpful`. That violates Planner ownership and creates four identical
  social-close contracts plus ten avoidable helpful payloads.
- Tone and affect are scheduled as marginals but several assigned pairs are
  semantically contradictory. Examples in v80 include 10 `approval+impolite`
  slots and 27 `neutral-affect+polite` slots. The Writer cannot preserve both,
  which contributes to emotion-label collapse.
- The focused Writer repeats the complete tone definition in two blocks. Its
  neutral affect rule also says both “keep emotional signal low” and “make the
  reaction legible through an evaluation/interjection/hedge.” These are prompt
  conflicts, not missing instructions.

## v81 changes and gates

All implementation gates below are complete. A paid generation/evaluation is
still required to measure the output distribution.

- DONE: make story/no-story a joint Planner contract in both directions.
- DONE: include fixed story/tone/affect/opener controls in direct-reply
  planning and condition reply-delta choice on them.
- DONE: clear development plans structurally when a slot has no long-form
  capacity; change the schema example to literal `none`; remove the root
  prompt's conflicting 35-word/16-beat formula so both planners display counts
  from `expected_development_beats`.
- DONE: remove semantic post-parse rewrites; retain only deterministic enum
  and anonymous-structure normalization.
- DONE: pair affect marginals with compatible tone/story slots before the
  Planner sees them.
- DONE: remove duplicated/conflicting instructions from focused, low-info, and
  full Writer prompts. A second review also caught and removed a conflict
  between neutral affect and impolite tone.
- DONE: allow rough but non-abusive Reddit surfaces in the assigned contracts:
  ordinary non-targeted profanity for impolite reactions and natural laughter
  tokens for amusement, without requiring a stock word or phrase.
- DONE: audit the first-pass candidate lifecycle against the rule that
  distribution diagnostics do not trigger per-comment best-of-N selection.
- DONE: delete the now-unreachable metric-repair strategies, candidate ranking,
  blocking repetition guard, dead CLI flags, and their historical-only tests.
  The Writer lifecycle now has one realization plus bounded recovery only for
  output that cannot be persisted.
- DONE: update tests, source pins, version docs, TODO, lessons, and handoff.

## Verification completed

- `ruff` passes on all changed generalized-card production and test modules.
- `259 passed` for the complete `generalized_card/tests` suite.
- `[generalized-self-test] PASS domain=camera_product`.
- Contract audit checks 72 pinned files: 0 missing, 0 drifted.
- The exact seed-8 generation command passes `--prepare-only`, resolves the
  v81 policy/config/source snapshot, and makes no API calls.
- Offline reconstruction of the 186-slot v80 structure assigns all 17 story
  slots, all 186 tone labels, and all 186 affect labels, with no
  story+gratitude/relief collision and no `approval+impolite` pairing.
- The source change removes more code than it adds: the scoped diff deletes the
  metric-guided retry system instead of layering another controller over it.

No paid run should start until unit tests, the backend self-test, source pins,
prompt snapshots, and an offline plan-field survival audit all pass.
