# Generalized CARD v83 worklog — 2026-08-17

## Scope

Trace every function called by `expand_matched_real_sample_to_tasks` that receives
the anonymous matched comment body. The allowed information is structural:
reply topology, word scale, punctuation, dominant link/quote form, identifier
typography, and optional speaker participation structure. Evaluation wording may
not assign a semantic, story, stance, tone, affect, or evidence control.

## Findings

v82 removed lexical gratitude from `infer_surface_texture`, but three indirect
paths remained:

1. `_allows_first_person` and `_allows_uncertainty` parsed matched wording. The
   Planner restore later overwrote both flags on healthy planned slots, so these
   regexes were dead on the required path and dangerous if restoration regressed.
2. `infer_surface_shape` labelled every 70+ word matched slot `story_rant`. That
   label can enter full Writer controls even for a `no_story` plan.
3. The same function classified lexical `side note/unrelated/FWIW/BTW` as a
   `side_tangent`, and `!template` in ordinary user text as template semantics.

## Changes

- Replace both lexical frame callbacks with one explicit boundary that always
  returns false. Planner restoration/distribution assigns the final flags.
- Rename long and identifier-bearing shapes to neutral structural labels and
  remove lexical tangent/template detection. Moderator author metadata and
  deleted placeholders remain structural special cases.
- Explicitly discard the `real_body` argument in the generalized anchor builder,
  documenting and enforcing that it cannot contribute Writer-visible facts.
- Delete unreachable generalized handling for lexical real-surface
  `thanks_ack`, `joke_reaction`, and `side_tangent` labels. Planned gratitude,
  jokes, and tangents remain live through payload/role controls; only the
  matched-text-derived aliases were dead.

## Tests and expected result

The end-to-end Planner-field test now uses matched wording containing `Side
note`, `Thanks`, `I think`, `maybe`, and `could`. The resulting planned rant is
still `full_answer + plain`, has neither semantic frame flag, and receives no
gratitude tone. Separate tests prove 80 words becomes `long_turn`, lexical
`!template` stays `full_answer` for an ordinary author, and moderator metadata
still produces `template_notice`.

The direct-reply end-to-end test was also strengthened: a planned
`correction + contrarian + blunt + hard_disagree` reply proves that its reply
delta and compact discourse fields each enter the focused Writer exactly once
after expansion and finalization.

Verification gates:

- complete `generalized_card/tests`: **263 passed**;
- Ruff passes on all changed generalized production and test modules;
- `[generalized-self-test] PASS domain=camera_product`;
- 72 pinned sources checked, 0 missing, 0 drifted;
- exact v83 seed-8 command passed `--prepare-only` with policy
  `generalized-card-v2-matched-text-semantic-isolation-v83-20260817`; no API
  calls were made and the formal tag remains unused.

This eliminates confounding and prompt conflicts; it does not by itself prove a
12-metric improvement. A paid v83 artifact and then a multi-thread evaluation
remain required.
