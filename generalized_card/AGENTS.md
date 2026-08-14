# Generalized CARD Engineering Rules

These rules apply to every file under `generalized_card/`. They supplement the
repository-level instructions.

## Trace The Active Path First

- Before changing behavior, trace the complete runtime path from the public CLI
  through argument and environment resolution, backend configuration, monkey
  patches, the shared generator or reviser, artifact persistence, cleanup, and
  evaluation.
- Read every implementation and call site on that path. Include subprocess
  entrypoints, dynamically assigned functions, snapshot compatibility checks,
  tests, and resume behavior. A definition is not evidence that code executes.
- State whether a finding is active behavior, inactive legacy code, a test-only
  helper, or a proposed design. Do not present a proposed fix as implemented.
- When generalized code wraps CARD code, verify both the original function and
  the active wrapper. Record which implementation wins at runtime.

## Keep Modules Cohesive

- Keep CLI files responsible for configuration and orchestration only. Keep
  backend adapters responsible for wiring and domain boundaries only.
- Put separable policy in focused modules. Examples include task-distribution
  calibration, story/affect allocation, lexical-quality checks, Writer local
  repair, revision memory, and metric projections.
- Do not add a substantial independent algorithm to the shared generator,
  `backend.py`, or `prompts.py` when it can be expressed behind a small typed
  interface in a dedicated module.
- Keep prompts close to the policy that supplies their structured controls, but
  do not make prompt text the only enforcement mechanism for a hard invariant.
- Preserve CARD behavior intentionally. Any generalized deviation must be named,
  justified, tested, and recorded in the policy/version metadata.

## Scientific Integrity

- Build domain profiles only from threads excluded from the evaluation seed
  pool. Preserve source hashes and verify zero seed/reference overlap.
- Never expose matched evaluation comment text to the Writer. Structural labels
  and leakage-safe reference abstractions must remain distinct.
- Do not tune generation against final test-set p-values. Calibrate controls on
  excluded reference data and report MWU, KS, Cliff's delta, and Wasserstein
  distance without hiding failures.
- Preserve run configuration, source provenance, cost, elapsed time, retry
  history, and accepted/rejected decisions in resumable artifacts.
- For first-pass Planner--Writer generation, preserve every matched structural
  slot.  Never shrink, cap, or omit a matched thread by default; any deliberate
  size ablation must be explicit in the run configuration.
- Make the Planner and one held-out reference template the sole owners of a
  slot's role, payload, tone, story, affect, and semantic controls. Do not let
  a later legacy surface pass rewrite those fields or inject fixed discourse
  words such as acknowledgements, jokes, or ellipses.

## Semantic Checks And Regex

- Use regex for syntax, schema residue, literal anchors, formatting, and narrow
  surface-pattern checks. Do not use regex as ground truth for story content,
  emotion, semantic equivalence, politeness, or discourse function.
- Use length-aware token/ngram checks or the same validated model family as the
  evaluation metric when semantic or distributional behavior matters.
- Derive thresholds from excluded reference data, conditioned on relevant
  factors such as thread size, comment length, and thread archetype. Do not
  encode examples from failed evaluation threads as one-off exceptions.
- Log every quality-guard diagnostic, hard failure, bounded recovery, fallback,
  and final decision. Collection-level lexical and semantic distribution
  diagnostics must inform the upstream plan and final evaluation; they must not
  drive repeated Writer sampling or best-of-N selection for one comment. Only
  hard schema, safety, factual-grounding, and empty-output failures block a
  comment or invoke bounded recovery.

## Dead Code

- Before removing apparently unused code, check repository-wide references,
  imports, reflection, monkey-patch assignments, subprocess entrypoints,
  reproducibility snapshots, and tests. Confirm the active runtime path.
- Delete code only after that audit and after updating its tests, contracts, and
  documentation. Do not keep commented-out implementations in production code;
  version control is the archive.
- Treat an accepted CLI argument whose value is never consumed as a correctness
  bug, not harmless cleanup. Add an end-to-end test that proves the argument
  changes the intended behavior.

## Reliability And Tests

- Persist completed post slots atomically and keep resume deterministic. A hard
  local Planner or Writer failure may retry the smallest safe unit within an
  explicit finite budget without losing healthy completed work. Do not use
  retries as a distribution-metric optimizer.
- Add focused unit tests for new policy modules, an integration test proving the
  active monkey-patched implementation, and a smoke test that exercises the
  public CLI.
- For distribution controls, test both directions, boundary sizes, short and
  long comments, low-information slots, and preservation of entities, numbers,
  parent relations, story status, and tree structure.
- Run the scoped test suite and the backend self-test before giving a generation
  command. Report anything that was not tested.
