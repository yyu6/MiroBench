# Generalized CARD v84 worklog — 2026-08-17

## Scope

Complete the current-path audit from bounded Writer recovery through post
persistence, output audit, and evaluation. Preserve every matched structural
slot without adding a canned fallback comment, metric-driven Writer resampling,
or hidden default API cost.

## Evidence

The paid v80 seed-8 artifact contains:

- 186 planned Writer tasks and generation records;
- 185 rendered comments;
- one skipped record, S99;
- three S99 attempts, all rejected for `parent_copy` (and repeated opening
  diagnostics);
- an output audit acceptance share of 185/186 = 0.9946.

S99's Prompt requested an exact parent-line quote. The same active Writer path
classifies parent copying as a hard realization failure. Current code still
scheduled the `quote` opener, rendered the exact-line instruction, persisted
valid comments after a skip, and allowed evaluation above a fractional acceptance
threshold. Each link was verified in the current source rather than inferred
from version history.

## Changes

- Reword the quote opener as a short exact markdown excerpt, not a whole-parent
  copy.
- Add a narrow syntax check for a Planner-assigned quote followed by at least six
  words of distinct reply. Only this form can remove `parent_copy` from the final
  hard-problem set.
- Require exact record/task/comment equality in `generation_coverage` and raise
  before post persistence when it fails.
- Make output audit independently reject recorded posts with unequal planned,
  recorded, generated, or rendered counts, or any skipped record.
- Correct obsolete run-config descriptions and delete an unreachable omission
  branch. Git and policy history remain the archive.

## Verification ledger

- Focused regression tests: 4 passed; complete suite: 266 passed.
- Existing v80 artifact replay: correctly rejected at 185/186 despite 99.46%
  accepted share.
- Ruff on changed production modules: passed.
- Camera-product backend self-test: passed.
- Planned quote-copy waiver is explicitly present in the persisted Writer
  distribution audit row, not only in the in-memory selection result.
- Source pins: 72 checked, 0 missing, 0 drifted.
- Exact formal v84 seed-8 command: `--prepare-only` passed with no API calls;
  the temporary directory was moved to macOS Trash and the formal tag is free.

## Expected result

Every artifact admitted to evaluation has exact matched structural coverage.
This change protects the interpretation of all 12 metrics; it does not claim
that content metrics will improve without a new paid artifact. The shorter quote
form may reduce lexical repetition, but that is a secondary hypothesis to
measure, not a promised result.
