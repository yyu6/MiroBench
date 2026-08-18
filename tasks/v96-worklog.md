# v96 worklog — selective factual grounding and ancestor-aware replies

Date: 2026-08-18

Status: zero-API gate complete; one paid seed-2 content gate is next. N=10 is
still blocked.

## Why v96 is necessary

The paid v95 seed-2 gate completed 45/45 comments in one attempt: 86 requests,
303,941 input tokens, 26,702 output tokens, $0.3481, and 301 seconds. This
confirms the v95 reliability repair. It does not confirm the content hypothesis.

The exact matched n=1 audit found:

- self-BLEU `0.0350` generated vs `0.0268` real;
- self-BERTScore `0.5306` vs `0.4892`;
- mean story probability `0.1015` vs `0.2321`;
- emotion entropy `1.6572` vs `1.9687`;
- digit-bearing comments `0.20` vs `0.60`;
- domain-vocabulary comments `0.1556` vs `0.5556`;
- distinct model designators `5` vs `40`;
- repeated 4-grams `0.0335` vs `0.0200`.

Length CV, average depth, and structural virality were already matched. The
n=1 MWU/KS values are descriptive only and are not treated as a statistical
pass.

## Root-cause trace

1. `--own-fact-license named` reached the Writer, but the active rule said not
   to repeat a name or figure used by another comment. Real replies naturally
   repeat the product name while changing the fact, condition, or stance. The
   rule suppressed shared domain vocabulary and pushed the Writer toward vague
   paraphrases.
2. With `--domain-claim off`, the root Planner saw evaluation-excluded reference
   text but was required to output no factual field. The Writer commonly
   received only one or two seed tokens (for example `Sony` or `VII`) and no
   safe factual payload. Asking it to be concrete could not create a reliable
   factual path.
3. The direct-reply Planner deliberately saw no evaluation-excluded reference
   rows at all. It could vary an abstract delta type, but it could not safely
   introduce diverse domain particulars.
4. Deep replies excluded only their immediate parent. In the seed-2 output,
   S37--S45 used different local delta labels while repeatedly returning to the
   same fixed-lens/commitment boundary.
5. Five story slots were planned and four were recognized as stories, so the
   primary story failure is not a dropped Planner label. The single matched
   thread's story rate also differs from the held-out template; v96 will not
   overfit story counts to one seed.

## Intended repair

- Preserve `domain-claim=off` and the historical ubiquitous `planned` arm.
- Add a distinct `selective` mode. Only capacity-compatible slots paired with
  a useful evaluation-excluded reference row may carry one Planner-restated
  general domain fact. Raw reference wording remains Planner-only.
- Give direct-reply planning its own source-diverse, evaluation-excluded
  reference window in selective mode.
- Show a reply the compact semantic coverage of its full ancestor chain, not
  just its parent.
- Permit normal reuse of the discussion's product names while prohibiting reuse
  of the same fact or number.
- Give named, first-person slots the same evaluation-excluded rotating equipment
  shortlist already available to the narrower `own` arm.

## Verification gate

Before any paid call: focused unit/integration tests, the full generalized-card
suite, Ruff, source pins, runtime parity, backend self-test, and an exact seed-2
prepare-only replay must pass. After that, run one paid seed only. N=10 remains
blocked until the generated discussion passes both artifact-health and content
review; a successful process exit alone is insufficient.

## Zero-API result

- 316 generalized-card tests passed, including selective schedule enforcement,
  direct-reference delivery, full-ancestor exclusions, Writer-anchor survival,
  and n=1 version-log handling.
- Ruff passed over the complete `generalized_card/` tree.
- Source contract: 95/95 files present and clean, no untracked active source,
  no unpinned local import, and zero hash drift.
- Active and active-plus-legacy parity both passed with no unexpected backend
  functions.
- Camera backend self-test passed with `domain-claim=selective` and
  `own-fact-license=named`.
- Exact seed-2 `--prepare-only` passed as
  `generalized_card_camera_gpt54_v96_selective_seed2_20260818_preflight_v2`;
  policy and all requested flags were preserved and no API call was made.
- The version index now marks n=1 output as `descriptive` instead of falsely
  counting twelve p-value passes.
