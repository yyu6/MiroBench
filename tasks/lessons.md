# Lessons

## 2026-08-13 — Read the whole related codebase before diagnosing, not slices of it

**What happened.** Diagnosing why generated long comments came out at ~0.72x their
matched length, I read `prompts.py`, `backend.py`, and `length_policy.py` in
partial line ranges and surveyed the 9,290-line
`scripts/sampling_generator/run_sampled_reddit_generator.py` only through an
outline and greps. From those slices I concluded the beat budget in
`long_form_planning.expected_development_beats` (one beat per 80 words) was the
cause, changed it to one per 35 words, spent an API run, and measured no
improvement (0.70 -> 0.72).

The real cause was visible only by reading a file I had never opened in full:
`reply_planning.render_direct_reply_planner_prompt` has its own compact JSON
schema that simply **omits** `development_plan`. Every long slot at depth >= 1
(33 of 77 long slots) therefore received no development guidance at all. The
beat budget was never the binding constraint for those slots.

**Why it matters.** A partial read produces a plausible-looking cause. Acting on
it costs a full generation run plus an evaluation run, and the null result is
ambiguous — it looks like "the fix was too weak" rather than "the fix was aimed
at the wrong thing." I also reported an improvement (0.70 -> 0.84) that was an
artifact of a double-counting bug in my own analysis script, and had to retract it.

**How to apply.**
- Before proposing a cause for behavior in this repo, read every file on the
  active path end to end, not the region a grep pointed at. In this codebase
  that means the CLI, the backend adapter, *every* prompt builder (there is more
  than one per role — root planner, direct-reply planner, low-info writer), the
  shared generator, and the policy modules.
- When several prompt builders exist for one role, diff their schemas against
  each other. A field present in one and absent in another is a likely defect.
- Verify an analysis script against a ground-truth count before trusting its
  aggregates. `generation_records.json` has one record per comment *and* each
  record nests its replies, so recursing into `replies` double-counts.
- Change one mechanism per API run, and predict the expected magnitude before
  spending the run so a null result is interpretable.
