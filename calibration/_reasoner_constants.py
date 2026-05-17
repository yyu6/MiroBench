"""Static text constants for the calibration reasoner prompts.

Extracted from reasoner.py to keep file size manageable. All values are
pure data (mostly large multi-line string templates injected into LLM
prompts).
"""
from __future__ import annotations


_REQUIRED_TEXT_KNOBS = (
    "persona.generation_guidance",
    "prompt.comment_style_guidance",
)

_MECHANISM_FAMILIES = (
    "semantic_diversity",
    "story_anecdote",
    "tone_civility",
    "length_variation",
    "structure",
)

_STAGNATION_TRIGGER = 3


# ---------------------------------------------------------------------------
# Static calibration knowledge injected into every reasoner prompt
# ---------------------------------------------------------------------------

METRIC_INTERPRETATION = """
## Metric Interpretation Guide

For each target metric, the goal is to drive |Cliff's delta| and Wasserstein
distance toward 0.  Do NOT use p-values.

Priority order for judging candidates:
1. abs_cliffs_delta (PRIMARY — lower is better, 0 = distributions match)
2. Wasserstein distance (PRIMARY — lower is better, 0 = perfect shape match)
3. quantile_error (supporting — lower is better)
4. abs_median_gap (supporting — lower is better)

The per-metric diagnostic section shows direction (generated_higher or
generated_lower) and tier (CRITICAL / SECONDARY / acceptable).  Use direction
to decide whether to push a raw metric up or down; use tier to prioritize
which metrics to fix first.
""".strip()

CALIBRATION_COMPARISON_STATS_GUIDE = """
## Calibration Comparison Statistics Guide

Use these calibration-only comparison statistics directly when reasoning about
candidate quality.

### The main statistics

- abs_cliffs_delta:
  Absolute Cliff's delta effect size between generated and real values.
  PRIMARY — lower is better, 0 = distributions match.

- Wasserstein distance:
  Overall shape mismatch between generated and real distributions for one metric.
  PRIMARY — lower is better, 0 = perfect match.

- quantile_error:
  Mean absolute gap between matched real/generated quantiles
  (10%, 25%, 50%, 75%, 90%). Supporting — lower is better.

- empirical_fail_rate:
  Fraction of generated rows that are obvious outliers relative to the real
  distribution (using empirical quantile thresholds). Supporting — lower is better.

- abs_median_gap:
  Absolute difference between generated and real medians. Supporting — lower is better.

Secondary diagnostics (out_of_range_count, percentile_distance, robust_z) are
sanity checks for direction errors or extreme tail mismatches.

### How to use them during calibration

- Judge candidates first on the current block's focus metrics using
  |Cliff's delta| and Wasserstein distance.
- Then check quantile_error, empirical_fail_rate, and abs_median_gap as
  supporting signals.
- Do not rank candidates by one grand average across all metrics during the
  focused blocks. Use the active focus metrics first, then preserve protected
  metrics.

### Interpretation rules

- Compare statistics per metric or within the current phase's focus metrics.
  Do not compare raw magnitudes across unrelated metrics.
- For length and structure metrics, slight upward overshoot can be acceptable
  if it produces lower |Cliff's delta| and Wasserstein distance overall.
""".strip()

CALIBRATION_PRINCIPLES = """
## Calibration Principles for Small-Batch Robust Scoring

### 1. Reference distribution
Use the real validation distribution as the during-calibration reference.
- Train examples are qualitative context only.
- Candidate ranking should judge whether the simulated batch looks closer to the real validation set.
- Do not optimize candidates against the train set since train set is only just qualitative context.

### 1b. Non-negotiable end state
- Every candidate should be designed with the final end state in mind:
  each target metric should look as if it came from the same distribution as
  the real validation set.
- That means:
  - generated thread-level distributions should match real validation as closely as possible;
  - abs_cliffs_delta and Wasserstein distance should move toward 0 for each
    target metric — these are the PRIMARY optimization targets;
  - quantile_error / empirical_fail_rate / abs_median_gap are supporting signals.
- Do NOT optimize for p-values.  Focus exclusively on driving Cliff's delta
  and Wasserstein distance toward 0.
- In early focused blocks you optimize only a subset of target metrics, but the
  subset should still be moved in the direction of that final end state.

### 2. Failure prioritization
Start from the active phase's target metrics, not from a global family average.
- Compare each active target metric on its own statistics.
- Look first at abs_cliffs_delta and Wasserstein distance for that metric —
  these are the two primary signals.
- Then look at quantile_error, empirical_fail_rate, and abs_median_gap as
  secondary supporting signals.
- Then use percentile_distance, robust_z, and out_of_range as directional
  sanity checks.
- Do not look at p-values for candidate ranking.
- Do not hide one bad metric behind improvements in other metrics.

### 3. Expected-improvement reasoning
Before choosing candidate directions, balance four factors:
- failure severity,
- controllability by persona.generation_guidance and prompt.comment_style_guidance,
- evidence from previous successful or partially successful edits,
- risk of worsening already acceptable metric groups.

Do not blindly optimize the metric with the largest failure signal.
Some metrics may be hard to move with prompt/persona edits alone, while others may respond more directly to changes in persona mix, comment shape, reply behavior, anecdote frequency, length variation, or tone.

Use past calibration history when available.
- If repeated edits toward aggression, average length, or reply depth produced little or no improvement, do not keep making the same edit stronger.
- Prefer candidate directions that showed even small positive movement before.
- Also consider plausible causal mechanisms that have not been tried yet.
- If a stubborn metric does not respond to direct instructions, target an indirect behavioral cause instead.

Example:
If avg_depth does not improve by simply asking for more replies, change the persona mix so users naturally disagree, ask follow-up questions, correct each other, or respond to narrow product claims.

### 4. Causal hypothesis requirement
Every candidate must be based on a clear causal hypothesis.
For each candidate, identify:
- which failure pattern is being targeted,
- what behavioral mistake is causing the mismatch,
- how persona generation should change,
- how comment/reply writing should change,
- which metrics should move and in which direction,
- which side effects should be avoided.

Do not write candidates based on vague realism, vibes, or generic diversity.

### 5. Only two editable knobs
There are exactly two editable knobs.

persona.generation_guidance controls WHO gets generated:
- user backgrounds,
- product-specific memories,
- prior experiences,
- repeated grievances,
- confidence level,
- taste biases,
- practical constraints,
- what they tend to notice, defend, regret, or complain about.

prompt.comment_style_guidance controls HOW they write:
- comment shape,
- reply behavior,
- disagreement style,
- anecdote usage,
- specificity,
- length variation,
- bluntness,
- whether they react to visible comments or post standalone takes.

Do not invent other knob names.
Every candidate must write both knobs.
One knob can be dominant, but both must be substantive and coordinated.

### 6. Patch quality requirements
The two knob values are actual content patches that will be injected.
They must be operational, not vague.

Each block should explain:
- what to do,
- what not to do,
- why it should help,
- what concrete outputs should look like.

Avoid weak edits.
Bad:
- "be more direct"
- "be more realistic"
- "be more diverse"
- "make comments natural"

Good:
- concrete rules,
- anti-patterns,
- example comment shapes,
- visible causal links to the failing metrics.

### 7. Metric-to-behavior mapping

Semantic diversity too repetitive / too homogeneous:
- If self_bleu, self_bertscore, or semantic cosine are too_high, increase variety in motives, product memories, personal constraints, disagreement angles, and comment shapes.
- Make different users care about different things for different reasons.
- Do not solve this with empty instructions like "be diverse."

Semantic diversity too scattered:
- If semantic metrics are too_low, strengthen thread anchoring.
- Make comments react to one visible claim, one product detail, one tradeoff, or one use case.

Story/anecdote too low:
- Add short, believable lived datapoints.
- Examples: application result, fee annoyance, return story, battery issue, fit issue, travel habit, or repeated regret.
- Do not make every comment a diary entry.

Story/anecdote too high:
- Mix in blunt verdicts, corrections, questions, and short reactions.
- Keep personal stories narrow and functional.

Length variation too low:
- Create a mix of one-liners, blunt caveats, short disagreements, medium advice, and occasional longer stories.
- Avoid making every comment equally complete.

Length variation too high:
- Reduce rambling while preserving natural variation.
- Long comments should be occasional and purpose-driven.

Structure too flat:
- If avg_depth, max_depth, avg_branching_factor, or structural_virality are too_low, increase natural replies to visible comments.
- Replies should challenge parent claims, ask pointed follow-ups, add counterexamples, or correct narrow claims.
- Do not only say "reply more"; create personas and situations where replies are naturally triggered.

Tone too sanitized:
- Allow ordinary Reddit roughness: bluntness, skeptical pushback, dismissive disagreement, sarcasm, and one-line rejections.
- The goal is natural friction, not unsafe content.

Tone too hostile:
- Reduce direct attacks, excessive profanity, and pointless hostility.
- Keep disagreement grounded in the product, claim, or use case.

### 8. Candidate search behavior
Before proposing the 5 candidates, briefly reason about actionability:
- Which failing metric groups are most severe?
- Which of them are realistically controllable through persona.generation_guidance and prompt.comment_style_guidance?
- Which previous edit directions helped, partially helped, or failed?
- Which candidate directions should be avoided because they repeat failed strategies?
Then allocate candidate slots toward the highest expected improvement, not only the largest failure signal.

Use the 5 candidate search slots as instructed in the Task section.
- Early iterations should explore distinct causal mechanisms.
- Later iterations should exploit, combine, or refine directions that showed improvement.
- Do not repeat failed strategies unless the new candidate changes the underlying mechanism.
- Prefer candidates with high expected actionability over candidates that only target the largest failure.

### 9. Candidate selection objective
Prefer candidates that, for the active target metrics:
1. reduce abs_cliffs_delta toward 0 (PRIMARY),
2. reduce Wasserstein distance toward 0 (PRIMARY),
3. reduce quantile_error (supporting),
4. reduce empirical_fail_rate (supporting),
5. reduce abs_median_gap (supporting),
6. avoid worsening already-protected metrics' Cliff's delta and Wasserstein,
7. build on partial wins from previous candidates in the same phase block.

The ultimate goal is to drive EVERY metric's |Cliff's delta| and Wasserstein
distance to 0 — meaning the generated distribution is indistinguishable from
the real validation distribution.  Do NOT use p-values for ranking.

The final candidate should be comprehensive, actionable, and easy for the downstream simulator to follow,
while still moving every targeted metric closer to the real validation distribution.

### Domain-general constraint
- Do not hard-code domain-specific jargon or behaviors unless the current domain context supports them.
- The same calibration logic must work across credit cards, laptops, cellphones, cameras, headphones, and future product domains.
- Use general slots:
  1. ownership_or_usage_history
  2. purchase_context
  3. failure_or_success_event
  4. comparison_target
  5. decision_constraint
  6. usage_pattern
  7. technical_or_value_detail
  8. confidence_level
- When domain context is available, instantiate these slots with domain-appropriate details.
- When domain context is not available, use neutral product-discussion language rather than domain-specific jargon.
""".strip()

TEXT_MATERIALIZER_PRINCIPLES = """
## Text Materializer Principles

You are the second-stage calibration writer.
The strategist already decided WHAT to change. Your job is to write the actual
persona/prompt text that will be injected into the simulator.

Global objective:
- The final runtime text should push every target metric's generated
  distribution as close as possible to the real validation distribution.
- The text should help drive |Cliff's delta| and Wasserstein distance toward 0
  for the active target metrics — these are the PRIMARY optimization targets.
- quantile_error / empirical_fail_rate / abs_median_gap are supporting signals.
- Do NOT optimize for p-values.
- In early focused blocks, concentrate on the active metrics first, but write
  with the eventual full-distribution match in mind.

There are exactly TWO text knobs you can write:
- persona.generation_guidance: Controls what kinds of people get generated (their backgrounds,
  habits, conflict styles, motivations, knowledge levels, blind spots, product experiences).
- prompt.comment_style_guidance: Controls how agents write comments and replies (tone, structure,
  length variety, reply behavior, nesting, paraphrasing avoidance, example comment shapes).

Rules:
- Always write both knobs. One can be dominant, but both must contain substantive operational text.
- Treat the two knobs as coordinated patches:
  persona.generation_guidance changes who shows up;
  prompt.comment_style_guidance changes how those people write and interact.
- Treat the strategist overlay_diff as a SEED SKETCH, not as final copy.
- You must substantially rewrite and expand the seed into better runtime text.
- Do not copy long spans verbatim from the strategist. Re-express the idea in clearer, more operational wording.
- Do not include meta commentary like "this should improve self-BLEU" inside the injected text unless it is phrased as an operational instruction.
- Do not write one-line slogans such as "be more diverse", "be more realistic", or "sound like Reddit".
- Write substantive, operational text — NOT one-line slogans.
- Treat each knob value as a ready-to-inject patch, not as notes to another editor.
- A good text block is 1-3 paragraphs and includes:
  (1) the concrete behavior to create
  (2) the causal logic for why that behavior should improve the target metrics
  (3) anti-patterns to avoid
  (4) the kinds of anecdotes, disagreements, lived details, or reply shapes that should appear
  (5) 2-4 short example snippets showing the desired comment/persona shape
- If your output is basically identical to the strategist seed, it is not doing the materializer job.

Persona guidance:
- Shape the cast of people, not just the tone.
- The persona guidance must be domain-general and reusable across product/community domains.
- Specify repeated grievances, blind spots, certainty mismatch, taste biases, product/service memories, what triggers them to reply, what kind of narrow angle they keep bringing up, and what practical constraints they have.
- Use domain-adaptive lived details when relevant:
  ownership_or_usage_history, purchase_context, failure_or_success_event, comparison_target, decision_constraint, usage_pattern, technical_or_value_detail, confidence_level, support/repair/return/warranty experience, cost sensitivity, quality concerns, compatibility issues, comfort or usability problems, brand loyalty, bad prior purchases, or narrow use cases.
- Do not rely on one domain's fixed jargon or institutions unless the active examples clearly belong to that domain.
- Avoid making every persona equally informed, equally polite, equally helpful, equally balanced, or equally anecdotal.

Comment-style guidance:
- Shape the visible writing as a domain-general online product/community discussion, not as a helpful assistant response.
- The materialized prompt.comment_style_guidance must be reusable across domains such as credit cards, laptops, cellphones, cameras, headphones, and future product domains.
- Do not hard-code one domain's jargon, rules, or product assumptions unless the active sample threads clearly support them.
- Use domain-adaptive slots instead:
  1. ownership_or_usage_history,
  2. purchase_context,
  3. failure_or_success_event,
  4. comparison_target,
  5. decision_constraint,
  6. usage_pattern,
  7. technical_or_value_detail,
  8. confidence_level.
- When writing examples, use neutral placeholders like X, Y, product, model, feature, cost, warranty, battery, comfort, fee, support, repair, compatibility, or use case unless the active domain is clearly known.
- Explain how to choose one angle, when to reply directly to another visible comment, how to avoid paraphrase, how to disagree, how to vary length, and what realistic comment shapes look like.
- Encourage a mix of one-liners, blunt caveats, skeptical corrections, short personal datapoints, medium practical advice, and occasional longer explanations.
- If structure is weak, explicitly tell writers to reply to visible comments, challenge parent claims, ask follow-ups, and create back-and-forth rather than only top-level standalone answers.
- If repetition is high, tell writers to change comment function rather than paraphrasing the same thread consensus.

Required structure for materialized prompt.comment_style_guidance:
Every prompt.comment_style_guidance block should be organized around these domain-general runtime behaviors when relevant to the active phase:

A. Narrative / first-hand or observed experience
- Use compact real-user evidence.
- About 35-50% of comments may contain one short first-hand or observed datapoint when story/anecdote is underrepresented.
- Good datapoints include: owned or used the product/service before; upgraded or switched from a related option; had a specific failure or success; compared it with an alternative; used it in a concrete situation; encountered cost, repair, comfort, battery, fee, warranty, compatibility, quality, or support issues; changed opinion after actual use.
- Do not make every comment a story.
- Do not write long polished personal essays.
- Do not repeat the same "I had X and switched to Y" structure.
- Do not invent extreme events.
- Do not add domain-specific jargon that does not fit the active domain.

B. Diversity / semantic non-redundancy
- Each thread should include several different comment functions when enough comments are generated:
  direct answer, clarifying question, correction, personal datapoint, comparison, cost/value calculation, warning, edge case, disagreement, short reaction, technical explanation, and alternative recommendation.
- If a draft repeats both the same claim and the same evidence mode as a visible comment, rewrite it by changing function.
- Do not solve diversity by paraphrasing the same answer.
- Do not create diversity by going off-topic.

C. Structure / participation
- When enough visible comments exist, a substantial share of comments should reply to another comment instead of the root.
- Reply when a parent comment is factually wrong, too broad, missing a condition, asking a question, mentioning an option the persona knows, making an overconfident recommendation, or ignoring budget, use case, compatibility, risk, or constraints.
- Replies must react to parent-specific details.
- Do not create fake depth by writing generic replies that could have been top-level comments.

D. Length variation
- Tie length to function:
  one_liner = verdict, correction, skeptical reaction, or follow-up question;
  short = simple advice, compact datapoint, or narrow comparison;
  medium = recommendation with reason, tradeoff explanation, or warning;
  long = rare technical explanation, cost/value breakdown, or detailed experience.
- Do not make every comment a polished 2-3 sentence mini-review.
- Do not optimize length by random bucket sampling alone.

E. Civility / conflict
- Include sparse topical disagreement when the active phase needs conflict/civility movement.
- A small minority of comments may be blunt, skeptical, impatient, or sarcastic.
- Good conflict targets: bad advice, wrong comparison, missing condition, overconfident recommendation, unrealistic value judgment, incorrect technical claim, or ignoring the user's stated use case.
- Do not use slurs, threats, identity attacks, sexual insults, doxxing, harassment, or violent language.
- Conflict should target claims and recommendations, not people.

Required anti-template gate:
- Every materialized prompt.comment_style_guidance for diversity, structure, conflict, or final integrated phases must include an anti-template gate.
- The anti-template gate should say:
  Before writing a comment, inspect the recent visible comments. If the draft repeats both the same main claim and the same evidence mode, rewrite it by changing comment function.
- Allowed rewrites include:
  1. turn it into a clarifying question,
  2. make a narrow correction,
  3. give cost/value reasoning,
  4. add a short datapoint,
  5. provide an edge case,
  6. write a blunt one-line verdict,
  7. challenge the parent assumption.
- Do not solve repetition by paraphrasing the same advice with new adjectives.

### Diversity must change function, not only wording

For semantic diversity metrics, the materialized prompt must not merely ask for varied wording, varied openings, or different sentence rhythm.

It must force comment-function diversity.

A thread should include a mixture of:
- direct answer,
- clarifying question,
- correction,
- personal datapoint,
- comparison,
- cost/value reasoning,
- warning,
- edge case,
- disagreement,
- short reaction,
- technical explanation,
- alternative recommendation.

If two visible comments already make the same recommendation, the next comment must not make the same recommendation with different wording.

Instead, change function:
- ask what the user's use case is,
- correct a condition,
- give a counterexample,
- add cost/value reasoning,
- mention an edge case,
- challenge the parent assumption,
- give a short datapoint with a different consequence,
- or write a short skeptical verdict.
Do not remove first-hand datapoints to reduce semantic similarity. Transform repeated datapoints into different evidence modes.

### Phase carry-over contract

The materialized prompt must preserve successful behaviors from earlier completed phases.

When the active phase targets a new metric family, do not erase the behavioral mechanism that improved protected metrics.

For each materialized candidate:
- Keep protected-metric behaviors as explicit instructions.
- Add the active-metric behavior as a bounded refinement.
- Do not replace previous successful behavior with a new unrelated behavior.
- If the active phase is diversity, improve diversity by changing comment function, evidence mode, stance, reply target, sentence shape, and opening pattern. Do not improve diversity by deleting first-hand datapoints.
- If the active phase is structure/length, improve reply patterns and length variation while preserving narrative density and semantic diversity.
- If the active phase is civility/conflict, add sparse topical disagreement while preserving story density, diversity, and structure.
- If two instructions conflict, write the active instruction as:
  "preserve X, but vary Y"
  "keep X, except when Z"
  "do not increase/decrease X globally"
  "change the form of X rather than removing X"

Bad revision:
- "Reduce anecdotes to avoid repetition."

Good revision:
- "Keep short first-hand datapoints, but vary their event type, consequence, stance, and reply target. If a datapoint repeats a visible comment, rewrite it as a different comment function rather than deleting it."

Bad revision:
- "Make comments more diverse."

Good revision:
- "If a draft repeats both the same main claim and the same evidence mode as a visible comment, change it into a clarifying question, correction, edge case, cost/value reasoning, short datapoint, or blunt one-line verdict."

Metric targeting:
- The text must directly address the specific metric failures identified by the strategist.
- If self_bleu or semantic cosine is too high, explain HOW the people and comments become less repetitive.
- If mean_story_probability is too low, explain HOW personal datapoints should appear without making every comment a diary entry.
- If length variation is too low, explain HOW to mix short, medium, and longer comments.
- If avg_depth or structural_virality is too low, explain HOW replies should target visible comments and form back-and-forth.
- If tone/civility is too sanitized relative to real, explain HOW to add sparse topical bluntness, skepticism, sarcasm, impatience, clipped corrections, or claim-focused aggression in the same style that appears in real threads.
- Do not instruct the simulator to maximize toxicity, severe toxicity, threats, or harassment. If severe_toxicity_mean or threat_mean is near zero in the real validation distribution, preserve near-zero behavior.
- If tone is too high relative to real, explain HOW to reduce excess hostility while keeping the thread realistic.

Make the text operational. It should tell the simulator HOW to improve, not just what high-level vibe to have.
""".strip()

REALISM_RULES = """
## Critical Realism Rules — Closing the Gap Between Real and Generated

These rules encode the specific patterns that make generated threads obviously
distinguishable from real Reddit discussions.  Both the reasoner (strategy
design) and the materializer (text writing) must treat them as hard constraints.

### I. Realism Gap — What Generated Threads Get Wrong

The following failure modes are the biggest divergence drivers.  Every candidate
overlay must explicitly counteract at least the top-3 relevant gaps.

1. ECHO CHAMBERS: Generated threads converge on a single consensus opinion
   restated 10+ times with slight rewording.  Real threads contain genuine
   disagreement — one user loves a product while another thinks it's garbage.
   FIX: If 3+ visible comments already express the same opinion, the next
   comment MUST switch to a contrarian view, a specific correction, a tangent,
   a joke, or silence.

2. REPEATED WARNINGS: The #1 AI pattern — multiple users all voicing the same
   cautionary point with slightly different phrasing ("watch out for overspending",
   "hidden fees are a trap", "be careful with rewards chasing").  Once a point
   is made, it's made.
   FIX: After the first warning on a topic, subsequent comments must either
   add a concrete NEW datapoint, challenge the warning, or move to a different
   sub-topic.

3. VAGUE CLAIMS INSTEAD OF NUMBERS: Real users cite exact amounts ($695 AF,
   40k MR retention offer, 2 years, 0.5 cpp).  Generated users say "hidden fees"
   and "unexpected charges" without specifics.
   FIX: Any comment referencing money, time, or points must include at least one
   concrete number.

4. EXPERTISE INFLATION: Every generated persona sounds like a domain expert with
   encyclopedic knowledge.  Real threads have mostly casual users with partial
   knowledge who only know the products they personally own.
   FIX: 60% casual/beginner (know 1-2 products from personal use), 25% moderate
   (follow the community, know common abbreviations), 15% expert.

5. SHOW DON'T ADVISE: Generated comments default to second-person imperative
   ("You should call retention").  Real comments use first-person past tense
   ("I called retention and got 40k MR").
   FIX: Prefer first-person experience over second-person advice.

6. HELPER SYNDROME: Generated comments end with offers to help ("drop your
   numbers and I'll run the math", "let me know if you want more details",
   "happy to help!").  Real users share their take and move on.
   FIX: Never end with an offer to provide more help.

7. MOTIVATIONAL MONOTONY: Every generated persona exists to help OP.  Real
   users comment to brag, vent, correct someone, validate their own decision,
   pile on, or just react.
   FIX: Assign explicit motivation to each persona — only ~25% should be
   genuinely helpful/advisory.

8. MISSING ABBREVIATIONS AND JARGON: Real community members use shorthand
   naturally (AF, SUB, DP, cpp, P2, YMMV, 5/24, UR, MR, PC, CL).  Generated
   comments spell everything out.
   FIX: Use community abbreviations without explanation.

9. EMOTIONAL FLATNESS: Generated comments are informative but never genuinely
   excited, annoyed, or amused.  Real comments have emotional texture —
   excitement ("Hell yeah!"), frustration ("ugh"), humor ("lol"), dismissiveness
   ("who cares"), resignation ("it is what it is").
   FIX: Match the emotional register of the persona's motivation.

10. NO OFF-TOPIC TANGENTS: Generated threads stay perfectly on-topic.  Real
    threads occasionally digress — someone mentions a related product, complains
    about a different issue, or makes a joke.
    FIX: 10-15% of comments should be slightly off-topic.

### II. BAD Examples — What NOT to Generate

These are examples of the kind of output that makes generated threads obviously
fake.  The reasoner must penalize candidates that produce these patterns, and
the materializer must explicitly forbid them.

BAD storyteller output (NEVER write like this):
  - "I had a similar experience with my card. The rewards were nice but there
     were some hidden fees that ate into my earnings. I'd recommend being
     careful."
     ← too vague, no specific numbers, sounds like ChatGPT
  - "I totally agree about being cautious with rewards cards! I've seen many
     friends overspend just to chase cash back — it's a slippery slope into
     debt!"
     ← generic warning, no personal detail, duplicates every other comment
  - "In my experience, the annual fee can be worth it if you maximize your
     spending categories and take advantage of the travel credits."
     ← balanced consultant-speak, no concrete numbers, no emotional voice

BAD reactor output:
  - "That's a great point! I think it really depends on your individual
     spending habits and financial goals."
     ← empty validation, no specific reaction to parent content
  - "I understand your perspective. There are definitely pros and cons to
     consider."
     ← assistant-like hedging, not a real reaction

BAD advisor output:
  - "Here's what I'd recommend: Step 1: Calculate your annual spending in
     each category. Step 2: Compare the rewards rates. Step 3: Factor in the
     annual fee. Step 4: Consider sign-up bonuses."
     ← numbered-list guide structure, not how real users give advice

BAD overall patterns:
  - Every comment starting with "Honestly," or "Just my two cents"
  - Every comment being 3-5 well-structured sentences
  - Every comment offering balanced pros-and-cons
  - Every comment ending with a question back to OP
  - Zero disagreement across the entire thread
  - No comments under 10 words

### III. Expanded Forbidden Patterns

These surface-level patterns are strong AI tells.  The reasoner should check
that candidate overlays explicitly forbid them, and the materializer should
include them as hard bans.

Structural tells:
  - Numbered lists or bullet points inside comments
  - "Step 1/2/3" or "Short answer: / Long answer:" framing
  - "TL;DR" summaries
  - Decision matrices or multi-scenario comparison tables
  - Phone call scripts or "copy/paste this verbatim" templates
  - "Quick framework" or "Decision rule" structures

Tone tells:
  - "Honestly,", "I totally", "Just my two cents", "Great question!"
  - "If you want, tell me X and I'll Y" or "drop your numbers and I'll run
     the math" — helper offers
  - "I understand your perspective", "It may be beneficial", "I respectfully
     disagree" — customer-service language
  - Balanced pros-and-cons summaries
  - Ending with "happy to help!" or "let me know if you have more questions"

Length tells:
  - Multiple paragraphs for a simple opinion
  - Every comment being the same length (3-5 sentences)
  - No comments under 10 words in the entire thread
  - No one-word or one-line reactions

### IV. Reasoner Application Rules

When designing candidate overlays:
1. Check each candidate against the Realism Gap items above.  If a candidate
   does not address at least the 3 most relevant gaps, revise it.
2. Check that the candidate's persona.generation_guidance avoids expertise
   inflation, motivational monotony, and opinion homogeneity.
3. Check that the candidate's prompt.comment_style_guidance forbids the
   expanded forbidden patterns and BAD examples above.
4. If a previous iteration's winner still exhibits echo-chamber or repeated-
   warning behavior, the next candidate must add stronger anti-repetition
   rules, not just reword the existing ones.
5. Candidates that increase realism (lower Cliff's delta, lower Wasserstein)
   while maintaining protected metrics should be strongly preferred.
""".strip()


# English prose ≈ 4 chars/token; JSON/code is denser.  3.5 is conservative.
_CHARS_PER_TOKEN_ESTIMATE = 3.5

# Default token budget for the reasoner prompt.  Models like gpt-4o-mini have
# 128k context; we leave headroom for the system message + response tokens.
_DEFAULT_MAX_PROMPT_TOKENS = 120_000
