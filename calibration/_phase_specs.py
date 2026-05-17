"""Manual-phase constants and metric guidance specs.

Extracted from orchestrator.py to keep the file under 1500 lines. All values
are pure data; no logic changes.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Metric display helpers
# ---------------------------------------------------------------------------

# A compact set of representative metrics shown in progress output.
_HEADLINE_METRICS = [
    ("self_bleu_4",            "diversity (self-BLEU-4)"),
    ("self_bertscore_mean_f1", "diversity (BERTScore-F1-mean)"),
    ("semantic_mean_cosine",   "semantic uniformity"),
    ("mean_story_probability", "story likelihood"),
    ("toxicity_mean",          "toxicity mean"),
    ("aggression_score_mean",  "aggression mean"),
    ("length_cv",              "length CV"),
    ("avg_depth",              "avg reply depth"),
    ("structural_virality",    "structural virality"),
]

_TARGET_METRICS_12 = [
    "self_bleu_4",
    "self_bertscore_mean_f1",
    "semantic_mean_cosine",
    "mean_story_probability",
    "toxicity_mean",
    "severe_toxicity_mean",
    "obscene_mean",
    "threat_mean",
    "aggression_score_mean",
    "length_cv",
    "avg_depth",
    "structural_virality",
]

_STAGNATION_TRIGGER = 3
_MANUAL_PHASE_MODE = True

_MANUAL_METRIC_GUIDANCE: dict[str, str] = {
    "self_bleu_4": (
        "High means too many comments reuse similar surface phrasing or clause shapes. "
        "To lower it, vary openings, syntax, rhetorical posture, and sentence rhythm. "
        "Do not lower it by making comments random or off-topic."
    ),
    "self_bertscore_mean_f1": (
        "High means comments are too close in wording and paraphrase structure. "
        "Lower it by varying how people explain, compare, question, dismiss, or narrate, "
        "not by adding meaningless noise."
    ),
    "semantic_mean_cosine": (
        "High means too many comments are making the same substantive point in slightly different words. "
        "Lower it by diversifying stance, evidence mode, grievance, comparison target, and persona-specific framing while staying on-topic."
    ),
    "mean_story_probability": (
        "Low means comments lack believable lived experience. "
        "Raise it with short, concrete, situational first-person datapoints tied to ownership, setup, support, return, upgrade, denial, or failure moments. "
        "Do not turn every comment into a long diary entry."
    ),
    "toxicity_mean": (
        "This tracks general hostility and abrasive language. "
        "If too low, the thread is over-sanitized; if too high, everyone sounds unrealistically hostile. "
        "Match real bluntness rather than forcing universal aggression."
    ),
    "severe_toxicity_mean": (
        "Severe toxicity is usually sparse and concentrated in a minority of comments, not evenly spread. "
        "If it must rise, do it through a smaller pocket of sharper comments rather than making the whole thread uniformly extreme."
    ),
    "obscene_mean": (
        "This tracks profanity and obscenity. "
        "Real threads usually use swearing in uneven pockets, not in every comment. "
        "Raise or lower it through realistic profanity density, not blanket censorship or blanket swearing."
    ),
    "threat_mean": (
        "Threat-like language is usually rare. "
        "If real threads are higher, increase it only through a small subset of aggressive replies with menace or intimidation flavor, not by making everybody explicitly threatening."
    ),
    "aggression_score_mean": (
        "Aggression comes from impatience, contempt, sharp correction, mockery, and combative reply posture. "
        "Move it through stance and interaction style, not only profanity."
    ),
    "length_cv": (
        "This is relative length variation inside the thread. "
        "Higher means a healthier mix of short, medium, and long comments. "
        "Real discussions are often broader than bland simulations, so do not collapse everything toward medium length. "
        "Matching real spread, or being slightly more varied than a too-narrow simulation, is usually safer than under-dispersion."
    ),
    "avg_depth": (
        "Higher means more nested replies and back-and-forth. "
        "Increase it by giving personas specific triggers to answer visible comments, defend themselves, correct others, or push a disagreement one step further. "
        "Do not force every comment into a reply."
    ),
    "structural_virality": (
        "Higher means the conversation branches and continues rather than staying flat. "
        "Increase it with disagreement, quote-reactive replies, follow-up questions, corrections, and cross-thread rebuttals. "
        "Real structure can be broader than sanitized simulations, so slight over-branching is usually less harmful than a flat, dead thread."
    ),
}

_MANUAL_PHASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "narrative_evidence_grounding",
        "label": "Narrative / Evidence Grounding",
        "iteration_start": 0,
        "iteration_end": 2,
        "focus_metrics": ("mean_story_probability",),
        "required_mechanism_family": "story_anecdote",
        "dominant_emphasis": "persona-heavy, but both knobs are always edited",
        "summary": (
            "Increase realistic short first-hand or observed datapoints without turning every "
            "comment into a polished story or repeating the same anecdote."
        ),
        "reasoner_rules": [
            "Do not use MBTI, salary-band, or over-built biographies as the main mechanism.",
            "Use domain-general experience fields instead: ownership_or_usage_history, purchase_context, failure_or_success_event, comparison_target, decision_constraint, usage_pattern, and confidence_level.",
            "Each persona should have a reason for caring about this domain, but not every persona should have a story.",
            "Only 35-50% of personas should be anecdote-capable; the rest should use questions, corrections, rule-based advice, short reactions, or comparisons.",
            "Personal datapoints must be short, concrete, and functional, usually one sentence.",
            "Avoid repeated skeletons such as: 'I had X, it was not worth it, I switched to Y.'",
            "Protect diversity: if multiple personas use anecdotes, they must differ in event type, product relation, consequence, and stance.",
            "persona.generation_guidance MUST define each persona's relationship to storytelling by specifying: "
            "(a) anecdote_capability: one of 'storyteller' (35-50% of personas — has concrete first-hand experiences to share), "
            "'reactor' (asks questions, gives reactions, corrects others, but no personal datapoints), "
            "'advisor' (gives rule-based advice from general knowledge, not personal experience), "
            "'lurker-commenter' (short reactions only — 'same', 'this', '+1', or one-sentence take); "
            "(b) for storytellers only — story_type: what kind of datapoint they naturally produce — "
            "'ownership_experience' (I had X for Y months), 'purchase_decision' (I chose X over Y because), "
            "'failure_event' (X broke/failed when), 'comparison' (X vs Y in my experience), 'regret' (I wish I had chosen), 'cost_math' (the real cost is); "
            "(c) for storytellers only — story_length: 'one-liner' (single sentence datapoint) vs 'short-paragraph' (2-3 sentences with context). "
            "Most storytellers should be 'one-liner' — only a few should produce short paragraphs.",
            "prompt.comment_style_guidance for narrative MUST instruct the comment generator: "
            "'Check your persona's anecdote_capability. If you are a storyteller, produce a concrete first-hand datapoint matching your story_type — "
            "keep it to the length specified in story_length. One-liner storytellers write ONE sentence of experience, not a full narrative. "
            "If you are a reactor, respond to what others said — agree, disagree, ask why, or express surprise, but do not invent personal experiences. "
            "If you are an advisor, give practical guidance from general knowledge without claiming personal use. "
            "If you are a lurker-commenter, write 1-2 words to one sentence max. "
            "Do NOT let reactors or advisors suddenly produce first-hand anecdotes — stay in your role.'",
        ],
        "example_moves": [
            "Persona guidance example for narrative roles: "
            "'User A (storyteller, ownership_experience, one-liner): anecdote_capability=storyteller, story_type=ownership_experience, story_length=one-liner. "
            "Example output: had the 13 pro for two years, battery still gets me through a full day. "
            "User B (storyteller, failure_event, short-paragraph): anecdote_capability=storyteller, story_type=failure_event, story_length=short-paragraph. "
            "Example output: My old card had a great rewards rate but the app crashed every time I tried to redeem points. Spent 45 minutes on hold with support and they basically told me to reinstall. Switched to a different card after that. "
            "User C (reactor): anecdote_capability=reactor. "
            "Example output: wait really? I thought they discontinued that model. "
            "User D (advisor): anecdote_capability=advisor. "
            "Example output: If your main concern is battery life, look at the specs for mAh rather than trusting the marketed hours. "
            "User E (lurker-commenter): anecdote_capability=lurker-commenter. "
            "Example output: same.'",
            "Prompt example: 'Use short datapoints: I used it for travel last winter and the battery died by noon.'",
            "Prompt example: 'If a comment repeats an earlier switched-from-X-to-Y story, rewrite it into a question, correction, comparison, cost math, or edge case.'",
        ],
    },
    {
        "name": "diversity_anti_template",
        "label": "Diversity / Anti-Template Control",
        "iteration_start": 3,
        "iteration_end": 5,
        "focus_metrics": (
            "self_bleu_4",
            "self_bertscore_mean_f1",
            "semantic_mean_cosine",
        ),
        "required_mechanism_family": "semantic_diversity",
        "dominant_emphasis": "prompt-heavy, but both knobs are always edited",
        "summary": (
            "Reduce lexical and semantic redundancy by forcing different comment functions, "
            "different evidence modes, and visible-comment-aware rewrites."
        ),
        "reasoner_rules": [
            "The main failure is not only similar wording; it is repeated comment function and repeated evidence mode.",
            "Require each thread to use at least 5 comment functions: direct answer, correction, clarifying question, datapoint, comparison, cost/value calculation, warning, edge case, disagreement, short reaction.",
            "Forbid repeated assistant-like openings: Honestly, Yeah, I totally get it, I feel you, Just my two cents.",
            "If a draft repeats the same main claim and evidence mode as a visible comment, it must change function: question, correction, edge case, cost/value calculation, or short reaction.",
            "Do not solve diversity by going off-topic or adding random slang.",
            "Hard protected-narrative rule: do not reduce the amount of short first-hand or observed datapoints that made mean_story_probability improve in the previous phase.",
            "Diversity must be improved by changing comment function, evidence mode, stance, reply target, and sentence shape, not by deleting anecdotes or making comments more generic.",
            "Preserve a similar narrative density from the narrative phase, but diversify the story events: different usage contexts, different failure/success events, different comparison targets, different consequences.",
            "If a draft anecdote repeats an earlier anecdote, rewrite it into a different kind of datapoint, correction, question, edge case, or cost/value reasoning. Do not remove the datapoint entirely unless story_probability is too high.",
            "Every diversity candidate must explicitly state how it preserves narrative metrics while lowering self-BLEU, BERTScore similarity, and semantic cosine.",
            "persona.generation_guidance MUST define individual speaking habits for each persona by specifying ALL of the following dimensions: "
            "(a) sentence structure: fragments vs. complete sentences vs. run-ons; "
            "(b) punctuation style: overuses ellipses, no periods, excessive exclamation marks, ALL CAPS for emphasis, or textbook-correct; "
            "(c) opening pattern: jumps in mid-thought, always starts with a question, opens with a tangent, leads with a disclaimer, or uses a greeting; "
            "(d) closing pattern: trails off, ends with a question, no conclusion, adds a PS, or wraps with a one-liner verdict; "
            "(e) verbal tics and filler: specific phrases they repeat (like 'tbh', 'ngl', 'idk man', 'lol', 'fwiw', 'imo'), or none; "
            "(f) formality level: text-speak abbreviations, casual lowercase, standard prose, or overly formal.",
            "Personas should have distinct writing voices distributed across these categories: "
            "~20% terse/fragment writers (1-2 sentences max, no greetings, often just a verdict or reaction), "
            "~30% casual mid-length (2-4 sentences, lowercase, some abbreviations, conversational), "
            "~30% standard commenters (3-6 sentences, proper grammar, organized thoughts), "
            "~10% ramblers (long paragraphs, no structure, topic drifts, asides), "
            "~10% distinctive quirk writers (heavy slang, ALL CAPS, excessive punctuation, or stream-of-consciousness).",
            "Do not give every persona clean, well-structured, assistant-like prose. Real Reddit users write unevenly: "
            "typos, incomplete thoughts, mid-sentence topic shifts, casual profanity, ALL CAPS for emphasis, or no capitalization at all. "
            "The persona field must encode these habits so the comment generator can follow them.",
            "Each persona's speaking habit must be concrete and actionable, not vague. "
            "BAD: 'writes casually' or 'uses informal tone'. "
            "GOOD: 'always opens with a lowercase tangent before getting to the point, uses ... between clauses, never capitalizes, ends abruptly without finishing the thought'. "
            "GOOD: 'writes in complete grammatical sentences but peppers in tbh and ngl, always ends with a rhetorical question'. "
            "GOOD: 'types like texting: no caps, abbreviations everywhere (u, ur, bc, rn), rarely more than two sentences'.",
            "prompt.comment_style_guidance for diversity MUST instruct the comment generator to follow each persona's defined speaking habits: "
            "'Before writing, check your persona's writing style fields. Match their sentence structure, punctuation habits, opening pattern, and verbal tics exactly. "
            "A terse persona should write 1-2 sentences max. A rambler should write a long paragraph with asides. "
            "A text-speak persona should use abbreviations and no capitalization. Do NOT default to clean, structured, assistant-like writing for any persona.'",
        ],
        "example_moves": [
            "Prompt example: 'If two visible comments already say X is not worth it, the next comment must not repeat that. It should ask about usage pattern, calculate value, suggest a downgrade/alternative, correct a detail, or give an edge case.'",
            "Prompt example: 'Bad: Honestly, I had it for a year and switched. Good: Depends what you need it for. If portability matters, the cheaper option may be better.'",
            "Persona guidance example for writing habits: "
            "'User A (terse fragment writer): opens mid-thought, no greeting, 1-2 sentences max, no periods, lowercase everything. "
            "Example output: lol no the battery on that thing is terrible just get the cheaper one. "
            "User B (casual rambler): starts with a tangent, writes one long paragraph, uses ... between thoughts, adds personal asides. "
            "Example output: ok so i had this exact same dilemma last year... ended up going with the premium one and honestly... its fine i guess but like my buddy got the budget version and his works just as well so idk maybe save ur money?? also the customer service is trash but thats a whole other thing. "
            "User C (standard commenter): proper grammar, 3-4 sentences, organized point. "
            "Example output: I owned the previous model for two years and it held up well. The main upgrade in the new version is the camera, which might not matter if you primarily use it for calls and texting. For the price difference, I would stick with the older model. "
            "User D (aggressive one-liner): no greeting, terse verdict, sometimes dismissive. "
            "Example output: thats not even close to worth it at that price. "
            "User E (quirky punctuation): excessive exclamation marks, ALL CAPS for emphasis, emotional. "
            "Example output: Wait WHAT?! You paid THAT much for it?! I got mine on sale for like half that and it works PERFECTLY fine!! Seriously check the deals section!!!'",
        ],
    },
    {
        "name": "structure_length_interaction",
        "label": "Structure / Length Interaction",
        "iteration_start": 6,
        "iteration_end": 8,
        "focus_metrics": (
            "length_cv",
            "avg_depth",
            "avg_branching_factor",
            "structural_virality",
        ),
        "required_mechanism_family": "structure",
        "dominant_emphasis": "prompt-heavy, but both knobs are always edited",
        "summary": (
            "Create realistic reply chains and length variation through corrections, questions, "
            "short rebuttals, medium advice, and occasional longer explanations."
        ),
        "reasoner_rules": [
            "Do not optimize length by random bucket sampling alone. Tie length to reply function.",
            "One-liners should usually be verdicts, corrections, skeptical replies, or follow-up questions.",
            "Medium comments should usually give advice, comparison, or one compact datapoint.",
            "Long comments should be rare and used for tradeoff analysis, cost/value explanation, technical clarification, or detailed personal datapoints.",
            "Increase replies by giving personas parent-specific triggers: incorrect claim, missing condition, overconfident advice, product-specific mention, or question.",
            "40-60% of comments should reply to visible comments when enough visible comments exist.",
            "Do not create artificial deep chains where every reply is a paragraph.",
            "Protect diversity by requiring each reply to react to a parent-specific detail, not just repeat the root answer.",
            "persona.generation_guidance MUST define each persona's natural comment length and reply behavior by specifying: "
            "(a) typical_length: one of 'terse' (1-2 sentences), 'medium' (3-5 sentences), 'long' (6+ sentences, rare, ~10% of personas); "
            "(b) reply_trigger: what makes this persona reply to someone else's comment instead of writing a top-level response — "
            "e.g., 'replies when someone gives wrong technical info', 'replies to correct price comparisons', 'replies only to ask follow-up questions', 'rarely replies, mostly posts top-level', 'replies to disagree with popular takes'; "
            "(c) reply_style: how they engage with the parent comment — 'quotes and rebuts specific claims', 'ignores parent and gives own take', 'asks a pointed follow-up question', 'adds a caveat or edge case to parent's advice', 'short agreement or disagreement then own experience'.",
            "Distribute length tendencies tied to persona type: "
            "lurkers and low-karma users (karma < 2000) should be 'terse' by default; "
            "regular users (karma 2000-10000) should mix 'terse' and 'medium'; "
            "power users and enthusiasts (karma > 10000) should be 'medium' or occasionally 'long' but not always — even power users sometimes write short reactions. "
            "Do NOT assign lengths randomly; tie them to the persona's motivation and knowledge_style.",
            "prompt.comment_style_guidance for structure MUST instruct the comment generator on reply behavior: "
            "'Check your persona's reply_trigger field. If a visible comment matches your trigger (wrong info, price error, overconfident claim), reply to that specific comment instead of writing top-level. "
            "When replying, react to a specific detail in the parent — quote it, correct it, question it, or add a caveat. Do not write a reply that ignores the parent and just restates your own opinion. "
            "Match your persona's typical_length: if you are terse, write 1-2 sentences max. If you are medium, write 3-5 sentences. Only write 6+ sentences if you are a long-form persona AND the topic warrants detailed explanation.'",
        ],
        "example_moves": [
            "Prompt example: 'Small correction: that model does not have the feature people keep mentioning.'",
            "Prompt example: 'Only if OP actually uses that feature.'",
            "Prompt example: 'What is your budget and main use case?'",
            "Prompt example: 'I had the older version; the issue was not performance, it was battery/comfort/repair cost/fees.'",
            "Persona guidance example for length/reply: "
            "'User A (karma 800, terse, reply_trigger=wrong_price): typical_length=terse, replies when someone quotes wrong prices. "
            "Example: the 128gb is $799 not $899, check bestbuy. "
            "User B (karma 5000, medium, reply_trigger=overconfident_advice): typical_length=medium, replies to push back on blanket recommendations. "
            "Example: That depends entirely on your use case. If you mostly browse and text, the base model is fine. The Pro only makes sense if you actually use the camera features daily. "
            "User C (karma 15000, medium-long, reply_trigger=missing_context): typical_length=medium, occasionally long when explaining tradeoffs. "
            "Example: I owned both the X and the Y for about a year each. The X had better battery life by roughly 2 hours in my usage, but the Y had a significantly better display. For the price difference, it really comes down to whether you value screen quality or endurance more.'",
        ],
    },
    {
        "name": "civility_conflict_calibration",
        "label": "Civility / Conflict Calibration",
        "iteration_start": 9,
        "iteration_end": 11,
        "focus_metrics": (
            "toxicity_mean",
            "severe_toxicity_mean",
            "obscene_mean",
            "threat_mean",
            "aggression_score_mean",
        ),
        "required_mechanism_family": "tone_civility",
        "dominant_emphasis": "prompt-heavy, but both knobs are always edited",
        "summary": (
            "Match realistic online disagreement style through sparse topical bluntness, "
            "skepticism, sarcasm, and corrections without maximizing unsafe hostility."
        ),
        "reasoner_rules": [
            "Do not maximize toxicity, severe toxicity, or threats. Match the real validation distribution.",
            "If severe_toxicity_mean or threat_mean is near zero in real data, preserve near-zero behavior.",
            "Move aggression mainly through topical disagreement: bad advice, wrong comparison, missing condition, overconfident recommendation, incorrect technical claim, or unrealistic value judgment.",
            "Allow a small minority, around 8-15%, of blunt or sarcastic commenters.",
            "Conflict should target claims, recommendations, assumptions, comparisons, or evidence, not identity or protected attributes.",
            "Avoid slurs, threats, sexual insults, doxxing, and harassment.",
            "Do not make every disagreement polite.",
            "Do not make every disagreement toxic.",
            "Conflict should often appear as replies to visible comments, not standalone insult-only comments.",
            "persona.generation_guidance MUST set individual conflict personalities by specifying ALL of the following for each persona: "
            "(a) conflict_style: one of blunt, sarcastic, argumentative, skeptical, calm, avoidant, passive-aggressive, gatekeeping; "
            "(b) conflict_trigger: what specific topic or behavior makes this persona push back — "
            "e.g., 'bad financial advice', 'recommending expensive products to budget shoppers', 'repeating marketing talking points as fact', "
            "'overconfident claims from obvious beginners', 'popular opinions they think are wrong', 'incorrect technical specs'; "
            "(c) conflict_intensity: low (hedges, softens, uses maybe/I think), medium (direct disagreement but respectful), "
            "high (blunt, dismissive, may use mild profanity or sarcasm); "
            "(d) conflict_target: what they attack — 'the claim itself', 'the reasoning behind it', 'the person's credibility', 'the comparison being made', 'missing context'.",
            "Distribute conflict personalities realistically across the cast: "
            "~40% calm/avoidant (rarely disagree, hedge when they do, use phrases like 'I could be wrong but' or 'just my experience'), "
            "~25% skeptical (question claims and ask for evidence, but not hostile — 'where did you see that?' or 'that hasn't been my experience'), "
            "~15% blunt/direct (disagree openly without softening — 'no, that's wrong' or 'bad advice'), "
            "~10% sarcastic/passive-aggressive (wrap disagreement in irony — 'oh sure, because the marketing team would never exaggerate'), "
            "~10% argumentative/gatekeeping (actively seek arguments, dismiss others' knowledge — 'you clearly haven't used it long enough to know').",
            "Some personas should have a low patience threshold and jump to dismissive replies quickly; others should be the type who writes a measured rebuttal. "
            "This variation MUST come from persona traits (conflict_style + conflict_intensity + conflict_trigger), not just prompt instructions. "
            "The comment generator should check the persona's conflict fields and only generate confrontational content when the persona's trigger is matched by a visible comment.",
            "Do not make every persona nice by default. Real subreddits have a mix: "
            "people who are genuinely helpful, people who correct others as a hobby, people who gatekeep their expertise, "
            "people who vent frustration about products/services, and people who just enjoy being contrarian. "
            "Each type should produce visibly different comment tones.",
            "prompt.comment_style_guidance for civility MUST instruct the comment generator: "
            "'Before writing, check your persona's conflict_style, conflict_trigger, conflict_intensity, and conflict_target fields. "
            "If a visible comment matches your conflict_trigger, respond according to your conflict_style and intensity: "
            "blunt personas should write short, direct pushback without softening; "
            "sarcastic personas should use irony, rhetorical questions, or mock agreement; "
            "skeptical personas should question the claim and ask for specifics; "
            "calm/avoidant personas should either skip the disagreement or hedge heavily with I think / maybe / not sure but. "
            "Do NOT make blunt personas polite or calm personas aggressive — stay in character. "
            "If no visible comment matches your trigger, write a normal comment without forced conflict.'",
        ],
        "example_moves": [
            "Prompt example (blunt, high intensity): 'No, that is bad advice. OP said they barely use that feature.'",
            "Prompt example (skeptical, medium intensity): 'Where are you getting that number? The MSRP is different from what retailers actually charge.'",
            "Prompt example (sarcastic, medium intensity): 'Oh sure, lets all recommend the $500 option to someone who said their budget is $200. Great advice.'",
            "Prompt example (calm, low intensity): 'I could be wrong, but I think the newer model actually fixed that issue. Might be worth checking.'",
            "Prompt example (gatekeeping, high intensity): 'You clearly havent used it for more than a week if you think thats a feature worth paying for.'",
            "Prompt example (avoidant, low intensity): 'idk I had a different experience but everyones situation is different I guess'",
            "Persona guidance example: "
            "'User A: conflict_style=blunt, conflict_trigger=bad_financial_advice, conflict_intensity=high, conflict_target=the_claim. "
            "Behavior: snaps at bad advice with 1-2 sentence dismissals. Example: nope. thats wrong, the annual fee alone eats any cashback you get. "
            "User B: conflict_style=sarcastic, conflict_trigger=marketing_parroting, conflict_intensity=medium, conflict_target=the_reasoning. "
            "Behavior: uses irony to expose flawed logic. Example: ah yes because the company that charges you $95/year definitely has your best interest at heart lol. "
            "User C: conflict_style=skeptical, conflict_trigger=overconfident_beginners, conflict_intensity=medium, conflict_target=credibility. "
            "Behavior: questions experience level. Example: how long have you actually had the card? because that perk changed like 6 months ago. "
            "User D: conflict_style=avoidant, conflict_trigger=none, conflict_intensity=low, conflict_target=none. "
            "Behavior: never directly disagrees. Example: yeah idk... i had a different experience but maybe it depends on the region or something.'",
        ],
    },
)


_DEDUP_ITERATION: int = int(_MANUAL_PHASE_SPECS[-1]["iteration_end"]) + 1
"""The 0-indexed manual-iteration number of the dedup-only round."""

_DEDUP_PHASE_SPEC: dict[str, Any] = {
    "name": "dedup_final",
    "label": "Final Deduplication",
    "iteration_start": _DEDUP_ITERATION,
    "iteration_end": _DEDUP_ITERATION,
    "focus_metrics": (),
    "required_mechanism_family": None,
    "dominant_emphasis": "text-only deduplication",
    "summary": (
        "Remove redundant, duplicated, or near-duplicated content from the "
        "accumulated structured overlay.  Do NOT add, change, or remove any "
        "rule or behavioral instruction — only compress."
    ),
    "reasoner_rules": [
        "Every unique rule must survive in the output.",
        "Do not merge rules from different phase blocks.",
        "Do not change number thresholds, percentages, or distributions.",
        "Replace duplicate rules with a short back-reference or remove them.",
        "Aim for 30-50% character reduction while preserving all unique behavioral content.",
    ],
    "example_moves": [],
    "is_dedup": True,
}

