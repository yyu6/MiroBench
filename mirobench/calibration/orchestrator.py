"""
Orchestrator for the calibration system.

Components
----------
CalibrationState      : Persistent state for resume support.
run_calibration_loop  : Main calibration loop.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

from .log import CalibrationLog
from .overlay import (
    STRUCTURED_PHASE_BLOCKS_KEY,
    diff_overlay,
    merge_overlay,
    render_structured_overlay,
    save_overlay,
)
from .reasoner import (
    build_dedup_prompt,
    build_reasoner_prompt,
    build_text_materializer_prompt,
    call_reasoner,
    generate_variants,
    materializer_response_format,
    parse_reasoner_response,
    parse_text_materializer_response,
)
from .registry import KnobRegistry
from .runner import run_candidates
from .scorer import (
    DEFAULT_METRICS,
    PRIMARY_CALIBRATION_METRICS,
    candidate_selection_key,
    compute_baseline_from_csv,
    load_thread_metrics,
    score_candidate,
    select_best_candidate,
)
from .stats import compare_before_after, evaluate_group_vs_real

import math as _math

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


def _manual_total_edited_iterations() -> int:
    """Return the fixed number of edited iterations in manual-phase mode.

    Includes the 4 regular phase blocks (12 iterations) plus 1 dedup round = 13.
    """
    return _DEDUP_ITERATION + 1


def _is_dedup_iteration(manual_iteration: int) -> bool:
    """Return True if this manual-iteration is the dedup-only round."""
    return manual_iteration == _DEDUP_ITERATION


def _parse_dedup_response(
    raw: str,
    num_candidates: int,
    fallback_overlay: dict[str, Any],
) -> dict[str, Any]:
    """Parse the dedup LLM response into a list of full overlay dicts.

    Returns ``{"overlays": [overlay_0, overlay_1, ...]}``.  Each overlay is a
    complete overlay dict (same shape as the input) ready to be simulated.
    If a candidate is missing or malformed, the original overlay is used as-is.
    """
    import re as _re

    # Strip markdown fencing if present
    cleaned = raw.strip()
    fence_match = _re.search(r"```(?:json)?\s*\n?(.*?)```", cleaned, _re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    overlays: list[dict[str, Any]] = []
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        print("  [dedup] WARNING: could not parse LLM response as JSON; using original overlay for all candidates")
        return {"overlays": [dict(fallback_overlay)] * num_candidates}

    if not isinstance(data, dict):
        return {"overlays": [dict(fallback_overlay)] * num_candidates}

    for i in range(num_candidates):
        key = f"candidate_{i}"
        candidate_data = data.get(key)
        if not isinstance(candidate_data, dict):
            overlays.append(dict(fallback_overlay))
            continue
        # Build a complete overlay from the candidate data
        overlay: dict[str, Any] = {}
        for field in ("persona.generation_guidance", "prompt.comment_style_guidance"):
            overlay[field] = str(candidate_data.get(field, fallback_overlay.get(field, "")))
        # Preserve _manual_phase_blocks if provided, otherwise keep original
        blocks = candidate_data.get("_manual_phase_blocks")
        if isinstance(blocks, dict) and blocks:
            overlay["_manual_phase_blocks"] = blocks
        else:
            orig_blocks = fallback_overlay.get("_manual_phase_blocks")
            if orig_blocks:
                overlay["_manual_phase_blocks"] = orig_blocks
        overlays.append(overlay)

    return {"overlays": overlays}


def _manual_phase_for_iteration(iteration: int) -> dict[str, Any]:
    """Return the deterministic phase spec for the manual schedule.

    Iterations 0-11 map to the 4 regular phase blocks.
    Iteration 12 is the dedup-only round.
    """
    if _is_dedup_iteration(iteration):
        phase = dict(_DEDUP_PHASE_SPEC)
        phase["phase_index"] = len(_MANUAL_PHASE_SPECS)
        phase["block_index"] = len(_MANUAL_PHASE_SPECS)
        return phase
    for idx, spec in enumerate(_MANUAL_PHASE_SPECS):
        if spec["iteration_start"] <= iteration <= spec["iteration_end"]:
            phase = dict(spec)
            phase["phase_index"] = idx
            phase["block_index"] = idx
            return phase
    phase = dict(_MANUAL_PHASE_SPECS[-1])
    phase["phase_index"] = len(_MANUAL_PHASE_SPECS) - 1
    phase["block_index"] = len(_MANUAL_PHASE_SPECS) - 1
    return phase


def _manual_phase_context(iteration: int) -> dict[str, Any]:
    """Return phase metadata plus protected metrics for the current iteration."""
    phase = _manual_phase_for_iteration(iteration)
    if phase.get("is_dedup"):
        # Dedup round: all metrics from all phases are protected; no focus metrics.
        all_metrics: list[str] = []
        for spec in _MANUAL_PHASE_SPECS:
            for metric in spec["focus_metrics"]:
                if metric not in all_metrics:
                    all_metrics.append(metric)
        phase["focus_metrics"] = []
        phase["protected_metrics"] = all_metrics
        phase["focus_metric_guidance"] = []
        phase["protected_metric_guidance"] = [
            {
                "metric": metric,
                "guidance": _MANUAL_METRIC_GUIDANCE.get(metric, ""),
            }
            for metric in all_metrics
            if metric in _MANUAL_METRIC_GUIDANCE
        ]
        phase["iteration_label"] = f"iter_{iteration + 1}_dedup"
        phase["block_label"] = f"iter_{iteration + 1} (dedup)"
        phase["candidate_plan"] = []
        return phase

    protected_metrics: list[str] = []
    for earlier in _MANUAL_PHASE_SPECS[: phase["phase_index"]]:
        for metric in earlier["focus_metrics"]:
            if metric not in protected_metrics:
                protected_metrics.append(metric)
    phase["focus_metrics"] = list(phase["focus_metrics"])
    phase["protected_metrics"] = protected_metrics
    phase["focus_metric_guidance"] = [
        {
            "metric": metric,
            "guidance": _MANUAL_METRIC_GUIDANCE.get(metric, ""),
        }
        for metric in phase["focus_metrics"]
    ]
    phase["protected_metric_guidance"] = [
        {
            "metric": metric,
            "guidance": _MANUAL_METRIC_GUIDANCE.get(metric, ""),
        }
        for metric in protected_metrics
        if metric in _MANUAL_METRIC_GUIDANCE
    ]
    phase["iteration_label"] = f"iter_{iteration + 1}"
    phase["block_label"] = f"iter_{phase['iteration_start'] + 1}-{phase['iteration_end'] + 1}"
    return phase


def _phase1_total_iterations(max_iterations: int) -> int:
    """Return total Phase-1 loop iterations, including baseline in manual mode."""
    return max_iterations + 1 if _MANUAL_PHASE_MODE else max_iterations


def _phase1_reported_iteration_count(completed_iterations: int) -> int:
    """Return the user-facing edited-iteration count for Phase 1 progress reporting."""
    if not _MANUAL_PHASE_MODE:
        return completed_iterations
    return max(0, completed_iterations - 1)


def _manual_phase_prompt_trajectory(
    trajectory: list[dict[str, Any]],
    phase_context: dict[str, Any],
    iteration: int,
) -> list[dict[str, Any]]:
    """Return all previous iterations inside the same manual phase block."""
    phase_name = str(phase_context.get("name", "")).strip()
    phase_start = int(phase_context.get("iteration_start", 0))
    phase_end = int(phase_context.get("iteration_end", phase_start))
    current_manual_iteration = max(0, iteration - 1)
    if not phase_name or current_manual_iteration <= phase_start:
        return []
    filtered: list[dict[str, Any]] = []
    for entry in trajectory:
        manual_ctx = ((entry.get("search_state", {}) or {}).get("manual_phase_context", {}) or {})
        if str(manual_ctx.get("name", "")).strip() != phase_name:
            continue
        entry_iteration_label = str(manual_ctx.get("iteration_label", "")).strip()
        if entry_iteration_label.startswith("iter_"):
            try:
                entry_manual_iteration = int(entry_iteration_label.split("_", 1)[1]) - 1
            except ValueError:
                continue
        else:
            continue
        if not (phase_start <= entry_manual_iteration <= phase_end):
            continue
        if entry_manual_iteration >= current_manual_iteration:
            continue
        filtered.append(entry)
    return filtered


def _manual_block_reference(state: "CalibrationState") -> dict[str, Any] | None:
    """Return the current manual phase-block incumbent payload, if any."""
    if state.manual_block_best_diagnostic and "quantile_fail_rate" in state.manual_block_best_diagnostic:
        return state.manual_block_best_diagnostic
    return state.manual_block_best_score


def _manual_start_block(
    state: "CalibrationState",
    phase_context: dict[str, Any],
) -> None:
    """Initialize a new phase block with no incumbent so the first iteration always wins.

    The base overlay (from previous phases) is kept as the search root for
    candidate generation, but the block-best score/diagnostic are cleared so
    that ``_manual_block_reference`` returns ``None``.  This guarantees the
    first iteration's winner is always saved — fulfilling the requirement that
    every phase must produce a block overlay.
    """
    state.manual_block_phase_name = str(phase_context.get("name", "")).strip() or None
    state.manual_block_best_overlay = dict(state.current_best_overlay)
    # Clear score/diagnostic so the first iteration has no incumbent to beat.
    state.manual_block_best_score = None
    state.manual_block_best_diagnostic = None
    state.manual_block_best_candidate_dir = None
    state.current_search_root_overlay = dict(state.current_best_overlay)
    state.current_search_root_diagnostic = state.current_best_diagnostic
    state.current_search_root_candidate_dir = state.current_best_candidate_dir
    state.current_search_root_mode = f"manual_phase:{phase_context.get('name')}"
    state.current_search_root_reason = "start new manual phase block from cumulative committed overlay"


def _manual_commit_block_best(
    state: "CalibrationState",
    phase_context: dict[str, Any],
) -> None:
    """Commit the current block incumbent into the cumulative overlay state."""
    phase_name = str(phase_context.get("name", "")).strip()
    if not phase_name:
        return
    if str(state.manual_block_phase_name or "").strip() != phase_name:
        return
    state.current_best_overlay = dict(state.manual_block_best_overlay)
    state.current_best_score = (
        dict(state.manual_block_best_score) if isinstance(state.manual_block_best_score, dict) else state.manual_block_best_score
    )
    state.current_best_diagnostic = (
        dict(state.manual_block_best_diagnostic) if isinstance(state.manual_block_best_diagnostic, dict) else state.manual_block_best_diagnostic
    )
    state.current_best_candidate_dir = state.manual_block_best_candidate_dir
    state.current_search_root_overlay = dict(state.current_best_overlay)
    state.current_search_root_diagnostic = state.current_best_diagnostic
    state.current_search_root_candidate_dir = state.current_best_candidate_dir
    state.current_search_root_mode = f"manual_phase:{phase_name}"
    state.current_search_root_reason = "committed block_best into cumulative overlay"
    _maybe_record_completed_phase_summary(state, phase_context)


def _subset_robust_stats(
    per_metric: dict[str, dict[str, Any]],
    metrics: list[str],
) -> dict[str, float | int]:
    """Aggregate robust scoring fields for an arbitrary metric subset."""
    items = [
        per_metric[m]
        for m in metrics
        if m in per_metric and per_metric[m].get("status") != "missing"
    ]
    if not items:
        return {
            "metric_count": 0,
            "out_of_range_count": 0,
            "mean_percentile_distance": float("inf"),
            "max_percentile_distance": float("inf"),
            "mean_abs_raw_robust_z": float("inf"),
            "max_abs_raw_robust_z": float("inf"),
        }
    percentile_distances = [float(item.get("percentile_distance", 0.0)) for item in items]
    raw_robust_zs = [float(item.get("abs_robust_z", 0.0)) for item in items]
    return {
        "metric_count": len(items),
        "out_of_range_count": sum(1 for item in items if item.get("status") != "in_range"),
        "mean_percentile_distance": float(np.mean(percentile_distances)),
        "max_percentile_distance": float(np.max(percentile_distances)),
        "mean_abs_raw_robust_z": float(np.mean(raw_robust_zs)),
        "max_abs_raw_robust_z": float(np.max(raw_robust_zs)),
    }


def _subset_group_eval_stats(
    per_metric: dict[str, dict[str, Any]],
    metrics: list[str],
) -> dict[str, float | int]:
    """Aggregate group-vs-real statistics for an arbitrary metric subset."""
    items = [per_metric[m] for m in metrics if m in per_metric]
    if not items:
        return {
            "metric_count": 0,
            "mwu_sig_count": 0,
            "ks_sig_count": 0,
            "mwu_pass_count": 0,
            "ks_pass_count": 0,
            "mean_wasserstein": float("inf"),
            "mean_quantile_error": float("inf"),
            "mean_empirical_fail_rate": float("inf"),
            "mean_abs_median_gap": float("inf"),
            "mean_abs_cliffs_delta": float("inf"),
        }
    wasserstein = [float(item.get("wasserstein_distance", float("inf"))) for item in items]
    quantile_error = [float(item.get("quantile_error", float("inf"))) for item in items]
    empirical_fail = [float(item.get("empirical_fail_rate", float("inf"))) for item in items]
    abs_median_gap = [abs(float(item.get("median_gap", item.get("abs_median_gap", float("inf"))))) for item in items]
    abs_cliffs = [abs(float(item.get("cliffs_delta", float("inf")))) for item in items]
    mwu_sig_count = sum(1 for item in items if float(item.get("mwu_p_value", 1.0)) <= 0.05)
    ks_sig_count = sum(1 for item in items if float(item.get("ks_p_value", 1.0)) <= 0.05)
    return {
        "metric_count": len(items),
        "mwu_sig_count": mwu_sig_count,
        "ks_sig_count": ks_sig_count,
        "mwu_pass_count": len(items) - mwu_sig_count,
        "ks_pass_count": len(items) - ks_sig_count,
        "mean_wasserstein": float(np.mean(wasserstein)),
        "mean_quantile_error": float(np.mean(quantile_error)),
        "mean_empirical_fail_rate": float(np.mean(empirical_fail)),
        "mean_abs_median_gap": float(np.mean(abs_median_gap)),
        "mean_abs_cliffs_delta": float(np.mean(abs_cliffs)),
    }


def _manual_phase_metric_rows(
    candidate: dict[str, Any],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Return ordered per-metric comparison rows for manual-phase selection."""
    robust_per_metric = candidate.get("per_metric", {}) or {}
    group_eval_per_metric = candidate.get("group_eval_per_metric", {}) or {}
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        robust = robust_per_metric.get(metric, {}) or {}
        group = group_eval_per_metric.get(metric, {}) or {}
        status = str(robust.get("status", "missing"))
        mwu_p = float(group.get("mwu_p_value", 1.0))
        ks_p = float(group.get("ks_p_value", 1.0))
        rows.append(
            {
                "metric": metric,
                "wasserstein": float(group.get("wasserstein_distance", float("inf"))),
                "quantile_error": float(group.get("quantile_error", float("inf"))),
                "empirical_fail_rate": float(group.get("empirical_fail_rate", float("inf"))),
                "abs_median_gap": abs(float(group.get("median_gap", group.get("abs_median_gap", float("inf"))))),
                "abs_cliffs_delta": abs(float(group.get("cliffs_delta", float("inf")))),
                "mwu_sig": int(mwu_p <= 0.05),
                "ks_sig": int(ks_p <= 0.05),
                "mwu_p_value": mwu_p,
                "ks_p_value": ks_p,
                "out_of_range": 0 if status == "in_range" else 1,
                "percentile_distance": float(robust.get("percentile_distance", float("inf"))),
                "abs_raw_robust_z": float(robust.get("abs_robust_z", float("inf"))),
                "status": status,
            }
        )
    return rows


def _target_metric_eval_summary(
    candidate: dict[str, Any],
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Return a compact all-target snapshot for one candidate or winner."""
    target_metrics = list(metrics or _TARGET_METRICS_12)
    rows = _manual_phase_metric_rows(candidate, target_metrics)
    if not rows:
        return {
            "metrics": target_metrics,
            "rows": [],
            "mwu_pass_count": 0,
            "ks_pass_count": 0,
            "mean_wasserstein": float("inf"),
            "mean_quantile_error": float("inf"),
            "mean_empirical_fail_rate": float("inf"),
            "mean_abs_median_gap": float("inf"),
            "mean_abs_cliffs_delta": float("inf"),
        }
    return {
        "metrics": target_metrics,
        "rows": rows,
        "mwu_pass_count": sum(1 for row in rows if float(row.get("mwu_p_value", 0.0)) > 0.05),
        "ks_pass_count": sum(1 for row in rows if float(row.get("ks_p_value", 0.0)) > 0.05),
        "mean_wasserstein": float(np.mean([float(row.get("wasserstein", float("inf"))) for row in rows])),
        "mean_quantile_error": float(np.mean([float(row.get("quantile_error", float("inf"))) for row in rows])),
        "mean_empirical_fail_rate": float(np.mean([float(row.get("empirical_fail_rate", float("inf"))) for row in rows])),
        "mean_abs_median_gap": float(np.mean([float(row.get("abs_median_gap", float("inf"))) for row in rows])),
        "mean_abs_cliffs_delta": float(np.mean([float(row.get("abs_cliffs_delta", float("inf"))) for row in rows])),
    }


def _manual_guard_threshold(
    baseline: float,
    *,
    multiplier: float,
    additive_floor: float,
) -> float:
    """Return a bounded regression threshold around a protected baseline value."""
    return max(baseline * multiplier, baseline + additive_floor)


def _manual_phase_guard_summary(
    candidate: dict[str, Any],
    reference_payload: dict[str, Any] | None,
    phase_context: dict[str, Any],
) -> dict[str, Any]:
    """Detect regressions on protected metrics using Cliff's delta and
    Wasserstein distance only (no p-values).

    A violation fires when a protected metric's |Cliff's delta| or Wasserstein
    distance increases by more than a relative tolerance compared to the
    reference (previous best).  This ensures earlier gains are preserved while
    the current phase optimizes its focus metrics.

    Tolerance: a protected metric is violated when its candidate value exceeds
    ``max(ref * 1.5, ref + 0.05)`` for either |Cliff's delta| or Wasserstein.
    """
    protected_metrics = list(phase_context.get("protected_metrics", []))
    if not protected_metrics or not reference_payload:
        return {
            "protected_metric_count": len(protected_metrics),
            "violation_count": 0,
            "max_severity": 0.0,
            "violations": [],
        }

    candidate_rows = {
        row["metric"]: row
        for row in _manual_phase_metric_rows(candidate, protected_metrics)
    }
    reference_rows = {
        row["metric"]: row
        for row in _manual_phase_metric_rows(reference_payload, protected_metrics)
    }

    violations: list[dict[str, Any]] = []
    max_severity = 0.0

    for metric in protected_metrics:
        cand_row = candidate_rows.get(metric)
        ref_row = reference_rows.get(metric)
        if not cand_row or not ref_row:
            continue

        ref_cd = float(ref_row.get("abs_cliffs_delta", float("inf")))
        cand_cd = float(cand_row.get("abs_cliffs_delta", float("inf")))
        ref_w = float(ref_row.get("wasserstein", float("inf")))
        cand_w = float(cand_row.get("wasserstein", float("inf")))

        triggered: list[str] = []
        # Check Cliff's delta regression
        cd_threshold = _manual_guard_threshold(ref_cd, multiplier=1.5, additive_floor=0.05)
        if cand_cd > cd_threshold and ref_cd < float("inf"):
            triggered.append("cliffs_delta_regressed")
        # Check Wasserstein regression
        w_threshold = _manual_guard_threshold(ref_w, multiplier=1.5, additive_floor=0.05)
        if cand_w > w_threshold and ref_w < float("inf"):
            triggered.append("wasserstein_regressed")

        if triggered:
            # Severity = how much worse the candidate is relative to reference
            cd_ratio = (cand_cd / max(ref_cd, 1e-9)) if ref_cd < float("inf") else 1.0
            w_ratio = (cand_w / max(ref_w, 1e-9)) if ref_w < float("inf") else 1.0
            severity = max(cd_ratio, w_ratio)
            max_severity = max(max_severity, severity)
            violations.append(
                {
                    "metric": metric,
                    "triggered_fields": triggered,
                    "severity": severity,
                    "reference": ref_row,
                    "candidate": cand_row,
                }
            )

    return {
        "protected_metric_count": len(protected_metrics),
        "violation_count": len(violations),
        "max_severity": float(max_severity),
        "violations": violations,
    }


def _manual_metric_row_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Return the comparison key for one metric.

    Lower key = better candidate.  Ranking uses only Cliff's delta and
    Wasserstein distance — both should be driven toward 0.

    1. |Cliff's delta| — lower is better (closer to 0 = distributions match).
    2. Wasserstein distance — lower is better (full-distribution shape match).
    3. |median gap| — tie-break on center-location mismatch.
    """
    return (
        float(row.get("abs_cliffs_delta", float("inf"))),
        float(row.get("wasserstein", float("inf"))),
        float(row.get("abs_median_gap", float("inf"))),
    )


def _manual_phase_score(
    candidate: dict[str, Any],
    phase_context: dict[str, Any],
) -> dict[str, Any]:
    """Build a phase-specific score using only the targeted metrics plus protected ones."""
    per_metric = candidate.get("per_metric", {}) or {}
    group_eval_per_metric = candidate.get("group_eval_per_metric", {}) or {}
    focus_metrics = list(phase_context.get("focus_metrics", []))
    protected_metrics = list(phase_context.get("protected_metrics", []))
    return {
        "phase_name": phase_context.get("name"),
        "focus_metrics": focus_metrics,
        "protected_metrics": protected_metrics,
        "focus_metric_rows": _manual_phase_metric_rows(candidate, focus_metrics),
        "protected_metric_rows": _manual_phase_metric_rows(candidate, protected_metrics),
        "focus_robust": _subset_robust_stats(per_metric, focus_metrics),
        "focus_group_eval": _subset_group_eval_stats(group_eval_per_metric, focus_metrics),
        "protected_robust": _subset_robust_stats(per_metric, protected_metrics),
        "protected_group_eval": _subset_group_eval_stats(group_eval_per_metric, protected_metrics),
    }


def _manual_phase_selection_key(candidate: dict[str, Any], phase_context: dict[str, Any]) -> tuple[float, ...]:
    """Return the deterministic selection key for the current manual phase.

    Selection uses only Cliff's delta and Wasserstein distance — no p-values.
    The goal is to drive both statistics toward 0 for ALL tracked metrics.

    Ranking tiers (lower is better for every component):
    1. Guard violation count — protected metrics whose |cd| or Wasserstein
       regressed beyond tolerance.  Fewer violations first.
    2. Mean |Cliff's delta| across ALL tracked metrics — overall effect-size
       proximity to real distribution.  Lower is better.
    3. Mean Wasserstein across ALL tracked metrics — overall distributional
       shape match.  Lower is better.
    4. Per-focus-metric tie-break using _manual_metric_row_key (cd, W, |med|).
    """
    phase_score = candidate.get("manual_phase_score") or _manual_phase_score(candidate, phase_context)
    guard = candidate.get("manual_phase_guard") or {}

    # Global mean |Cliff's delta| and mean Wasserstein across ALL group_eval metrics
    group_eval = candidate.get("group_eval_per_metric", {}) or {}
    cd_values: list[float] = []
    w_values: list[float] = []
    for _metric_name, metric_info in group_eval.items():
        cd = abs(float(metric_info.get("cliffs_delta", float("inf"))))
        w = float(metric_info.get("wasserstein_distance", float("inf")))
        if cd < float("inf"):
            cd_values.append(cd)
        if w < float("inf"):
            w_values.append(w)

    mean_cd = float(np.mean(cd_values)) if cd_values else float("inf")
    mean_w = float(np.mean(w_values)) if w_values else float("inf")

    focus_rows = phase_score.get("focus_metric_rows", [])

    key: list[float] = [
        float(guard.get("violation_count", 0)),   # fewer guard violations first
        mean_cd,                                    # lower mean |Cliff's delta| first
        mean_w,                                     # lower mean Wasserstein first
    ]
    # Per-focus-metric tie-break
    for row in focus_rows:
        key.extend(_manual_metric_row_key(row))
    return tuple(key)


def _fmt(v: float, fmt: str = ".4f") -> str:
    return f"{v:{fmt}}" if not _math.isnan(v) else "  N/A"


def _fmt_signed(v: float, fmt: str = ".1f") -> str:
    return f"{v:+{fmt}}" if not _math.isnan(v) else "   N/A"


def _print_candidate_score_summary(score: dict) -> None:
    """Print a compact robust-distribution summary for headline metrics."""
    pm = score.get("per_metric", {})
    if not pm:
        return

    has_robust = any("sim_median" in pm.get(key, {}) for key, _ in _HEADLINE_METRICS if key in pm)
    if not has_robust:
        print(f"  {'Metric':<28} {'real_med':>8} {'gen_med':>8} {'fail%':>6} {'direction'}")
        print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*6} {'-'*16}")
        for key, label in _HEADLINE_METRICS:
            if key not in pm:
                continue
            m = pm[key]
            real_med = float(m.get("real_median", float("nan")))
            gen_med = float(m.get("generated_median", float("nan")))
            fail_pct = float(m.get("fail_rate", 0.0)) * 100
            direction = m.get("direction", "")
            print(f"  {label:<28} {_fmt(real_med):>8} {_fmt(gen_med):>8} {fail_pct:>5.1f}% {direction}")
        return

    print(
        f"  {'metric':<28} {'sim_med':>8} {'real_p10':>10} {'real_p50':>10} "
        f"{'real_p90':>10} {'pct_rank':>9} {'robust_z':>10} {'status'}"
    )
    print(
        f"  {'-'*28} {'-'*8} {'-'*10} {'-'*10} "
        f"{'-'*10} {'-'*9} {'-'*10} {'-'*10}"
    )
    for key, label in _HEADLINE_METRICS:
        if key not in pm:
            continue
        m = pm[key]
        sim_med = float(m.get("sim_median", m.get("generated_median", float("nan"))))
        real_p10 = float(m.get("real_p10", float("nan")))
        real_p50 = float(m.get("real_median", float("nan")))
        real_p90 = float(m.get("real_p90", float("nan")))
        pct_rank = float(m.get("percentile_rank", float("nan")))
        robust_z = float(m.get("robust_z", float("nan")))
        status = m.get("status", "N/A")
        print(
            f"  {label:<28} {_fmt(sim_med):>8} {_fmt(real_p10):>10} {_fmt(real_p50):>10} "
            f"{_fmt(real_p90):>10} {_fmt(pct_rank, '.2f'):>9} {_fmt_signed(robust_z):>10} {status}"
        )


def _print_group_eval_summary(group_eval: dict, label_a: str = "group_a", label_b: str = "group_b") -> None:
    """Print Cliff's delta and significance for headline metrics from evaluate_group_vs_real output.

    Accepts both the flat {metric: stats} shape returned by evaluate_group_vs_real
    and the wrapped {"per_metric": {metric: stats}} shape.
    """
    pm = group_eval.get("per_metric", group_eval)
    if not pm:
        return
    print(f"  {'Metric':<28} {'cliff_d':>8} {'mwu_p':>10} {'sig?':>5}")
    print(f"  {'-'*28} {'-'*8} {'-'*10} {'-'*5}")
    for key, label in _HEADLINE_METRICS:
        if key not in pm:
            continue
        m = pm[key]
        cd    = float(m.get("cliffs_delta", float("nan")))
        mwu_p = float(m.get("mwu_p_value", float("nan")))
        sig   = "YES" if not _math.isnan(mwu_p) and mwu_p < 0.05 else "no"
        p_str = f"{mwu_p:.2e}" if not _math.isnan(mwu_p) else "    N/A"
        print(f"  {label:<28} {_fmt(cd):>8} {p_str:>10} {sig:>5}")


def _print_improvement_table(improvement: dict) -> None:
    """Print before→after per-headline-metric improvement table."""
    pm = improvement.get("per_metric", {})
    if not pm:
        return
    print(f"  {'Metric':<28} {'cd_before':>9} {'cd_after':>9} {'Δfail%':>7} {'impr?':>5}")
    print(f"  {'-'*28} {'-'*9} {'-'*9} {'-'*7} {'-'*5}")
    for key, label in _HEADLINE_METRICS:
        if key not in pm:
            continue
        m = pm[key]
        cd_b  = float(m.get("before_cliffs_delta", float("nan")))
        cd_a  = float(m.get("after_cliffs_delta",  float("nan")))
        dfail = float(m.get("fail_rate_reduction") or 0.0) * 100
        impr  = "YES" if m.get("improved") else "no"
        d_str = f"{dfail:+.1f}%" if not _math.isnan(dfail) else "  N/A"
        print(f"  {label:<28} {_fmt(cd_b):>9} {_fmt(cd_a):>9} {d_str:>7} {impr:>5}")


def _selection_ranking_rows(scored_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return candidates sorted by the active selection key with compact fields."""
    ranked = sorted(scored_candidates, key=_candidate_selection_key)
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ranked, start=1):
        family_scores = candidate.get("selection_family_scores", {}) or {}
        guardrail = family_scores.get("guardrail_core", {}) or {}
        semantic = family_scores.get("semantic_core", {}) or {}
        engagement = family_scores.get("engagement_core", {}) or {}
        length = family_scores.get("length_core", {}) or {}
        rows.append(
            {
                "rank": rank,
                "candidate_id": candidate.get("candidate_id"),
                "strategy_label": candidate.get("strategy_label", ""),
                "mechanism_family": candidate.get("mechanism_family", ""),
                "primary_layer": candidate.get("primary_layer", ""),
                "guardrail_out_of_range_count": int(guardrail.get("out_of_range_count", 0)),
                "guardrail_max_percentile_distance": float(
                    guardrail.get("max_percentile_distance", float("nan"))
                ),
                "semantic_out_of_range_count": int(semantic.get("out_of_range_count", 0)),
                "semantic_mean_percentile_distance": float(
                    semantic.get("mean_percentile_distance", float("nan"))
                ),
                "semantic_max_percentile_distance": float(
                    semantic.get("max_percentile_distance", float("nan"))
                ),
                "semantic_mean_abs_robust_z": float(
                    semantic.get("mean_abs_robust_z", float("nan"))
                ),
                "semantic_mean_abs_raw_robust_z": float(
                    semantic.get("mean_abs_raw_robust_z", float("nan"))
                ),
                "engagement_out_of_range_count": int(engagement.get("out_of_range_count", 0)),
                "engagement_mean_percentile_distance": float(
                    engagement.get("mean_percentile_distance", float("nan"))
                ),
                "engagement_max_percentile_distance": float(
                    engagement.get("max_percentile_distance", float("nan"))
                ),
                "engagement_mean_abs_robust_z": float(
                    engagement.get("mean_abs_robust_z", float("nan"))
                ),
                "engagement_mean_abs_raw_robust_z": float(
                    engagement.get("mean_abs_raw_robust_z", float("nan"))
                ),
                "length_mean_percentile_distance": float(
                    length.get("mean_percentile_distance", float("nan"))
                ),
                "length_mean_abs_raw_robust_z": float(
                    length.get("mean_abs_raw_robust_z", float("nan"))
                ),
                "quantile_fail_rate": float(candidate.get("quantile_fail_rate", float("nan"))),
                "mean_percentile_distance": float(
                    candidate.get("mean_percentile_distance", float("nan"))
                ),
                "mean_abs_robust_z": float(candidate.get("mean_abs_robust_z", float("nan"))),
                "ranking_mean_abs_delta": float(
                    candidate.get("ranking_mean_abs_delta", float("nan"))
                ),
                "ranking_fail_rate": float(candidate.get("ranking_fail_rate", float("nan"))),
            }
        )
    return rows


def _print_selection_ranking(scored_candidates: list[dict[str, Any]]) -> None:
    """Print the actual selection ordering used for best-candidate choice."""
    rows = _selection_ranking_rows(scored_candidates)
    if not rows:
        return

    print("    selection ranking (best→worst by actual winner key):")
    print(
        f"    {'cand':<6} {'g_oor':>5} {'s_oor':>5} {'s_mean':>7} {'s_max':>6} "
        f"{'s_rawz':>7} {'e_oor':>5} {'e_mean':>7} {'e_rawz':>7} {'l_mean':>7} "
        f"{'qfail':>7} {'pct':>7} {'r_z':>6} {'r|d|':>6} {'r_fail':>7}"
    )
    print(
        f"    {'-'*6} {'-'*5} {'-'*5} {'-'*7} {'-'*6} "
        f"{'-'*7} {'-'*5} {'-'*7} {'-'*7} {'-'*7} "
        f"{'-'*7} {'-'*6} {'-'*6} {'-'*7}"
    )
    for row in rows:
        print(
            f"    c{row['candidate_id']!s:<5} "
            f"{row['guardrail_out_of_range_count']:>5d} "
            f"{row['semantic_out_of_range_count']:>5d} "
            f"{_fmt(row['semantic_mean_percentile_distance'], '.2f'):>7} "
            f"{_fmt(row['semantic_max_percentile_distance'], '.2f'):>6} "
            f"{_fmt(row['semantic_mean_abs_raw_robust_z'], '.2f'):>7} "
            f"{row['engagement_out_of_range_count']:>5d} "
            f"{_fmt(row['engagement_mean_percentile_distance'], '.2f'):>7} "
            f"{_fmt(row['engagement_mean_abs_raw_robust_z'], '.2f'):>7} "
            f"{_fmt(row['length_mean_percentile_distance'], '.2f'):>7} "
            f"{_fmt(row['quantile_fail_rate'], '.4f'):>7} "
            f"{_fmt(row['mean_percentile_distance'], '.4f'):>7} "
            f"{_fmt(row['mean_abs_robust_z'], '.2f'):>6} "
            f"{_fmt(row['ranking_mean_abs_delta'], '.4f'):>6} "
            f"{_fmt(row['ranking_fail_rate'], '.4f'):>7}"
        )


def _manual_phase_ranking_rows(
    scored_candidates: list[dict[str, Any]],
    phase_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return candidates sorted by the active manual-phase selection key."""
    ranked = sorted(
        scored_candidates,
        key=lambda candidate: _manual_phase_selection_key(candidate, phase_context),
    )
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ranked, start=1):
        phase_score = candidate.get("manual_phase_score") or _manual_phase_score(candidate, phase_context)
        rows.append(
            {
                "rank": rank,
                "candidate_id": candidate.get("candidate_id"),
                "strategy_label": candidate.get("strategy_label", ""),
                "primary_layer": candidate.get("primary_layer", ""),
                "manual_phase_guard": candidate.get("manual_phase_guard", {}),
                "focus_metric_rows": phase_score.get("focus_metric_rows", []),
                "protected_metric_rows": phase_score.get("protected_metric_rows", []),
            }
        )
    return rows


def _print_manual_phase_selection_ranking(
    scored_candidates: list[dict[str, Any]],
    phase_context: dict[str, Any],
) -> None:
    """Print the actual ranking used in deterministic manual phase mode."""
    rows = _manual_phase_ranking_rows(scored_candidates, phase_context)
    if not rows:
        return
    print("    manual phase ranking (best→worst by active block metrics):")
    for row in rows:
        print(
            f"    c{row['candidate_id']!s:<5} "
            f"strategy={row['strategy_label'] or 'candidate'} "
            f"layer={row['primary_layer'] or 'both'}"
        )
        guard = row.get("manual_phase_guard", {}) or {}
        if guard:
            print(
                "      guard "
                f"violations={int(guard.get('violation_count', 0))} "
                f"max_severity={_fmt(float(guard.get('max_severity', 0.0)), '.3f')}"
            )
        for metric_row in row.get("focus_metric_rows", []):
            print(
                "      focus "
                f"{metric_row['metric']:<24} "
                f"W={_fmt(metric_row['wasserstein'], '.4f')} "
                f"Q={_fmt(metric_row['quantile_error'], '.4f')} "
                f"fail={_fmt(metric_row['empirical_fail_rate'], '.4f')} "
                f"|med|={_fmt(metric_row['abs_median_gap'], '.4f')} "
                f"|cd|={_fmt(metric_row['abs_cliffs_delta'], '.4f')} "
                f"mwu_p={_fmt(metric_row['mwu_p_value'], '.4f')} "
                f"ks_p={_fmt(metric_row['ks_p_value'], '.4f')} "
                f"oor={metric_row['out_of_range']} "
                f"pct={_fmt(metric_row['percentile_distance'], '.4f')} "
                f"raw_z={_fmt(metric_row['abs_raw_robust_z'], '.4f')}"
            )
        for metric_row in row.get("protected_metric_rows", []):
            print(
                "      prot  "
                f"{metric_row['metric']:<24} "
                f"W={_fmt(metric_row['wasserstein'], '.4f')} "
                f"Q={_fmt(metric_row['quantile_error'], '.4f')} "
                f"fail={_fmt(metric_row['empirical_fail_rate'], '.4f')} "
                f"|med|={_fmt(metric_row['abs_median_gap'], '.4f')} "
                f"|cd|={_fmt(metric_row['abs_cliffs_delta'], '.4f')} "
                f"mwu_p={_fmt(metric_row['mwu_p_value'], '.4f')} "
                f"ks_p={_fmt(metric_row['ks_p_value'], '.4f')} "
                f"oor={metric_row['out_of_range']} "
                f"pct={_fmt(metric_row['percentile_distance'], '.4f')} "
                f"raw_z={_fmt(metric_row['abs_raw_robust_z'], '.4f')}"
            )


def _serialize_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return JSON-friendly copies of per-metric comparison rows."""
    serialized: list[dict[str, Any]] = []
    for row in rows:
        serialized.append(
            {
                "metric": row.get("metric"),
                "wasserstein": float(row.get("wasserstein", float("inf"))),
                "quantile_error": float(row.get("quantile_error", float("inf"))),
                "empirical_fail_rate": float(row.get("empirical_fail_rate", float("inf"))),
                "abs_median_gap": float(row.get("abs_median_gap", float("inf"))),
                "abs_cliffs_delta": float(row.get("abs_cliffs_delta", float("inf"))),
                "mwu_sig": int(row.get("mwu_sig", 1)),
                "ks_sig": int(row.get("ks_sig", 1)),
                "mwu_p_value": float(row.get("mwu_p_value", 0.0)),
                "ks_p_value": float(row.get("ks_p_value", 0.0)),
                "out_of_range": int(row.get("out_of_range", 1)),
                "percentile_distance": float(row.get("percentile_distance", float("inf"))),
                "abs_raw_robust_z": float(row.get("abs_raw_robust_z", float("inf"))),
                "status": row.get("status", "missing"),
            }
        )
    return serialized


def _print_phase_watch_metrics(
    phase_context: dict[str, Any],
    focus_rows: list[dict[str, Any]],
    protected_rows: list[dict[str, Any]],
) -> None:
    """Print the metrics that matter for the current manual phase iteration."""
    print("  → Watch metrics this iteration:")
    print(f"    focus     : {phase_context.get('focus_metrics', [])}")
    if phase_context.get("protected_metrics"):
        print(f"    protected : {phase_context.get('protected_metrics', [])}")
    if focus_rows:
        print("    current focus metric stats (search-root / block incumbent):")
        for row in focus_rows:
            print(
                "      "
                f"{row['metric']}: "
                f"W={_fmt(row['wasserstein'], '.4f')}  "
                f"Q={_fmt(row['quantile_error'], '.4f')}  "
                f"fail={_fmt(row['empirical_fail_rate'], '.4f')}  "
                f"|med|={_fmt(row['abs_median_gap'], '.4f')}  "
                f"|cd|={_fmt(row['abs_cliffs_delta'], '.4f')}  "
                f"mwu_p={_fmt(row['mwu_p_value'], '.4f')}  "
                f"ks_p={_fmt(row['ks_p_value'], '.4f')}  "
                f"oor={row['out_of_range']}  "
                f"pct={_fmt(row['percentile_distance'], '.4f')}  "
                f"raw_z={_fmt(row['abs_raw_robust_z'], '.4f')}"
            )
    if protected_rows:
        print("    protected metric stats to preserve:")
        for row in protected_rows:
            print(
                "      "
                f"{row['metric']}: "
                f"W={_fmt(row['wasserstein'], '.4f')}  "
                f"Q={_fmt(row['quantile_error'], '.4f')}  "
                f"fail={_fmt(row['empirical_fail_rate'], '.4f')}  "
                f"|med|={_fmt(row['abs_median_gap'], '.4f')}  "
                f"|cd|={_fmt(row['abs_cliffs_delta'], '.4f')}  "
                f"mwu_p={_fmt(row['mwu_p_value'], '.4f')}  "
                f"ks_p={_fmt(row['ks_p_value'], '.4f')}  "
                f"oor={row['out_of_range']}  "
                f"pct={_fmt(row['percentile_distance'], '.4f')}  "
                f"raw_z={_fmt(row['abs_raw_robust_z'], '.4f')}"
            )


def _print_winner_selection_breakdown(winner: dict[str, Any]) -> None:
    """Print the family-level fields that actually determined the winner."""
    family_scores = winner.get("selection_family_scores", {}) or {}
    guardrail = family_scores.get("guardrail_core", {}) or {}
    semantic = family_scores.get("semantic_core", {}) or {}
    engagement = family_scores.get("engagement_core", {}) or {}
    length = family_scores.get("length_core", {}) or {}

    print("    selection breakdown:")
    print(
        "      guardrail_core: "
        f"oor={int(guardrail.get('out_of_range_count', 0))} "
        f"max_pct={_fmt(float(guardrail.get('max_percentile_distance', float('nan'))), '.3f')} "
        f"raw_z={_fmt(float(guardrail.get('mean_abs_raw_robust_z', guardrail.get('mean_abs_robust_z', float('nan')))), '.3f')}"
    )
    print(
        "      semantic_core : "
        f"oor={int(semantic.get('out_of_range_count', 0))} "
        f"mean_pct={_fmt(float(semantic.get('mean_percentile_distance', float('nan'))), '.3f')} "
        f"max_pct={_fmt(float(semantic.get('max_percentile_distance', float('nan'))), '.3f')} "
        f"raw_z={_fmt(float(semantic.get('mean_abs_raw_robust_z', semantic.get('mean_abs_robust_z', float('nan')))), '.3f')}"
    )
    print(
        "      engagement_core: "
        f"oor={int(engagement.get('out_of_range_count', 0))} "
        f"mean_pct={_fmt(float(engagement.get('mean_percentile_distance', float('nan'))), '.3f')} "
        f"max_pct={_fmt(float(engagement.get('max_percentile_distance', float('nan'))), '.3f')} "
        f"raw_z={_fmt(float(engagement.get('mean_abs_raw_robust_z', engagement.get('mean_abs_robust_z', float('nan')))), '.3f')}"
    )
    print(
        "      length_core   : "
        f"mean_pct={_fmt(float(length.get('mean_percentile_distance', float('nan'))), '.3f')} "
        f"raw_z={_fmt(float(length.get('mean_abs_raw_robust_z', length.get('mean_abs_robust_z', float('nan')))), '.3f')}"
    )
    print(
        "      overall       : "
        f"quantile_fail={_fmt(float(winner.get('quantile_fail_rate', float('nan'))), '.4f')} "
        f"pct_dist={_fmt(float(winner.get('mean_percentile_distance', float('nan'))), '.4f')} "
        f"robust_z={_fmt(float(winner.get('mean_abs_robust_z', float('nan'))), '.4f')} "
        f"ranking_|delta|={_fmt(float(winner.get('ranking_mean_abs_delta', float('nan'))), '.4f')} "
        f"ranking_fail={_fmt(float(winner.get('ranking_fail_rate', float('nan'))), '.4f')}"
    )


def _group_eval_selection_summary(group_eval: dict) -> dict[str, float | int]:
    """Summarize group-level evaluation into candidate selection metrics."""
    per_metric = group_eval.get("per_metric", group_eval)
    if not per_metric:
        return {
            "group_mean_abs_cliffs_delta": float("inf"),
            "group_overall_fail_rate": float("inf"),
            "group_metrics_sig_different": 0,
        }

    abs_deltas = [abs(float(info.get("cliffs_delta", 0.0))) for info in per_metric.values()]
    fail_rates = [float(info.get("empirical_fail_rate", 0.0)) for info in per_metric.values()]
    sig_count = sum(
        1
        for info in per_metric.values()
        if float(info.get("mwu_p_value", 1.0)) < 0.05
        or float(info.get("ks_p_value", 1.0)) < 0.05
    )
    return {
        "group_mean_abs_cliffs_delta": float(np.mean(abs_deltas)) if abs_deltas else 0.0,
        "group_overall_fail_rate": float(np.mean(fail_rates)) if fail_rates else 0.0,
        "group_metrics_sig_different": sig_count,
    }


def _candidate_selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    """Return the comparison key for candidate selection and incumbent checks."""
    if "quantile_fail_rate" in candidate:
        return candidate_selection_key(candidate)

    if (
        "group_mean_abs_cliffs_delta" in candidate
        or "group_overall_fail_rate" in candidate
    ):
        return (
            float(candidate.get("group_mean_abs_cliffs_delta", candidate.get("mean_abs_delta", float("inf")))),
            float(candidate.get("group_overall_fail_rate", candidate.get("fail_rate", float("inf")))),
            float(candidate.get("mean_abs_delta", float("inf"))),
            float(candidate.get("fail_rate", float("inf"))),
        )

    return (
        float(candidate.get("fail_rate", float("inf"))),
        float(candidate.get("mean_abs_delta", float("inf"))),
    )


def _current_best_selection_reference(state: "CalibrationState") -> dict[str, Any] | None:
    """Return the richest persisted incumbent payload for winner comparison."""
    if state.current_best_diagnostic and "quantile_fail_rate" in state.current_best_diagnostic:
        return state.current_best_diagnostic
    return state.current_best_score


def _group_score_key(group_info: dict[str, Any] | None) -> tuple[float, ...]:
    """Return a stable comparison key for one metric-group summary."""
    if not group_info:
        return (float("inf"), float("inf"), float("inf"))
    return (
        float(group_info.get("quantile_fail_rate", float("inf"))),
        float(group_info.get("mean_percentile_distance", float("inf"))),
        float(group_info.get("mean_abs_robust_z", float("inf"))),
    )


def _group_severity_key(group_info: dict[str, Any] | None) -> tuple[float, ...]:
    """Return a severity key where larger values mean the group is worse."""
    if not group_info:
        return (float("-inf"), float("-inf"), float("-inf"))
    return (
        float(group_info.get("quantile_fail_rate", 0.0)),
        float(group_info.get("mean_percentile_distance", 0.0)),
        float(group_info.get("mean_abs_robust_z", 0.0)),
    )


def _worst_group_order(score: dict[str, Any] | None) -> list[str]:
    """Return group names sorted from worst to best for the provided score."""
    if not score:
        return []
    group_scores = score.get("group_scores", {}) or {}
    return [
        name
        for name, _info in sorted(
            group_scores.items(),
            key=lambda item: _group_severity_key(item[1]),
            reverse=True,
        )
    ]


def _stagnation_count_from_entries(entries: list[dict[str, Any]]) -> int:
    """Return the number of consecutive iterations without a new best."""
    count = 0
    for entry in reversed(entries):
        if entry.get("selection", {}).get("beat_current_best", False):
            break
        count += 1
    return count


def _slim_candidate_diagnostic(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a persisted diagnostic payload for a scored candidate."""
    return {
        k: v
        for k, v in candidate.items()
        if k not in {"candidate_id", "candidate_dir", "overlay"}
    }


def _make_frontier_entry(
    candidate: dict[str, Any],
    preview: dict[str, Any] | None,
    iteration: int,
) -> dict[str, Any]:
    """Persist a scored candidate as a reusable frontier/search-root entry."""
    return {
        "iteration": iteration,
        "candidate_id": candidate.get("candidate_id"),
        "candidate_dir": candidate.get("candidate_dir"),
        "overlay": candidate.get("overlay", {}),
        "diagnostic": _slim_candidate_diagnostic(candidate),
        "strategy_label": (preview or {}).get("strategy_label", candidate.get("strategy_label", "")),
        "strategy": (preview or {}).get("strategy", ""),
        "primary_layer": (preview or {}).get("primary_layer", "both"),
        "mechanism_family": (preview or {}).get("mechanism_family", candidate.get("mechanism_family", "mixed")),
        "anti_incumbent": bool((preview or {}).get("anti_incumbent", False)),
    }


def _update_frontier(
    frontier: dict[str, dict[str, Any]],
    scored: list[dict[str, Any]],
    preview_by_id: dict[int, dict[str, Any]],
    iteration: int,
) -> dict[str, dict[str, Any]]:
    """Update the per-group frontier with any newly superior candidates."""
    updated = dict(frontier or {})
    for candidate in scored:
        group_scores = candidate.get("group_scores", {}) or {}
        preview = preview_by_id.get(int(candidate.get("candidate_id", -1)), {})
        for group_name, group_info in group_scores.items():
            existing = updated.get(group_name)
            candidate_key = _group_score_key(group_info) + _candidate_selection_key(candidate)
            existing_key = (
                _group_score_key((existing or {}).get("diagnostic", {}).get("group_scores", {}).get(group_name))
                + _candidate_selection_key((existing or {}).get("diagnostic", {}))
                if existing
                else (float("inf"),) * 8
            )
            if existing is None or candidate_key < existing_key:
                updated[group_name] = _make_frontier_entry(candidate, preview, iteration)
    return updated


def _frontier_prompt_summary(frontier: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return a compact frontier summary suitable for the reasoner prompt."""
    summary: dict[str, dict[str, Any]] = {}
    for group_name, entry in (frontier or {}).items():
        diagnostic = entry.get("diagnostic", {})
        group_info = (diagnostic.get("group_scores", {}) or {}).get(group_name, {})
        summary[group_name] = {
            "iteration": entry.get("iteration"),
            "candidate_id": entry.get("candidate_id"),
            "strategy_label": entry.get("strategy_label"),
            "mechanism_family": entry.get("mechanism_family"),
            "primary_layer": entry.get("primary_layer"),
            "anti_incumbent": bool(entry.get("anti_incumbent", False)),
            "group_quantile_fail_rate": group_info.get("quantile_fail_rate"),
            "group_mean_percentile_distance": group_info.get("mean_percentile_distance"),
            "group_mean_abs_robust_z": group_info.get("mean_abs_robust_z"),
            "overall_quantile_fail_rate": diagnostic.get("quantile_fail_rate"),
            "overall_mean_percentile_distance": diagnostic.get("mean_percentile_distance"),
            "overall_mean_abs_robust_z": diagnostic.get("mean_abs_robust_z"),
        }
    return summary


def _choose_search_root(
    state: "CalibrationState",
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None, str, str]:
    """Choose the overlay/diagnostic that the next iteration should branch from."""
    default_reason = "global_best"
    if not state.current_best_overlay:
        return {}, None, None, "global_best", default_reason

    if state.stagnation_count < _STAGNATION_TRIGGER:
        return (
            state.current_best_overlay,
            state.current_best_diagnostic,
            state.current_best_candidate_dir,
            "global_best",
            default_reason,
        )

    current_best_score = state.current_best_score or {}
    current_best_dir = state.current_best_candidate_dir
    frontier = state.frontier or {}
    worst_groups = _worst_group_order(current_best_score)
    for group_name in worst_groups:
        frontier_entry = frontier.get(group_name)
        if not frontier_entry:
            continue
        if frontier_entry.get("candidate_dir") == current_best_dir:
            continue
        challenger_group = (
            frontier_entry.get("diagnostic", {})
            .get("group_scores", {})
            .get(group_name, {})
        )
        incumbent_group = current_best_score.get("group_scores", {}).get(group_name, {})
        if _group_score_key(challenger_group) < _group_score_key(incumbent_group):
            return (
                frontier_entry.get("overlay", {}) or state.current_best_overlay,
                frontier_entry.get("diagnostic") or state.current_best_diagnostic,
                frontier_entry.get("candidate_dir") or state.current_best_candidate_dir,
                "challenger_root",
                (
                    f"stagnation_count={state.stagnation_count}; branching from frontier"
                    f" best for worst group '{group_name}' via "
                    f"{frontier_entry.get('strategy_label', 'candidate')}"
                ),
            )

    return (
        state.current_best_overlay,
        state.current_best_diagnostic,
        state.current_best_candidate_dir,
        "global_best",
        f"stagnation_count={state.stagnation_count}; no challenger frontier entry beat global_best on its target group",
    )


def _sanitize_overlay(
    registry: KnobRegistry,
    overlay: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Sanitize one overlay against the registry and deduplicate errors."""
    cleaned, errors = registry.sanitize_overlay(overlay)
    if STRUCTURED_PHASE_BLOCKS_KEY in overlay:
        cleaned[STRUCTURED_PHASE_BLOCKS_KEY] = overlay[STRUCTURED_PHASE_BLOCKS_KEY]
        cleaned = render_structured_overlay(cleaned)
        structured_error = f"Unknown knob: '{STRUCTURED_PHASE_BLOCKS_KEY}'."
        errors = [err for err in errors if err != structured_error]
    deduped = list(dict.fromkeys(errors))
    return cleaned, deduped


def _composite_thread_key(product: str | None, thread_id: str | None) -> str:
    """Return a stable thread key compatible with split-aware few-shot filters."""
    product_str = str(product or "").strip()
    thread_str = str(thread_id or "").strip()
    if product_str and thread_str:
        return f"{product_str}::{thread_str}"
    return thread_str


def _format_terminal_value(value: Any) -> str:
    """Render a knob value for terminal output without truncation."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, str):
        return value.strip()

    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _maybe_record_completed_phase_summary(
    state: "CalibrationState",
    phase_context: dict[str, Any],
) -> None:
    """Persist the current cumulative best as the completed best for a phase block."""
    if not state.current_best_overlay or not state.current_best_diagnostic:
        return
    phase_name = str(phase_context.get("name", "")).strip()
    if not phase_name:
        return
    if any(summary.get("phase_name") == phase_name for summary in state.completed_phase_summaries):
        return
    summary = {
        "phase_name": phase_name,
        "phase_label": phase_context.get("label"),
        "block_label": phase_context.get("block_label"),
        "iteration_end": phase_context.get("iteration_end"),
        "focus_metrics": list(phase_context.get("focus_metrics", [])),
        "overlay": dict(state.current_best_overlay),
        "diagnostic": dict(state.current_best_diagnostic),
        "candidate_dir": state.current_best_candidate_dir,
    }
    state.completed_phase_summaries.append(summary)


def _completed_phase_prompt_summary(completed_phase_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact phase-best summary for prompt consumption."""
    prompt_rows: list[dict[str, Any]] = []
    for summary in completed_phase_summaries:
        diagnostic = summary.get("diagnostic", {}) or {}
        prompt_rows.append(
            {
                "phase_name": summary.get("phase_name"),
                "phase_label": summary.get("phase_label"),
                "block_label": summary.get("block_label"),
                "focus_metrics": summary.get("focus_metrics", []),
                "quantile_fail_rate": diagnostic.get("quantile_fail_rate"),
                "mean_percentile_distance": diagnostic.get("mean_percentile_distance"),
                "mean_abs_robust_z": diagnostic.get("mean_abs_robust_z"),
                "overlay": summary.get("overlay", {}),
            }
        )
    return prompt_rows


def _knob_runtime_location(name: str, knob: dict[str, Any]) -> str:
    """Describe where a persisted calibration text slot is consumed."""
    location_overrides = {
        "persona.generation_guidance": "persona generator / calibration persona guidance block",
        "prompt.comment_style_guidance": "system prompt + action prompt / calibration comment guidance block",
    }
    if name in location_overrides:
        return location_overrides[name]
    return f"{knob['layer']} runtime / {knob['domain']}"


def _overlay_change_records(
    registry: KnobRegistry,
    previous_overlay: dict[str, Any],
    candidate_overlay: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return structured change records for the effective candidate changes."""
    changed = diff_overlay(previous_overlay, candidate_overlay)
    records: list[dict[str, Any]] = []

    def _sort_key(name: str) -> tuple[str, str]:
        try:
            knob = registry.get(name)
        except KeyError:
            return ("zz_internal", name)
        return (str(knob.get("layer", "")), name)

    for name in sorted(changed.keys(), key=_sort_key):
        try:
            knob = registry.get(name)
        except KeyError:
            continue
        old_value = previous_overlay.get(name, knob["default"])
        new_value = candidate_overlay.get(name, knob["default"])
        record: dict[str, Any] = {
            "name": name,
            "layer": knob["layer"],
            "domain": knob["domain"],
            "type": knob["type"],
            "description": knob["description"],
            "runtime_location": _knob_runtime_location(name, knob),
            "old_value": old_value,
            "new_value": new_value,
        }

        if knob["type"] == "distribution":
            old_dist = old_value if isinstance(old_value, dict) else {}
            new_dist = new_value if isinstance(new_value, dict) else {}
            subchanges: list[dict[str, Any]] = []
            for key in knob["keys"]:
                old_key_val = float(old_dist.get(key, 0.0))
                new_key_val = float(new_dist.get(key, 0.0))
                if abs(old_key_val - new_key_val) > 1e-12:
                    subchanges.append({
                        "key": key,
                        "old_value": old_key_val,
                        "new_value": new_key_val,
                    })
            record["changed_keys"] = subchanges

        records.append(record)

    return records


def _print_candidate_change_preview(
    candidate_id: int,
    strategy_label: str,
    primary_layer: str,
    strategy: str,
    rationale: str,
    changes: list[dict[str, Any]],
    validation_errors: list[str] | None = None,
) -> None:
    """Print one candidate's exact persisted text edits to stdout."""
    print(f"      [{candidate_id}] {strategy_label} (layer={primary_layer})")
    if strategy:
        print(f"          strategy : {strategy}")
    if rationale:
        print(f"          rationale: {rationale}")

    if not changes:
        print("          changes  : no effective edits after validation")
    else:
        print(f"          changes  : {len(changes)} applied edit(s)")
        for change in changes:
            header = (
                f"          - {change['name']} "
                f"[{change['layer']} | {change['domain']}]"
            )
            print(header)
            runtime_location = change.get(
                "runtime_location",
                f"{change['layer']} runtime / {change['domain']}",
            )
            print(f"            applies at: {runtime_location}")
            if change["type"] == "distribution":
                subchanges = change.get("changed_keys", [])
                if subchanges:
                    joined = "; ".join(
                        f"{entry['key']} {entry['old_value']:.4f} -> {entry['new_value']:.4f}"
                        for entry in subchanges
                    )
                    print(f"            {joined}")
                else:
                    print("            no distribution entries changed")
            else:
                old_value = _format_terminal_value(change["old_value"])
                new_value = _format_terminal_value(change["new_value"])
                if change["type"] == "text":
                    print("            old:")
                    for line in old_value.splitlines() or [""]:
                        print(f"              {line}")
                    print("            new:")
                    for line in new_value.splitlines() or [""]:
                        print(f"              {line}")
                else:
                    print(f"            {old_value} -> {new_value}")

    if validation_errors:
        print("          validation:")
        for err in validation_errors:
            print(f"            - {err}")


# ---------------------------------------------------------------------------
# Sample thread extraction (for reasoner prompt)
# ---------------------------------------------------------------------------

def _extract_sample_real_thread(
    few_shot_dir: Path,
    train_thread_ids: list[str],
    max_comments: int = 15,
    count: int = 2,
) -> str:
    """Extract *count* real Reddit threads as readable text samples.

    Picks random train-only threads from the few_shot_dir.
    Returns a formatted string suitable for inclusion in the reasoner prompt.
    """
    import random as _rnd

    # Find .comments.jsonl files
    comment_files: list[Path] = []
    for sub in sorted(few_shot_dir.iterdir()):
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.name.endswith(".comments.jsonl"):
                comment_files.append(f)

    if not comment_files:
        return ""

    # Collect all eligible threads across files
    all_threads: list[tuple[str, list[dict]]] = []
    _rnd.shuffle(comment_files)
    for cf in comment_files[:20]:
        threads: dict[str, list[dict]] = {}
        try:
            for line in cf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                tid = str(obj.get("post_id", ""))
                composite_id = _composite_thread_key(cf.parent.name, tid)
                if tid and (
                    not train_thread_ids
                    or tid in train_thread_ids
                    or composite_id in train_thread_ids
                ):
                    threads.setdefault(tid, []).append(obj)
        except Exception:
            continue

        for tid, comments in threads.items():
            if len(comments) >= 3:
                all_threads.append((tid, comments))

    if not all_threads:
        return ""

    # Shuffle and pick up to *count* threads
    _rnd.shuffle(all_threads)
    samples: list[str] = []
    for tid, comments in all_threads[:count]:
        lines = [f"[Thread ID: {tid}]"]
        title = comments[0].get("post_title", "")
        if title:
            lines.append(f"Title: {title}")
        lines.append("")
        for c in comments[:max_comments]:
            author = c.get("author", "anonymous")
            body = c.get("body", "").strip()
            depth = c.get("depth", 0)
            indent = "  " * int(depth)
            lines.append(f"{indent}[{author}] (depth={depth}): {body[:200]}")
        samples.append("\n".join(lines))

    return "\n\n---\n\n".join(samples)


def _extract_sample_sim_thread(
    best_candidate_dir: Path | None,
    max_comments: int = 15,
    count: int = 2,
) -> str:
    """Extract up to *count* simulated threads from the best candidate's discussion.json.

    Returns a formatted string suitable for inclusion in the reasoner prompt.
    """
    if best_candidate_dir is None:
        return ""

    # Allow the stored path to point directly at a simulation directory.
    direct_discussion = best_candidate_dir / "discussion.json"
    if direct_discussion.exists():
        discussion_path = direct_discussion
    else:
        discussion_path = None

    # Find discussion.json in sim_output
    sim_output = best_candidate_dir / "sim_output"
    if discussion_path is None and sim_output.exists():
        for sub in sim_output.iterdir():
            if sub.is_dir():
                dp = sub / "discussion.json"
                if dp.exists():
                    discussion_path = dp
                    break

    if discussion_path is None:
        return ""

    try:
        data = json.loads(discussion_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    posts = data.get("posts", [])
    if not posts:
        return ""

    # Pick up to *count* posts with enough comments
    samples: list[str] = []
    for post in posts:
        comments = post.get("comments", [])
        if len(comments) >= 3:
            lines = [f"[Simulated Thread]"]
            lines.append(f"Post: {post.get('content', '')[:200]}")
            lines.append("")
            for c in comments[:max_comments]:
                author = c.get("author", "agent")
                body = c.get("content", "").strip()
                depth = c.get("depth", 0)
                indent = "  " * int(depth)
                lines.append(f"{indent}[{author}] (depth={depth}): {body[:200]}")
            samples.append("\n".join(lines))
            if len(samples) >= count:
                break

    return "\n\n---\n\n".join(samples)


def _find_reusable_vanilla_sim_dir(vanilla_scores_csv: Path | None) -> Path | None:
    """Return one existing vanilla simulation directory for iter-0 reuse."""
    if vanilla_scores_csv is None:
        return None
    runs_dir = vanilla_scores_csv.parent / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(
        path for path in runs_dir.iterdir()
        if path.is_dir() and (path / "thread_metrics_summary.csv").exists()
    )
    return candidates[0] if candidates else None


def _resolve_eval_thread_target(reference_thread_count: int, requested_cap: int) -> int:
    """Return the effective evaluation-thread target.

    A positive ``requested_cap`` is treated as an upper bound, so evaluation
    runs target ``min(reference_thread_count, requested_cap)`` threads.
    Non-positive values preserve the prior "no explicit cap" behavior.
    """

    if requested_cap <= 0:
        return 0
    if reference_thread_count <= 0:
        return 0
    return min(reference_thread_count, requested_cap)


def _make_reused_baseline_candidate_result(
    iter_dir: Path,
    overlay: dict[str, Any],
    source_sim_dir: Path,
) -> dict[str, Any]:
    """Create a pseudo candidate result that reuses a precomputed vanilla sim."""
    candidate_dir = iter_dir / "candidates" / "candidate_0"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    save_overlay(overlay, candidate_dir / "overlay.json")
    (candidate_dir / "reused_from.txt").write_text(
        str(source_sim_dir), encoding="utf-8",
    )
    return {
        "candidate_id": 0,
        "candidate_dir": str(source_sim_dir),
        "sim_dir": str(source_sim_dir),
        "success": True,
        "returncode": 0,
        "reused": True,
        "reused_from": str(source_sim_dir),
    }


def _load_iteration_checkpoint(iter_dir: Path) -> dict[str, Any] | None:
    """Load a partially-completed iteration's saved candidate set, if present."""
    diagnosis_path = iter_dir / "diagnosis.json"
    if not diagnosis_path.exists():
        return None

    try:
        payload = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    candidate_dir_root = iter_dir / "candidates"
    if not candidate_dir_root.exists():
        return None

    candidate_dirs = sorted(
        path for path in candidate_dir_root.iterdir()
        if path.is_dir() and path.name.startswith("candidate_")
    )
    if not candidate_dirs:
        return None

    overlays: list[dict[str, Any]] = []
    for candidate_dir in candidate_dirs:
        overlay_path = candidate_dir / "overlay.json"
        if not overlay_path.exists():
            return None
        try:
            overlays.append(json.loads(overlay_path.read_text(encoding="utf-8")))
        except Exception:
            return None

    candidate_previews = payload.get("candidates", [])
    validation_errors = payload.get("validation_errors", [])
    overlay_diff = payload.get("overlay_diff", {})

    return {
        "strategy_label": payload.get("strategy_label", "resumed_iteration"),
        "diagnosis": payload.get("diagnosis", ""),
        "candidate_previews": candidate_previews,
        "overlays": overlays,
        "overlay_diff": overlay_diff if isinstance(overlay_diff, dict) else {},
        "validation_errors": validation_errors if isinstance(validation_errors, list) else [],
    }


# ---------------------------------------------------------------------------
# CalibrationState
# ---------------------------------------------------------------------------

class CalibrationState:
    """Persistent state for a calibration run, with resume support.

    The state is serialised to ``output_dir/calibration_state.json``.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.state_path = self.output_dir / "calibration_state.json"
        self.current_best_overlay: dict = {}
        self.current_best_score: dict | None = None
        self.current_best_diagnostic: dict | None = None
        self.current_best_candidate_dir: str | None = None
        self.current_search_root_overlay: dict = {}
        self.current_search_root_diagnostic: dict | None = None
        self.current_search_root_candidate_dir: str | None = None
        self.current_search_root_mode: str = "global_best"
        self.current_search_root_reason: str = "global_best"
        self.frontier: dict[str, dict[str, Any]] = {}
        self.stagnation_count: int = 0
        self.completed_iterations: int = 0
        self.current_phase_name: str | None = None
        self.completed_phase_summaries: list[dict[str, Any]] = []
        self.manual_block_phase_name: str | None = None
        self.manual_block_best_overlay: dict = {}
        self.manual_block_best_score: dict | None = None
        self.manual_block_best_diagnostic: dict | None = None
        self.manual_block_best_candidate_dir: str | None = None

        if self.state_path.exists():
            self._load()

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "current_best_overlay": self.current_best_overlay,
            "current_best_score": self.current_best_score,
            "current_best_diagnostic": self.current_best_diagnostic,
            "current_best_candidate_dir": self.current_best_candidate_dir,
            "current_search_root_overlay": self.current_search_root_overlay,
            "current_search_root_diagnostic": self.current_search_root_diagnostic,
            "current_search_root_candidate_dir": self.current_search_root_candidate_dir,
            "current_search_root_mode": self.current_search_root_mode,
            "current_search_root_reason": self.current_search_root_reason,
            "frontier": self.frontier,
            "stagnation_count": self.stagnation_count,
            "completed_iterations": self.completed_iterations,
            "current_phase_name": self.current_phase_name,
            "completed_phase_summaries": self.completed_phase_summaries,
            "manual_block_phase_name": self.manual_block_phase_name,
            "manual_block_best_overlay": self.manual_block_best_overlay,
            "manual_block_best_score": self.manual_block_best_score,
            "manual_block_best_diagnostic": self.manual_block_best_diagnostic,
            "manual_block_best_candidate_dir": self.manual_block_best_candidate_dir,
        }
        self.state_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load(self) -> None:
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.current_best_overlay = render_structured_overlay(raw.get("current_best_overlay", {}))
        self.current_best_score = raw.get("current_best_score")
        self.current_best_diagnostic = raw.get("current_best_diagnostic")
        self.current_best_candidate_dir = raw.get("current_best_candidate_dir")
        self.current_search_root_overlay = render_structured_overlay(raw.get(
            "current_search_root_overlay",
            self.current_best_overlay,
        ))
        self.current_search_root_diagnostic = raw.get(
            "current_search_root_diagnostic",
            self.current_best_diagnostic,
        )
        self.current_search_root_candidate_dir = raw.get(
            "current_search_root_candidate_dir",
            self.current_best_candidate_dir,
        )
        self.current_search_root_mode = raw.get("current_search_root_mode", "global_best")
        self.current_search_root_reason = raw.get("current_search_root_reason", "global_best")
        self.frontier = raw.get("frontier", {})
        self.stagnation_count = raw.get("stagnation_count", 0)
        self.completed_iterations = raw.get("completed_iterations", 0)
        self.current_phase_name = raw.get("current_phase_name")
        self.completed_phase_summaries = raw.get("completed_phase_summaries", [])
        self.manual_block_phase_name = raw.get("manual_block_phase_name")
        self.manual_block_best_overlay = render_structured_overlay(raw.get(
            "manual_block_best_overlay",
            self.current_best_overlay,
        ))
        self.manual_block_best_score = raw.get(
            "manual_block_best_score",
            self.current_best_score,
        )
        self.manual_block_best_diagnostic = raw.get(
            "manual_block_best_diagnostic",
            self.current_best_diagnostic,
        )
        self.manual_block_best_candidate_dir = raw.get(
            "manual_block_best_candidate_dir",
            self.current_best_candidate_dir,
        )


# ---------------------------------------------------------------------------
# run_calibration_loop
# ---------------------------------------------------------------------------

def run_calibration_loop(
    output_dir: Path,
    real_train_csv: Path,
    real_val_csv: Path,
    real_test_csv: Path,
    reference_run_config: dict,
    max_iterations: int = 10,
    candidates_per_iter: int = 5,
    parallel: int = 1,
    calibration_model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    seed: int = 42,
    python: str = sys.executable,
    repo_root: Path | None = None,
    metrics: list[str] | None = None,
    metric_definitions: str = "",
    device: str = "cpu",
    final_sim_runs: int = 12,
    vanilla_scores_csv: Path | None = None,
    rerun_phase0_vanilla: bool = False,
    min_sim_threads: int = 0,
    metric_parallel: int = 2,
    calibration_reasoning_effort: str | None = None,
    simulation_reasoning_effort: str | None = None,
    stop_after_phase1: bool = False,
    calibration_rounds: int | None = None,
    combination_start_iteration: int | None = None,
) -> dict:
    """Main calibration loop with train/val/test splits.

    Phases
    ------
    Phase 0  Before-calibration group evaluation (vanilla vs real_test).
    Phase 1  Calibration loop (per-thread empirical p-value diagnostics).
    Phase 2  After-calibration group evaluation (calibrated vs real_test).
    Phase 3  Improvement analysis (before vs after).

    Parameters
    ----------
    real_train_csv : Path
        Thread scores CSV for the train split — used by the LLM reasoner.
    real_val_csv : Path
        Thread scores CSV for the validation split — used for candidate scoring.
    real_test_csv : Path
        Thread scores CSV for the test split — used for before/after evaluation.
    vanilla_scores_csv : Path | None
        Pre-existing vanilla simulation scores CSV.  If provided, used as the
        before-calibration baseline for the improvement analysis.  If absent,
        the improvement analysis is skipped.
    final_sim_runs : int
        Number of fresh simulations for the after-calibration evaluation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _MANUAL_PHASE_MODE:
        manual_iterations = _manual_total_edited_iterations()
        if int(max_iterations) != manual_iterations:
            print(
                f"  [manual phase] overriding edited iterations "
                f"{int(max_iterations)} → {manual_iterations}",
                flush=True,
            )
        max_iterations = manual_iterations

    if metrics is None:
        metrics = DEFAULT_METRICS
    ranking_metrics = PRIMARY_CALIBRATION_METRICS
    if combination_start_iteration is None:
        combination_start_iteration = max_iterations // 2
    combination_start_iteration = max(1, min(int(combination_start_iteration), max_iterations))
    if repo_root is None:
        repo_root = Path(__file__).parent.parent

    # -----------------------------------------------------------------------
    # Initialise components
    # -----------------------------------------------------------------------
    registry = KnobRegistry()
    log = CalibrationLog(output_dir / "calibration_log.json")
    state = CalibrationState(output_dir=output_dir)
    sanitized_best_overlay, best_overlay_errors = _sanitize_overlay(
        registry, state.current_best_overlay
    )
    if best_overlay_errors:
        print("  [overlay validation] Sanitized persisted best overlay:")
        for err in best_overlay_errors:
            print(f"    - {err}")
        state.current_best_overlay = sanitized_best_overlay
        state.save()
    if state.current_best_overlay and not state.current_search_root_overlay:
        state.current_search_root_overlay = dict(state.current_best_overlay)
        state.current_search_root_diagnostic = state.current_best_diagnostic
        state.current_search_root_candidate_dir = state.current_best_candidate_dir
        state.current_search_root_mode = "global_best"
        state.current_search_root_reason = "global_best"
        state.save()
    if OpenAI is not None:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        # openai < 1.0: store credentials on a simple namespace
        import types
        client = types.SimpleNamespace(api_key=api_key, base_url=base_url)

    # -----------------------------------------------------------------------
    # Compute baselines from train and val splits
    # -----------------------------------------------------------------------
    val_df = pd.read_csv(real_val_csv)
    real_test_df = pd.read_csv(real_test_csv)
    train_baseline = compute_baseline_from_csv(real_train_csv, metrics)
    val_baseline = compute_baseline_from_csv(real_val_csv, metrics)
    before_generated_df = pd.read_csv(vanilla_scores_csv) if vanilla_scores_csv is not None else None
    reusable_vanilla_sim_dir = _find_reusable_vanilla_sim_dir(vanilla_scores_csv)

    def _baseline_summary(baseline: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for m, v in baseline.items():
            arr = np.asarray(v["values"], dtype=float)
            summary[m] = {
                "median": v["median"],
                "mean": v["mean"],
                "std": float(np.std(arr)) if arr.size > 0 else 0.0,
                "p10": float(np.percentile(arr, 10)) if arr.size > 0 else float("nan"),
                "p25": float(np.percentile(arr, 25)) if arr.size > 0 else float("nan"),
                "p75": float(np.percentile(arr, 75)) if arr.size > 0 else float("nan"),
                "p90": float(np.percentile(arr, 90)) if arr.size > 0 else float("nan"),
                "min": float(np.min(arr)) if arr.size > 0 else float("nan"),
                "max": float(np.max(arr)) if arr.size > 0 else float("nan"),
                "n": int(arr.size),
            }
        return summary

    # Train remains the qualitative source split; validation is the actual
    # reference distribution used for candidate scoring and ranking.
    train_summary = _baseline_summary(train_baseline)
    val_summary = _baseline_summary(val_baseline)

    (output_dir / "real_train_baseline_metrics.json").write_text(
        json.dumps(train_summary, indent=2), encoding="utf-8",
    )
    (output_dir / "real_val_baseline_metrics.json").write_text(
        json.dumps(val_summary, indent=2), encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Extract train thread keys for few-shot filtering (no val/test leakage)
    # -----------------------------------------------------------------------
    train_df = pd.read_csv(real_train_csv)
    train_thread_ids: list[str] = []
    if "thread_id" in train_df.columns:
        if "product" in train_df.columns:
            train_pairs = train_df[["product", "thread_id"]].dropna(subset=["thread_id"])
            train_thread_ids = [
                _composite_thread_key(product, thread_id)
                for product, thread_id in zip(
                    train_pairs["product"].astype(str),
                    train_pairs["thread_id"].astype(str),
                )
                if str(thread_id).strip()
            ]
        else:
            train_thread_ids = train_df["thread_id"].dropna().astype(str).tolist()
        train_thread_ids = list(dict.fromkeys(train_thread_ids))
        train_ids_path = output_dir / "train_thread_ids.json"
        train_ids_path.write_text(
            json.dumps(train_thread_ids, ensure_ascii=False), encoding="utf-8",
        )
        reference_run_config["few_shot_thread_ids"] = str(train_ids_path)

    # -----------------------------------------------------------------------
    # Phase 0: Before-calibration group evaluation (vanilla vs real_test)
    # -----------------------------------------------------------------------
    before_eval: dict[str, dict] | None = None
    _before_eval_path = output_dir / "before_calibration_group_eval.json"
    _before_generated_scores_path = output_dir / "before_calibration_generated_scores.csv"
    _before_reused_sim_path = output_dir / "before_calibration_reused_sim_dir.txt"
    if rerun_phase0_vanilla:
        if _before_eval_path.exists() and _before_generated_scores_path.exists():
            before_eval = json.loads(_before_eval_path.read_text(encoding="utf-8"))
            before_generated_df = pd.read_csv(_before_generated_scores_path)
            if _before_reused_sim_path.exists():
                reusable_vanilla_sim_dir = Path(
                    _before_reused_sim_path.read_text(encoding="utf-8").strip()
                )
            print("\n" + "=" * 60)
            print("PHASE 0: Before-calibration group evaluation (skipped — already done)")
            print("=" * 60)
        else:
            before_reference_run_config = _force_vanilla_backbone(
                reference_run_config
            )
            before_eval, before_generated_df, reusable_vanilla_sim_dir = _run_before_calibration_evaluation(
                output_dir=output_dir,
                real_test_csv=real_test_csv,
                reference_run_config=before_reference_run_config,
                sim_runs=final_sim_runs,
                metrics=metrics,
                python=python,
                repo_root=repo_root,
                device=device,
                min_sim_threads=min_sim_threads,
                metric_parallel=metric_parallel,
                simulation_reasoning_effort=simulation_reasoning_effort,
            )
        if before_eval:
            print("  Vanilla vs real_test (key metrics):")
            _print_group_eval_summary(before_eval)
    elif vanilla_scores_csv is not None:
        if _before_eval_path.exists():
            before_eval = json.loads(_before_eval_path.read_text(encoding="utf-8"))
            print("\n" + "=" * 60)
            print("PHASE 0: Before-calibration group evaluation (skipped — already done)")
            print("=" * 60)
        else:
            before_eval, before_generated_df, _ = _run_before_calibration_evaluation(
                output_dir=output_dir,
                real_test_csv=real_test_csv,
                metrics=metrics,
                vanilla_scores_csv=vanilla_scores_csv,
            )
        if before_eval:
            print("  Vanilla vs real_test (key metrics):")
            _print_group_eval_summary(before_eval)

    # -----------------------------------------------------------------------
    # Phase 1: Calibration loop — resume status
    # -----------------------------------------------------------------------
    total_phase1_iterations = _phase1_total_iterations(max_iterations)
    starting_completed_iterations = state.completed_iterations
    reported_completed_iterations = _phase1_reported_iteration_count(state.completed_iterations)
    remaining = max(0, max_iterations - reported_completed_iterations)
    print(f"\n{'='*60}")
    print("PHASE 1: Calibration loop")
    print(f"{'='*60}")
    if state.completed_iterations > 0:
        best = state.current_best_score or {}
        print(f"  Resumed at iteration {reported_completed_iterations}/{max_iterations}")
        if best.get("quantile_fail_rate") is not None:
            print(
                f"  Best so far → quantile_fail={best['quantile_fail_rate']:.4f}  "
                f"pct_dist={best.get('mean_percentile_distance', float('nan')):.4f}  "
                f"robust_z={best.get('mean_abs_robust_z', float('nan')):.4f}"
            )
            print(
                f"                 legacy fail_rate={best.get('fail_rate', float('nan')):.4f}  "
                f"|delta|={best.get('mean_abs_delta', float('nan')):.4f}"
            )
        else:
            print(
                f"  Best so far → fail_rate={best.get('fail_rate', float('nan')):.4f}  "
                f"|delta|={best.get('mean_abs_delta', float('nan')):.4f}"
            )
        if state.current_search_root_mode != "global_best":
            print(
                f"  Search root → {state.current_search_root_mode} "
                f"({state.current_search_root_reason})"
            )
    else:
        print(f"  Starting fresh — {max_iterations} iterations planned")
    print(f"  Iterations remaining: {remaining}")

    # Build a separate run config for Phase 1 calibration iterations.
    # If --calibration-rounds is set, use fewer rounds during candidate
    # ranking (iterations 1+) for faster turnaround.  Iteration 0 (baseline)
    # and Phase 2 (final evaluation) always use the full rounds.
    phase1_run_config = dict(reference_run_config)
    if calibration_rounds is not None and calibration_rounds != reference_run_config.get("rounds"):
        phase1_run_config["rounds"] = calibration_rounds
        print(f"  Phase 1 calibration rounds: {calibration_rounds} (full: {reference_run_config.get('rounds')})")

    for iteration in range(state.completed_iterations, total_phase1_iterations):
        iter_dir = output_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        iteration_checkpoint = _load_iteration_checkpoint(iter_dir)
        resume_iteration = iteration_checkpoint is not None
        manual_iteration = iteration - 1 if _MANUAL_PHASE_MODE else iteration
        manual_block_active = _MANUAL_PHASE_MODE and iteration > 0
        phase_context = (
            _manual_phase_context(manual_iteration)
            if manual_block_active else {}
        )
        if _MANUAL_PHASE_MODE:
            active_phase_name = str(phase_context.get("name", "")).strip()
            if iteration == 0:
                state.current_phase_name = None
                state.manual_block_phase_name = None
            else:
                if state.current_phase_name and state.current_phase_name != active_phase_name:
                    previous_phase = _manual_phase_context(manual_iteration - 1)
                    _manual_commit_block_best(state, previous_phase)
                if state.manual_block_phase_name != active_phase_name:
                    _manual_start_block(state, phase_context)
                state.current_phase_name = active_phase_name
            state.save()

        display_iteration = manual_iteration + 1 if _MANUAL_PHASE_MODE and iteration > 0 else 0
        print(f"\n[Iter {display_iteration}/{max_iterations}] ── {'baseline run' if iteration == 0 else 'LLM reasoner → generate candidates'}")
        if _MANUAL_PHASE_MODE:
            if iteration == 0:
                print(
                    "  → Manual phase warm-start baseline "
                    f"(not counted against the {max_iterations} edited iterations)"
                )
            else:
                print(
                    "  → Manual phase: "
                    f"{phase_context.get('label')} "
                    f"({phase_context.get('block_label')}, focus={phase_context.get('focus_metrics')})"
                )
        if resume_iteration:
            print("  → Resuming from saved iteration checkpoint...")

        # -------------------------------------------------------------------
        # Build candidate overlays
        # -------------------------------------------------------------------
        parsed: dict = {}
        candidates_list: list[dict] = []
        candidate_previews: list[dict[str, Any]] = []
        candidate_validation_errors: dict[int, list[str]] = {}
        validation_errors: list[str] = []
        reasoner_prompt_path: Path | None = None
        reasoner_response_path: Path | None = None
        materializer_prompt_path: Path | None = None
        materializer_response_path: Path | None = None
        if resume_iteration:
            assert iteration_checkpoint is not None
            strategy_label = iteration_checkpoint["strategy_label"]
            diagnosis = iteration_checkpoint["diagnosis"]
            overlay_diff = iteration_checkpoint["overlay_diff"]
            overlays = iteration_checkpoint["overlays"]
            candidate_previews = list(iteration_checkpoint["candidate_previews"])
            validation_errors = list(iteration_checkpoint["validation_errors"])
            candidate_validation_errors = {
                int(preview.get("candidate_id", idx)): list(preview.get("validation_errors", []))
                for idx, preview in enumerate(candidate_previews)
                if preview.get("validation_errors")
            }
            parsed = {
                "strategy_label": strategy_label,
                "diagnosis": diagnosis,
                "overlay_diff": overlay_diff,
                "primary_layer": "resumed",
            }
            print(f"  → Restored {len(overlays)} candidate overlay(s) from {iter_dir / 'diagnosis.json'}")
        elif iteration == 0:
            # Iteration 0: single candidate with default overlay to establish
            # a baseline diagnostic for the reasoner.
            strategy_label = "defaults"
            if reusable_vanilla_sim_dir is not None:
                diagnosis = "Initial baseline iteration reusing one pre-calibration vanilla simulation."
            else:
                diagnosis = "Initial baseline run using default knob values."
            overlay_diff: dict = {}
            overlays = [dict(state.current_best_overlay)]
            if reusable_vanilla_sim_dir is not None:
                print(f"  → 1 candidate (reused pre-calibration vanilla simulation)")
                print(f"    source: {reusable_vanilla_sim_dir}")
            else:
                print(f"  → 1 candidate (default overlay)")
        else:
            # Build prompt — validation is the scoring reference; train remains
            # the qualitative source split for sample-thread context.
            print(f"  → Calling {calibration_model} for strategy...", flush=True)
            if _MANUAL_PHASE_MODE:
                search_root_overlay = dict(state.manual_block_best_overlay or state.current_best_overlay)
                search_root_diagnostic = state.manual_block_best_diagnostic or state.current_best_diagnostic
                search_root_candidate_dir = state.manual_block_best_candidate_dir or state.current_best_candidate_dir
                search_root_mode = f"manual_phase:{phase_context.get('name')}"
                search_root_reason = (
                    "deterministic manual phase-block schedule; "
                    "branch from current block_best built on cumulative committed overlay"
                )
                phase_context["current_focus_metric_rows"] = _manual_phase_metric_rows(
                    search_root_diagnostic or {},
                    list(phase_context.get("focus_metrics", [])),
                )
                phase_context["current_protected_metric_rows"] = _manual_phase_metric_rows(
                    search_root_diagnostic or {},
                    list(phase_context.get("protected_metrics", [])),
                )
                state.current_search_root_overlay = dict(search_root_overlay)
                state.current_search_root_diagnostic = search_root_diagnostic
                state.current_search_root_candidate_dir = search_root_candidate_dir
                state.current_search_root_mode = search_root_mode
                state.current_search_root_reason = search_root_reason
                state.stagnation_count = 0
                state.save()
                print(f"  → Search mode: {search_root_mode}")
                print(f"    reason: {search_root_reason}")
                _print_phase_watch_metrics(
                    phase_context,
                    phase_context.get("current_focus_metric_rows", []),
                    phase_context.get("current_protected_metric_rows", []),
                )
            else:
                state.stagnation_count = _stagnation_count_from_entries(log.entries())
                (
                    search_root_overlay,
                    search_root_diagnostic,
                    search_root_candidate_dir,
                    search_root_mode,
                    search_root_reason,
                ) = _choose_search_root(state)
                state.current_search_root_overlay = dict(search_root_overlay)
                state.current_search_root_diagnostic = search_root_diagnostic
                state.current_search_root_candidate_dir = search_root_candidate_dir
                state.current_search_root_mode = search_root_mode
                state.current_search_root_reason = search_root_reason
                state.save()
                print(
                    f"  → Search mode: {search_root_mode} "
                    f"(stagnation_count={state.stagnation_count})"
                )
                if search_root_reason and search_root_reason != "global_best":
                    print(f"    reason: {search_root_reason}")

            # ── Dedup-only round: skip normal reasoner/materializer ────────
            is_dedup_round = bool(phase_context.get("is_dedup"))
            if is_dedup_round:
                print("  → DEDUP ROUND: deduplicating accumulated overlay text")
                dedup_source_overlay = dict(state.current_best_overlay)
                prompt = build_dedup_prompt(
                    current_overlay=dedup_source_overlay,
                    num_candidates=candidates_per_iter,
                )
                reasoner_prompt_path = iter_dir / "dedup_prompt.txt"
                reasoner_prompt_path.write_text(prompt, encoding="utf-8")

                raw_response = call_reasoner(
                    client,
                    calibration_model,
                    prompt,
                    reasoning_effort=calibration_reasoning_effort,
                    schema_kind=None,
                )
                reasoner_response_path = iter_dir / "dedup_raw_response.json"
                reasoner_response_path.write_text(raw_response, encoding="utf-8")

                # Parse dedup response: expect {candidate_0: {overlay}, ...}
                dedup_parsed = _parse_dedup_response(
                    raw_response, candidates_per_iter, dedup_source_overlay,
                )
                overlays = dedup_parsed["overlays"]
                strategy_label = "dedup_final"
                diagnosis = "Final deduplication round — removing overlay text redundancy."
                overlay_diff: dict = {}
                parsed = {
                    "strategy_label": strategy_label,
                    "diagnosis": diagnosis,
                    "overlay_diff": overlay_diff,
                    "primary_layer": "both",
                }
                candidates_list = [
                    {
                        "strategy_label": f"dedup_variant_{i}",
                        "mechanism_family": "dedup",
                        "primary_layer": "both",
                        "anti_incumbent": False,
                    }
                    for i in range(len(overlays))
                ]
                print(f"  → {len(overlays)} dedup candidate(s) generated")
                for ci, ov in enumerate(overlays):
                    persona_len = len(str(ov.get("persona.generation_guidance", "")))
                    prompt_len = len(str(ov.get("prompt.comment_style_guidance", "")))
                    print(f"      [{ci}] persona={persona_len} chars, prompt={prompt_len} chars")

            else:
                # ── Normal reasoner + materializer flow ──────────────────────

                # Extract sample threads for qualitative context
                _few_shot_dir = Path(reference_run_config.get("few_shot_source", ""))
                _best_cand_dir = (
                    Path(search_root_candidate_dir)
                    if search_root_candidate_dir else None
                )
                sample_real = _extract_sample_real_thread(
                    _few_shot_dir, train_thread_ids,
                ) if _few_shot_dir.exists() else ""
                sample_sim = _extract_sample_sim_thread(
                    _best_cand_dir,
                ) if _best_cand_dir else ""

                prompt_trajectory = (
                    _manual_phase_prompt_trajectory(log.trajectory(), phase_context, iteration)
                    if _MANUAL_PHASE_MODE else log.trajectory()
                )

                prompt = build_reasoner_prompt(
                    registry=registry,
                    current_overlay=search_root_overlay,
                    current_diagnostic=search_root_diagnostic or {},
                    real_baseline=val_summary,
                    trajectory=prompt_trajectory,
                    failed_strategies=log.failed_strategies(),
                    metric_definitions=metric_definitions,
                    sample_real_thread=sample_real,
                    sample_sim_thread=sample_sim,
                    iteration=manual_iteration if _MANUAL_PHASE_MODE else iteration,
                    max_iterations=max_iterations,
                    combination_start_iteration=combination_start_iteration,
                    global_best_overlay=state.current_best_overlay,
                    global_best_diagnostic=state.current_best_diagnostic or {},
                    frontier=_frontier_prompt_summary(state.frontier),
                    stagnation_count=state.stagnation_count,
                    search_mode=search_root_mode,
                    search_root_reason=search_root_reason,
                    phase_context=phase_context if _MANUAL_PHASE_MODE else None,
                    completed_phase_summaries=None,
                )
                reasoner_prompt_path = iter_dir / "reasoner_prompt.txt"
                reasoner_prompt_path.write_text(prompt, encoding="utf-8")

                raw_response = call_reasoner(
                    client,
                    calibration_model,
                    prompt,
                    reasoning_effort=calibration_reasoning_effort,
                    schema_kind="strategist",
                )
                reasoner_response_path = iter_dir / "reasoner_raw_response.json"
                reasoner_response_path.write_text(
                    raw_response,
                    encoding="utf-8",
                )
                parsed = parse_reasoner_response(raw_response)
                diagnosis_preview = " ".join(str(parsed.get("diagnosis", "")).split())
                if diagnosis_preview:
                    print(f"  → Diagnosis: {diagnosis_preview[:300]}")

                # Second-stage text materializer:
                # The strategist chooses what to modify; a second LLM call writes the
                # actual prompt/persona text blocks used at runtime.
                parsed_candidates = parsed.get("candidates", [])
                if _MANUAL_PHASE_MODE and parsed_candidates:
                    required_family = str(phase_context.get("required_mechanism_family", "")).strip()
                    for candidate in parsed_candidates[:5]:
                        candidate["primary_layer"] = "both"
                        if required_family:
                            candidate["mechanism_family"] = required_family
                if parsed_candidates:
                    materializer_prompt = build_text_materializer_prompt(
                        registry=registry,
                        current_overlay=search_root_overlay,
                        current_diagnostic=search_root_diagnostic or {},
                        diagnosis=parsed.get("diagnosis", ""),
                        candidates=parsed_candidates,
                        real_baseline=val_summary,
                        trajectory=prompt_trajectory,
                        failed_strategies=log.failed_strategies(),
                        metric_definitions=metric_definitions,
                        sample_real_thread=sample_real,
                        sample_sim_thread=sample_sim,
                        iteration=manual_iteration if _MANUAL_PHASE_MODE else iteration,
                        max_iterations=max_iterations,
                        combination_start_iteration=combination_start_iteration,
                        global_best_overlay=state.current_best_overlay,
                        global_best_diagnostic=state.current_best_diagnostic or {},
                        frontier=_frontier_prompt_summary(state.frontier),
                        stagnation_count=state.stagnation_count,
                        search_mode=search_root_mode,
                        search_root_reason=search_root_reason,
                        phase_context=phase_context if _MANUAL_PHASE_MODE else None,
                        completed_phase_summaries=None,
                    )
                    materializer_prompt_path = iter_dir / "materializer_prompt.txt"
                    materializer_prompt_path.write_text(materializer_prompt, encoding="utf-8")
                    expected_materialized_candidates = min(5, len(parsed_candidates))
                    raw_materialized = call_reasoner(
                        client,
                        calibration_model,
                        materializer_prompt,
                        reasoning_effort=calibration_reasoning_effort,
                        schema_kind="materializer",
                        response_format_override=materializer_response_format(
                            expected_materialized_candidates
                        ),
                    )
                    materializer_response_path = iter_dir / "materializer_raw_response.json"
                    materializer_response_path.write_text(
                        raw_materialized,
                        encoding="utf-8",
                    )
                    materialized = parse_text_materializer_response(
                        raw_materialized,
                        expected_candidates=expected_materialized_candidates,
                    )
                    for ci, candidate in enumerate(parsed_candidates[:5]):
                        text_diff = materialized.get(ci, {})
                        filtered_text_diff: dict[str, Any] = {}
                        for name, value in text_diff.items():
                            try:
                                if registry.get(name)["type"] == "text":
                                    filtered_text_diff[name] = value
                            except KeyError:
                                continue
                        if filtered_text_diff:
                            candidate["materialized_text_overlay_diff"] = filtered_text_diff
                            candidate["overlay_diff"] = merge_overlay(
                                candidate.get("overlay_diff", {}),
                                filtered_text_diff,
                            )
                            if ci == 0 and isinstance(parsed.get("overlay_diff"), dict):
                                parsed["overlay_diff"] = merge_overlay(
                                    parsed.get("overlay_diff", {}),
                                    filtered_text_diff,
                                )
                if "overlay_diff" in parsed:
                    parsed["overlay_diff"], overlay_errors = _sanitize_overlay(
                        registry,
                        parsed.get("overlay_diff", {}),
                    )
                    candidate_validation_errors.setdefault(0, []).extend(overlay_errors)
                    validation_errors.extend(
                        [f"base overlay_diff: {err}" for err in overlay_errors]
                    )
                for ci, candidate in enumerate(parsed_candidates[:5]):
                    cleaned_diff, candidate_errors = _sanitize_overlay(
                        registry,
                        candidate.get("overlay_diff", {}),
                    )
                    candidate["overlay_diff"] = cleaned_diff
                    if _MANUAL_PHASE_MODE:
                        candidate["primary_layer"] = "both"
                        required_family = str(phase_context.get("required_mechanism_family", "")).strip()
                        if required_family:
                            candidate["mechanism_family"] = required_family
                    if candidate_errors:
                        candidate_validation_errors.setdefault(ci, []).extend(candidate_errors)
                        candidate["validation_errors"] = list(
                            dict.fromkeys(candidate_validation_errors[ci])
                        )
                        validation_errors.extend(
                            [f"candidate_{ci}: {err}" for err in candidate_errors]
                        )

                strategy_label = parsed["strategy_label"]
                diagnosis = parsed["diagnosis"]
                overlay_diff = parsed.get("overlay_diff", {})
                candidates_list = parsed_candidates
                if _MANUAL_PHASE_MODE:
                    parsed["primary_layer"] = "both"

                overlays = generate_variants(
                    current_overlay=search_root_overlay,
                    base_diff=overlay_diff,
                    prompt_alternatives=parsed.get("prompt_alternatives", {}),
                    registry=registry,
                    seed=seed + iteration,
                    conservative_diff=parsed.get("conservative_diff"),
                    parsed_candidates=candidates_list,
                    append_text_mode=_MANUAL_PHASE_MODE,
                    structured_phase_name=str(phase_context.get("name", "")).strip() if _MANUAL_PHASE_MODE else None,
                    structured_phase_label=str(phase_context.get("label", "")).strip() if _MANUAL_PHASE_MODE else None,
                    structured_phase_order=int(phase_context.get("phase_index", 0)) if _MANUAL_PHASE_MODE else None,
                )
                if len(candidates_list) >= 5:
                    print(f"  → 5 independent strategies from LLM:")
                    for ci, c in enumerate(candidates_list[:5]):
                        print(f"      [{ci}] {c.get('strategy_label','?')} ({c.get('primary_layer','?')})")
                else:
                    print(f"  → Strategy: {strategy_label}")
                print(f"  → {len(overlays)} candidate(s) generated")

        sanitized_overlays: list[dict[str, Any]] = []
        for ci, overlay in enumerate(overlays):
            cleaned_overlay, overlay_errors = _sanitize_overlay(registry, overlay)
            sanitized_overlays.append(cleaned_overlay)
            if overlay_errors:
                candidate_validation_errors.setdefault(ci, []).extend(overlay_errors)
            validation_errors.extend(
                [f"candidate_{ci} merged overlay: {err}" for err in overlay_errors]
            )
        overlays = sanitized_overlays
        candidate_validation_errors = {
            ci: list(dict.fromkeys(errors))
            for ci, errors in candidate_validation_errors.items()
        }
        validation_errors = list(dict.fromkeys(validation_errors))
        if validation_errors:
            print("  [overlay validation] Dropped invalid candidate overlay entries:")
            for err in validation_errors:
                print(f"    - {err}")

        print("  → Candidate guidance changes:")
        if candidate_previews:
            for preview in candidate_previews:
                _print_candidate_change_preview(
                    candidate_id=preview.get("candidate_id", 0),
                    strategy_label=preview.get("strategy_label", "strategy"),
                    primary_layer=preview.get("primary_layer", "both"),
                    strategy=preview.get("strategy", ""),
                    rationale=preview.get("rationale", ""),
                    changes=preview.get("effective_changes", []),
                    validation_errors=preview.get("validation_errors"),
                )
        elif candidates_list and len(candidates_list) >= 5:
            for ci, cand in enumerate(candidates_list[:5]):
                effective_changes = _overlay_change_records(
                    registry,
                    search_root_overlay,
                    overlays[ci],
                )
                preview = {
                    "candidate_id": ci,
                    "strategy_label": cand.get("strategy_label", f"candidate_{ci}"),
                    "strategy": cand.get("strategy", ""),
                    "primary_layer": cand.get("primary_layer", "both"),
                    "mechanism_family": cand.get("mechanism_family", "mixed"),
                    "anti_incumbent": bool(cand.get("anti_incumbent", False)),
                    "rationale": cand.get("rationale", ""),
                    "overlay_diff": cand.get("overlay_diff", {}),
                    "materialized_text_overlay_diff": cand.get("materialized_text_overlay_diff", {}),
                    "effective_changes": effective_changes,
                }
                if candidate_validation_errors.get(ci):
                    preview["validation_errors"] = candidate_validation_errors[ci]
                candidate_previews.append(preview)
                _print_candidate_change_preview(
                    candidate_id=ci,
                    strategy_label=preview["strategy_label"],
                    primary_layer=preview["primary_layer"],
                    strategy=preview["strategy"],
                    rationale=preview["rationale"],
                    changes=effective_changes,
                    validation_errors=preview.get("validation_errors"),
                )
        else:
            base_overlay = state.current_best_overlay
            if iteration > 0:
                base_overlay = state.current_search_root_overlay or state.current_best_overlay
            overlay = overlays[0] if overlays else dict(base_overlay)
            effective_changes = _overlay_change_records(registry, base_overlay, overlay)
            preview = {
                "candidate_id": 0,
                "strategy_label": strategy_label,
                "strategy": parsed.get("strategy", diagnosis),
                "primary_layer": parsed.get("primary_layer", "both"),
                "mechanism_family": parsed.get("mechanism_family", "mixed"),
                "anti_incumbent": bool(parsed.get("anti_incumbent", False)),
                "rationale": parsed.get("rationale", ""),
                "overlay_diff": overlay_diff,
                "materialized_text_overlay_diff": parsed.get("materialized_text_overlay_diff", {}),
                "effective_changes": effective_changes,
            }
            if candidate_validation_errors.get(0):
                preview["validation_errors"] = candidate_validation_errors[0]
            candidate_previews.append(preview)
            _print_candidate_change_preview(
                candidate_id=0,
                strategy_label=preview["strategy_label"],
                primary_layer=preview["primary_layer"],
                strategy=preview["strategy"],
                rationale=preview["rationale"],
                changes=effective_changes,
                validation_errors=preview.get("validation_errors"),
            )

        # Save diagnosis
        diag_payload: dict = {
            "iteration": iteration,
            "strategy_label": strategy_label,
            "diagnosis": diagnosis,
        }
        if manual_block_active:
            diag_payload["manual_phase_context"] = dict(phase_context)
            diag_payload["watch_metrics"] = {
                "focus_metric_rows": _serialize_metric_rows(
                    list(phase_context.get("current_focus_metric_rows", []))
                ),
                "protected_metric_rows": _serialize_metric_rows(
                    list(phase_context.get("current_protected_metric_rows", []))
                ),
            }
        diag_payload["artifacts"] = {
            "reasoner_prompt": str(reasoner_prompt_path) if reasoner_prompt_path else None,
            "reasoner_response": str(reasoner_response_path) if reasoner_response_path else None,
            "materializer_prompt": str(materializer_prompt_path) if materializer_prompt_path else None,
            "materializer_response": str(materializer_response_path) if materializer_response_path else None,
        }
        diag_payload["candidates"] = candidate_previews
        if not (candidates_list and len(candidates_list) >= 5):
            diag_payload["overlay_diff"] = overlay_diff
        if validation_errors:
            diag_payload["validation_errors"] = validation_errors
        (iter_dir / "diagnosis.json").write_text(
            json.dumps(diag_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # -------------------------------------------------------------------
        # Run candidates (different overlays share the same seed)
        # -------------------------------------------------------------------
        if iteration == 0 and reusable_vanilla_sim_dir is not None:
            print("  → Reusing 1 precomputed vanilla simulation...", flush=True)
            candidate_results = [
                _make_reused_baseline_candidate_result(
                    iter_dir=iter_dir,
                    overlay=overlays[0],
                    source_sim_dir=reusable_vanilla_sim_dir,
                )
            ]
        else:
            # Iteration 0 uses full rounds (baseline); iterations 1+ use
            # phase1_run_config which may have reduced rounds for speed.
            active_run_config = reference_run_config if iteration == 0 else phase1_run_config
            print(f"  → Running {len(overlays)} simulation(s)...", flush=True)
            candidate_results = run_candidates(
                overlays=overlays,
                iter_dir=iter_dir,
                reference_run_config=active_run_config,
                parallel=len(overlays),
                python=python,
                repo_root=repo_root,
                device=device,
                metric_parallel=metric_parallel,
                simulation_reasoning_effort=simulation_reasoning_effort,
                reuse_existing=resume_iteration,
            )
        succeeded = sum(1 for r in candidate_results if r["success"])
        print(f"  → {succeeded}/{len(overlays)} simulation(s) succeeded")

        # -------------------------------------------------------------------
        # Score candidates against VALIDATION baseline (per-thread empirical p)
        # -------------------------------------------------------------------
        print(f"  → Scoring candidates against val baseline...", flush=True)
        scored: list[dict] = []
        preview_by_id = {
            int(preview.get("candidate_id", idx)): preview
            for idx, preview in enumerate(candidate_previews)
        }
        manual_reference_payload = _manual_block_reference(state) if manual_block_active else None
        for result in candidate_results:
            if not result["success"] or result["sim_dir"] is None:
                continue
            sim_dir = Path(result["sim_dir"])
            try:
                sim_df = load_thread_metrics(sim_dir)
                sc = score_candidate(
                    sim_dir,
                    val_baseline,
                    metrics,
                    ranking_metrics=ranking_metrics,
                )
                group_eval = evaluate_group_vs_real(val_df, sim_df, metrics)
                sc.update(_group_eval_selection_summary(group_eval))
                sc["group_eval_per_metric"] = group_eval.get("per_metric", group_eval)
                sc["candidate_id"] = result["candidate_id"]
                sc["candidate_dir"] = result["candidate_dir"]
                sc["overlay"] = overlays[result["candidate_id"]]
                preview = preview_by_id.get(int(result["candidate_id"]), {})
                sc["strategy_label"] = preview.get("strategy_label", "")
                sc["mechanism_family"] = preview.get("mechanism_family", "mixed")
                sc["primary_layer"] = preview.get("primary_layer", "both")
                sc["anti_incumbent"] = bool(preview.get("anti_incumbent", False))
                if manual_block_active:
                    sc["manual_phase_context"] = dict(phase_context)
                    sc["manual_phase_score"] = _manual_phase_score(sc, phase_context)
                    sc["manual_phase_guard"] = _manual_phase_guard_summary(
                        sc,
                        manual_reference_payload,
                        phase_context,
                    )
                scored.append(sc)
            except Exception:
                pass
        print(f"  → {len(scored)} candidate(s) scored")
        state.frontier = _update_frontier(
            state.frontier,
            scored,
            preview_by_id,
            iteration,
        )

        # -------------------------------------------------------------------
        # Select best candidate
        # -------------------------------------------------------------------
        if manual_block_active and scored:
            winner = min(
                scored,
                key=lambda candidate: _manual_phase_selection_key(candidate, phase_context),
            )
        else:
            winner = select_best_candidate(scored) if scored else None

        # -------------------------------------------------------------------
        # Check if winner beats current best
        # -------------------------------------------------------------------
        beat_current_best = False
        winner_target_eval: dict[str, Any] | None = None
        if winner is not None:
            if manual_block_active:
                prev_payload = _manual_block_reference(state)
                if prev_payload is None:
                    beat_current_best = True
                else:
                    prev = _manual_phase_selection_key(prev_payload, phase_context)
                    new = _manual_phase_selection_key(winner, phase_context)
                    if new < prev:
                        beat_current_best = True
            elif state.current_best_score is None:
                beat_current_best = True
            else:
                prev_payload = _current_best_selection_reference(state)
                prev = _candidate_selection_key(prev_payload or state.current_best_score)
                new = _candidate_selection_key(winner)
                if new < prev:
                    beat_current_best = True

            if beat_current_best:
                winner_score_payload = {
                    "fail_rate": winner["fail_rate"],
                    "mean_abs_delta": winner["mean_abs_delta"],
                    "ranking_fail_rate": winner.get("ranking_fail_rate"),
                    "ranking_mean_abs_delta": winner.get("ranking_mean_abs_delta"),
                    "quantile_fail_rate": winner.get("quantile_fail_rate"),
                    "mean_percentile_distance": winner.get("mean_percentile_distance"),
                    "mean_abs_robust_z": winner.get("mean_abs_robust_z"),
                    "mean_group_percentile_distance": winner.get("mean_group_percentile_distance"),
                    "group_scores": winner.get("group_scores"),
                    "selection_family_scores": winner.get("selection_family_scores"),
                    "group_mean_abs_cliffs_delta": winner.get("group_mean_abs_cliffs_delta"),
                    "group_overall_fail_rate": winner.get("group_overall_fail_rate"),
                    "group_metrics_sig_different": winner.get("group_metrics_sig_different"),
                    "manual_phase_context": winner.get("manual_phase_context"),
                    "manual_phase_score": winner.get("manual_phase_score"),
                }
                winner_diagnostic_payload = {
                    k: v for k, v in winner.items()
                    if k not in ("candidate_id", "candidate_dir", "overlay")
                }
                if manual_block_active:
                    state.manual_block_best_overlay = winner.get("overlay", {})
                    state.manual_block_best_score = winner_score_payload
                    state.manual_block_best_diagnostic = winner_diagnostic_payload
                    state.manual_block_best_candidate_dir = winner.get("candidate_dir")
                    state.current_search_root_overlay = dict(state.manual_block_best_overlay)
                    state.current_search_root_diagnostic = state.manual_block_best_diagnostic
                    state.current_search_root_candidate_dir = state.manual_block_best_candidate_dir
                    state.current_search_root_mode = f"manual_phase:{phase_context.get('name')}"
                    state.current_search_root_reason = "winner became current block_best incumbent"
                else:
                    state.current_best_overlay = winner.get("overlay", {})
                    state.current_best_score = winner_score_payload
                    state.current_best_diagnostic = winner_diagnostic_payload
                    state.current_best_candidate_dir = winner.get("candidate_dir")
                    state.current_search_root_overlay = dict(state.current_best_overlay)
                    state.current_search_root_diagnostic = state.current_best_diagnostic
                    state.current_search_root_candidate_dir = state.current_best_candidate_dir
                    state.current_search_root_mode = "global_best"
                    state.current_search_root_reason = "global_best"
                state.stagnation_count = 0
            else:
                if manual_block_active:
                    state.stagnation_count = 0
                    state.current_search_root_overlay = dict(
                        state.manual_block_best_overlay or state.current_best_overlay
                    )
                    state.current_search_root_diagnostic = (
                        state.manual_block_best_diagnostic or state.current_best_diagnostic
                    )
                    state.current_search_root_candidate_dir = (
                        state.manual_block_best_candidate_dir or state.current_best_candidate_dir
                    )
                    state.current_search_root_mode = f"manual_phase:{phase_context.get('name')}"
                    state.current_search_root_reason = "no update; keep current block_best incumbent"
                else:
                    state.stagnation_count = _stagnation_count_from_entries(log.entries()) + 1
                    (
                        state.current_search_root_overlay,
                        state.current_search_root_diagnostic,
                        state.current_search_root_candidate_dir,
                        state.current_search_root_mode,
                        state.current_search_root_reason,
                    ) = _choose_search_root(state)

        if winner:
            winner_target_eval = _target_metric_eval_summary(winner)
            (iter_dir / "winner_target_metric_eval.json").write_text(
                json.dumps(winner_target_eval, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if manual_block_active:
                improved_str = " ✓ new block best" if beat_current_best else ""
            else:
                improved_str = " ✓ new best" if beat_current_best else ""
            if manual_block_active:
                phase_score = winner.get("manual_phase_score") or _manual_phase_score(winner, phase_context)
                focus_rows = phase_score.get("focus_metric_rows", [])
                print(
                    f"  → Winner: candidate_{winner['candidate_id']}  "
                    f"focus_metrics={phase_context.get('focus_metrics', [])}{improved_str}"
                )
                for metric_row in focus_rows:
                    print(
                        "    "
                        f"{metric_row['metric']}: "
                        f"W={_fmt(metric_row['wasserstein'], '.4f')}  "
                        f"Q={_fmt(metric_row['quantile_error'], '.4f')}  "
                        f"fail={_fmt(metric_row['empirical_fail_rate'], '.4f')}  "
                        f"|med|={_fmt(metric_row['abs_median_gap'], '.4f')}  "
                        f"|cd|={_fmt(metric_row['abs_cliffs_delta'], '.4f')}  "
                        f"mwu_p={_fmt(metric_row['mwu_p_value'], '.4f')}  "
                        f"ks_p={_fmt(metric_row['ks_p_value'], '.4f')}  "
                        f"oor={metric_row['out_of_range']}  "
                        f"pct={_fmt(metric_row['percentile_distance'], '.4f')}  "
                        f"raw_z={_fmt(metric_row['abs_raw_robust_z'], '.4f')}"
                    )
                print(
                    "    target-12 summary: "
                    f"MWU>0.05 {winner_target_eval['mwu_pass_count']}/{len(winner_target_eval['metrics'])}  "
                    f"KS>0.05 {winner_target_eval['ks_pass_count']}/{len(winner_target_eval['metrics'])}  "
                    f"meanW={_fmt(winner_target_eval['mean_wasserstein'], '.4f')}  "
                    f"meanQ={_fmt(winner_target_eval['mean_quantile_error'], '.4f')}  "
                    f"mean|cd|={_fmt(winner_target_eval['mean_abs_cliffs_delta'], '.4f')}  "
                    f"meanFail={_fmt(winner_target_eval['mean_empirical_fail_rate'], '.4f')}"
                )
                _print_manual_phase_selection_ranking(scored, phase_context)
            elif winner.get("quantile_fail_rate") is not None:
                print(
                    f"  → Winner: candidate_{winner['candidate_id']}  "
                    f"quantile_fail={winner['quantile_fail_rate']:.4f}  "
                    f"pct_dist={winner['mean_percentile_distance']:.4f}  "
                    f"robust_z={winner['mean_abs_robust_z']:.4f}{improved_str}"
                )
                print(
                    f"    legacy fail_rate={winner['fail_rate']:.4f}  "
                    f"|delta|={winner['mean_abs_delta']:.4f}"
                )
                _print_winner_selection_breakdown(winner)
                _print_selection_ranking(scored)
            else:
                print(
                    f"  → Winner: candidate_{winner['candidate_id']}  "
                    f"fail_rate={winner['fail_rate']:.4f}  "
                    f"|delta|={winner['mean_abs_delta']:.4f}{improved_str}"
                )
            if winner.get("group_mean_abs_cliffs_delta") is not None:
                print(
                    f"    group |delta|={winner['group_mean_abs_cliffs_delta']:.4f}  "
                    f"group fail={winner['group_overall_fail_rate']:.4f}"
                )
            _print_candidate_score_summary(winner)

        # -------------------------------------------------------------------
        # Log entry
        # -------------------------------------------------------------------
        def _slim(c: dict) -> dict:
            slim = {k: v for k, v in c.items() if k != "per_metric"}
            slim["per_metric_summary"] = {
                m: {sk: sv for sk, sv in md.items() if sk != "threads"}
                for m, md in c.get("per_metric", {}).items()
            }
            return slim

        if winner_target_eval is not None:
            diag_payload["winner_target_metric_eval"] = winner_target_eval
        if winner and winner.get("manual_phase_guard") is not None:
            diag_payload["winner_manual_phase_guard"] = winner.get("manual_phase_guard", {})
        (iter_dir / "diagnosis.json").write_text(
            json.dumps(diag_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Build per-candidate strategy metadata for trajectory
        candidate_strategies: list[dict] = []
        for preview in candidate_previews:
            cs_entry: dict = {
                "candidate_id": preview.get("candidate_id", 0),
                "strategy_label": preview.get("strategy_label", "strategy"),
                "strategy": preview.get("strategy", ""),
                "primary_layer": preview.get("primary_layer", "both"),
                "mechanism_family": preview.get("mechanism_family", "mixed"),
                "anti_incumbent": bool(preview.get("anti_incumbent", False)),
                "overlay_diff": preview.get("overlay_diff", {}),
                "rationale": preview.get("rationale", ""),
                "effective_changes": preview.get("effective_changes", []),
            }
            if preview.get("materialized_text_overlay_diff"):
                cs_entry["materialized_text_overlay_diff"] = preview["materialized_text_overlay_diff"]
            matched = [
                s for s in scored
                if s.get("candidate_id") == cs_entry["candidate_id"]
            ]
            if matched:
                cs_entry["fail_rate"] = matched[0]["fail_rate"]
                cs_entry["mean_abs_delta"] = matched[0]["mean_abs_delta"]
                cs_entry["ranking_fail_rate"] = matched[0].get("ranking_fail_rate")
                cs_entry["ranking_mean_abs_delta"] = matched[0].get("ranking_mean_abs_delta")
                cs_entry["quantile_fail_rate"] = matched[0].get("quantile_fail_rate")
                cs_entry["mean_percentile_distance"] = matched[0].get("mean_percentile_distance")
                cs_entry["mean_abs_robust_z"] = matched[0].get("mean_abs_robust_z")
                cs_entry["group_mean_abs_cliffs_delta"] = matched[0].get("group_mean_abs_cliffs_delta")
                cs_entry["group_overall_fail_rate"] = matched[0].get("group_overall_fail_rate")
                cs_entry["group_scores"] = matched[0].get("group_scores", {})
                cs_entry["selection_family_scores"] = matched[0].get("selection_family_scores", {})
                if matched[0].get("manual_phase_score") is not None:
                    cs_entry["manual_phase_score"] = matched[0].get("manual_phase_score")
                if matched[0].get("manual_phase_guard") is not None:
                    cs_entry["manual_phase_guard"] = matched[0].get("manual_phase_guard")
                cs_entry["target_metric_eval"] = _target_metric_eval_summary(matched[0])
                # Store headline metric values for trajectory visibility
                m_pm = matched[0].get("per_metric", {})
                headline_vals: dict[str, dict] = {}
                for hkey, _hlabel in _HEADLINE_METRICS:
                    hinfo = m_pm.get(hkey, {})
                    if not hinfo:
                        continue
                    headline_vals[hkey] = {
                        "sim_median": hinfo.get("sim_median"),
                        "real_median": hinfo.get("real_median"),
                        "percentile_distance": hinfo.get("percentile_distance"),
                        "robust_z": hinfo.get("robust_z"),
                        "status": hinfo.get("status"),
                    }
                if headline_vals:
                    cs_entry["headline_metrics"] = headline_vals
            if preview.get("validation_errors"):
                cs_entry["validation_errors"] = preview["validation_errors"]
            candidate_strategies.append(cs_entry)

        log.upsert_iteration({
            "iteration": iteration,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "strategy_label": strategy_label,
            "primary_layer": parsed.get("primary_layer", ""),
            "diagnosis": diagnosis,
            "overlay_diff": overlay_diff,
            "artifacts": {
                "iteration_dir": str(iter_dir),
                "diagnosis_path": str(iter_dir / "diagnosis.json"),
                "reasoner_prompt": str(reasoner_prompt_path) if reasoner_prompt_path else None,
                "reasoner_response": str(reasoner_response_path) if reasoner_response_path else None,
                "materializer_prompt": str(materializer_prompt_path) if materializer_prompt_path else None,
                "materializer_response": str(materializer_response_path) if materializer_response_path else None,
            },
            "validation_errors": validation_errors,
            "candidate_strategies": candidate_strategies,
            "candidates": [_slim(c) for c in scored],
            "selection": {
                "winner_candidate_id": winner["candidate_id"] if winner else None,
                "beat_current_best": beat_current_best,
                "best_fail_rate": winner["fail_rate"] if winner else None,
                "best_mean_abs_delta": winner["mean_abs_delta"] if winner else None,
                "best_ranking_fail_rate": winner.get("ranking_fail_rate") if winner else None,
                "best_ranking_mean_abs_delta": winner.get("ranking_mean_abs_delta") if winner else None,
                "best_quantile_fail_rate": winner.get("quantile_fail_rate") if winner else None,
                "best_mean_percentile_distance": (
                    winner.get("mean_percentile_distance") if winner else None
                ),
                "best_mean_abs_robust_z": winner.get("mean_abs_robust_z") if winner else None,
                "best_group_mean_abs_cliffs_delta": (
                    winner.get("group_mean_abs_cliffs_delta") if winner else None
                ),
                "best_group_overall_fail_rate": (
                    winner.get("group_overall_fail_rate") if winner else None
                ),
                "winner_selection_family_scores": (
                    winner.get("selection_family_scores", {}) if winner else {}
                ),
                "winner_manual_phase_score": (
                    winner.get("manual_phase_score", {}) if winner else {}
                ),
                "winner_manual_phase_guard": (
                    winner.get("manual_phase_guard", {}) if winner else {}
                ),
                "winner_target_metric_eval": winner_target_eval or {},
                "selection_ranking": (
                    _manual_phase_ranking_rows(scored, phase_context)
                    if _MANUAL_PHASE_MODE else _selection_ranking_rows(scored)
                ),
            },
            "search_state": {
                "stagnation_count": state.stagnation_count,
                "search_root_mode": state.current_search_root_mode,
                "search_root_reason": state.current_search_root_reason,
                "search_root_candidate_dir": state.current_search_root_candidate_dir,
                "manual_phase_context": phase_context if _MANUAL_PHASE_MODE else {},
                "watch_metrics": (
                    {
                        "focus_metric_rows": _serialize_metric_rows(
                            list(phase_context.get("current_focus_metric_rows", []))
                        ),
                        "protected_metric_rows": _serialize_metric_rows(
                            list(phase_context.get("current_protected_metric_rows", []))
                        ),
                    }
                    if manual_block_active else {}
                ),
            },
        })

        state.completed_iterations = iteration + 1
        state.save()

    if _MANUAL_PHASE_MODE and state.completed_iterations > 1:
        final_phase = _manual_phase_context(state.completed_iterations - 2)
        _manual_commit_block_best(state, final_phase)
        state.save()

    if stop_after_phase1:
        summary = {
            "best_overlay": state.current_best_overlay,
            "best_score": state.current_best_score,
            "completed_iterations": _phase1_reported_iteration_count(state.completed_iterations),
            "output_dir": str(output_dir),
            "after_calibration_evaluation": {},
            "improvement": None,
            "stopped_after_phase1": True,
        }
        (output_dir / "calibration_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n{'='*60}")
        print("STOPPED AFTER PHASE 1")
        print(f"{'='*60}")
        print("  Phase 2 and Phase 3 were skipped by request.")
        return summary

    # -----------------------------------------------------------------------
    # Export best overlay
    # -----------------------------------------------------------------------
    save_overlay(state.current_best_overlay, output_dir / "best_overlay.json")

    # -----------------------------------------------------------------------
    # Phase 2: After-calibration group evaluation (calibrated vs real_test)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 2: After-calibration group evaluation")
    print(f"{'='*60}")
    _after_eval_path = output_dir / "after_calibration" / "after_calibration_evaluation.json"
    ran_new_iterations = state.completed_iterations > starting_completed_iterations
    if _after_eval_path.exists() and not ran_new_iterations:
        print(f"  Skipped — already done (found {_after_eval_path})")
        after_eval = json.loads(_after_eval_path.read_text(encoding="utf-8"))
        # reload group_eval too
        _group_eval_path = output_dir / "after_calibration" / "after_calibration_group_eval.json"
        if _group_eval_path.exists():
            after_eval["group_eval"] = json.loads(_group_eval_path.read_text(encoding="utf-8"))
    else:
        print(f"  Running {final_sim_runs} fresh simulation(s) with best overlay "
              f"(stop at {min_sim_threads} threads)...", flush=True)
        after_eval = _run_after_calibration_evaluation(
            output_dir=output_dir,
            best_overlay=state.current_best_overlay,
            real_test_csv=real_test_csv,
            reference_run_config=reference_run_config,
            sim_runs=final_sim_runs,
            metrics=metrics,
            python=python,
            repo_root=repo_root,
            device=device,
            min_sim_threads=min_sim_threads,
            metric_parallel=metric_parallel,
            simulation_reasoning_effort=simulation_reasoning_effort,
        )

    if after_eval.get("group_eval"):
        print("  Calibrated vs real_test (key metrics):")
        _print_group_eval_summary(after_eval["group_eval"])

    # -----------------------------------------------------------------------
    # Phase 3: Improvement analysis (before vs after)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 3: Improvement analysis")
    print(f"{'='*60}")
    improvement: dict | None = None
    if before_eval is not None and after_eval.get("group_eval") is not None:
        improvement = compare_before_after(
            before_eval,
            after_eval["group_eval"],
            real_df=real_test_df,
            before_df=before_generated_df,
            after_df=after_eval.get("_all_sim_df"),
            metrics=metrics,
        )
        (output_dir / "before_after_improvement_summary.json").write_text(
            json.dumps(improvement, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _print_improvement_summary(improvement)
        _print_improvement_table(improvement)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    summary = {
        "best_overlay": state.current_best_overlay,
        "best_score": state.current_best_score,
        "completed_iterations": _phase1_reported_iteration_count(state.completed_iterations),
        "output_dir": str(output_dir),
        "after_calibration_evaluation": {
            k: v for k, v in after_eval.items() if k not in {"group_eval", "_all_sim_df"}
        },
        "improvement": improvement,
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


# ---------------------------------------------------------------------------
# Phase 0: Before-calibration group evaluation
# ---------------------------------------------------------------------------

def _run_before_calibration_evaluation(
    output_dir: Path,
    real_test_csv: Path,
    metrics: list[str],
    vanilla_scores_csv: Path | None = None,
    reference_run_config: dict[str, Any] | None = None,
    sim_runs: int = 12,
    python: str = sys.executable,
    repo_root: Path | None = None,
    device: str = "cpu",
    min_sim_threads: int = 0,
    metric_parallel: int = 2,
    simulation_reasoning_effort: str | None = None,
) -> tuple[dict[str, dict], pd.DataFrame, Path | None]:
    """Evaluate a before-calibration vanilla baseline against real_test.

    If ``reference_run_config`` is provided, run fresh vanilla simulations with the
    current config. Otherwise load the precomputed ``vanilla_scores_csv``.
    """
    print(f"\n{'='*60}")
    print("PHASE 0: Before-calibration group evaluation (vanilla vs real_test)")
    print(f"{'='*60}")

    real_test_df = pd.read_csv(real_test_csv)
    reused_sim_dir: Path | None = None
    target_threads = _resolve_eval_thread_target(len(real_test_df), min_sim_threads)

    if reference_run_config is not None:
        if repo_root is None:
            repo_root = Path(__file__).parent.parent

        before_dir = output_dir / "before_calibration"
        before_dir.mkdir(parents=True, exist_ok=True)
        overlays = [{} for _ in range(sim_runs)]
        seed_offsets = list(range(sim_runs))

        print(
            f"  Running {sim_runs} fresh vanilla simulation(s) "
            f"(target {target_threads if target_threads > 0 else 'unbounded'} threads)...",
            flush=True,
        )
        results = run_candidates(
            overlays=overlays,
            iter_dir=before_dir,
            reference_run_config=reference_run_config,
            python=python,
            repo_root=repo_root,
            device=device,
            seed_offsets=seed_offsets,
            min_threads=target_threads,
            metric_parallel=metric_parallel,
            batch_schedule=[sim_runs],
            simulation_reasoning_effort=simulation_reasoning_effort,
        )

        frames: list[pd.DataFrame] = []
        for result in results:
            if not result["success"] or result["sim_dir"] is None:
                continue
            sim_dir = Path(result["sim_dir"])
            if reused_sim_dir is None:
                reused_sim_dir = sim_dir
            try:
                df = load_thread_metrics(sim_dir)
                df["_run_id"] = result["candidate_id"]
                frames.append(df)
            except Exception:
                pass

        if not frames:
            raise RuntimeError("No successful vanilla baseline simulations for Phase 0.")

        vanilla_df = pd.concat(frames, ignore_index=True)
        if target_threads > 0 and len(vanilla_df) > target_threads:
            vanilla_df = vanilla_df.iloc[:target_threads].copy()
        vanilla_df.to_csv(output_dir / "before_calibration_generated_scores.csv", index=False)
        if reused_sim_dir is not None:
            (output_dir / "before_calibration_reused_sim_dir.txt").write_text(
                str(reused_sim_dir),
                encoding="utf-8",
            )
    else:
        if vanilla_scores_csv is None:
            raise ValueError("Either vanilla_scores_csv or reference_run_config must be provided.")
        vanilla_df = pd.read_csv(vanilla_scores_csv)

    print(f"  real_test threads: {len(real_test_df)}")
    print(f"  vanilla threads:   {len(vanilla_df)}")

    before_eval = evaluate_group_vs_real(real_test_df, vanilla_df, metrics)

    (output_dir / "before_calibration_group_eval.json").write_text(
        json.dumps(before_eval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  → Saved before_calibration_group_eval.json")
    return before_eval, vanilla_df, reused_sim_dir


def _force_vanilla_backbone(
    reference_run_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a copy of ``reference_run_config`` that always uses vanilla OASIS."""

    if reference_run_config is None:
        return None
    forced = dict(reference_run_config)
    forced["discussion_backbone"] = "vanilla_oasis"
    return forced


# ---------------------------------------------------------------------------
# Phase 2: After-calibration group evaluation
# ---------------------------------------------------------------------------

def _run_after_calibration_evaluation(
    output_dir: Path,
    best_overlay: dict,
    real_test_csv: Path,
    reference_run_config: dict,
    sim_runs: int,
    metrics: list[str],
    python: str,
    repo_root: Path,
    device: str = "cpu",
    min_sim_threads: int = 0,
    metric_parallel: int = 2,
    simulation_reasoning_effort: str | None = None,
) -> dict:
    """Generate fresh simulations with best_overlay, evaluate against real_test.

    Each run uses a different seed (base_seed + i) since the overlay is the
    same across all runs.

    Returns a dict with:
      - ``group_eval``: output of evaluate_group_vs_real (for improvement analysis)
      - ``fail_rate``, ``mean_abs_delta``: aggregated per-thread diagnostics
      - ``sim_runs``, ``total_threads``: counts
    """
    final_dir = output_dir / "after_calibration"
    final_dir.mkdir(parents=True, exist_ok=True)
    real_test_df = pd.read_csv(real_test_csv)
    target_threads = _resolve_eval_thread_target(len(real_test_df), min_sim_threads)

    # Save the overlay used
    save_overlay(best_overlay, final_dir / "overlay.json")

    print(f"\n{'='*60}")
    print(f"PHASE 2: After-calibration evaluation ({sim_runs} fresh sims with best overlay)")
    print(f"{'='*60}")
    print(
        f"  target threads: {target_threads if target_threads > 0 else 'unbounded'}",
        flush=True,
    )

    # Each run gets a different seed (same overlay → must differ by seed)
    overlays = [dict(best_overlay) for _ in range(sim_runs)]
    seed_offsets = list(range(sim_runs))

    results = run_candidates(
        overlays=overlays,
        iter_dir=final_dir,
        reference_run_config=reference_run_config,
        python=python,
        repo_root=repo_root,
        device=device,
        seed_offsets=seed_offsets,
        min_threads=target_threads,
        metric_parallel=metric_parallel,
        batch_schedule=[4, 3, 2] + [1] * max(0, sim_runs - 9),
        simulation_reasoning_effort=simulation_reasoning_effort,
    )

    # Collect all thread metrics from successful runs into one DataFrame
    frames: list[pd.DataFrame] = []
    for result in results:
        if not result["success"] or result["sim_dir"] is None:
            continue
        sim_dir = Path(result["sim_dir"])
        try:
            df = load_thread_metrics(sim_dir)
            df["_run_id"] = result["candidate_id"]
            frames.append(df)
        except Exception:
            pass

    if not frames:
        print("[WARN] No successful after-calibration simulations.")
        return {
            "group_eval": None,
            "_all_sim_df": None,
            "fail_rate": None,
            "mean_abs_delta": None,
            "sim_runs": 0,
            "total_threads": 0,
            "target_threads": target_threads,
        }

    all_sim_df = pd.concat(frames, ignore_index=True)
    if target_threads > 0 and len(all_sim_df) > target_threads:
        all_sim_df = all_sim_df.iloc[:target_threads].copy()

    print(f"  calibrated threads: {len(all_sim_df)}")
    print(f"  real_test threads:  {len(real_test_df)}")

    # Group-level evaluation: MWU, KS, Cliff's delta per metric
    group_eval = evaluate_group_vs_real(real_test_df, all_sim_df, metrics)

    (final_dir / "after_calibration_group_eval.json").write_text(
        json.dumps(group_eval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Also compute aggregate per-thread empirical diagnostics
    test_baseline = compute_baseline_from_csv(real_test_csv, metrics)
    all_fail_rates: list[float] = []
    all_abs_deltas: list[float] = []
    for result in results:
        if not result["success"] or result["sim_dir"] is None:
            continue
        sim_dir = Path(result["sim_dir"])
        try:
            sc = score_candidate(sim_dir, test_baseline, metrics)
            all_fail_rates.append(sc["fail_rate"])
            all_abs_deltas.append(sc["mean_abs_delta"])
        except Exception:
            pass

    after_result = {
        "group_eval": group_eval,
        "_all_sim_df": all_sim_df,
        "fail_rate": float(np.mean(all_fail_rates)) if all_fail_rates else None,
        "mean_abs_delta": float(np.mean(all_abs_deltas)) if all_abs_deltas else None,
        "sim_runs": len(frames),
        "total_threads": len(all_sim_df),
        "target_threads": target_threads,
    }

    (final_dir / "after_calibration_evaluation.json").write_text(
        json.dumps(
            {k: v for k, v in after_result.items() if k not in {"group_eval", "_all_sim_df"}},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"  avg fail rate:    {after_result['fail_rate']:.4f}" if after_result["fail_rate"] else "")
    print(f"  avg |delta|:      {after_result['mean_abs_delta']:.4f}" if after_result["mean_abs_delta"] else "")
    print(f"  successful runs:  {after_result['sim_runs']}/{sim_runs}")
    print(f"  → Saved after_calibration_group_eval.json")

    return after_result


# ---------------------------------------------------------------------------
# Improvement summary printer
# ---------------------------------------------------------------------------

def _print_improvement_summary(improvement: dict) -> None:
    """Print a concise terminal summary of before vs after improvement."""
    s = improvement.get("summary", {})
    pm = improvement.get("per_metric", {})

    def _yn(flag: bool) -> str:
        return "YES" if flag else "no"

    print(f"\n{'='*60}")
    print("IMPROVEMENT ANALYSIS (before vs after calibration)")
    print(f"{'='*60}")
    print(f"  Metrics sig. different before: {s.get('metrics_sig_different_before', '?')}")
    print(f"  Metrics sig. different after:  {s.get('metrics_sig_different_after', '?')}")
    print(f"  Avg |Cliff's delta| before:    {s.get('avg_abs_cliffs_delta_before', 0):.4f}")
    print(f"  Avg |Cliff's delta| after:     {s.get('avg_abs_cliffs_delta_after', 0):.4f}")
    print(f"  Overall fail rate before:      {s.get('overall_fail_rate_before', 0):.4f}")
    print(f"  Overall fail rate after:       {s.get('overall_fail_rate_after', 0):.4f}")
    print(f"  Overall pass rate before:      {s.get('overall_pass_rate_before', 0):.4f}")
    print(f"  Overall pass rate after:       {s.get('overall_pass_rate_after', 0):.4f}")
    print(
        f"  Avg Wasserstein before/after:  "
        f"{s.get('avg_wasserstein_distance_before', float('nan')):.4f} → "
        f"{s.get('avg_wasserstein_distance_after', float('nan')):.4f}"
    )
    print(
        f"  Avg quantile err before/after: "
        f"{s.get('avg_quantile_error_before', float('nan')):.4f} → "
        f"{s.get('avg_quantile_error_after', float('nan')):.4f}"
    )

    metric_count = len(pm)
    print(
        "\n  Strict improved"
        f" (|Cliff's delta|↓ AND fail_rate↓): {s.get('strict_improved_count', 0)}/{metric_count}"
    )
    print(
        "  Closer-to-real by Wasserstein:       "
        f"{s.get('closer_by_wasserstein_count', 0)}/{metric_count}"
    )
    print(
        "  Closer-to-real by quantile error:    "
        f"{s.get('closer_by_quantile_count', 0)}/{metric_count}"
    )
    print(
        "  Closer-to-real by mean gap:          "
        f"{s.get('closer_by_mean_gap_count', 0)}/{metric_count}"
    )
    print(
        "  Closer-to-real by abs median gap:    "
        f"{s.get('closer_by_abs_median_gap_count', 0)}/{metric_count}"
    )

    print(
        "\n  Note:"
        "\n    - 'strict improved' is the conservative pass/fail view used by phase 3."
        "\n    - the 'closer-to-real' counts show directional numeric movement even when strict improved stays false."
    )

    if not pm:
        return

    print(
        f"\n  {'Metric':<28} {'cd_before':>9} {'cd_after':>9} {'Δfail%':>7} "
        f"{'strict':>6} {'wass':>6} {'quant':>6} {'mean':>6}"
    )
    print(
        f"  {'-'*28} {'-'*9} {'-'*9} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6}"
    )
    for key, label in _HEADLINE_METRICS:
        info = pm.get(key)
        if not info:
            continue
        print(
            f"  {label:<28} "
            f"{float(info.get('before_cliffs_delta', float('nan'))):>9.4f} "
            f"{float(info.get('after_cliffs_delta', float('nan'))):>9.4f} "
            f"{float(info.get('fail_rate_reduction', float('nan'))) * 100:>+6.1f}% "
            f"{_yn(bool(info.get('improved'))):>6} "
            f"{_yn(bool(info.get('closer_by_wasserstein'))):>6} "
            f"{_yn(bool(info.get('closer_by_quantile'))):>6} "
            f"{_yn(bool(info.get('closer_by_mean_gap'))):>6}"
        )
