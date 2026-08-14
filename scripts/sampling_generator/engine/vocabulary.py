from __future__ import annotations


PAYLOAD_TYPES = (
    "low_info_reaction",
    "bare_answer",
    "fragment_datapoint",
    "soft_helpful",
    "correction",
    "narrow_question",
    "personal_story",
    "rant",
    "joke",
    "side_tangent",
    "meta_or_template",
    "advice",
)

LOW_INFO_PAYLOAD_TYPES = {
    "low_info_reaction",
    "bare_answer",
    "fragment_datapoint",
    "narrow_question",
    "joke",
    "side_tangent",
    "meta_or_template",
}

TERSE_PAYLOAD_TYPES = {
    "low_info_reaction",
    "bare_answer",
    "joke",
    "meta_or_template",
}

SPEAKER_ROLES = (
    "advisor",
    "confused_asker",
    "op_followup",
    "gratitude_reply",
    "jokester",
    "mod_meta",
    "contrarian",
    "datapoint_only",
    "ranter",
    "side_observer",
)

# The value space `infer_utterance_mode` and `infer_surface_texture` can produce.
# Nothing reads these at run time; they document the field domain and let the
# domain-residue sweep enumerate every writer-prompt variant.
UTTERANCE_MODES = (
    "fragment_only",
    "direct_answer",
    "question_only",
    "one_datapoint",
    "op_followup",
    "joke_only",
    "template_notice",
    "complaint_only",
    "side_tangent",
    "correction_only",
    "local_advice",
)

SURFACE_TEXTURES = (
    "plain",
    "no_punct_fragment",
    "abbrev_shorthand",
    "emoji_or_sarcasm",
    "markdown_quote",
    "link_reference",
    "messy_punctuation",
    "gratitude_social",
)

LOW_INFO_UTTERANCE_MODES = {
    "fragment_only",
    "direct_answer",
    "question_only",
    "joke_only",
    "template_notice",
    "side_tangent",
    "op_followup",
}

CAPPED_OPENER_FAMILIES = {
    "first_person_experience",
    "conditional_advice",
    "uncertainty_preface",
    "helpful_directive",
    "generic_empathy",
    "yeah_that_ack",
    "thanks_that_ack",
    "good_to_know_ack",
}

HARD_REAL_SURFACE_SHAPES = {
    "deleted_removed",
    "template_notice",
    "link_reference",
    "quote_link_reference",
    "micro_reaction",
    "short_direct_answer",
    "short_question",
    "thanks_ack",
    "joke_reaction",
    "side_tangent",
}

def is_hard_real_surface_shape(shape: str) -> bool:
    return bool(shape) and shape in HARD_REAL_SURFACE_SHAPES

CAPPED_TEMPLATE_PHRASE_FAMILIES = {
    "first_person_experience_frame",
    "uncertainty_frame",
    "worth_frame",
    "generic_advice_frame",
    "gpt_part_frame",
    "gpt_basically_frame",
    "gpt_kind_of_frame",
    "gpt_feels_like_frame",
    "gpt_good_to_know_frame",
}

COMMENT_FUNCTIONS = (
    "reaction",
    "question_followup",
    "correction_caveat",
    "personal_datapoint",
    "recommendation_advice",
    "verdict_evaluation",
    "explanation_analysis",
    "offtopic_noise",
)

CONTENT_ANGLES = (
    "cost_value",
    "rules_constraints",
    "risk_reliability_support",
    "comparison_alternative",
    "setup_troubleshooting",
    "availability_timing",
    "fit_use_case",
    "unclear_mixed",
)

CONTEXT_APERTURES = (
    "full_seed",
    "seed_gist_only",
    "title_only",
    "semantic_only",
    "parent_only",
)

EVIDENCE_MODES = (
    "none_assertion",
    "firsthand_experience",
    "technical_or_policy_reasoning",
    "calculation_math",
    "hearsay_consensus",
    "link_quote_reference",
    "small_observation",
)

STORY_MODES = (
    "no_story",
    "tiny_personal_context",
    "specific_personal_story",
    "messy_multi_step_story",
)

VOICE_MODES = (
    "blunt",
    "casual_neutral",
    "polite_soft",
    "sarcastic",
    "annoyed",
    "uncertain",
    "grateful",
)

TONE_SHAPES = (
    "soft_ack",
    "personal_dp",
    "neutral_fact",
    "plain_question",
    "mild_caveat",
    "light_joke",
    "direct_correction",
    "rant",
    "bare_answer",
)

LENGTH_BUCKET_BOUNDS = {
    "micro": (1, 5),
    "short": (6, 10),
    "medium": (11, 45),
    "long": (46, 100),
    "very_long": (120, 220),
}

SYSTEM_PROMPTS = {
    "qwen8_v13": (
        "/no_think Write one realistic Reddit-style comment. "
        "Do not sound like an assistant. Return only the comment."
    ),
    "qwen14_labelaware": (
        "/no_think Write one realistic Reddit-style comment. Do not sound like an "
        "assistant. Follow the controls naturally. Return only the comment."
    ),
    "gpt54_reddit_writer": (
        "You are writing one human Reddit comment in r/CreditCards. "
        "Follow the sampled role, length, tone, and local context exactly, but do not sound "
        "like an assistant, customer support, or a polished explainer. Real Reddit comments "
        "can be partial, thankful, uncertain, neutral, mildly skeptical, messy, or low-information. "
        "Casual Reddit voice does not mean scolding the OP, dunking on another user, or writing a "
        "judge-like verdict. Keep disagreement local: a correction can be plain, but should not "
        "stack extra contempt, loaded insults, or a lecture after the local point. Use small "
        "acknowledgement/report-back when the sampled role asks for polite, grateful, caveated, "
        "or follow-up tone; make that social move visible but informal. For substantive polite-helpful "
        "slots, use a constructive Reddit frame around the same local advice/datapoint instead of a "
        "generic thank-you wrapper. For those social slots, "
        "do not bury the acknowledgement inside a correction or immediately flip it with but/not/don't. Use previous "
        "thread comments only to avoid repeating the same wording, function, or discourse "
        "shape. Do not make comments unrelated just to be diverse. Do not invent case numbers, "
        "exact dates, exact fees, reward values, phone numbers, policy names, or product "
        "details unless they are visible in the prompt. Never output planner labels, skeleton "
        "labels, bracket placeholders, or fake link markers. "
        "Return only the comment."
    ),
    "osim8b": (
        "/no_think Social context: you are an ordinary r/CreditCards Reddit user "
        "writing the next human-side turn in a discussion. Your goal is to sound "
        "like a real commenter with a local reaction, datapoint, question, joke, "
        "or correction. Do not act like a helpful assistant. Return only the comment."
    ),
    "osim8b_minimal_context": (
        "/no_think You are an ordinary Reddit user. Write one short Reddit-style "
        "comment. Do not explain. Return only the comment."
    ),
    "osim8b_qwen_style": (
        "/no_think Write one realistic Reddit-style comment. Do not sound like an "
        "assistant. Return only the comment."
    ),
    "minimal": (
        "/no_think Write one realistic Reddit-style comment. "
        "Do not sound like an assistant. Return only the comment."
    ),
}
