from __future__ import annotations

from sampling_generator.engine.model import BranchPlan
from sampling_generator.engine.model import CommentTask
from sampling_generator.engine.vocabulary import HARD_REAL_SURFACE_SHAPES
from sampling_generator.engine.vocabulary import TERSE_PAYLOAD_TYPES
from sampling_generator.engine.vocabulary import TONE_SHAPES
import math
import re

def length_bucket_for_word_count(word_count: int) -> str:
    if word_count <= 5:
        return "micro"
    if word_count <= 10:
        return "short"
    if word_count <= 45:
        return "medium"
    if word_count <= 100:
        return "long"
    return "very_long"

def length_bucket_for_payload(*, payload_type: str, word_count: int) -> str:
    """Preserve real length for normal comments, but keep noisy payloads noisy.

    The planner can correctly identify a real comment as a bare answer or small
    reaction while the real word count still falls into the medium bucket.  If
    we pass that contradiction to Qwen, it writes a polished explanation.  This
    deterministic override keeps low-information payloads short before the
    prompt is built.
    """

    if payload_type in TERSE_PAYLOAD_TYPES:
        if word_count <= 5:
            return "micro"
        if word_count <= 10:
            return "short"
        return length_bucket_for_word_count(word_count)
    if payload_type == "narrow_question":
        if word_count <= 5:
            return "micro"
        if word_count <= 12:
            return "short"
        return length_bucket_for_word_count(word_count)
    if payload_type == "fragment_datapoint":
        if word_count <= 5:
            return "micro"
        if word_count <= 10:
            return "short"
        return length_bucket_for_word_count(word_count)
    if payload_type == "side_tangent":
        if word_count <= 5:
            return "micro"
        if word_count <= 12:
            return "short"
        return length_bucket_for_word_count(word_count)
    return length_bucket_for_word_count(word_count)

def infer_payload_type(
    *,
    word_count: int,
    comment_function: str,
    reply_relation: str,
    stance: str,
) -> str:
    if word_count <= 5:
        return "bare_answer" if stance in {"agree", "disagree", "neutral"} else "low_info_reaction"
    if word_count <= 10:
        if comment_function == "question_followup" or reply_relation == "asks_narrow_followup":
            return "narrow_question"
        return "fragment_datapoint" if comment_function == "personal_datapoint" else "low_info_reaction"
    if comment_function == "question_followup":
        return "narrow_question"
    if comment_function == "personal_datapoint":
        return "personal_story" if word_count > 45 else "fragment_datapoint"
    if comment_function == "offtopic_noise":
        return "side_tangent"
    if comment_function == "reaction" and stance == "joking":
        return "joke"
    if comment_function == "recommendation_advice":
        return "advice"
    if comment_function == "correction_caveat":
        return "correction"
    if comment_function == "verdict_evaluation":
        return "rant" if stance == "disagree" else "soft_helpful"
    return "soft_helpful"

def voice_for_payload(
    *,
    planned_voice: str,
    payload_type: str,
    comment_function: str,
    branch: BranchPlan,
) -> str:
    if payload_type in {"soft_helpful", "advice"} and planned_voice in {"", "casual_neutral"}:
        return "polite_soft"
    if payload_type == "personal_story" and planned_voice in {"", "casual_neutral", "uncertain"}:
        return "grateful" if "grateful" in branch.tone_palette else "polite_soft"
    if planned_voice and planned_voice != "casual_neutral":
        return planned_voice
    return infer_voice_for_payload(
        payload_type=payload_type,
        comment_function=comment_function,
        branch=branch,
    )

def infer_voice_for_payload(*, payload_type: str, comment_function: str, branch: BranchPlan) -> str:
    if payload_type in {"soft_helpful", "advice"}:
        return "polite_soft"
    if payload_type in {"narrow_question", "fragment_datapoint"}:
        return "uncertain"
    if payload_type in {"rant", "bare_answer"}:
        return "blunt"
    if payload_type == "joke":
        return "sarcastic"
    if payload_type == "personal_story":
        return "grateful" if "grateful" in branch.tone_palette else "uncertain"
    if payload_type == "side_tangent":
        return "annoyed" if "annoyed" in branch.tone_palette else "uncertain"
    if comment_function == "reaction":
        return "blunt"
    return first_choice(branch.tone_palette, "uncertain")

def infer_speaker_role(*, payload_type: str, comment_function: str, voice: str, reply_relation: str) -> str:
    relation = str(reply_relation or "")
    if payload_type == "meta_or_template":
        return "mod_meta"
    if payload_type == "joke":
        return "jokester"
    if payload_type == "rant" or voice == "annoyed":
        return "ranter"
    if payload_type == "side_tangent" or relation == "shifts_to_side_detail":
        return "side_observer"
    if payload_type == "narrow_question" or comment_function == "question_followup" or relation == "asks_narrow_followup":
        return "confused_asker"
    if payload_type == "fragment_datapoint" or comment_function == "personal_datapoint":
        return "datapoint_only"
    if voice == "grateful":
        return "gratitude_reply"
    if relation in {"challenges_parent", "corrects_detail"} or payload_type == "correction":
        return "contrarian"
    return "advisor"

def infer_utterance_mode(
    *,
    payload_type: str,
    speaker_role: str,
    comment_function: str,
    voice: str,
    real_word_count: int,
) -> str:
    # Matched-real length is the surface authority. A Planner may choose a
    # low-information semantic move, but it must not collapse a medium real
    # slot into a micro fragment.
    if real_word_count <= 5:
        return "fragment_only"
    if payload_type == "low_info_reaction" and real_word_count <= 20:
        return "direct_answer"
    if payload_type == "bare_answer":
        return "direct_answer"
    if speaker_role == "mod_meta" or payload_type == "meta_or_template":
        return "template_notice"
    if speaker_role == "jokester" or payload_type == "joke":
        return "joke_only"
    if speaker_role == "confused_asker" or payload_type == "narrow_question" or comment_function == "question_followup":
        return "question_only"
    if speaker_role == "op_followup" or speaker_role == "gratitude_reply":
        return "op_followup"
    if speaker_role == "datapoint_only" or payload_type in {"fragment_datapoint", "personal_story"}:
        return "one_datapoint"
    if speaker_role == "ranter" or payload_type == "rant" or voice == "annoyed":
        return "complaint_only"
    if speaker_role == "side_observer" or payload_type == "side_tangent" or comment_function == "offtopic_noise":
        return "side_tangent"
    if speaker_role == "contrarian" or payload_type == "correction" or comment_function == "correction_caveat":
        return "correction_only"
    return "local_advice"

def infer_surface_texture(
    real_text: str,
    *,
    payload_type: str,
    speaker_role: str,
    utterance_mode: str,
) -> str:
    text = str(real_text or "")
    lowered = text.lower()
    tokens = re.findall(r"[A-Za-z0-9/']+", text)
    if re.search(r"https?://|www\.", text, re.IGNORECASE):
        return "link_reference"
    if ">" in text or "&gt;" in lowered or re.search(r"\[[^\]]+\]\(", text):
        return "markdown_quote"
    if re.search(r"[\U0001F300-\U0001FAFF]", text) or "/s" in lowered:
        return "emoji_or_sarcasm"
    if any(word in lowered for word in ("thank you", "thanks", "appreciate", "good to know", "best of luck")):
        return "gratitude_social"
    if re.search(r"\b(CFU|CFF|CSR|SUB|AF|CLI|HUCA|PC|DP|USBAR|BCP|BCE)\b|5/24|1/12", text):
        return "abbrev_shorthand"
    if re.search(r"\b[A-Z]{2,}\b", text) and len(tokens) <= 30:
        return "abbrev_shorthand"
    if utterance_mode == "fragment_only" or (len(tokens) <= 8 and not re.search(r"[.!?]\s*$", text.strip())):
        return "no_punct_fragment"
    if "..." in text or "…" in text or text.count("!") >= 1:
        return "messy_punctuation"
    if payload_type in {"joke", "side_tangent"} or speaker_role in {"jokester", "side_observer"}:
        return "messy_punctuation"
    return "plain"

PROMPT_VISIBLE_REAL_TONE_SLOTS = {
    "pure_acknowledgement",
    "op_appreciative_followup",
    "thanks_plus_question",
    "acknowledgement_plus_local_detail",
    "mild_local_pushback",
    "process_friction_aside",
    "hard_process_complaint",
}

def real_tone_slot_for_prompt(task: CommentTask) -> tuple[str, str]:
    """Return only high-signal real tone slots for writer prompts.

    Default slots like neutral_local_observation/plain_datapoint are useful
    metadata, but showing them on every comment made v45 too uniform and hurt
    Self-BLEU/semantic on small-sample smoke.  Only surface the slots that
    correct the known tone failure modes.
    """

    slot = str(task.real_tone_slot or "")
    if slot in PROMPT_VISIBLE_REAL_TONE_SLOTS:
        return slot, task.real_tone_instruction
    return "", ""

def overrides_for_real_surface_shape(*, shape: str, word_count: int, has_parent: bool) -> dict[str, str]:
    if shape not in HARD_REAL_SURFACE_SHAPES:
        return {}
    if shape == "deleted_removed":
        return {
            "payload_type": "meta_or_template",
            "comment_function": "offtopic_noise",
            "evidence_mode": "none_assertion",
            "story_mode": "no_story",
            "voice": "casual_neutral",
            "speaker_role": "mod_meta",
            "length_bucket": "micro",
            "utterance_mode": "template_notice",
            "surface_texture": "no_punct_fragment",
        }
    if shape == "template_notice":
        return {
            "payload_type": "meta_or_template",
            "comment_function": "offtopic_noise",
            "evidence_mode": "link_quote_reference",
            "story_mode": "no_story",
            "voice": "casual_neutral",
            "speaker_role": "mod_meta",
            "length_bucket": length_bucket_for_word_count(word_count),
            "utterance_mode": "template_notice",
            "surface_texture": "markdown_quote",
        }
    if shape in {"link_reference", "quote_link_reference"}:
        return {
            "payload_type": "meta_or_template",
            "comment_function": "offtopic_noise",
            "evidence_mode": "link_quote_reference",
            "story_mode": "no_story",
            "voice": "casual_neutral",
            "speaker_role": "side_observer" if has_parent else "mod_meta",
            "length_bucket": length_bucket_for_word_count(word_count),
            "utterance_mode": "template_notice",
            "surface_texture": "link_reference" if shape == "link_reference" else "markdown_quote",
        }
    if shape == "micro_reaction":
        return {
            "payload_type": "low_info_reaction",
            "comment_function": "reaction",
            "evidence_mode": "none_assertion",
            "story_mode": "no_story",
            "voice": "blunt",
            "speaker_role": "side_observer",
            "length_bucket": "micro",
            "utterance_mode": "fragment_only",
            "surface_texture": "no_punct_fragment",
        }
    if shape == "short_direct_answer":
        return {
            "payload_type": "bare_answer",
            "comment_function": "reaction",
            "evidence_mode": "none_assertion",
            "story_mode": "no_story",
            "voice": "blunt",
            "speaker_role": "side_observer",
            "length_bucket": "short",
            "utterance_mode": "direct_answer",
            "surface_texture": "no_punct_fragment",
        }
    if shape == "short_question":
        return {
            "payload_type": "narrow_question",
            "comment_function": "question_followup",
            "evidence_mode": "none_assertion",
            "story_mode": "no_story",
            "voice": "uncertain",
            "speaker_role": "confused_asker",
            "length_bucket": "short" if word_count <= 10 else "medium",
            "utterance_mode": "question_only",
            "surface_texture": "plain",
        }
    if shape == "thanks_ack":
        return {
            "payload_type": "bare_answer",
            "comment_function": "reaction",
            "evidence_mode": "none_assertion",
            "story_mode": "no_story",
            "voice": "grateful",
            "speaker_role": "gratitude_reply",
            "length_bucket": length_bucket_for_word_count(word_count),
            "utterance_mode": "op_followup",
            "surface_texture": "gratitude_social",
        }
    if shape == "joke_reaction":
        return {
            "payload_type": "joke",
            "comment_function": "reaction",
            "evidence_mode": "none_assertion",
            "story_mode": "no_story",
            "voice": "sarcastic",
            "speaker_role": "jokester",
            "length_bucket": length_bucket_for_word_count(word_count),
            "utterance_mode": "joke_only",
            "surface_texture": "emoji_or_sarcasm",
        }
    if shape == "side_tangent":
        return {
            "payload_type": "side_tangent",
            "comment_function": "offtopic_noise",
            "evidence_mode": "small_observation",
            "story_mode": "no_story",
            "voice": "uncertain",
            "speaker_role": "side_observer",
            "length_bucket": length_bucket_for_word_count(word_count),
            "utterance_mode": "side_tangent",
            "surface_texture": "messy_punctuation",
        }
    return {}

def is_question_like_task(task: CommentTask) -> bool:
    return (
        task.speaker_role == "confused_asker"
        or task.payload_type == "narrow_question"
        or task.comment_function == "question_followup"
        or task.utterance_mode == "question_only"
    )

def resolved_tone_shape(task: CommentTask) -> str:
    inferred = infer_tone_shape_for_task(task)
    if task.tone_shape in TONE_SHAPES and tone_shape_is_compatible(task.tone_shape, task):
        return task.tone_shape
    return inferred

def tone_shape_is_compatible(shape: str, task: CommentTask) -> bool:
    if shape == "soft_ack":
        return task.speaker_role in {"gratitude_reply", "op_followup"} or task.voice in {"grateful", "polite_soft"}
    if shape == "personal_dp":
        return (
            task.speaker_role in {"datapoint_only", "op_followup"}
            or task.payload_type in {"fragment_datapoint", "personal_story"}
            or task.evidence_mode == "firsthand_experience"
            or task.story_mode in {"tiny_personal_context", "specific_personal_story", "messy_multi_step_story"}
        )
    if shape == "plain_question":
        return is_question_like_task(task)
    if shape == "direct_correction":
        return (
            task.speaker_role == "contrarian"
            or task.payload_type == "correction"
            or task.comment_function == "correction_caveat"
            or task.utterance_mode == "correction_only"
        )
    if shape == "rant":
        return task.speaker_role == "ranter" or task.payload_type == "rant" or task.utterance_mode == "complaint_only"
    if shape == "light_joke":
        return task.speaker_role == "jokester" or task.payload_type == "joke" or task.utterance_mode == "joke_only"
    if shape == "bare_answer":
        return task.payload_type in TERSE_PAYLOAD_TYPES or task.utterance_mode in {"fragment_only", "direct_answer"}
    return shape in {"neutral_fact", "mild_caveat"}

def infer_tone_shape_for_task(task: CommentTask) -> str:
    if is_question_like_task(task):
        return "plain_question"
    if task.speaker_role == "gratitude_reply" or task.voice == "grateful" or task.surface_texture == "gratitude_social":
        return "soft_ack"
    if task.speaker_role == "ranter" or task.payload_type == "rant" or task.utterance_mode == "complaint_only":
        return "rant"
    if task.speaker_role == "jokester" or task.payload_type == "joke" or task.utterance_mode == "joke_only":
        return "light_joke"
    if (
        task.speaker_role == "contrarian"
        or task.payload_type == "correction"
        or task.comment_function == "correction_caveat"
        or task.utterance_mode == "correction_only"
    ):
        return "direct_correction"
    if (
        task.speaker_role in {"datapoint_only", "op_followup"}
        or task.payload_type in {"fragment_datapoint", "personal_story"}
        or task.evidence_mode == "firsthand_experience"
        or task.story_mode in {"tiny_personal_context", "specific_personal_story", "messy_multi_step_story"}
    ):
        return "personal_dp"
    if task.payload_type in TERSE_PAYLOAD_TYPES or task.utterance_mode in {"fragment_only", "direct_answer"}:
        return "bare_answer"
    if task.voice in {"polite_soft", "uncertain"} or task.comment_function in {"verdict_evaluation", "correction_caveat"}:
        return "mild_caveat"
    return "neutral_fact"

def is_surface_restyled_answer(task: CommentTask) -> bool:
    return task.surface_skeleton in {
        "caveat-first local answer",
        "datapoint-first local answer",
        "plain answer plus local reason",
        "parenthetical aside",
        "terms/reference aside",
        "soft acknowledgement plus detail",
        "uneven caveat detail",
    }

def surface_texture_for_task(task: CommentTask, *, utterance_mode: str) -> str:
    if task.surface_texture and task.surface_texture != "plain":
        return task.surface_texture
    shape = infer_tone_shape_for_task(task)
    if shape == "soft_ack":
        return "gratitude_social"
    if shape == "personal_dp":
        return "abbrev_shorthand"
    if shape in {"neutral_fact", "mild_caveat", "plain_question"}:
        return "plain"
    if utterance_mode == "fragment_only":
        return "no_punct_fragment"
    if utterance_mode == "joke_only" or task.speaker_role == "jokester":
        return "emoji_or_sarcasm"
    if utterance_mode == "op_followup" or task.speaker_role == "gratitude_reply":
        return "gratitude_social"
    if task.speaker_role == "mod_meta":
        return "markdown_quote"
    if task.speaker_role in {"contrarian", "ranter", "side_observer"}:
        return "messy_punctuation"
    return "plain"

def first_choice(choices: tuple[str, ...], default: str) -> str:
    return choices[0] if choices else default

def claim_family_budget_for_total(total_tasks: int, *, max_share: float, min_budget: int) -> int:
    if total_tasks <= 0 or max_share <= 0:
        return 0
    return max(1, min_budget, int(math.ceil(total_tasks * max_share)))
