from __future__ import annotations

from sampling_generator.engine.anchors import ANCHOR_OVERLAP_STOPWORDS
from sampling_generator.engine.model import CommentTask
from sampling_generator.engine.slot_inference import is_question_like_task
from sampling_generator.engine.util import compact
from sampling_generator.engine.util import normalize_apostrophe_text
from sampling_generator.engine.util import safe_int
from sampling_generator.engine.vocabulary import LOW_INFO_PAYLOAD_TYPES
from sampling_generator.engine.vocabulary import LOW_INFO_UTTERANCE_MODES
from sampling_generator.engine.vocabulary import TERSE_PAYLOAD_TYPES
from typing import Any
import re

def task_allows_question_mark(task: CommentTask) -> bool:
    return (
        is_question_like_task(task)
        or (task.allow_uncertainty_frame and task.speaker_role in {"op_followup", "confused_asker"})
    )

def looks_like_parent_copy(*, text_norm: str, parent_norm: str) -> bool:
    if not text_norm or not parent_norm:
        return False
    if text_norm == parent_norm:
        return True
    shared_len = min(len(text_norm), len(parent_norm), 220)
    return shared_len >= 120 and text_norm[:shared_len] == parent_norm[:shared_len]

def length_too_long(word_count: int, bucket: str, task: CommentTask | None = None) -> bool:
    """Cap output against the matched slot itself, not just its coarse bucket.

    The bucket ceiling is far away from most slots inside it: a 15-word real
    comment lands in ``medium`` (11-45 real words) whose ceiling is 65, so it
    could be answered at four times its scale and still pass. Measured over v70,
    slots under 30 real words came out 1.20x too long while the 200+ tail came
    out at 0.57x, and both directions compress the thread's length spread
    (generated CV 0.874 against a matched real 1.038). When a matched slot
    exists its own word count is the scale; the bucket table is the fallback for
    a slot that has none.
    """

    real_words = safe_int(getattr(task, "real_word_count", 0), 0) if task is not None else 0
    if real_words > 0:
        return word_count > max(8, int(round(real_words * 1.6)) + 6)
    max_words = {
        "micro": 8,
        "short": 14,
        "medium": 65,
        "long": 135,
        "very_long": 260,
    }.get(bucket, 65)
    return word_count > max_words

def real_slot_requires_substantive_writer(task: CommentTask) -> bool:
    real_words = safe_int(task.real_word_count, 0)
    if real_words >= 46:
        return True
    return task.real_surface_shape in {"full_answer", "story_rant"} and real_words >= 35

def real_slot_min_words(task: CommentTask) -> int:
    """Scale the floor with the slot instead of flattening above 140 words.

    The tiers used to stop at 70 words, so a 274-word matched comment asked for
    the same floor as a 140-word one -- 26% of its scale. That is why the long
    tail is where generation falls furthest short. The proportion below is the
    one the old tiers already implied (0.54-0.55 at 70 and 100 words); it simply
    keeps applying above 140.
    """

    real_words = safe_int(task.real_word_count, 0)
    if real_words >= 70:
        return int(round(real_words * 0.55))
    if task.real_surface_shape in {"full_answer", "story_rant"} and real_words >= 46:
        return 32
    return 0

def real_slot_too_short(word_count: int, task: CommentTask) -> bool:
    minimum = real_slot_min_words(task)
    if not minimum:
        return False
    if task.real_surface_shape in {"micro_reaction", "short_direct_answer", "short_question", "thanks_ack", "joke_reaction"}:
        return False
    return word_count < minimum

def low_info_word_limit(task: CommentTask) -> int | None:
    if real_slot_requires_substantive_writer(task):
        return None
    real_words = safe_int(task.real_word_count, 0)
    if task.utterance_mode == "fragment_only" or (real_words and real_words <= 5):
        return 6
    if task.utterance_mode in {"direct_answer", "question_only", "joke_only", "template_notice"} and real_words <= 10:
        return 10
    if task.utterance_mode in LOW_INFO_UTTERANCE_MODES and real_words and real_words <= 20:
        return min(18, max(8, real_words + 3))
    if task.payload_type in TERSE_PAYLOAD_TYPES:
        if task.length_bucket == "micro":
            return 6
        if task.length_bucket == "short":
            return 10
        if real_words and real_words <= 18:
            return min(18, max(10, real_words + 3))
        return None
    if task.payload_type in {"narrow_question", "fragment_datapoint", "side_tangent", "meta_or_template"}:
        if task.length_bucket in {"micro", "short"}:
            return 12
        if real_words and real_words <= 18:
            return min(20, max(12, real_words + 4))
        return None
    if task.length_bucket in {"micro", "short"}:
        return 12
    return None

def opening_signature(text: str, *, words: int = 7) -> str:
    return compact(" ".join(str(text or "").split()[:words]), 120)

def opening_guard_signature(text: str, *, words: int = 5) -> str:
    return compact(" ".join(str(text or "").lower().split()[:words]), 90)

def opener_family_signature(text: str) -> str:
    normalized = normalize_apostrophe_text(text)
    tokens = re.findall(r"[a-z0-9']+", normalized)
    if not tokens:
        return ""
    first = tokens[0]
    second = tokens[1] if len(tokens) > 1 else ""
    third = tokens[2] if len(tokens) > 2 else ""
    start2 = f"{first} {second}".strip()
    start3 = f"{first} {second} {third}".strip()
    start4 = " ".join(tokens[:4])
    if first in {"yeah", "yep"} and second in {"that", "that's", "this", "it"}:
        return "yeah_that_ack"
    if first in {"thanks", "thank"} and second in {"that", "that's", "yeah", "yep"}:
        return "thanks_that_ack"
    if start3 == "good to know" or start4 == "good to know thanks":
        return "good_to_know_ack"
    if start2 in {"i've been", "ive been", "i've had", "ive had", "i had", "i have", "my last", "my experience"}:
        return "first_person_experience"
    if start2 in {"if you", "if you're", "if youre"} or start3 in {"if you re", "so if you", "but if you"}:
        return "conditional_advice"
    if start2 in {"not sure", "i think", "i'm not", "im not"} or start3 in {"i m not"} or first in {"maybe", "probably"}:
        return "uncertainty_preface"
    if first in {"check", "try", "call"} or start2 in {"you should", "you can", "you could", "make sure"}:
        return "helpful_directive"
    if start3 in {"i totally get", "i get the", "i get why"}:
        return "generic_empathy"
    if first in {"no", "nope", "yes", "yep", "yeah", "nah"}:
        return "direct_answer"
    if first in {"lol", "haha", "lmao"}:
        return "joke_reaction"
    if first in {"thanks", "thank", "appreciate"}:
        return "thanks_ack"
    return ""

def template_phrase_signature(text: str) -> str:
    normalized = normalize_apostrophe_text(text)
    tokens = re.findall(r"[a-z0-9']+", normalized)
    if not tokens:
        return ""
    head = " ".join(tokens[:28])
    if re.search(r"\b(that's|that is|thats)\s+the\s+part\b|\bthe\s+(annoying\s+)?part\s+that\b", head):
        return "gpt_part_frame"
    if re.search(r"\b(that's|that is|thats|that)\s+basically\b|\bbasically\s+(the|how|where|what)\b", head):
        return "gpt_basically_frame"
    if re.search(r"\bkind\s+of\b|\bkinda\b", head):
        return "gpt_kind_of_frame"
    if re.search(r"\bfeels?\s+like\b|\bfelt\s+like\b", head):
        return "gpt_feels_like_frame"
    if re.search(r"\bgood\s+to\s+know\b", head):
        return "gpt_good_to_know_frame"
    if re.search(r"\b(i've|ive|i have|i had|my)\b.{0,80}\b(similar|same|been|had|seen|noticed|found|used|heard|got|experience|issue|problem|situation)\b", head):
        return "first_person_experience_frame"
    if re.search(r"\b(not sure|i think|i'm not|im not|maybe|probably|i wonder|seems like)\b", head):
        return "uncertainty_frame"
    if re.search(r"\b(worth|value|pays for itself|justify)\b", head):
        return "worth_frame"
    if re.search(r"\b(check|try|call|make sure|you should|you can|you could)\b", head):
        return "generic_advice_frame"
    return ""

def previous_comment_texts(previous_comments: list[dict[str, Any]] | None) -> list[str]:
    return [
        str(comment.get("content") or "").strip()
        for comment in (previous_comments or [])
        if str(comment.get("content") or "").strip()
    ]

def contains_writer_placeholder_literal(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if re.search(r"\[(source|link|wiki|sidebar|placeholder)[^\]]*\]", lowered):
        return True
    if re.search(r"\b(sidebar cashback card lists|quote/link reference|card recommendation template)\b", lowered):
        return True
    first_line = lowered.splitlines()[0].strip()
    if re.match(r"^>\s*(the\s+)?(sidebar|card recommendation template|application context template|source link|quote/link reference|cashback card lists|cashback card resources)\b", first_line):
        return True
    return False

PLANNER_RESIDUE_PATTERNS = (
    "bonus acronym explained",
    "acronym explained",
    "tiny fragment reaction",
    "single narrow question",
    "short direct answer",
    "quote/link reference",
    "messy first-person datapoint",
    "compact answer with parenthetical aside",
    "blunt short verdict",
    "2-sentence local comment",
    "3-sentence local comment",
    "1-sentence local comment",
    "brief thanks or acknowledgement",
    "elliptical aside",
    "joke or sarcastic aside",
)

def is_meta_template_task(task: CommentTask) -> bool:
    return (
        task.payload_type == "meta_or_template"
        or task.utterance_mode == "template_notice"
        or task.real_surface_shape in {"template_notice", "link_reference", "quote_link_reference"}
    )

def is_gpt_long_helpful_task(task: CommentTask) -> bool:
    return (
        task.length_bucket in {"long", "very_long"}
        and task.payload_type in {"advice", "soft_helpful"}
        and task.utterance_mode in {"local_advice", "op_followup", "correction_only"}
    )

def task_requires_concrete_anchor_density(task: CommentTask) -> bool:
    if not task.concrete_anchors:
        return False
    if task.payload_type in LOW_INFO_PAYLOAD_TYPES or task.utterance_mode in LOW_INFO_UTTERANCE_MODES:
        return False
    if is_meta_template_task(task) and safe_int(task.real_word_count, 0) < 35:
        return False
    return (
        real_slot_requires_substantive_writer(task)
        or is_gpt_long_helpful_task(task)
        or task.length_bucket in {"long", "very_long"}
        or safe_int(task.real_word_count, 0) >= 55
    )

def has_task_anchor_overlap(text: str, task: CommentTask) -> bool:
    lowered = normalize_apostrophe_text(text)
    source = " ".join(
        item
        for item in (
            task.local_anchor,
            task.detail_focus,
            task.local_topic,
            task.semantic_move,
            task.planner_intent,
            task.claim_key,
            task.claim_family,
        )
        if item
    )
    terms = {
        token
        for token in re.findall(r"[a-z0-9/]+", normalize_apostrophe_text(source))
        if len(token) >= 4 and token not in ANCHOR_OVERLAP_STOPWORDS
    }
    if not terms:
        return False
    hits = [
        term
        for term in terms
        if re.search(rf"\b{re.escape(term)}\b", lowered)
    ]
    return len(hits) >= 2
