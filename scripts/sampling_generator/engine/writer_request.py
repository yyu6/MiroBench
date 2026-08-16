from __future__ import annotations

from sampling_generator.engine.model import CommentTask
from sampling_generator.engine.slot_inference import real_tone_slot_for_prompt
from sampling_generator.engine.slot_inference import resolved_tone_shape
from sampling_generator.engine.util import safe_int
from sampling_generator.engine.vocabulary import LOW_INFO_PAYLOAD_TYPES
from sampling_generator.engine.vocabulary import LOW_INFO_UTTERANCE_MODES
from sampling_generator.engine.writer_validation import real_slot_requires_substantive_writer
from typing import Any
import re

def should_use_low_info_writer(task: CommentTask) -> bool:
    if real_slot_requires_substantive_writer(task):
        return False
    if task.utterance_mode in LOW_INFO_UTTERANCE_MODES and safe_int(task.real_word_count, 999) <= 20:
        return True
    if task.payload_type not in LOW_INFO_PAYLOAD_TYPES:
        return False
    return task.length_bucket in {"micro", "short"} or safe_int(task.real_word_count, 999) <= 18

def render_sampled_plan_block(task: CommentTask) -> str:
    if task.context_transform in {"parent_hidden", "semantic_plan_only", "parent_gist", "parent_jittered", "seed_jittered"}:
        rows = [
            ("payload_type", task.payload_type),
            ("speaker_role", task.speaker_role),
            ("tone_shape", resolved_tone_shape(task)),
            ("utterance_mode", task.utterance_mode),
            ("surface_texture", task.surface_texture),
            ("real_surface_shape", task.real_surface_shape),
            ("surface_skeleton", task.surface_skeleton),
            ("reply_relation", task.reply_relation),
            ("stance", task.stance),
            ("tone_target", task.tone_target),
            ("story_mode", task.story_mode),
            ("affect_role", task.affect_role),
            ("local_cue", task.local_anchor or task.local_topic),
        ]
    elif task.context_transform == "minor_detail_focus":
        rows = [
            ("payload_type", task.payload_type),
            ("speaker_role", task.speaker_role),
            ("tone_shape", resolved_tone_shape(task)),
            ("utterance_mode", task.utterance_mode),
            ("surface_texture", task.surface_texture),
            ("real_surface_shape", task.real_surface_shape),
            ("surface_skeleton", task.surface_skeleton),
            ("reply_relation", task.reply_relation),
            ("stance", task.stance),
            ("tone_target", task.tone_target),
            ("story_mode", task.story_mode),
            ("affect_role", task.affect_role),
            ("minor_cue", task.detail_focus or task.local_topic or task.local_anchor),
        ]
    else:
        rows = [
            ("payload_type", task.payload_type),
            ("speaker_role", task.speaker_role),
            ("tone_shape", resolved_tone_shape(task)),
            ("utterance_mode", task.utterance_mode),
            ("surface_texture", task.surface_texture),
            ("real_surface_shape", task.real_surface_shape),
            ("surface_skeleton", task.surface_skeleton),
            ("real_tone_slot", real_tone_slot_for_prompt(task)[0]),
            ("tone_target", task.tone_target),
            ("story_mode", task.story_mode),
            ("affect_role", task.affect_role),
            ("branch_goal", task.branch_goal),
            ("branch_exclusion", task.branch_exclusion),
            ("semantic_move", task.semantic_move),
            ("local_topic", task.local_topic),
            ("reply_relation", task.reply_relation),
            ("stance", task.stance),
            ("detail_focus", task.detail_focus),
            ("perspective_id", task.perspective_id),
            ("domain_intent", task.domain_intent),
            ("decision_boundary", task.decision_boundary),
            ("opening_style", task.opening_style),
            ("development_plan", task.development_plan),
        ]
    lines = [f"- {key}: {value}" for key, value in rows if value]
    if not lines:
        return ""
    return "\nSampled semantic plan:\n" + "\n".join(lines) + "\n"

def controls_for_task(task: CommentTask) -> dict[str, str]:
    controls = {
        "length": task.length_bucket,
        "payload_type": task.payload_type,
        "speaker_role": task.speaker_role,
        "tone_shape": resolved_tone_shape(task),
        "utterance_mode": task.utterance_mode,
        "surface_texture": task.surface_texture,
        "real_surface_shape": task.real_surface_shape,
        "surface_skeleton": task.surface_skeleton,
        "comment_function": task.comment_function,
        "content_angle": task.content_angle,
        "evidence_mode": task.evidence_mode,
        "story_mode": task.story_mode,
        "affect_role": task.affect_role,
        "voice": task.voice,
        "perspective_id": task.perspective_id,
        "domain_intent": task.domain_intent,
        "decision_boundary": task.decision_boundary,
        "claim_key": task.claim_key,
        "claim_family": task.claim_family,
    }
    prompt_tone_slot = real_tone_slot_for_prompt(task)[0]
    if prompt_tone_slot:
        controls["real_tone_slot"] = prompt_tone_slot
    if task.tone_target:
        controls["tone_target"] = task.tone_target
    if task.story_instruction:
        controls["story_instruction"] = task.story_instruction
    if task.affect_instruction:
        controls["affect_instruction"] = task.affect_instruction
    return controls

def writer_temperature(task: CommentTask, *, profile: str = "", attempt_idx: int = 0) -> float:
    if profile in {"osim8b_minimal_context", "osim8b_qwen_style"}:
        return min(0.45, 0.30 + attempt_idx * 0.05)
    if task.length_bucket in {"micro", "short"}:
        base = 0.88
    elif task.comment_function in {"offtopic_noise", "reaction"}:
        base = 0.95
    else:
        base = 0.82
    return min(1.08, base + attempt_idx * 0.07)

def writer_extra_body(profile: str) -> dict[str, Any] | None:
    if profile in {"osim8b_minimal_context", "osim8b_qwen_style"}:
        return {"repetition_penalty": 1.2}
    return None

def max_tokens_for_length(bucket: str) -> int:
    if bucket == "micro":
        return 32
    if bucket == "short":
        return 48
    if bucket == "long":
        return 170
    if bucket == "very_long":
        return 300
    return 110

def strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

def remove_thinking_blocks(text: str) -> str:
    current = text
    while True:
        start = current.find("<think>")
        end = current.find("</think>")
        if start >= 0 and end > start:
            current = (current[:start] + current[end + len("</think>") :]).strip()
            continue
        return current.replace("<think>", "").replace("</think>", "").strip()

def remove_space_token_leakage(text: str) -> str:
    """Remove leading literal `space` artifacts emitted by local writer models."""

    cleaned = re.sub(r"(?im)^[ \t]*space(?:[ \t]*\n+|[ \t]+)", "", str(text or ""))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
