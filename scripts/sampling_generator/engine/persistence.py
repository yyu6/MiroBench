from __future__ import annotations

from pathlib import Path
from sampling_generator.engine.model import CommentTask
from sampling_generator.engine.slot_inference import resolved_tone_shape
from sampling_generator.engine.util import first_line
from sampling_generator.engine.util import increment
from sampling_generator.engine.util import median_int
from sampling_generator.engine.util import normalize_exact
from sampling_generator.engine.util import safe_int
from sampling_generator.engine.util import utc_now
from sampling_generator.engine.util import write_json
from typing import Any
import argparse
import json

def completed_seed_slots(discussion: dict[str, Any]) -> set[int]:
    slots: set[int] = set()
    for post in discussion.get("posts") or []:
        slot = post.get("post_slot")
        if slot is None:
            post_id = str(post.get("post_id") or "")
            marker = "_post"
            pos = post_id.find(marker)
            if pos >= 0 and len(post_id) >= pos + len(marker) + 2:
                slot = safe_int(post_id[pos + len(marker) : pos + len(marker) + 2], -1)
        slot_int = safe_int(slot, -1)
        if slot_int >= 0:
            slots.add(slot_int)
    return slots

def replace_or_append_post(discussion: dict[str, Any], post_slot: int, post: dict[str, Any]) -> None:
    post["post_slot"] = post_slot
    posts = discussion.setdefault("posts", [])
    for idx, old in enumerate(posts):
        if safe_int(old.get("post_slot"), -1) == post_slot:
            posts[idx] = post
            return
    posts.append(post)
    posts.sort(key=lambda item: safe_int(item.get("post_slot"), 0))

def write_discussion_bundle(run_dir: Path, discussion: dict[str, Any], global_memory: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "discussion.json", discussion)
    (run_dir / "discussion.md").write_text(render_markdown(discussion), encoding="utf-8")
    write_json(run_dir / "summary.json", summarize_discussion(discussion))
    write_json(run_dir / "generation_records.json", flatten_generation_records(discussion))
    write_json(run_dir / "global_memory_snapshot.json", global_memory)

def render_markdown(discussion: dict[str, Any]) -> str:
    meta = discussion.get("meta") or {}
    lines = [
        f"# Sampled Reddit discussion - {meta.get('product_category', 'credit_cards')}",
        f"*Run: {meta.get('run_id')} | Generator: {meta.get('generator')} | Model: {meta.get('model')}*",
        "",
        "---",
        "",
    ]
    for post in discussion.get("posts") or []:
        title = str(post.get("title") or first_line(str(post.get("content") or "")))
        lines.extend(
            [
                f"## [{post.get('likes', 0)} up] {title}",
                f"**u/{post.get('author', 'op')}**",
                "",
                str(post.get("content") or ""),
                "",
            ]
        )
        lines.extend(render_comment_markdown(post.get("comments") or []))
        lines.extend(["---", ""])
    return "\n".join(lines)

def render_comment_markdown(comments: list[dict[str, Any]], level: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = ">" * (level + 1)
    for comment in comments:
        meta = (
            f"job={comment.get('comment_job')} | "
            f"payload={comment.get('payload_type')} | "
            f"len={comment.get('length_bucket')} | "
            f"role={comment.get('speaker_role')} | "
            f"tone={comment.get('tone_shape')} | "
            f"utt={comment.get('utterance_mode')} | "
            f"texture={comment.get('surface_texture')} | "
            f"real_shape={comment.get('real_surface_shape')} | "
            f"skeleton={comment.get('surface_skeleton')} | "
            f"voice={comment.get('voice')} | "
            f"angle={comment.get('content_angle')} | "
            f"scope={comment.get('visibility_scope')} | "
            f"context={comment.get('context_transform')} | "
            f"depth={comment.get('depth')}"
        )
        lines.extend(
            [
                f"{prefix} **u/{comment.get('author')}** [{comment.get('likes', 0)} up] `{meta}`",
                prefix,
            ]
        )
        for line in str(comment.get("content") or "").splitlines():
            lines.append(f"{prefix} {line}")
        lines.append("")
        lines.extend(render_comment_markdown(comment.get("replies") or [], level + 1))
    return lines

def summarize_discussion(discussion: dict[str, Any]) -> dict[str, Any]:
    flat = list(iter_comments(discussion))
    word_counts = [len(str(comment.get("content") or "").split()) for comment in flat]
    by_depth: dict[str, int] = {}
    by_function: dict[str, int] = {}
    by_angle: dict[str, int] = {}
    by_voice: dict[str, int] = {}
    by_payload_type: dict[str, int] = {}
    by_length: dict[str, int] = {}
    by_speaker_role: dict[str, int] = {}
    by_tone_shape: dict[str, int] = {}
    by_utterance_mode: dict[str, int] = {}
    by_surface_texture: dict[str, int] = {}
    by_real_surface_shape: dict[str, int] = {}
    by_surface_skeleton: dict[str, int] = {}
    by_claim_family: dict[str, int] = {}
    by_context_aperture: dict[str, int] = {}
    by_context_transform: dict[str, int] = {}
    by_opener_family: dict[str, int] = {}
    by_template_phrase_family: dict[str, int] = {}
    for comment in flat:
        increment(by_depth, str(comment.get("depth", 0)))
        increment(by_function, str(comment.get("comment_function") or comment.get("comment_job") or "unknown"))
        increment(by_angle, str(comment.get("content_angle") or "unknown"))
        increment(by_voice, str(comment.get("voice") or "unknown"))
        increment(by_payload_type, str(comment.get("payload_type") or "unknown"))
        increment(by_length, str(comment.get("length_bucket") or "unknown"))
        increment(by_speaker_role, str(comment.get("speaker_role") or "unknown"))
        increment(by_tone_shape, str(comment.get("tone_shape") or "unknown"))
        increment(by_utterance_mode, str(comment.get("utterance_mode") or "unknown"))
        increment(by_surface_texture, str(comment.get("surface_texture") or "unknown"))
        increment(by_real_surface_shape, str(comment.get("real_surface_shape") or "unknown"))
        increment(by_surface_skeleton, str(comment.get("surface_skeleton") or "unknown"))
        increment(by_claim_family, str(comment.get("claim_family") or "unknown"))
        increment(by_context_aperture, str(comment.get("context_aperture") or "unknown"))
        increment(by_context_transform, str(comment.get("context_transform") or "unknown"))
        if comment.get("opener_family"):
            increment(by_opener_family, str(comment.get("opener_family")))
        if comment.get("template_phrase_family"):
            increment(by_template_phrase_family, str(comment.get("template_phrase_family")))
    exact_texts = [normalize_exact(str(comment.get("content") or "")) for comment in flat]
    return {
        "posts": len(discussion.get("posts") or []),
        "comments": len(flat),
        "word_len_min": min(word_counts) if word_counts else 0,
        "word_len_median": median_int(word_counts),
        "word_len_max": max(word_counts) if word_counts else 0,
        "short_le_10": sum(value <= 10 for value in word_counts),
        "long_ge_80": sum(value >= 80 for value in word_counts),
        "exact_duplicate_count": len(exact_texts) - len(set(exact_texts)),
        "by_depth": by_depth,
        "by_comment_function": by_function,
        "by_content_angle": by_angle,
        "by_voice": by_voice,
        "by_payload_type": by_payload_type,
        "by_length_bucket": by_length,
        "by_speaker_role": by_speaker_role,
        "by_tone_shape": by_tone_shape,
        "by_utterance_mode": by_utterance_mode,
        "by_surface_texture": by_surface_texture,
        "by_real_surface_shape": by_real_surface_shape,
        "by_surface_skeleton": by_surface_skeleton,
        "by_claim_family": by_claim_family,
        "by_context_aperture": by_context_aperture,
        "by_context_transform": by_context_transform,
        "by_opener_family": by_opener_family,
        "by_template_phrase_family": by_template_phrase_family,
    }

def flatten_generation_records(discussion: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for post in discussion.get("posts") or []:
        for record in post.get("generation_records") or []:
            records.append(
                {
                    "post_id": post.get("post_id"),
                    "seed_index": post.get("seed_index"),
                    **record,
                }
            )
    return records

def iter_comments(discussion: dict[str, Any]):
    for post in discussion.get("posts") or []:
        yield from iter_comment_tree(post.get("comments") or [])

def iter_comment_tree(comments: list[dict[str, Any]]):
    for comment in comments:
        yield comment
        yield from iter_comment_tree(comment.get("replies") or [])

def update_global_memory(global_memory: dict[str, Any], tasks: list[CommentTask]) -> None:
    for key in (
        "comment_function",
        "content_angle",
        "evidence_mode",
        "voice",
        "payload_type",
        "story_mode",
        "length_bucket",
        "speaker_role",
        "utterance_mode",
        "surface_texture",
        "real_surface_shape",
        "tone_shape",
        "claim_family",
        "perspective_id",
        "claim_key",
        "context_aperture",
        "context_transform",
    ):
        global_memory.setdefault(key, {})
    for task in tasks:
        for key, value in (
            ("comment_function", task.comment_function),
            ("content_angle", task.content_angle),
            ("evidence_mode", task.evidence_mode),
            ("voice", task.voice),
            ("payload_type", task.payload_type),
            ("story_mode", task.story_mode),
            ("length_bucket", task.length_bucket),
            ("speaker_role", task.speaker_role),
            ("utterance_mode", task.utterance_mode),
            ("surface_texture", task.surface_texture),
            ("real_surface_shape", task.real_surface_shape),
            ("tone_shape", resolved_tone_shape(task)),
            ("claim_family", task.claim_family),
            ("perspective_id", task.perspective_id or "seed_local"),
            ("claim_key", task.claim_key or "local_claim"),
            ("context_aperture", task.context_aperture),
            ("context_transform", task.context_transform),
        ):
            counts = global_memory.setdefault(key, {})
            counts[value] = safe_int(counts.get(value), 0) + 1
    global_memory["updated_at"] = utc_now()

def load_global_memory(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "global_memory.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "comment_function": {},
        "content_angle": {},
        "evidence_mode": {},
        "voice": {},
        "payload_type": {},
        "story_mode": {},
        "length_bucket": {},
        "speaker_role": {},
        "utterance_mode": {},
        "surface_texture": {},
        "real_surface_shape": {},
        "tone_shape": {},
        "claim_family": {},
        "perspective_id": {},
        "claim_key": {},
        "context_aperture": {},
        "context_transform": {},
    }

def upsert_run_manifest(
    *,
    manifest: dict[str, Any],
    run_index: int,
    run_dir: Path,
    completed_posts: int,
    total_posts: int,
) -> None:
    runs = manifest.setdefault("runs", [])
    row = {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "completed_posts": completed_posts,
        "total_posts": total_posts,
        "status": "complete" if completed_posts >= total_posts else "in_progress",
        "updated_at": utc_now(),
    }
    for idx, existing in enumerate(runs):
        if safe_int(existing.get("run_index"), -1) == run_index:
            runs[idx] = row
            return
    runs.append(row)

def task_to_dict(task: CommentTask) -> dict[str, Any]:
    return {
        "local_task_id": task.local_task_id,
        "local_parent_task_id": task.local_parent_task_id,
        "depth": task.depth,
        "branch_id": task.branch_id,
        "branch_goal": task.branch_goal,
        "branch_exclusion": task.branch_exclusion,
        "owned_decision_subject": task.owned_decision_subject,
        "forbidden_decision_subjects": task.forbidden_decision_subjects,
        "visible_scope": task.visible_scope,
        "local_anchor": task.local_anchor,
        "comment_function": task.comment_function,
        "content_angle": task.content_angle,
        "evidence_mode": task.evidence_mode,
        "story_mode": task.story_mode,
        "voice": task.voice,
        "payload_type": task.payload_type,
        "length_bucket": task.length_bucket,
        "speaker_role": task.speaker_role,
        "tone_shape": resolved_tone_shape(task),
        "utterance_mode": task.utterance_mode,
        "surface_texture": task.surface_texture,
        "allow_first_person_frame": task.allow_first_person_frame,
        "allow_uncertainty_frame": task.allow_uncertainty_frame,
        "planner_intent": task.planner_intent,
        "must_not_do": task.must_not_do,
        "real_sample_id": task.real_sample_id,
        "real_parent_sample_id": task.real_parent_sample_id,
        "real_word_count": task.real_word_count,
        "real_surface_shape": task.real_surface_shape,
        "surface_skeleton": task.surface_skeleton,
        "surface_instruction": task.surface_instruction,
        "real_tone_slot": task.real_tone_slot,
        "real_tone_instruction": task.real_tone_instruction,
        "tone_overlay_slot": task.tone_overlay_slot,
        "tone_overlay_instruction": task.tone_overlay_instruction,
        "tone_target": task.tone_target,
        "tone_target_instruction": task.tone_target_instruction,
        "story_instruction": task.story_instruction,
        "affect_role": task.affect_role,
        "affect_instruction": task.affect_instruction,
        "distribution_assignment": task.distribution_assignment,
        "concrete_anchors": list(task.concrete_anchors),
        "semantic_move": task.semantic_move,
        "local_topic": task.local_topic,
        "reply_relation": task.reply_relation,
        "stance": task.stance,
        "detail_focus": task.detail_focus,
        "avoid_repeating": task.avoid_repeating,
        "claim_family": task.claim_family,
        "claim_key": task.claim_key,
        "perspective_id": task.perspective_id,
        "domain_intent": task.domain_intent,
        "decision_boundary": task.decision_boundary,
        "reply_delta": task.reply_delta,
        "reply_delta_type": task.reply_delta_type,
        "reply_novelty_anchor": task.reply_novelty_anchor,
        "parent_semantic_move": task.parent_semantic_move,
        "parent_decision_boundary": task.parent_decision_boundary,
        "opening_style": task.opening_style,
        "development_plan": task.development_plan,
        "context_aperture": task.context_aperture,
        "context_transform": task.context_transform,
    }

def config_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    hidden_keys = {"planner_api_key", "writer_api_key"}
    result = {}
    for key, value in vars(args).items():
        result[key] = "***" if key in hidden_keys and value else value
    return result

def scrub_argv(argv: list[str]) -> list[str]:
    scrubbed: list[str] = []
    redact_next = False
    secret_flags = {"--planner-api-key", "--writer-api-key"}
    for item in argv:
        if redact_next:
            scrubbed.append("***")
            redact_next = False
            continue
        if item in secret_flags:
            scrubbed.append(item)
            redact_next = True
            continue
        if any(item.startswith(flag + "=") for flag in secret_flags):
            flag = item.split("=", 1)[0]
            scrubbed.append(flag + "=***")
            continue
        scrubbed.append(item)
    return scrubbed
