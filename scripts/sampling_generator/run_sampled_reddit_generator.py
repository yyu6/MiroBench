#!/usr/bin/env python3
"""Sample-planned Reddit discussion generator.

This generator is intentionally separate from the older controlled Qwen
generator.  The design is:

1. A GPT branch planner reads the seed post plus the matched real thread and
   samples broad discussion branches.
2. A GPT per-comment planner samples payload type plus a light local move from
   stratified matched real comments.
3. Deterministic code expands the matched real depth/parent/length skeleton into
   one task per comment and applies claim-family / opening guards.
4. A Qwen-style writer receives one local task at a time and writes exactly one
   Reddit-style comment.  Replies only see their parent comment by default.
5. A final deterministic rebalance caps repeated complete-answer shapes before
   writing, because those shapes were the main source of high self-BERT.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

# The adapter loads this file by path, so it has no package of its own. Put the
# scripts directory on the path once and import the engine absolutely; the
# facade stays a fresh module per load while the engine modules are shared.
_SCRIPTS_ROOT = str(Path(__file__).resolve().parent.parent)
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

from sampling_generator.engine.util import (  # noqa: E402
    compact,
    first_line,
    increment,
    is_gratitude_text,
    load_json,
    median_int,
    nonempty,
    normalize_apostrophe_text,
    normalize_claim_key,
    normalize_exact,
    normalize_vocab_list,
    normalize_vocab_value,
    run_git_text,
    safe_getsource,
    safe_int,
    utc_now,
    weighted_choice,
    write_json,
)
from sampling_generator.engine.model import (  # noqa: E402
    BranchPlan,
    CommentTask,
    SeedPost,
    ThreadTarget,
)
from sampling_generator.engine.anchors import (  # noqa: E402
    ANCHOR_OVERLAP_STOPWORDS,
    GENERIC_ANCHOR_WORDS,
    clean_anchor_label,
    concrete_anchor_base_label,
    concrete_anchor_key,
    concrete_anchor_tokens,
    dedup_anchors,
    extract_money_number_anchors,
    extract_short_dp_anchors,
    extract_url_anchor_labels,
)
from sampling_generator.engine.vocabulary import (  # noqa: E402
    CAPPED_OPENER_FAMILIES,
    CAPPED_TEMPLATE_PHRASE_FAMILIES,
    COMMENT_FUNCTIONS,
    CONTENT_ANGLES,
    CONTEXT_APERTURES,
    EVIDENCE_MODES,
    HARD_REAL_SURFACE_SHAPES,
    LENGTH_BUCKET_BOUNDS,
    LOW_INFO_PAYLOAD_TYPES,
    LOW_INFO_UTTERANCE_MODES,
    PAYLOAD_TYPES,
    SPEAKER_ROLES,
    STORY_MODES,
    SURFACE_TEXTURES,
    SYSTEM_PROMPTS,
    TERSE_PAYLOAD_TYPES,
    TONE_SHAPES,
    UTTERANCE_MODES,
    VOICE_MODES,
    is_hard_real_surface_shape,
)
from sampling_generator.engine.slot_inference import (  # noqa: E402
    PROMPT_VISIBLE_REAL_TONE_SLOTS,
    claim_family_budget_for_total,
    first_choice,
    infer_payload_type,
    infer_speaker_role,
    infer_surface_texture,
    infer_tone_shape_for_task,
    infer_utterance_mode,
    infer_voice_for_payload,
    is_question_like_task,
    is_surface_restyled_answer,
    length_bucket_for_payload,
    length_bucket_for_word_count,
    overrides_for_real_surface_shape,
    real_tone_slot_for_prompt,
    resolved_tone_shape,
    surface_texture_for_task,
    tone_shape_is_compatible,
    voice_for_payload,
)
from sampling_generator.engine.persistence import (  # noqa: E402
    completed_seed_slots,
    config_snapshot,
    flatten_generation_records,
    iter_comment_tree,
    iter_comments,
    load_global_memory,
    render_comment_markdown,
    render_markdown,
    replace_or_append_post,
    scrub_argv,
    summarize_discussion,
    task_to_dict,
    update_global_memory,
    upsert_run_manifest,
    write_discussion_bundle,
)
from sampling_generator.engine.writer_validation import (  # noqa: E402
    PLANNER_RESIDUE_PATTERNS,
    contains_writer_placeholder_literal,
    has_task_anchor_overlap,
    is_gpt_long_helpful_task,
    is_meta_template_task,
    length_too_long,
    looks_like_parent_copy,
    low_info_word_limit,
    opener_family_signature,
    opening_guard_signature,
    opening_signature,
    previous_comment_texts,
    real_slot_min_words,
    real_slot_requires_substantive_writer,
    real_slot_too_short,
    task_allows_question_mark,
    task_requires_concrete_anchor_density,
    template_phrase_signature,
)
from sampling_generator.engine.writer_request import (  # noqa: E402
    controls_for_task,
    max_tokens_for_length,
    remove_space_token_leakage,
    remove_thinking_blocks,
    render_sampled_plan_block,
    should_use_low_info_writer,
    strip_code_fence,
    writer_extra_body,
    writer_temperature,
)
from sampling_generator.engine.thread_structure import (  # noqa: E402
    branch_by_id,
    comments_for_shape,
    evenly_spaced_indices,
    order_comments_by_thread_tree,
    planner_batch_ranges_by_depth,
    real_comment_keys,
    render_top_counts,
    sample_thread_target,
    selected_matched_comments,
)
from sampling_generator.engine.cli import (  # noqa: E402
    DEFAULT_PLANNER_MODEL,
    DEFAULT_WRITER_BASE_URL,
    DEFAULT_WRITER_MODEL,
    describe_bad_endpoint,
    is_recoverable_post_error,
    load_seed_posts,
    make_openai_client,
    parse_args,
    record_post_failure,
)
from sampling_generator.engine.context_policy import (  # noqa: E402
    choose_context_transform,
    default_reply_delta,
)
from sampling_generator.engine.parent_alignment import (  # noqa: E402
    align_task_to_generated_parent,
    nearest_generated_ancestor,
    writer_payload_for_token_cap,
)


GENERATOR_NAME = "sampled_planner_gpt_writer_v54_constructive_polite_frame"
CLAIM_FAMILIES = (
    "low_limit_amount",
    "approval_datapoint",
    "cli_process",
    "hard_soft_pull",
    "credit_transfer_exposure",
    "product_comparison",
    "utilization_score",
    "asset_requirement",
    "temporary_activation",
    "support_process",
    "cashback_points_value",
    "joke_reaction",
    "issuer_policy",
    "miscellaneous",
)

def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_reproducibility_snapshot(output_dir, args)

    seeds = load_seed_posts(Path(args.seed_post_pool_json).expanduser())
    if not seeds:
        raise SystemExit("No seed posts loaded.")

    real_bank = load_real_thread_bank(
        Path(args.real_comments_dir).expanduser(),
        max_threads=args.max_real_threads_loaded,
    )

    planner_client = make_openai_client(
        base_url=args.planner_base_url,
        api_key=args.planner_api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY"),
        timeout=args.planner_timeout,
    )
    writer_client = make_openai_client(
        base_url=args.writer_base_url,
        api_key=args.writer_api_key,
        timeout=args.writer_timeout,
    )
    preflight_openai_compatible_endpoint(
        role="planner",
        base_url=args.planner_base_url,
        api_key=args.planner_api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY"),
        allow_remote=True,
    )
    preflight_openai_compatible_endpoint(
        role="writer",
        base_url=args.writer_base_url,
        api_key=args.writer_api_key,
        allow_remote=False,
    )

    manifest_path = output_dir / "manifest.json"
    manifest = load_json(manifest_path, default={})
    manifest.setdefault("generator", GENERATOR_NAME)
    manifest.setdefault("created_at", utc_now())
    manifest.setdefault("runs", [])
    manifest["updated_at"] = utc_now()
    manifest["config"] = config_snapshot(args)
    write_json(manifest_path, manifest)

    global_memory = load_global_memory(output_dir)
    total_posts = args.runs * args.posts_per_run
    if args.max_total_posts:
        total_posts = min(total_posts, args.max_total_posts)

    for run_index in range(args.runs):
        run_dir = output_dir / f"run_{run_index:02d}_sampled_reddit"
        discussion = load_or_init_discussion(
            run_dir=run_dir,
            run_index=run_index,
            args=args,
        )
        completed_slots = completed_seed_slots(discussion)

        for post_slot in range(args.posts_per_run):
            global_post_slot = run_index * args.posts_per_run + post_slot
            if global_post_slot >= total_posts:
                break
            seed_index = args.start_seed_index + global_post_slot
            if seed_index >= len(seeds):
                if args.wrap_seed_posts:
                    seed_index = seed_index % len(seeds)
                else:
                    print(f"Stopping: requested seed index {seed_index}, pool has {len(seeds)} posts.")
                    break
            if post_slot in completed_slots and not args.force_post:
                print(f"[resume] run={run_index:02d} post_slot={post_slot:02d} already complete")
                continue

            generated = run_post_with_recovery(
                operation=lambda: generate_one_post_slot(
                    args=args,
                    seeds=seeds,
                    real_bank=real_bank,
                    planner_client=planner_client,
                    writer_client=writer_client,
                    global_memory=global_memory,
                    discussion=discussion,
                    output_dir=output_dir,
                    run_dir=run_dir,
                    run_index=run_index,
                    post_slot=post_slot,
                    seed_index=seed_index,
                ),
                output_dir=output_dir,
                run_index=run_index,
                post_slot=post_slot,
                seed_index=seed_index,
                retry_limit=args.post_retry_limit,
                retry_delay=args.post_retry_delay,
            )
            if generated:
                manifest = load_json(manifest_path, default={})
                manifest["updated_at"] = utc_now()
                manifest.setdefault("runs", [])
                upsert_run_manifest(
                    manifest=manifest,
                    run_index=run_index,
                    run_dir=run_dir,
                    completed_posts=len(completed_seed_slots(discussion)),
                    total_posts=args.posts_per_run,
                )
                write_json(manifest_path, manifest)

    print(f"Done. Output: {output_dir}")


def run_post_with_recovery(
    *,
    operation: Callable[[], bool],
    output_dir: Path,
    run_index: int,
    post_slot: int,
    seed_index: int,
    retry_limit: int,
    retry_delay: float,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    attempt = 0
    while True:
        attempt += 1
        try:
            return operation()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - classified below
            if not is_recoverable_post_error(exc):
                raise
            record_post_failure(
                output_dir=output_dir,
                run_index=run_index,
                post_slot=post_slot,
                seed_index=seed_index,
                attempt=attempt,
                error=exc,
            )
            if retry_limit > 0 and attempt >= retry_limit:
                raise RuntimeError(
                    "Recoverable post generation failed after configured retries: "
                    f"run={run_index:02d} post_slot={post_slot:02d} "
                    f"seed={seed_index} attempts={attempt}"
                ) from exc
            delay = min(300.0, max(0.0, retry_delay) * attempt)
            print(
                f"[post-retry] run={run_index:02d} post_slot={post_slot:02d} "
                f"seed={seed_index} attempt={attempt} "
                f"error={type(exc).__name__}:{exc} sleep={delay:.1f}s",
                flush=True,
            )
            if delay:
                sleep(delay)


def generate_one_post_slot(
    *,
    args: argparse.Namespace,
    seeds: list[SeedPost],
    real_bank: Any,
    planner_client: Any,
    writer_client: Any,
    global_memory: dict[str, Any],
    discussion: dict[str, Any],
    output_dir: Path,
    run_dir: Path,
    run_index: int,
    post_slot: int,
    seed_index: int,
) -> bool:
    """Generate and atomically persist one post after all quality gates pass."""

    seed_post = seeds[seed_index]
    post_rng = random.Random(args.seed + 1009 * seed_index + 9176 * run_index)
    target = sample_thread_target(
        seed_post=seed_post,
        rng=post_rng,
        max_comments_per_post=args.max_comments_per_post,
        count_scale=args.comment_count_scale,
        exact_matched_thread_size=args.exact_matched_thread_size,
    )
    print(
        f"[plan] run={run_index:02d} post_slot={post_slot:02d} "
        f"seed={seed_index} target={target.target_comments} "
        f"shape={target.shape_label} title={seed_post.title[:80]!r}"
    )

    matched_real_thread = find_matched_real_thread(real_bank, seed_post)
    matched_note = (
        f"matched_real={matched_real_thread.get('post_id')} comments={len(matched_real_thread.get('comments') or [])}"
        if matched_real_thread
        else "matched_real=missing"
    )
    print(f"       {matched_note}")
    if not matched_real_thread:
        print(
            f"       [skip] no exact matched real thread for seed={seed_index} "
            f"source={seed_post.source_raw_post_id}"
        )
        return False
    plan = plan_thread(
        client=planner_client,
        model=args.planner_model,
        seed_post=seed_post,
        target=target,
        matched_real_thread=matched_real_thread,
        matched_real_comments=args.matched_real_comments,
        global_memory=global_memory,
        retries=args.planner_retries,
        max_tokens=args.planner_max_tokens,
    )
    comment_plans = plan_comment_moves(
        client=planner_client,
        model=args.planner_model,
        seed_post=seed_post,
        target=target,
        branches=plan,
        matched_real_thread=matched_real_thread,
        matched_real_comments=args.matched_real_comments,
        batch_size=args.comment_planner_batch_size,
        retries=args.planner_retries,
        max_tokens=args.comment_planner_max_tokens,
    )
    tasks = expand_plan_to_tasks(
        branches=plan,
        target=target,
        seed_post=seed_post,
        matched_real_thread=matched_real_thread,
        matched_real_comments=args.matched_real_comments,
        comment_plans=comment_plans,
        rng=post_rng,
        context_dropout_rate=args.context_dropout_rate,
        context_jitter_rate=args.context_jitter_rate,
    )
    tasks = rebalance_tasks_for_diversity(
        tasks,
        rng=post_rng,
        advisor_max_share=args.advisor_max_share,
        question_max_share=args.question_max_share,
        micro_target_share=args.micro_target_share,
        short_max_share=args.short_max_share,
        social_noise_min_share=args.social_noise_min_share,
        gratitude_min_share=args.gratitude_min_share,
        tone_harsh_max_share=args.tone_harsh_max_share,
        tone_calm_min_share=args.tone_calm_min_share,
        tone_personal_min_share=args.tone_personal_min_share,
        tone_polite_min_share=args.tone_polite_min_share,
    )
    post = generate_post_from_tasks(
        writer_client=writer_client,
        writer_model=args.writer_model,
        writer_profile=args.writer_profile,
        seed_post=seed_post,
        tasks=tasks,
        run_index=run_index,
        post_slot=post_slot,
        seed_index=seed_index,
        rng=post_rng,
        max_writer_tokens=args.writer_max_tokens,
        writer_retries=args.writer_retries,
        claim_key_budget=args.claim_key_budget,
        claim_family_max_share=args.claim_family_max_share,
        claim_family_min_budget=args.claim_family_min_budget,
        opening_reuse_budget=args.opening_reuse_budget,
        opener_family_reuse_budget=args.opener_family_reuse_budget,
        template_phrase_reuse_budget=args.template_phrase_reuse_budget,
    )
    replace_or_append_post(discussion, post_slot, post)
    update_global_memory(global_memory, tasks)
    write_discussion_bundle(run_dir, discussion, global_memory)
    write_json(output_dir / "global_memory.json", global_memory)
    return True


def plan_thread(
    *,
    client: Any,
    model: str,
    seed_post: SeedPost,
    target: ThreadTarget,
    matched_real_thread: dict[str, Any] | None,
    matched_real_comments: int,
    global_memory: dict[str, Any],
    retries: int,
    max_tokens: int,
) -> list[BranchPlan]:
    prompt = build_planner_prompt(
        seed_post=seed_post,
        target=target,
        matched_real_thread=matched_real_thread,
        matched_real_comments=matched_real_comments,
        global_memory=global_memory,
    )
    last_error = ""
    for attempt in range(max(1, retries + 1)):
        final_prompt = prompt
        if last_error:
            final_prompt += (
                "\n\nPrevious response was invalid for this reason:\n"
                f"{last_error}\nReturn corrected JSON only."
            )
        raw = chat_completion_text(
            client=client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sampling planner for realistic Reddit discussions. "
                        "Return strict JSON only. Do not write comments."
                    ),
                },
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.55,
            max_tokens=max_tokens,
            response_format_json=True,
        )
        try:
            payload = parse_json_object(raw)
            branches = normalize_branch_plan(payload, target=target)
            if branches:
                return branches
            last_error = "branch_plans was empty"
        except Exception as exc:  # noqa: BLE001 - planner retry feedback
            last_error = str(exc)
    raise RuntimeError(f"Planner failed after retries: {last_error}")


def normalize_branch_plan(payload: dict[str, Any], *, target: ThreadTarget) -> list[BranchPlan]:
    raw_branches = payload.get("branch_plans") or payload.get("branches") or []
    if not isinstance(raw_branches, list):
        raise ValueError("branch_plans must be a list")
    branches: list[BranchPlan] = []
    seen_ids: set[int] = set()
    for idx, row in enumerate(raw_branches, start=1):
        if not isinstance(row, dict):
            continue
        branch_id = safe_int(row.get("branch_id"), idx)
        if branch_id in seen_ids:
            branch_id = idx
        seen_ids.add(branch_id)
        branch = BranchPlan(
            branch_id=branch_id,
            anchor_quote=nonempty(row.get("anchor_quote"), "one concrete seed-post detail"),
            anchor_source=nonempty(row.get("anchor_source"), "seed_post_detail"),
            detour_type=nonempty(row.get("detour_type"), "main_answer"),
            branch_goal=nonempty(row.get("branch_goal"), "react to one local detail"),
            allowed_functions=normalize_vocab_list(row.get("allowed_functions"), COMMENT_FUNCTIONS, ("reaction",)),
            evidence_modes=normalize_vocab_list(row.get("evidence_modes"), EVIDENCE_MODES, ("none_assertion",)),
            tone_palette=normalize_vocab_list(row.get("tone_palette"), VOICE_MODES, ("casual_neutral",)),
            story_modes=normalize_vocab_list(row.get("story_modes"), STORY_MODES, ("no_story",)),
            content_angles=normalize_vocab_list(row.get("content_angles"), CONTENT_ANGLES, ("unclear_mixed",)),
            perspective_id=nonempty(row.get("perspective_id"), "seed_local"),
            decision_boundary=nonempty(
                row.get("decision_boundary"),
                "one branch-local decision condition",
            ),
            branch_exclusion=nonempty(
                row.get("branch_exclusion"),
                "do not repeat another branch's decision axis",
            ),
            owned_decision_subject=nonempty(
                row.get("owned_decision_subject"),
                nonempty(
                    row.get("decision_boundary"),
                    "one branch-local decision condition",
                ),
            ),
        )
        branches.append(branch)
    # A branch owns one independently rooted discussion chain. Reusing a small
    # fixed branch pool across many roots makes unrelated chains inherit the
    # same decision subject, producing semantic and lexical repetition before
    # the Writer is called. Replies still share their parent's root branch.
    min_branches = min(
        max(3, target.top_level_comments),
        max(3, target.target_comments),
    )
    if len(branches) < min_branches:
        raise ValueError(f"not enough branch plans: {len(branches)} < {min_branches}")
    return branches[: max(3, target.target_comments)]


def plan_comment_moves(
    *,
    client: Any,
    model: str,
    seed_post: SeedPost,
    target: ThreadTarget,
    branches: list[BranchPlan],
    matched_real_thread: dict[str, Any] | None,
    matched_real_comments: int,
    batch_size: int,
    retries: int,
    max_tokens: int,
) -> dict[int, dict[str, str]]:
    """Extract one semantic writing plan per matched real comment.

    The matched real comments are allowed to inform the GPT planner, but the
    resulting plan intentionally contains no real text.  Qwen receives only this
    abstract plan plus the visible seed/parent context.
    """

    if not matched_real_thread:
        return {}
    comments = selected_matched_comments(
        matched_real_thread=matched_real_thread,
        target=target,
        matched_real_comments=matched_real_comments,
    )
    if not comments:
        return {}
    if batch_size <= 0:
        batch_size = len(comments)
    # A depth stage receives all available roots or replies together, but a
    # child stage is planned only after every parent has a committed ledger
    # entry. This keeps one-shot Planner control while making a reply delta a
    # genuine addition rather than a restated root claim.
    batch_size = min(batch_size, len(comments))
    merged: dict[int, dict[str, str]] = {}
    for start, end in planner_batch_ranges_by_depth(comments, batch_size=batch_size):
        batch = comments[start:end]
        batch_plans = plan_comment_move_batch(
            client=client,
            model=model,
            seed_post=seed_post,
            target=target,
            branches=branches,
            matched_real_thread=matched_real_thread,
            comments=batch,
            all_comments=comments,
            sample_offset=start,
            retries=retries,
            max_tokens=max_tokens,
        )
        merged.update(batch_plans)
    return merged


def plan_comment_move_batch(
    *,
    client: Any,
    model: str,
    seed_post: SeedPost,
    target: ThreadTarget,
    branches: list[BranchPlan],
    matched_real_thread: dict[str, Any],
    comments: list[dict[str, Any]],
    all_comments: list[dict[str, Any]],
    sample_offset: int,
    retries: int,
    max_tokens: int,
) -> dict[int, dict[str, str]]:
    prompt = build_comment_move_planner_prompt(
        seed_post=seed_post,
        target=target,
        branches=branches,
        matched_real_thread=matched_real_thread,
        comments=comments,
        all_comments=all_comments,
        sample_offset=sample_offset,
    )
    last_error = ""
    for _attempt in range(max(1, retries + 1)):
        final_prompt = prompt
        if last_error:
            final_prompt += (
                "\n\nPrevious response was invalid for this reason:\n"
                f"{last_error}\nReturn corrected JSON only."
            )
        raw = chat_completion_text(
            client=client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract abstract semantic writing plans from real Reddit comments. "
                        "Return strict JSON only. Do not write final comments."
                    ),
                },
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.70,
            max_tokens=max_tokens,
            response_format_json=True,
        )
        try:
            payload = parse_json_object(raw)
            plans = normalize_comment_move_plans(payload, branches=branches)
            if plans:
                return plans
            last_error = "comment_plans was empty"
        except Exception as exc:  # noqa: BLE001 - planner retry feedback
            last_error = str(exc)
    batch_start = sample_offset + 1
    batch_end = sample_offset + len(comments)
    raise RuntimeError(f"Comment move planner failed for S{batch_start}-S{batch_end} after retries: {last_error}")


def normalize_comment_move_plans(
    payload: dict[str, Any],
    *,
    branches: list[BranchPlan],
) -> dict[int, dict[str, str]]:
    raw_plans = payload.get("comment_plans") or payload.get("plans") or []
    if not isinstance(raw_plans, list):
        raise ValueError("comment_plans must be a list")
    branch_ids = {branch.branch_id for branch in branches}
    result: dict[int, dict[str, str]] = {}
    for row in raw_plans:
        if not isinstance(row, dict):
            continue
        # The prompt names anonymous structural rows S1, S2, ... . Providers
        # commonly preserve that notation in JSON even when the schema example
        # displays an integer. Treat the prefix as transport syntax, never as a
        # semantic label, so a complete plan batch is not discarded.
        sample_value = str(row.get("sample_id") or "").strip()
        prefixed_sample = re.fullmatch(r"[sS]\s*(\d+)", sample_value)
        sample_id = (
            int(prefixed_sample.group(1))
            if prefixed_sample is not None
            else safe_int(row.get("sample_id"), -1)
        )
        if sample_id <= 0:
            continue
        branch_id = safe_int(row.get("branch_id"), 0)
        if branch_id not in branch_ids:
            branch_id = 0
        result[sample_id] = {
            "reference_id": compact(nonempty(row.get("reference_id"), "none"), 32),
            "branch_id": str(branch_id),
            "payload_type": normalize_vocab_value(row.get("payload_type"), PAYLOAD_TYPES, ""),
            "comment_function": normalize_vocab_value(row.get("comment_function"), COMMENT_FUNCTIONS, "reaction"),
            "content_angle": normalize_vocab_value(row.get("content_angle"), CONTENT_ANGLES, "unclear_mixed"),
            "evidence_mode": normalize_vocab_value(row.get("evidence_mode"), EVIDENCE_MODES, "none_assertion"),
            "story_mode": normalize_vocab_value(row.get("story_mode"), STORY_MODES, "no_story"),
            "voice": normalize_vocab_value(row.get("voice"), VOICE_MODES, ""),
            "speaker_role": normalize_vocab_value(row.get("speaker_role"), SPEAKER_ROLES, ""),
            "semantic_move": compact(nonempty(row.get("semantic_move"), "make one concrete local point"), 220),
            "local_topic": compact(nonempty(row.get("local_topic"), "one local thread detail"), 180),
            "reply_relation": compact(nonempty(row.get("reply_relation"), "adds_datapoint"), 80),
            "stance": compact(nonempty(row.get("stance"), "neutral"), 60),
            "detail_focus": compact(nonempty(row.get("detail_focus"), "one specific detail"), 220),
            "avoid_repeating": compact(nonempty(row.get("avoid_repeating"), "do not repeat generic nearby claims"), 220),
            "claim_family": normalize_vocab_value(row.get("claim_family"), CLAIM_FAMILIES, "miscellaneous"),
            "claim_key": normalize_claim_key(row.get("claim_key")),
            "perspective_id": compact(nonempty(row.get("perspective_id"), "seed_local"), 48),
            "domain_intent": compact(nonempty(row.get("domain_intent"), "one seed-grounded local move"), 180),
            "decision_boundary": compact(nonempty(row.get("decision_boundary"), "one local decision condition"), 180),
            "reply_delta": compact(nonempty(row.get("reply_delta"), ""), 180),
            "reply_delta_type": compact(nonempty(row.get("reply_delta_type"), ""), 48),
            "reply_novelty_anchor": compact(nonempty(row.get("reply_novelty_anchor"), ""), 180),
            "opening_style": compact(nonempty(row.get("opening_style"), "direct concrete point first"), 120),
            "context_aperture": normalize_vocab_value(row.get("context_aperture"), CONTEXT_APERTURES, "seed_gist_only"),
        }
    return result


def expand_plan_to_tasks(
    *,
    branches: list[BranchPlan],
    target: ThreadTarget,
    seed_post: SeedPost | None = None,
    matched_real_thread: dict[str, Any] | None = None,
    matched_real_comments: int = 0,
    comment_plans: dict[int, dict[str, str]] | None = None,
    rng: random.Random,
    context_dropout_rate: float = 0.0,
    context_jitter_rate: float = 0.0,
) -> list[CommentTask]:
    if target.target_comments <= 0:
        return []
    return expand_matched_real_sample_to_tasks(
        branches=branches,
        target=target,
        seed_post=seed_post,
        matched_real_thread=matched_real_thread,
        matched_real_comments=matched_real_comments,
        comment_plans=comment_plans or {},
        rng=rng,
        context_dropout_rate=context_dropout_rate,
        context_jitter_rate=context_jitter_rate,
    )


def expand_matched_real_sample_to_tasks(
    *,
    branches: list[BranchPlan],
    target: ThreadTarget,
    seed_post: SeedPost | None,
    matched_real_thread: dict[str, Any] | None,
    matched_real_comments: int,
    comment_plans: dict[int, dict[str, str]],
    rng: random.Random,
    context_dropout_rate: float = 0.0,
    context_jitter_rate: float = 0.0,
) -> list[CommentTask]:
    """Use the matched real thread's structural skeleton when it is available.

    The real text itself is not sent to the Qwen writer.  We only preserve
    comment order, parent linkage, depth tendency, and word-count bucket.
    """

    if not matched_real_thread:
        return []
    selected = selected_matched_comments(
        matched_real_thread=matched_real_thread,
        target=target,
        matched_real_comments=matched_real_comments,
    )
    if not selected:
        return []

    fullname_to_sample: dict[str, int] = {}
    for sample_id, row in enumerate(selected, start=1):
        for key in real_comment_keys(row):
            fullname_to_sample[key] = sample_id

    tasks: list[CommentTask] = []
    sample_to_task: dict[int, CommentTask] = {}
    branch_cycle = list(branches)
    rng.shuffle(branch_cycle)
    root_count = 0

    for sample_id, row in enumerate(selected, start=1):
        planned = comment_plans.get(sample_id) or {}
        parent_sample = fullname_to_sample.get(str(row.get("parent_id") or ""))
        parent_task = sample_to_task.get(parent_sample or -1)
        planned_branch_id = safe_int(planned.get("branch_id"), 0)
        if parent_task is None:
            branch = branch_by_id(branches, planned_branch_id) or branch_cycle[root_count % len(branch_cycle)]
            root_count += 1
            depth = 0
            visible_scope = "seed_post"
            anchor = planned.get("local_topic") or branch.anchor_quote
            intent = (
                f"Fill matched real sample slot S{sample_id}: top-level comment, "
                f"real_words={len(str(row.get('body') or '').split())}. "
                f"Semantic move: {planned.get('semantic_move') or 'make one concrete local point'}."
            )
            must_not_do = "Do not cover every issue in the OP. Do not write a balanced card review."
        else:
            branch = branch_by_id(branches, planned_branch_id) or branch_by_id(branches, parent_task.branch_id) or branches[0]
            depth = parent_task.depth + 1
            visible_scope = "parent_only"
            anchor = planned.get("local_topic") or f"matched real reply slot S{sample_id} replying to S{parent_sample}"
            intent = (
                f"Fill matched real sample slot S{sample_id}: reply to S{parent_sample}, "
                f"real_words={len(str(row.get('body') or '').split())}. "
                f"Semantic move: {planned.get('semantic_move') or 'add one new parent-local point'}."
            )
            # The parent's own text is already carried as a structured
            # exclusion ("parent contribution not to restate"). Repeating the
            # prohibition here as prose put the parent's wording in front of the
            # Writer three times in one prompt, each time labelled as forbidden.
            must_not_do = (
                "Do not re-answer the OP. Do not mention information outside "
                "the parent unless implied by the parent."
            )

        other_branch_subjects = "; ".join(
            item.owned_decision_subject
            for item in branches
            if item.branch_id != branch.branch_id and item.owned_decision_subject
        )
        parent_move = (
            str(parent_task.semantic_move or parent_task.local_topic or "").strip()
            if parent_task is not None
            else ""
        )
        parent_boundary = (
            str(parent_task.decision_boundary or "").strip()
            if parent_task is not None
            else ""
        )
        reply_delta = str(planned.get("reply_delta") or "").strip()
        reply_delta_type = str(planned.get("reply_delta_type") or "").strip()
        reply_novelty_anchor = str(planned.get("reply_novelty_anchor") or "").strip()
        if reply_novelty_anchor.casefold() in {"none", "n/a", "na"}:
            reply_novelty_anchor = ""
        reply_delta_is_fallback = False
        if parent_task is not None and reply_delta.casefold() in {"", "none", "n/a", "na"}:
            reply_delta = default_reply_delta(
                str(planned.get("reply_relation") or "")
            )
            reply_delta_is_fallback = True
        planned_semantic_move = str(planned.get("semantic_move") or "").strip()
        planned_decision_boundary = str(planned.get("decision_boundary") or "").strip()
        # A direct reply may share a root decision subject, but it must realize
        # only the Planner's named increment. This routes an already planned
        # contract; it does not create or rank alternate Writer candidates.
        #
        # The delta and the novelty anchor reach the Writer as their own rendered
        # fields, so they do not need to displace the move. They used to: a
        # reply's semantic_move was replaced by its anchor, which is a bare noun
        # phrase. Measured over v70 that cut a reply's semantic contract from the
        # 21.4-word proposition a top-level slot gets to 7.7 words, for 61 of 61
        # replies and 45% of the thread.
        realized_semantic_move = planned_semantic_move
        realized_decision_boundary = planned_decision_boundary
        if parent_task is not None and reply_delta and not reply_delta_is_fallback:
            intent = (
                f"Fill matched real sample slot S{sample_id}: reply to S{parent_sample}, "
                f"real_words={len(str(row.get('body') or '').split())}. "
                f"Semantic move: {planned_semantic_move or reply_delta}. "
                f"New reply increment: {reply_novelty_anchor or reply_delta}."
            )
        # `avoid_repeating` is the Planner's own list of things this slot should
        # not re-cover. The parent contribution used to be appended here as a
        # second copy of an exclusion the semantic contract already renders.
        avoid_repeating = str(planned.get("avoid_repeating") or "").strip()

        body_words = len(str(row.get("body") or "").split())
        real_body = str(row.get("body") or "")
        function = planned.get("comment_function") or first_choice(branch.allowed_functions, "reaction")
        evidence = planned.get("evidence_mode") or first_choice(branch.evidence_modes, "none_assertion")
        payload_type = planned.get("payload_type") or infer_payload_type(
            word_count=body_words,
            comment_function=function,
            reply_relation=planned.get("reply_relation") or "",
            stance=planned.get("stance") or "",
        )
        voice = voice_for_payload(
            planned_voice=planned.get("voice") or "",
            payload_type=payload_type,
            comment_function=function,
            branch=branch,
        )
        story_mode = planned.get("story_mode") or first_choice(branch.story_modes, "no_story")
        angle = planned.get("content_angle") or first_choice(branch.content_angles, "unclear_mixed")
        context_aperture = planned.get("context_aperture") or ("parent_only" if parent_task is not None else "seed_gist_only")
        length_bucket = length_bucket_for_payload(payload_type=payload_type, word_count=body_words)
        speaker_role = planned.get("speaker_role") or infer_speaker_role(
            payload_type=payload_type,
            comment_function=function,
            voice=voice,
            reply_relation=planned.get("reply_relation") or "",
        )
        overrides = infer_real_comment_social_overrides(row, payload_type=payload_type, speaker_role=speaker_role)
        if overrides:
            payload_type = overrides.get("payload_type", payload_type)
            function = overrides.get("comment_function", function)
            evidence = overrides.get("evidence_mode", evidence)
            story_mode = overrides.get("story_mode", story_mode)
            voice = overrides.get("voice", voice)
            speaker_role = overrides.get("speaker_role", speaker_role)
            length_bucket = overrides.get("length_bucket", length_bucket)
        utterance_mode = infer_utterance_mode(
            payload_type=payload_type,
            speaker_role=speaker_role,
            comment_function=function,
            voice=voice,
            real_word_count=body_words,
        )
        surface_texture = infer_surface_texture(
            real_body,
            payload_type=payload_type,
            speaker_role=speaker_role,
            utterance_mode=utterance_mode,
        )
        real_surface_shape = infer_real_surface_shape(row)
        surface_skeleton, surface_instruction = infer_surface_skeleton(real_body)
        real_tone_slot, real_tone_instruction = infer_real_tone_slot(
            row,
            payload_type=payload_type,
            speaker_role=speaker_role,
            voice=voice,
            real_surface_shape=real_surface_shape,
            surface_texture=surface_texture,
        )
        surface_updates = overrides_for_real_surface_shape(
            shape=real_surface_shape,
            word_count=body_words,
            has_parent=parent_task is not None,
        )
        if surface_updates:
            payload_type = surface_updates.get("payload_type", payload_type)
            function = surface_updates.get("comment_function", function)
            evidence = surface_updates.get("evidence_mode", evidence)
            story_mode = surface_updates.get("story_mode", story_mode)
            voice = surface_updates.get("voice", voice)
            speaker_role = surface_updates.get("speaker_role", speaker_role)
            length_bucket = surface_updates.get("length_bucket", length_bucket)
            utterance_mode = surface_updates.get("utterance_mode", utterance_mode)
            surface_texture = surface_updates.get("surface_texture", surface_texture)
        allow_first_person_frame = real_text_allows_first_person_frame(real_body)
        allow_uncertainty_frame = real_text_allows_uncertainty_frame(real_body)
        concrete_anchors = build_concrete_anchors_for_task(
            real_body=real_body,
            seed_post=seed_post,
            branch=branch,
            planned=planned,
            anchor=anchor,
            parent_task=parent_task,
        )
        if payload_type in LOW_INFO_PAYLOAD_TYPES:
            evidence = "none_assertion"
            story_mode = "no_story"
        context_transform = choose_context_transform(
            rng=rng,
            has_parent=parent_task is not None,
            payload_type=payload_type,
            comment_function=function,
            reply_relation=planned.get("reply_relation") or "",
            context_aperture=context_aperture,
            dropout_rate=context_dropout_rate,
            jitter_rate=context_jitter_rate,
        )
        task = CommentTask(
            local_task_id=len(tasks) + 1,
            local_parent_task_id=None if parent_task is None else parent_task.local_task_id,
            depth=depth,
            branch_id=branch.branch_id,
            branch_goal=branch.branch_goal,
            branch_exclusion=branch.branch_exclusion,
            owned_decision_subject=branch.owned_decision_subject,
            forbidden_decision_subjects=other_branch_subjects,
            visible_scope=visible_scope,
            local_anchor=anchor,
            comment_function=function,
            content_angle=angle,
            evidence_mode=evidence,
            story_mode=story_mode,
            voice=voice,
            payload_type=payload_type,
            length_bucket=length_bucket,
            speaker_role=speaker_role,
            utterance_mode=utterance_mode,
            surface_texture=surface_texture,
            allow_first_person_frame=allow_first_person_frame,
            allow_uncertainty_frame=allow_uncertainty_frame,
            planner_intent=intent,
            must_not_do=must_not_do,
            real_sample_id=sample_id,
            real_parent_sample_id=parent_sample,
            real_word_count=body_words,
            semantic_move=realized_semantic_move,
            local_topic=planned.get("local_topic") or "",
            reply_relation=planned.get("reply_relation") or "",
            stance=planned.get("stance") or "",
            detail_focus=planned.get("detail_focus") or "",
            avoid_repeating=avoid_repeating,
            claim_key=planned.get("claim_key") or "",
            claim_family=planned.get("claim_family") or "miscellaneous",
            perspective_id=planned.get("perspective_id") or "seed_local",
            domain_intent=planned.get("domain_intent") or "",
            decision_boundary=realized_decision_boundary,
            reply_delta=reply_delta,
            reply_delta_type=reply_delta_type,
            reply_novelty_anchor=reply_novelty_anchor,
            parent_semantic_move=parent_move,
            parent_decision_boundary=parent_boundary,
            opening_style=planned.get("opening_style") or "",
            development_plan=planned.get("development_plan") or "",
            context_aperture=context_aperture,
            context_transform=context_transform,
            real_surface_shape=real_surface_shape,
            surface_skeleton=surface_skeleton,
            surface_instruction=surface_instruction,
            real_tone_slot=real_tone_slot,
            real_tone_instruction=real_tone_instruction,
            concrete_anchors=concrete_anchors,
        )
        tasks.append(task)
        sample_to_task[sample_id] = task
    return tasks


def infer_real_surface_shape(row: dict[str, Any]) -> str:
    body = str(row.get("body") or "").strip()
    lowered = body.lower()
    words = len(body.split())
    author = str(row.get("author") or "").lower()

    if lowered in {"[deleted]", "[removed]"}:
        return "deleted_removed"
    if author in {"automoderator", "moderator"} or "!template" in lowered or "template for card recommendation" in lowered:
        return "template_notice"
    if re.search(r"https?://|www\.", body, re.IGNORECASE):
        return "quote_link_reference" if re.search(r"\[[^\]]+\]\(", body) or ">" in body else "link_reference"
    if ">" in body or "&gt;" in lowered or re.search(r"\[[^\]]+\]\(", body):
        return "quote_link_reference"
    if is_gratitude_text(body):
        return "thanks_ack"
    if re.search(r"\b(lol|lmao|haha)\b|/s|[\U0001F300-\U0001FAFF]", body, re.IGNORECASE):
        return "joke_reaction"
    if words <= 5:
        return "micro_reaction"
    if "?" in body and words <= 18:
        return "short_question"
    if words <= 10:
        return "short_direct_answer"
    if re.search(r"\b(CFU|CFF|CSR|SUB|AF|CLI|HUCA|PC|DP|USBAR|BCP|BCE)\b|5/24|1/12", body):
        return "compact_datapoint"
    if "?" in body and words <= 32:
        return "short_question"
    if re.search(r"\b(no|nope|wrong|incorrect|not true|doesn't|does not|can't|cannot)\b", lowered) and words <= 45:
        return "parent_grounded_correction"
    if words >= 70 or re.search(r"\b(i had|i've|ive|my|called|applied|denied|approved|customer service|fraud)\b", lowered):
        return "story_rant"
    if re.search(r"\b(side note|also|unrelated|fwiw|btw)\b", lowered):
        return "side_tangent"
    return "full_answer"


def infer_surface_skeleton(text: str) -> tuple[str, str]:
    """Return a domain-light sentence-shape skeleton from a real Reddit comment.

    This is intentionally not a semantic label.  The writer should preserve the
    rough surface form while replacing the actual content with the sampled local
    point, which targets Self-BERT phrase diversity more directly than another
    payload label.
    """

    body = str(text or "").strip()
    compacted = re.sub(r"\s+", " ", body)
    lowered = compacted.lower()
    words = compacted.split()
    word_count = len(words)
    sentence_count = max(1, len(re.findall(r"[.!?]+(?:\s|$)", compacted)) or len(re.split(r"\s{2,}", body)))
    has_quote = bool(re.search(r"(^|\n)\s*>|&gt;|\[[^\]]+\]\(", body))
    has_link = bool(re.search(r"https?://|www\.", body, re.IGNORECASE))
    has_thanks = is_gratitude_text(body)
    has_joke = bool(re.search(r"\b(lol|lmao|haha)\b|/s|[\U0001F300-\U0001FAFF]", body, re.IGNORECASE))
    has_question = "?" in compacted
    has_first = bool(re.search(r"\b(i|i'm|i’ve|i've|ive|my|mine|we|us)\b", lowered))
    has_blunt = bool(re.search(r"^(no|nope|yes|yep|depends|hard pass|wrong|not worth)\b", lowered))
    has_tiny_blunt = bool(re.search(r"^(same|this)\b", lowered)) and word_count <= 8
    has_ellipsis = "..." in compacted or "…" in compacted
    has_parenthetical = "(" in compacted and ")" in compacted

    if lowered in {"[deleted]", "[removed]"}:
        return ("deleted placeholder", "Use a deleted/removed placeholder shape.")
    if has_link or has_quote:
        return (
            "quote/link reference plus short reaction",
            "Use a quote/link/reference-style surface. Do not expand it into a complete explanation.",
        )
    if word_count <= 5:
        return (
            "tiny fragment reaction",
            "Use one tiny fragment, roughly one to five words. No explanation.",
        )
    if has_thanks and word_count <= 20:
        return (
            "brief thanks or acknowledgement",
            "Use a brief social acknowledgement. Do not add a new advice paragraph.",
        )
    if has_joke and word_count <= 28:
        return (
            "joke or sarcastic aside",
            "Use a joke/sarcastic aside as the whole comment. Do not explain it.",
        )
    if has_question and word_count <= 18:
        return (
            "single narrow question",
            "Ask one narrow question only. Do not answer it yourself.",
        )
    if (has_blunt or has_tiny_blunt) and word_count <= 18:
        return (
            "blunt short verdict plus optional fragment",
            "Use a blunt short verdict, optionally followed by one small fragment.",
        )
    if has_first and word_count >= 45:
        return (
            f"messy first-person datapoint across {min(sentence_count, 4)} short chunks",
            "Use a first-person datapoint/rant shape with imperfect Reddit pacing, not a polished answer.",
        )
    if has_question:
        return (
            "question embedded in a local reply",
            "Keep the question as part of the surface form, but do not turn the whole comment into advice.",
        )
    if has_ellipsis:
        return (
            "elliptical aside with incomplete pacing",
            "Use ellipses or an unfinished aside shape. Avoid clean essay structure.",
        )
    if has_parenthetical:
        return (
            "compact answer with parenthetical aside",
            "Use one compact answer with a parenthetical aside or abbreviation.",
        )
    if word_count <= 10:
        return (
            "short direct answer",
            "Use a short direct answer with no extra setup.",
        )
    if word_count >= 80:
        return (
            f"long uneven Reddit paragraph with {min(sentence_count, 5)} moves",
            "Use a longer uneven Reddit paragraph with concrete details and imperfect pacing; avoid generic helpful structure.",
        )
    return (
        f"{min(sentence_count, 3)}-sentence local comment",
        "Use the same rough sentence count and pacing, but replace the actual content with the sampled local point.",
    )


CONCRETE_TERM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bSUB\b|sign[- ]?up bonus|signup bonus", "SUB / signup bonus"),
    (r"\bAF\b|annual fee", "annual fee / AF"),
    (r"\bFTF\b|foreign transaction", "foreign transaction fee / FTF"),
    (r"\bCLI\b|credit limit increase", "CLI / credit limit increase"),
    (r"\bprequal(?:ification)?\b|prequalified|preselected", "prequal / preselected offer"),
    (r"\butilization\b|statement balance|report(?:ed|ing)? balance", "utilization / statement balance"),
    (r"\bsecured card\b|secured credit card|graduate to unsecured|graduation", "secured card / graduation"),
    (r"\btransfer partner|transfer partners|UR points|MR points|miles\b", "points / transfer partners"),
    (r"\bGlobal Entry|Priority Pass|trip delay|travel insurance", "travel benefit / insurance"),
    (r"\bportal\b|hotel credit|resort credit|statement credit", "portal / statement credit"),
    (r"\bhard pull\b|soft pull\b|inquiry|5/24", "hard pull / 5/24"),
    (r"\bmailers?\b|targeted offer|offer in the mail", "targeted mailer DP"),
    (r"\bchecking account|credit union|local bank", "banking relationship / credit union"),
)


def extract_concrete_anchors(text: str, *, source_label: str = "", max_items: int = 12) -> list[str]:
    body = str(text or "")
    if not body.strip():
        return []
    anchors: list[str] = []
    anchors.extend(extract_url_anchor_labels(body))
    anchors.extend(extract_money_number_anchors(body))
    anchors.extend(extract_product_anchors(body))
    anchors.extend(extract_term_anchors(body))
    anchors.extend(extract_short_dp_anchors(body))
    cleaned = dedup_anchors(anchors, max_anchors=max_items)
    if source_label and cleaned:
        return [f"{item} ({source_label})" for item in cleaned]
    return cleaned


def finalize_rebalanced_task(task: CommentTask) -> CommentTask:
    if is_surface_restyled_answer(task):
        utterance_mode = task.utterance_mode or infer_utterance_mode(
            payload_type=task.payload_type,
            speaker_role=task.speaker_role,
            comment_function=task.comment_function,
            voice=task.voice,
            real_word_count=safe_int(task.real_word_count, 0),
        )
        surface_texture = task.surface_texture
    else:
        utterance_mode = infer_utterance_mode(
            payload_type=task.payload_type,
            speaker_role=task.speaker_role,
            comment_function=task.comment_function,
            voice=task.voice,
            real_word_count=safe_int(task.real_word_count, 0),
        )
        surface_texture = surface_texture_for_task(task, utterance_mode=utterance_mode)
    provisional = replace(task, utterance_mode=utterance_mode, surface_texture=surface_texture)
    tone_shape = (
        task.tone_shape
        if task.tone_shape in TONE_SHAPES and tone_shape_is_compatible(task.tone_shape, provisional)
        else infer_tone_shape_for_task(provisional)
    )
    return replace(
        task,
        utterance_mode=utterance_mode,
        surface_texture=surface_texture,
        tone_shape=tone_shape,
        allow_first_person_frame=task.allow_first_person_frame
        and (
            task.speaker_role in {"datapoint_only", "gratitude_reply", "op_followup"}
            or task.tone_target == "constructive_polite_helpful"
        ),
        allow_uncertainty_frame=task.allow_uncertainty_frame
        and (
            task.speaker_role in {"confused_asker", "op_followup"}
            or task.tone_target == "constructive_polite_helpful"
        ),
    )


def generate_post_from_tasks(
    *,
    writer_client: Any,
    writer_model: str,
    writer_profile: str,
    seed_post: SeedPost,
    tasks: list[CommentTask],
    run_index: int,
    post_slot: int,
    seed_index: int,
    rng: random.Random,
    max_writer_tokens: int,
    writer_retries: int,
    claim_key_budget: int,
    claim_family_max_share: float,
    claim_family_min_budget: int,
    opening_reuse_budget: int,
    opener_family_reuse_budget: int,
    template_phrase_reuse_budget: int,
) -> dict[str, Any]:
    actual_by_task: dict[int, dict[str, Any]] = {}
    root_comments: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    generated_comments: list[dict[str, Any]] = []
    seen_exact_texts: set[str] = set()
    recent_openings: list[str] = []
    claim_counts: dict[str, int] = {}
    claim_family_counts: dict[str, int] = {}
    claim_family_budget = claim_family_budget_for_total(
        len(tasks),
        max_share=claim_family_max_share,
        min_budget=claim_family_min_budget,
    )
    opening_counts: dict[str, int] = {}
    opener_family_counts: dict[str, int] = {}
    template_phrase_counts: dict[str, int] = {}
    task_by_id = {task.local_task_id: task for task in tasks}
    base_ts = datetime.now(timezone.utc) - timedelta(hours=rng.randint(3, 72))

    for task in tasks:
        original_task = task
        task, parent_comment = align_task_to_generated_parent(
            task,
            task_by_id=task_by_id,
            actual_by_task=actual_by_task,
        )
        if task.claim_key and claim_key_budget > 0 and claim_counts.get(task.claim_key, 0) >= claim_key_budget:
            task = backfill_repeated_claim_task(task, reason="claim_key_budget")
        if (
            task.claim_family
            and task.claim_family != "miscellaneous"
            and claim_family_budget > 0
            and claim_family_counts.get(task.claim_family, 0) >= claim_family_budget
        ):
            task = backfill_repeated_claim_task(task, reason="claim_family_budget")
        writer_result = generate_writer_text_with_guards(
            writer_client=writer_client,
            writer_model=writer_model,
            writer_profile=writer_profile,
            seed_post=seed_post,
            task=task,
            parent_comment=parent_comment,
            seen_exact_texts=seen_exact_texts,
            recent_openings=recent_openings,
            previous_comments=generated_comments,
            opening_counts=opening_counts,
            opening_reuse_budget=opening_reuse_budget,
            opener_family_counts=opener_family_counts,
            opener_family_reuse_budget=opener_family_reuse_budget,
            template_phrase_counts=template_phrase_counts,
            template_phrase_reuse_budget=template_phrase_reuse_budget,
            max_writer_tokens=max_writer_tokens,
            writer_retries=writer_retries,
        )
        if writer_result["skip"]:
            records.append(
                {
                    "task": task_to_dict(task),
                    "prompt": writer_result.get("prompt", ""),
                    "raw": writer_result.get("raw", ""),
                    "comment": None,
                    "attempts": writer_result.get("attempts", []),
                    "skipped": True,
                    "skip_reason": writer_result.get("skip_reason", "writer_guard_failed"),
                }
            )
            continue
        text = str(writer_result["text"])
        raw = str(writer_result["raw"])
        prompt = str(writer_result["prompt"])
        opener_family = opener_family_signature(text)
        template_phrase_family = template_phrase_signature(text)
        comment_id = run_index * 100000 + post_slot * 10000 + task.local_task_id
        comment = {
            "comment_id": comment_id,
            "parent_comment_id": parent_comment.get("comment_id") if parent_comment else None,
            "author": f"sampled_user_{run_index}_{post_slot}_{task.local_task_id}",
            "author_karma": int(rng.randint(12, 84000)),
            "content": text,
            "created_at": (base_ts + timedelta(minutes=task.local_task_id * rng.randint(2, 19))).isoformat(),
            "timestamp": (base_ts + timedelta(minutes=task.local_task_id * rng.randint(2, 19))).isoformat(),
            "likes": int(max(0, rng.gauss(8, 22))),
            "depth": task.depth,
            "replies": [],
            "visibility_scope": task.visible_scope,
            "comment_job": task.comment_function,
            "comment_function": task.comment_function,
            "content_angle": task.content_angle,
            "evidence_mode": task.evidence_mode,
            "story_mode": task.story_mode,
            "story_instruction": task.story_instruction,
            "affect_role": task.affect_role,
            "affect_instruction": task.affect_instruction,
            "distribution_assignment": task.distribution_assignment,
            "tone_target": task.tone_target or task.voice,
            "tone_target_instruction": task.tone_target_instruction,
            "voice": task.voice,
            "payload_type": task.payload_type,
            "length_bucket": task.length_bucket,
            "speaker_role": task.speaker_role,
            "tone_shape": resolved_tone_shape(task),
            "utterance_mode": task.utterance_mode,
            "surface_texture": task.surface_texture,
            "real_surface_shape": task.real_surface_shape,
            "surface_skeleton": task.surface_skeleton,
            "surface_instruction": task.surface_instruction,
            "real_tone_slot": task.real_tone_slot,
            "real_tone_instruction": task.real_tone_instruction,
            "tone_overlay_slot": task.tone_overlay_slot,
            "tone_overlay_instruction": task.tone_overlay_instruction,
            "concrete_anchors": list(task.concrete_anchors),
            "allow_first_person_frame": task.allow_first_person_frame,
            "allow_uncertainty_frame": task.allow_uncertainty_frame,
            "branch_id": task.branch_id,
            "branch_goal": task.branch_goal,
            "branch_exclusion": task.branch_exclusion,
            "owned_decision_subject": task.owned_decision_subject,
            "forbidden_decision_subjects": task.forbidden_decision_subjects,
            "local_anchor": task.local_anchor,
            "planner_intent": task.planner_intent,
            "semantic_move": task.semantic_move,
            "local_topic": task.local_topic,
            "reply_relation": task.reply_relation,
            "stance": task.stance,
            "detail_focus": task.detail_focus,
            "avoid_repeating": task.avoid_repeating,
            "claim_key": task.claim_key,
            "claim_family": task.claim_family,
            "perspective_id": task.perspective_id,
            "domain_intent": task.domain_intent,
            "decision_boundary": task.decision_boundary,
            "reply_delta": task.reply_delta,
            "reply_delta_type": task.reply_delta_type,
            "reply_novelty_anchor": task.reply_novelty_anchor,
            "parent_semantic_move": task.parent_semantic_move,
            "parent_decision_boundary": task.parent_decision_boundary,
            "opening_style": task.opening_style,
            "context_aperture": task.context_aperture,
            "context_transform": task.context_transform,
            "opener_family": opener_family,
            "template_phrase_family": template_phrase_family,
            "word_count": len(text.split()),
            "generator": GENERATOR_NAME,
            "writer_attempts": len(writer_result.get("attempts") or []),
            "guard_degraded": bool(writer_result.get("degraded")),
            "degraded_from": writer_result.get("degraded_from", []),
        }
        seen_exact_texts.add(normalize_exact(text))
        opening = opening_signature(text)
        opening_guard = opening_guard_signature(text)
        if opening_guard:
            opening_counts[opening_guard] = opening_counts.get(opening_guard, 0) + 1
        if opener_family:
            opener_family_counts[opener_family] = opener_family_counts.get(opener_family, 0) + 1
        if template_phrase_family:
            template_phrase_counts[template_phrase_family] = template_phrase_counts.get(template_phrase_family, 0) + 1
        # Every opening already used in this thread stays visible. The window
        # used to keep the last 18, so on a long thread the Writer could reuse an
        # opening it had already used earlier and see no conflict; v70 threads
        # reached 44 prior comments against that 18-line ledger. The renderer
        # downstream is what decides how many to show, and it dedupes.
        recent_openings.append(opening)
        if task.claim_key:
            claim_counts[task.claim_key] = claim_counts.get(task.claim_key, 0) + 1
        if task.claim_family:
            claim_family_counts[task.claim_family] = claim_family_counts.get(task.claim_family, 0) + 1
        actual_by_task[task.local_task_id] = comment
        generated_comments.append(comment)
        if parent_comment:
            parent_comment.setdefault("replies", []).append(comment)
        else:
            root_comments.append(comment)
        records.append(
            {
                "task": task_to_dict(task),
                "prompt": prompt,
                "raw": raw,
                "comment": comment,
                "attempts": writer_result.get("attempts", []),
                "backfilled_from": task_to_dict(original_task) if task != original_task else None,
            }
        )

    post_id = f"sampled_run{run_index:02d}_post{post_slot:02d}_seed{seed_index:03d}"
    return {
        "post_id": post_id,
        "seed_index": seed_index,
        "source_raw_post_id": seed_post.source_raw_post_id,
        "title": seed_post.title,
        "content": seed_post.content,
        "author": f"sampled_op_{run_index}_{post_slot}",
        "author_karma": int(rng.randint(25, 55000)),
        "timestamp": base_ts.isoformat(),
        "likes": int(max(1, rng.gauss(35, 95))),
        "comments": root_comments,
        "thread_plan": {
            "target_comments": len(tasks),
            "max_depth_goal": max((task.depth for task in tasks), default=0),
            "branch_count": len({task.branch_id for task in tasks}),
            "claim_family_budget": claim_family_budget,
            "generator": GENERATOR_NAME,
        },
        "generation_records": records,
    }


def backfill_repeated_claim_task(task: CommentTask, *, reason: str) -> CommentTask:
    perspective = task.perspective_id or "seed_local"
    return replace(
        task,
        claim_key=f"{task.claim_key or 'local_claim'}__slot_{task.local_task_id}",
        claim_family="miscellaneous",
        semantic_move=(
            f"Make a distinct seed- or parent-grounded move for {perspective}; "
            "do not restate the saturated claim."
        ),
        avoid_repeating=(
            f"The original {reason} was saturated; choose another visible local detail "
            "without adding a new fact."
        ),
    )


def generate_writer_text_with_guards(
    *,
    writer_client: Any,
    writer_model: str,
    writer_profile: str,
    seed_post: SeedPost,
    task: CommentTask,
    parent_comment: dict[str, Any] | None,
    seen_exact_texts: set[str],
    recent_openings: list[str],
    previous_comments: list[dict[str, Any]],
    opening_counts: dict[str, int],
    opening_reuse_budget: int,
    opener_family_counts: dict[str, int],
    opener_family_reuse_budget: int,
    template_phrase_counts: dict[str, int],
    template_phrase_reuse_budget: int,
    max_writer_tokens: int,
    writer_retries: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    retry_note = ""
    total_attempts = max(1, writer_retries + 1)
    last_prompt = ""
    last_raw = ""
    last_text = ""
    last_problems: list[str] = []
    for attempt_idx in range(total_attempts):
        prompt = build_writer_prompt(
            profile=writer_profile,
            seed_post=seed_post,
            task=task,
            parent_comment=parent_comment,
            recent_openings=recent_openings,
            previous_comments=previous_comments,
            retry_note=retry_note,
        )
        raw = chat_completion_text(
            client=writer_client,
            model=writer_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[writer_profile]},
                {"role": "user", "content": prompt},
            ],
            temperature=writer_temperature(task, profile=writer_profile, attempt_idx=attempt_idx),
            max_tokens=writer_token_cap(
                task.length_bucket,
                payload_type=writer_payload_for_token_cap(task),
                profile=writer_profile,
                max_writer_tokens=max_writer_tokens,
            ),
            response_format_json=False,
            extra_body=writer_extra_body(writer_profile),
        )
        text = shape_writer_text_for_task(sanitize_writer_text(raw), task)
        problems = validate_writer_text(
            text=text,
            task=task,
            parent_comment=parent_comment,
            seen_exact_texts=seen_exact_texts,
            previous_comments=previous_comments,
            opening_counts=opening_counts,
            opening_reuse_budget=opening_reuse_budget,
            opener_family_counts=opener_family_counts,
            opener_family_reuse_budget=opener_family_reuse_budget,
            template_phrase_counts=template_phrase_counts,
            template_phrase_reuse_budget=template_phrase_reuse_budget,
            writer_profile=writer_profile,
        )
        attempts.append(
            {
                "attempt": attempt_idx + 1,
                "word_count": len(text.split()),
                "problems": problems,
                "text": text,
                "raw": raw,
                "prompt": prompt,
            }
        )
        last_prompt = prompt
        last_raw = raw
        last_text = text
        last_problems = problems
        if not problems:
            return {
                "skip": False,
                "text": text,
                "raw": raw,
                "prompt": prompt,
                "attempts": attempts,
            }
        retry_note = retry_note_for_problems(problems, task)

    if has_blocking_guard_failure(last_problems):
        degraded = generate_degraded_writer_text_after_guard_failure(
            writer_client=writer_client,
            writer_model=writer_model,
            writer_profile=writer_profile,
            seed_post=seed_post,
            task=task,
            parent_comment=parent_comment,
            seen_exact_texts=seen_exact_texts,
            recent_openings=recent_openings,
            previous_comments=previous_comments,
            opening_counts=opening_counts,
            opening_reuse_budget=opening_reuse_budget,
            opener_family_counts=opener_family_counts,
            opener_family_reuse_budget=opener_family_reuse_budget,
            template_phrase_counts=template_phrase_counts,
            template_phrase_reuse_budget=template_phrase_reuse_budget,
            max_writer_tokens=max_writer_tokens,
            previous_attempts=attempts,
            previous_problems=last_problems,
        )
        if degraded is not None:
            return degraded
        return {
            "skip": True,
            "text": last_text,
            "raw": last_raw,
            "prompt": last_prompt,
            "attempts": attempts,
            "skip_reason": ",".join(last_problems),
        }
    return {
        "skip": False,
        "text": last_text,
        "raw": last_raw,
        "prompt": last_prompt,
        "attempts": attempts,
    }


def has_blocking_guard_failure(problems: list[str]) -> bool:
    return any(
        problem
        in {
            "exact_duplicate",
            "parent_copy",
            "opener_family_reused",
            "template_phrase_reused",
            "first_person_frame_unwanted",
            "uncertainty_frame_unwanted",
            "question_mark_unwanted",
            "placeholder_literal",
            "planner_skeleton_residue",
            "meta_template_quote_heading",
            "long_helpful_too_generic",
            "missing_concrete_anchor",
            "real_slot_too_short",
        }
        for problem in problems
    )


def generate_degraded_writer_text_after_guard_failure(
    *,
    writer_client: Any,
    writer_model: str,
    writer_profile: str,
    seed_post: SeedPost,
    task: CommentTask,
    parent_comment: dict[str, Any] | None,
    seen_exact_texts: set[str],
    recent_openings: list[str],
    previous_comments: list[dict[str, Any]],
    opening_counts: dict[str, int],
    opening_reuse_budget: int,
    opener_family_counts: dict[str, int],
    opener_family_reuse_budget: int,
    template_phrase_counts: dict[str, int],
    template_phrase_reuse_budget: int,
    max_writer_tokens: int,
    previous_attempts: list[dict[str, Any]],
    previous_problems: list[str],
) -> dict[str, Any] | None:
    if any(problem in {"exact_duplicate", "parent_copy"} for problem in previous_problems):
        return None
    degraded_task = degraded_task_for_guard_failure(task, previous_problems)
    retry_note = retry_note_for_problems(previous_problems, degraded_task) + guard_fallback_retry_note(previous_problems)
    prompt = build_writer_prompt(
        profile=writer_profile,
        seed_post=seed_post,
        task=degraded_task,
        parent_comment=parent_comment,
        recent_openings=recent_openings,
        previous_comments=previous_comments,
        retry_note=retry_note,
    )
    raw = chat_completion_text(
        client=writer_client,
        model=writer_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS[writer_profile]},
            {"role": "user", "content": prompt},
        ],
        temperature=min(1.05, writer_temperature(degraded_task, profile=writer_profile, attempt_idx=1) + 0.04),
        max_tokens=writer_token_cap(
            degraded_task.length_bucket,
            payload_type=writer_payload_for_token_cap(degraded_task),
            profile=writer_profile,
            max_writer_tokens=max_writer_tokens,
        ),
        response_format_json=False,
        extra_body=writer_extra_body(writer_profile),
    )
    text = shape_writer_text_for_task(sanitize_writer_text(raw), degraded_task)
    problems = validate_writer_text(
        text=text,
        task=degraded_task,
        parent_comment=parent_comment,
        seen_exact_texts=seen_exact_texts,
        previous_comments=previous_comments,
        opening_counts=opening_counts,
        opening_reuse_budget=opening_reuse_budget,
        opener_family_counts=opener_family_counts,
        opener_family_reuse_budget=opener_family_reuse_budget,
        template_phrase_counts=template_phrase_counts,
        template_phrase_reuse_budget=template_phrase_reuse_budget,
        writer_profile=writer_profile,
    )
    attempts = list(previous_attempts) + [
        {
            "attempt": len(previous_attempts) + 1,
            "word_count": len(text.split()),
            "problems": problems,
            "text": text,
            "raw": raw,
            "prompt": prompt,
            "degraded": True,
        }
    ]
    if has_blocking_guard_failure(problems):
        return {
            "skip": True,
            "text": text,
            "raw": raw,
            "prompt": prompt,
            "attempts": attempts,
            "skip_reason": ",".join(problems),
            "degraded": True,
            "degraded_from": previous_problems,
            "degraded_task": task_to_dict(degraded_task),
        }
    return {
        "skip": False,
        "text": text,
        "raw": raw,
        "prompt": prompt,
        "attempts": attempts,
        "degraded": True,
        "degraded_from": previous_problems,
        "degraded_task": task_to_dict(degraded_task),
    }


def validate_writer_text(
    *,
    text: str,
    task: CommentTask,
    parent_comment: dict[str, Any] | None,
    seen_exact_texts: set[str],
    previous_comments: list[dict[str, Any]] | None = None,
    opening_counts: dict[str, int],
    opening_reuse_budget: int,
    opener_family_counts: dict[str, int],
    opener_family_reuse_budget: int,
    template_phrase_counts: dict[str, int],
    template_phrase_reuse_budget: int,
    writer_profile: str = "",
) -> list[str]:
    problems: list[str] = []
    normalized = normalize_exact(text)
    if not normalized:
        problems.append("empty")
    if normalized and normalized in seen_exact_texts:
        problems.append("exact_duplicate")
    if parent_comment is not None and looks_like_parent_copy(
        text_norm=normalized,
        parent_norm=normalize_exact(str(parent_comment.get("content") or "")),
    ):
        problems.append("parent_copy")
    if contains_writer_placeholder_literal(text):
        problems.append("placeholder_literal")
    if contains_planner_skeleton_residue(text, task):
        problems.append("planner_skeleton_residue")
    if writer_profile == "gpt54_reddit_writer" and is_meta_template_task(task) and str(text or "").lstrip().startswith(">"):
        problems.append("meta_template_quote_heading")
    if writer_profile == "gpt54_reddit_writer" and is_gpt_long_helpful_too_generic(text, task):
        problems.append("long_helpful_too_generic")
    if (
        writer_profile == "gpt54_reddit_writer"
        and task_requires_concrete_anchor_density(task)
        and not output_uses_visible_concrete_anchor(text, task)
    ):
        problems.append("missing_concrete_anchor")
    low_info_limit = low_info_word_limit(task)
    if low_info_limit and len(text.split()) > low_info_limit:
        problems.append("low_info_too_long")
    if length_too_long(len(text.split()), task.length_bucket, task):
        problems.append("length_too_long")
    if writer_profile == "gpt54_reddit_writer" and real_slot_too_short(len(text.split()), task):
        problems.append("real_slot_too_short")
    if writer_profile == "gpt54_reddit_writer":
        overlap_problem = lexical_overlap_problem(text=text, previous_comments=previous_comments, task=task)
        if overlap_problem:
            problems.append(overlap_problem)
    opening = opening_guard_signature(text)
    if opening and opening_reuse_budget > 0 and opening_counts.get(opening, 0) >= opening_reuse_budget:
        problems.append("opening_reused")
    opener_family = opener_family_signature(text)
    if (
        opener_family
        and opener_family_reuse_budget > 0
        and opener_family in CAPPED_OPENER_FAMILIES
        and opener_family_counts.get(opener_family, 0) >= opener_family_reuse_budget
    ):
        problems.append("opener_family_reused")
    template_phrase = template_phrase_signature(text)
    if template_phrase == "first_person_experience_frame" and not task.allow_first_person_frame:
        problems.append("first_person_frame_unwanted")
    if template_phrase == "uncertainty_frame" and not task.allow_uncertainty_frame:
        problems.append("uncertainty_frame_unwanted")
    if "?" in text and not task_allows_question_mark(task):
        problems.append("question_mark_unwanted")
    if (
        template_phrase
        and template_phrase_reuse_budget > 0
        and template_phrase in CAPPED_TEMPLATE_PHRASE_FAMILIES
        and template_phrase_counts.get(template_phrase, 0) >= template_phrase_reuse_budget
    ):
        problems.append("template_phrase_reused")
    return problems


def contains_planner_skeleton_residue(text: str, task: CommentTask) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    lowered = normalize_apostrophe_text(raw)
    compacted = re.sub(r"\s+", " ", lowered).strip(" .:;-")
    if compacted in PLANNER_RESIDUE_PATTERNS:
        return True
    if any(pattern in compacted for pattern in PLANNER_RESIDUE_PATTERNS):
        if len(compacted.split()) <= 8:
            return True
    skeleton = normalize_apostrophe_text(task.surface_skeleton or "")
    if skeleton and skeleton in compacted and len(compacted.split()) <= max(8, len(skeleton.split()) + 2):
        return True
    if re.fullmatch(r"(sub|sub|signup|sign up|bonus|af|apr|cli|dp)\s+acronym\s+explained", compacted):
        return True
    return False


def output_uses_visible_concrete_anchor(text: str, task: CommentTask) -> bool:
    if not task.concrete_anchors:
        return False
    output_keys = {
        concrete_anchor_key(anchor)
        for anchor in extract_concrete_anchors(str(text or ""), source_label="", max_items=30)
        if concrete_anchor_key(anchor)
    }
    task_keys = {
        concrete_anchor_key(anchor)
        for anchor in task.concrete_anchors
        if concrete_anchor_key(anchor)
    }
    if output_keys & task_keys:
        return True
    lowered = normalize_apostrophe_text(text)
    for anchor in task.concrete_anchors:
        base = concrete_anchor_base_label(anchor)
        if not base:
            continue
        key = concrete_anchor_key(base)
        if key and re.search(rf"\b{re.escape(key)}\b", lowered):
            return True
        tokens = concrete_anchor_tokens(base)
        if not tokens:
            continue
        numeric_tokens = [token for token in tokens if re.search(r"\d", token)]
        non_numeric_tokens = [token for token in tokens if not re.search(r"\d", token)]
        numeric_hit = any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)", lowered) for token in numeric_tokens)
        non_numeric_hits = sum(1 for token in non_numeric_tokens if re.search(rf"\b{re.escape(token)}\b", lowered))
        if numeric_tokens and numeric_hit and (not non_numeric_tokens or non_numeric_hits >= 1):
            return True
        if len(non_numeric_tokens) == 1 and re.search(rf"\b{re.escape(non_numeric_tokens[0])}\b", lowered):
            return True
        if len(non_numeric_tokens) >= 2 and non_numeric_hits >= min(2, len(non_numeric_tokens)):
            return True
    return False


def is_gpt_long_helpful_too_generic(text: str, task: CommentTask) -> bool:
    if not is_gpt_long_helpful_task(task):
        return False
    if len(str(text or "").split()) < 85:
        return False
    return not has_realistic_long_helpful_anchor(text, task=task)


def seed_post_gist(seed_post: SeedPost) -> str:
    title = compact(seed_post.title, 160)
    return (
        f"The OP is discussing a credit-card situation from this title: {title}. "
        "Possible reply branches include issuer rules, approval datapoints, credit-limit handling, "
        "utilization, rewards value, product comparisons, support/process details, jokes, and side tangents."
    )


def writer_token_cap(
    bucket: str,
    *,
    payload_type: str = "",
    profile: str = "",
    max_writer_tokens: int,
) -> int:
    if profile in {"osim8b_minimal_context", "osim8b_qwen_style"}:
        bucket_cap = 30
        if max_writer_tokens <= 0:
            return bucket_cap
        return max(12, min(max_writer_tokens, bucket_cap))
    if payload_type in TERSE_PAYLOAD_TYPES and bucket in {"micro", "short"}:
        bucket_cap = 22 if bucket == "micro" else 34
        if max_writer_tokens <= 0:
            return bucket_cap
        return max(12, min(max_writer_tokens, bucket_cap))
    if payload_type in LOW_INFO_PAYLOAD_TYPES and bucket in {"micro", "short", "medium"}:
        bucket_cap = 42 if bucket in {"micro", "short"} else 64
        if max_writer_tokens <= 0:
            return bucket_cap
        return max(12, min(max_writer_tokens, bucket_cap))
    bucket_cap = max_tokens_for_length(bucket)
    if max_writer_tokens <= 0:
        return bucket_cap
    return max(16, min(max_writer_tokens, bucket_cap))


def shape_writer_text_for_task(text: str, task: CommentTask) -> str:
    shaped = str(text or "").strip()
    if task.real_surface_shape == "deleted_removed":
        return "[deleted]" if task.local_task_id % 2 else "[removed]"
    if task.real_surface_shape == "micro_reaction" and len(shaped.split()) > 6:
        micro_options = ("Nope", "Same", "This", "Yep", "Hard pass", "Fair")
        shaped = micro_options[task.local_task_id % len(micro_options)]
    if (
        task.surface_texture == "no_punct_fragment"
        and task.utterance_mode in LOW_INFO_UTTERANCE_MODES
        and len(shaped.split()) <= 12
        and task.utterance_mode != "question_only"
    ):
        shaped = shaped.rstrip(" .!")
    if task.surface_texture == "emoji_or_sarcasm" and len(shaped.split()) <= 24:
        lowered = shaped.lower()
        if not re.search(r"\b(lol|lmao|haha)\b|/s|[\U0001F300-\U0001FAFF]", lowered):
            shaped = shaped.rstrip(" .") + (" lol" if task.local_task_id % 2 else " /s")
    if task.surface_texture == "messy_punctuation" and len(shaped.split()) <= 35:
        if "..." not in shaped and "…" not in shaped and "!" not in shaped:
            shaped = shaped.rstrip(".") + "..."
    if task.surface_texture == "gratitude_social":
        lowered = shaped.lower()
        if len(shaped.split()) <= 18 and not any(marker in lowered for marker in ("thanks", "thank you", "appreciate", "good to know", "that helps", "fair point")):
            shaped = "Thanks, " + shaped[0].lower() + shaped[1:] if shaped else "Thanks"
    return shaped or text


def write_reproducibility_snapshot(output_dir: Path, args: argparse.Namespace) -> None:
    snapshot_dir = output_dir / "_reproducibility"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(__file__).resolve()
    shutil.copy2(source_path, snapshot_dir / "generator_source_snapshot.py")
    write_json(
        snapshot_dir / "generator_config_snapshot.json",
        {
            "generator": GENERATOR_NAME,
            "created_at": utc_now(),
            "config": config_snapshot(args),
        },
    )
    (snapshot_dir / "generator_cli.txt").write_text(
        " ".join(shlex.quote(part) for part in scrub_argv(sys.argv)) + "\n",
        encoding="utf-8",
    )
    # Every prompt builder is supplied by the configured adapter, so resolve the
    # names bound at run time rather than the ones this file happens to define.
    # A name no adapter supplies is recorded as absent instead of raising, which
    # keeps the snapshot honest about what actually rendered.
    prompt_sources = {}
    for label in (
        "build_planner_prompt",
        "build_comment_move_planner_prompt",
        "build_writer_prompt",
        "render_parent_context_for_writer",
        "render_seed_context_for_writer",
        "rebalance_tasks_for_diversity",
    ):
        bound = globals().get(label)
        prompt_sources[label] = safe_getsource(bound) if bound is not None else "(absent)"
    write_json(
        snapshot_dir / "prompt_snapshot.json",
        {"system_prompts": SYSTEM_PROMPTS, "prompt_functions": prompt_sources},
    )
    (snapshot_dir / "git_status.txt").write_text(run_git_text(["git", "status", "--short"]), encoding="utf-8")
    try:
        diff_path = str(source_path.relative_to(Path.cwd()))
    except ValueError:
        diff_path = str(source_path)
    diff = run_git_text(["git", "diff", "--no-ext-diff", "--", diff_path])
    if not diff.strip():
        diff = "(git diff is empty; generator_source_snapshot.py is the authoritative source snapshot for this run.)\n"
    (snapshot_dir / "git_diff.patch").write_text(diff, encoding="utf-8")


if __name__ == "__main__":
    main()
