from __future__ import annotations

import json
import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .data import find_matched_real_thread, load_real_thread_bank
from .domain import DomainConfig
from .domain_profile import load_domain_profile
from .planning_quality import evaluate_plan_batch
from .prompts import INTERNAL_CONTROL_ID_RE


PLACEHOLDER_PATTERNS = (
    r"\b(?:lorem ipsum|insert (?:comment|response)|generated comment)\b",
    r"\[(?:insert|placeholder|todo|tbd)[^]]*\]",
    r"^\s*(?:placeholder|placeholder (?:comment|response)|generated comment)\s*[.!]?\s*$",
    r"\b(?:as an ai|i cannot provide|i can't provide)\b",
)
PROMPT_LEAK_PATTERNS = (
    r"\b(?:private sampled controls|semantic_move|claim_family|surface_skeleton|context_aperture)\b",
    r"\b(?:you are revising one|return strict json|planner labels?)\b",
)

# Very short exact matches are often shared forum reactions rather than
# evidence that a generated comment copied its matched reference thread.
MIN_COPY_OVERLAP_TOKENS = 8


def audit_generated_root(
    root: Path,
    *,
    config: DomainConfig | None = None,
    seed_pool: Path | None = None,
    domain_profile: Path | None = None,
    min_accepted_share: float = 0.50,
    min_unique_share: float = 0.80,
    min_mean_words: float = 5.0,
    max_plan_collision_rate: float = 0.10,
    max_perspective_share: float = 0.34,
) -> dict[str, Any]:
    discussions = sorted(root.expanduser().resolve().glob("run_*_sampled_reddit/discussion.json"))
    posts = 0
    comments = 0
    planned_comments = 0
    total_words = 0
    zero_comment_posts = 0
    placeholders: list[dict[str, Any]] = []
    prompt_leaks: list[dict[str, Any]] = []
    internal_control_leaks: list[dict[str, Any]] = []
    duplicate_comments: list[dict[str, Any]] = []
    matched_real_overlap: list[dict[str, Any]] = []
    drawn_link_rows: list[dict[str, Any]] = []
    drawn_link_in_matched_real: list[dict[str, Any]] = []
    reference_viewpoint_overlap: list[dict[str, Any]] = []
    writer_rejections = Counter()
    perspective_counts: Counter[str] = Counter()
    claim_counts: Counter[str] = Counter()
    recorded_slots = 0
    recorded_posts = 0
    skipped_slots = 0
    backfilled_slots = 0
    incomplete_recorded_posts = 0
    incomplete_structural_posts = 0
    incomplete_coverage_examples: list[dict[str, Any]] = []
    claim_collision_posts = 0
    semantic_collision_posts = 0
    semantic_colliding_comments = 0
    substantive_plan_comments = 0
    overconcentrated_perspective_posts = 0
    reply_contract_violations = 0
    plan_quality_examples: list[dict[str, Any]] = []
    seeds = _load_seeds(seed_pool) if seed_pool else []
    real_bank = load_real_thread_bank(config.raw_discussions_dir) if config and seeds else []
    profile = load_domain_profile(domain_profile) if domain_profile else {}
    reference_viewpoint_index = _build_overlap_index(
        [
            str(item.get("text") or "")
            for item in (profile.get("reference_viewpoints") or [])
            if isinstance(item, dict)
        ]
    )
    perspective_ids = {
        str(item.get("perspective_id") or "").upper()
        for item in (profile.get("perspectives") or [])
        if isinstance(item, dict) and item.get("perspective_id")
    }
    for path in discussions:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for post in payload.get("posts") or []:
            if not isinstance(post, dict):
                continue
            posts += 1
            visible_seed_text = " ".join(
                str(post.get(key) or "") for key in ("title", "body", "content")
            )
            visible_control_ids = {
                match.group(0).upper()
                for match in INTERNAL_CONTROL_ID_RE.finditer(visible_seed_text)
            }
            rows = list(_flatten(post.get("comments") or []))
            records = [
                row
                for row in (post.get("generation_records") or [])
                if isinstance(row, dict)
            ]
            if records:
                recorded_posts += 1
            recorded_slots += len(records)
            post_skipped_slots = sum(bool(row.get("skipped")) for row in records)
            skipped_slots += post_skipped_slots
            backfilled_slots += sum(row.get("backfilled_from") is not None for row in records)
            post_planned_comments = int(
                (post.get("thread_plan") or {}).get("target_comments") or len(rows)
            )
            planned_comments += post_planned_comments
            generated_records = sum(
                isinstance(row.get("comment"), dict) for row in records
            )
            rendered_coverage_complete = len(rows) == post_planned_comments
            recorded_coverage_complete = not records or (
                len(records) == post_planned_comments
                and generated_records == len(records)
                and post_skipped_slots == 0
            )
            if not rendered_coverage_complete or not recorded_coverage_complete:
                incomplete_structural_posts += 1
                if records:
                    incomplete_recorded_posts += 1
                if len(incomplete_coverage_examples) < 20:
                    incomplete_coverage_examples.append(
                        {
                            "file": str(path),
                            "post_id": post.get("post_id") or post.get("id"),
                            "planned_comments": post_planned_comments,
                            "rendered_comments": len(rows),
                            "generation_records": len(records),
                            "generated_records": generated_records,
                            "skipped_records": post_skipped_slots,
                        }
                    )
            real_comments = _matched_real_comments(post, seeds=seeds, real_bank=real_bank)
            comments += len(rows)
            if not rows:
                zero_comment_posts += 1
            normalized: Counter[str] = Counter()
            for row in rows:
                text = str(row.get("content") or row.get("body") or "").strip()
                normalized[" ".join(text.lower().split())] += 1
                perspective = str(row.get("perspective_id") or "seed_local").strip()
                claim = str(row.get("claim_key") or "local_claim").strip()
                perspective_counts[perspective] += 1
                claim_counts[claim] += 1
                total_words += len(text.split())
                location = {
                    "file": str(path),
                    "post_id": post.get("post_id") or post.get("id"),
                    "comment_id": row.get("comment_id") or row.get("id"),
                    "text": text[:240],
                }
                if any(re.search(pattern, text, flags=re.I) for pattern in PLACEHOLDER_PATTERNS):
                    placeholders.append(location)
                has_prompt_pattern = any(
                    re.search(pattern, text, flags=re.I) for pattern in PROMPT_LEAK_PATTERNS
                )
                if has_prompt_pattern:
                    prompt_leaks.append(location)
                row_perspective = str(row.get("perspective_id") or "").upper()
                candidate_perspective_ids = set(perspective_ids)
                if re.fullmatch(r"P\d{2}", row_perspective):
                    candidate_perspective_ids.add(row_perspective)
                leaked_labels = sorted(
                    {
                        match.group(0).upper()
                        for match in INTERNAL_CONTROL_ID_RE.finditer(text)
                        if match.group(0).upper() not in visible_control_ids
                        and (
                            match.group(0).upper() in candidate_perspective_ids
                            or match.group(0).upper().startswith(("S", "B"))
                        )
                    }
                )
                if leaked_labels:
                    leak = {**location, "labels": leaked_labels}
                    internal_control_leaks.append(leak)
                    if not has_prompt_pattern:
                        prompt_leaks.append(leak)
                copied = _closest_real_overlap(text, real_comments)
                if copied is not None:
                    matched_real_overlap.append({**location, **copied})
                # A drawn reference link is a single high-entropy token string.
                # `_closest_real_overlap` is a 5-gram Jaccard test with a 0.65
                # floor, so one shared URL inside an ordinary comment scores far
                # below it and slips through. A URL that also appears in this
                # post's matched real thread is evaluation-set content sitting in
                # generated output, which is what `evaluable` exists to stop, so
                # it gets its own exact test rather than riding on the phrase one.
                for url in _urls(text):
                    row_out = {**location, "url": url}
                    drawn_link_rows.append(row_out)
                    if any(url in real for real in real_comments):
                        drawn_link_in_matched_real.append(row_out)
                reference_copied = _closest_overlap(text, reference_viewpoint_index)
                if reference_copied is not None:
                    reference_viewpoint_overlap.append({**location, **reference_copied})
                reason = str(row.get("writer_rejection_reason") or "").strip()
                if reason:
                    writer_rejections[reason.split(":", 1)[0]] += 1
            post_claims = Counter(
                str(row.get("claim_key") or "local_claim").strip() for row in rows
            )
            if len(rows) >= 5 and post_claims and max(post_claims.values()) / len(rows) > 0.35:
                claim_collision_posts += 1
            plan_report = evaluate_plan_batch(
                _planner_rows_for_audit(rows=rows, records=records),
                max_perspective_share=max_perspective_share,
                require_reply_novelty=True,
            )
            substantive_plan_comments += plan_report.substantive_count
            semantic_colliding_comments += len(plan_report.colliding_samples)
            reply_contract_violations += sum(
                issue.code == "reply_increment_conflict"
                for issue in plan_report.issues
            )
            if plan_report.collision_rate > max_plan_collision_rate:
                semantic_collision_posts += 1
            if (
                plan_report.substantive_count >= 8
                and plan_report.dominant_perspective_share > max_perspective_share + 0.12
            ):
                overconcentrated_perspective_posts += 1
            if plan_report.issues and len(plan_quality_examples) < 20:
                plan_quality_examples.append(
                    {
                        "file": str(path),
                        "post_id": post.get("post_id") or post.get("id"),
                        **plan_report.to_dict(),
                    }
                )
            for text, count in normalized.items():
                if text and count > 1:
                    duplicate_comments.append(
                        {"file": str(path), "post_id": post.get("id"), "count": count, "text": text[:240]}
                    )
    accepted_share = comments / max(1, planned_comments)
    unique_count = comments - sum(max(0, row["count"] - 1) for row in duplicate_comments)
    unique_share = unique_count / max(1, comments)
    mean_words = total_words / max(1, comments)
    semantic_collision_rate = semantic_colliding_comments / max(1, substantive_plan_comments)
    profile_isolation = _profile_isolation_report(domain_profile, seeds)
    # Evaluation must reject unusable or contaminated artifacts, while still
    # allowing distribution-quality failures to be measured. Perspective and
    # semantic concentration remain strict generation-health diagnostics below.
    evaluable = (
        bool(posts)
        and not zero_comment_posts
        and not placeholders
        and not prompt_leaks
        and not matched_real_overlap
        and not drawn_link_in_matched_real
        and not reference_viewpoint_overlap
        and not incomplete_structural_posts
        and accepted_share >= min_accepted_share
        and unique_share >= min_unique_share
        and mean_words >= min_mean_words
        and profile_isolation["valid"]
    )
    healthy = (
        evaluable
        and semantic_collision_rate <= max_plan_collision_rate
        and not overconcentrated_perspective_posts
        and not reply_contract_violations
    )
    return {
        "healthy": healthy,
        "evaluable": evaluable,
        "discussion_files": len(discussions),
        "posts": posts,
        "comments": comments,
        "planned_comments": planned_comments,
        "accepted_share": round(accepted_share, 4),
        "unique_share": round(unique_share, 4),
        "mean_words": round(mean_words, 2),
        "recorded_generation_slots": recorded_slots,
        "posts_with_generation_records": recorded_posts,
        "incomplete_recorded_posts": incomplete_recorded_posts,
        "incomplete_structural_posts": incomplete_structural_posts,
        "complete_structural_coverage": incomplete_structural_posts == 0,
        "skipped_generation_slots": skipped_slots,
        "backfilled_generation_slots": backfilled_slots,
        "structural_slot_fidelity": round(comments / max(1, recorded_slots), 4),
        "perspective_unique": len(perspective_counts),
        "perspective_entropy": round(_counter_entropy(perspective_counts), 6),
        "top_perspectives": dict(perspective_counts.most_common(12)),
        "claim_unique": len(claim_counts),
        "claim_collision_posts": claim_collision_posts,
        "top_claims": dict(claim_counts.most_common(12)),
        "semantic_plan_collision_rate": round(semantic_collision_rate, 6),
        "semantic_plan_collision_posts": semantic_collision_posts,
        "semantic_plan_colliding_comments": semantic_colliding_comments,
        "substantive_plan_comments": substantive_plan_comments,
        "overconcentrated_perspective_posts": overconcentrated_perspective_posts,
        "reply_contract_violations": reply_contract_violations,
        "domain_profile_isolation": profile_isolation,
        "thresholds": {
            "min_accepted_share": min_accepted_share,
            "min_unique_share": min_unique_share,
            "min_mean_words": min_mean_words,
            "max_plan_collision_rate": max_plan_collision_rate,
            "max_perspective_share": max_perspective_share,
        },
        "zero_comment_posts": zero_comment_posts,
        "placeholder_comments": len(placeholders),
        "prompt_leak_comments": len(prompt_leaks),
        "internal_control_label_comments": len(internal_control_leaks),
        "matched_real_copy_risks": len(matched_real_overlap),
        "drawn_link_comments": len(drawn_link_rows),
        "drawn_link_distinct": len({row["url"] for row in drawn_link_rows}),
        "drawn_link_repeated_in_thread": _link_repeats_within_thread(drawn_link_rows),
        "drawn_link_in_matched_real": len(drawn_link_in_matched_real),
        "reference_viewpoint_copy_risks": len(reference_viewpoint_overlap),
        "exact_duplicate_groups": len(duplicate_comments),
        "writer_rejection_reasons": dict(writer_rejections),
        "placeholder_examples": placeholders[:20],
        "prompt_leak_examples": prompt_leaks[:20],
        "internal_control_label_examples": internal_control_leaks[:20],
        "matched_real_copy_examples": matched_real_overlap[:20],
        "drawn_link_examples": drawn_link_rows[:20],
        "drawn_link_in_matched_real_examples": drawn_link_in_matched_real[:20],
        "reference_viewpoint_copy_examples": reference_viewpoint_overlap[:20],
        "duplicate_examples": duplicate_comments[:20],
        "incomplete_coverage_examples": incomplete_coverage_examples,
        "plan_quality_examples": plan_quality_examples,
    }


def _planner_rows_for_audit(
    *,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Recover plan metadata without relying on rendered comment traversal.

    The persisted generation record is the source of truth for anonymous
    parent-slot IDs and Planner controls. Final comments retain these fields for
    inspection, but their tree IDs are not the original S# sequence after a
    breadth-first plan is rendered recursively. Falling back to final comments
    keeps legacy artifacts auditable without falsely marking them malformed.
    """

    planned: dict[int, dict[str, Any]] = {}
    for record in records:
        task = record.get("task") if isinstance(record, dict) else None
        if not isinstance(task, dict):
            continue
        try:
            sample_id = int(task.get("local_task_id") or 0)
        except (TypeError, ValueError):
            continue
        if sample_id <= 0:
            continue
        plan = dict(task)
        plan["sample_id"] = sample_id
        plan["parent_sample_id"] = task.get("local_parent_task_id") or ""
        planned[sample_id] = plan
    if len(planned) == len(rows):
        return planned

    index_by_comment_id: dict[str, int] = {}
    for sample_id, row in enumerate(rows, start=1):
        for value in (row.get("comment_id"), row.get("id")):
            key = str(value or "").strip()
            if key:
                index_by_comment_id[key] = sample_id
    fallback: dict[int, dict[str, Any]] = {}
    for sample_id, row in enumerate(rows, start=1):
        plan = dict(row)
        plan["sample_id"] = sample_id
        parent_key = str(row.get("parent_comment_id") or row.get("parent_id") or "").strip()
        plan["parent_sample_id"] = index_by_comment_id.get(parent_key, "")
        fallback[sample_id] = plan
    return fallback


def _flatten(comments: list[dict[str, Any]]):
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        yield comment
        yield from _flatten(comment.get("replies") or comment.get("children") or [])


def _load_seeds(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("seed_posts") if isinstance(payload, dict) else payload
    return rows if isinstance(rows, list) else []


def _matched_real_comments(
    post: dict[str, Any],
    *,
    seeds: list[dict[str, Any]],
    real_bank: list[dict[str, Any]],
) -> list[str]:
    try:
        seed_index = int(post.get("seed_index"))
    except (TypeError, ValueError):
        return []
    if seed_index < 0 or seed_index >= len(seeds):
        return []

    class Seed:
        pass

    seed = Seed()
    seed.source_raw_post_id = str(seeds[seed_index].get("source_raw_post_id") or "")
    seed.metadata = seeds[seed_index]
    matched = find_matched_real_thread(real_bank, seed)
    if not matched:
        return []
    return [str(row.get("body") or "").strip() for row in matched.get("comments") or []]


def _closest_real_overlap(text: str, real_comments: list[str]) -> dict[str, Any] | None:
    return _closest_overlap(text, _build_overlap_index(real_comments))


def _build_overlap_index(
    comments: list[str],
) -> list[tuple[str, str, set[tuple[str, ...]]]]:
    index = []
    for comment in comments:
        tokens = _tokens(comment)
        if len(tokens) < 5:
            continue
        index.append((comment, " ".join(tokens), _ngrams(tokens, 5)))
    return index


def _closest_overlap(
    text: str,
    real_index: list[tuple[str, str, set[tuple[str, ...]]]],
) -> dict[str, Any] | None:
    generated_tokens = _tokens(text)
    if len(generated_tokens) < MIN_COPY_OVERLAP_TOKENS:
        return None
    generated_ngrams = _ngrams(generated_tokens, 5)
    best_score = 0.0
    best_text = ""
    best_exact = False
    normalized = " ".join(generated_tokens)
    for real, real_normalized, real_ngrams in real_index:
        exact = normalized == real_normalized
        union = generated_ngrams | real_ngrams
        score = len(generated_ngrams & real_ngrams) / max(1, len(union))
        if exact or score > best_score:
            best_score = score
            best_text = real
            best_exact = exact
    if not best_exact and (len(generated_ngrams) < 4 or best_score < 0.65):
        return None
    return {
        "exact_match": best_exact,
        "fivegram_jaccard": round(best_score, 4),
        "matched_real_text": best_text[:240],
    }


_URL_RE = re.compile(r"https?://\S+|\bwww\.\S+", re.I)


def _urls(text: str) -> list[str]:
    return [match.rstrip(").,;\"'") for match in _URL_RE.findall(str(text or ""))]


def _link_repeats_within_thread(rows: list[dict[str, Any]]) -> int:
    """Count links used more than once inside one post.

    A repeated link is a repeated n-gram, which pushes `self_bleu_4` and
    `self_bertscore` the wrong way -- the exact direction the drawn link exists
    to fix. The per-slot SHA-256 draw is meant to make this zero.
    """

    seen: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row.get("post_id") or ""), str(row.get("url") or ""))
        seen[key] = seen.get(key, 0) + 1
    return sum(1 for count in seen.values() if count > 1)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(text).lower())


def _ngrams(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _counter_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count > 0
    )


def _profile_isolation_report(
    domain_profile: Path | None,
    seeds: list[dict[str, Any]],
) -> dict[str, Any]:
    if domain_profile is None:
        return {"valid": True, "status": "not_checked"}
    try:
        profile = load_domain_profile(domain_profile)
    except (OSError, ValueError) as exc:
        return {"valid": False, "status": "invalid_profile", "error": str(exc)}
    source = profile.get("source") or {}
    seed_ids = {
        str(row.get("source_raw_post_id") or "").strip()
        for row in seeds
        if isinstance(row, dict) and str(row.get("source_raw_post_id") or "").strip()
    }
    seed_hash = hashlib.sha256("\n".join(sorted(seed_ids)).encode("utf-8")).hexdigest()
    reference_rows = [
        row for row in (profile.get("reference_viewpoints") or []) if isinstance(row, dict)
    ]
    reference_source_ids = {
        str(row.get("source_post_id") or "").strip()
        for row in reference_rows
        if str(row.get("source_post_id") or "").strip()
    }
    checks = {
        "test_content_visible_false": source.get("test_content_visible") is False,
        "seed_hash_matches": str(source.get("seed_ids_sha256") or "") == seed_hash,
        "seed_reference_overlap_zero": int(source.get("seed_reference_overlap_count") or 0) == 0,
        "excluded_seed_count_matches": int(source.get("excluded_seed_count") or -1) == len(seed_ids),
        "has_non_seed_references": int(source.get("reference_thread_count") or 0) >= 20,
        "reference_viewpoint_count_matches": int(
            source.get("reference_viewpoint_count") or 0
        ) == len(reference_rows),
        "reference_viewpoint_seed_overlap_zero": not (seed_ids & reference_source_ids),
    }
    return {
        "valid": all(checks.values()),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "profile": str(domain_profile),
        "profile_sha256": profile.get("profile_sha256"),
    }
