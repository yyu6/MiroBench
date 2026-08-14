from __future__ import annotations

from sampling_generator.engine.model import BranchPlan
from sampling_generator.engine.model import SeedPost
from sampling_generator.engine.model import ThreadTarget
from sampling_generator.engine.util import safe_int
from sampling_generator.engine.util import weighted_choice
from typing import Any
import math
import random

def sample_thread_target(
    *,
    seed_post: SeedPost,
    rng: random.Random,
    max_comments_per_post: int,
    count_scale: float,
    exact_matched_thread_size: bool = False,
) -> ThreadTarget:
    real_count = max(0, seed_post.real_num_comments)
    if real_count:
        scaled = real_count if exact_matched_thread_size else int(round(real_count * max(0.05, count_scale) * rng.uniform(0.85, 1.18)))
        target_comments = max(1, min(max_comments_per_post, scaled) if max_comments_per_post > 0 else scaled)
    else:
        shape = weighted_choice(
            rng,
            (
                ("quiet", 0.25),
                ("normal", 0.52),
                ("busy", 0.18),
                ("viral_chain", 0.05),
            ),
        )
        target_comments = comments_for_shape(shape, rng)
        target_comments = min(max_comments_per_post, target_comments) if max_comments_per_post > 0 else target_comments

    if target_comments <= 2:
        shape_label = "dead"
        max_depth_goal = 1
    elif target_comments <= 8:
        shape_label = "quiet"
        max_depth_goal = rng.choice([1, 2])
    elif target_comments <= 25:
        shape_label = "normal"
        max_depth_goal = rng.choice([2, 3, 4])
    elif target_comments <= 60:
        shape_label = "busy"
        max_depth_goal = rng.choice([4, 5, 6])
    else:
        shape_label = "viral_chain"
        max_depth_goal = rng.choice([6, 7, 8, 9])

    top_min = 1 if target_comments <= 3 else 3
    top_max = max(top_min, min(target_comments, int(math.sqrt(target_comments) * 2.4) + 2))
    top_level_comments = rng.randint(top_min, top_max)
    return ThreadTarget(
        target_comments=target_comments,
        top_level_comments=top_level_comments,
        max_depth_goal=max_depth_goal,
        shape_label=shape_label,
        length_mix_note=(
            "Target length mix: some micro/short fragments, mostly medium comments, "
            "a few long/story comments when locally natural."
        ),
    )

def comments_for_shape(shape: str, rng: random.Random) -> int:
    if shape == "quiet":
        return rng.randint(3, 8)
    if shape == "busy":
        return rng.randint(26, 60)
    if shape == "viral_chain":
        return rng.randint(70, 120)
    return rng.randint(9, 25)

def render_top_counts(global_memory: dict[str, Any]) -> str:
    lines: list[str] = []
    for key in ("comment_function", "content_angle", "evidence_mode", "voice", "story_mode"):
        counts = dict(global_memory.get(key) or {})
        if not counts:
            lines.append(f"- {key}: none yet")
            continue
        top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
        lines.append("- " + key + ": " + ", ".join(f"{name}={count}" for name, count in top))
    return "\n".join(lines)

def planner_batch_ranges_by_depth(
    comments: list[dict[str, Any]], *, batch_size: int
) -> list[tuple[int, int]]:
    """Return contiguous Planner batches that never mix parent and child depths.

    ``selected_matched_comments`` has already put rows in breadth-first tree
    order. Keeping a depth group intact lets all sibling replies be planned
    together while the preceding group is available through the private plan
    ledger. The function sees only anonymous topology and depth metadata.
    """

    if not comments:
        return []
    size = max(1, int(batch_size))
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(comments):
        depth = max(0, safe_int(comments[start].get("depth"), 0))
        group_end = start + 1
        while (
            group_end < len(comments)
            and max(0, safe_int(comments[group_end].get("depth"), 0)) == depth
        ):
            group_end += 1
        for batch_start in range(start, group_end, size):
            ranges.append((batch_start, min(group_end, batch_start + size)))
        start = group_end
    return ranges

def selected_matched_comments(
    *,
    matched_real_thread: dict[str, Any],
    target: ThreadTarget,
    matched_real_comments: int,
) -> list[dict[str, Any]]:
    raw_comments = list(matched_real_thread.get("comments") or [])
    cap = target.target_comments
    if matched_real_comments > 0:
        cap = min(cap, matched_real_comments)
    cap = max(0, min(cap, len(raw_comments)))
    if cap <= 0:
        return []
    if len(raw_comments) <= cap:
        # Keep direct parents ahead of replies even when no structural sampling
        # is needed. Otherwise quiet threads bypass the tree-ordering contract
        # and a child can be planned before its parent has a ledger entry.
        return order_comments_by_thread_tree(raw_comments)

    parent_to_index: dict[str, int] = {}
    for idx, row in enumerate(raw_comments):
        for key in real_comment_keys(row):
            parent_to_index[key] = idx

    selected: set[int] = set()

    def ancestry_for(index: int) -> list[int]:
        chain: list[int] = []
        seen: set[int] = set()
        current = index
        while current not in seen:
            seen.add(current)
            chain.append(current)
            parent_id = str(raw_comments[current].get("parent_id") or "")
            parent_index = parent_to_index.get(parent_id)
            if parent_index is None:
                break
            current = parent_index
        return list(reversed(chain))

    def add_with_ancestors(index: int) -> bool:
        chain = [item for item in ancestry_for(index) if item not in selected]
        if len(selected) + len(chain) > cap:
            return False
        selected.update(chain)
        return True

    def add_evenly(candidates: list[int], quota: int) -> None:
        if quota <= 0 or not candidates:
            return
        added = 0
        for index in evenly_spaced_indices(candidates, min(len(candidates), quota * 3)):
            if len(selected) >= cap:
                return
            if add_with_ancestors(index):
                added += 1
            if added >= quota:
                return

    count = len(raw_comments)
    thirds = {
        "early": range(0, max(1, count // 3)),
        "middle": range(max(1, count // 3), max(2, (count * 2) // 3)),
        "late": range(max(2, (count * 2) // 3), count),
    }
    short = [i for i, row in enumerate(raw_comments) if len(str(row.get("body") or "").split()) <= 10]
    deep = [i for i, row in enumerate(raw_comments) if safe_int(row.get("depth"), 0) >= 3]
    roots = [i for i, row in enumerate(raw_comments) if str(row.get("parent_id") or "").startswith("t3_")]
    late = list(thirds["late"])

    add_evenly(list(thirds["early"]), max(1, round(cap * 0.18)))
    add_evenly(list(thirds["middle"]), max(1, round(cap * 0.18)))
    add_evenly(late, max(1, round(cap * 0.22)))
    add_evenly(short, max(1, round(cap * 0.20)))
    add_evenly(deep, max(1, round(cap * 0.18)))
    add_evenly(roots, max(1, round(cap * 0.14)))
    add_evenly(list(range(count)), cap)

    selected_rows = [raw_comments[index] for index in sorted(selected)[:cap]]
    return order_comments_by_thread_tree(selected_rows)

def order_comments_by_thread_tree(comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order anonymous slots breadth-first so parent plans precede replies.

    A depth-first order puts a root and its replies into one Planner call. The
    Planner then has only a structural parent ID, not the parent's settled
    semantic contract, and tends to restate the root claim. Breadth-first
    ordering creates a root-plan ledger before direct replies are planned.
    This uses only identifiers and parent links, never matched comment text.
    """

    if len(comments) < 2:
        return list(comments)
    identifiers: dict[str, int] = {}
    for index, row in enumerate(comments):
        for key in real_comment_keys(row):
            identifiers[str(key)] = index
    children: dict[int, list[int]] = {index: [] for index in range(len(comments))}
    roots: list[int] = []
    for index, row in enumerate(comments):
        parent = identifiers.get(str(row.get("parent_id") or "").strip())
        if parent is None or parent == index:
            roots.append(index)
        else:
            children[parent].append(index)
    ordered: list[int] = []
    visited: set[int] = set()
    frontier = list(roots)
    while frontier:
        next_frontier: list[int] = []
        for index in frontier:
            if index in visited:
                continue
            visited.add(index)
            ordered.append(index)
            next_frontier.extend(children.get(index, []))
        frontier = next_frontier
    # Preserve malformed/orphaned rows rather than losing an anonymous slot.
    for index in range(len(comments)):
        if index not in visited:
            ordered.append(index)
    normalized: list[dict[str, Any]] = []
    for index in ordered:
        row = comments[index]
        parent_raw = str(row.get("parent_id") or "").strip()
        # Raw Reddit exports can retain a reply while its parent was deleted or
        # filtered out of the corpus. The task expander already has no parent
        # to attach in that case, so make the Planner see the same effective
        # root topology. This changes only anonymous structural metadata, not
        # the real comment body supplied to the private Planner.
        if (
            parent_raw
            and not parent_raw.startswith("t3_")
            and parent_raw not in identifiers
        ):
            normalized_row = dict(row)
            post_id = str(row.get("post_id") or "orphaned_post").strip()
            normalized_row["parent_id"] = f"t3_{post_id}"
            normalized_row["depth"] = 0
            normalized.append(normalized_row)
        else:
            normalized.append(row)
    return normalized

def evenly_spaced_indices(candidates: list[int], count: int) -> list[int]:
    if count <= 0 or not candidates:
        return []
    if count >= len(candidates):
        return list(candidates)
    if count == 1:
        return [candidates[len(candidates) // 2]]
    result: list[int] = []
    last_position = len(candidates) - 1
    for step in range(count):
        position = round(step * last_position / (count - 1))
        value = candidates[position]
        if value not in result:
            result.append(value)
    return result

def real_comment_keys(row: dict[str, Any]) -> tuple[str, ...]:
    keys: list[str] = []
    for value in (row.get("comment_fullname"), row.get("fullname"), row.get("name")):
        text = str(value or "").strip()
        if text and text not in keys:
            keys.append(text)
    comment_id = str(row.get("comment_id") or row.get("id") or "").strip()
    if comment_id:
        if comment_id not in keys:
            keys.append(comment_id)
        prefixed = comment_id if comment_id.startswith("t1_") else f"t1_{comment_id}"
        if prefixed not in keys:
            keys.append(prefixed)
    return tuple(keys)

def branch_by_id(branches: list[BranchPlan], branch_id: int) -> BranchPlan | None:
    for branch in branches:
        if branch.branch_id == branch_id:
            return branch
    return None
