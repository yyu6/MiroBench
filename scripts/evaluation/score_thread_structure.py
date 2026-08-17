#!/usr/bin/env python3
"""Score thread-level structural metrics for Reddit discussions.

This evaluator implements the metrics described in
`thread_metrics_implementation_plan.md`:

- `length_std`
- `length_iqr`
- `length_cv`
- `max_depth`
- `avg_depth`
- `avg_branching_factor`
- `structural_virality`

The script loads either generated `discussion.json` runs or scraped real Reddit
bundles, normalizes each thread into a list of comments with `comment_id`,
`parent_id`, and comment text, and then exports one metric row per thread.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
from typing import Any

from score_thread_disagreement import detect_target_kind
from score_thread_semantic_uniformity import (
    ThreadComment,
    load_generated_comments,
    load_real_comments,
    median,
    weighted_average,
)


METRIC_NAMES = (
    "length_std",
    "length_iqr",
    "length_cv",
    "max_depth",
    "avg_depth",
    "avg_branching_factor",
    "structural_virality",
)


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Score thread-level structural metrics for Reddit discussions."
    )
    parser.add_argument("input", help="Real discussion folder/file or generated run folder/file.")
    parser.add_argument(
        "--target-kind",
        choices=["auto", "real", "generated"],
        default="auto",
        help="Input schema. auto detects generated discussion.json vs real Reddit bundle.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Where to write JSON results. Defaults next to input.",
    )
    parser.add_argument("--max-threads", type=int, default=0)
    parser.add_argument("--max-comments-per-thread", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    target_kind = args.target_kind
    if target_kind == "auto":
        target_kind = detect_target_kind(input_path)

    if target_kind == "generated":
        comments_by_thread, source_meta = load_generated_comments(input_path)
    else:
        comments_by_thread, source_meta = load_real_comments(input_path)

    if args.max_threads and args.max_threads > 0:
        comments_by_thread = dict(list(comments_by_thread.items())[: args.max_threads])
    if args.max_comments_per_thread and args.max_comments_per_thread > 0:
        comments_by_thread = {
            thread_id: comments[: args.max_comments_per_thread]
            for thread_id, comments in comments_by_thread.items()
        }

    thread_results = []
    for thread_id, comments in comments_by_thread.items():
        thread_results.append(score_thread(thread_id=thread_id, comments=comments))

    thread_results.sort(
        key=lambda row: (
            row["structural_virality"],
            row["avg_depth"],
            row["comment_count"],
        ),
        reverse=True,
    )
    result = {
        "meta": {
            "input": str(input_path),
            "target_kind": target_kind,
            "metric": "Thread structural metrics",
            "metric_names": list(METRIC_NAMES),
            "thread_count": len(thread_results),
            "comment_count": sum(int(row["comment_count"]) for row in thread_results),
            "source": source_meta,
        },
        "overall": aggregate_overall(thread_results),
        "threads": thread_results,
    }

    output_file = Path(args.output_file).expanduser() if args.output_file else default_output_path(input_path)
    if not output_file.is_absolute():
        output_file = (Path.cwd() / output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = result["overall"]
    print(f"Wrote {output_file}")
    print(
        "overall: "
        f"comments={overall.get('comment_count', 0)} threads={overall.get('thread_count', 0)} "
        f"length_cv={overall.get('weighted_length_cv', 0):.4f} "
        f"avg_depth={overall.get('weighted_avg_depth', 0):.4f} "
        f"structural_virality={overall.get('weighted_structural_virality', 0):.4f}"
    )


def score_thread(thread_id: str, comments: list[ThreadComment]) -> dict[str, Any]:
    """Compute all structural metrics for one thread."""

    lengths = sorted(tokenize_len(comment.text) for comment in comments)
    children = build_children(comments)
    depths = compute_depths(children)
    internal_branch_counts = [len(children[node_id]) for node_id in get_comment_ids(comments) if len(children[node_id]) > 0]

    return {
        "thread_id": thread_id,
        "thread_title": comments[0].thread_title if comments else "",
        "comment_count": len(comments),
        "length_std": compute_length_std(lengths),
        "length_iqr": compute_length_iqr(lengths),
        "length_cv": compute_length_cv(lengths),
        "max_depth": float(max(depths.values())) if depths else 0.0,
        "avg_depth": mean(list(depths.values())),
        "avg_branching_factor": mean(internal_branch_counts),
        "structural_virality": compute_structural_virality(comments),
    }


def tokenize_len(text: str) -> int:
    """Approximate comment length with whitespace token count."""

    return len((text or "").split())


def compute_length_std(lengths: list[int]) -> float:
    """Population standard deviation of comment lengths."""

    if len(lengths) < 2:
        return 0.0
    avg = sum(lengths) / len(lengths)
    return float(math.sqrt(sum((value - avg) ** 2 for value in lengths) / len(lengths)))


def compute_length_iqr(lengths: list[int]) -> float:
    """Interquartile range of comment lengths."""

    if not lengths:
        return 0.0
    return float(percentile(lengths, 0.75) - percentile(lengths, 0.25))


def compute_length_cv(lengths: list[int]) -> float:
    """Coefficient of variation of comment lengths."""

    if not lengths:
        return 0.0
    avg = sum(lengths) / len(lengths)
    if avg <= 0:
        return 0.0
    return float(compute_length_std(lengths) / avg)


def percentile(sorted_values: list[int], q: float) -> float:
    """Linear-interpolation percentile on an already sorted list."""

    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def get_comment_ids(comments: list[ThreadComment]) -> set[str]:
    """Return the set of normalized comment ids inside the thread."""

    return {str(comment.comment_id) for comment in comments}


def normalize_parent_id(parent_id: str, comment_ids: set[str]) -> str | None:
    """Map missing or non-comment parents to the synthetic root."""

    parent = str(parent_id or "")
    return parent if parent in comment_ids else None


def build_children(comments: list[ThreadComment]) -> dict[str | None, list[str]]:
    """Build parent -> children adjacency, with missing parents attached to root."""

    comment_ids = get_comment_ids(comments)
    children: dict[str | None, list[str]] = defaultdict(list)
    for comment in comments:
        parent = normalize_parent_id(comment.parent_id, comment_ids)
        children[parent].append(str(comment.comment_id))
        children.setdefault(str(comment.comment_id), [])
    return children


def compute_depths(children: dict[str | None, list[str]]) -> dict[str, int]:
    """Compute BFS depths where top-level comments have depth 1."""

    depth: dict[str, int] = {}
    queue: deque[str] = deque()
    for top_id in children.get(None, []):
        depth[top_id] = 1
        queue.append(top_id)

    while queue:
        parent = queue.popleft()
        for child in children.get(parent, []):
            if child in depth:
                continue
            depth[child] = depth[parent] + 1
            queue.append(child)
    return depth


def build_undirected_adjacency(comments: list[ThreadComment]) -> dict[str, list[str]]:
    """Build an undirected graph over real comments only."""

    comment_ids = get_comment_ids(comments)
    adjacency: dict[str, list[str]] = defaultdict(list)
    for comment in comments:
        comment_id = str(comment.comment_id)
        adjacency.setdefault(comment_id, [])
        parent = normalize_parent_id(comment.parent_id, comment_ids)
        if parent is None:
            continue
        adjacency[comment_id].append(parent)
        adjacency[parent].append(comment_id)
    return adjacency


def shortest_path_lengths_from(source: str, adjacency: dict[str, list[str]]) -> dict[str, int]:
    """Return BFS distances from one comment to all reachable comments."""

    distances = {source: 0}
    queue: deque[str] = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[node] + 1
            queue.append(neighbor)
    return distances


def compute_structural_virality(comments: list[ThreadComment]) -> float:
    """Average shortest-path distance across all unordered comment pairs."""

    comment_ids = sorted(get_comment_ids(comments))
    if len(comment_ids) < 2:
        return 0.0

    adjacency = build_undirected_adjacency(comments)
    all_distances = {
        comment_id: shortest_path_lengths_from(comment_id, adjacency)
        for comment_id in comment_ids
    }

    total_distance = 0.0
    pair_count = 0
    for left_id, right_id in combinations(comment_ids, 2):
        distance = all_distances[left_id].get(right_id)
        if distance is None:
            continue
        total_distance += distance
        pair_count += 1
    if pair_count == 0:
        return 0.0
    return float(total_distance / pair_count)


def mean(values: list[int] | list[float]) -> float:
    """Mean with empty-list protection."""

    if not values:
        return 0.0
    return float(sum(values) / len(values))


def aggregate_overall(thread_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate thread-level structural metrics into one summary."""

    comment_count = sum(int(row["comment_count"]) for row in thread_results)
    valid_rows = [row for row in thread_results if int(row["comment_count"]) > 0]
    if not valid_rows:
        return {"thread_count": len(thread_results), "comment_count": comment_count}

    result: dict[str, Any] = {
        "thread_count": len(thread_results),
        "comment_count": comment_count,
    }
    for metric_name in METRIC_NAMES:
        result[f"weighted_{metric_name}"] = weighted_average(
            [(float(row[metric_name]), int(row["comment_count"])) for row in valid_rows]
        )
        result[f"mean_thread_{metric_name}"] = mean([float(row[metric_name]) for row in valid_rows])
        result[f"median_thread_{metric_name}"] = median([float(row[metric_name]) for row in valid_rows])
        result[f"max_thread_{metric_name}"] = max(float(row[metric_name]) for row in valid_rows)
    return result


def default_output_path(input_path: Path) -> Path:
    """Choose a default output path next to the evaluated input."""

    if input_path.is_dir():
        return input_path / "thread_structure_results.json"
    return input_path.with_suffix(input_path.suffix + ".thread_structure_results.json")


if __name__ == "__main__":
    main()
