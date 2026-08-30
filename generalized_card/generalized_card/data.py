from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .domain import DomainConfig


REMOVED = {"", "[deleted]", "[removed]", "deleted", "removed"}


def load_real_thread_bank(raw_dir: Path, *, max_threads: int = 0) -> list[dict[str, Any]]:
    """Load product-domain Reddit threads with stable source metadata.

    The legacy loader matched only ``post_id``. Product corpora can contain the
    same Reddit post under multiple product folders, so this loader retains the
    product directory and source file and deduplicates each comment tree.
    """

    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)

    posts = _load_post_metadata(raw_dir)
    comments = _load_comments(raw_dir)
    bank: list[dict[str, Any]] = []
    for key, rows in comments.items():
        if not rows:
            continue
        source_dir, post_id = key
        post = posts.get(key, {})
        source_file = str(post.get("_source_file") or "")
        bank.append(
            {
                "post_id": post_id,
                "thread_key": f"{Path(source_dir).name}:{post_id}",
                "title": str(post.get("title") or rows[0].get("post_title") or ""),
                "body": str(post.get("selftext") or post.get("body") or ""),
                "subreddit": str(post.get("subreddit") or ""),
                "source_dir": source_dir,
                "source_file": source_file,
                "source_product": Path(source_dir).name,
                "comments": rows,
                "comment_count": len(rows),
                "post": post,
            }
        )
    bank.sort(key=lambda item: (-int(item["comment_count"]), item["thread_key"]))
    if max_threads and len(bank) > max_threads:
        bank = bank[:max_threads]
    return bank


def find_matched_real_thread(
    real_bank: list[dict[str, Any]],
    seed_post: Any,
) -> dict[str, Any] | None:
    source_id = str(getattr(seed_post, "source_raw_post_id", "") or "").strip()
    if not source_id:
        return None
    metadata = getattr(seed_post, "metadata", {}) or {}
    wanted_product = str(
        metadata.get("source_product_dir")
        or metadata.get("source_product")
        or Path(str(metadata.get("source_file") or "")).parent.name
        or ""
    )
    candidates = [item for item in real_bank if str(item.get("post_id") or "") == source_id]
    if wanted_product:
        exact = [item for item in candidates if str(item.get("source_product") or "") == wanted_product]
        if exact:
            return exact[0]
    return candidates[0] if candidates else None


def build_seed_pool(
    config: DomainConfig,
    output_path: Path,
    *,
    count: int,
    seed: int,
    min_comments: int | None = None,
    exclude_keys: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Sample a matched-real seed pool from the domain's thread bank.

    `exclude_keys` holds `(source_product_dir, source_raw_post_id)` pairs to drop
    before sampling, and defaults to None so every existing caller is unchanged.
    It exists for one job: a **calibration** pool disjoint from the evaluation
    pools, so a realization matrix can be measured without any of the threads it
    will later be judged on (`tone_realization.py`, `docs/DECISIONS.md` G161).
    """

    minimum = max(1, int(min_comments or config.min_comments))
    bank = load_real_thread_bank(config.raw_discussions_dir)
    eligible = [item for item in _deduplicate_threads(bank) if item["comment_count"] >= minimum]
    if exclude_keys:
        eligible = [
            item
            for item in eligible
            if (str(item["source_product"]), str(item["post_id"])) not in exclude_keys
        ]
    if len(eligible) < count:
        raise ValueError(
            f"{config.domain_id} has only {len(eligible)} unique threads with >= {minimum} comments; "
            f"requested {count}."
        )

    selected = _distribution_preserving_sample(eligible, count=count, seed=seed)
    rows = []
    for index, item in enumerate(selected):
        post = item.get("post") or {}
        title = str(item.get("title") or "").strip()
        body = str(item.get("body") or "").strip()
        rows.append(
            {
                "seed_index": index,
                "title": title,
                "body": body,
                "content": "\n\n".join(part for part in (title, body) if part).strip(),
                "source_raw_post_id": str(item["post_id"]),
                "source_product": str(item["source_product"]),
                "source_product_dir": str(item["source_product"]),
                "source_file": str(item.get("source_file") or ""),
                "subreddit": str(item.get("subreddit") or ""),
                "real_num_comments": int(item["comment_count"]),
                "real_score": _safe_int(post.get("score", post.get("ups", 0))),
                "real_created_utc": post.get("created_utc"),
                "domain_id": config.domain_id,
            }
        )

    payload = {
        "meta": {
            "builder": "generalized_card.data.build_seed_pool",
            "domain": config.to_public_dict(),
            "count": len(rows),
            "eligible_threads": len(eligible),
            "min_comments": minimum,
            "sampling_seed": seed,
            "excluded_threads": len(exclude_keys or ()),
            "comment_count_summary": _count_summary(selected),
            "subreddit_counts": dict(Counter(str(item.get("subreddit") or "unknown") for item in selected)),
        },
        "seed_posts": rows,
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def audit_domain(config: DomainConfig) -> dict[str, Any]:
    bank = _deduplicate_threads(load_real_thread_bank(config.raw_discussions_dir))
    return {
        "domain_id": config.domain_id,
        "unique_threads": len(bank),
        "thresholds": {
            str(threshold): sum(1 for item in bank if int(item["comment_count"]) >= threshold)
            for threshold in (1, 3, 5, 8, 10, 15, 20)
        },
        "comments": sum(int(item["comment_count"]) for item in bank),
    }


def _load_post_metadata(raw_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    posts: dict[tuple[str, str], dict[str, Any]] = {}
    loaded_sources: set[Path] = set()
    for path in sorted(raw_dir.rglob("*.jsonl")):
        if any(
            path.name.endswith(suffix)
            for suffix in (".comments.jsonl", ".comments.raw.jsonl", ".raw.jsonl")
        ):
            continue
        for row in _read_json_lines(path):
            _record_post(posts, row, path)
        loaded_sources.add(path.with_suffix(".json"))

    # Some product folders contain only the aggregate JSON file.
    for path in sorted(raw_dir.rglob("*.json")):
        if path in loaded_sources or _is_derived_json(path):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = payload.get("posts") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                _record_post(posts, row, path)
    return posts


def _record_post(
    posts: dict[tuple[str, str], dict[str, Any]],
    row: dict[str, Any],
    path: Path,
) -> None:
    post_id = str(row.get("id") or row.get("post_id") or "").strip()
    if not post_id:
        return
    copied = dict(row)
    copied["_source_file"] = str(path.resolve())
    posts[(str(path.parent.resolve()), post_id)] = copied


def _load_comments(raw_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for path in sorted(raw_dir.rglob("*.comments.jsonl")):
        source_dir = str(path.parent.resolve())
        for row in _read_json_lines(path):
            body = str(row.get("body") or "").strip()
            if not usable_comment(body):
                continue
            post_id = str(row.get("post_id") or "").strip()
            comment_id = str(row.get("comment_id") or row.get("id") or "").strip()
            if not post_id or not comment_id:
                continue
            grouped.setdefault((source_dir, post_id), {}).setdefault(comment_id, row)
    return {
        key: sorted(rows.values(), key=_comment_order)
        for key, rows in grouped.items()
    }


def usable_comment(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered not in REMOVED and "i am a bot and this action was performed automatically" not in lowered


def _read_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _is_derived_json(path: Path) -> bool:
    name = path.name
    return (
        name == "subreddit_cache.json"
        or "summary" in name
        or name.endswith("_results.json")
        or name in {
            "politeness_results.json",
            "go_emotions_results.json",
            "storyseeker_results.json",
            "stance_disagreement_results.json",
            "self_bertscore_results.json",
            "self_bleu_results.json",
            "detoxify_results.json",
            "thread_structure_results.json",
            "semantic_uniformity_results.json",
        }
    )


def _comment_order(row: dict[str, Any]) -> tuple[float, int, str]:
    try:
        created = float(row.get("created_utc") or 0)
    except (TypeError, ValueError):
        created = 0.0
    return (created, _safe_int(row.get("depth")), str(row.get("comment_id") or ""))


def _deduplicate_threads(bank: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for item in bank:
        post_id = str(item["post_id"])
        old = chosen.get(post_id)
        if old is None or _thread_priority(item) > _thread_priority(old):
            chosen[post_id] = item
    return sorted(chosen.values(), key=lambda item: str(item["post_id"]))


def _thread_priority(item: dict[str, Any]) -> tuple[int, int, str]:
    post = item.get("post") or {}
    relevant = 1 if post.get("is_relevant", True) is not False else 0
    return (relevant, int(item["comment_count"]), str(item.get("source_product") or ""))


def _distribution_preserving_sample(
    eligible: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample across comment-count strata without hand-selecting easy threads."""

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in ("5-9", "10-19", "20-39", "40-79", "80+")}
    for item in eligible:
        n = int(item["comment_count"])
        key = "5-9" if n < 10 else "10-19" if n < 20 else "20-39" if n < 40 else "40-79" if n < 80 else "80+"
        buckets[key].append(item)
    for rows in buckets.values():
        rng.shuffle(rows)

    # Largest-remainder proportional allocation preserves the eligible real distribution.
    raw_targets = {key: count * len(rows) / len(eligible) for key, rows in buckets.items()}
    targets = {key: min(len(buckets[key]), int(value)) for key, value in raw_targets.items()}
    remaining = count - sum(targets.values())
    order = sorted(
        buckets,
        key=lambda key: (raw_targets[key] - int(raw_targets[key]), len(buckets[key]), key),
        reverse=True,
    )
    while remaining:
        progress = False
        for key in order:
            if targets[key] < len(buckets[key]):
                targets[key] += 1
                remaining -= 1
                progress = True
                if not remaining:
                    break
        if not progress:
            raise RuntimeError("Unable to allocate seed sample across strata")

    selected = [item for key in buckets for item in buckets[key][: targets[key]]]
    rng.shuffle(selected)
    return selected


def _count_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(int(item["comment_count"]) for item in rows)
    if not values:
        return {}
    return {
        "min": values[0],
        "median": values[len(values) // 2],
        "mean": sum(values) / len(values),
        "max": values[-1],
    }


def _safe_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0
