#!/usr/bin/env python3
"""Run OASIS on the same real seed posts used by sampled CARD runs."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from product_reddit_sim.exporter import export_discussion
from product_reddit_sim.runner import run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OASIS baseline threads from a fixed real seed-post pool."
    )
    parser.add_argument(
        "--seed-post-pool-json",
        type=Path,
        default=Path("artifacts/seed_posts/credit_cards_test_real_distribution_seed_pool_154_20260609.json"),
        help="Seed pool JSON containing real Credit Card root posts.",
    )
    parser.add_argument(
        "--template-run-dir",
        type=Path,
        default=Path("artifacts/baselines/credit_cards_gpt4omini/vanilla/runs/credit_cards_20260507_025804"),
        help="Existing OASIS run whose profiles/config are reused.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root containing run_*_sampled_reddit directories.",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--start-seed-index", type=int, default=0)
    parser.add_argument("--max-seeds", type=int, default=150)
    parser.add_argument("--posts-per-run", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--run-retries", type=int, default=0)
    parser.add_argument("--retry-delay", type=float, default=60.0)
    parser.add_argument("--sleep-between-runs", type=float, default=0.0)
    parser.add_argument(
        "--min-comments-per-post",
        type=int,
        default=1,
        help="Retry or fail a generated run when any exported seed thread has fewer comments.",
    )
    parser.add_argument("--env-file", type=Path, default=Path("third_party/MiroFish/.env"))
    parser.add_argument(
        "--reference-scores-csv",
        type=Path,
        default=None,
        help="Optional CARD score CSV. When set, use its seed_index order instead of a contiguous slice.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate existing run directories.")
    parser.add_argument("--dry-run", action="store_true", help="Write configs only; do not call OASIS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_env_file(args.env_file)
    _ensure_api_key()

    seed_posts = _load_seed_posts(args.seed_post_pool_json)
    selected = _select_seed_records(seed_posts, args)

    template_dir = args.template_run_dir.expanduser().resolve()
    _check_template(template_dir)
    profiles = _read_json(template_dir / "reddit_profiles.json")
    template_config = _read_json(template_dir / "simulation_config.json")
    template_analysis = _read_json(template_dir / "product_analysis.json")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matched_seed_manifest.json").write_text(
        json.dumps(
            {
                "seed_post_pool_json": str(args.seed_post_pool_json),
                "template_run_dir": str(template_dir),
                "model": args.model,
                "start_seed_index": args.start_seed_index,
                "max_seeds": args.max_seeds,
                "posts_per_run": args.posts_per_run,
                "reference_scores_csv": str(args.reference_scores_csv) if args.reference_scores_csv else None,
                "selected_seed_indices": [seed_index for seed_index, _seed in selected],
                "created_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total_runs = (len(selected) + args.posts_per_run - 1) // args.posts_per_run
    for run_id in range(total_runs):
        batch_start = run_id * args.posts_per_run
        batch = selected[batch_start : batch_start + args.posts_per_run]
        seed_indices = [seed_index for seed_index, _seed in batch]
        run_dir = output_root / f"run_{run_id:03d}_sampled_reddit"
        if (run_dir / "discussion.json").exists() and not args.force:
            print(f"[oasis-resume] run={run_id:03d} exists -> {run_dir}")
            continue
        if run_dir.exists():
            if args.force:
                print(f"[oasis-force] removing existing run={run_id:03d} -> {run_dir}")
                shutil.rmtree(run_dir)
            else:
                print(f"[oasis-partial] removing incomplete run={run_id:03d} -> {run_dir}")
                shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[oasis-run] run={run_id:03d}/{total_runs - 1:03d} "
            f"seeds={','.join(str(seed_index) for seed_index in seed_indices)} "
            f"posts={len(batch)}"
        )
        _prepare_run_dir(
            run_dir=run_dir,
            run_id=run_id,
            batch=batch,
            profiles=profiles,
            template_config=template_config,
            template_analysis=template_analysis,
            args=args,
        )
        if args.dry_run:
            print(f"[oasis-dry-run] wrote {run_dir / 'simulation_config.json'}")
            continue

        _run_and_export_with_retries(
            run_dir=run_dir,
            run_id=run_id,
            batch=batch,
            profiles=profiles,
            template_config=template_config,
            template_analysis=template_analysis,
            args=args,
        )
        print(f"[oasis-done] run={run_id:03d} -> {run_dir / 'discussion.json'}")
        if args.sleep_between_runs > 0 and run_id < total_runs - 1:
            print(f"[oasis-sleep] seconds={args.sleep_between_runs:g}")
            time.sleep(args.sleep_between_runs)

    print(f"[done] oasis_matched_seed_root={output_root}")


def _run_and_export_with_retries(
    *,
    run_dir: Path,
    run_id: int,
    batch: list[tuple[int, dict[str, Any]]],
    profiles: list[dict[str, Any]],
    template_config: dict[str, Any],
    template_analysis: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    attempts = max(0, args.run_retries) + 1
    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
                _prepare_run_dir(
                    run_dir=run_dir,
                    run_id=run_id,
                    batch=batch,
                    profiles=profiles,
                    template_config=template_config,
                    template_analysis=template_analysis,
                    args=args,
                )
            config_path = run_dir / "simulation_config.json"
            run_simulation(str(config_path), max_rounds=args.rounds)
            db_path = run_dir / "reddit_simulation.db"
            if not db_path.exists():
                raise FileNotFoundError(f"OASIS DB not found after simulation: {db_path}")
            export_discussion(
                str(db_path),
                str(run_dir / "reddit_profiles.json"),
                str(run_dir),
                meta={
                    "product_category": "credit_cards",
                    "baseline": "oasis",
                    "model": args.model,
                    "run_id": run_dir.name,
                    "seed_indices": [seed_index for seed_index, _seed in batch],
                    "seed_count": len(batch),
                    "rounds": args.rounds,
                    "simulated_hours": args.hours,
                },
            )
            _attach_seed_metadata(run_dir / "discussion.json", batch=batch)
            _validate_discussion_quality(
                run_dir / "discussion.json",
                min_comments_per_post=args.min_comments_per_post,
            )
            return
        except Exception as exc:
            if attempt >= attempts:
                raise
            print(
                f"[oasis-retry] run={run_id:03d} attempt={attempt}/{attempts} "
                f"error={type(exc).__name__}: {exc} sleep={args.retry_delay:g}s",
                flush=True,
            )
            time.sleep(args.retry_delay)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def _ensure_api_key() -> None:
    api_key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_KEY")
        or os.environ.get("PLANNER_API_KEY")
    )
    if not api_key:
        raise SystemExit("Set LLM_API_KEY or OPENAI_API_KEY before running OASIS.")
    os.environ.setdefault("LLM_API_KEY", api_key)


def _load_seed_posts(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    posts = data.get("seed_posts") or data.get("posts") or data.get("initial_posts")
    if not isinstance(posts, list):
        raise ValueError(f"Cannot find seed post list in {path}")
    return posts


def _select_seed_records(seed_posts: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    if args.reference_scores_csv:
        seed_indices = _load_reference_seed_indices(args.reference_scores_csv)
        selected_indices = seed_indices[args.start_seed_index : args.start_seed_index + args.max_seeds]
    else:
        selected_indices = list(range(args.start_seed_index, args.start_seed_index + args.max_seeds))
    if len(selected_indices) < args.max_seeds:
        raise SystemExit(
            f"Requested {args.max_seeds} seeds from offset {args.start_seed_index}, "
            f"but only {len(selected_indices)} are available."
        )
    missing = [idx for idx in selected_indices if idx < 0 or idx >= len(seed_posts)]
    if missing:
        raise SystemExit(f"Selected seed indices outside seed pool: {missing[:10]}")
    return [(idx, seed_posts[idx]) for idx in selected_indices]


def _load_reference_seed_indices(path: Path) -> list[int]:
    import csv

    indices: list[int] = []
    seen: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = str(row.get("seed_index", "")).strip()
            if not raw:
                continue
            idx = int(float(raw))
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
    if not indices:
        raise ValueError(f"No seed_index values found in reference scores CSV: {path}")
    return indices


def _check_template(template_dir: Path) -> None:
    required = ["product_analysis.json", "reddit_profiles.json", "simulation_config.json"]
    missing = [name for name in required if not (template_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Template run is missing {missing}: {template_dir}")


def _prepare_run_dir(
    *,
    run_dir: Path,
    run_id: int,
    batch: list[tuple[int, dict[str, Any]]],
    profiles: list[dict[str, Any]],
    template_config: dict[str, Any],
    template_analysis: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    config = copy.deepcopy(template_config)
    config["simulation_id"] = run_dir.name
    config["graph_id"] = "credit_cards"
    config["llm_model"] = args.model
    config["llm_base_url"] = args.base_url
    config["generated_at"] = datetime.now().isoformat()
    if isinstance(config.get("time_config"), dict):
        config["time_config"]["total_simulation_hours"] = args.hours

    agent_count = max(1, len(profiles))
    initial_posts = []
    seed_records = []
    for slot, (seed_index, seed) in enumerate(batch):
        poster_id = _safe_int(seed.get("poster_agent_id"), seed_index) % agent_count
        content = _seed_content(seed)
        initial_posts.append(
            {
                "poster_agent_id": poster_id,
                "content": content,
                "post_type": seed.get("post_type") or "real_seed",
                "seed_index": seed_index,
                "source_raw_post_id": seed.get("source_raw_post_id"),
            }
        )
        seed_records.append(_seed_record(seed, seed_index, slot, content))

    event_config = dict(config.get("event_config") or {})
    event_config["initial_posts"] = initial_posts
    event_config["allow_new_threads_during_simulation"] = False
    event_config["max_total_threads"] = len(initial_posts)
    config["event_config"] = event_config

    analysis = copy.deepcopy(template_analysis)
    analysis["_matched_seed_baseline"] = {
        "baseline": "oasis",
        "model": args.model,
        "run_id": run_id,
        "seed_indices": [seed_index for seed_index, _seed in batch],
        "seed_count": len(batch),
    }
    _write_json(run_dir / "product_analysis.json", analysis)
    _write_json(run_dir / "reddit_profiles.json", profiles)
    _write_json(run_dir / "simulation_config.json", config)
    _write_json(run_dir / "matched_seed_posts.json", seed_records)


def _attach_seed_metadata(path: Path, *, batch: list[tuple[int, dict[str, Any]]]) -> None:
    discussion = _read_json(path)
    seed_records = [
        _seed_record(seed, seed_index, slot, _seed_content(seed))
        for slot, (seed_index, seed) in enumerate(batch)
    ]
    by_content = {record["content"]: record for record in seed_records}
    unmatched = []
    kept_posts = []
    used_slots: set[int] = set()

    for post in discussion.get("posts") or []:
        record = by_content.get(str(post.get("content") or ""))
        if record is None:
            unmatched.append(post)
            continue
        _apply_post_metadata(post, record)
        used_slots.add(int(record["post_slot"]))
        kept_posts.append(post)

    if len(kept_posts) < len(seed_records):
        for record, post in zip(
            [r for r in seed_records if int(r["post_slot"]) not in used_slots],
            unmatched,
        ):
            _apply_post_metadata(post, record)
            kept_posts.append(post)

    kept_posts = kept_posts[: len(seed_records)]
    kept_posts.sort(key=lambda item: _safe_int(item.get("post_slot"), 0))
    discussion["posts"] = kept_posts
    _write_json(path, discussion)


def _apply_post_metadata(post: dict[str, Any], record: dict[str, Any]) -> None:
    post["post_slot"] = record["post_slot"]
    post["seed_index"] = record["seed_index"]
    post["source_raw_post_id"] = record["source_raw_post_id"]
    post["source_product_dir"] = record["source_product_dir"]
    post["source_file"] = record["source_file"]
    post["post_type"] = record["post_type"]
    post["title"] = record["title"]


def _validate_discussion_quality(path: Path, *, min_comments_per_post: int) -> None:
    if min_comments_per_post <= 0:
        return
    discussion = _read_json(path)
    failed = []
    for post in discussion.get("posts") or []:
        count = _count_comments(post.get("comments") or post.get("replies") or [])
        if count < min_comments_per_post:
            failed.append((post.get("seed_index"), count))
    if failed:
        details = ", ".join(f"seed={seed}:comments={count}" for seed, count in failed[:10])
        raise RuntimeError(
            f"Generated OASIS run failed quality gate "
            f"min_comments_per_post={min_comments_per_post}: {details}"
        )


def _count_comments(comments: Any) -> int:
    if isinstance(comments, dict):
        return 1 + _count_comments(comments.get("replies") or comments.get("children") or comments.get("comments") or [])
    if not isinstance(comments, list):
        return 0
    total = 0
    for comment in comments:
        total += _count_comments(comment)
    return total


def _seed_record(seed: dict[str, Any], seed_index: int, post_slot: int, content: str) -> dict[str, Any]:
    return {
        "seed_index": seed_index,
        "post_slot": post_slot,
        "source_raw_post_id": _clean_text(seed.get("source_raw_post_id")),
        "source_product_dir": _clean_text(seed.get("source_product") or seed.get("source_product_dir")),
        "source_file": _clean_text(seed.get("source_file")),
        "post_type": _clean_text(seed.get("post_type") or "real_seed"),
        "title": _clean_text(seed.get("title") or ""),
        "content": content,
    }


def _seed_content(seed: dict[str, Any]) -> str:
    content = _clean_text(seed.get("content")).strip()
    if content:
        return content
    title = _clean_text(seed.get("title")).strip()
    body = _clean_text(seed.get("body")).strip()
    return "\n\n".join(part for part in (title, body) if part).strip()


def _clean_text(value: Any) -> str:
    """Normalize JSON strings that contain UTF-16 surrogate escape fragments."""
    if value is None:
        return ""
    text = str(value)
    try:
        text = text.encode("utf-16", "surrogatepass").decode("utf-16")
    except UnicodeError:
        pass
    return text.encode("utf-8", "replace").decode("utf-8")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _read_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
