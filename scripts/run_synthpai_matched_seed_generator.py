#!/usr/bin/env python3
"""Run SynthPAI comment generation on fixed real seed posts."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


FEATURES = [
    "city_country",
    "age",
    "income_level",
    "income",
    "education",
    "occupation",
    "sex",
    "relationship_status",
    "birth_city_country",
]

SYNTHPAI_COMMENT_LABEL_RE = re.compile(
    r"\bmy[\s_-]*(?:new[\s_-]*)?comment\s*:\s*",
    re.IGNORECASE,
)
SYNTHPAI_INTERNAL_TEXT_RE = re.compile(
    r"\b("
    r"here is what i know about this subthread|"
    r"here is what i know about myself|"
    r"summarize the topic of this subreddit|"
    r"reply to its last comment|"
    r"\[your new comment\]"
    r")|"
    r"^\s*(guess|certainty|hardness|reasoning|ground truth|model ans):\s",
    re.IGNORECASE,
)
BAD_GENERATED_COMMENT_RE = re.compile(
    r"\b("
    r"lorem ipsum|placeholder|insert comment|sample text|dummy text|"
    r"i[' ]?m sorry,?\s+but\s+i\s+can[' ]?t\s+help|"
    r"i\s+cannot\s+(assist|help)\s+with|"
    r"as\s+an\s+ai\s+language\s+model|"
    r"rate limit|quota exceeded|traceback|apierror|invalid response object|"
    r"contents is not specified|please pass a valid api key"
    r")\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SynthPAI baseline threads from a fixed real seed-post pool."
    )
    parser.add_argument(
        "--seed-post-pool-json",
        type=Path,
        default=Path("artifacts/seed_posts/credit_cards_test_real_distribution_seed_pool_154_20260609.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--synthpai-dir", type=Path, default=Path("SynthPAI"))
    parser.add_argument(
        "--config-path",
        default="configs/thread/thread_gpt4omini_city_country.yaml",
        help="SynthPAI thread config, relative to --synthpai-dir unless absolute.",
    )
    parser.add_argument(
        "--reference-scores-csv",
        type=Path,
        default=None,
        help="Optional CARD score CSV. When set, use its seed_index order instead of a contiguous slice.",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--env-file", type=Path, default=Path("third_party/MiroFish/.env"))
    parser.add_argument("--start-seed-index", type=int, default=0)
    parser.add_argument("--max-seeds", type=int, default=150)
    parser.add_argument("--posts-per-run", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thread-retries", type=int, default=0)
    parser.add_argument("--retry-delay", type=float, default=60.0)
    parser.add_argument("--sleep-between-seeds", type=float, default=0.0)
    parser.add_argument(
        "--min-comments-per-post",
        type=int,
        default=1,
        help="Retry or fail a seed when exported comments are below this count.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_env_file(args.env_file)
    api_key = _api_key()
    if not api_key and not args.dry_run:
        raise SystemExit("Set OPENAI_API_KEY or LLM_API_KEY before running SynthPAI.")

    seed_posts = _load_seed_posts(args.seed_post_pool_json)
    selected = _select_seed_records(seed_posts, args)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "matched_seed_manifest.json",
        {
            "seed_post_pool_json": str(args.seed_post_pool_json),
            "baseline": "synthpai",
            "model": args.model,
            "start_seed_index": args.start_seed_index,
            "max_seeds": args.max_seeds,
            "posts_per_run": args.posts_per_run,
            "reference_scores_csv": str(args.reference_scores_csv) if args.reference_scores_csv else None,
            "selected_seed_indices": [seed_index for seed_index, _seed in selected],
            "created_at": datetime.now().isoformat(),
        },
    )

    if args.dry_run:
        _write_dry_run_outputs(args, selected, output_root)
        print(f"[done] synthpai_matched_seed_dry_run_root={output_root}")
        return

    synthpai_dir = args.synthpai_dir.expanduser().resolve()
    cfg_path = Path(args.config_path)
    if not cfg_path.is_absolute():
        cfg_path = synthpai_dir / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"SynthPAI config not found: {cfg_path}")

    sys.path.insert(0, str(synthpai_dir))
    os.chdir(synthpai_dir)

    import openai  # type: ignore
    from src.models.model_factory import get_model  # type: ignore
    from src.prompts import Conversation, Prompt  # type: ignore
    from src.thread.run_thread import Node, RedditThread  # type: ignore
    from src.thread.user_bot_system_prompt_builder import build_tagging_comment_prompt  # type: ignore
    from src.utils.initialization import read_config_from_yaml, seed_everything  # type: ignore

    openai.api_key = api_key
    openai.organization = os.environ.get("OPENAI_ORG", "")
    if args.base_url and args.base_url != "https://api.openai.com/v1":
        openai.api_base = args.base_url

    cfg = read_config_from_yaml(str(cfg_path))
    seed_everything(args.seed)
    _override_model(cfg, args.model)

    author = get_model(cfg.task_config.author_bot)
    user = get_model(cfg.task_config.user_bot)
    checker = get_model(cfg.task_config.checker_bot)

    author_prompt = _read_text(Path(cfg.task_config.author_bot_system_prompt_path))
    user_prompt = _read_text(Path(cfg.task_config.user_bot_system_prompt_path))
    profile_checker_prompt = _read_text(Path(cfg.task_config.profile_checker_prompt_path))
    user_profiles = _read_json(Path(cfg.task_config.user_bot_personalities_path))

    del author_prompt  # Fixed-root mode does not ask SynthPAI to write the root post.
    rng = random.Random(args.seed)

    total_runs = (len(selected) + args.posts_per_run - 1) // args.posts_per_run
    for run_id in range(total_runs):
        batch_start = run_id * args.posts_per_run
        batch = selected[batch_start : batch_start + args.posts_per_run]
        seed_indices = [seed_index for seed_index, _seed in batch]
        run_dir = output_root / f"run_{run_id:03d}_sampled_reddit"
        run_dir.mkdir(parents=True, exist_ok=True)
        discussion_path = run_dir / "discussion.json"
        discussion = _load_or_new_discussion(discussion_path, args.force)
        completed = {
            int(post.get("seed_index"))
            for post in discussion.get("posts", [])
            if str(post.get("seed_index", "")).strip().isdigit()
        }

        print(
            f"[synthpai-run] run={run_id:03d}/{total_runs - 1:03d} "
            f"seeds={','.join(str(seed_index) for seed_index in seed_indices)}"
        )
        for slot, (seed_index, seed) in enumerate(batch):
            if seed_index in completed and not args.force:
                print(f"[synthpai-resume] seed={seed_index} already exists")
                continue
            feature = FEATURES[seed_index % len(FEATURES)]
            print(f"[synthpai-thread] seed={seed_index} slot={slot} feature={feature}")
            post = _generate_one_post_with_retries(
                args=args,
                seed=seed,
                seed_index=seed_index,
                post_slot=slot,
                run_id=run_id,
                feature=feature,
                cfg=cfg,
                user_profiles=user_profiles,
                user_model=user,
                checker_model=checker,
                profile_checker_prompt=profile_checker_prompt,
                user_prompt=user_prompt,
                rng=rng,
                Node=Node,
                RedditThread=RedditThread,
                Conversation=Conversation,
                Prompt=Prompt,
                build_tagging_comment_prompt=build_tagging_comment_prompt,
            )
            _replace_post(discussion, post)
            _write_json(discussion_path, discussion)
            print(
                f"[synthpai-done-thread] seed={seed_index} "
                f"comments={_count_comments(post.get('comments') or [])}"
            )
            if args.sleep_between_seeds > 0:
                print(f"[synthpai-sleep] seconds={args.sleep_between_seeds:g}")
                time.sleep(args.sleep_between_seeds)

    print(f"[done] synthpai_matched_seed_root={output_root}")


def _generate_one_post_with_retries(*, args: argparse.Namespace, **kwargs: Any) -> dict[str, Any]:
    attempts = max(0, args.thread_retries) + 1
    seed_index = int(kwargs["seed_index"])
    for attempt in range(1, attempts + 1):
        try:
            post = _generate_one_post(**kwargs)
            comment_count = _count_comments(post.get("comments") or [])
            if comment_count < args.min_comments_per_post:
                raise RuntimeError(
                    f"SynthPAI thread failed quality gate "
                    f"min_comments_per_post={args.min_comments_per_post}: "
                    f"seed={seed_index} comments={comment_count}"
                )
            return post
        except Exception as exc:
            if attempt >= attempts:
                raise
            print(
                f"[synthpai-retry] seed={seed_index} attempt={attempt}/{attempts} "
                f"error={type(exc).__name__}: {exc} sleep={args.retry_delay:g}s",
                flush=True,
            )
            time.sleep(args.retry_delay)
    raise RuntimeError("unreachable")


def _generate_one_post(
    *,
    seed: dict[str, Any],
    seed_index: int,
    post_slot: int,
    run_id: int,
    feature: str,
    cfg: Any,
    user_profiles: dict[str, dict[str, Any]],
    user_model: Any,
    checker_model: Any,
    profile_checker_prompt: str,
    user_prompt: str,
    rng: random.Random,
    Node: Any,
    RedditThread: Any,
    Conversation: Any,
    Prompt: Any,
    build_tagging_comment_prompt: Any,
) -> dict[str, Any]:
    del build_tagging_comment_prompt  # Comment tagging remains inside SynthPAI's add_comment.
    thread = RedditThread()
    keys = list(user_profiles.keys())
    rng.shuffle(keys)
    thread_author = keys[seed_index % len(keys)]
    thread_profile = user_profiles[thread_author]
    available_profiles = {key: user_profiles[key] for key in keys if key != thread_author}
    root_text = _seed_content(seed)
    username = thread_profile.get("username", str(thread_author))
    thread.root = Node(thread_author, thread_profile, username, root_text, Node(None, None, None, None, None))
    thread.root.guesses = []
    thread.comments.append(thread.root)

    sampled = thread.choose_profiles(
        checker_model,
        profile_checker_prompt,
        available_profiles,
        root_text,
        cfg.task_config.no_profiles,
    )
    sampled_profiles = {key: available_profiles[key] for key in sampled if key in available_profiles}
    if not sampled_profiles:
        sampled_profiles = {
            key: available_profiles[key]
            for key in list(available_profiles.keys())[: min(5, len(available_profiles))]
        }
        print(f"[synthpai-fallback] seed={seed_index} sampled_profiles={len(sampled_profiles)}")

    for pers in sampled_profiles:
        path_empty: list[Any] = []
        score, path = thread.score_path(pers, path_empty, thread.root, score=0.0)
        thread.root.scores[pers] = float(score / float(max(1, len(path))))

    nc = cfg.task_config.no_actions
    default_comment_prob = cfg.task_config.default_comment_prob
    default_abstain_prob = 10 - default_comment_prob
    no_profiles = cfg.task_config.no_profiles
    p_critic = cfg.task_config.p_critic

    for round_idx in range(cfg.task_config.no_rounds):
        sampled_keys = list(sampled_profiles.keys())
        if not sampled_keys:
            continue
        n_critics = min(int(no_profiles * p_critic), len(sampled_keys))
        thread.critics = np.random.choice(sampled_keys, n_critics) if n_critics > 0 else []
        for pers, profile in list(sampled_profiles.items()):
            choices = ["comment", "abstain"]
            for _ in range(random.choice(list(range(1, nc + 1)))):
                if round_idx < default_comment_prob:
                    prob_comment = default_comment_prob - round_idx
                    prob_abstain = default_abstain_prob + round_idx
                else:
                    prob_comment = 1
                    prob_abstain = 9
                action = random.choices(choices, weights=[prob_comment, prob_abstain])[0]
                if action != "comment":
                    continue
                try:
                    thread.add_comment(
                        user_model,
                        pers,
                        profile,
                        user_prompt,
                        cfg.task_config.min_comment_len,
                        cfg.task_config.max_comment_len,
                        sampled_profiles,
                        cfg,
                        checker_model,
                    )
                except Exception as exc:
                    print(f"[synthpai-warn] seed={seed_index} profile={pers} error={exc}")
        if round_idx == 0 and thread.comments and thread.comments[0] is thread.root:
            thread.comments.remove(thread.root)

    return _thread_to_geo_post(
        thread=thread,
        seed=seed,
        seed_index=seed_index,
        post_slot=post_slot,
        run_id=run_id,
        feature=feature,
    )


def _thread_to_geo_post(
    *,
    thread: Any,
    seed: dict[str, Any],
    seed_index: int,
    post_slot: int,
    run_id: int,
    feature: str,
) -> dict[str, Any]:
    counter = [0]
    post_id = f"sampled_run{run_id:02d}_post{post_slot:02d}_seed{seed_index:03d}"
    return {
        "post_id": post_id,
        "post_slot": post_slot,
        "seed_index": seed_index,
        "source_raw_post_id": _clean_text(seed.get("source_raw_post_id")),
        "source_product_dir": _clean_text(seed.get("source_product") or seed.get("source_product_dir")),
        "source_file": _clean_text(seed.get("source_file")),
        "post_type": _clean_text(seed.get("post_type") or "real_seed"),
        "title": _clean_text(seed.get("title") or ""),
        "author": thread.root.username or "OP",
        "author_karma": 1000,
        "content": thread.root.text or _seed_content(seed),
        "timestamp": "",
        "likes": 0,
        "dislikes": 0,
        "synthpai_feature": feature,
        "comments": _node_children_to_comments(thread.root.children, counter, depth=0, parent_id=None),
    }


def _node_children_to_comments(
    children: list[Any],
    counter: list[int],
    *,
    depth: int,
    parent_id: int | None,
) -> list[dict[str, Any]]:
    comments = []
    for child in children:
        content = _clean_generated_comment_text(child.text or "")
        if _is_bad_generated_comment(content):
            print(
                "[synthpai-skip-comment] "
                f"depth={depth} reason=non_discussion_text "
                f"text={_clean_text(child.text or '')[:160]}"
            )
            comments.extend(
                _node_children_to_comments(
                    child.children,
                    counter,
                    depth=depth,
                    parent_id=parent_id,
                )
            )
            continue
        counter[0] += 1
        cid = counter[0]
        comments.append(
            {
                "comment_id": cid,
                "author": child.username or child.author or f"user_{cid}",
                "author_karma": 1000,
                "content": content,
                "timestamp": "",
                "likes": 0,
                "dislikes": 0,
                "parent_comment_id": parent_id,
                "depth": depth,
                "replies": _node_children_to_comments(
                    child.children,
                    counter,
                    depth=depth + 1,
                    parent_id=cid,
                ),
            }
        )
    return comments


def _clean_generated_comment_text(value: Any) -> str:
    text = " ".join(_clean_text(value).split()).strip()
    if not text:
        return ""

    matches = list(SYNTHPAI_COMMENT_LABEL_RE.finditer(text))
    if matches:
        text = text[matches[-1].end() :].strip()
    return text.strip().strip('"').strip("'").strip()


def _is_bad_generated_comment(text: str) -> bool:
    if not text:
        return True
    if SYNTHPAI_INTERNAL_TEXT_RE.search(text):
        return True
    if BAD_GENERATED_COMMENT_RE.search(text):
        return True
    return False


def _override_model(cfg: Any, model: str) -> None:
    cfg.gen_model.name = model
    cfg.task_config.author_bot.name = model
    cfg.task_config.user_bot.name = model
    cfg.task_config.checker_bot.name = model


def _write_dry_run_outputs(
    args: argparse.Namespace,
    selected: list[tuple[int, dict[str, Any]]],
    output_root: Path,
) -> None:
    total_runs = (len(selected) + args.posts_per_run - 1) // args.posts_per_run
    for run_id in range(total_runs):
        run_dir = output_root / f"run_{run_id:03d}_sampled_reddit"
        run_dir.mkdir(parents=True, exist_ok=True)
        posts = []
        batch_start = run_id * args.posts_per_run
        batch = selected[batch_start : batch_start + args.posts_per_run]
        for slot, (seed_index, seed) in enumerate(batch):
            posts.append(
                {
                    "post_id": f"sampled_run{run_id:02d}_post{slot:02d}_seed{seed_index:03d}",
                    "post_slot": slot,
                    "seed_index": seed_index,
                    "source_raw_post_id": _clean_text(seed.get("source_raw_post_id")),
                    "source_product_dir": _clean_text(seed.get("source_product") or seed.get("source_product_dir")),
                    "source_file": _clean_text(seed.get("source_file")),
                    "title": _clean_text(seed.get("title") or ""),
                    "author": "dry_run",
                    "author_karma": 0,
                    "content": _seed_content(seed),
                    "timestamp": "",
                    "likes": 0,
                    "dislikes": 0,
                    "comments": [],
                }
            )
        _write_json(run_dir / "discussion.json", _new_discussion(posts))


def _load_or_new_discussion(path: Path, force: bool) -> dict[str, Any]:
    if path.exists() and not force:
        return _read_json(path)
    return _new_discussion([])


def _new_discussion(posts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "meta": {
            "product_category": "credit_cards",
            "baseline": "synthpai",
            "created_at": datetime.now().isoformat(),
        },
        "posts": posts,
    }


def _replace_post(discussion: dict[str, Any], post: dict[str, Any]) -> None:
    posts = list(discussion.get("posts") or [])
    slot = int(post.get("post_slot", 0))
    replaced = False
    for idx, existing in enumerate(posts):
        if int(existing.get("post_slot", -1)) == slot:
            posts[idx] = post
            replaced = True
            break
    if not replaced:
        posts.append(post)
    posts.sort(key=lambda item: int(item.get("post_slot", 0)))
    discussion["posts"] = posts


def _count_comments(comments: list[dict[str, Any]]) -> int:
    total = 0
    for comment in comments:
        total += 1 + _count_comments(comment.get("replies") or [])
    return total


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def _api_key() -> str:
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_KEY")
        or os.environ.get("PLANNER_API_KEY")
        or ""
    )


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


def _seed_content(seed: dict[str, Any]) -> str:
    content = _clean_text(seed.get("content")).strip()
    if content:
        return content
    title = _clean_text(seed.get("title")).strip()
    body = _clean_text(seed.get("body")).strip()
    return "\n\n".join(part for part in (title, body) if part).strip()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    try:
        text = text.encode("utf-16", "surrogatepass").decode("utf-16")
    except UnicodeError:
        pass
    return text.encode("utf-8", "replace").decode("utf-8")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
