from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import praw
from dotenv import load_dotenv

from .io_utils import read_csv_rows, write_csv_rows, write_jsonl
from .text_utils import aliases_from_pipe, compact_text, text_mentions_alias


TARGET_SUBREDDITS = ("creditcards", "personalfinance", "churning")


def _build_reddit_client() -> praw.Reddit:
    load_dotenv()
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )


def _primary_subject(title: str, body: str, aliases: list[str]) -> bool:
    combined = compact_text(title, body).lower()
    title_lower = title.lower()
    alias_hits = sum(1 for alias in aliases if alias.lower() in combined)
    return alias_hits > 0 and any(alias.lower() in title_lower for alias in aliases)


def collect_reddit_items(
    seed_csv: Path,
    output_jsonl: Path,
    coverage_csv: Path,
    per_card_target: int = 100,
    max_submissions_per_query: int = 30,
) -> None:
    reddit = _build_reddit_client()
    cards = read_csv_rows(seed_csv)

    items: list[dict[str, object]] = []
    coverage: dict[str, dict[str, int]] = defaultdict(
        lambda: {"submission_count": 0, "comment_count": 0, "total_items": 0}
    )

    for card in cards:
        aliases = aliases_from_pipe(card["reddit_aliases"])
        seen_item_ids: set[str] = set()
        collected = 0

        for subreddit_name in TARGET_SUBREDDITS:
            subreddit = reddit.subreddit(subreddit_name)

            for alias in aliases[:3]:
                if collected >= per_card_target:
                    break

                for submission in subreddit.search(alias, sort="relevance", limit=max_submissions_per_query):
                    if collected >= per_card_target:
                        break

                    if not _primary_subject(submission.title, submission.selftext or "", aliases):
                        continue

                    submission_key = f"t3_{submission.id}"
                    if submission_key not in seen_item_ids:
                        items.append(
                            {
                                "card_id": card["card_id"],
                                "card_name": card["card_name"],
                                "issuer": card["issuer"],
                                "item_type": "submission",
                                "subreddit": subreddit_name,
                                "item_id": submission.id,
                                "root_submission_id": submission.id,
                                "parent_id": "",
                                "title": submission.title,
                                "body": submission.selftext or "",
                                "score": int(submission.score),
                                "depth": 0,
                                "created_utc": int(submission.created_utc),
                                "permalink": f"https://www.reddit.com{submission.permalink}",
                            }
                        )
                        seen_item_ids.add(submission_key)
                        coverage[card["card_id"]]["submission_count"] += 1
                        coverage[card["card_id"]]["total_items"] += 1
                        collected += 1

                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list():
                        if collected >= per_card_target:
                            break
                        body = getattr(comment, "body", "")
                        if not text_mentions_alias(body, aliases):
                            continue
                        comment_key = f"t1_{comment.id}"
                        if comment_key in seen_item_ids:
                            continue

                        items.append(
                            {
                                "card_id": card["card_id"],
                                "card_name": card["card_name"],
                                "issuer": card["issuer"],
                                "item_type": "comment",
                                "subreddit": subreddit_name,
                                "item_id": comment.id,
                                "root_submission_id": submission.id,
                                "parent_id": comment.parent_id,
                                "title": submission.title,
                                "body": body,
                                "score": int(comment.score),
                                "depth": getattr(comment, "depth", 0),
                                "created_utc": int(comment.created_utc),
                                "permalink": f"https://www.reddit.com{comment.permalink}",
                            }
                        )
                        seen_item_ids.add(comment_key)
                        coverage[card["card_id"]]["comment_count"] += 1
                        coverage[card["card_id"]]["total_items"] += 1
                        collected += 1

                if collected >= per_card_target:
                    break

    write_jsonl(output_jsonl, items)

    coverage_rows: list[dict[str, object]] = []
    for card in cards:
        stats = coverage[card["card_id"]]
        coverage_rows.append(
            {
                "card_id": card["card_id"],
                "card_name": card["card_name"],
                "issuer": card["issuer"],
                "submission_count": stats["submission_count"],
                "comment_count": stats["comment_count"],
                "total_items": stats["total_items"],
            }
        )

    write_csv_rows(
        coverage_csv,
        coverage_rows,
        ["card_id", "card_name", "issuer", "submission_count", "comment_count", "total_items"],
    )
