from __future__ import annotations

from pathlib import Path
from sampling_generator.engine.model import SeedPost
from sampling_generator.engine.util import compact
from sampling_generator.engine.util import first_line
from sampling_generator.engine.util import utc_now
from sampling_generator.engine.vocabulary import SYSTEM_PROMPTS
from typing import Any
import argparse
import json
import os

DEFAULT_PLANNER_MODEL = "gpt-5.4-mini"

DEFAULT_WRITER_MODEL = "reddit-qwen3"

DEFAULT_WRITER_BASE_URL = "http://127.0.0.1:11435/v1"

def is_recoverable_post_error(error: Exception) -> bool:
    """Retry model/output failures without hiding programming or config bugs."""

    message = str(error).lower()
    permanent_markers = (
        "insufficient balance",
        "invalid api key",
        "authentication",
        "unauthorized",
        "permission denied",
        "unsupported value",
        "core contract mismatch",
    )
    if any(marker in message for marker in permanent_markers):
        return False
    if isinstance(error, (RuntimeError, ValueError, TimeoutError, ConnectionError, OSError)):
        return True
    module = type(error).__module__.split(".", 1)[0]
    return module in {"openai", "httpx", "httpcore", "urllib3"}

def record_post_failure(
    *,
    output_dir: Path,
    run_index: int,
    post_slot: int,
    seed_index: int,
    attempt: int,
    error: Exception,
) -> None:
    path = output_dir / "_generation_failures" / "post_retries.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "recorded_at": utc_now(),
        "run_index": run_index,
        "post_slot": post_slot,
        "seed_index": seed_index,
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error": compact(str(error), 2000),
        "action": "retry_same_post",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sampled planner + Qwen writer Reddit generator.")
    parser.add_argument("--seed-post-pool-json", required=False, default="")
    parser.add_argument("--real-comments-dir", default="data/raw/discussions/credit_cards")
    parser.add_argument("--output-dir", default="artifacts/sample_planner_qwen_generator_v1")
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--posts-per-run", type=int, default=6)
    parser.add_argument("--start-seed-index", type=int, default=0)
    parser.add_argument("--wrap-seed-posts", action="store_true")
    parser.add_argument("--max-total-posts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-comments-per-post", type=int, default=80)
    parser.add_argument("--comment-count-scale", type=float, default=0.65)
    parser.add_argument("--exact-matched-thread-size", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--planner-model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument("--planner-base-url", default=os.environ.get("PLANNER_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--planner-api-key", default=os.environ.get("PLANNER_API_KEY", ""))
    parser.add_argument("--planner-retries", type=int, default=2)
    parser.add_argument("--planner-max-tokens", type=int, default=10000)
    parser.add_argument(
        "--planner-timeout",
        type=float,
        default=float(os.environ.get("PLANNER_TIMEOUT", "600")),
        help="Per-request timeout in seconds for planner API calls.",
    )
    parser.add_argument(
        "--comment-planner-max-tokens",
        type=int,
        default=12000,
        help="Max tokens for the per-comment semantic-move planner.",
    )
    parser.add_argument(
        "--comment-planner-batch-size",
        type=int,
        default=8,
        help="Number of matched real comments to abstract per planner call. Small batches avoid omitted JSON slots on busy threads while the shared ledger preserves thread-wide planning.",
    )
    parser.add_argument("--writer-model", default=os.environ.get("WRITER_MODEL", DEFAULT_WRITER_MODEL))
    parser.add_argument("--writer-base-url", default=os.environ.get("WRITER_BASE_URL", DEFAULT_WRITER_BASE_URL))
    parser.add_argument("--writer-api-key", default=os.environ.get("WRITER_API_KEY", "dummy"))
    parser.add_argument(
        "--writer-timeout",
        type=float,
        default=float(os.environ.get("WRITER_TIMEOUT", "600")),
        help="Per-request timeout in seconds for writer API calls. Local Transformers servers may need a larger value.",
    )
    parser.add_argument(
        "--writer-profile",
        choices=sorted(SYSTEM_PROMPTS),
        default="qwen14_labelaware",
        help="Prompt shape. qwen8_v13/qwen14_labelaware are Qwen styles; osim8b uses an OSim social-context writer prompt.",
    )
    parser.add_argument(
        "--writer-max-tokens",
        type=int,
        default=260,
        help="Hard upper cap for writer tokens. Per-length caps still apply below this value.",
    )
    parser.add_argument(
        "--writer-retries",
        type=int,
        default=2,
        help="Retry a writer call when it duplicates an existing comment, copies the parent, or badly misses a short length bucket.",
    )
    parser.add_argument(
        "--post-retry-limit",
        type=int,
        default=0,
        help=(
            "Maximum recoverable attempts for one unfinished post. Zero retries "
            "until success or Ctrl-C; permanent configuration/authentication "
            "errors still fail immediately."
        ),
    )
    parser.add_argument(
        "--post-retry-delay",
        type=float,
        default=15.0,
        help="Base delay in seconds between recoverable post attempts.",
    )
    parser.add_argument(
        "--claim-key-budget",
        type=int,
        default=2,
        help="Maximum accepted comments with the same planner-provided semantic claim key in one generated thread.",
    )
    parser.add_argument(
        "--claim-family-max-share",
        type=float,
        default=0.24,
        help="Maximum accepted share for one broad claim family in a generated thread.",
    )
    parser.add_argument(
        "--claim-family-min-budget",
        type=int,
        default=3,
        help="Minimum accepted comments for any broad claim family before family share capping applies.",
    )
    parser.add_argument(
        "--opening-reuse-budget",
        type=int,
        default=1,
        help="Maximum accepted comments with the same generated opening signature in one generated thread.",
    )
    parser.add_argument(
        "--opener-family-reuse-budget",
        type=int,
        default=5,
        help=(
            "Maximum accepted comments per generated thread for overused opening families "
            "such as first-person experience, conditional advice, uncertainty prefaces, and helpful directives."
        ),
    )
    parser.add_argument(
        "--template-phrase-reuse-budget",
        type=int,
        default=4,
        help=(
            "Maximum accepted comments per generated thread for overused template phrase families "
            "such as first-person experience frames, uncertainty frames, worth-it frames, and generic advice frames."
        ),
    )
    parser.add_argument(
        "--advisor-max-share",
        type=float,
        default=0.45,
        help=(
            "Maximum share of advisor/advice-like tasks before deterministic rebalance. "
            "Keep this loose; Self-BERT is handled by surface restyling, not broad deletion/noise conversion."
        ),
    )
    parser.add_argument(
        "--question-max-share",
        type=float,
        default=0.22,
        help=(
            "Maximum share of question-like tasks before deterministic rebalance converts surplus "
            "questions into direct answers, datapoints, side observations, jokes, or social acknowledgements."
        ),
    )
    parser.add_argument(
        "--micro-target-share",
        type=float,
        default=0.07,
        help="Target share of true micro tasks (roughly 1-5 words) after rebalance.",
    )
    parser.add_argument(
        "--short-max-share",
        type=float,
        default=0.20,
        help="Maximum combined share of micro/short tasks before surplus short tasks are expanded to medium.",
    )
    parser.add_argument(
        "--social-noise-min-share",
        type=float,
        default=0.13,
        help="Minimum share of joke/template/link/quote/messy side-turn tasks after rebalance.",
    )
    parser.add_argument(
        "--gratitude-min-share",
        type=float,
        default=0.10,
        help="Minimum share of gratitude or soft acknowledgement tasks after rebalance.",
    )
    parser.add_argument(
        "--tone-harsh-max-share",
        type=float,
        default=0.14,
        help=(
            "Maximum share of harsh tone shapes after rebalance. Lower values reduce "
            "GPT overproduction of hard advice, hard disagreement, and scolding."
        ),
    )
    parser.add_argument(
        "--tone-calm-min-share",
        type=float,
        default=0.78,
        help="Minimum share of calm tone shapes after rebalance.",
    )
    parser.add_argument(
        "--tone-personal-min-share",
        type=float,
        default=0.16,
        help="Minimum share of compact personal-datapoint tone shapes after rebalance.",
    )
    parser.add_argument(
        "--tone-polite-min-share",
        type=float,
        default=0.24,
        help=(
            "Minimum share of explicit Reddit-polite tone slots: local acknowledgements, "
            "caveated helpful turns, and brief report-backs rather than formal support prose."
        ),
    )
    parser.add_argument(
        "--matched-real-comments",
        type=int,
        default=80,
        help="Max comments from the matched real thread to show the planner.",
    )
    parser.add_argument(
        "--context-dropout-rate",
        type=float,
        default=0.08,
        help=(
            "Probability of hiding or abstracting visible seed/parent context before the writer call. "
            "Keep low for the main generator; strong dropout caused semantic drift in v30."
        ),
    )
    parser.add_argument(
        "--context-jitter-rate",
        type=float,
        default=0.08,
        help=(
            "Probability of showing a lightly distorted seed/parent context instead of exact visible wording. "
            "Keep low for the main generator; surface variation should do most Self-BERT work."
        ),
    )
    parser.add_argument("--max-real-threads-loaded", type=int, default=0, help="0 loads all real threads.")
    parser.add_argument("--force-post", action="store_true", help="Regenerate posts even if the slot exists.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and not args.seed_post_pool_json:
        parser.error("--seed-post-pool-json is required unless --self-test is used")
    if args.post_retry_limit < 0 or args.post_retry_delay < 0:
        parser.error("--post-retry-limit and --post-retry-delay must be non-negative")
    return args

def make_openai_client(*, base_url: str, api_key: str | None, timeout: float | None = None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("The openai package is required for this generator.") from exc
    if not api_key:
        raise SystemExit("Planner/writer API key is missing. Set OPENAI_API_KEY/PLANNER_API_KEY/WRITER_API_KEY.")
    kwargs: dict[str, Any] = {"base_url": base_url, "api_key": api_key}
    if timeout and timeout > 0:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)

def describe_bad_endpoint(*, role: str, base_url: str, content_type: str, body: str) -> str:
    lowered = body.lower()
    if "django tried these url patterns" in lowered or "debug = true" in lowered:
        hint = (
            "This URL is serving your Django app, not a local LLM server. "
            "Point the writer to an OpenAI-compatible Qwen endpoint such as "
            "`http://127.0.0.1:11435/v1` for `scripts/ollama_openai_proxy.py`, "
            "or your MLX/vLLM server port."
        )
    else:
        hint = "This endpoint did not return an OpenAI-compatible response."
    return (
        f"{role} endpoint is misconfigured: {base_url}\n"
        f"Content-Type: {content_type or 'unknown'}\n"
        f"{hint}"
    )

def load_seed_posts(path: Path) -> list[SeedPost]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("seed_posts") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Seed post pool must contain a list: {path}")
    result: list[SeedPost] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        body = str(row.get("body") or "").strip()
        content = str(row.get("content") or "").strip()
        if not content:
            content = f"{title}\n\n{body}".strip()
        if not title and content:
            title = first_line(content)
        try:
            real_num_comments = int(row.get("real_num_comments") or 0)
        except (TypeError, ValueError):
            real_num_comments = 0
        result.append(
            SeedPost(
                index=idx,
                title=title,
                body=body,
                content=content,
                source_raw_post_id=str(row.get("source_raw_post_id") or row.get("post_id") or ""),
                real_num_comments=real_num_comments,
                metadata=row,
            )
        )
    return result
