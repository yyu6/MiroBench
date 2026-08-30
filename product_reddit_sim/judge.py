"""Binary LLM judge for real-vs-generated Reddit discussions.

This module is the core implementation behind the "LLM-as-a-judge" flow used
in this repository. It does four jobs:

1. Normalize different discussion sources into a shared in-memory thread shape.
   The judge can read:
   - generated simulation output (`artifacts/simulations/.../discussion.json`)
   - actual Reddit discussion bundles (`data/raw/discussions/credit_cards/<card>/<card>.json`)
   - manifest files that point to many actual bundles

2. Optionally retrieve supporting context for the judge.
   - product facts are retrieved from product-description JSON files
   - reference threads are retrieved from the training manifest of actual data

3. Build the final judge prompt and call the model.
   The model is asked to output strict JSON with a binary label:
   - `1` = more likely real human Reddit discussion
   - `0` = more likely AI-generated discussion

4. Return structured results that are easy to save, evaluate, and aggregate.

The intended usage pattern is:
`load threads -> retrieve context -> build prompt -> call judge -> aggregate`.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse
import random


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

try:
    from token_usage_tracker import record_openai_usage
except Exception:  # pragma: no cover - tracking must never block judging.
    def record_openai_usage(*args: Any, **kwargs: Any) -> None:
        return


@dataclass
class DiscussionComment:
    """A visible comment snippet attached to a discussion thread.

    The judge does not need the full Reddit object model. It only needs the
    parts that affect realism decisions: author identity, text, score, nesting
    depth, and rough time metadata.
    """

    author: str
    content: str
    comment_id: Optional[str] = None
    score: Optional[int] = None
    depth: int = 0
    timestamp: Optional[str] = None


@dataclass
class DiscussionThread:
    """A normalized thread representation consumed by the judge.

    Both actual Reddit data and generated simulation outputs are converted into
    this shape so the rest of the pipeline can stay source-agnostic.
    """

    source_kind: str
    thread_id: str
    title: str
    body: str
    author: str
    score: Optional[int]
    comment_count: int
    comments: list[DiscussionComment]
    subreddit: Optional[str] = None
    topic: Optional[str] = None
    url: Optional[str] = None


@dataclass
class ProductFact:
    """A compact product record that can be inserted into the judge prompt.

    These facts are intentionally short. The judge should see only the most
    relevant card facts rather than the full raw product catalog.
    """

    product_id: str
    title: str
    brand: Optional[str]
    source_path: str
    summary: str


@dataclass
class JudgeThreadResult:
    """Structured result for one judged thread."""

    thread_id: str
    title: str
    label: int
    confidence: float
    reason: str
    source_kind: str
    comment_count: int
    product_facts: list[dict[str, Any]]
    reference_threads: list[dict[str, Any]]
    few_shot_examples: list[dict[str, Any]]
    raw_response: dict[str, Any]


@dataclass
class FewShotExample:
    """A labeled calibration example inserted into the judge prompt."""

    label: int
    thread: DiscussionThread
    origin: str


@dataclass
class DiscussionCommentTarget:
    """A single comment plus enough local thread context for judging.

    The comment-level judge should not see an isolated sentence with no
    surrounding context. This object keeps the target comment together with the
    parent post and a few nearby visible comments.
    """

    source_kind: str
    thread_id: str
    thread_title: str
    thread_body: str
    thread_author: str
    thread_score: Optional[int]
    thread_comment_count: int
    comment_id: str
    comment_author: str
    comment_content: str
    comment_score: Optional[int] = None
    comment_depth: int = 0
    comment_timestamp: Optional[str] = None
    sibling_comments: list[DiscussionComment] = field(default_factory=list)
    subreddit: Optional[str] = None
    topic: Optional[str] = None
    url: Optional[str] = None


@dataclass
class JudgeCommentResult:
    """Structured result for one judged comment."""

    comment_id: str
    thread_id: str
    thread_title: str
    label: int
    confidence: float
    reason: str
    source_kind: str
    comment_author: str
    comment_depth: int
    product_facts: list[dict[str, Any]]
    reference_comments: list[dict[str, Any]]
    few_shot_examples: list[dict[str, Any]]
    raw_response: dict[str, Any]


@dataclass
class FewShotCommentExample:
    """A labeled calibration example for comment-level judging."""

    label: int
    target: DiscussionCommentTarget
    origin: str


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of discussion realism.

Your task is binary classification:
- Output 1 if the TARGET DISCUSSION is more likely a real human Reddit discussion.
- Output 0 if the TARGET DISCUSSION is more likely AI-generated.

Be skeptical. Fluency alone is not evidence of being real.
"""


COMMENT_JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of Reddit comment realism.

Your task is binary classification:
- Output 1 if the TARGET COMMENT is more likely a real human Reddit comment.
- Output 0 if the TARGET COMMENT is more likely AI-generated.

Judge the comment in context, not in isolation. Be skeptical. Fluency alone is
not evidence of being real.
"""


def sanitize_text(text: Any) -> str:
    """Remove control characters and invalid surrogates from arbitrary text.

    This helper is used aggressively across the judge pipeline because both raw
    product data and raw Reddit dumps can contain malformed characters that
    break prompt construction or JSON serialization.
    """

    sanitized_chars: list[str] = []
    for ch in str(text or ""):
        codepoint = ord(ch)
        if codepoint in (9, 10, 13):
            sanitized_chars.append(ch)
            continue
        if codepoint < 32:
            sanitized_chars.append(" ")
            continue
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        sanitized_chars.append(ch)
    return "".join(sanitized_chars)


def auto_detect_target_kind(path: str | Path) -> str:
    """Infer which loader should be used for a target path.

    Returns one of:
    - `generated`: simulation output from this repo
    - `actual_bundle`: one scraped Reddit bundle for one product/card
    - `actual_manifest`: a manifest pointing to multiple scraped bundles
    """

    path = Path(path)
    if path.is_dir():
        if (path / "discussion.json").exists():
            return "generated"
        raise ValueError(f"Cannot detect target kind from directory: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and {"meta", "posts"}.issubset(data.keys()):
        return "generated"
    if isinstance(data, dict) and {"topic", "posts"}.issubset(data.keys()):
        return "actual_bundle"
    if isinstance(data, list) and data and isinstance(data[0], dict) and "files" in data[0]:
        return "actual_manifest"
    raise ValueError(f"Cannot detect target kind for {path}")


def load_generated_threads(
    path: str | Path,
    max_threads: int = 10,
    max_comments: int = 8,
) -> list[DiscussionThread]:
    """Load simulated discussion threads from a `discussion.json` file.

    The generated discussion format is flatter than real Reddit data, so this
    loader mostly does field renaming plus defensive sanitation.
    """

    path = Path(path)
    if path.is_dir():
        path = path / "discussion.json"

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    posts = data.get("posts") or []
    threads: list[DiscussionThread] = []
    for post in posts[:max_threads]:
        comments = _flatten_generated_comments(
            post.get("comments") or [],
            max_comments=max_comments,
        )
        threads.append(
            DiscussionThread(
                source_kind="generated",
                thread_id=str(post.get("post_id")),
                title=_truncate_text(sanitize_text(post.get("content") or ""), 120),
                body=sanitize_text(post.get("content") or ""),
                author=sanitize_text(post.get("author") or "unknown"),
                score=_safe_int(post.get("likes")),
                comment_count=len(post.get("comments") or []),
                comments=comments,
                subreddit=sanitize_text((data.get("meta") or {}).get("product_category") or "generated_discussion"),
                topic=sanitize_text((data.get("meta") or {}).get("product_category") or ""),
                url=None,
            )
        )
    return threads


def load_actual_threads(
    bundle_json_path: str | Path,
    comments_jsonl_path: str | Path | None = None,
    max_threads: int = 10,
    max_comments: int = 8,
) -> list[DiscussionThread]:
    """Load one real Reddit bundle and its comments into judge threads.

    The bundle JSON stores post-level metadata, while the `.comments.jsonl`
    sidecar stores flattened comments. This function joins them and ranks posts
    so the judge sees the most relevant/high-signal threads first.
    """

    bundle_json_path = Path(bundle_json_path)
    if comments_jsonl_path is None:
        comments_jsonl_path = bundle_json_path.with_name(
            bundle_json_path.stem + ".comments.jsonl"
        )
    comments_jsonl_path = Path(comments_jsonl_path)

    with bundle_json_path.open("r", encoding="utf-8") as handle:
        bundle = json.load(handle)

    comments_by_post = _load_actual_comments(comments_jsonl_path)
    posts = bundle.get("posts") or []
    ranked_posts = sorted(
        posts,
        key=lambda post: (
            not bool(post.get("is_relevant", True)),
            _safe_int(post.get("search_rank"), default=10**9),
            -_safe_int(post.get("num_comments"), default=0),
            -_safe_int(post.get("score"), default=0),
        ),
    )

    threads: list[DiscussionThread] = []
    for post in ranked_posts[:max_threads]:
        post_id = str(post.get("id"))
        raw_comments = comments_by_post.get(post_id, [])
        selected_comments = sorted(
            raw_comments,
            key=lambda comment: (
                _safe_int(comment.get("depth"), default=0),
                -_safe_int(comment.get("score"), default=0),
                comment.get("created_iso") or "",
            ),
        )[:max_comments]
        comments = [
            DiscussionComment(
                author=sanitize_text(comment.get("author") or "unknown"),
                content=sanitize_text(comment.get("body") or ""),
                comment_id=sanitize_text(comment.get("comment_id") or "") or None,
                score=_safe_int(comment.get("score")),
                depth=_safe_int(comment.get("depth"), default=0),
                timestamp=sanitize_text(comment.get("created_iso") or ""),
            )
            for comment in selected_comments
        ]

        title = sanitize_text(post.get("title") or post.get("post_title") or "")
        body = sanitize_text(post.get("selftext") or "")
        if not body:
            body = title

        threads.append(
            DiscussionThread(
                source_kind="actual",
                thread_id=post_id,
                title=title or _truncate_text(body, 120),
                body=body,
                author=sanitize_text(post.get("author") or "unknown"),
                score=_safe_int(post.get("score")),
                comment_count=len(raw_comments) if raw_comments else _safe_int(post.get("num_comments"), default=0),
                comments=comments,
                subreddit=sanitize_text(post.get("subreddit") or ""),
                topic=sanitize_text(bundle.get("topic") or ""),
                url=sanitize_text(post.get("permalink") or post.get("url") or ""),
            )
        )
    return threads


def load_threads_from_manifest(
    manifest_path: str | Path,
    max_items: int = 10,
    threads_per_item: int = 2,
    max_comments: int = 8,
) -> list[DiscussionThread]:
    """Load representative actual threads from a manifest of many cards.

    This is mainly used when:
    - building a batch evaluation set
    - retrieving a few actual reference threads for the judge
    """

    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    threads: list[DiscussionThread] = []
    for item in manifest[:max_items]:
        files = item.get("files") or {}
        if item.get("discussion_bundle_status") != "json_bundle":
            continue
        bundle_path = _resolve_repo_path(files.get("json"))
        comments_path = _resolve_repo_path(files.get("comments_jsonl"))
        if not bundle_path or not comments_path:
            continue
        threads.extend(
            load_actual_threads(
                bundle_json_path=bundle_path,
                comments_jsonl_path=comments_path,
                max_threads=threads_per_item,
                max_comments=max_comments,
            )
        )
    return threads


def load_product_facts(
    product_file: str | Path,
    max_products: int = 100,
    max_description_chars: int = 900,
) -> list[ProductFact]:
    """Convert product-description JSON into compact retrievable fact records.

    The loader accepts multiple schemas because this repo contains product data
    from several pipelines. The downstream retriever only relies on title,
    optional brand, and a truncated description summary.
    """

    with Path(product_file).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    raw_list = _detect_product_schema(data)
    facts: list[ProductFact] = []
    for idx, raw in enumerate(raw_list[:max_products]):
        title = _resolve_product_title(raw)
        brand = _resolve_product_brand(raw)
        description = sanitize_text(
            raw.get("page_description")
            or raw.get("full_description")
            or raw.get("description")
            or ""
        )
        summary = _truncate_text(description, max_description_chars)
        facts.append(
            ProductFact(
                product_id=str(raw.get("row_index", idx)),
                title=title,
                brand=brand,
                source_path=str(product_file),
                summary=summary,
            )
        )
    return facts


def retrieve_product_facts(
    thread: DiscussionThread,
    product_facts: list[ProductFact],
    top_k: int = 4,
) -> list[ProductFact]:
    """Retrieve the most relevant product facts for one target thread.

    This is a lightweight lexical retriever, not a full embedding system. It
    intentionally favors:
    - exact card-name mentions
    - issuer/brand mentions
    - word overlap between the thread and product summaries
    """

    query_text = render_thread_for_retrieval(thread)
    query_tokens = _tokenize(query_text)

    scored: list[tuple[float, ProductFact]] = []
    normalized_query = query_text.lower()
    for fact in product_facts:
        title_tokens = _tokenize(fact.title)
        brand_tokens = _tokenize(fact.brand or "")
        summary_tokens = _tokenize(fact.summary)

        overlap = len(query_tokens & (title_tokens | brand_tokens | summary_tokens))
        exact_title_bonus = 10 if fact.title.lower() in normalized_query else 0
        brand_bonus = 3 if fact.brand and fact.brand.lower() in normalized_query else 0
        score = exact_title_bonus + brand_bonus + overlap
        if score > 0:
            scored.append((float(score), fact))

    scored.sort(key=lambda item: (-item[0], item[1].title))
    return [fact for _, fact in scored[:top_k]]


def load_reference_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Load the aligned discussion manifest used for reference retrieval.

    Relative file paths are preserved in the JSON on disk, but resolved to
    absolute paths in memory so callers do not depend on the current working
    directory.
    """

    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    for item in records:
        files = item.get("files") or {}
        item["_manifest_path"] = str(manifest_path)
        for key, value in list(files.items()):
            files[key] = str(_resolve_repo_path(value)) if value else None
    return records


def retrieve_reference_threads(
    thread: DiscussionThread,
    manifest_records: list[dict[str, Any]],
    max_cards: int = 2,
    max_comments: int = 4,
) -> list[DiscussionThread]:
    """Retrieve a small number of actual reference threads for calibration.

    These are not used as labels or exact matches. They act as a distributional
    anchor so the judge sees what real credit-card Reddit threads tend to look
    like in the training split.
    """

    query_tokens = _tokenize(render_thread_for_retrieval(thread))
    scored_records: list[tuple[float, dict[str, Any]]] = []

    for item in manifest_records:
        if item.get("discussion_bundle_status") != "json_bundle":
            continue
        text = " ".join(
            str(part)
            for part in (
                item.get("card_name"),
                item.get("topic"),
                item.get("folder_name"),
            )
            if part
        )
        overlap = len(query_tokens & _tokenize(text))
        if overlap > 0:
            scored_records.append((float(overlap), item))

    scored_records.sort(key=lambda item: (-item[0], item[1].get("folder_name", "")))
    reference_threads: list[DiscussionThread] = []
    for _, item in scored_records[:max_cards]:
        files = item.get("files") or {}
        bundle_path = files.get("json")
        comments_path = files.get("comments_jsonl")
        if not bundle_path or not comments_path:
            continue
        reference_threads.extend(
            load_actual_threads(
                bundle_json_path=bundle_path,
                comments_jsonl_path=comments_path,
                max_threads=1,
                max_comments=max_comments,
            )
        )
    return reference_threads


def sample_few_shot_examples(
    real_threads: list[DiscussionThread] | None = None,
    generated_threads: list[DiscussionThread] | None = None,
    per_label: int = 2,
    seed: int = 42,
    exclude_threads: list[DiscussionThread] | None = None,
) -> list[FewShotExample]:
    """Randomly sample labeled few-shot examples from real and generated pools.

    The sampling is intentionally simple and deterministic under `seed`. It is
    meant to provide a compact decision-boundary reminder to the judge, not a
    large retrieval context.
    """

    rng = random.Random(seed)
    excluded = {
        _thread_identity(thread)
        for thread in (exclude_threads or [])
    }

    sampled: list[FewShotExample] = []
    for label, origin, pool in (
        (1, "real", list(real_threads or [])),
        (0, "generated", list(generated_threads or [])),
    ):
        candidates = [
            thread
            for thread in pool
            if _thread_identity(thread) not in excluded
        ]
        rng.shuffle(candidates)
        for thread in candidates[: max(0, per_label)]:
            sampled.append(FewShotExample(label=label, thread=thread, origin=origin))

    rng.shuffle(sampled)
    return sampled


def extract_comment_targets(
    threads: list[DiscussionThread],
    max_items: int = 20,
    sibling_comments: int = 3,
) -> list[DiscussionCommentTarget]:
    """Explode thread-level data into comment-level judging targets.

    Each extracted target keeps the parent post plus a few non-target comments
    so the comment judge can assess whether the target fits the local thread.
    """

    targets: list[DiscussionCommentTarget] = []
    for thread in threads:
        for idx, comment in enumerate(thread.comments, start=1):
            siblings = [
                sibling
                for sibling in thread.comments
                if (sibling.comment_id or "") != (comment.comment_id or "")
                or sibling.content != comment.content
            ][: max(0, sibling_comments)]
            targets.append(
                DiscussionCommentTarget(
                    source_kind=thread.source_kind,
                    thread_id=thread.thread_id,
                    thread_title=thread.title,
                    thread_body=thread.body,
                    thread_author=thread.author,
                    thread_score=thread.score,
                    thread_comment_count=thread.comment_count,
                    comment_id=comment.comment_id or f"{thread.thread_id}_comment_{idx}",
                    comment_author=comment.author,
                    comment_content=comment.content,
                    comment_score=comment.score,
                    comment_depth=comment.depth,
                    comment_timestamp=comment.timestamp,
                    sibling_comments=siblings,
                    subreddit=thread.subreddit,
                    topic=thread.topic,
                    url=thread.url,
                )
            )
            if len(targets) >= max_items:
                return targets
    return targets


def sample_few_shot_comment_examples(
    real_comments: list[DiscussionCommentTarget] | None = None,
    generated_comments: list[DiscussionCommentTarget] | None = None,
    per_label: int = 2,
    seed: int = 42,
    exclude_comments: list[DiscussionCommentTarget] | None = None,
) -> list[FewShotCommentExample]:
    """Randomly sample labeled few-shot examples for comment-level judging."""

    rng = random.Random(seed)
    excluded = {
        _comment_identity(comment)
        for comment in (exclude_comments or [])
    }

    sampled: list[FewShotCommentExample] = []
    for label, origin, pool in (
        (1, "real", list(real_comments or [])),
        (0, "generated", list(generated_comments or [])),
    ):
        candidates = [
            comment
            for comment in pool
            if _comment_identity(comment) not in excluded
        ]
        rng.shuffle(candidates)
        for comment in candidates[: max(0, per_label)]:
            sampled.append(FewShotCommentExample(label=label, target=comment, origin=origin))

    rng.shuffle(sampled)
    return sampled


def render_thread_for_prompt(
    thread: DiscussionThread,
    max_comments: int = 8,
    include_source_kind: bool = False,
) -> str:
    """Render a normalized thread into a compact prompt block.

    `include_source_kind` is off by default to avoid leaking target labels into
    the judge prompt.
    """

    lines = [
        f"THREAD_ID: {thread.thread_id}",
        f"TOPIC: {thread.topic or ''}",
        f"SUBREDDIT: {thread.subreddit or ''}",
        f"TITLE: {thread.title}",
        f"AUTHOR: {thread.author}",
        f"SCORE: {thread.score if thread.score is not None else 'unknown'}",
        f"COMMENT_COUNT: {thread.comment_count}",
        "POST_BODY:",
        sanitize_text(thread.body),
        "",
        "COMMENTS:",
    ]
    if include_source_kind:
        lines.insert(1, f"SOURCE_KIND: {thread.source_kind}")
    if not thread.comments:
        lines.append("- [no visible comments]")
    else:
        for idx, comment in enumerate(thread.comments[:max_comments], start=1):
            lines.append(
                f"- comment_{idx} | author={comment.author} | depth={comment.depth} | "
                f"score={comment.score if comment.score is not None else 'unknown'}"
            )
            lines.append(f"  {sanitize_text(comment.content)}")
    return "\n".join(lines)


def render_thread_for_retrieval(thread: DiscussionThread) -> str:
    """Render a thread into a short text view optimized for lexical retrieval."""

    comment_text = " ".join(comment.content for comment in thread.comments)
    return sanitize_text(
        f"{thread.title}\n{thread.body}\n{thread.topic or ''}\n{comment_text}"
    )


def render_comment_target_for_prompt(
    target: DiscussionCommentTarget,
    max_siblings: int = 3,
    include_source_kind: bool = False,
) -> str:
    """Render one target comment plus local thread context for the prompt."""

    lines = [
        f"THREAD_ID: {target.thread_id}",
        f"TOPIC: {target.topic or ''}",
        f"SUBREDDIT: {target.subreddit or ''}",
        f"THREAD_TITLE: {target.thread_title}",
        f"THREAD_AUTHOR: {target.thread_author}",
        f"THREAD_SCORE: {target.thread_score if target.thread_score is not None else 'unknown'}",
        f"THREAD_COMMENT_COUNT: {target.thread_comment_count}",
        "POST_BODY:",
        sanitize_text(target.thread_body),
        "",
        "TARGET_COMMENT:",
        (
            f"comment_id={target.comment_id} | author={target.comment_author} | "
            f"depth={target.comment_depth} | score={target.comment_score if target.comment_score is not None else 'unknown'}"
        ),
        sanitize_text(target.comment_content),
        "",
        "OTHER_VISIBLE_COMMENTS:",
    ]
    if include_source_kind:
        lines.insert(1, f"SOURCE_KIND: {target.source_kind}")
    if not target.sibling_comments:
        lines.append("- [no other visible comments]")
    else:
        for idx, comment in enumerate(target.sibling_comments[:max_siblings], start=1):
            lines.append(
                f"- sibling_{idx} | author={comment.author} | depth={comment.depth} | "
                f"score={comment.score if comment.score is not None else 'unknown'}"
            )
            lines.append(f"  {sanitize_text(comment.content)}")
    return "\n".join(lines)


def render_comment_target_for_retrieval(target: DiscussionCommentTarget) -> str:
    """Render a comment target into short retrieval text."""

    sibling_text = " ".join(comment.content for comment in target.sibling_comments)
    return sanitize_text(
        "\n".join(
            [
                target.thread_title,
                target.thread_body,
                target.comment_content,
                target.topic or "",
                sibling_text,
            ]
        )
    )


def build_binary_judge_prompt(
    thread: DiscussionThread,
    product_facts: list[ProductFact] | None = None,
    reference_threads: list[DiscussionThread] | None = None,
    few_shot_examples: list[FewShotExample] | None = None,
) -> str:
    """Build the full user prompt sent to the judge model.

    The prompt keeps three sources of information separate:
    - retrieved product facts
    - optional actual reference threads
    - the target discussion to classify

    Keeping these blocks explicit makes the prompt easier to audit and reduces
    the risk of leaking accidental label information.
    """

    product_facts = list(product_facts or [])
    reference_threads = list(reference_threads or [])
    few_shot_examples = list(few_shot_examples or [])

    fact_block = ["PRODUCT_FACTS:"]
    if not product_facts:
        fact_block.append("- [no external product facts provided]")
    else:
        for fact in product_facts:
            fact_block.append(
                f"- {fact.title} | brand={fact.brand or 'unknown'} | {fact.summary}"
            )

    reference_block = ["ACTUAL_REFERENCE_THREADS:"]
    if not reference_threads:
        reference_block.append("- [no reference threads provided]")
    else:
        for idx, ref_thread in enumerate(reference_threads, start=1):
            reference_block.append(f"[REFERENCE_{idx}]")
            reference_block.append(render_thread_for_prompt(ref_thread, max_comments=4))

    few_shot_block = ["LABELED_FEW_SHOT_EXAMPLES:"]
    if not few_shot_examples:
        few_shot_block.append("- [no few-shot examples provided]")
    else:
        few_shot_block.extend(
            [
                "Use these as rough calibration examples for the decision boundary.",
                "Do not do exact similarity matching. Judge the target on its own merits.",
            ]
        )
        for idx, example in enumerate(few_shot_examples, start=1):
            label_text = "real" if example.label == 1 else "ai-generated"
            few_shot_block.append(
                f"[FEW_SHOT_{idx}] KNOWN_LABEL={example.label} ({label_text}) | origin={example.origin}"
            )
            few_shot_block.append(
                render_thread_for_prompt(example.thread, max_comments=4)
            )

    return sanitize_text(
        "\n".join(
            [
                "Classify the TARGET DISCUSSION.",
                "Output 1 if it is more likely real human Reddit discussion.",
                "Output 0 if it is more likely ai-generated discussion.",
                "Be skeptical. Fluency alone is not evidence of being real.",
                "",
                "Judge the discussion on:",
                "1. Thread realism: are comments reacting to the post and to each other?",
                "2. Human variation: do commenters sound like different people with uneven knowledge and different motivations?",
                "3. Epistemic realism: do people show uncertainty, bias, partial knowledge, correction, or overconfidence in believable ways?",
                "4. Product grounding: are claims tied to plausible credit-card facts, use cases, or lived experience?",
                "5. Reddit messiness: are there short replies, disagreement, nitpicking, sarcasm, narrow takes, or low-information comments?",
                "6. Synthetic artifacts: do many comments sound uniformly polished, repetitive, overly balanced, or like mini reviews?",
                "",
                *fact_block,
                "",
                *reference_block,
                "",
                *few_shot_block,
                "",
                "TARGET_DISCUSSION:",
                render_thread_for_prompt(thread),
                "",
                "Return strict JSON:",
                "{",
                '  "label": 0 | 1,',
                '  "confidence": 0.0-1.0,',
                '  "reason": "short paragraph explaining the decisive signals that makes you think the discussion is real or generated"',
                "}",
                "",
                "Decision rule:",
                "- Use 1 only if the discussion is more likely real than generated.",
                "- Use 0 if there are stronger synthetic signals than human signals.",
            ]
        )
    )


def build_binary_comment_judge_prompt(
    target: DiscussionCommentTarget,
    product_facts: list[ProductFact] | None = None,
    reference_comments: list[DiscussionCommentTarget] | None = None,
    few_shot_examples: list[FewShotCommentExample] | None = None,
) -> str:
    """Build the full user prompt for comment-level realism judging."""

    product_facts = list(product_facts or [])
    reference_comments = list(reference_comments or [])
    few_shot_examples = list(few_shot_examples or [])

    fact_block = ["PRODUCT_FACTS:"]
    if not product_facts:
        fact_block.append("- [no external product facts provided]")
    else:
        for fact in product_facts:
            fact_block.append(
                f"- {fact.title} | brand={fact.brand or 'unknown'} | {fact.summary}"
            )

    reference_block = ["ACTUAL_REFERENCE_COMMENTS:"]
    if not reference_comments:
        reference_block.append("- [no reference comments provided]")
    else:
        for idx, reference in enumerate(reference_comments, start=1):
            reference_block.append(f"[REFERENCE_COMMENT_{idx}]")
            reference_block.append(
                render_comment_target_for_prompt(reference, max_siblings=3)
            )

    few_shot_block = ["LABELED_FEW_SHOT_EXAMPLES:"]
    if not few_shot_examples:
        few_shot_block.append("- [no few-shot examples provided]")
    else:
        few_shot_block.extend(
            [
                "Use these as rough calibration examples for the decision boundary.",
                "Do not do exact similarity matching. Judge the target on its own merits.",
            ]
        )
        for idx, example in enumerate(few_shot_examples, start=1):
            label_text = "real" if example.label == 1 else "ai-generated"
            few_shot_block.append(
                f"[FEW_SHOT_{idx}] KNOWN_LABEL={example.label} ({label_text}) | origin={example.origin}"
            )
            few_shot_block.append(
                render_comment_target_for_prompt(example.target, max_siblings=3)
            )

    return sanitize_text(
        "\n".join(
            [
                "Classify the TARGET COMMENT.",
                "Output 1 if it is more likely a real human Reddit comment.",
                "Output 0 if it is more likely an ai-generated comment.",
                "Judge the comment in context, not in isolation.",
                "Be skeptical. Fluency alone is not evidence of being real.",
                "",
                "Judge the comment on:",
                "1. Local reaction realism: does it plausibly respond to the post or visible thread context?",
                "2. Human voice variation: does it sound like a specific person rather than a generic assistant voice?",
                "3. Epistemic realism: does it show believable uncertainty, bias, partial knowledge, annoyance, or overconfidence?",
                "4. Product grounding: are claims tied to plausible facts, use cases, lived experience, or normal Reddit folklore?",
                "5. Reddit comment texture: could a real user actually write it at this depth, in this thread, at this level of effort?",
                "6. Synthetic artifacts: is it overly polished, overly complete, too balanced, too helpful, or too much like a mini review?",
                "7. Context fit: does the wording fit the surrounding thread, or does it feel dropped in from outside?",
                "",
                *fact_block,
                "",
                *reference_block,
                "",
                *few_shot_block,
                "",
                "TARGET_COMMENT_CONTEXT:",
                render_comment_target_for_prompt(target, max_siblings=3),
                "",
                "Return strict JSON:",
                "{",
                '  "label": 0 | 1,',
                '  "confidence": 0.0-1.0,',
                '  "reason": "short paragraph explaining the decisive signals"',
                "}",
                "",
                "Decision rule:",
                "- Use 1 only if the comment is more likely real than generated.",
                "- Use 0 if there are stronger synthetic signals than human signals.",
            ]
        )
    )


def call_binary_judge(
    client: Any,
    model: str,
    thread: DiscussionThread,
    product_facts: list[ProductFact] | None = None,
    reference_threads: list[DiscussionThread] | None = None,
    few_shot_examples: list[FewShotExample] | None = None,
) -> JudgeThreadResult:
    """Call the chat model once and parse a binary realism judgment.

    The model is forced into JSON mode because downstream evaluation depends on
    reliable machine-readable output.
    """

    prompt = build_binary_judge_prompt(
        thread=thread,
        product_facts=product_facts,
        reference_threads=reference_threads,
        few_shot_examples=few_shot_examples,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    record_openai_usage(response, model=model, component="product_reddit_sim_thread_judge")
    raw_content = response.choices[0].message.content
    parsed = _parse_judge_response(raw_content)
    return JudgeThreadResult(
        thread_id=thread.thread_id,
        title=thread.title,
        label=parsed["label"],
        confidence=parsed["confidence"],
        reason=parsed["reason"],
        source_kind=thread.source_kind,
        comment_count=thread.comment_count,
        product_facts=[asdict(fact) for fact in (product_facts or [])],
        reference_threads=[
            {
                "thread_id": ref.thread_id,
                "title": ref.title,
                "source_kind": ref.source_kind,
            }
            for ref in (reference_threads or [])
        ],
        few_shot_examples=[
            {
                "label": example.label,
                "origin": example.origin,
                "thread_id": example.thread.thread_id,
                "title": example.thread.title,
                "source_kind": example.thread.source_kind,
            }
            for example in (few_shot_examples or [])
        ],
        raw_response=parsed,
    )


def call_binary_comment_judge(
    client: Any,
    model: str,
    target: DiscussionCommentTarget,
    product_facts: list[ProductFact] | None = None,
    reference_comments: list[DiscussionCommentTarget] | None = None,
    few_shot_examples: list[FewShotCommentExample] | None = None,
) -> JudgeCommentResult:
    """Call the chat model once and parse a binary realism judgment for a comment."""

    prompt = build_binary_comment_judge_prompt(
        target=target,
        product_facts=product_facts,
        reference_comments=reference_comments,
        few_shot_examples=few_shot_examples,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COMMENT_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    record_openai_usage(response, model=model, component="product_reddit_sim_comment_judge")
    raw_content = response.choices[0].message.content
    parsed = _parse_judge_response(raw_content)
    return JudgeCommentResult(
        comment_id=target.comment_id,
        thread_id=target.thread_id,
        thread_title=target.thread_title,
        label=parsed["label"],
        confidence=parsed["confidence"],
        reason=parsed["reason"],
        source_kind=target.source_kind,
        comment_author=target.comment_author,
        comment_depth=target.comment_depth,
        product_facts=[asdict(fact) for fact in (product_facts or [])],
        reference_comments=[
            {
                "comment_id": reference.comment_id,
                "thread_id": reference.thread_id,
                "thread_title": reference.thread_title,
                "source_kind": reference.source_kind,
            }
            for reference in (reference_comments or [])
        ],
        few_shot_examples=[
            {
                "label": example.label,
                "origin": example.origin,
                "comment_id": example.target.comment_id,
                "thread_id": example.target.thread_id,
                "source_kind": example.target.source_kind,
            }
            for example in (few_shot_examples or [])
        ],
        raw_response=parsed,
    )


def retrieve_product_facts_for_comment(
    target: DiscussionCommentTarget,
    product_facts: list[ProductFact],
    top_k: int = 4,
) -> list[ProductFact]:
    """Retrieve product facts relevant to a single target comment."""

    query_text = render_comment_target_for_retrieval(target)
    query_tokens = _tokenize(query_text)

    scored: list[tuple[float, ProductFact]] = []
    normalized_query = query_text.lower()
    for fact in product_facts:
        title_tokens = _tokenize(fact.title)
        brand_tokens = _tokenize(fact.brand or "")
        summary_tokens = _tokenize(fact.summary)

        overlap = len(query_tokens & (title_tokens | brand_tokens | summary_tokens))
        exact_title_bonus = 10 if fact.title.lower() in normalized_query else 0
        brand_bonus = 3 if fact.brand and fact.brand.lower() in normalized_query else 0
        score = exact_title_bonus + brand_bonus + overlap
        if score > 0:
            scored.append((float(score), fact))

    scored.sort(key=lambda item: (-item[0], item[1].title))
    return [fact for _, fact in scored[:top_k]]


def retrieve_reference_comment_targets(
    target: DiscussionCommentTarget,
    manifest_records: list[dict[str, Any]],
    max_cards: int = 2,
    max_comments: int = 4,
    max_targets: int = 2,
) -> list[DiscussionCommentTarget]:
    """Retrieve a small number of actual reference comments for calibration."""

    pseudo_thread = DiscussionThread(
        source_kind=target.source_kind,
        thread_id=target.thread_id,
        title=target.thread_title,
        body=target.thread_body,
        author=target.thread_author,
        score=target.thread_score,
        comment_count=target.thread_comment_count,
        comments=[
            DiscussionComment(
                author=target.comment_author,
                content=target.comment_content,
                comment_id=target.comment_id,
                score=target.comment_score,
                depth=target.comment_depth,
                timestamp=target.comment_timestamp,
            ),
            *list(target.sibling_comments),
        ],
        subreddit=target.subreddit,
        topic=target.topic,
        url=target.url,
    )
    reference_threads = retrieve_reference_threads(
        pseudo_thread,
        manifest_records,
        max_cards=max_cards,
        max_comments=max_comments,
    )
    return extract_comment_targets(
        reference_threads,
        max_items=max_targets,
        sibling_comments=min(3, max_comments),
    )


def aggregate_judge_results(results: Iterable[JudgeThreadResult]) -> dict[str, Any]:
    """Aggregate multiple thread-level verdicts into one overall decision.

    The aggregation rule converts each thread result into a "probability of
    being real" and then averages across threads. This keeps the overall label
    stable even when a batch contains mixed evidence.
    """

    results = list(results)
    if not results:
        return {
            "label": 0,
            "confidence": 0.0,
            "mean_real_probability": 0.0,
            "thread_count": 0,
        }

    real_probabilities = [
        result.confidence if result.label == 1 else max(0.0, 1.0 - result.confidence)
        for result in results
    ]
    mean_real_probability = sum(real_probabilities) / len(real_probabilities)
    overall_label = 1 if mean_real_probability >= 0.5 else 0
    overall_confidence = abs(mean_real_probability - 0.5) * 2.0
    return {
        "label": overall_label,
        "confidence": round(overall_confidence, 4),
        "mean_real_probability": round(mean_real_probability, 4),
        "thread_count": len(results),
    }


def _load_actual_comments(comments_jsonl_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load flattened Reddit comments and group them by post id."""

    comments_by_post: dict[str, list[dict[str, Any]]] = {}
    if not comments_jsonl_path.exists():
        return comments_by_post

    with comments_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            post_id = str(record.get("post_id"))
            comments_by_post.setdefault(post_id, []).append(record)
    return comments_by_post


def _flatten_generated_comments(
    comments: list[dict[str, Any]],
    max_comments: int,
    depth: int = 0,
) -> list[DiscussionComment]:
    """Flatten nested generated comments into judge-friendly comment objects."""

    flattened: list[DiscussionComment] = []
    for comment in comments:
        flattened.append(
            DiscussionComment(
                author=sanitize_text(comment.get("author") or "unknown"),
                content=sanitize_text(comment.get("content") or ""),
                comment_id=sanitize_text(comment.get("comment_id") or "") or None,
                score=_safe_int(comment.get("likes")),
                depth=_safe_int(comment.get("depth"), default=depth) or depth,
                timestamp=sanitize_text(comment.get("timestamp") or ""),
            )
        )
        if len(flattened) >= max_comments:
            return flattened[:max_comments]
        flattened.extend(
            _flatten_generated_comments(
                comment.get("replies") or [],
                max_comments=max_comments - len(flattened),
                depth=depth + 1,
            )
        )
        if len(flattened) >= max_comments:
            return flattened[:max_comments]
    return flattened[:max_comments]


def _detect_product_schema(data: Any) -> list[dict[str, Any]]:
    """Detect the list of raw product records inside a JSON payload."""

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "products" in data and isinstance(data["products"], list):
            return data["products"]
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
    raise ValueError(f"Cannot detect product schema for judge input: {type(data).__name__}")


def _resolve_repo_path(path: str | Path | None) -> Path | None:
    """Resolve a manifest/file path relative to the repo root when needed."""

    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _thread_identity(thread: DiscussionThread) -> tuple[str, str, str]:
    """Build a stable identity tuple used for few-shot exclusion."""

    return (
        str(thread.source_kind),
        str(thread.thread_id),
        _truncate_text(f"{thread.title}\n{thread.body}", 160),
    )


def _comment_identity(target: DiscussionCommentTarget) -> tuple[str, str, str, str]:
    """Build a stable identity tuple used for comment few-shot exclusion."""

    return (
        str(target.source_kind),
        str(target.thread_id),
        str(target.comment_id),
        _truncate_text(target.comment_content, 160),
    )


def _resolve_product_title(raw: dict[str, Any]) -> str:
    """Resolve a product title across multiple product-data schemas."""

    return str(
        raw.get("title")
        or raw.get("card_name")
        or raw.get("product_name")
        or raw.get("name")
        or "Unknown Product"
    )


def _resolve_product_brand(raw: dict[str, Any]) -> Optional[str]:
    """Infer a product brand from explicit metadata or the source URL host."""

    if raw.get("brand"):
        return str(raw["brand"])

    url = raw.get("final_url") or raw.get("official_product_url")
    if not url:
        return None

    host = urlparse(str(url)).netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    brand_map = {
        "americanexpress.com": "American Express",
        "creditcards.chase.com": "Chase",
        "chase.com": "Chase",
        "capitalone.com": "Capital One",
        "bankofamerica.com": "Bank of America",
        "citi.com": "Citi",
        "citicards.citi.com": "Citi",
        "discover.com": "Discover",
        "wellsfargo.com": "Wells Fargo",
        "creditcards.wellsfargo.com": "Wells Fargo",
        "usbank.com": "U.S. Bank",
        "barclaycardus.com": "Barclays",
        "dcu.org": "DCU",
        "chime.com": "Chime",
        "synchrony.com": "Synchrony",
    }
    return brand_map.get(host)


def _parse_judge_response(raw_content: str) -> dict[str, Any]:
    """Parse and validate the model response from the binary judge call.

    The parser is deliberately defensive:
    - it accepts strict JSON
    - it falls back to extracting the first JSON object from noisy text
    - it normalizes `label` and `confidence`
    """

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_content, flags=re.S)
        if not match:
            raise ValueError(f"Judge did not return valid JSON: {raw_content[:300]}")
        parsed = json.loads(match.group(0))

    label = parsed.get("label")
    if isinstance(label, str) and label.isdigit():
        label = int(label)
    if label not in (0, 1):
        raise ValueError(f"Judge label must be 0 or 1, got {label!r}")

    confidence = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = sanitize_text(parsed.get("reason") or parsed.get("overall_verdict") or "")
    return {
        "label": int(label),
        "confidence": confidence,
        "reason": reason,
        **parsed,
    }


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Best-effort integer conversion used for heterogeneous raw inputs."""

    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _truncate_text(text: str, limit: int) -> str:
    """Normalize whitespace and trim a string to a bounded prompt length."""

    text = sanitize_text(text)
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a small lowercase word set for lexical retrieval."""

    return {
        token
        for token in re.findall(r"[a-z0-9]+", sanitize_text(text).lower())
        if len(token) >= 3
    }
