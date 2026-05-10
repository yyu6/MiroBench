#!/usr/bin/env python3
"""Score thread-level semantic uniformity with comment embeddings.

This evaluator measures whether comments inside the same thread are too
semantically similar. It embeds each individual comment with a sentence
embedding model, computes all pairwise cosine similarities within each thread,
and reports:

- mean cosine similarity
- median cosine similarity
- top-k mean cosine similarity

Higher values mean the thread is more semantically uniform. For generated
discussions, this is a useful complement to the Stance_Rel disagreement scorer:
stance can detect direct agree/disagree relationships, while embedding
uniformity catches repeated paraphrases and low perspective diversity.

The default model is `sentence-transformers/all-mpnet-base-v2`. The script uses
`sentence-transformers` when installed; otherwise it falls back to local
Transformers mean pooling, matching the model card's standard pooling recipe.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .score_thread_disagreement import clean_text, detect_target_kind, find_single, read_jsonl


DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


@dataclass
class ThreadComment:
    """One comment to embed for semantic-uniformity scoring."""

    thread_id: str
    thread_title: str
    comment_id: str
    parent_id: str
    author: str
    text: str
    depth: int


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Score within-thread comment semantic uniformity with all-mpnet-base-v2."
    )
    parser.add_argument("input", help="Real discussion folder/file or generated run folder/file.")
    parser.add_argument(
        "--target-kind",
        choices=["auto", "real", "generated"],
        default="auto",
        help="Input schema. auto detects generated discussion.json vs real Reddit bundle.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Hugging Face sentence embedding model name or local path.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Where to write JSON results. Defaults next to input.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-threads", type=int, default=0)
    parser.add_argument("--max-comments-per-thread", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Embedding device. auto uses cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Do not write per-comment embedding vectors to JSON.",
    )
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

    comments = [comment for thread_comments in comments_by_thread.values() for comment in thread_comments]
    embedder = CommentEmbedder(
        model_name=args.model_name,
        device=args.device,
        max_length=max(8, args.max_length),
    )
    embeddings = embedder.encode(
        [comment.text for comment in comments],
        batch_size=max(1, args.batch_size),
    )
    comment_embeddings = {
        comment.comment_id: embeddings[idx]
        for idx, comment in enumerate(comments)
    }

    thread_results = []
    for thread_id, thread_comments in comments_by_thread.items():
        thread_embeddings = np.array(
            [comment_embeddings[comment.comment_id] for comment in thread_comments],
            dtype=np.float32,
        )
        thread_results.append(
            score_thread(
                thread_id=thread_id,
                comments=thread_comments,
                embeddings=thread_embeddings,
                top_k=max(1, args.top_k),
                include_embeddings=not args.no_embeddings,
            )
        )

    thread_results.sort(
        key=lambda row: (
            row["mean_cosine_similarity"],
            row["comment_count"],
        ),
        reverse=True,
    )
    result = {
        "meta": {
            "input": str(input_path),
            "target_kind": target_kind,
            "model_name": args.model_name,
            "embedding_backend": embedder.backend_name,
            "device": embedder.device,
            "max_length": args.max_length,
            "top_k": args.top_k,
            "comment_count": len(comments),
            "thread_count": len(thread_results),
            "embedding_dim": int(embeddings.shape[1]) if len(comments) else 0,
            "include_embeddings": not args.no_embeddings,
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
        f"comments={overall['comment_count']} threads={overall['thread_count']} "
        f"mean_cosine={overall['weighted_mean_cosine_similarity']:.4f} "
        f"median_thread_cosine={overall['median_thread_cosine_similarity']:.4f} "
        f"top_k_mean={overall['weighted_top_k_mean_cosine_similarity']:.4f}"
    )


class CommentEmbedder:
    """Sentence embedding wrapper for all-mpnet-base-v2 style models."""

    def __init__(self, model_name: str, device: str, max_length: int) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.backend_name = "transformers_mean_pooling"
        self.device = "cpu"

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._sentence_model = None
        else:
            self._sentence_model = SentenceTransformer(model_name, device=self._resolve_sentence_device(device))
            self.backend_name = "sentence_transformers"
            self.device = str(self._sentence_model.device)
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit(
                "This scorer requires either sentence-transformers or torch+transformers."
            ) from exc

        self.torch = torch
        self.device = self._resolve_torch_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode comment texts into normalized embeddings."""

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self._sentence_model is not None:
            return np.asarray(
                self._sentence_model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )

        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                output = self.model(**encoded)
                pooled = mean_pooling(output.last_hidden_state, encoded["attention_mask"])
                normalized = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(normalized.detach().cpu().numpy().astype(np.float32))
        return np.vstack(embeddings)

    def _resolve_sentence_device(self, requested: str) -> str | None:
        """Resolve device for sentence-transformers."""

        if requested == "auto":
            return None
        return requested

    def _resolve_torch_device(self, requested: str) -> str:
        """Resolve device for plain Transformers."""

        if requested != "auto":
            return requested
        if self.torch.cuda.is_available():
            return "cuda"
        if getattr(self.torch.backends, "mps", None) and self.torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def mean_pooling(token_embeddings: Any, attention_mask: Any) -> Any:
    """Mean-pool token embeddings with an attention mask."""

    import torch

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1),
        min=1e-9,
    )


def load_generated_comments(input_path: Path) -> tuple[dict[str, list[ThreadComment]], dict[str, Any]]:
    """Load comments from generated GEO `discussion.json`."""

    json_path = input_path / "discussion.json" if input_path.is_dir() else input_path
    data = json.loads(json_path.read_text(encoding="utf-8"))
    comments_by_thread: dict[str, list[ThreadComment]] = {}

    for post in data.get("posts", []):
        thread_id = str(post.get("post_id", ""))
        post_text = clean_text(post.get("content") or "")
        thread_title = post_text[:120]
        rows: list[ThreadComment] = []

        def walk(comments: list[dict[str, Any]]) -> None:
            for comment in comments:
                text = clean_text(comment.get("content") or "")
                if is_usable_comment(text):
                    rows.append(
                        ThreadComment(
                            thread_id=thread_id,
                            thread_title=thread_title,
                            comment_id=str(comment.get("comment_id", "")),
                            parent_id=str(comment.get("parent_comment_id") or f"post:{thread_id}"),
                            author=str(comment.get("author") or ""),
                            text=text,
                            depth=int(comment.get("depth") or 0),
                        )
                    )
                walk(comment.get("replies") or [])

        walk(post.get("comments") or [])
        comments_by_thread[thread_id] = rows

    return comments_by_thread, {"json_path": str(json_path), "post_count": len(data.get("posts", []))}


def load_real_comments(input_path: Path) -> tuple[dict[str, list[ThreadComment]], dict[str, Any]]:
    """Load comments from scraped Reddit JSONL bundle."""

    folder = input_path if input_path.is_dir() else input_path.parent
    comments_path = find_single(folder, "*.comments.jsonl")
    posts_path = find_single(
        folder,
        "*.jsonl",
        exclude_suffixes=(".comments.jsonl", ".comments.raw.jsonl", ".raw.jsonl"),
    )
    posts = {str(row.get("id")): row for row in read_jsonl(posts_path)}
    comments_by_thread: dict[str, list[ThreadComment]] = {}
    raw_comments = read_jsonl(comments_path)

    for row in raw_comments:
        thread_id = str(row.get("post_id") or "")
        post = posts.get(thread_id, {})
        text = clean_text(row.get("body") or "")
        if not is_usable_comment(text):
            continue
        comments_by_thread.setdefault(thread_id, []).append(
            ThreadComment(
                thread_id=thread_id,
                thread_title=str(post.get("title") or row.get("post_title") or ""),
                comment_id=str(row.get("comment_fullname") or row.get("comment_id") or ""),
                parent_id=str(row.get("parent_id") or ""),
                author=str(row.get("author") or ""),
                text=text,
                depth=int(row.get("depth") or 0),
            )
        )

    return comments_by_thread, {
        "folder": str(folder),
        "posts_path": str(posts_path),
        "comments_path": str(comments_path),
        "post_count": len(posts),
        "raw_comment_count": len(raw_comments),
    }


def score_thread(
    thread_id: str,
    comments: list[ThreadComment],
    embeddings: np.ndarray,
    top_k: int,
    include_embeddings: bool,
) -> dict[str, Any]:
    """Compute semantic-uniformity metrics for one thread."""

    similarities = pairwise_cosine_values(embeddings)
    pair_count = int(len(similarities))
    top_values = np.sort(similarities)[-min(top_k, pair_count) :] if pair_count else np.array([])
    comment_rows = []
    for idx, comment in enumerate(comments):
        row = asdict(comment)
        if include_embeddings:
            row["embedding"] = [round(float(value), 6) for value in embeddings[idx].tolist()]
        comment_rows.append(row)

    return {
        "thread_id": thread_id,
        "thread_title": comments[0].thread_title if comments else "",
        "comment_count": len(comments),
        "pair_count": pair_count,
        "mean_cosine_similarity": float(np.mean(similarities)) if pair_count else 0.0,
        "median_cosine_similarity": float(np.median(similarities)) if pair_count else 0.0,
        "top_k": top_k,
        "top_k_mean_cosine_similarity": float(np.mean(top_values)) if len(top_values) else 0.0,
        "p90_cosine_similarity": float(np.percentile(similarities, 90)) if pair_count else 0.0,
        "max_cosine_similarity": float(np.max(similarities)) if pair_count else 0.0,
        "min_cosine_similarity": float(np.min(similarities)) if pair_count else 0.0,
        "comments": comment_rows,
    }


def pairwise_cosine_values(embeddings: np.ndarray) -> np.ndarray:
    """Return upper-triangle pairwise cosine similarities for normalized vectors."""

    if embeddings.shape[0] < 2:
        return np.array([], dtype=np.float32)
    embeddings = np.nan_to_num(
        embeddings.astype(np.float64, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    indices = np.triu_indices(embeddings.shape[0], k=1)
    similarities = np.einsum("ij,ij->i", embeddings[indices[0]], embeddings[indices[1]])
    return np.clip(similarities, -1.0, 1.0).astype(np.float32)


def aggregate_overall(thread_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-thread metrics into one summary."""

    total_comments = sum(int(row["comment_count"]) for row in thread_results)
    total_pairs = sum(int(row["pair_count"]) for row in thread_results)
    if not thread_results or total_pairs == 0:
        return {
            "comment_count": total_comments,
            "thread_count": len(thread_results),
            "pair_count": total_pairs,
            "weighted_mean_cosine_similarity": 0.0,
            "median_thread_cosine_similarity": 0.0,
            "weighted_top_k_mean_cosine_similarity": 0.0,
        }

    weighted_mean = weighted_average(
        [(row["mean_cosine_similarity"], row["pair_count"]) for row in thread_results]
    )
    weighted_top_k = weighted_average(
        [(row["top_k_mean_cosine_similarity"], row["pair_count"]) for row in thread_results]
    )
    return {
        "comment_count": total_comments,
        "thread_count": len(thread_results),
        "pair_count": total_pairs,
        "weighted_mean_cosine_similarity": weighted_mean,
        "median_thread_cosine_similarity": median(
            [float(row["median_cosine_similarity"]) for row in thread_results]
        ),
        "mean_thread_cosine_similarity": float(
            np.mean([float(row["mean_cosine_similarity"]) for row in thread_results])
        ),
        "weighted_top_k_mean_cosine_similarity": weighted_top_k,
        "max_thread_mean_cosine_similarity": max(
            float(row["mean_cosine_similarity"]) for row in thread_results
        ),
    }


def weighted_average(values: Iterable[tuple[float, int]]) -> float:
    """Compute a weighted average."""

    values = list(values)
    denom = sum(weight for _, weight in values)
    if denom <= 0:
        return 0.0
    return float(sum(value * weight for value, weight in values) / denom)


def median(values: list[float]) -> float:
    """Median with empty-list protection."""

    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2)


def is_usable_comment(text: str) -> bool:
    """Filter deleted or extremely short comments."""

    return len(clean_text(text).split()) >= 2


def default_output_path(input_path: Path) -> Path:
    """Choose a default output path next to the evaluated input."""

    if input_path.is_dir():
        return input_path / "semantic_uniformity_results.json"
    return input_path.with_suffix(input_path.suffix + ".semantic_uniformity_results.json")


if __name__ == "__main__":
    main()
