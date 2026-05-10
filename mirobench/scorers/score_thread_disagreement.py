#!/usr/bin/env python3
"""Score thread-level disagreement with the local Stance_Rel checkpoint.

The Stance_Rel artifact in this repo contains a trained RoBERTa checkpoint plus
graph/relation weights, but it does not include the original inference code.
This script provides a practical evaluation wrapper for GEO discussions:

1. Load real Reddit bundles or generated `discussion.json` files.
2. Extract parent -> reply text pairs within each thread.
3. Run the local Stance_Rel RoBERTa stance head on each pair.
4. Aggregate pair-level stance predictions into per-thread disagreement scores.

Important model note:
The saved classifier head expects a 1536-dimensional input: 768 RoBERTa pooled
features plus 768 relation/graph features. The original graph inference path is
not present in `Stance_Rel`, so this runner uses a zero relation feature and
loads the trained RoBERTa + stance classifier weights. This is enough for a
consistent text-pair disagreement score, but it is not a byte-for-byte
reproduction of the original graph-augmented inference pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


LABELS = ("disagree", "neutral", "agree")


@dataclass
class StancePair:
    """One parent -> reply pair that can be scored for stance."""

    thread_id: str
    thread_title: str
    parent_id: str
    reply_id: str
    parent_author: str
    reply_author: str
    parent_text: str
    reply_text: str
    depth: int


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Score Reddit thread disagreement using local Stance_Rel weights."
    )
    parser.add_argument("input", help="Real discussion folder/file or generated run folder/file.")
    parser.add_argument(
        "--target-kind",
        choices=["auto", "real", "generated"],
        default="auto",
        help="Input schema. auto detects generated discussion.json vs real Reddit bundle.",
    )
    parser.add_argument(
        "--model-dir",
        default="Stance_Rel/RoBERT_rel_1.5e-05",
        help="Directory containing config/tokenizer/pytorch_model.bin.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Where to write JSON results. Defaults next to input.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional cap for quick smoke tests.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device. auto uses cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--label-order",
        default=",".join(LABELS),
        help="Comma-separated classifier row order. Default: disagree,neutral,agree.",
    )
    parser.add_argument(
        "--graph-author",
        default="reply",
        choices=["reply", "parent", "none"],
        help=(
            "Which author node to use for the Stance_Rel graph feature. "
            "Use none for zero-graph text-only scoring."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.is_absolute():
        model_dir = (Path.cwd() / model_dir).resolve()

    target_kind = args.target_kind
    if target_kind == "auto":
        target_kind = detect_target_kind(input_path)

    if target_kind == "generated":
        pairs, source_meta = load_generated_pairs(input_path)
    else:
        pairs, source_meta = load_real_pairs(input_path)

    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    label_order = tuple(label.strip() for label in args.label_order.split(",") if label.strip())
    if set(label_order) != set(LABELS) or len(label_order) != 3:
        raise ValueError("--label-order must contain exactly: disagree,neutral,agree")

    scorer = StanceRelScorer(
        model_dir=model_dir,
        label_order=label_order,
        device=args.device,
        max_length=args.max_length,
        graph_author=args.graph_author,
    )
    scored_pairs = scorer.score_pairs(pairs, batch_size=max(1, args.batch_size))
    thread_scores = aggregate_thread_scores(scored_pairs)

    result = {
        "meta": {
            "input": str(input_path),
            "target_kind": target_kind,
            "model_dir": str(model_dir),
            "model_note": (
                "Uses local Stance_Rel RoBERTa stance head plus available RGCN user "
                "embeddings. Missing authors fall back to a zero graph feature because "
                "the original graph inference code is not present."
            ),
            "label_order": list(label_order),
            "graph_author": args.graph_author,
            "graph_feature_coverage": scorer.graph_feature_coverage,
            "pair_count": len(scored_pairs),
            "thread_count": len(thread_scores),
            "source": source_meta,
        },
        "overall": aggregate_overall(scored_pairs, thread_scores),
        "threads": thread_scores,
        "pairs": scored_pairs,
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
        f"pairs={overall['pair_count']} threads={overall['thread_count']} "
        f"mean_disagree_probability={overall['mean_disagree_probability']:.4f} "
        f"hard_disagree_rate={overall['hard_disagree_rate']:.4f}"
    )


class StanceRelScorer:
    """Load Stance_Rel weights and score text pairs."""

    def __init__(
        self,
        model_dir: Path,
        label_order: tuple[str, str, str],
        device: str,
        max_length: int,
        graph_author: str,
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, RobertaConfig, RobertaModel
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit(
                "This scorer requires torch and transformers. Use a Python environment "
                "that has them installed, or install them before running this script."
            ) from exc

        self.torch = torch
        self.label_order = label_order
        self.max_length = max_length
        self.graph_author = graph_author
        self.device = self._resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

        config = RobertaConfig.from_pretrained(model_dir, local_files_only=True)
        self.model = ZeroGraphRobertaStance(config, RobertaModel)
        state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        allowed_missing: set[str] = set()
        allowed_unexpected = {"roberta.embeddings.position_ids"}
        unexpected_non_graph = [
            key
            for key in unexpected
            if not key.startswith(("g1.", "d1.", "d2."))
            and key not in allowed_unexpected
        ]
        if missing and set(missing) != allowed_missing:
            raise RuntimeError(f"Missing model weights: {missing}")
        if unexpected_non_graph:
            raise RuntimeError(f"Unexpected non-graph model weights: {unexpected_non_graph}")
        self.model.to(self.device)
        self.model.eval()
        self.author_embeddings = self._load_author_embeddings(model_dir)
        self.graph_feature_coverage = {
            "known_author_pairs": 0,
            "total_pairs": 0,
            "coverage": 0.0,
        }

    def score_pairs(self, pairs: list[StancePair], batch_size: int) -> list[dict[str, Any]]:
        """Return pair dictionaries with stance probabilities attached."""

        scored: list[dict[str, Any]] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = self.tokenizer(
                [pair.parent_text for pair in batch],
                [pair.reply_text for pair in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            graph_vectors, known_count = self._graph_vectors_for_batch(batch)
            with self.torch.inference_mode():
                logits = self.model(**encoded, graph_vectors=graph_vectors)
                probs = self.torch.softmax(logits, dim=-1).detach().cpu().tolist()
            self.graph_feature_coverage["known_author_pairs"] += known_count
            self.graph_feature_coverage["total_pairs"] += len(batch)
            total_pairs = self.graph_feature_coverage["total_pairs"]
            self.graph_feature_coverage["coverage"] = (
                self.graph_feature_coverage["known_author_pairs"] / total_pairs
                if total_pairs
                else 0.0
            )
            for pair, row in zip(batch, probs):
                prob_map = {label: float(row[idx]) for idx, label in enumerate(self.label_order)}
                pred_label = max(prob_map, key=prob_map.get)
                scored.append(
                    {
                        **asdict(pair),
                        "stance_probs": prob_map,
                        "pred_label": pred_label,
                        "disagree_probability": prob_map["disagree"],
                    }
                )
        return scored

    def _load_author_embeddings(self, model_dir: Path) -> dict[str, Any]:
        """Load Stance_Rel user graph embeddings when available."""

        if self.graph_author == "none":
            return {}
        stance_root = model_dir.parent
        mapping_path = stance_root / "utils" / "unique_nodes_n_mapping_bert.pkl"
        rgcn_path = model_dir / "_rgcn.pt"
        if not mapping_path.exists() or not rgcn_path.exists():
            return {}

        with mapping_path.open("rb") as handle:
            mapping = pickle.load(handle)
        rgcn_state = self.torch.load(rgcn_path, map_location="cpu")
        weights = rgcn_state.get("entity_embedding.weight")
        if weights is None:
            return {}
        return {
            "mapping": mapping,
            "weights": weights,
        }

    def _graph_vectors_for_batch(self, batch: list[StancePair]) -> tuple[Any, int]:
        """Return author graph vectors for the batch plus known-author count."""

        vectors = []
        known_count = 0
        weights = self.author_embeddings.get("weights") if self.author_embeddings else None
        mapping = self.author_embeddings.get("mapping") if self.author_embeddings else None
        for pair in batch:
            vector = None
            if weights is not None and mapping is not None:
                author = pair.reply_author if self.graph_author == "reply" else pair.parent_author
                author_index = mapping.get(author)
                if author_index is None:
                    author_index = mapping.get(str(author).lower())
                if author_index is not None:
                    vector = weights[int(author_index)]
                    known_count += 1
            if vector is None:
                vector = self.torch.zeros(100, dtype=self.torch.float32)
            vectors.append(vector)
        stacked = self.torch.stack(vectors).to(self.device)
        return stacked, known_count

    def _resolve_device(self, requested: str) -> str:
        """Resolve `auto` into a usable torch device string."""

        if requested != "auto":
            return requested
        if self.torch.cuda.is_available():
            return "cuda"
        if getattr(self.torch.backends, "mps", None) and self.torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class ZeroGraphRobertaStance:
    """Small nn.Module matching the saved Stance_Rel state dict.

    It is defined dynamically so importing this script without torch installed
    still shows a useful CLI error instead of failing at import time.
    """

    def __new__(cls, config: Any, roberta_model_cls: Any):  # type: ignore[override]
        import torch

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.roberta = roberta_model_cls(config)
                self.g1 = torch.nn.Linear(100, 768)
                self.d1 = torch.nn.Linear(768, 768)
                self.d2 = torch.nn.Linear(768, 100)
                self.sent_cls = torch.nn.Linear(1536, 3)

            def forward(
                self,
                input_ids: Any,
                attention_mask: Any | None = None,
                graph_vectors: Any | None = None,
                **kwargs: Any,
            ) -> Any:
                del kwargs
                outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.pooler_output
                if pooled is None:
                    pooled = outputs.last_hidden_state[:, 0, :]
                if graph_vectors is None:
                    graph_vectors = torch.zeros(
                        (pooled.shape[0], 100),
                        dtype=pooled.dtype,
                        device=pooled.device,
                    )
                graph_feature = self.g1(graph_vectors.to(dtype=pooled.dtype, device=pooled.device))
                return self.sent_cls(torch.cat([pooled, graph_feature], dim=-1))

        return _Model()


def detect_target_kind(input_path: Path) -> str:
    """Infer real vs generated discussion schema."""

    json_path = input_path / "discussion.json" if input_path.is_dir() else input_path
    if json_path.name == "discussion.json" and json_path.exists():
        return "generated"
    if input_path.is_dir() and any(input_path.glob("*.comments.jsonl")):
        return "real"
    return "real"


def load_generated_pairs(input_path: Path) -> tuple[list[StancePair], dict[str, Any]]:
    """Extract parent->reply pairs from generated GEO discussion JSON."""

    json_path = input_path / "discussion.json" if input_path.is_dir() else input_path
    data = json.loads(json_path.read_text(encoding="utf-8"))
    pairs: list[StancePair] = []

    for post in data.get("posts", []):
        thread_id = str(post.get("post_id", ""))
        post_text = clean_text(post.get("content") or "")
        thread_title = post_text[:100]
        post_author = str(post.get("author") or "OP")

        def walk(comments: list[dict[str, Any]], parent_id: str, parent_text: str, parent_author: str) -> None:
            for comment in comments:
                reply_text = clean_text(comment.get("content") or "")
                if is_usable_pair(parent_text, reply_text):
                    pairs.append(
                        StancePair(
                            thread_id=thread_id,
                            thread_title=thread_title,
                            parent_id=parent_id,
                            reply_id=str(comment.get("comment_id", "")),
                            parent_author=parent_author,
                            reply_author=str(comment.get("author") or ""),
                            parent_text=parent_text,
                            reply_text=reply_text,
                            depth=int(comment.get("depth") or 0),
                        )
                    )
                walk(
                    comment.get("replies") or [],
                    parent_id=str(comment.get("comment_id", "")),
                    parent_text=reply_text,
                    parent_author=str(comment.get("author") or ""),
                )

        walk(post.get("comments") or [], f"post:{thread_id}", post_text, post_author)

    return pairs, {"json_path": str(json_path), "post_count": len(data.get("posts", []))}


def load_real_pairs(input_path: Path) -> tuple[list[StancePair], dict[str, Any]]:
    """Extract parent->reply pairs from scraped Reddit JSONL bundles."""

    folder = input_path if input_path.is_dir() else input_path.parent
    comments_path = find_single(folder, "*.comments.jsonl")
    posts_path = find_single(folder, "*.jsonl", exclude_suffixes=(".comments.jsonl", ".comments.raw.jsonl", ".raw.jsonl"))

    posts = {str(row.get("id")): row for row in read_jsonl(posts_path)}
    comments = read_jsonl(comments_path)
    comments_by_fullname = {
        str(row.get("comment_fullname") or f"t1_{row.get('comment_id')}"): row
        for row in comments
    }
    pairs: list[StancePair] = []

    for row in comments:
        post_id = str(row.get("post_id") or "")
        post = posts.get(post_id, {})
        parent_id = str(row.get("parent_id") or "")
        if parent_id.startswith("t1_"):
            parent_row = comments_by_fullname.get(parent_id, {})
            parent_text = clean_text(parent_row.get("body") or "")
            parent_author = str(parent_row.get("author") or "")
        else:
            parent_text = clean_text(
                "\n\n".join(
                    part
                    for part in [
                        str(post.get("title") or row.get("post_title") or ""),
                        str(post.get("selftext") or ""),
                    ]
                    if part
                )
            )
            parent_author = str(post.get("author") or "OP")

        reply_text = clean_text(row.get("body") or "")
        if not is_usable_pair(parent_text, reply_text):
            continue
        pairs.append(
            StancePair(
                thread_id=post_id,
                thread_title=str(post.get("title") or row.get("post_title") or ""),
                parent_id=parent_id,
                reply_id=str(row.get("comment_fullname") or row.get("comment_id") or ""),
                parent_author=parent_author,
                reply_author=str(row.get("author") or ""),
                parent_text=parent_text,
                reply_text=reply_text,
                depth=int(row.get("depth") or 0),
            )
        )

    return pairs, {
        "folder": str(folder),
        "posts_path": str(posts_path),
        "comments_path": str(comments_path),
        "post_count": len(posts),
        "raw_comment_count": len(comments),
    }


def aggregate_thread_scores(scored_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate pair-level stance scores by thread."""

    by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in scored_pairs:
        by_thread[pair["thread_id"]].append(pair)

    threads: list[dict[str, Any]] = []
    for thread_id, pairs in by_thread.items():
        disagree_probs = [float(pair["disagree_probability"]) for pair in pairs]
        pred_counts = Counter(str(pair["pred_label"]) for pair in pairs)
        title = str(pairs[0].get("thread_title") or "")
        threads.append(
            {
                "thread_id": thread_id,
                "thread_title": title,
                "pair_count": len(pairs),
                "mean_disagree_probability": safe_mean(disagree_probs),
                "hard_disagree_rate": pred_counts.get("disagree", 0) / len(pairs),
                "label_counts": dict(pred_counts),
                "top_disagreement_pairs": top_pairs(pairs, n=5),
            }
        )

    threads.sort(
        key=lambda row: (
            float(row["mean_disagree_probability"]),
            int(row["pair_count"]),
        ),
        reverse=True,
    )
    return threads


def aggregate_overall(scored_pairs: list[dict[str, Any]], thread_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all pair and thread scores into one summary."""

    if not scored_pairs:
        return {
            "pair_count": 0,
            "thread_count": len(thread_scores),
            "mean_disagree_probability": 0.0,
            "hard_disagree_rate": 0.0,
            "label_counts": {},
        }
    pred_counts = Counter(str(pair["pred_label"]) for pair in scored_pairs)
    disagree_probs = [float(pair["disagree_probability"]) for pair in scored_pairs]
    return {
        "pair_count": len(scored_pairs),
        "thread_count": len(thread_scores),
        "mean_disagree_probability": safe_mean(disagree_probs),
        "median_thread_disagree_probability": median(
            [float(thread["mean_disagree_probability"]) for thread in thread_scores]
        ),
        "hard_disagree_rate": pred_counts.get("disagree", 0) / len(scored_pairs),
        "label_counts": dict(pred_counts),
    }


def top_pairs(pairs: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Return a compact sample of the strongest disagreement pairs."""

    out: list[dict[str, Any]] = []
    for pair in sorted(pairs, key=lambda item: float(item["disagree_probability"]), reverse=True)[:n]:
        out.append(
            {
                "reply_id": pair["reply_id"],
                "parent_author": pair["parent_author"],
                "reply_author": pair["reply_author"],
                "pred_label": pair["pred_label"],
                "disagree_probability": pair["disagree_probability"],
                "parent_text": truncate(pair["parent_text"], 220),
                "reply_text": truncate(pair["reply_text"], 220),
            }
        )
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries."""

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_single(folder: Path, pattern: str, exclude_suffixes: tuple[str, ...] = ()) -> Path:
    """Find one file in a folder matching a glob pattern."""

    candidates = [
        path
        for path in folder.glob(pattern)
        if not any(path.name.endswith(suffix) for suffix in exclude_suffixes)
    ]
    if not candidates:
        raise FileNotFoundError(f"No {pattern} file found in {folder}")
    candidates.sort(key=lambda path: (len(path.name), path.name))
    return candidates[0]


def clean_text(text: Any) -> str:
    """Normalize Reddit text and remove deleted/empty placeholders."""

    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if cleaned.lower() in {"[deleted]", "[removed]", "deleted", "removed"}:
        return ""
    return cleaned


def is_usable_pair(parent_text: str, reply_text: str) -> bool:
    """Return whether a text pair is substantial enough to score."""

    return len(parent_text.split()) >= 3 and len(reply_text.split()) >= 2


def safe_mean(values: Iterable[float]) -> float:
    """Mean with empty-list protection."""

    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def median(values: list[float]) -> float:
    """Return the median of numeric values."""

    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2)


def truncate(text: str, limit: int) -> str:
    """Trim long text for JSON samples."""

    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def default_output_path(input_path: Path) -> Path:
    """Choose a default JSON output path next to the input."""

    if input_path.is_dir():
        return input_path / "stance_disagreement_results.json"
    return input_path.with_suffix(input_path.suffix + ".stance_disagreement_results.json")


if __name__ == "__main__":
    main()
