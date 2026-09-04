#!/usr/bin/env python3
"""Read and write a run's discussion.json without disturbing anything else.

The reviser edits comment text in place. Everything else in the artifact -- the
reply tree, ids, authors, timestamps, every planner field -- is preserved
byte-for-byte, which is what lets `avg_depth` and `structural_virality` be
skipped during rescoring rather than assumed to be unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
import sys
from pathlib import Path
from typing import Any

_EVAL = Path(__file__).resolve().parents[1] / "scripts" / "evaluation"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))


@dataclass
class Thread:
    run_dir: Path
    post_id: str
    title: str
    nodes: list[dict[str, Any]]      # flattened, tree order
    parents: list[int]               # index of parent, -1 for top level
    scored: list[int] = field(default_factory=list)   # nodes the scorers read

    @property
    def scored_texts(self) -> list[str]:
        """The comment texts as the official scorers see them, in their order."""
        from score_thread_semantic_uniformity import clean_text

        return [clean_text(str(self.nodes[i].get("content") or "")) for i in self.scored]

    @property
    def texts(self) -> list[str]:
        return [str(node.get("content") or "") for node in self.nodes]

    def parent_text(self, index: int) -> str:
        p = self.parents[index]
        return str(self.nodes[p].get("content") or "") if p >= 0 else ""

    def set_text(self, index: int, value: str) -> None:
        self.nodes[index]["content"] = value
        self.nodes[index]["word_count"] = len(value.split())


def _flatten(comments: list[dict[str, Any]], parent: int,
             nodes: list[dict[str, Any]], parents: list[int]) -> None:
    for comment in comments or []:
        nodes.append(comment)
        parents.append(parent)
        index = len(nodes) - 1
        _flatten(comment.get("replies") or [], index, nodes, parents)


def load(run_dir: Path) -> Thread:
    """Flatten the thread, then keep only what the official scorers see.

    Two things the scorers do that a naive walk does not: `clean_text`
    normalizes whitespace, and `is_usable_comment` drops anything under two
    words -- one "Same" in seed 1 is invisible to every metric. Targeting a
    comment the scorers never read would spend an API call that cannot move a
    number, and, worse, would put this module's indices out of step with the
    per-comment rows the scorers emit. So `scored` carries the official view
    and every index in this package refers to it.
    """
    payload = json.loads((run_dir / "discussion.json").read_text())
    post = (payload.get("posts") or [{}])[0]
    nodes: list[dict[str, Any]] = []
    parents: list[int] = []
    _flatten(post.get("comments") or [], -1, nodes, parents)
    thread = Thread(run_dir=run_dir, post_id=str(post.get("post_id") or run_dir.name),
                    title=str(post.get("title") or ""), nodes=nodes, parents=parents)
    thread._payload = payload  # type: ignore[attr-defined]
    thread.scored = _scored_indices(nodes)
    return thread


def _scored_indices(nodes: list[dict[str, Any]]) -> list[int]:
    from score_thread_semantic_uniformity import clean_text, is_usable_comment

    return [i for i, node in enumerate(nodes)
            if is_usable_comment(clean_text(str(node.get("content") or "")))]


def save(thread: Thread) -> None:
    """Persist edits. `nodes` holds references into the payload, so the tree is
    already updated; this only rewrites the file."""
    # The trailing newline matches what the generator writes, so a rejected
    # round restores the file byte-for-byte rather than off by one character.
    (thread.run_dir / "discussion.json").write_text(
        json.dumps(thread._payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"  # type: ignore[attr-defined]
    )


def snapshot(thread: Thread) -> list[str]:
    return list(thread.texts)


def restore(thread: Thread, texts: list[str]) -> None:
    for index, value in enumerate(texts):
        thread.set_text(index, value)
    save(thread)
