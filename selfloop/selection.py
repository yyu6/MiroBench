#!/usr/bin/env python3
"""Which comments to spend an API call on.

The budget is small on purpose -- a round rewrites a tenth of a thread, not all
of it -- so the ranking decides whether the round moves anything. Each metric
ranks by that comment's own contribution to the thread's score, computed by
leave-one-out on the real quantity rather than by a proxy:

    contribution(i) = thread_score(all) - thread_score(all but i)

A comment whose removal drops the thread's semantic cosine most is the comment
that is most redundant with its neighbours. That is exactly what should be
rewritten, and it needs no heuristic about what "too similar" looks like.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_scorer as C  # noqa: E402


def _drop(values: list, index: int) -> list:
    return values[:index] + values[index + 1:]


def semantic_contributions(texts: Sequence[str]) -> list[float]:
    import numpy as np

    vectors = C.embed(list(texts))
    n = len(texts)
    if n < 3:
        return [0.0] * n
    full = C.semantic_mean_cosine(vectors)
    out = []
    for i in range(n):
        keep = np.delete(vectors, i, axis=0)
        out.append(full - C.semantic_mean_cosine(keep))
    return out


def self_bleu_contributions(texts: Sequence[str]) -> list[float]:
    """Leave-one-out self-BLEU, from one pair matrix instead of n rescores.

    Dropping comment i removes exactly the n-1 pairs that touch it, so the
    thread mean without i is (total - row_i) / (pairs - (n-1)). Recomputing the
    whole thread per comment made this O(n^3) and cost 79s on a 96-comment
    thread; this is O(n^2) and identical, which
    `test_leave_one_out_matches_rescore` checks.
    """
    import score_thread_self_bleu as sb

    n = len(texts)
    if n < 3:
        return [0.0] * n
    tokens = [sb.tokenize(t) for t in texts]
    rows = [0.0] * n
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            value = sb.symmetric_pair_bleu(tokens[i], tokens[j], 4)
            total += value
            rows[i] += value
            rows[j] += value
    pairs = n * (n - 1) // 2
    full = total / pairs
    remaining = pairs - (n - 1)
    if remaining <= 0:
        return [0.0] * n
    return [full - (total - rows[i]) / remaining for i in range(n)]


def bertscore_contributions(texts: Sequence[str]) -> list[float]:
    """Rank by lexical + semantic redundancy, with no BERTScore call at all.

    Measured 2026-09-04 against the real per-comment `bert_pair_f1` ordering:
    Spearman +0.761 on a 42-comment thread (p<0.001), and the proxy's top 15%
    contains 4 of the model's top 6. Running the model instead cost 91s on that
    thread and 142s on a 96-comment one, made BERTScore selection the slowest
    step in the loop, and was where the process was being killed between
    rounds.

    This is a SELECTION heuristic. It decides which comments get an API call;
    the candidate that call produces is still scored with the real BERTScore,
    and the round is still gated on the official scorer. Being 76% right about
    which comments to try costs a little search efficiency and nothing else.
    """
    n = len(texts)
    if n < 3:
        return [0.0] * n
    lexical = self_bleu_contributions(texts)
    semantic = semantic_contributions(texts)

    def unit(values: list[float]) -> list[float]:
        span = max(values) - min(values)
        return [0.0] * len(values) if span <= 0 else [(v - min(values)) / span for v in values]

    return [a + b for a, b in zip(unit(lexical), unit(semantic))]


def per_comment_labels(texts: Sequence[str], metric: str) -> list[float]:
    """For the rate metrics, a comment scores 1 when it is in the wrong class."""
    if metric in ("polite_rate", "impolite_rate", "neutral_rate"):
        want = metric.replace("_rate", "")
        labels = C.politeness_labels(list(texts))
        return [1.0 if label == want else 0.0 for label in labels]
    return [0.0] * len(texts)


def length_contributions(texts: Sequence[str]) -> list[float]:
    """Distance from the thread's own mean length, in words."""
    counts = [len(t.split()) for t in texts]
    mean = sum(counts) / max(1, len(counts))
    return [abs(c - mean) for c in counts]


RANKERS: dict[str, Callable[[Sequence[str]], list[float]]] = {
    "semantic_mean_cosine": semantic_contributions,
    "self_bleu_4": self_bleu_contributions,
    "self_bertscore_mean_f1": bertscore_contributions,
    "length_cv": length_contributions,
}


def rank(texts: Sequence[str], metric: str, *, too_high: bool) -> list[int]:
    """Comment indices, most worth rewriting first."""
    if metric in RANKERS:
        scores = RANKERS[metric](texts)
    elif metric in ("polite_rate", "impolite_rate", "neutral_rate"):
        # Too many of a class -> rewrite members of it. Too few -> rewrite the
        # ones that are not, since those are the ones that can be moved in.
        scores = per_comment_labels(texts, metric)
        if not too_high:
            scores = [1.0 - s for s in scores]
    else:
        # story / emotion / disagreement: no cheap per-comment contribution that
        # is worth its model call at selection time, so fall back to the longest
        # comments, which carry the most classifier signal.
        scores = [float(len(t.split())) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: -scores[i])
    return order


def budget(n: int, share: float, *, floor: int = 1, ceiling: int = 12) -> int:
    return max(floor, min(ceiling, int(round(n * share))))
