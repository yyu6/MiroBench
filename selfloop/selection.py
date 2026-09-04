#!/usr/bin/env python3
"""Which comments to spend an API call on, and what to tell the model about them.

The budget is small on purpose -- a round rewrites a tenth of a thread, not all
of it -- so the ranking decides whether the round moves anything. Each target
ranks by that comment's own contribution to the thread's score, computed by
leave-one-out on the real quantity rather than by a proxy:

    contribution(i) = thread_score(all) - thread_score(all but i)

A comment whose removal drops the thread's semantic cosine most is the comment
that is most redundant with its neighbours. That is exactly what should be
rewritten, and it needs no heuristic about what "too similar" looks like.

`evidence` is the other half: the same measurements, rendered for the prompt, so
the model is told which comments it is duplicating and on which words instead of
being told a metric name and two floats.

Every ranker takes the ThreadCache and GuardCache the round already built. Both
hold exactly what the rankers need -- embeddings, tokens, per-comment classifier
outputs -- so passing them in removes a second embedding pass and a second
O(n^2) BLEU matrix per thread per round.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import candidate_scorer as C  # noqa: E402
import strategies as S  # noqa: E402


def semantic_contributions(texts: Sequence[str], vectors: Any = None) -> list[float]:
    import numpy as np

    n = len(texts)
    if n < 3:
        return [0.0] * n
    if vectors is None:
        vectors = C.embed(list(texts))
    full = C.semantic_mean_cosine(vectors)
    return [full - C.semantic_mean_cosine(np.delete(vectors, i, axis=0))
            for i in range(n)]


def self_bleu_contributions(texts: Sequence[str], tokens: Any = None) -> list[float]:
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
    if tokens is None:
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


def similarity_contributions(texts: Sequence[str], vectors: Any = None,
                             tokens: Any = None) -> list[float]:
    """Rank by lexical + semantic redundancy, with no BERTScore call at all.

    This is the ranker for the whole similarity group, and it was already the
    right one for `self_bertscore` alone. Measured 2026-09-04 against the real
    per-comment `bert_pair_f1` ordering: Spearman +0.761 on a 42-comment thread
    (p<0.001), and the proxy's top 15% contains 4 of the model's top 6. Running
    the model instead cost 91s on that thread and 142s on a 96-comment one, and
    was where the process kept being killed between rounds.

    This is a SELECTION heuristic. It decides which comments get an API call;
    the candidate that call produces is still scored with the real BERTScore,
    and the round is still gated on the official scorer.
    """
    n = len(texts)
    if n < 3:
        return [0.0] * n
    lexical = self_bleu_contributions(texts, tokens)
    semantic = semantic_contributions(texts, vectors)

    def unit(values: list[float]) -> list[float]:
        span = max(values) - min(values)
        return [0.0] * len(values) if span <= 0 else [(v - min(values)) / span for v in values]

    return [a + b for a, b in zip(unit(lexical), unit(semantic))]


def register_contributions(texts: Sequence[str], guard: Any,
                           wants: dict[str, float]) -> list[float]:
    """Which comments hold the thread's register back, from the guard's own
    per-comment outputs -- no extra model pass.

    Direction is read off the gap, never assumed: when the thread has too much
    neutrality the neutral comments score, and when it has too little the others
    do. The fallback this replaces ranked by comment length, which is only a
    proxy for "carries classifier signal" and is otherwise arbitrary.
    """
    n = len(texts)
    if guard is None or n < 3:
        return [float(len(t.split())) for t in texts]
    have = guard.values()
    score = [0.0] * n
    for metric, klass in (("polite_rate", "polite"), ("neutral_rate", "neutral"),
                          ("impolite_rate", "impolite")):
        if metric not in wants:
            continue
        gap = have[metric] - wants[metric]
        for i in range(n):
            member = guard.polite[i] == klass
            if member if gap > 0 else not member:
                score[i] += abs(gap)
    if "mean_story_probability" in wants:
        gap = have["mean_story_probability"] - wants["mean_story_probability"]
        for i in range(n):
            score[i] += abs(gap) * (guard.story[i] if gap > 0 else 1.0 - guard.story[i])
    if "emotion_entropy" in wants:
        gap = have["emotion_entropy"] - wants["emotion_entropy"]
        counts = collections.Counter(guard.emotion)
        for i in range(n):
            # Too little spread: move the comments carrying the crowd's feeling.
            # Too much: move the outliers back toward it.
            share = counts[guard.emotion[i]] / n
            score[i] += abs(gap) * (share if gap < 0 else 1.0 - share)
    return score


def politeness_offenders(texts: Sequence[str], metric: str, guard: Any,
                         too_high: bool) -> list[float]:
    """A comment scores 1 when it is in the wrong class for a single-metric
    politeness round. Too many of a class -> rewrite members of it; too few ->
    rewrite the ones that are not, since those are what can be moved in."""
    want = metric.replace("_rate", "")
    labels = guard.polite if guard is not None else C.politeness_labels(list(texts))
    return [1.0 if (label == want) == too_high else 0.0 for label in labels]


def length_contributions(texts: Sequence[str]) -> list[float]:
    """Distance from the thread's own mean length, in words."""
    counts = [len(t.split()) for t in texts]
    mean = sum(counts) / max(1, len(counts))
    return [abs(c - mean) for c in counts]


def rank(texts: Sequence[str], target: str, *, too_high: bool,
         cache: Any = None, guard: Any = None,
         wants: dict[str, float] | None = None) -> list[int]:
    """Comment indices, most worth rewriting first. `target` is a group or a metric."""
    vectors = getattr(cache, "vectors", None)
    tokens = getattr(cache, "tokens", None)
    if target == "similarity" or target in S.SIMILARITY:
        scores = similarity_contributions(texts, vectors, tokens)
    elif target == "register":
        scores = register_contributions(texts, guard, wants or {})
    elif target in ("polite_rate", "impolite_rate", "neutral_rate"):
        scores = politeness_offenders(texts, target, guard, too_high)
    elif target == "length_cv":
        scores = length_contributions(texts)
    else:
        # No cheap per-comment contribution worth its model call at selection
        # time, so fall back to the longest comments, which carry the most
        # classifier signal.
        scores = [float(len(t.split())) for t in texts]
    return sorted(range(len(texts)), key=lambda i: -scores[i])


def budget(n: int, share: float, *, floor: int = 1, ceiling: int = 12) -> int:
    return max(floor, min(ceiling, int(round(n * share))))


# ------------------------------------------------------------------ evidence
def evidence(texts: Sequence[str], index: int, target: str, *, position: int,
             cache: Any = None, guard: Any = None,
             wants: dict[str, float] | None = None) -> str:
    """What the model is told about WHY this comment was picked.

    Every line is measured on this thread. It replaces two blocks of the old
    prompt: "semantic_mean_cosine = 0.2277, a real thread sits at 0.1792",
    which is a name and two floats the model cannot act on, and a dump of eight
    neighbours truncated at 220 characters, which need not even contain the
    comment this one is actually duplicating.
    """
    if target == "similarity" or target in S.SIMILARITY:
        return _redundancy_evidence(texts, index, position, cache)
    return _register_evidence(texts, index, guard, wants or {})


def _redundancy_evidence(texts: Sequence[str], index: int, position: int,
                         cache: Any) -> str:
    vectors = getattr(cache, "vectors", None)
    if vectors is None:
        vectors = C.embed(list(texts))
    sims = vectors @ vectors[index]
    # Only neighbours it is genuinely close to. The thread's own mean pair
    # cosine is the reference, so the bar comes from the data rather than a
    # constant. Without it a short thread listed its LEAST similar comment
    # under "these are the ones it is closest to" -- telling the model to avoid
    # a point it was nowhere near.
    floor = C.semantic_mean_cosine(vectors)
    nearest = [i for i in sorted((j for j in range(len(texts)) if j != index),
                                 key=lambda j: -float(sims[j]))[:2]
               if float(sims[i]) > floor]
    lines = [f"WHY THIS ONE: of the thread's {len(texts)} comments, it is the "
             f"#{position + 1} biggest contributor to how much they repeat each other."]
    if nearest:
        lines.append("These are the ones it is closest to. It must not end up "
                     "making either of their points, or sounding like them:")
        lines += [f"  [{float(sims[i]):.2f} similar] {texts[i]}" for i in nearest]
    shared = _repeated_runs(texts, index, getattr(cache, "tokens", None))
    if shared:
        lines.append("Word runs it reuses from elsewhere in the thread, which have "
                     "to go: " + " · ".join(f'"{run}"' for run in shared))
    return "\n".join(lines)


def _repeated_runs(texts: Sequence[str], index: int, tokens: Any = None,
                   size: int = 4, limit: int = 6) -> list[str]:
    """The word runs this comment shares with any other comment in the thread.

    Four is the order `self_bleu_4` is scored at, so these are the exact spans
    the metric is charging for.
    """
    import score_thread_self_bleu as sb

    if tokens is None:
        tokens = [sb.tokenize(t) for t in texts]
    mine = tokens[index]
    elsewhere: set[tuple[str, ...]] = set()
    for j, other in enumerate(tokens):
        if j == index:
            continue
        elsewhere.update(tuple(other[k:k + size]) for k in range(len(other) - size + 1))
    out: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for k in range(len(mine) - size + 1):
        gram = tuple(mine[k:k + size])
        if gram in elsewhere and gram not in seen:
            seen.add(gram)
            out.append(" ".join(gram))
        if len(out) >= limit:
            break
    return out


def _register_evidence(texts: Sequence[str], index: int, guard: Any,
                       wants: dict[str, float]) -> str:
    around = "\n".join(f"  - {t[:200]}" for i, t in enumerate(texts[:6]) if i != index)
    if guard is None:
        return "THE THREAD AROUND IT:\n" + around
    have = guard.values()
    feeling = guard.emotion[index] or "none"
    # Whatever this round is targeting, not a fixed list: a `length_cv` round
    # printed an empty "a real thread :" line when the names were hardcoded.
    label = {"mean_story_probability": "narrative", "emotion_entropy": "spread of feelings"}
    real = ", ".join(f"{label.get(key, key.replace('_rate', ''))} {wants[key]:.2f}"
                     for key in wants)
    return (
        "WHAT THE CLASSIFIERS SEE:\n"
        f"  this comment  : {guard.polite[index]}, feeling \"{feeling}\", "
        f"narrative {guard.story[index]:.2f}\n"
        f"  this thread   : {have['polite_rate']:.0%} polite / "
        f"{have['impolite_rate']:.0%} impolite / {have['neutral_rate']:.0%} neutral, "
        f"narrative {have['mean_story_probability']:.2f}, "
        f"spread of feelings {have['emotion_entropy']:.2f}\n"
        f"  a real thread : {real}\n"
        "THE THREAD AROUND IT:\n" + around)
