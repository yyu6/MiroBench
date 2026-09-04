#!/usr/bin/env python3
"""Fast per-thread scoring, used ONLY to rank the reviser's candidates.

Division of labour, and it matters: this module never decides whether a round
is accepted. It answers "of these six rewrites, which one moves the target
metric most" -- a ranking question, where being a hair off costs a little search
quality and nothing else. Every accept/reject decision in the loop is made by
`metric_engine`, which runs the official scorers.

It is fast because it reuses one embedding matrix per thread: swapping comment i
for a candidate changes one row, so the thread's pair statistics are a rank-one
update rather than a rescore. The models are the same ones the official scorers
use, so the ranking is on the same scale as the gate.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "scripts" / "evaluation"
for path in (str(REPO_ROOT / "scripts"), str(EVAL_DIR)):
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

SEMANTIC_MODEL = "sentence-transformers/all-mpnet-base-v2"
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"

_EMBEDDER: Any = None
_BERT: Any = None
_POLITE: Any = None
_STORY: Any = None
_EMOTION: Any = None


def embedder() -> Any:
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(SEMANTIC_MODEL, device="cpu")
    return _EMBEDDER


def embed(texts: Sequence[str]):
    import numpy as np

    if not texts:
        return np.zeros((0, 768), dtype="float32")
    return embedder().encode(
        list(texts), normalize_embeddings=True, show_progress_bar=False,
        convert_to_numpy=True, batch_size=32,
    )


def semantic_mean_cosine(vectors) -> float:
    """Mean pairwise cosine over a thread, the quantity the metric aggregates."""
    import numpy as np

    n = len(vectors)
    if n < 2:
        return 0.0
    sim = vectors @ vectors.T
    iu = np.triu_indices(n, 1)
    return float(sim[iu].mean())


# ---------------------------------------------------------------- self-BLEU
# Imported, never reimplemented. A first draft of this file rewrote sentence
# BLEU and got three things wrong against the official scorer: add-one
# smoothing applies at EVERY order (not only above unigrams), pairs are scored
# symmetrically in both directions and averaged, and each unordered pair is
# counted once rather than twice. A ranking built on a subtly different BLEU
# would push the reviser toward candidates the gate then rejects.
def self_bleu_4(texts: Sequence[str]) -> float:
    import score_thread_self_bleu as sb

    tokenized = [sb.tokenize(t) for t in texts]
    return float(sb.pairwise_self_bleu_for_order(tokenized, 4))


# ------------------------------------------------------------ per-comment
def politeness_labels(texts: Sequence[str]) -> list[str]:
    """Polite-guard labels, using the official scorer's own comment record."""
    global _POLITE
    import score_thread_politeness as mod

    if _POLITE is None:
        _POLITE = mod.PolitenessScorer(mod.DEFAULT_MODEL, "cpu", 512)
    comments = [
        mod.ThreadComment(thread_id="t", thread_title="", comment_id=str(i),
                          parent_id="", author="", text=text, depth=0)
        for i, text in enumerate(texts)
    ]
    rows = _POLITE.score_comments(comments, batch_size=16, include_text=False)
    return [str(row["pred_label"]) for row in rows]


def shared_bert_scorer():
    """Reuse the BERTScorer `metric_engine` already loaded.

    deberta-xlarge-mnli is 2.6 GB resident. A second copy took the process from
    4.5 GB to 7.1 GB and the first full-cohort run was killed by the OS during
    round 1 with no traceback. The engine caches the official
    `load_bert_scorer`, so its instance is reachable and identical.
    """
    import metric_engine as engine

    for key, value in engine._MODEL_CACHE.items():
        if key[1] == "load_bert_scorer":
            # load_bert_scorer returns a tuple whose first element is the scorer
            return value[0] if isinstance(value, tuple) else value
    from bert_score import BERTScorer

    return BERTScorer(model_type=BERTSCORE_MODEL, lang="en", idf=False,
                      rescale_with_baseline=False, device="cpu")


def bert_pair_f1(texts: Sequence[str], focus: int) -> float:
    """Mean BERTScore F1 of one comment against the rest of its thread.

    Only the pairs that touch `focus` are computed, which is what changes when a
    single comment is rewritten.
    """
    global _BERT
    if _BERT is None:
        _BERT = shared_bert_scorer()
    others = [t for i, t in enumerate(texts) if i != focus]
    if not others:
        return 0.0
    cands = [texts[focus]] * len(others)
    # batch_size 4, not 16: deberta-xlarge on a 96-comment thread spikes hard
    # enough at 16 to get the process killed between rounds.
    _, _, f1 = _BERT.score(cands, others, verbose=False, batch_size=4)
    return float(f1.mean())


# ---------------------------------------------------- incremental variants
# Swapping one comment changes only the rows and pairs that touch it. Embedding
# the whole thread per candidate cost 8s on a 96-comment thread, and a round
# evaluates ~70 candidates there -- ten minutes for one thread, which is what
# made the first live run unusable. These are exact, not approximations:
# `test_incremental_matches_full` asserts equality with the full recompute.
class ThreadCache:
    """Per-thread state that survives candidate evaluation."""

    def __init__(self, texts: Sequence[str]) -> None:
        import score_thread_self_bleu as sb

        self.texts = list(texts)
        self.vectors = embed(self.texts)
        self.tokens = [sb.tokenize(t) for t in self.texts]
        self._sb = sb
        n = len(self.texts)
        self.pair_count = n * (n - 1) // 2
        self.bleu_total = sum(
            sb.symmetric_pair_bleu(self.tokens[i], self.tokens[j], 4)
            for i in range(n) for j in range(i + 1, n)
        )

    def semantic_if(self, index: int, candidate: str) -> float:
        import numpy as np

        n = len(self.texts)
        if n < 2:
            return 0.0
        new_row = embed([candidate])[0]
        iu = np.triu_indices(n, 1)
        total = float((self.vectors @ self.vectors.T)[iu].sum())
        old_row = self.vectors[index]
        # The pairs that touch `index` are its dot product with every other row.
        # Subtract the self term (1.0 for a normalized row) rather than the row
        # count, then add the same sum computed against the candidate.
        old_pairs = float((self.vectors @ old_row).sum()) - float(old_row @ old_row)
        new_pairs = float((self.vectors @ new_row).sum()) - float(old_row @ new_row)
        return (total - old_pairs + new_pairs) / len(iu[0])

    def self_bleu_if(self, index: int, candidate: str) -> float:
        if self.pair_count == 0:
            return 0.0
        new_tokens = self._sb.tokenize(candidate)
        removed = sum(
            self._sb.symmetric_pair_bleu(self.tokens[index], self.tokens[j], 4)
            for j in range(len(self.texts)) if j != index
        )
        added = sum(
            self._sb.symmetric_pair_bleu(new_tokens, self.tokens[j], 4)
            for j in range(len(self.texts)) if j != index
        )
        return float((self.bleu_total - removed + added) / self.pair_count)

    def commit(self, index: int, candidate: str) -> None:
        self.texts[index] = candidate
        self.vectors[index] = embed([candidate])[0]
        self.tokens[index] = self._sb.tokenize(candidate)
        n = len(self.texts)
        self.bleu_total = sum(
            self._sb.symmetric_pair_bleu(self.tokens[i], self.tokens[j], 4)
            for i in range(n) for j in range(i + 1, n)
        )


# ------------------------------------------------- per-comment guard models
# story, emotion and politeness are means over per-comment classifier outputs,
# so swapping comment i moves the thread mean by (new_i - old_i) / n exactly.
# One forward pass per candidate makes them affordable to guard, which matters:
# the loop's first full round was rejected because a rewrite aimed at semantic
# cosine moved story and emotion, neither of which was being watched.
def story_probability(texts: Sequence[str]) -> list[float]:
    global _STORY
    import score_thread_storyseeker as mod

    if _STORY is None:
        _STORY = mod.StorySeekerScorer(mod.DEFAULT_MODEL, "cpu", 512)
    rows = _STORY.score_comments(_comments(texts), batch_size=16,
                                 threshold=0.5, include_text=False)
    return [float(r.get("story_probability", 0.0)) for r in rows]


def dominant_emotions(texts: Sequence[str]) -> list[str]:
    """The per-comment label the official entropy is computed over."""
    global _EMOTION
    import score_thread_go_emotions as mod

    if _EMOTION is None:
        _EMOTION = mod.GoEmotionsScorer(mod.DEFAULT_MODEL, "cpu", 512)
    rows = _EMOTION.score_comments(_comments(texts), batch_size=16,
                                   threshold=0.5, include_text=False)
    return [str(row["dominant_emotion"]) for row in rows]


def _comments(texts: Sequence[str]):
    from score_thread_semantic_uniformity import ThreadComment

    return [ThreadComment(thread_id="t", thread_title="", comment_id=str(i),
                          parent_id="", author="", text=text, depth=0)
            for i, text in enumerate(texts)]


def emotion_entropy(dominant: Sequence[str]) -> float:
    """Shannon entropy over dominant-emotion counts -- the official aggregation
    (`score_thread_go_emotions.aggregate_thread`, line 314)."""
    from collections import Counter

    import score_thread_go_emotions as mod

    return float(mod.shannon_entropy(Counter(dominant).values()))


class GuardCache:
    """Per-comment guard values for one thread, updated one comment at a time.

    Every metric here is a mean or an entropy over per-comment outputs, so a
    swap costs one forward pass instead of a thread rescore. The values equal
    the official scorers' (verified to 1e-8 or exactly) so a candidate rejected
    here would have been rejected at the gate.
    """

    def __init__(self, texts: Sequence[str]) -> None:
        self.n = len(texts)
        self.story = story_probability(texts)
        self.polite = politeness_labels(texts)
        self.emotion = dominant_emotions(texts)
        self.words = [len(t.split()) for t in texts]

    def _single(self, text: str) -> tuple[float, str, str]:
        return (story_probability([text])[0], politeness_labels([text])[0],
                dominant_emotions([text])[0])

    def values(self, index: int | None = None,
               candidate: str | None = None) -> dict[str, float]:
        story, polite, emotion, words = self.story, self.polite, self.emotion, self.words
        if candidate is not None:
            s, p, e = self._single(candidate)
            story = story[:index] + [s] + story[index + 1:]
            polite = polite[:index] + [p] + polite[index + 1:]
            emotion = emotion[:index] + [e] + emotion[index + 1:]
            words = words[:index] + [len(candidate.split())] + words[index + 1:]
        n = max(1, self.n)
        return {
            "length_cv": _cv(words),
            "mean_story_probability": sum(story) / n,
            "polite_rate": sum(1 for x in polite if x == "polite") / n,
            "impolite_rate": sum(1 for x in polite if x == "impolite") / n,
            "neutral_rate": sum(1 for x in polite if x == "neutral") / n,
            "emotion_entropy": emotion_entropy(emotion),
        }

    def commit(self, index: int, candidate: str) -> None:
        s, p, e = self._single(candidate)
        self.story[index], self.polite[index], self.emotion[index] = s, p, e
        self.words[index] = len(candidate.split())


def _cv(words: Sequence[int]) -> float:
    """Length CV, from the official scorer. It sorts before computing, which a
    hand-rolled population CV does not, and `tokenize_len` is whitespace split
    rather than the regex tokenizer self-BLEU uses."""
    import score_thread_structure as st

    return float(st.compute_length_cv(sorted(words)))
