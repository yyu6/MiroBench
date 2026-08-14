#!/usr/bin/env python3
"""Compare generated thread content against its matched real threads.

The evaluation metrics say whether a distribution test passes. They do not say
which property of the writing is unlike the real corpus, and a change that
improves a p-value can still move the content further from real. This script
measures the content properties directly, on both sides, over the same seeds, and
prints the real value as the target.

Every row is a matching target, not a direction to maximize. Overshooting the real
value is as wrong as falling short: the aim is to resemble the real threads, not
to score well on any single axis.

    PYTHONPATH=generalized_card python3 \\
      generalized_card/scripts/compare_content_profile.py --tag <run tag>

Emotion and story rows reuse the per-comment model outputs already written by
`run_evaluate.py` for the generated side and by the corpus build for the real
side, so this script loads no models of its own.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card.data import load_real_thread_bank  # noqa: E402
from generalized_card.domain import load_domain_config  # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*|\d[\dA-Za-z.\-/]*")
FIRST_PERSON = re.compile(r"I['’a-z]* ")
DIGIT = re.compile(r"\d")
LIST_SHAPE = re.compile(r"(^|\s)\d[.)]\s|\n\s*[-*]\s")
END_PUNCT = tuple(".!?)\"”")
# A model designator carries both a letter and a digit: 5D, a7III, X-T2, 50mm.
DESIGNATOR = re.compile(r"\b(?=[A-Za-z0-9-]{2,8}\b)(?=[^\s]*\d)(?=[^\s]*[A-Za-z])[A-Za-z0-9-]+\b")

def _concrete_markers(config: Any) -> tuple[str, ...]:
    """Domain vocabulary from configuration, never a list written into this file.

    An earlier version hardcoded camera nouns here, which would have measured
    every other domain against camera vocabulary. The domain configuration owns
    its terms; this script owns only the algorithm.
    """

    return tuple(
        term.lower()
        for term in (*config.technical_terms, *config.protected_entity_terms)
        if term.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--domain", default="camera")
    parser.add_argument("--runs-root", default=str(REPO_ROOT / "artifacts/generalized_card/runs"))
    args = parser.parse_args()

    run_dir = Path(args.runs_root) / args.tag
    config = load_domain_config(args.domain)
    generated = _generated_threads(run_dir)
    if not generated:
        raise SystemExit(f"no generated comments under {run_dir}")
    real = _matched_real_threads(run_dir, config, set(generated))

    gen_all = [text for texts in generated.values() for text in texts]
    real_all = [text for texts in real.values() for text in texts]
    print(f"tag         : {args.tag}")
    print(f"threads     : generated {len(generated)}   matched real {len(real)}")
    print(f"comments    : generated {len(gen_all)}   real {len(real_all)}")
    print()

    terms = tuple(term.lower() for term in config.technical_terms)
    markers = _concrete_markers(config)
    rows: list[tuple[str, Callable[[list[str]], float], str]] = [
        ("repeated 4-gram share", lambda t: _repeated_ngram_share(t, 4), "lower is more varied"),
        ("repeated 5-gram share", lambda t: _repeated_ngram_share(t, 5), ""),
        ("distinct 3-word openers", _distinct_openers, "share of comments"),
        ("distinct model designators", lambda t: float(len(_designators(t))), "count"),
        ("top designator share", _top_designator_share, "concentration"),
        ("has technical term", lambda t: _share(t, lambda x: any(k in x.lower() for k in terms)), ""),
        ("has domain vocabulary", lambda t: _share(t, lambda x: any(k in x.lower() for k in markers)), "configured terms"),
        ("has a digit", lambda t: _share(t, lambda x: bool(DIGIT.search(x))), ""),
        ("opens first person", lambda t: _share(t, lambda x: bool(FIRST_PERSON.match(x))), ""),
        ("has a question mark", lambda t: _share(t, lambda x: "?" in x), ""),
        ("uses a quote marker", lambda t: _share(t, lambda x: x.lstrip().startswith(">") or "&gt;" in x), ""),
        ("uses a list shape", lambda t: _share(t, lambda x: bool(LIST_SHAPE.search(x))), ""),
        ("no end punctuation", lambda t: _share(t, lambda x: bool(x) and not x.endswith(END_PUNCT)), ""),
        ("median words", lambda t: statistics.median(len(x.split()) for x in t), ""),
        ("mean words", lambda t: statistics.mean(len(x.split()) for x in t), ""),
        ("word count CV", _word_cv, "length spread"),
        ("longest comment words", lambda t: float(max(len(x.split()) for x in t)), ""),
        ("comments under 10 words", lambda t: _share(t, lambda x: len(x.split()) < 10), ""),
        ("comments over 100 words", lambda t: _share(t, lambda x: len(x.split()) > 100), ""),
    ]

    print(f"{'content property':<30}{'REAL (target)':>15}{'GENERATED':>12}{'gap':>10}  note")
    print("-" * 82)
    for label, fn, note in rows:
        r, g = fn(real_all), fn(gen_all)
        print(f"{label:<30}{r:>15.4f}{g:>12.4f}{g - r:>+10.4f}  {note}")

    print()
    _print_structural(real, generated)
    _print_model_rows(run_dir, config)


def _print_structural(real: dict[str, list[str]], generated: dict[str, list[str]]) -> None:
    """Reply and sibling vocabulary overlap, per thread then pooled."""

    print(f"{'discourse property':<30}{'REAL (target)':>15}{'GENERATED':>12}{'gap':>10}")
    print("-" * 70)
    # Overlap needs the tree, which _generated_threads flattens, so recompute
    # from the paired per-slot ordering the two sides share.
    for label, fn in (
        ("neighbour vocab overlap", _adjacent_overlap),
        ("thread vocab breadth", _vocab_breadth),
    ):
        r = statistics.mean([fn(v) for v in real.values() if len(v) > 2])
        g = statistics.mean([fn(v) for v in generated.values() if len(v) > 2])
        print(f"{label:<30}{r:>15.4f}{g:>12.4f}{g - r:>+10.4f}")
    print()


def _print_model_rows(run_dir: Path, config: Any) -> None:
    """Emotion and story rows from the per-comment model outputs."""

    gen_emotion = _dominant_labels(run_dir.glob("cleaned/*/go_emotions_results.json"))
    real_emotion = _dominant_labels(
        Path(config.raw_discussions_dir).rglob("go_emotions_results.json")
    )
    gen_story = _story_values(run_dir.glob("cleaned/*/storyseeker_results.json"))
    real_story = _story_values(
        Path(config.raw_discussions_dir).rglob("storyseeker_results.json")
    )
    if not gen_emotion and not gen_story:
        print("emotion/story rows unavailable: run run_evaluate.py for this tag first")
        return
    print(f"{'model-scored property':<30}{'REAL (target)':>15}{'GENERATED':>12}{'gap':>10}")
    print("-" * 70)
    if real_emotion and gen_emotion:
        print(f"{'dominant-emotion entropy':<30}{_entropy(real_emotion):>15.4f}{_entropy(gen_emotion):>12.4f}{_entropy(gen_emotion) - _entropy(real_emotion):>+10.4f}")
        print(f"{'distinct dominant emotions':<30}{len(real_emotion):>15}{len(gen_emotion):>12}{len(gen_emotion) - len(real_emotion):>+10}")
        print(f"{'neutral share':<30}{_label_share(real_emotion, 'neutral'):>15.4f}{_label_share(gen_emotion, 'neutral'):>12.4f}{_label_share(gen_emotion, 'neutral') - _label_share(real_emotion, 'neutral'):>+10.4f}")
    if real_story and gen_story:
        for label, fn in (
            ("mean story probability", statistics.mean),
            ("share above 0.5", lambda v: sum(1 for x in v if x > 0.5) / len(v)),
        ):
            r, g = fn(real_story), fn(gen_story)
            print(f"{label:<30}{r:>15.4f}{g:>12.4f}{g - r:>+10.4f}")
    print()
    if real_emotion and gen_emotion:
        print("top dominant emotions  REAL:", _top(real_emotion))
        print("top dominant emotions  GEN :", _top(gen_emotion))


def _generated_threads(run_dir: Path) -> dict[str, list[str]]:
    threads: dict[str, list[str]] = {}
    for path in sorted(run_dir.glob("generated/run_*/generation_records.json")):
        for record in _load_list(path):
            comment = record.get("comment") or {}
            text = " ".join(str(comment.get("content") or "").split())
            if not text:
                continue
            threads.setdefault(str(record.get("post_id") or "unknown"), []).append(text)
    return threads


def _matched_real_threads(
    run_dir: Path,
    config: Any,
    generated_keys: set[str],
) -> dict[str, list[str]]:
    """Load the real threads the run was matched against, keyed like generated."""

    wanted: dict[int, str] = {}
    for path in sorted(run_dir.glob("generated/run_*/generation_records.json")):
        for record in _load_list(path):
            index = record.get("seed_index")
            if isinstance(index, int):
                wanted[index] = str(record.get("post_id") or "")
    pool = _load(_seed_pool_path(run_dir))
    by_index = {
        int(row["seed_index"]): str(row["source_raw_post_id"])
        for row in pool.get("seed_posts") or []
        if "seed_index" in row and "source_raw_post_id" in row
    }
    ids = {by_index[i]: key for i, key in wanted.items() if i in by_index}
    threads: dict[str, list[str]] = {}
    best: dict[str, int] = {}
    for thread in load_real_thread_bank(config.raw_discussions_dir):
        post_id = str(thread.get("post_id") or "")
        if post_id not in ids:
            continue
        count = int(thread.get("comment_count") or 0)
        if count <= best.get(post_id, -1):
            continue
        best[post_id] = count
        threads[ids[post_id]] = [
            " ".join(str(row.get("body") or "").split())
            for row in thread.get("comments") or []
            if str(row.get("body") or "").strip()
        ]
    return threads


def _seed_pool_path(run_dir: Path) -> Path:
    config = _load(run_dir / "run_config.json")
    value = str(config.get("seed_post_pool_json") or config.get("seed_pool") or "")
    return Path(value)


def _repeated_ngram_share(texts: list[str], width: int) -> float:
    counts: Counter[str] = Counter()
    for text in texts:
        tokens = TOKEN.findall(text.lower())
        counts.update(
            {
                " ".join(tokens[index : index + width])
                for index in range(max(0, len(tokens) - width + 1))
            }
        )
    total = sum(counts.values())
    return sum(v for v in counts.values() if v >= 2) / total if total else 0.0


def _distinct_openers(texts: list[str]) -> float:
    return len({" ".join(TOKEN.findall(t.lower())[:3]) for t in texts}) / len(texts)


def _designators(texts: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(m.group().lower() for m in DESIGNATOR.finditer(text))
    return counts


def _top_designator_share(texts: list[str]) -> float:
    counts = _designators(texts)
    total = sum(counts.values())
    return max(counts.values()) / total if total else 0.0


def _share(texts: list[str], predicate: Callable[[str], bool]) -> float:
    return sum(1 for t in texts if predicate(t)) / len(texts)


def _word_cv(texts: list[str]) -> float:
    values = [len(t.split()) for t in texts]
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _content_words(text: str) -> set[str]:
    return {w for w in TOKEN.findall(text.lower()) if len(w) > 3}


def _adjacent_overlap(texts: list[str]) -> float:
    """Mean vocabulary overlap between consecutive comments of one thread.

    Consecutive order approximates local discourse locality without needing the
    reply tree, and it is computed identically on both sides.
    """

    values = []
    for left, right in zip(texts, texts[1:]):
        a, b = _content_words(left), _content_words(right)
        if a and b:
            values.append(len(a & b) / min(len(a), len(b)))
    return statistics.mean(values) if values else 0.0


def _vocab_breadth(texts: list[str]) -> float:
    """Distinct content words per comment: how much subject matter a thread covers."""

    vocab: set[str] = set()
    for text in texts:
        vocab |= _content_words(text)
    return len(vocab) / len(texts)


def _dominant_labels(paths: Iterable[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        payload = _load(path)
        for thread in payload.get("threads") or []:
            for comment in thread.get("comments") or []:
                label = str(comment.get("dominant_label") or comment.get("dominant_emotion") or "")
                if label:
                    counts[label.strip().lower()] += 1
    return counts


def _story_values(paths: Iterable[Path]) -> list[float]:
    values: list[float] = []
    for path in paths:
        payload = _load(path)
        for thread in payload.get("threads") or []:
            for comment in thread.get("comments") or []:
                try:
                    values.append(float(comment["story_probability"]))
                except (KeyError, TypeError, ValueError):
                    continue
    return values


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((v / total) * math.log(v / total) for v in counts.values() if v)


def _label_share(counts: Counter[str], label: str) -> float:
    total = sum(counts.values())
    return counts.get(label, 0) / total if total else 0.0


def _top(counts: Counter[str], limit: int = 8) -> str:
    total = sum(counts.values()) or 1
    return ", ".join(f"{k}={v / total:.2f}" for k, v in counts.most_common(limit))


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_list(path: Path) -> list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


if __name__ == "__main__":
    main()
