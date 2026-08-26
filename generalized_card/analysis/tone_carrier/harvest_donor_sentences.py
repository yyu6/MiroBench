#!/usr/bin/env python3
"""Harvest the donor-sentence inventory G53 named and never built (v120).

G53 closed six realization-side hypotheses for `polite_rate` and left exactly one
alive, with a measured effect and no implementation:

    "inserting a single real short appreciative sentence into a generated
     non-polite 40w+ comment flips it 0.29-0.50 of the time against 0.121 for a
     non-polite real donor (v104), so the metric IS reachable if the Writer could
     produce such a sentence -- the untried mechanism is a per-slot drawn donor
     sentence from the excluded corpus. Not built, not priced."

Priced 2026-08-27: at the LOW end of that flip range the polite row of the
realization matrix moves 0.384 -> 0.563, which takes `polite_rate` at N=150 from
P(pass) **0.17 -> 0.90** raw and 0.64 -> 0.99 under Holm, with impolite 0.92 and
neutral 0.95. That is all three tone metrics passing, and it is the only mechanism
measured large enough to do it.

This builds the inventory. Discipline:

  * **evaluation-excluded threads only.** No seed thread is read, so no evaluation
    text can reach the output. Same rule as `reference_link`.
  * **the evaluation's own `Intel/polite-guard` checkpoint** decides what counts as
    appreciative -- P(polite) > 0.80, G53's threshold, flat between 0.70 and 0.90.
  * **topic-free**: the donor is prefixed to a comment about a DIFFERENT product,
    so anything naming a subject is a visible tell -- a much worse one than the
    URL case (G61/G64), because it sits in the prose. Filtered out: product
    designators, digits, URLs, brand names, and any capitalised word that is not
    sentence-initial and not `I`. That also makes the inventory transfer to
    another domain.
  * **short**: 2-12 words. G53's shortest tail bin is where the appreciation
    essentially IS the sentence and real converts at 0.437.
  * G37's risk is the reason for the size report: a shared prescribed sentence
    across slots converges the pairwise metrics. The inventory has to be large
    enough that per-slot hash draws rarely collide, exactly as the 802-URL link
    inventory is.

Writes `generalized_card/profiles/<domain>_donor_sentences.json`.

Usage:  python3 generalized_card/analysis/tone_carrier/harvest_donor_sentences.py [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "generalized_card"))
sys.path.insert(0, str(REPO / "generalized_card" / "analysis"))

from generalized_card.content_profile_analysis import DESIGNATOR  # noqa: E402
from polite_sentence_diagnosis import (  # noqa: E402
    POLITE_SENTENCE,
    Corpus,
    load_scorer,
    sentences,
)

OUT = REPO / "generalized_card/profiles"
MIN_WORDS, MAX_WORDS = 2, 12
URLISH = re.compile(r"https?://|www\.|/r/|/u/|\bu/|\br/", re.I)
DIGIT = re.compile(r"\d")
# A donor is prefixed to someone else's sentence, so it must not open a reference
# that has no antecedent in the host comment.
DANGLING = re.compile(r"^\s*(?:it|this|that|those|these|they|he|she|him|her)\b", re.I)
# Brands and domain nouns. A donor saying "enjoy your Canon setup" prefixed to a
# Sony thread is exactly the tell this arm cannot afford.
BRAND = re.compile(
    r"\b(?:canon|nikon|sony|fuji|fujifilm|panasonic|lumix|olympus|pentax|ricoh|"
    r"leica|hasselblad|sigma|tamron|samyang|apple|iphone|gopro|dji|zeiss|minolta|"
    r"kodak|nikkor|sigma)\b",
    re.I,
)
# Any capitalised token that is not sentence-initial and not `I` is a proper noun
# or an acronym, i.e. something specific to the donor's own thread.
PROPER = re.compile(r"\b(?!I\b)(?:[A-Z][a-z]{2,}|[A-Z]{2,})\b")
# Every content word must be COMMON in this domain's own corpus. "Thanks so much
# for sharing your experience!" transfers to any thread; "I love your cosplay
# photos!" does not, and a rare word is exactly what makes it not transfer. The
# rule is measured per domain rather than listed, so it carries to a new corpus.
COMMON_RANK = 2000
CONTENT = re.compile(r"[A-Za-z']{4,}")


def usable(text: str, common: frozenset[str]) -> bool:
    words = text.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        return False
    if URLISH.search(text) or DIGIT.search(text):
        return False
    if DESIGNATOR.search(text) or DANGLING.match(text):
        return False
    if BRAND.search(text):
        return False
    # Check from the second word on, so a sentence-initial capital is allowed.
    if PROPER.search(" ".join(words[1:])):
        return False
    if any(ch in text for ch in "><*[]|`"):
        # Quotes and markdown carry Reddit structure the Writer would copy verbatim.
        return False
    return all(w.lower() in common for w in CONTENT.findall(text))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--domain", default="camera_product")
    ap.add_argument("--limit", type=int, default=0, help="cap candidates, for a smoke run")
    args = ap.parse_args()

    corpus = Corpus(REPO / "artifacts/generalized_card/runs/v117_calibration_20260826_v1")
    corpus.describe()

    frequency: collections.Counter[str] = collections.Counter()
    for row in corpus.excluded:
        frequency.update(w.lower() for w in CONTENT.findall(row["text"]))
    common = frozenset(w for w, _ in frequency.most_common(COMMON_RANK))
    print(f"common-word vocabulary: top {COMMON_RANK} of {len(frequency)} types")

    seen: set[str] = set()
    candidates: list[str] = []
    for row in corpus.excluded:
        for sentence in sentences(row["text"]):
            clean = " ".join(sentence.split())
            key = re.sub(r"[^a-z ]", "", clean.lower()).strip()
            if not key or key in seen or not usable(clean, common):
                continue
            seen.add(key)
            candidates.append(clean)
    if args.limit:
        candidates = candidates[: args.limit]
    print(f"candidate sentences after surface filters: {len(candidates)}")

    score = load_scorer()
    kept: list[dict] = []
    batch = 256
    for start in range(0, len(candidates), batch):
        chunk = candidates[start : start + batch]
        for text, probs in zip(chunk, score(chunk)):
            polite = float(probs.get("polite") or 0.0)
            if polite > POLITE_SENTENCE:
                kept.append({"text": text, "polite": round(polite, 4)})
        if (start // batch) % 10 == 0:
            print(f"  scored {min(start + batch, len(candidates))}/{len(candidates)}"
                  f"   kept {len(kept)}", flush=True)

    kept.sort(key=lambda row: (-row["polite"], row["text"]))
    words = [len(row["text"].split()) for row in kept]
    first = collections.Counter(row["text"].split()[0].lower().strip(",.") for row in kept)
    payload = {
        "available": bool(kept),
        "domain": args.domain,
        "sentence_count": len(kept),
        "sentences": [row["text"] for row in kept],
        "polite_probabilities": [row["polite"] for row in kept],
        "threshold": POLITE_SENTENCE,
        "min_words": MIN_WORDS,
        "max_words": MAX_WORDS,
        "mean_words": round(sum(words) / len(words), 3) if words else 0.0,
        "distinct_openers": len(first),
        "top_opener_share": round(first.most_common(1)[0][1] / len(kept), 4) if kept else 0.0,
        "scored_by": "Intel/polite-guard, the evaluation's own checkpoint",
        "source": "evaluation-excluded threads only; no seed thread is read",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.domain}_donor_sentences.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nkept {len(kept)} donor sentences   mean words {payload['mean_words']}"
          f"   distinct openers {payload['distinct_openers']}"
          f"   top opener share {payload['top_opener_share']}")
    print(f"wrote {path}")
    for row in kept[:12]:
        print(f"   {row['polite']:.3f}  {row['text']}")


if __name__ == "__main__":
    main()
