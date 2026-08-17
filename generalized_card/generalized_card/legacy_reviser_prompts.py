from __future__ import annotations

from collections import Counter
import re
from typing import Any

from .domain import DomainConfig


_CARD_STATIC_DOMAIN_REPLACEMENTS = {
    "selfbleu": (
        ("card names", "product/model names"),
        (
            "new facts, cards, banks, dates, fees, percentages, URLs, or reward numbers",
            "new facts, products, brands, dates, specifications, measurements, prices, or URLs",
        ),
    ),
    "selfbert": (
        (
            "issuer, card, fee, reward, denial reason, recon/CLI/BT action, timing, or quoted premise",
            "product, model, component, specification, observed behavior, workflow action, timing, or quoted premise",
        ),
        (
            "new issuers, products, fees, dates, rewards, or numbers",
            "new brands, products, models, specifications, dates, prices, measurements, or numbers",
        ),
        ("Bank X, Card Y, issuer X", "Brand X, Product Y, model X"),
        ('"Yeah, that limit is tiny."', '"Yeah, that detail is pretty rough."'),
        (
            '"That limit is the weird part, honestly."',
            '"That behavior is the weird part, honestly."',
        ),
        (
            '\\"No downside\\" only works if the fee is actually zero.',
            '\\"No downside\\" only works if the tradeoff is actually zero.',
        ),
        (
            '"Nope, that is not treated like normal spend."',
            '"Nope, that does not behave like the standard mode."',
        ),
        (
            '"Did they say if it was age or utilization?"',
            '"Did they say if it was the setting or the component?"',
        ),
        (
            '"My recon letter said almost the same thing."',
            '"My support ticket said almost the same thing."',
        ),
        (
            '"The annoying part is the timing after the statement cuts."',
            '"The annoying part is the timing after that step finishes."',
        ),
        (
            '"This is also why those side perks get messy fast."',
            '"This is also why those extra features get messy fast."',
        ),
    ),
    "tone": (
        ("new card/bank/number", "new product/brand/quantity"),
        (
            "numbers, card names, bank names, fees, and parent relationship",
            "numbers, product/model names, brand/source names, specifications, and parent relationship",
        ),
        (
            "numbers, card names, bank names, and parent relationship",
            "numbers, product/model names, brand/source names, and parent relationship",
        ),
        (
            "same concrete cards, banks, APRs, fees, limits, or URLs",
            "same concrete products, brands, specifications, prices, limits, or URLs",
        ),
        (
            "new card names, new banks, new numbers, new URLs",
            "new product/model names, new brands, new numbers, new URLs",
        ),
        (
            "new card names, new banks, new numbers",
            "new product/model names, new brands, new numbers",
        ),
        (
            "same card/bank/APR/fee/SUB point",
            "same product/model/specification/price point",
        ),
        (
            "same card/bank/APR/SUB detail",
            "same product/model/specification detail",
        ),
        (
            "same concrete card/bank/APR/fee detail",
            "same concrete product/model/specification/price detail",
        ),
        (
            "new facts, new cards, new banks, new numbers",
            "new facts, new products, new brands, new numbers",
        ),
        (
            "new facts, new cards, new banks, new numbers, or a new story",
            "new facts, new products, new brands, new numbers, or a new story",
        ),
        (
            "new card names, new numbers, or generic customer-service tone",
            "new product/model names, new numbers, or generic customer-service tone",
        ),
        (
            "entities, numbers, cards, banks, fees, percentages, and URLs",
            "entities, numbers, products, brands, specifications, measurements, and URLs",
        ),
    ),
    "story": (
        ("r/CreditCards", "the target Reddit community"),
        ("financial point", "domain point"),
        ("concrete financial point", "concrete domain point"),
        (
            "card/issuer names, and financial facts",
            "product/model and brand names, and domain facts",
        ),
        (
            "financial numbers such as fees, rates, limits, rewards, balances, approval amounts, and issuer rules",
            "domain quantities such as prices, specifications, limits, measurements, observed outcomes, and product rules",
        ),
        ("a financial rule", "a domain rule"),
    ),
}


def adapt_card_reviser_prompt(
    config: DomainConfig,
    prompt: str,
    *,
    kind: str,
) -> str:
    """Keep CARD's prompt structure while replacing static finance wording.

    Replacements intentionally target complete phrases rather than isolated
    words.  This prevents changing dynamic discussion content such as "SD
    card" while removing the finance-only instructions embedded in CARD.
    """

    adapted = str(prompt)
    for old, new in _CARD_STATIC_DOMAIN_REPLACEMENTS.get(kind, ()):
        adapted = adapted.replace(old, new)
    preamble = (
        "Domain context for this revision: "
        f"{config.community_context}. Preserve domain-specific product/model "
        "names, technical terms, measurements, prices, dates, links, and "
        "parent-local facts.\n"
    )
    first_break = adapted.find("\n")
    if first_break < 0:
        return preamble + adapted
    return adapted[: first_break + 1] + "\n" + preamble + adapted[first_break + 1 :]


_NGRAM_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


def selfbleu_ngram_diagnostic(
    config: DomainConfig,
    *,
    comments: list[Any],
    target: Any,
) -> str:
    """Expose actionable 1/2-gram repetition omitted by CARD's phrase list."""

    target_tokens = _ngram_tokens(str(getattr(target, "content", "") or ""))
    other_tokens = [
        _ngram_tokens(str(getattr(comment, "content", "") or ""))
        for comment in comments
        if getattr(comment, "comment_id", None) != getattr(target, "comment_id", None)
    ]
    unigram_counts: Counter[tuple[str, ...]] = Counter()
    bigram_counts: Counter[tuple[str, ...]] = Counter()
    for tokens in other_tokens:
        unigram_counts.update((token,) for token in tokens)
        bigram_counts.update(zip(tokens, tokens[1:]))

    protected = {
        token
        for phrase in (*config.technical_terms, *config.protected_entity_terms)
        for token in _ngram_tokens(phrase)
    }
    repeated_unigrams = Counter()
    for token in target_tokens:
        if (
            unigram_counts[(token,)] > 0
            and token not in protected
            and token not in _NGRAM_STOP_WORDS
            and len(token) > 2
        ):
            repeated_unigrams[token] += unigram_counts[(token,)]
    repeated_bigrams = Counter()
    for pair in zip(target_tokens, target_tokens[1:]):
        if bigram_counts[pair] <= 0:
            continue
        if all(token in _NGRAM_STOP_WORDS for token in pair):
            continue
        repeated_bigrams[" ".join(pair)] += bigram_counts[pair]

    unigrams = (
        ", ".join(item for item, _ in repeated_unigrams.most_common(8)) or "(none)"
    )
    bigrams = ", ".join(item for item, _ in repeated_bigrams.most_common(8)) or "(none)"
    return f"""Cross-domain n-gram diagnostic:
- The exact Self-BLEU evaluator aggregates 1-, 2-, 3-, and 4-gram overlap after each candidate is inserted into the full thread.
- CARD's thread-local phrase list above identifies actionable repeated 3/4-grams.
- Repeated non-anchor unigrams in this target: {unigrams}.
- Repeated bigrams in this target: {bigrams}.
- Change repeated function wording and sentence paths first. Preserve technical anchors even when they repeat; the exact metric gate decides whether the full rewrite helps.
"""


def insert_reviser_guidance(prompt: str, guidance: str) -> str:
    marker = "Return strict JSON only:"
    if marker in prompt:
        return prompt.replace(marker, guidance.rstrip() + "\n\n" + marker, 1)
    return prompt.rstrip() + "\n\n" + guidance


def _ngram_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())
