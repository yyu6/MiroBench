"""Generate the final Reddit personas used by the simulator.

This module converts high-level archetypes from `analyzer.py` into concrete
Reddit-style user profiles. It is the second LLM stage in the pipeline.

Design goals:
- keep personas aligned with archetype weights
- make personas feel like messy subreddit users rather than polished bios
- keep OASIS-required fields present and correctly indexed
- batch large persona requests so API requests stay reliable
"""
from __future__ import annotations

import json
import random
from datetime import date
from math import ceil
from typing import Any

from openai import OpenAI

from .analyzer import PersonaArchetype, ProductAnalysis
from .loader import NormalizedProduct
from .llm_utils import create_json_object_completion


def generate_personas(
    analysis: ProductAnalysis,
    n_agents: int,
    products: list[NormalizedProduct],
    client: OpenAI,
    model: str,
    seed: int,
    overlay: dict | None = None,
) -> tuple[list[dict], str, str]:
    """Generate `n_agents` personas and return `(profiles, prompt, raw)`.

    The function may issue multiple LLM calls when the requested population is
    large. Batched prompts/raw responses are concatenated with `---BATCH---` so
    the caller can still save one audit trail per run.
    """
    distribution = _distribute_agents(analysis.persona_archetypes, n_agents)
    product_summary = _sanitize_text(_build_product_summary(products))

    batch_distributions = _split_distribution_into_batches(
        distribution,
        max_batch_size=10,
    )

    prompts: list[str] = []
    raw_chunks: list[str] = []
    profiles: list[dict] = []

    for batch_distribution in batch_distributions:
        batch_profiles, batch_prompt, batch_raw = _generate_persona_batch(
            analysis=analysis,
            distribution=batch_distribution,
            product_summary=product_summary,
            client=client,
            model=model,
            overlay=overlay,
        )
        prompts.append(batch_prompt)
        raw_chunks.append(batch_raw)
        profiles.extend(batch_profiles)

    if len(profiles) != n_agents:
        raise ValueError(
            f"Expected {n_agents} personas across batches, got {len(profiles)}"
        )

    _ensure_runtime_persona_diversity(
        profiles,
        analysis=analysis,
        seed=seed,
    )
    _ensure_product_opinion_coverage(
        profiles,
        products=products,
        seed=seed,
    )

    # OASIS's Reddit agent graph is indexed from 0, so keep persona IDs aligned
    # with downstream runtime scheduling and export code.
    for i, p in enumerate(profiles):
        p["user_id"] = i
        # Ensure created_at field exists (OASIS requires it)
        if "created_at" not in p:
            p["created_at"] = str(date.today())

    return profiles, "\n\n---BATCH---\n\n".join(prompts), "\n\n---BATCH---\n\n".join(raw_chunks)


def _generate_persona_batch(
    analysis: ProductAnalysis,
    distribution: dict[str, int],
    product_summary: str,
    client: OpenAI,
    model: str,
    overlay: dict | None = None,
    max_attempts: int = 3,
) -> tuple[list[dict], str, str]:
    """Generate one persona batch and recover missing rows with micro-batches."""

    expected_total = sum(distribution.values())
    personas, prompt_log, raw_log, exact = _attempt_persona_batch_generation(
        analysis=analysis,
        distribution=distribution,
        product_summary=product_summary,
        client=client,
        model=model,
        overlay=overlay,
        max_attempts=max_attempts,
    )
    if exact:
        return personas, prompt_log, raw_log

    if 0 < len(personas) < expected_total:
        combined, supplemental_prompt, supplemental_raw = _fill_persona_batch_with_micro_batches(
            analysis=analysis,
            target_distribution=distribution,
            initial_personas=personas,
            product_summary=product_summary,
            client=client,
            model=model,
            overlay=overlay,
            max_attempts=max_attempts,
        )
        if len(combined) == expected_total:
            prompt_parts = [part for part in [prompt_log, supplemental_prompt] if part]
            raw_parts = [part for part in [raw_log, supplemental_raw] if part]
            return (
                combined,
                "\n\n---RETRY---\n\n".join(prompt_parts),
                "\n\n---RETRY---\n\n".join(raw_parts),
            )

    raise ValueError(
        f"Expected {expected_total} personas in batch after fallback recovery, "
        f"got {len(personas)}"
    )


def _attempt_persona_batch_generation(
    analysis: ProductAnalysis,
    distribution: dict[str, int],
    product_summary: str,
    client: OpenAI,
    model: str,
    overlay: dict | None,
    max_attempts: int,
) -> tuple[list[dict], str, str, bool]:
    """Attempt one logical persona batch and return the best partial result."""

    expected_total = sum(distribution.values())
    corrective_feedback = ""
    prompt_attempts: list[str] = []
    raw_attempts: list[str] = []
    best_personas: list[dict] = []

    for _attempt in range(1, max_attempts + 1):
        prompt = _sanitize_text(
            _build_prompt(
                analysis,
                distribution,
                product_summary,
                overlay=overlay,
                corrective_feedback=corrective_feedback,
            )
        )
        prompt_attempts.append(prompt)

        raw = create_json_object_completion(
            client=client,
            model=model,
            prompt=prompt,
            temperature=0.9,
        )
        raw_attempts.append(raw)
        data = json.loads(raw)
        if "personas" not in data:
            raise ValueError(f"LLM response missing 'personas' key. Raw: {raw[:200]}")

        personas = data["personas"]
        if not isinstance(personas, list):
            raise ValueError(f"LLM response 'personas' must be a list. Raw: {raw[:200]}")

        count = len(personas)
        if count == expected_total:
            return (
                personas,
                "\n\n---RETRY---\n\n".join(prompt_attempts),
                "\n\n---RETRY---\n\n".join(raw_attempts),
                True,
            )
        if count < expected_total and count > len(best_personas):
            best_personas = personas

        corrective_feedback = (
            f"COUNT CORRECTION: your previous response returned {count} personas, "
            f"but this batch requires exactly {expected_total}. Regenerate the full JSON "
            f"array from scratch with exactly {expected_total} persona objects. "
            "Do not omit any item. Do not return more than requested."
        )

    return (
        best_personas,
        "\n\n---RETRY---\n\n".join(prompt_attempts),
        "\n\n---RETRY---\n\n".join(raw_attempts),
        False,
    )


def _fill_persona_batch_with_micro_batches(
    analysis: ProductAnalysis,
    target_distribution: dict[str, int],
    initial_personas: list[dict],
    product_summary: str,
    client: OpenAI,
    model: str,
    overlay: dict | None = None,
    max_attempts: int = 3,
    max_batch_size: int = 3,
) -> tuple[list[dict], str, str]:
    """Fill an underfull batch by shrinking supplemental micro-batches to 1.

    The LLM sometimes returns 8/10 or 11/12 personas for a large JSON batch.
    Rather than abandon the whole simulation, we keep the valid partial result,
    infer how many personas are still missing, and then request only those
    missing personas in progressively smaller batches: 3 -> 2 -> 1.
    """

    expected_total = sum(target_distribution.values())
    combined = list(initial_personas)
    prompt_logs: list[str] = []
    raw_logs: list[str] = []
    current_batch_size = max(1, min(max_batch_size, expected_total - len(combined)))
    while current_batch_size >= 1:
        missing_distribution = _infer_missing_distribution(
            target_distribution=target_distribution,
            personas=combined,
        )
        missing_total = sum(missing_distribution.values())
        if missing_total <= 0 or len(combined) >= expected_total:
            return (
                combined[:expected_total],
                "\n\n---MICRO-BATCH---\n\n".join(prompt_logs),
                "\n\n---MICRO-BATCH---\n\n".join(raw_logs),
            )

        current_batch_size = min(current_batch_size, missing_total)
        previous_count = len(combined)
        micro_batches = _split_distribution_into_batches(
            missing_distribution,
            max_batch_size=current_batch_size,
        )

        for micro_distribution in micro_batches:
            profiles, prompt, raw, _exact = _attempt_persona_batch_generation(
                analysis=analysis,
                distribution=micro_distribution,
                product_summary=product_summary,
                client=client,
                model=model,
                overlay=overlay,
                max_attempts=max_attempts,
            )
            if prompt:
                prompt_logs.append(prompt)
            if raw:
                raw_logs.append(raw)
            if not profiles:
                continue
            remaining_slots = expected_total - len(combined)
            if remaining_slots <= 0:
                break
            combined.extend(profiles[:remaining_slots])
            if len(combined) >= expected_total:
                break

        if len(combined) >= expected_total:
            return (
                combined[:expected_total],
                "\n\n---MICRO-BATCH---\n\n".join(prompt_logs),
                "\n\n---MICRO-BATCH---\n\n".join(raw_logs),
            )
        if len(combined) == previous_count:
            current_batch_size -= 1

    return (
        combined[:expected_total],
        "\n\n---MICRO-BATCH---\n\n".join(prompt_logs),
        "\n\n---MICRO-BATCH---\n\n".join(raw_logs),
    )


def _infer_missing_distribution(
    target_distribution: dict[str, int],
    personas: list[dict],
) -> dict[str, int]:
    """Infer which archetype counts are still missing from an underfull batch."""

    observed = {name: 0 for name in target_distribution}
    for persona in personas:
        archetype = str(persona.get("archetype") or "").strip()
        if archetype in observed:
            observed[archetype] += 1

    missing: dict[str, int] = {}
    for archetype, target_count in target_distribution.items():
        deficit = max(0, int(target_count) - int(observed.get(archetype, 0)))
        if deficit > 0:
            missing[archetype] = deficit
    return missing


def _distribute_agents(
    archetypes: list[PersonaArchetype], n: int
) -> dict[str, int]:
    """Distribute agents across archetypes while preserving diversity.

    The allocation is roughly proportional to archetype weights, but every
    archetype still gets at least one persona so the final discussion does not
    collapse into only the dominant types.
    """
    if n < len(archetypes):
        raise ValueError(
            f"n_agents ({n}) must be >= number of archetypes ({len(archetypes)})"
        )
    distribution: dict[str, int] = {}
    remaining = n
    for arch in archetypes[:-1]:
        count = max(1, round(arch.weight * n))
        count = min(count, remaining - (len(archetypes) - len(distribution) - 1))
        distribution[arch.name] = count
        remaining -= count
    distribution[archetypes[-1].name] = max(1, remaining)
    return distribution


def _build_product_summary(products: list[NormalizedProduct]) -> str:
    """Build a compact product roster for persona-generation prompts.

    Personas need product-specific hooks so later comments can contain distinct
    lived opinions. The roster uses stable product ids and short facts instead
    of long descriptions so the LLM can assign each persona a small opinion
    portfolio without blowing up the prompt.
    """

    lines = []
    for idx, p in enumerate(products[:60], start=1):
        price_str = f"${p.price}" if p.price is not None else "price unknown"
        rating_str = f"{p.rating}/5" if p.rating is not None else "unrated"
        feature_text = "; ".join(p.features[:2]) if p.features else ""
        feature_suffix = f" | {feature_text}" if feature_text else ""
        lines.append(
            f"- P{idx:03d}: {p.title} ({p.brand or 'brand unknown'}, {price_str}, {rating_str}){feature_suffix}"
        )
    return "\n".join(lines) if lines else "No products provided."


def _split_distribution_into_batches(
    distribution: dict[str, int],
    max_batch_size: int,
) -> list[dict[str, int]]:
    """Split one persona-allocation plan into API-sized batches."""

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")

    remaining = dict(distribution)
    total = sum(remaining.values())
    batch_count = max(1, ceil(total / max_batch_size))
    batches: list[dict[str, int]] = []

    for _ in range(batch_count):
        current: dict[str, int] = {}
        current_total = 0
        for archetype, count in list(remaining.items()):
            if count <= 0 or current_total >= max_batch_size:
                continue
            take = min(count, max_batch_size - current_total)
            if take <= 0:
                continue
            current[archetype] = take
            remaining[archetype] -= take
            current_total += take
        if current:
            batches.append(current)

    leftover = {k: v for k, v in remaining.items() if v > 0}
    if leftover:
        raise ValueError(f"Failed to allocate all personas into batches: {leftover}")

    return batches


def _build_prompt(
    analysis: ProductAnalysis,
    distribution: dict[str, int],
    product_summary: str,
    overlay: dict | None = None,
    corrective_feedback: str = "",
) -> str:
    """Build one batch prompt for persona generation."""

    total = sum(distribution.values())
    dist_lines = "\n".join(
        f"  - {name}: {count} personas" for name, count in distribution.items()
    )
    archetype_details = "\n".join(
        f"  [{a.name}]: {a.description}"
        for a in analysis.persona_archetypes
    )
    themes = ", ".join(analysis.key_themes[:5])

    correction_block = (
        f"\nIMPORTANT RETRY INSTRUCTION:\n{corrective_feedback}\n"
        if corrective_feedback
        else ""
    )
    calibration_guidance = str((overlay or {}).get("persona.generation_guidance", "") or "").strip()
    calibration_guidance_block = (
        "\nCALIBRATION PERSONA GUIDANCE:\n"
        "Treat the following as the actual calibration patch for this batch. "
        "Bake it concretely into the cast of people you generate, not just into prettier wording. "
        "Translate it into visible consequences in: the persona notes, motivations, blind spots, "
        "product opinions, likely_comment_angle, confidence level, repeated grievances, and what kinds "
        "of threads each user gets pulled into.\n\n"
        f"{calibration_guidance}\n\n"
        "When relevant, make the guidance visible through: what triggers the user to reply, "
        "what they keep repeating from personal experience, how certain or annoying they sound, "
        "what products they defend or regret, what narrow angle they keep bringing up, and what kinds "
        "of concrete anecdotes they would naturally use. Do not simply restate the guidance in generic terms; "
        "turn it into distinct, castable people with uneven knowledge and recurring biases.\n"
        if calibration_guidance
        else ""
    )

    return f"""You are creating {total} realistic Reddit user personas for an academic NeurIPS simulation of authentic online discussions about {analysis.product_category}.{correction_block}

PERSONA DISTRIBUTION (you must generate exactly these counts):
{dist_lines}

ARCHETYPE DESCRIPTIONS:
{archetype_details}

KEY DISCUSSION THEMES: {themes}

PRODUCTS POOL (agents should be aware of and able to reference these):
{product_summary}
{calibration_guidance_block}

Generate exactly {total} personas. Return JSON with this structure:
{{
  "personas": [
    {{
      "user_id": 0,
      "username": "unique_reddit_style_username",
      "name": "Full Name",
      "bio": "Very short Reddit bio or tagline, casual and internet-native",
      "persona": "Dense internal character notes covering real-life context, product relationship, writing habits on Reddit, recurring opinions, blind spots, and how they behave in threads",
      "karma": 4500,
      "age": 29,
      "gender": "male",
      "mbti": "INTJ",
      "country": "USA",
      "profession": "Software Engineer",
      "interested_topics": ["topic1", "topic2"],
      "primary_motivation": "correcting people",
      "knowledge_style": "overconfident_half_expert",
      "conflict_style": "skeptical",
      "product_opinions": [
        {{
          "product_id": "P003",
          "product_title": "Exact product title from PRODUCTS POOL",
          "relationship": "owns",
          "personal_angle": "One concrete personal experience, bias, annoyance, regret, or reason they care about this product",
          "likely_comment_angle": "How this opinion would show up in a Reddit reply"
        }}
      ],
      "archetype": "Audiophile"
    }}
  ]
}}

RULES:
- Each persona must be MEANINGFULLY DIFFERENT even within the same archetype
- Treat this like casting real subreddit users, not writing polished brand-safe bios
- Include lurkers, habitual commenters, skeptics, nitpickers, enthusiasts, normal people with limited knowledge, and a few mildly annoying know-it-alls
- Usernames: realistic Reddit-style, no spaces, and not all cute/alliterative by default
- Karma range: casual 200–2000 | regular 2000–15000 | power user 15000–50000
- `bio` should be short; fragments are fine
- `persona` should read like internal casting notes, not a self-introduction or polished essay
- In `persona`, include: day job/life context, what they own or want, what they care about, what kinds of threads trigger them, how they write online, and what bad takes or biases they repeatedly show
- Add `primary_motivation` showing why they usually reply on Reddit. Good values include: helping, venting, showing expertise, correcting people, defending their own setup, bargain-hunting, complaining, validation-seeking, or joking around
- Add `knowledge_style` showing what kind of knowledge they usually bring. Good values include: beginner, casual_user, experienced_owner, specialist, or overconfident_half_expert
- Add `conflict_style` showing how they disagree. Good values include: calm, skeptical, blunt, sarcastic, argumentative, or avoidant
- Add `product_opinions`: exactly 2-4 product-specific opinions for every persona, using products from PRODUCTS POOL
- Each product opinion must use a real `product_id` and exact `product_title` from PRODUCTS POOL
- `relationship` should vary across personas: owns, used_to_own, tried, applied_for, rejected, considering, avoided, downgraded_from, recommended_to_friend, regret_buying, wants_but_cannot_justify
- `personal_angle` must be concrete and personal: a specific fee pain point, approval/return/retention outcome, everyday use case, travel habit, budget constraint, annoyance, regret, or loyalty bias
- `likely_comment_angle` should describe how this persona would naturally bring that product into a Reddit comment without sounding like a review blurb
- Across the batch, cover many different products. Do not let everyone have opinions about the same 2-3 obvious products
- Opinions may contradict each other. One persona can love a product that another persona hates for personal reasons
- Include skeptics, enthusiasts, beginners, experts, and people who only know enough to be dangerous
- Personas should reference specific products from the pool when relevant to their archetype and product_opinions
- Make personas feel like real humans — give them habits, pet peeves, quirks, contradictions, and uneven knowledge
- Do not make everyone helpful, articulate, upbeat, or balanced
- Across the batch, vary motivations and certainty levels on purpose. Some personas should sound tentative, some should sound overconfident, some should only reply when annoyed, and some should mostly chime in to validate or complain
- Across the batch, do not let everyone have the same conflict style. Mix calm practical people with skeptics, nitpickers, mildly rude users, sarcastic users, and a few people who are conflict-averse
- Avoid assistant-style phrasing, motivational language, and generic "I love helping others" filler"""


def _sanitize_text(text: str) -> str:
    """Strip bad control characters from prompt text before API calls."""

    sanitized_chars: list[str] = []
    for ch in str(text):
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


def _ensure_runtime_persona_diversity(
    profiles: list[dict],
    analysis: ProductAnalysis,
    seed: int,
) -> None:
    """Backfill persona-style runtime fields when the LLM leaves them vague.

    Judge feedback repeatedly flagged that commenters sounded too similar in
    motivation, certainty, and disagreement style. The persona prompt now asks
    for explicit fields, but this helper ensures the runtime still gets a
    diverse mix even if the model omits some of them.
    """

    rng = random.Random(seed + 17)
    archetype_names = [a.name for a in analysis.persona_archetypes]
    archetype_defaults = {
        name: _default_trait_bundle(name)
        for name in archetype_names
    }

    for profile in profiles:
        archetype = str(profile.get("archetype", "") or "")
        defaults = archetype_defaults.get(archetype, _default_trait_bundle(archetype))
        karma = int(profile.get("karma", 0) or 0)

        profile.setdefault(
            "primary_motivation",
            rng.choice(defaults["primary_motivation"]),
        )
        profile.setdefault(
            "knowledge_style",
            _default_knowledge_style(karma=karma, choices=defaults["knowledge_style"], rng=rng),
        )
        profile.setdefault(
            "conflict_style",
            rng.choice(defaults["conflict_style"]),
        )


def _ensure_product_opinion_coverage(
    profiles: list[dict],
    products: list[NormalizedProduct],
    seed: int,
) -> None:
    """Ensure every persona has concrete opinions about selected products.

    The LLM prompt asks for these directly, but this fallback is important for
    reliability and older test fixtures. Each persona receives a small,
    deterministic portfolio of product relationships so runtime prompts can
    surface actual product-specific reasons instead of asking agents to invent
    generic anecdotes on the fly.
    """

    if not products:
        for profile in profiles:
            profile.setdefault("product_opinions", [])
        return

    rng = random.Random(seed + 31)
    product_pool = list(products[:60])
    product_lookup = {_normalize_product_title(product.title): idx for idx, product in enumerate(product_pool)}

    for idx, profile in enumerate(profiles):
        existing = _clean_product_opinions(
            profile.get("product_opinions"),
            product_lookup=product_lookup,
            products=product_pool,
        )
        if len(existing) >= 2:
            profile["product_opinions"] = existing[:4]
            continue

        needed = rng.randint(2, 4) - len(existing)
        assigned_products = {opinion["product_title"] for opinion in existing}
        candidates = _candidate_products_for_profile(
            profile_index=idx,
            products=product_pool,
            rng=rng,
        )
        for product_id, product in candidates:
            if needed <= 0:
                break
            if product.title in assigned_products:
                continue
            existing.append(
                _build_default_product_opinion(
                    profile=profile,
                    product_id=product_id,
                    product=product,
                    rng=rng,
                )
            )
            assigned_products.add(product.title)
            needed -= 1
        profile["product_opinions"] = existing[:4]


def _clean_product_opinions(
    raw_opinions: Any,
    product_lookup: dict[str, int],
    products: list[NormalizedProduct],
) -> list[dict[str, str]]:
    """Normalize LLM-provided product opinions and drop invalid entries."""

    if not isinstance(raw_opinions, list):
        return []

    cleaned: list[dict[str, str]] = []
    for raw in raw_opinions:
        if not isinstance(raw, dict):
            continue
        product_title = str(raw.get("product_title") or "").strip()
        lookup_idx = product_lookup.get(_normalize_product_title(product_title))
        if lookup_idx is None:
            continue
        product = products[lookup_idx]
        product_id = str(raw.get("product_id") or f"P{lookup_idx + 1:03d}").strip()
        relationship = str(raw.get("relationship") or "considering").strip()
        personal_angle = str(raw.get("personal_angle") or "").strip()
        likely_comment_angle = str(raw.get("likely_comment_angle") or "").strip()
        if not personal_angle:
            personal_angle = f"Has a specific bias about {product.title}."
        if not likely_comment_angle:
            likely_comment_angle = f"Would bring up {product.title} as a personal datapoint."
        cleaned.append(
            {
                "product_id": product_id,
                "product_title": product.title,
                "relationship": relationship,
                "personal_angle": personal_angle,
                "likely_comment_angle": likely_comment_angle,
            }
        )
    return cleaned


def _candidate_products_for_profile(
    profile_index: int,
    products: list[NormalizedProduct],
    rng: random.Random,
) -> list[tuple[str, NormalizedProduct]]:
    """Return a rotated candidate product list for one persona."""

    if not products:
        return []
    start = profile_index % len(products)
    indexed_products = list(enumerate(products, start=1))
    ordered = indexed_products[start:] + indexed_products[:start]
    head = ordered[:8]
    tail = ordered[8:]
    rng.shuffle(head)
    return [
        (f"P{product_index:03d}", product)
        for product_index, product in head + tail
    ]


def _build_default_product_opinion(
    profile: dict,
    product_id: str,
    product: NormalizedProduct,
    rng: random.Random,
) -> dict[str, str]:
    """Create a concrete fallback opinion for one persona/product pair."""

    motivation = str(profile.get("primary_motivation", "") or "").lower()
    knowledge_style = str(profile.get("knowledge_style", "") or "").lower()
    relationship_choices = [
        "owns",
        "tried",
        "considering",
        "rejected",
        "recommended_to_friend",
        "wants_but_cannot_justify",
    ]
    if "complain" in motivation or "vent" in motivation:
        relationship_choices.extend(["regret_buying", "avoided"])
    if "beginner" in knowledge_style:
        relationship_choices.extend(["considering", "applied_for"])
    relationship = rng.choice(relationship_choices)

    title = product.title
    price_or_fee = f"${product.price}" if product.price is not None else "the cost/fee"
    angle_templates = [
        f"{title} is tied to their budget math; {price_or_fee} is the part they keep fixating on.",
        f"They have a specific annoyance with {title} and tend to bring up that one drawback first.",
        f"They think {title} only makes sense for a narrow use case similar to their own.",
        f"They compare {title} against what they already use and are biased by that setup.",
        f"They heard enough mixed stories about {title} that they are skeptical even without perfect information.",
    ]
    comment_templates = [
        f"Would mention {title} as a personal datapoint rather than giving a broad category answer.",
        f"Would push the thread toward the one practical tradeoff they associate with {title}.",
        f"Would disagree if others treat {title} like an obvious default.",
        f"Would use {title} to explain why their own setup or regret colors their advice.",
    ]
    return {
        "product_id": product_id,
        "product_title": title,
        "relationship": relationship,
        "personal_angle": rng.choice(angle_templates),
        "likely_comment_angle": rng.choice(comment_templates),
    }


def _normalize_product_title(title: str) -> str:
    """Normalize product titles for opinion validation."""

    return " ".join(str(title or "").lower().split())


def _default_trait_bundle(archetype_name: str) -> dict[str, list[str]]:
    """Return plausible runtime trait choices for one archetype name."""

    lowered = archetype_name.lower()
    if "newbie" in lowered or "beginner" in lowered or "student" in lowered:
        return {
            "primary_motivation": ["validation-seeking", "helping", "complaining"],
            "knowledge_style": ["beginner", "casual_user", "overconfident_half_expert"],
            "conflict_style": ["avoidant", "skeptical", "calm"],
        }
    if "skeptic" in lowered or "critic" in lowered:
        return {
            "primary_motivation": ["correcting people", "complaining", "venting"],
            "knowledge_style": ["casual_user", "experienced_owner", "overconfident_half_expert"],
            "conflict_style": ["skeptical", "blunt", "sarcastic"],
        }
    if "travel" in lowered or "rewards" in lowered or "hacker" in lowered:
        return {
            "primary_motivation": ["showing expertise", "correcting people", "defending their own setup"],
            "knowledge_style": ["experienced_owner", "specialist", "overconfident_half_expert"],
            "conflict_style": ["skeptical", "argumentative", "sarcastic"],
        }
    if "budget" in lowered or "cashback" in lowered or "frugal" in lowered:
        return {
            "primary_motivation": ["warning people", "bargain-hunting", "complaining", "helping"],
            "knowledge_style": ["casual_user", "experienced_owner", "beginner"],
            "conflict_style": ["blunt", "skeptical", "calm"],
        }
    return {
        "primary_motivation": ["helping", "correcting people", "validation-seeking", "complaining"],
        "knowledge_style": ["casual_user", "experienced_owner", "overconfident_half_expert"],
        "conflict_style": ["calm", "skeptical", "blunt", "avoidant"],
    }


def _default_knowledge_style(
    karma: int,
    choices: list[str],
    rng: random.Random,
) -> str:
    """Choose a plausible knowledge style with mild karma-based bias."""

    normalized_choices = list(choices or ["casual_user"])
    if karma <= 1500 and "beginner" in normalized_choices:
        return "beginner"
    if karma >= 12000 and "specialist" in normalized_choices:
        return "specialist"
    if 3000 <= karma <= 12000 and "experienced_owner" in normalized_choices:
        return "experienced_owner"
    return rng.choice(normalized_choices)
