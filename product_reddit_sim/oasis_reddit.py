"""Patch layer that adapts OASIS/MiroFish into the current GEO Reddit runtime.

This module is where GEO intentionally diverges from vanilla OASIS Reddit
behavior without forking the upstream package. It does three main things:

1. Prompt adaptation
   Replaces weak/generic Reddit prompts with prompts that better fit normal
   subreddit behavior and the personas generated in this repo.

2. Runtime adaptation
   Exposes GEO-generated agent configs and discussion-planning hints to the
   OASIS runtime so activity level, delay, and stance actually affect behavior.

3. Platform guardrails
   Adds a few realism constraints to the simplified Reddit interface, such as:
   - no top-level self-replies
   - no repeated top-level comments from the same user on one thread

The overall goal is not to replace OASIS's architecture. It is to keep the
OASIS execution skeleton while making the resulting discussion look more like a
real Reddit thread.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Mapping

from .prompt_examples import render_generation_few_shot_block

_RUNTIME_AGENT_CONFIGS: dict[int, dict[str, Any]] = {}
_RUNTIME_CONTEXT: dict[str, Any] = {}
_RUNTIME_DB_PATH: str | None = None
_MAX_VISIBLE_ROOT_COMMENTS = 3
_MAX_VISIBLE_COMMENT_NODES = 10


def configure_reddit_runtime(
    config: Mapping[str, Any] | None = None,
    db_path: str | None = None,
) -> None:
    """Cache run-specific config so later prompt/runtime hooks can read it.

    The OASIS hooks we patch do not naturally receive the full GEO config. This
    function stores the per-run settings in module globals so patched methods
    can access:
    - agent behavior configs
    - high-level simulation context
    - the SQLite DB path for "what has this agent already done?" checks
    """
    global _RUNTIME_AGENT_CONFIGS, _RUNTIME_CONTEXT, _RUNTIME_DB_PATH

    config = dict(config or {})
    _RUNTIME_AGENT_CONFIGS = {
        int(agent_cfg["agent_id"]): dict(agent_cfg)
        for agent_cfg in config.get("agent_configs", [])
        if agent_cfg.get("agent_id") is not None
    }
    _RUNTIME_CONTEXT = {
        "simulation_requirement": config.get("simulation_requirement"),
        "hot_topics": list(
            (config.get("event_config", {}) or {}).get("hot_topics") or []
        ),
        "narrative_direction": (config.get("event_config", {}) or {}).get(
            "narrative_direction"
        ),
        "allow_new_threads_during_simulation": (config.get("event_config", {}) or {}).get(
            "allow_new_threads_during_simulation"
        ),
        "max_total_threads": (config.get("event_config", {}) or {}).get(
            "max_total_threads"
        ),
        "generation_few_shot_examples": list(
            (config.get("prompt_config", {}) or {}).get(
                "generation_few_shot_examples"
            )
            or []
        ),
        "overlay": dict(config.get("overlay") or {}),
    }
    _RUNTIME_DB_PATH = db_path


def _runtime_allows_new_threads() -> bool:
    """Return whether agents may open new top-level threads mid-simulation."""
    allow_new_threads = _RUNTIME_CONTEXT.get("allow_new_threads_during_simulation")
    if allow_new_threads is None:
        return True
    return bool(allow_new_threads)


def _runtime_max_total_threads() -> int | None:
    """Return the configured thread cap, if one exists."""
    value = _RUNTIME_CONTEXT.get("max_total_threads")
    if value in (None, ""):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _count_runtime_threads() -> int:
    """Count current top-level posts already stored in the runtime DB."""
    if not _RUNTIME_DB_PATH or not os.path.exists(_RUNTIME_DB_PATH):
        return 0
    try:
        with sqlite3.connect(_RUNTIME_DB_PATH) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM post")
            row = cursor.fetchone()
        return int(row[0] or 0) if row else 0
    except sqlite3.Error:
        return 0


def _new_thread_creation_block_reason() -> str | None:
    """Return the reason new top-level threads should be blocked, if any."""
    max_total_threads = _runtime_max_total_threads()
    current_threads = _count_runtime_threads()
    if max_total_threads is not None and current_threads >= max_total_threads:
        return (
            "This run is capped at a small fixed set of starter threads. "
            "Reply inside existing threads instead of opening a new one."
        )
    if not _runtime_allows_new_threads():
        return (
            "This run uses a fixed set of starter threads. "
            "Do not open a new thread; comment or reply within the existing ones."
        )
    return None


def create_reddit_platform(db_path: str):
    """Create the OASIS Reddit platform with GEO-specific defaults."""
    from oasis.social_platform.platform import Platform

    platform = Platform(
        db_path=db_path,
        recsys_type="reddit",
        allow_self_rating=False,
        show_score=True,
        max_rec_post_len=100,
        refresh_rec_post_count=5,
    )
    _ensure_nested_comment_schema(platform.db, platform.db_cursor)
    return platform


def resolve_reddit_agent_id(
    agent_record: Mapping[str, Any],
    fallback_index: int,
) -> int:
    """Resolve the runtime agent id while preserving persona/profile alignment."""
    if "user_id" in agent_record and agent_record["user_id"] is not None:
        return int(agent_record["user_id"])
    return int(fallback_index)


def build_reddit_system_prompt(
    user_name: str | None,
    display_name: str | None,
    bio: str | None,
    other_info: Mapping[str, Any] | None,
    overlay: dict | None = None,
) -> str:
    """Build the Reddit system prompt for one simulated user.

    This prompt combines:
    - persona identity fields
    - sampled behavior tendencies from `agent_configs`
    - run-level community context from the simulation config
    - Reddit-style constraints intended to suppress assistant-like behavior
    """

    other = dict(other_info or {})
    persona = str(other.get("user_profile", "") or "").strip()
    mbti = other.get("mbti")
    gender = other.get("gender")
    age = other.get("age")
    country = other.get("country")
    profession = other.get("profession")
    interests = other.get("interested_topics") or []
    primary_motivation = other.get("primary_motivation")
    knowledge_style = other.get("knowledge_style")
    conflict_style = other.get("conflict_style")
    product_opinions = other.get("product_opinions") or []
    if isinstance(interests, str):
        interests = [interests]

    identity_lines = []
    if display_name:
        identity_lines.append(f"- Display name: {display_name}")
    if user_name:
        identity_lines.append(f"- Username: {user_name}")
    if bio:
        identity_lines.append(f"- Bio: {bio}")
    if persona:
        identity_lines.append(f"- Persona notes: {persona}")

    demographic_bits = []
    if age is not None:
        demographic_bits.append(f"{age} years old")
    if gender:
        demographic_bits.append(str(gender))
    if country:
        demographic_bits.append(f"from {country}")
    if profession:
        demographic_bits.append(f"works as {profession}")
    if mbti:
        demographic_bits.append(f"MBTI {mbti}")
    if demographic_bits:
        identity_lines.append(f"- Background: {', '.join(demographic_bits)}")
    if interests:
        identity_lines.append(f"- Interested topics: {', '.join(map(str, interests))}")
    if primary_motivation:
        identity_lines.append(f"- Typical reason for replying: {primary_motivation}")
    if knowledge_style:
        identity_lines.append(f"- Knowledge style: {knowledge_style}")
    if conflict_style:
        identity_lines.append(f"- Conflict style: {conflict_style}")
    opinion_lines = _format_product_opinions(product_opinions)
    if opinion_lines:
        identity_lines.append("- Product-specific opinions and experiences:")
        identity_lines.extend(f"  - {line}" for line in opinion_lines)

    behavior_lines = _build_behavior_guidance(other)
    if behavior_lines:
        behavior_block = "\n".join(f"- {line}" for line in behavior_lines)
    else:
        behavior_block = "- No extra behavioral tendencies provided."

    context_lines = _build_context_guidance()
    context_block = (
        "\n".join(f"- {line}" for line in context_lines)
        if context_lines
        else "- No extra community context provided."
    )

    description_block = (
        "\n".join(identity_lines)
        if identity_lines
        else "- No extra profile details available."
    )

    _overlay = dict(overlay or {})
    tone_guidance = _overlay.get("prompt.tone_guidance") or "Real threads mix tones. Some replies are calm and practical, some are firsthand datapoints, some are dismissive or sarcastic, and some are mildly aggressive."
    anti_paraphrase = _overlay.get("prompt.anti_paraphrase_instruction") or "If several visible comments are already making the same point, your default should be to add a different lens, a different reason, or a direct challenge. Pure paraphrase should be relatively rare."

    prompt = f"""# OBJECTIVE
You're a Reddit user browsing threads and reacting like a real person with limited time, imperfect knowledge, and specific opinions.
After you inspect the feed, decide what you would actually do next.

# SELF-DESCRIPTION
Your behavior must stay consistent with this identity.
{description_block}

# BEHAVIORAL TENDENCIES
{behavior_block}

# COMMUNITY CONTEXT
{context_block}

# REDDIT STYLE
- Sound like a normal subreddit user, not a helpful assistant, salesperson, reviewer, or customer-support rep.
- Prefer concise, specific, casual writing. One grounded sentence is better than a polished paragraph.
- Most real Reddit comments push one angle, not a balanced pros-and-cons summary.
- Prefer 1-2 short sentences when commenting unless this persona is the type to ramble.
- Very short replies are normal. Sometimes the most realistic response is just "No.", "probably not", "why do you want it?", or one concrete caveat.
- {tone_guidance}
- Do not make every comment equally polite, equally hostile, or equally helpful. Vary the tone in a way that fits this persona and the thread.
- Many real replies are a little abrasive or blunt: "hard pass", "that's not worth it", "lol no", "you're overthinking this", or one specific complaint. That range is allowed when it fits.
- Real threads also mix motivations. Some people are genuinely helpful, some are flexing knowledge, some are venting, some are defending their own setup, some are correcting misinformation, and some just want to dunk on a bad take.
- It is normal to be skeptical, uncertain, blunt, mildly messy, biased, sarcastic, or to ignore the thread entirely.
- Voting, lurking, and skipping threads are all normal, but unanswered advice-seeking threads usually pull a few people into replying.
- Do not default to upbeat recommendation language or balanced summaries.
- Avoid phrases like "I highly recommend", "you won't regret it", "game-changer", "feel free to ask", "I'd be happy to help", or "let me know if you have questions".
- Avoid checklist-y helper phrasing like "make sure", "just ensure", or "be sure to" unless that is genuinely the one thing you care about.
- Do not write like a marketing blurb.
- Avoid emoji unless this persona would obviously use them.
- Avoid hashtags unless you are joking.
- If you have not personally used something, do not pretend you have. Speak from what you have seen, read, or heard.
- Uneven knowledge is normal. You do not need to sound comprehensive, neutral, or fully informed. Partial knowledge, hearsay, overconfidence, and one-sided takes can all be realistic if they fit the persona.
- Stay consistent with your budget, habits, expertise, brand loyalties, and blind spots.
- React to the OP's concrete details instead of rewriting the whole topic. If they mention a fee, hard pull, budget, existing setup, trade-in, battery issue, fit issue, or one specific product, it is more realistic to answer that one detail than to give a full generic overview.
- Personal experience is especially valuable. If this persona could plausibly have a lived datapoint, use one concrete thing: what you paid, what annoyed you, what perk you actually used, what approval or retention outcome you got, or why you regretted keeping the card.
- Your product-specific opinions are part of your memory. When a visible post mentions one of those products, or a close substitute, use that personal angle before falling back to generic category advice.
- Do not mention every product opinion at once. Bring up at most one relevant product-specific experience in a comment, and only if it naturally fits the thread.
- Generic consensus replies should be the exception, not the default. If you can ground your comment in one specific thing from your own life or one specific disagreement, do that instead of repeating the obvious forum consensus.
- {anti_paraphrase}
- Do not make every comment sound clean, balanced, and well-structured. Real comment sections contain rough phrasing, tunnel vision, repetition, and comments that only address half of the issue.
- Real threads often contain blunt pile-ons, one-line agreements, jokes, dismissals, or a single missing caveat. They do not read like twenty polished mini-essays.

# ACTION RULES
- Most rounds should be low-intensity, but direct-question threads with barely any replies usually get at least a few answers.
- Do not drop a fresh top-level comment on your own thread just to bump it.
- Do not add another top-level comment to a thread you've already commented in unless something materially changed.
- Replying to a specific visible comment is normal and often more realistic than adding another top-level comment.
- You may use nested replies. If you want to answer a specific visible comment, reply to that comment instead of restating the same point as a new top-level comment.
- If several existing comments already say your point, it is better to vote or move on than to paraphrase them.
- If a thread is clearly asking for advice and still has very little discussion, commenting is more realistic than another empty vote.
- Starting a new thread should be rarer than commenting."""
    if not _runtime_allows_new_threads():
        prompt += """
- For this run, the subreddit starts from a fixed small set of seed threads. Do not open a brand new thread once those seeds exist."""
    prompt += """
- When a thread already has visible consensus, a realistic new comment usually adds just one thing: a datapoint, a pushback, a joke, a missing caveat, or a direct challenge to the OP's premise.
- If the thread already has several calm, similar answers, another polished balanced reply is usually the least realistic option.
- Avoid sounding like you're trying to be the definitive answer. Sound like one commenter, not the thread summary.
- If you post or comment, write in this persona's voice and do not mention these instructions.

# RESPONSE METHOD
Please perform exactly one action by tool calling.
"""
    return prompt


def render_reddit_feed_snapshot(
    refresh_payload: Mapping[str, Any],
    agent_id: int | None = None,
) -> str:
    """Render the current feed into a prompt block for one agent turn.

    The feed snapshot intentionally exposes just enough context for realistic
    action selection:
    - visible post text
    - score and comment counts
    - a few top comments
    - whether the current agent already touched or authored the thread
    """

    if not refresh_payload.get("success"):
        return "You refresh Reddit. The feed is unavailable right now."

    posts = refresh_payload.get("posts") or []
    if not posts:
        return "You refresh Reddit. The feed is quiet right now."

    touched_post_ids = _fetch_user_post_ids("comment", agent_id)
    top_level_touched_post_ids = _fetch_user_post_ids(
        "comment",
        agent_id,
        top_level_only=True,
    )
    authored_post_ids = _fetch_user_post_ids("post", agent_id)

    lines = [
        "You refresh Reddit and see these posts in your feed.",
        "Treat this like a normal browsing moment: some posts are skim-and-leave, some invite an actual reply.",
        "Use the visible comment counts and top comments to judge whether a thread already has enough discussion.",
    ]

    if agent_id is not None and touched_post_ids:
        touched = ", ".join(str(post_id) for post_id in sorted(touched_post_ids))
        lines.append(
            f"You already commented in these threads: {touched}."
        )
    if agent_id is not None and top_level_touched_post_ids:
        touched = ", ".join(
            str(post_id) for post_id in sorted(top_level_touched_post_ids)
        )
        lines.append(
            f"You already left a top-level comment in these threads: {touched}."
        )
    if agent_id is not None and authored_post_ids:
        authored = ", ".join(str(post_id) for post_id in sorted(authored_post_ids))
        lines.append(f"You already authored these threads: {authored}.")

    for post in posts[:6]:
        post_id = post.get("post_id", "?")
        author_id = post.get("user_id", "?")
        score = post.get("score", post.get("num_likes", 0))
        content = _compact_text(post.get("content", ""), limit=280)
        comments = sorted(post.get("comments") or [], key=_comment_sort_key, reverse=True)
        total_visible = _count_comment_nodes(comments)
        top_level_count = len(comments)
        nested_reply_count = max(0, total_visible - top_level_count)
        max_depth = _max_comment_depth(comments)
        markers = []
        if agent_id is not None and post_id in top_level_touched_post_ids:
            markers.append("you_already_commented=yes")
            markers.append("you_already_top_level_commented=yes")
        elif agent_id is not None and post_id in touched_post_ids:
            markers.append("you_already_commented=yes")
        if agent_id is not None and post_id in authored_post_ids:
            markers.append("your_post=yes")
        marker_text = f" | {' | '.join(markers)}" if markers else ""
        lines.append(
            f"- post_id={post_id} | author_user_id={author_id} | score={score} | comments={total_visible} | top_level_comments={top_level_count} | replies={nested_reply_count} | max_depth={max_depth}{marker_text}"
        )
        lines.append(f"  post: {content}")
        structure_note = _thread_structure_note(
            top_level_count=top_level_count,
            nested_reply_count=nested_reply_count,
            max_depth=max_depth,
        )
        if structure_note:
            lines.append(f"  structure: {structure_note}")
        perspective_note = _thread_perspective_note(comments)
        if perspective_note:
            lines.append(f"  perspective: {perspective_note}")
        rendered_count = 0
        for idx, comment in enumerate(comments[:_MAX_VISIBLE_ROOT_COMMENTS], start=1):
            rendered_count += _append_comment_preview_lines(
                lines,
                comment,
                label=f"top_comment_{idx}",
                level=0,
                remaining=_MAX_VISIBLE_COMMENT_NODES - rendered_count,
            )
            if rendered_count >= _MAX_VISIBLE_COMMENT_NODES:
                break
        if total_visible > rendered_count:
            lines.append(f"  ... {total_visible - rendered_count} more comments not shown")

    return "\n".join(lines)


def build_reddit_action_prompt(
    env_prompt: str,
    discussion_plan: Mapping[str, Any] | None = None,
    overlay: dict | None = None,
) -> str:
    """Wrap the feed snapshot in per-turn action instructions.

    `discussion_plan` is a soft scheduler hint produced outside the model. It
    does not hard-code which thread to comment on. It only nudges some agents to
    contribute when the board is too quiet overall.
    """

    plan = dict(discussion_plan or {})
    _overlay = dict(overlay or {})
    structure_pref_guidance = _overlay.get("prompt.structure_preference_weight") or "Balance standalone answers and back-and-forth."
    depth_cap_instruction = _overlay.get("prompt.depth_soft_cap_instruction") or "Once a visible chain is already around that depth, prefer branching out or making a fresh top-level comment unless you are directly continuing that exact sub-argument."
    consensus_handling = _overlay.get("prompt.consensus_handling") or "Treat repeated consensus as a warning sign. If three visible comments already say some version of the same thing, a fourth version should usually be very short or should say something meaningfully different."
    prompt_lines = [
        "Decide what you would realistically do on Reddit right now.",
        "",
        env_prompt,
        "",
        "Choose the single next action that best matches your actual level of interest.",
        "If you comment or post, keep it natural, specific, and in character.",
        "If you comment, usually give one concrete angle instead of a full shopping summary.",
        "Reply to one detail that is actually visible in the post or comments, not the whole topic.",
        "Before commenting, scan the visible comments and identify what the dominant angle already is.",
        "If your planned point mostly matches the dominant visible angle, do not write a full paraphrase. Either add one new detail, challenge it, reply directly to a visible claim, or pick a different angle.",
        "Most new comments should contribute a different perspective, a different reason, a different use case, or a different personal datapoint. Exact perspective overlap should be the minority, not the norm.",
        "Across a realistic thread, comments should vary: some are personal anecdotes, some are calm practical takes, some are skeptical challenges, and some are blunt or mildly rude.",
        "Across a realistic thread, motivations should vary too: help, vent, brag, warn, complain, correct, justify a past decision, or pile on.",
        "Not every reply should read like advice. Some comments just disagree, correct, pile on, ask one pointed question, or add one datapoint.",
        "If this persona plausibly has firsthand experience, a concrete lived detail is usually more realistic than generic recommendations.",
        "Default toward one specific personal angle, one regret, one annoyance, one thing you actually used, or one concrete disagreement before falling back to a generic recommendation.",
        "If this persona would only know part of the story, do not magically become comprehensive. A narrow, imperfect, or slightly biased comment is often more realistic.",
        "Sharper language, sarcasm, or mild offensiveness can happen, but only when it fits the persona and the thread. Do not make every comment combative.",
        "If the visible comments are already calm and helpful, another polished recommendation is usually less realistic than a sharper disagreement, joke, complaint, or one specific datapoint.",
        "Do not launder disagreement into overly careful hedging. A normal Reddit reply can be direct.",
        "Do not make every reply equally articulate. Some can be clipped, annoyed, repetitive, half-informed, or tunnel-visioned.",
        "If you are replying to another comment, react to that exact claim. Agree, disagree, qualify it, joke about it, or correct it. Do not just restate your generic answer to the original post.",
        "Do not add a fresh top-level comment to your own thread just to bump it.",
        "Do not restate the same advice already sitting in the thread.",
        "If you want to answer a visible comment, use create_comment with post_id, content, and parent_comment_id.",
        "Replying to a visible comment is often more realistic than adding another top-level comment.",
        "Do not add another top-level comment to a thread you already touched unless something materially changed.",
        "Use search only when you genuinely want more context before speaking.",
        "Avoid generic mini-reviews, balanced wrap-ups, and assistant-like advice paragraphs.",
        "If you agree with an existing comment, agree asymmetrically: add one personal datapoint, one stronger wording choice, one caveat, or one new reason. Do not just restate it in cleaner prose.",
        consensus_handling,
        "More realistic comment shapes are: blunt verdict, single caveat, personal datapoint, skeptical question, disagreement, joke, or one-line pile-on.",
        "Low-effort comments are normal if they still fit the thread.",
        "Top-level comments and replies should be mixed naturally. Do not turn every thread into isolated standalone answers, and do not bury everything in one deep chain.",
    ]

    thread_block_reason = _new_thread_creation_block_reason()
    if thread_block_reason:
        prompt_lines.extend(
            [
                thread_block_reason,
                "Do not use create_post, repost, or quote_post here.",
                "Stay inside the visible threads and choose between commenting, replying, voting, searching, or doing nothing.",
            ]
        )

    structure_preference = str(plan.get("structure_preference", "") or "").lower()
    if structure_preference == "top_level":
        prompt_lines.extend(
            [
                "Structure guidance for this turn: prefer a top-level comment on a thread that still lacks enough independent answers.",
                "Do not keep drilling deeper into an already active reply chain unless you are directly correcting or challenging that exact comment.",
            ]
        )
    elif structure_preference == "reply":
        prompt_lines.extend(
            [
                "Structure guidance for this turn: prefer replying to a visible comment so the thread feels conversational, not just a stack of standalone answers.",
                "Pick a visible comment you genuinely want to react to and answer that one point.",
            ]
        )
    elif structure_preference == "balanced":
        prompt_lines.extend(
            [
                f"Structure guidance for this turn: {structure_pref_guidance}",
                "If a thread only has one or two top-level takes, another top-level comment can be natural. If it already has several standalone takes but little interaction, reply to a visible comment instead.",
            ]
        )

    depth_soft_cap = plan.get("depth_soft_cap")
    try:
        depth_soft_cap_int = int(depth_soft_cap) if depth_soft_cap is not None else None
    except (TypeError, ValueError):
        depth_soft_cap_int = None
    if depth_soft_cap_int is not None and depth_soft_cap_int >= 1:
        prompt_lines.append(
            f"Soft depth guard: depth {depth_soft_cap_int} — {depth_cap_instruction}"
        )

    if plan.get("required"):
        reason = str(plan.get("reason", "") or "").strip()
        prompt_lines.extend(
            [
                "For this turn, you were selected to help keep the discussion moving instead of only voting or lurking.",
                "Look through the feed and pick one thread that still feels unresolved, under-discussed, or clearly asking for opinions.",
                "Prefer a post with very few comments over one that already has several visible replies.",
                "If you can answer one thread in character, create_comment is the expected action for this turn.",
                (
                    f"Why you're being nudged to comment: {reason}."
                    if reason
                    else "Several threads still look sparse."
                ),
                "Do not use do_nothing or vote-only actions unless none of the visible threads can be answered in character.",
                "Even when you do comment, keep it narrow. You are helping the thread move, not writing the final answer.",
                "For this forced-discussion turn, do not default to the safest generic advice. Pick a concrete motive and a concrete angle.",
            ]
        )
    else:
        prompt_lines.extend(
            [
                "If nothing strongly pulls you in, using do_nothing is normal.",
                "If a post is openly asking for advice or only has a couple of comments, replying can be more realistic than another empty vote.",
                "If a thread already has several comments saying the same thing, adding another polished version of that same advice is less realistic than skipping it or leaving one short additive remark.",
            ]
        )

    few_shot_block = render_generation_few_shot_block(
        _RUNTIME_CONTEXT.get("generation_few_shot_examples") or [],
        heading="REAL REDDIT STYLE EXAMPLES",
    )
    if few_shot_block:
        prompt_lines.extend(
            [
                "",
                few_shot_block,
                "",
                "Match the examples' realism signals: specific lived detail, uneven knowledge, short narrow takes, visible disagreement, and comments that do not sound like mini reviews.",
                "Notice that realistic comments are often incomplete, repetitive in a human way, or only half-helpful. They should not all sound crisp, patient, and optimized.",
                "Notice that realistic threads also distribute perspectives. Several commenters may overlap, but most do not sound like the same person rephrasing the same answer.",
            ]
        )

    prompt_lines.append("Use exactly one tool call.")
    return "\n".join(prompt_lines)


def apply_oasis_reddit_realism_patch() -> None:
    """Monkey patch upstream OASIS Reddit behavior in place.

    The patched hooks replace:
    - the Reddit system prompt constructor
    - the environment text prompt
    - the per-turn action wrapper
    - the top-level comment creation guardrails
    """
    from camel.messages import BaseMessage
    from oasis.social_agent.agent import ALL_SOCIAL_ACTIONS, SocialAgent, agent_log
    from oasis.social_agent.agent_action import SocialAction
    from oasis.social_agent.agent_environment import SocialEnvironment
    from oasis.social_platform.config import UserInfo
    from oasis.social_platform.platform import Platform
    from oasis.social_platform.platform_utils import PlatformUtils
    from oasis.social_platform.typing import ActionType

    if getattr(UserInfo.to_reddit_system_message, "__name__", "") == "_patched_to_reddit_system_message":
        return

    original_create_post = Platform.create_post
    original_repost = Platform.repost
    original_quote_post = Platform.quote_post

    def _patched_to_reddit_system_message(self: Any) -> str:
        return build_reddit_system_prompt(
            user_name=getattr(self, "user_name", None),
            display_name=getattr(self, "name", None),
            bio=getattr(self, "description", None),
            other_info=((self.profile or {}).get("other_info") or {}),
            overlay=_RUNTIME_CONTEXT.get("overlay"),
        )

    async def _patched_to_text_prompt(
        self: Any,
        include_posts: bool = True,
        include_followers: bool = True,
        include_follows: bool = True,
    ) -> str:
        del include_followers, include_follows
        if not include_posts:
            return "You are on Reddit right now."
        refresh_payload = await self.action.refresh()
        return render_reddit_feed_snapshot(
            refresh_payload,
            agent_id=getattr(self.action, "agent_id", None),
        )

    async def _patched_perform_action_by_llm(self: Any):
        env_prompt = await self.env.to_text_prompt()
        discussion_plan = dict(
            getattr(self, "_geo_discussion_plan", None) or {}
        )
        removed_tools = {}
        if _new_thread_creation_block_reason() is not None:
            for tool_name in ("create_post", "repost", "quote_post"):
                tool = self._internal_tools.pop(tool_name, None)
                if tool is not None:
                    removed_tools[tool_name] = tool
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=build_reddit_action_prompt(
                env_prompt,
                discussion_plan=discussion_plan,
                overlay=_RUNTIME_CONTEXT.get("overlay"),
            ),
        )
        try:
            agent_log.info(
                f"Agent {self.social_agent_id} observing environment: {env_prompt}"
            )
            response = await self.astep(user_msg)
            for tool_call in response.info["tool_calls"]:
                action_name = tool_call.tool_name
                args = tool_call.args
                agent_log.info(
                    f"Agent {self.social_agent_id} performed action: "
                    f"{action_name} with args: {args}"
                )
                if action_name not in ALL_SOCIAL_ACTIONS:
                    agent_log.info(
                        f"Agent {self.social_agent_id} get the result: "
                        f"{tool_call.result}"
                    )
                return response
        except Exception as exc:  # pragma: no cover - matches upstream behavior
            agent_log.error(f"Agent {self.social_agent_id} error: {exc}")
            return exc
        finally:
            for tool_name, tool in removed_tools.items():
                self._internal_tools[tool_name] = tool
            if hasattr(self, "_geo_discussion_plan"):
                delattr(self, "_geo_discussion_plan")

    async def _patched_create_post(self: Any, agent_id: int, content: str):
        max_total_threads = _runtime_max_total_threads()
        current_threads = _count_runtime_threads()
        if max_total_threads is not None and current_threads >= max_total_threads:
            return {
                "success": False,
                "error": (
                    "Thread limit reached for this run. Reply inside the existing "
                    "threads instead of creating a new one."
                ),
            }
        if not _runtime_allows_new_threads() and max_total_threads in (None, 0):
            return {
                "success": False,
                "error": (
                    "This run uses a fixed set of starter threads. "
                    "Create a comment or nested reply instead."
                ),
            }
        return await original_create_post(self, agent_id, content)

    async def _patched_repost(self: Any, agent_id: int, post_id: int):
        if _new_thread_creation_block_reason() is not None:
            return {
                "success": False,
                "error": (
                    "Reposts are disabled for this run. Stay inside the existing "
                    "threads and reply there."
                ),
            }
        return await original_repost(self, agent_id, post_id)

    async def _patched_quote_post(
        self: Any,
        agent_id: int,
        quote_message: tuple,
    ):
        if _new_thread_creation_block_reason() is not None:
            return {
                "success": False,
                "error": (
                    "Quote-posting is disabled for this run. Stay inside the existing "
                    "threads and reply there."
                ),
            }
        return await original_quote_post(self, agent_id, quote_message)

    async def _patched_create_comment(self: Any, agent_id: int, comment_message: tuple):
        post_id, content, parent_comment_id = _unpack_comment_message(comment_message)
        if self.recsys_type.name == "REDDIT":
            current_time = self.sandbox_clock.time_transfer(
                datetime.now(), self.start_time
            )
        else:
            current_time = self.sandbox_clock.get_time_step()
        try:
            post_type_result = self.pl_utils._get_post_type(post_id)
            if post_type_result and post_type_result.get("type") == "repost":
                post_id = post_type_result["root_post_id"]
            if not post_type_result:
                return {"success": False, "error": "Post not found."}

            user_id = agent_id
            depth = 0
            if parent_comment_id is None:
                author_query = "SELECT user_id FROM post WHERE post_id = ?"
                self.pl_utils._execute_db_command(author_query, (post_id,))
                author_result = self.db_cursor.fetchone()
                if author_result and int(author_result[0]) == int(agent_id):
                    return {
                        "success": False,
                        "error": (
                            "User already authored this thread. Avoid fresh top-level "
                            "self-replies; reply to someone specific instead."
                        ),
                    }
                check_query = (
                    "SELECT 1 FROM comment WHERE post_id = ? AND user_id = ? "
                    "AND (parent_comment_id IS NULL OR parent_comment_id = 0) LIMIT 1"
                )
                self.pl_utils._execute_db_command(check_query, (post_id, agent_id))
                if self.db_cursor.fetchone():
                    return {
                        "success": False,
                        "error": (
                            "User already left a top-level comment on this post. "
                            "Reply to a specific comment instead of stacking another top-level one."
                        ),
                    }
            else:
                parent_query = (
                    "SELECT post_id, depth FROM comment WHERE comment_id = ? LIMIT 1"
                )
                self.pl_utils._execute_db_command(parent_query, (parent_comment_id,))
                parent_row = self.db_cursor.fetchone()
                if not parent_row:
                    return {
                        "success": False,
                        "error": "Parent comment not found.",
                    }
                parent_post_id = int(parent_row[0])
                if int(parent_post_id) != int(post_id):
                    return {
                        "success": False,
                        "error": "Parent comment does not belong to this post.",
                    }
                depth = int(parent_row[1] or 0) + 1

            comment_insert_query = (
                "INSERT INTO comment (post_id, user_id, content, created_at, parent_comment_id, depth) "
                "VALUES (?, ?, ?, ?, ?, ?)"
            )
            self.pl_utils._execute_db_command(
                comment_insert_query,
                (post_id, user_id, content, current_time, parent_comment_id, depth),
                commit=True,
            )
            comment_id = self.db_cursor.lastrowid

            action_info = {
                "content": content,
                "post_id": post_id,
                "comment_id": comment_id,
            }
            if parent_comment_id is not None:
                action_info["parent_comment_id"] = int(parent_comment_id)
            self.pl_utils._record_trace(
                user_id,
                ActionType.CREATE_COMMENT.value,
                action_info,
                current_time,
            )
            return {"success": True, "comment_id": comment_id}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _patched_add_comments_to_posts(self: Any, posts_results: Any):
        posts = []
        for row in posts_results:
            (
                post_id,
                user_id,
                original_post_id,
                content,
                quote_content,
                created_at,
                num_likes,
                num_dislikes,
                num_shares,
            ) = row
            post_type_result = self._get_post_type(post_id)
            if post_type_result is None:
                continue
            original_user_id_query = "SELECT user_id FROM post WHERE post_id = ?"
            if post_type_result["type"] == "repost":
                self.db_cursor.execute(original_user_id_query, (original_post_id,))
                original_user_id = self.db_cursor.fetchone()[0]
                original_post_id = post_id
                post_id = post_type_result["root_post_id"]
                self.db_cursor.execute(
                    "SELECT content, quote_content, created_at, num_likes, "
                    "num_dislikes, num_shares, num_reports FROM post "
                    "WHERE post_id = ?",
                    (post_id,),
                )
                original_post_result = self.db_cursor.fetchone()
                (
                    content,
                    quote_content,
                    created_at,
                    num_likes,
                    num_dislikes,
                    num_shares,
                    num_reports,
                ) = original_post_result
                post_content = (
                    f"User {user_id} reposted a post from User "
                    f"{original_user_id}. Repost content: {content}. "
                )
            elif post_type_result["type"] == "quote":
                self.db_cursor.execute(original_user_id_query, (original_post_id,))
                original_user_id = self.db_cursor.fetchone()[0]
                post_content = (
                    f"User {user_id} quoted a post from User "
                    f"{original_user_id}. Quote content: {quote_content}. "
                    f"Original Content: {content}"
                )
                self.db_cursor.execute(
                    "SELECT num_reports FROM post WHERE post_id = ?",
                    (post_id,),
                )
                num_reports = self.db_cursor.fetchone()[0]
            else:
                post_content = content
                self.db_cursor.execute(
                    "SELECT num_reports FROM post WHERE post_id = ?",
                    (post_id,),
                )
                num_reports = self.db_cursor.fetchone()[0]

            self.db_cursor.execute(
                "SELECT comment_id, post_id, user_id, content, created_at, "
                "num_likes, num_dislikes, parent_comment_id, depth "
                "FROM comment WHERE post_id = ? ORDER BY created_at ASC, comment_id ASC",
                (post_id,),
            )
            comments_results = self.db_cursor.fetchall()
            comments = _nest_runtime_comments(
                comments_results,
                show_score=self.show_score,
            )

            if num_reports >= self.report_threshold:
                warning_message = ("[Warning: This post has been reported"
                                   f" {num_reports} times]")
                post_content = f"{warning_message}\n{post_content}"

            posts.append({
                "post_id": post_id if post_type_result["type"] != "repost" else original_post_id,
                "user_id": user_id,
                "content": post_content,
                "created_at": created_at,
                **({
                    "score": num_likes - num_dislikes
                } if self.show_score else {
                    "num_likes": num_likes,
                    "num_dislikes": num_dislikes,
                }),
                "num_shares": num_shares,
                "num_reports": num_reports,
                "comments": comments,
            })
        return posts

    async def _patched_social_action_create_comment(
        self: Any,
        post_id: int,
        content: str,
        parent_comment_id: int | None = None,
    ):
        """Create a comment on a post or reply to an existing comment.

        Args:
            post_id: The thread to comment in.
            content: The text of the comment.
            parent_comment_id: Optional comment id to reply to. Omit this for a
                new top-level comment.
        """
        comment_message = (
            (post_id, content, parent_comment_id)
            if parent_comment_id is not None
            else (post_id, content)
        )
        return await self.perform_action(
            comment_message,
            ActionType.CREATE_COMMENT.value,
        )

    _patched_social_action_create_comment.__name__ = "create_comment"

    UserInfo.to_reddit_system_message = _patched_to_reddit_system_message
    SocialEnvironment.to_text_prompt = _patched_to_text_prompt
    SocialAgent.perform_action_by_llm = _patched_perform_action_by_llm
    Platform.create_post = _patched_create_post
    Platform.repost = _patched_repost
    Platform.quote_post = _patched_quote_post
    Platform.create_comment = _patched_create_comment
    PlatformUtils._add_comments_to_posts = _patched_add_comments_to_posts
    SocialAction.create_comment = _patched_social_action_create_comment


async def generate_reddit_agent_graph(
    profile_path: str,
    model: Any = None,
    available_actions: list[Any] | None = None,
):
    """Create OASIS agents from GEO profiles without losing persona alignment.

    This function is the adapter between `reddit_profiles.json` and OASIS's
    `SocialAgent` graph. It injects GEO-specific `other_info` fields so the
    patched prompts can see persona text plus behavior knobs like stance,
    activity level, and response delays.
    """
    from oasis.social_agent import AgentGraph, SocialAgent
    from oasis.social_platform.config import UserInfo

    with open(profile_path, "r", encoding="utf-8") as file:
        agent_records = json.load(file)

    agent_graph = AgentGraph()
    explicit_ids = [record.get("user_id") for record in agent_records]
    use_profile_ids = (
        all(agent_id is not None for agent_id in explicit_ids)
        and sorted(int(agent_id) for agent_id in explicit_ids)
        == list(range(len(agent_records)))
    )

    async def process_agent(index: int) -> None:
        record = agent_records[index]
        record_user_id = resolve_reddit_agent_id(record, index)
        behavior_cfg = _RUNTIME_AGENT_CONFIGS.get(record_user_id, {})
        other_info = {
            "user_profile": record.get("persona", ""),
            "mbti": record.get("mbti"),
            "gender": record.get("gender"),
            "age": record.get("age"),
            "country": record.get("country"),
            "profession": record.get("profession"),
            "interested_topics": record.get("interested_topics"),
            "primary_motivation": record.get("primary_motivation"),
            "knowledge_style": record.get("knowledge_style"),
            "conflict_style": record.get("conflict_style"),
            "product_opinions": record.get("product_opinions"),
            "karma": record.get("karma"),
            "activity_level": behavior_cfg.get("activity_level"),
            "posts_per_hour": behavior_cfg.get("posts_per_hour"),
            "comments_per_hour": behavior_cfg.get("comments_per_hour"),
            "response_delay_min": behavior_cfg.get("response_delay_min"),
            "response_delay_max": behavior_cfg.get("response_delay_max"),
            "sentiment_bias": behavior_cfg.get("sentiment_bias"),
            "stance": behavior_cfg.get("stance"),
        }
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {
                key: value
                for key, value in other_info.items()
                if value not in (None, "", [])
            },
        }

        user_info = UserInfo(
            user_name=record.get("username"),
            name=record.get("name") or record.get("realname") or record.get("username"),
            description=record.get("bio"),
            profile=profile,
            recsys_type="reddit",
        )

        agent = SocialAgent(
            agent_id=(record_user_id if use_profile_ids else index),
            user_info=user_info,
            agent_graph=agent_graph,
            model=model,
            available_actions=available_actions,
        )
        agent_graph.add_agent(agent)

    await asyncio.gather(*(process_agent(i) for i in range(len(agent_records))))
    return agent_graph


def _build_behavior_guidance(other_info: Mapping[str, Any]) -> list[str]:
    """Translate numeric behavior controls into prompt-visible natural language."""

    lines: list[str] = []

    stance = str(other_info.get("stance", "") or "").lower()
    primary_motivation = str(other_info.get("primary_motivation", "") or "").lower()
    knowledge_style = str(other_info.get("knowledge_style", "") or "").lower()
    conflict_style = str(other_info.get("conflict_style", "") or "").lower()

    if primary_motivation:
        if "correct" in primary_motivation:
            lines.append("You often reply because you want to correct people or puncture bad advice.")
        elif "complain" in primary_motivation or "vent" in primary_motivation:
            lines.append("You often show up to vent, warn people off, or complain about annoying tradeoffs.")
        elif "defend" in primary_motivation or "validation" in primary_motivation:
            lines.append("You often reply to justify your own setup or look for validation that your choice made sense.")
        elif "jok" in primary_motivation:
            lines.append("You sometimes reply just to make a joke or pile on with a snarky aside.")
        elif "help" in primary_motivation:
            lines.append("You do sometimes try to be helpful, but you should still sound like a normal commenter rather than a neat explainer.")

    if knowledge_style:
        if "beginner" in knowledge_style:
            lines.append("Your knowledge is patchy. It is fine to sound uncertain, simplistic, or focused on one obvious angle.")
        elif "casual" in knowledge_style:
            lines.append("You know enough for normal use but not enough to sound like the resident expert.")
        elif "overconfident" in knowledge_style or "half" in knowledge_style:
            lines.append("You often sound more certain than your actual expertise justifies.")
        elif "experienced" in knowledge_style or "owner" in knowledge_style:
            lines.append("You can draw on lived experience, not just generic advice.")
        elif "specialist" in knowledge_style:
            lines.append("You know the niche details, but you still should not flatten the whole thread into a polished lecture.")

    if conflict_style:
        if "sarcastic" in conflict_style:
            lines.append("When you disagree, sarcasm comes naturally.")
        elif "blunt" in conflict_style:
            lines.append("When you disagree, you tend to be blunt instead of cushioning the point.")
        elif "argumentative" in conflict_style:
            lines.append("You do not mind escalating a disagreement if a take sounds obviously wrong.")
        elif "skeptical" in conflict_style:
            lines.append("You tend to poke holes in claims before agreeing with them.")
        elif "avoidant" in conflict_style:
            lines.append("You usually avoid long fights and prefer one quick comment over a drawn-out argument.")
        elif "calm" in conflict_style:
            lines.append("Even when you disagree, you usually stay measured rather than heated.")

    if stance == "observer":
        lines.append("You are more of a lurker. Commenting should be relatively rare for you.")
    elif stance == "supportive":
        lines.append("When you speak up, you lean measured, practical, or encouraging rather than abrasive.")
    elif stance == "opposing":
        lines.append("You are comfortable sounding sharp, dismissive, or mildly rude when a take sounds wrong, overhyped, or wasteful.")
        lines.append("Your motivation is often to push back, correct, or puncture hype rather than to give a full helpful overview.")

    sentiment_bias = other_info.get("sentiment_bias")
    if isinstance(sentiment_bias, (int, float)):
        if sentiment_bias <= -0.2:
            lines.append("You skew skeptical and notice downsides first.")
            lines.append("If a post feels naive, wasteful, or overhyped, you can sound impatient or a little cutting.")
        elif sentiment_bias >= 0.2:
            lines.append("You skew favorable when something looks like a decent buy or a fair tradeoff.")
            lines.append("When you think something is fine, you are more likely to sound reassuring than hostile.")

    comments_per_hour = _to_float(other_info.get("comments_per_hour"))
    posts_per_hour = _to_float(other_info.get("posts_per_hour"))
    if comments_per_hour is not None and posts_per_hour is not None:
        if comments_per_hour >= 2.0 and comments_per_hour > posts_per_hour:
            lines.append("You jump into comment sections more often than you start threads.")
        elif comments_per_hour <= 0.8 and posts_per_hour <= 0.4:
            lines.append("You are fairly quiet and often just read or vote.")

    karma = _to_float(other_info.get("karma"))
    if karma is not None:
        if karma <= 1500:
            lines.append("You are not the most authoritative voice in the room. It is normal for you to sound tentative, half-informed, or to latch onto one simple point.")
        elif karma >= 10000:
            lines.append("You sound like someone who has been around these discussions for a while, but you still should not turn every reply into a polished lecture.")

    delay_min = _to_float(other_info.get("response_delay_min"))
    delay_max = _to_float(other_info.get("response_delay_max"))
    if delay_min is not None and delay_max is not None:
        avg_delay = (delay_min + delay_max) / 2
        if avg_delay >= 60:
            lines.append("You usually read for a while before interacting, not instantly.")
        elif avg_delay <= 20:
            lines.append("You react fairly quickly when a thread hooks you.")

    activity_level = _to_float(other_info.get("activity_level"))
    if activity_level is not None:
        if activity_level <= 0.35:
            lines.append("You are not online constantly. Skipping most threads is normal for you.")
        elif activity_level >= 0.8:
            lines.append("You check Reddit pretty often, but that still does not mean you need to comment every time.")

    return lines


def _format_product_opinions(product_opinions: Any) -> list[str]:
    """Render a persona's product-specific memory into short prompt bullets."""

    if not isinstance(product_opinions, list):
        return []

    lines: list[str] = []
    for raw in product_opinions[:4]:
        if not isinstance(raw, Mapping):
            continue
        title = _compact_text(raw.get("product_title") or "", limit=90)
        if not title:
            continue
        relationship = _compact_text(raw.get("relationship") or "has an opinion on", limit=40)
        personal_angle = _compact_text(raw.get("personal_angle") or "", limit=170)
        likely_comment_angle = _compact_text(
            raw.get("likely_comment_angle") or "",
            limit=150,
        )
        pieces = [f"{relationship} {title}"]
        if personal_angle:
            pieces.append(personal_angle)
        if likely_comment_angle:
            pieces.append(f"Usually brings it up as: {likely_comment_angle}")
        lines.append(" | ".join(pieces))
    return lines


def _build_context_guidance() -> list[str]:
    """Render run-level simulation context into prompt-visible bullet points."""

    lines: list[str] = []
    simulation_requirement = _RUNTIME_CONTEXT.get("simulation_requirement")
    if simulation_requirement:
        lines.append(str(simulation_requirement))

    hot_topics = [str(topic) for topic in (_RUNTIME_CONTEXT.get("hot_topics") or []) if topic]
    if hot_topics:
        lines.append(f"Recurring discussion themes: {', '.join(hot_topics[:5])}.")

    narrative_direction = _RUNTIME_CONTEXT.get("narrative_direction")
    if narrative_direction:
        lines.append(str(narrative_direction))

    return lines


def _comment_sort_key(comment: Mapping[str, Any]) -> tuple[float, str]:
    """Sort visible comments by score first, then time."""

    score = comment.get("score", comment.get("num_likes", 0)) or 0
    created_at = str(comment.get("created_at", "") or "")
    return float(score), created_at


def _compact_text(text: Any, limit: int) -> str:
    """Collapse whitespace and trim long text for prompt display."""

    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _fetch_user_post_ids(
    table_name: str,
    agent_id: int | None,
    top_level_only: bool = False,
) -> set[int]:
    """Read the run DB to see which threads the agent already authored/touched."""

    if agent_id is None:
        return set()

    db_path = _resolve_runtime_db_path()
    if not db_path or not os.path.exists(db_path):
        return set()

    if table_name == "comment":
        condition = ""
        if top_level_only:
            try:
                conn = sqlite3.connect(db_path)
                try:
                    cursor = conn.cursor()
                    if _table_has_column(cursor, "comment", "parent_comment_id"):
                        condition = " AND (parent_comment_id IS NULL OR parent_comment_id = 0)"
                finally:
                    conn.close()
            except sqlite3.Error:
                condition = ""
        query = f"SELECT DISTINCT post_id FROM {table_name} WHERE user_id = ?{condition}"
    else:
        query = "SELECT DISTINCT post_id FROM post WHERE user_id = ?"

    try:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(query, (agent_id,))
            return {int(row[0]) for row in cursor.fetchall()}
        finally:
            conn.close()
    except sqlite3.Error:
        return set()


def _resolve_runtime_db_path() -> str | None:
    """Return the SQLite path for the current simulation run, if known."""

    if _RUNTIME_DB_PATH:
        return _RUNTIME_DB_PATH
    return os.environ.get("OASIS_DB_PATH")


def _to_float(value: Any) -> float | None:
    """Best-effort float conversion used for optional behavior controls."""

    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensure_nested_comment_schema(db: Any, db_cursor: Any) -> None:
    """Add reply-tree columns to the OASIS comment table if they are missing."""

    for column_name, ddl in (
        ("parent_comment_id", "ALTER TABLE comment ADD COLUMN parent_comment_id INTEGER"),
        ("depth", "ALTER TABLE comment ADD COLUMN depth INTEGER DEFAULT 0"),
    ):
        if _table_has_column(db_cursor, "comment", column_name):
            continue
        db_cursor.execute(ddl)
        db.commit()


def _table_has_column(db_cursor: Any, table_name: str, column_name: str) -> bool:
    """Return whether the SQLite table contains the requested column."""

    db_cursor.execute(f"PRAGMA table_info({table_name})")
    return any(str(row[1]) == column_name for row in db_cursor.fetchall())


def _unpack_comment_message(comment_message: tuple[Any, ...]) -> tuple[int, str, int | None]:
    """Normalize CREATE_COMMENT payloads from legacy and patched tool schemas."""

    if len(comment_message) == 2:
        post_id, content = comment_message
        return int(post_id), str(content), None
    if len(comment_message) == 3:
        post_id, content, parent_comment_id = comment_message
        parent = None if parent_comment_id in (None, "", 0) else int(parent_comment_id)
        return int(post_id), str(content), parent
    raise ValueError(f"Unexpected comment message shape: {comment_message!r}")


def _nest_runtime_comments(
    comments_results: list[Any],
    show_score: bool,
) -> list[dict[str, Any]]:
    """Convert raw SQLite comment rows into a nested reply tree."""

    comment_map: dict[int, dict[str, Any]] = {}
    ordered_comments: list[dict[str, Any]] = []
    for (
        comment_id,
        post_id,
        user_id,
        content,
        created_at,
        num_likes,
        num_dislikes,
        parent_comment_id,
        depth,
    ) in comments_results:
        comment = {
            "comment_id": comment_id,
            "post_id": post_id,
            "user_id": user_id,
            "content": content,
            "created_at": created_at,
            "parent_comment_id": parent_comment_id,
            "depth": int(depth or 0),
            "replies": [],
            **({
                "score": num_likes - num_dislikes
            } if show_score else {
                "num_likes": num_likes,
                "num_dislikes": num_dislikes,
            }),
        }
        comment_map[int(comment_id)] = comment
        ordered_comments.append(comment)

    roots: list[dict[str, Any]] = []
    for comment in ordered_comments:
        parent_comment_id = comment.get("parent_comment_id")
        if parent_comment_id in (None, "", 0):
            roots.append(comment)
            continue
        parent = comment_map.get(int(parent_comment_id))
        if parent is None:
            roots.append(comment)
            continue
        parent.setdefault("replies", []).append(comment)

    return roots


def _append_comment_preview_lines(
    lines: list[str],
    comment: Mapping[str, Any],
    label: str,
    level: int,
    remaining: int,
) -> int:
    """Render one visible comment subtree into the feed snapshot."""

    if remaining <= 0:
        return 0

    indent = "  " * (level + 1)
    comment_id = comment.get("comment_id", "?")
    parent_comment_id = comment.get("parent_comment_id")
    comment_author = comment.get("user_id", "?")
    comment_score = comment.get("score", comment.get("num_likes", 0))
    comment_text = _compact_text(
        comment.get("content") or comment.get("comment") or "",
        limit=180,
    )
    extra = (
        f" | reply_to_comment_id={parent_comment_id}"
        if parent_comment_id not in (None, "", 0)
        else ""
    )
    lines.append(
        f"{indent}{label}: comment_id={comment_id} | depth={comment.get('depth', level)} "
        f"| user_id={comment_author} | score={comment_score}{extra} | {comment_text}"
    )
    rendered = 1
    replies = sorted(comment.get("replies") or [], key=_comment_sort_key, reverse=True)
    for idx, reply in enumerate(replies[:2], start=1):
        child_label = f"reply_{idx}"
        rendered += _append_comment_preview_lines(
            lines,
            reply,
            label=child_label,
            level=level + 1,
            remaining=remaining - rendered,
        )
        if rendered >= remaining:
            break
    return rendered


def _thread_perspective_note(comments: list[Mapping[str, Any]]) -> str:
    """Describe whether visible comments already converge on one angle.

    This note is intentionally lightweight. It does not try to fully summarize
    thread semantics. It only gives the model a prompt-visible reminder when the
    visible comments are already paraphrasing each other, so the next reply is
    nudged toward a different angle or a direct reaction rather than another
    interchangeable answer.
    """

    visible_texts = _flatten_visible_comment_texts(comments, limit=8)
    if not visible_texts:
        return ""
    if len(visible_texts) == 1:
        return (
            "Only one visible perspective is on the board so far. A different "
            "angle, personal datapoint, or probing follow-up could make the "
            "thread feel more real."
        )

    repeated_opening_ratio = _repeated_opening_ratio(visible_texts)
    lexical_overlap = _average_text_overlap(visible_texts)
    if repeated_opening_ratio >= 0.34 or lexical_overlap >= 0.23:
        return (
            "Several visible comments are already circling the same point. If "
            "you add a new comment, it will feel more realistic if it brings a "
            "different perspective, a concrete personal experience, a caveat, "
            "or a direct challenge instead of another paraphrase."
        )
    if len(visible_texts) >= 3:
        return (
            "There are already a few visible angles here. If you comment, react "
            "to one specific claim or add a distinct reason rather than writing "
            "a detached generic summary."
        )
    return ""


def _flatten_visible_comment_texts(
    comments: list[Mapping[str, Any]],
    limit: int,
) -> list[str]:
    """Collect a bounded list of visible comment texts from the nested tree."""

    collected: list[str] = []

    def walk(nodes: list[Mapping[str, Any]]) -> None:
        if len(collected) >= limit:
            return
        for node in nodes:
            text = _compact_text(node.get("content") or node.get("comment") or "", 220)
            if text:
                collected.append(text)
                if len(collected) >= limit:
                    return
            replies = sorted(node.get("replies") or [], key=_comment_sort_key, reverse=True)
            walk(replies)
            if len(collected) >= limit:
                return

    walk(comments)
    return collected


def _repeated_opening_ratio(texts: list[str]) -> float:
    """Estimate how many visible comments start with near-identical phrasing."""

    openings: dict[str, int] = {}
    for text in texts:
        tokens = _normalize_overlap_tokens(text)
        if not tokens:
            continue
        opening = " ".join(tokens[:6])
        if not opening:
            continue
        openings[opening] = openings.get(opening, 0) + 1
    if not openings:
        return 0.0
    max_count = max(openings.values())
    return max_count / max(1, len(texts))


def _average_text_overlap(texts: list[str]) -> float:
    """Estimate how interchangeable visible comments look at the token level."""

    token_sets = [set(_normalize_overlap_tokens(text)) for text in texts]
    token_sets = [tokens for tokens in token_sets if tokens]
    if len(token_sets) < 2:
        return 0.0

    overlap_total = 0.0
    pair_count = 0
    for idx, left in enumerate(token_sets):
        for right in token_sets[idx + 1 :]:
            union = left | right
            if not union:
                continue
            overlap_total += len(left & right) / len(union)
            pair_count += 1
    if pair_count == 0:
        return 0.0
    return overlap_total / pair_count


def _normalize_overlap_tokens(text: str) -> list[str]:
    """Normalize comment text into coarse lexical tokens for overlap checks."""

    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "do",
        "for",
        "from",
        "have",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "just",
        "me",
        "my",
        "not",
        "of",
        "on",
        "or",
        "really",
        "so",
        "that",
        "the",
        "their",
        "them",
        "there",
        "they",
        "this",
        "to",
        "was",
        "with",
        "you",
        "your",
    }
    normalized = []
    for raw_token in str(text or "").lower().replace("’", "'").split():
        token = "".join(ch for ch in raw_token if ch.isalnum() or ch == "'").strip("'")
        if len(token) < 3 or token in stopwords:
            continue
        normalized.append(token)
    return normalized


def _count_comment_nodes(comments: list[Mapping[str, Any]]) -> int:
    """Count all visible comments in a nested reply tree."""

    total = 0
    for comment in comments:
        total += 1
        total += _count_comment_nodes(list(comment.get("replies") or []))
    return total


def _max_comment_depth(comments: list[Mapping[str, Any]]) -> int:
    """Return the maximum visible reply depth in a nested comment tree."""

    max_depth = 0
    for comment in comments:
        try:
            comment_depth = int(comment.get("depth", 0) or 0)
        except (TypeError, ValueError):
            comment_depth = 0
        max_depth = max(max_depth, comment_depth)
        max_depth = max(max_depth, _max_comment_depth(list(comment.get("replies") or [])))
    return max_depth


def _thread_structure_note(
    top_level_count: int,
    nested_reply_count: int,
    max_depth: int,
) -> str:
    """Summarize whether a thread needs more top-level takes or more interaction."""

    if top_level_count <= 1 and nested_reply_count >= 1:
        return "Most of the activity is buried under one comment. Another standalone top-level answer would feel natural."
    if top_level_count <= 1:
        return "This thread still has very few independent answers."
    if top_level_count >= 2 and nested_reply_count == 0:
        return "This thread has standalone answers but almost no back-and-forth yet."
    if max_depth >= 2:
        return "There is already a deeper reply chain here. Another branch or a fresh top-level comment may feel more natural than going even deeper."
    return ""
