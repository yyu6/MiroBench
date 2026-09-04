#!/usr/bin/env python3
"""The LLM half of the self-loop: propose rewrites, never decide.

One call per selected comment returns several candidates; the caller ranks them
with `candidate_scorer` and the round is then gated by `metric_engine` on the
official scorers. The model is asked for alternatives, never for a judgement --
"the LLM proposes, metric gates decide" is the rule the CARD controller was
built on and it is kept.

Domain adaptivity is structural, not a prompt variable: nothing here names a
product category. What the model sees about the domain is the thread itself,
the community string from the domain config, and the anchors extracted from the
comment under revision.
"""
from __future__ import annotations

import concurrent.futures as futures
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "selfloop"))
import strategies as S  # noqa: E402

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass
class Target:
    thread_id: str
    index: int          # position in the thread's flattened comment list
    comment_id: str
    text: str
    parent_text: str
    instruction: str    # what to change, from the group's strategy
    evidence: str       # why THIS comment, measured on its own thread
    anchors: list[str]  # facts it already states, which a rewrite must keep


def load_env(path: Path = REPO_ROOT / "third_party/MiroFish/.env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def client(base_url: str = DEFAULT_BASE_URL, api_key_env: str = "LLM_API_KEY"):
    from openai import OpenAI

    load_env()
    key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(f"no API key in ${api_key_env}")
    return OpenAI(api_key=key, base_url=base_url, timeout=300)


def build_prompt(target: Target, *, community: str, keep: str,
                 candidates: int, feedback: str = "") -> str:
    """One comment, one problem, one set of rules.

    Five sections, and each earns its place: the comment, what it answers, what
    is wrong with it (in prose, then in measurements), what must survive, and
    the output shape. Two sections were removed rather than added to. The first
    printed the metric name and two floats -- "semantic_mean_cosine = 0.2277,
    a real thread sits at 0.1792" -- which a language model cannot act on. The
    second dumped eight neighbours truncated at 220 characters, which both cut
    off the overlapping spans and need not have contained the comment this one
    was actually duplicating; `evidence` names the right ones in full instead.
    """
    anchor_block = ", ".join(target.anchors) if target.anchors else "(none — it states no specific fact)"
    parent_block = target.parent_text[:400] if target.parent_text else "(it replies to the post itself)"
    feedback_block = f"\n{feedback}\n" if feedback else ""
    return f"""You are rewriting one comment from a discussion thread on {community}.

THE COMMENT:
{target.text}

IT REPLIES TO:
{parent_block}

WHAT IS WRONG WITH IT:
{target.instruction}

{target.evidence}

FACTS IT ALREADY STATES — keep the ones it used, add none:
{anchor_block}

RULES:
{S.SHARED_INVARIANTS}
{keep}
{feedback_block}
Return strict JSON and nothing else:
{{"candidates": [{{"text": "<rewritten comment>", "what_changed": "<six words>"}}]}}
Give exactly {candidates} candidates that differ from each other in substance,
not only in wording."""


_JSON = re.compile(r"\{.*\}", re.S)


def parse(raw: str) -> list[dict[str, str]]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    match = _JSON.search(text)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for item in payload.get("candidates") or []:
        body = str((item or {}).get("text") or "").strip()
        if body:
            out.append({"text": body, "what_changed": str(item.get("what_changed") or "")})
    return out


def propose(
    api,
    targets: Sequence[Target],
    *,
    community: str,
    keep: str,
    model: str = DEFAULT_MODEL,
    candidates: int = 5,
    workers: int = 8,
    feedback: dict[str, str] | None = None,
) -> dict[tuple[str, int], list[dict[str, str]]]:
    """One call per target, run concurrently. Returns candidates per target."""

    feedback = feedback or {}

    def one(target: Target) -> tuple[tuple[str, int], list[dict[str, str]]]:
        prompt = build_prompt(
            target, community=community, keep=keep, candidates=candidates,
            feedback=feedback.get(f"{target.thread_id}:{target.index}", ""),
        )
        for attempt in range(3):
            try:
                response = api.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                return (target.thread_id, target.index), parse(response.choices[0].message.content or "")
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    return (target.thread_id, target.index), []
                time.sleep(3 * (attempt + 1))
        return (target.thread_id, target.index), []

    out: dict[tuple[str, int], list[dict[str, str]]] = {}
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for key, value in pool.map(one, targets):
            out[key] = value
    return out
