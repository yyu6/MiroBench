#!/usr/bin/env python3
"""Replay shipped Writer prompts through a different model, offline.

G109 measured the model-swap null on two threads with a 4-bit 8B local model.
This replays the SAME rendered prompts an existing paid run already produced,
so the Planner cost is zero and only Writer tokens are spent. The output is a
JSON file of {thread_id, slot, original, replayed} that
`writer_model_mix_score.py` turns into within-thread pairwise BERTScore for the
original corpus, the replayed corpus, and a simulated 50/50 mixed-writer policy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path("/Users/yaoningyu/Desktop/UIUC/GEO")


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def collect(run: Path, min_size: int, max_size: int, max_threads: int) -> list[dict]:
    """Select WHOLE threads whose natural size falls in [min_size, max_size].

    Truncating a long thread to its first N comments would take a shallow
    prefix, and G125 shows coverage truncation depresses `self_bertscore`
    artefactually. Taking complete mid-sized threads avoids both.
    """

    by_thread: dict[str, list[dict]] = {}
    for path in sorted((run / "cleaned").glob("run_*_sampled_reddit/discussion.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for post in payload.get("posts") or []:
            seed = post.get("seed_index")
            tid = f"seed{seed:03d}" if isinstance(seed, int) else str(seed)
            rows = by_thread.setdefault(tid, [])
            for record in post.get("generation_records") or []:
                comment = record.get("comment")
                prompt = record.get("prompt")
                if not isinstance(comment, dict) or not isinstance(prompt, str):
                    continue
                text = str(comment.get("content") or "").strip()
                if not text or not prompt.strip():
                    continue
                rows.append(
                    {
                        "thread_id": tid,
                        "comment_id": comment.get("comment_id"),
                        "prompt": prompt,
                        "original": text,
                    }
                )
    eligible = {
        tid: rows
        for tid, rows in by_thread.items()
        if min_size <= len(rows) <= max_size
    }
    chosen = sorted(eligible, key=lambda t: -len(eligible[t]))[:max_threads]
    print(
        "thread sizes: "
        + ", ".join(f"{t}={len(r)}" for t, r in sorted(by_thread.items(), key=lambda kv: -len(kv[1])))
        + f"\n  eligible [{min_size},{max_size}]: {sorted(eligible)}  chosen: {sorted(chosen)}",
        flush=True,
    )
    out: list[dict] = []
    for tid in chosen:
        out.extend(eligible[tid])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v128_interaction_n10_20260828_v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-threads", type=int, default=4)
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--max-size", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="smoke cap on total calls")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--frequency-penalty", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args()

    load_env(REPO / "third_party" / "MiroFish" / ".env")
    run = REPO / "artifacts" / "generalized_card" / "runs" / args.run
    items = collect(run, args.min_size, args.max_size, args.max_threads)
    if args.limit:
        items = items[: args.limit]
    chars = sum(len(x["prompt"]) for x in items)
    print(
        f"slots={len(items)} threads={len({x['thread_id'] for x in items})} "
        f"prompt_chars={chars} approx_input_tokens={chars // 4}",
        flush=True,
    )
    if args.estimate_only:
        return 0

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    reasoning = args.model.lower().startswith("gpt-5") or args.model.lower().startswith("o")
    results = []
    for index, item in enumerate(items):
        kwargs: dict = {
            "model": args.model,
            "messages": [{"role": "user", "content": item["prompt"]}],
        }
        if reasoning:
            kwargs["max_completion_tokens"] = args.max_tokens + 256
            if args.reasoning_effort:
                kwargs["reasoning_effort"] = args.reasoning_effort
        else:
            kwargs["max_tokens"] = args.max_tokens
        # backend.py:3735 claims gpt-5 endpoints reject non-default temperature.
        # Probed 2026-08-28: gpt-5.4-mini accepts temperature, top_p,
        # frequency_penalty and presence_penalty. The shipped path sets none of
        # them, so every gpt-5.x run to date used default sampling.
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            kwargs["top_p"] = args.top_p
        if args.frequency_penalty is not None:
            kwargs["frequency_penalty"] = args.frequency_penalty
        if args.presence_penalty is not None:
            kwargs["presence_penalty"] = args.presence_penalty
        text = ""
        for attempt in range(4):
            try:
                response = client.chat.completions.create(**kwargs)
                text = str(response.choices[0].message.content or "").strip()
                if text:
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"  [retry {attempt}] {type(exc).__name__}: {exc}", flush=True)
                time.sleep(3 * (attempt + 1))
        results.append({**item, "replayed": text})
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(items)}", flush=True)
    Path(args.out).write_text(
        json.dumps({"model": args.model, "run": args.run,
                    "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                                 "frequency_penalty": args.frequency_penalty,
                                 "presence_penalty": args.presence_penalty,
                                 "reasoning_effort": args.reasoning_effort},
                    "items": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    empty = sum(1 for r in results if not r["replayed"])
    print(f"wrote {args.out}  slots={len(results)}  empty={empty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
