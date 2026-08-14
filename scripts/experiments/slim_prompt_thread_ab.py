#!/usr/bin/env python3
"""Does slimming the Writer prompt cost within-thread diversity?

A 30-slot single-comment A/B showed that a 532-character plain-language prompt
beat the 21,920-character shipped prompt on register (the converged
"the part that" frame fell from 26.7% to 6.7%) and matched it on length. That
test generated each slot independently, so it could not measure the thing the
long prompt is actually buying: the thread-level ledgers are 78% of the prompt
and exist to stop comments in one thread from repeating each other.

This regenerates one whole thread in slot order against a slimmed prompt that
keeps a compressed ledger, then scores both variants with the same
`scripts/evaluation` implementations the real evaluation uses, so the numbers are
comparable to a run's `self_bleu_4` and `self_bertscore_mean_f1`.

Variant A is free: it is the thread the run already produced.

    python3 scripts/experiments/slim_prompt_thread_ab.py \
        --tag generalized_card_camera_gpt54_v73_smoke10_20260814_v1 \
        --thread-index 4 --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "artifacts" / "generalized_card" / "runs"
WARM = re.compile(
    r"\b(love|amazing|awesome|great|thanks|thank you|appreciate|glad|happy|nice"
    r"|excited|congrats|beautiful|enjoy|fantastic|perfect)\b",
    re.I,
)
FRAME = re.compile(
    r"\b(that'?s|it'?s) the (part|bit|thing|one|only|annoying)"
    r"|the (part|bit|thing) that\b",
    re.I,
)
# The ledger is what the long prompt is for. Keeping it, compressed, is the
# point of the experiment: if a short prompt with a short ledger holds
# within-thread diversity, the other ~17,000 characters are not paying for it.
LEDGER_OPENING_WORDS = 6
LEDGER_MOVE_CHARS = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--thread-index",
        type=int,
        default=4,
        help="Which run_NN thread to rebuild. Pick a mid-size one; cost scales with it.",
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to artifacts/experiments/slim_prompt_ab/<tag>_run<NN>.",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Self-BLEU only. Faster, and enough to see a lexical-diversity regression.",
    )
    return parser.parse_args()


def load_thread(tag: str, index: int) -> tuple[dict, list[dict]]:
    """Return the run's post record and its generation records in slot order."""

    generated = RUNS / tag / "generated" / f"run_{index:02d}_sampled_reddit"
    discussion = json.load(open(generated / "discussion.json"))
    post = discussion["posts"][0]
    records = json.load(open(generated / "generation_records.json"))
    records.sort(key=lambda r: int((r.get("task") or {}).get("local_task_id") or 0))
    return post, records


def slim_prompt(
    task: dict,
    *,
    community: str,
    seed_title: str,
    parent_text: str,
    prior: list[dict],
) -> str:
    """The same requirement in plain language, plus a compressed ledger."""

    def field(name: str) -> str:
        return " ".join(str(task.get(name) or "").split())

    words = int(task.get("real_word_count") or 0)
    lines = [
        f"You are commenting in {community}. Write one comment and nothing else.",
        "",
        f"Discussion: {seed_title}",
    ]
    if parent_text:
        lines.append(f"You are replying to: {parent_text[:600]}")
    lines += ["", f"What you want to say: {field('semantic_move')}"]
    for label, name in (
        ("The detail you have in mind", "detail_focus"),
        ("What you add that the parent did not", "reply_delta"),
    ):
        value = field(name)
        if value and value != "none":
            lines.append(f"{label}: {value}")
    lines += [
        "",
        f"You are the kind of commenter who is: {field('speaker_role').replace('_', ' ')}",
        f"How you come across: {field('tone_target') or field('voice')}",
        f"How you feel about it: {field('affect_role')}",
        f"Length: around {words} words.",
    ]
    if prior:
        openings = []
        moves = []
        for row in prior:
            text = " ".join(str(row.get("text") or "").split())
            if text:
                openings.append(" ".join(text.split()[:LEDGER_OPENING_WORDS]))
            move = " ".join(str(row.get("semantic_move") or "").split())
            if move:
                moves.append(move[:LEDGER_MOVE_CHARS])
        if openings:
            lines += [
                "",
                "Comments already in this thread opened like this. Start differently:",
                *(f"- {value}" for value in openings[-24:]),
            ]
        if moves:
            lines += [
                "",
                "These points are already covered. Do not repeat them:",
                *(f"- {value}" for value in moves[-16:]),
            ]
    lines += ["", "Write it the way someone actually types on Reddit."]
    return "\n".join(lines)


def generate(args: argparse.Namespace, post: dict, records: list[dict]) -> list[dict]:
    from openai import OpenAI

    key = os.environ.get(args.api_key_env, "").strip()
    if not key:
        raise SystemExit(f"{args.api_key_env} is not set")
    client = OpenAI(api_key=key, base_url=args.base_url)
    community = "Reddit camera and photography communities"
    seed_title = " ".join(str(post.get("title") or "").split())

    by_id: dict[int, str] = {}
    prior: list[dict] = []
    out: list[dict] = []
    for position, record in enumerate(records, start=1):
        task = record.get("task") or {}
        parent_id = task.get("local_parent_task_id")
        prompt = slim_prompt(
            task,
            community=community,
            seed_title=seed_title,
            parent_text=by_id.get(int(parent_id), "") if parent_id else "",
            prior=prior,
        )
        words = int(task.get("real_word_count") or 0)
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max(260, int(words * 1.7) + 64),
        )
        text = (response.choices[0].message.content or "").strip()
        task_id = int(task.get("local_task_id") or position)
        by_id[task_id] = text
        prior.append({"text": text, "semantic_move": task.get("semantic_move")})
        out.append(
            {
                "local_task_id": task_id,
                "local_parent_task_id": parent_id,
                "depth": int(task.get("depth") or 0),
                "real_word_count": words,
                "text": text,
                "prompt_chars": len(prompt),
            }
        )
        print(f"  slot {position}/{len(records)} ({len(text.split())}w)", flush=True)
    return out


def write_discussion(path: Path, post: dict, rows: list[dict], label: str) -> Path:
    """Write the variant in the schema the evaluation scorers already read."""

    path.mkdir(parents=True, exist_ok=True)
    comments = [
        {
            "comment_id": str(600000 + row["local_task_id"]),
            "parent_comment_id": (
                str(600000 + int(row["local_parent_task_id"]))
                if row["local_parent_task_id"]
                else None
            ),
            "author": f"slim_user_{row['local_task_id']}",
            "content": row["text"],
            "depth": row["depth"],
            "replies": [],
        }
        for row in rows
        if row["text"].strip()
    ]
    payload = {
        "meta": {"run_id": f"slim_prompt_ab_{label}", "generator": "slim_prompt_ab"},
        "posts": [
            {
                "post_id": post.get("post_id"),
                "title": post.get("title"),
                "content": post.get("content"),
                "author": post.get("author"),
                "comments": comments,
            }
        ],
    }
    (path / "discussion.json").write_text(json.dumps(payload, indent=1))
    return path


def score(directory: Path, *, device: str, skip_bertscore: bool) -> dict[str, float]:
    scores: dict[str, float] = {}
    bleu_out = directory / "self_bleu_results.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/evaluation/score_thread_self_bleu.py"),
            str(directory),
            "--target-kind",
            "generated",
            "--output-file",
            str(bleu_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    bleu = json.load(open(bleu_out))
    threads = bleu.get("threads") or []
    if threads:
        scores["self_bleu_4"] = float(threads[0]["self_bleu_4"])
    if skip_bertscore:
        return scores
    bert_out = directory / "self_bertscore_results.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/evaluation/score_thread_self_bertscore.py"),
            str(directory),
            "--target-kind",
            "generated",
            "--device",
            device,
            "--output-file",
            str(bert_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    bert = json.load(open(bert_out))
    threads = bert.get("threads") or []
    if threads:
        # The scorer's own field is `mean_bert_f1`; `self_bertscore_mean_f1` is
        # the column name the downstream CSV export uses.
        scores["self_bertscore_mean_f1"] = float(threads[0]["mean_bert_f1"])
    return scores


def register(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {}
    counts = [len(t.split()) for t in texts]
    return {
        "warm_pct": mean(1 if WARM.search(t) else 0 for t in texts) * 100,
        "frame_pct": mean(1 if FRAME.search(t) else 0 for t in texts) * 100,
        "mean_words": mean(counts),
        "max_words": max(counts),
    }


def main() -> None:
    args = parse_args()
    post, records = load_thread(args.tag, args.thread_index)
    out_root = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT
        / "artifacts"
        / "experiments"
        / "slim_prompt_ab"
        / f"{args.tag}_run{args.thread_index:02d}"
    )
    print(f"thread run_{args.thread_index:02d}: {len(records)} slots")
    print(f"shipped prompt mean: {mean(len(r['prompt']) for r in records if r.get('prompt')):.0f} chars")

    shipped_rows = [
        {
            "local_task_id": int((r.get("task") or {}).get("local_task_id") or 0),
            "local_parent_task_id": (r.get("task") or {}).get("local_parent_task_id"),
            "depth": int((r.get("task") or {}).get("depth") or 0),
            "real_word_count": int((r.get("task") or {}).get("real_word_count") or 0),
            "text": str((r.get("comment") or {}).get("content") or ""),
            "prompt_chars": len(r.get("prompt") or ""),
        }
        for r in records
    ]

    # Persist the generation before anything else can fail. A first version
    # scored the shipped variant first, the scorer raised on a field name, and
    # 38 paid completions that only existed in memory were lost.
    cache = out_root / "slim_generation.json"
    if cache.exists():
        slim_rows = json.loads(cache.read_text())
        print(f"\nreusing the cached slim generation: {cache}")
    else:
        print("\ngenerating the slim variant in slot order")
        slim_rows = generate(args, post, records)
        out_root.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(slim_rows, indent=1))
        print(f"cached the generation to {cache}")
    print(f"slim prompt mean: {mean(r['prompt_chars'] for r in slim_rows):.0f} chars")

    results = {}
    for label, rows in (("shipped", shipped_rows), ("slim", slim_rows)):
        directory = write_discussion(out_root / label, post, rows, label)
        metrics = score(directory, device=args.device, skip_bertscore=args.skip_bertscore)
        metrics.update(register([r["text"] for r in rows if r["text"].strip()]))
        metrics["prompt_chars"] = mean(r["prompt_chars"] for r in rows)
        results[label] = metrics

    real_words = [r["real_word_count"] for r in shipped_rows if r["real_word_count"]]
    keys = [
        "prompt_chars",
        "self_bleu_4",
        "self_bertscore_mean_f1",
        "warm_pct",
        "frame_pct",
        "mean_words",
        "max_words",
    ]
    print(f"\n{'metric':<26}{'shipped':>12}{'slim':>12}")
    print("-" * 50)
    for key in keys:
        if key in results["shipped"] and key in results["slim"]:
            print(f"{key:<26}{results['shipped'][key]:>12.4f}{results['slim'][key]:>12.4f}")
    if real_words:
        print(f"\nmatched real slot words: mean={mean(real_words):.1f} max={max(real_words)}")
    print("\nLower self_bleu_4 and self_bertscore_mean_f1 are closer to real.")
    print("A slim variant that does not raise them means the long ledgers are not")
    print("what is buying within-thread diversity.")

    (out_root / "summary.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out_root}")


if __name__ == "__main__":
    main()
