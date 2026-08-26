#!/usr/bin/env python3
"""Do DECODING knobs move `self_bertscore` / `self_bleu_4` at all? Zero API cost.

Why this test exists. G28 (`prompt_convergence_diagnosis.py`, v108) established
that the Writer's *inputs* already separate -- prompt line-Jaccard falls
0.3516 -> 0.2481 with depth while realized text similarity stays flat
(0.0971 / 0.0865 / 0.0917 / 0.0969 / 0.0818), r(prompt, text) = 0.320 -- and
concluded the convergence is produced *inside* the Writer, downstream of every
lever the project pulls. Every mechanism tried since has been input-side, which
is why they keep failing.

That rejection covers **input** levers. It says nothing about the **sampler**,
which sits downstream of the prompt -- exactly where G28 localises the defect.
And the sampler is, right now, nearly constant across slots
(`scripts/sampling_generator/engine/writer_request.py`):

    length_bucket in {micro, short}                 -> 0.88
    comment_function in {offtopic_noise, reaction}  -> 0.95
    else                                            -> 0.82   <- 69.7% of v119 slots

Three temperatures, keyed on length and function, **blind to depth** -- while
G3/G26 put the entire defect in the reply population, depth bins [2,4) and
[4,7) carrying 82.7% of it. `top_p`, `frequency_penalty` and `presence_penalty`
are never set on the API path at all (`writer_extra_body` returns None for every
non-local profile).

**This script does not measure how much a knob is worth.** It runs the run's own
saved Writer prompts back through a LOCAL model at several decoding settings and
asks whether the two priority metrics respond, and in which direction. The local
model is not `gpt-5.4-mini`, so no magnitude here transfers. What transfers is
existence and sign: if a knob cannot move pairwise similarity on any model, it is
not a candidate; if it moves it cleanly, it earns one paid arm.

Every setting reads the SAME prompts, so the prompt side is held exactly fixed
and only the sampler varies -- the contrast G28's observational r=0.320 could
not make.

Two interpreters, on purpose:

    # 1. generate -- needs MLX
    .venv_mlx_qwen/bin/python .../decoding_diversity_probe.py generate --out probe.json

    # 2. score -- needs transformers 4.48.0 for BERTScore parity with the artifact
    python3 .../decoding_diversity_probe.py score --in probe.json

Scoring uses the project's own scorers (`score_thread_self_bertscore`,
`score_thread_self_bleu`), not a reimplementation.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "artifacts/generalized_card/runs"
DEFAULT_RUN = "v119_tonequota_only_n10_20260827_v1"
DEFAULT_MODEL_PATH = "mlx-community/Qwen3-8B-4bit"

# Two mid-sized threads: enough comments for a stable pairwise mean, small
# enough that a seven-setting sweep finishes on a laptop.
DEFAULT_THREADS = ("sampled_run00_post00_seed002", "sampled_run00_post01_seed003")

# `base` reproduces the production sampler for the 69.7% majority bucket.
# The rest move exactly one knob each, so any response is attributable.
SETTINGS: dict[str, dict[str, float]] = {
    "base_T0.82":      {"temp": 0.82, "top_p": 1.0},
    "temp_T1.00":      {"temp": 1.00, "top_p": 1.0},
    "temp_T1.15":      {"temp": 1.15, "top_p": 1.0},
    "topp_0.85":       {"temp": 0.82, "top_p": 0.85},
    "freqpen_0.6":     {"temp": 0.82, "top_p": 1.0, "frequency_penalty": 0.6},
    "freqpen_1.2":     {"temp": 0.82, "top_p": 1.0, "frequency_penalty": 1.2},
    "prespen_0.6":     {"temp": 0.82, "top_p": 1.0, "presence_penalty": 0.6},
}


def max_tokens_for_length(bucket: str) -> int:
    """Mirrors `scripts/sampling_generator/engine/writer_request.py`."""
    return {"micro": 32, "short": 48, "long": 170, "very_long": 300}.get(bucket, 110)


def load_slots(run: str, threads: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    """The run's own saved Writer prompts, grouped by thread."""
    out: dict[str, list[dict[str, Any]]] = {t: [] for t in threads}
    for path in sorted((RUNS / run).glob("cleaned/run_*_sampled_reddit/generation_records.json")):
        for rec in json.loads(path.read_text(encoding="utf-8")):
            tid = str(rec.get("post_id") or "")
            if tid not in out:
                continue
            task = rec.get("task") or {}
            prompt = rec.get("prompt")
            if not prompt:
                continue
            out[tid].append(
                {
                    "comment_id": str(task.get("comment_id") or task.get("local_task_id") or len(out[tid])),
                    "prompt": prompt,
                    "length_bucket": str(task.get("length_bucket") or "medium"),
                    "depth": int(task.get("depth") or 0),
                    "shipped": str(rec.get("raw") or ""),
                }
            )
    return {t: rows for t, rows in out.items() if rows}


# --------------------------------------------------------------------------- #
# stage 1 -- generate (MLX)
# --------------------------------------------------------------------------- #


def cmd_generate(args: argparse.Namespace) -> None:
    from mlx_lm import generate, load  # noqa: PLC0415
    from mlx_lm.sample_utils import make_logits_processors, make_sampler  # noqa: PLC0415

    slots = load_slots(args.run, tuple(args.threads))
    total = sum(len(v) for v in slots.values())
    print(f"loading {args.model} ...", flush=True)
    model, tokenizer = load(args.model)
    print(f"threads={len(slots)} slots={total} settings={len(SETTINGS)}", flush=True)

    results: dict[str, Any] = {
        "meta": {
            "run": args.run,
            "model": args.model,
            "threads": list(slots),
            "settings": SETTINGS,
            "note": "local model; magnitudes do not transfer to gpt-5.4-mini",
        },
        "shipped": {t: [r["shipped"] for r in rows] for t, rows in slots.items()},
        "generated": {},
    }
    for name, cfg in SETTINGS.items():
        sampler = make_sampler(temp=cfg["temp"], top_p=cfg.get("top_p", 1.0))
        processors = make_logits_processors(
            frequency_penalty=cfg.get("frequency_penalty"),
            presence_penalty=cfg.get("presence_penalty"),
        )
        per_thread: dict[str, list[str]] = {}
        done = 0
        for tid, rows in slots.items():
            texts = []
            for row in rows:
                messages = [{"role": "user", "content": row["prompt"]}]
                chat = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,  # Qwen3 reasoning would dominate the budget
                )
                text = generate(
                    model, tokenizer, prompt=chat, verbose=False,
                    max_tokens=max_tokens_for_length(row["length_bucket"]),
                    sampler=sampler, logits_processors=processors,
                )
                texts.append(strip_thinking(text).strip())
                done += 1
            per_thread[tid] = texts
            print(f"  {name}: {tid} {done}/{total}", flush=True)
        results["generated"][name] = per_thread
        Path(args.out).write_text(json.dumps(results), encoding="utf-8")
    print(f"wrote {args.out}")


def strip_thinking(text: str) -> str:
    while "<think>" in text and "</think>" in text:
        a, b = text.find("<think>"), text.find("</think>")
        if a < 0 or b < a:
            break
        text = (text[:a] + text[b + len("</think>"):]).strip()
    return text.replace("<think>", "").replace("</think>", "").strip()


# --------------------------------------------------------------------------- #
# stage 2 -- score (system python3, transformers 4.48.0)
# --------------------------------------------------------------------------- #


def cmd_score(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO / "scripts" / "evaluation"))
    from score_thread_self_bertscore import (  # noqa: PLC0415
        DEFAULT_BERT_SCORE_PATH, DEFAULT_MODEL, load_bert_scorer,
        score_pairs_with_device_fallback,
    )
    from score_thread_self_bleu import pairwise_self_bleu_for_order, tokenize  # noqa: PLC0415

    data = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    arms: dict[str, dict[str, list[str]]] = dict(data["generated"])
    arms["SHIPPED_gpt5.4mini"] = data["shipped"]

    class C:
        """Minimal stand-in for ThreadComment. The scorer reads `comment_id` and
        `author` onto every pair row, so both must exist, not just `text`."""

        __slots__ = ("text", "comment_id", "author", "thread_id", "parent_id", "depth")

        def __init__(self, text: str, comment_id: str) -> None:
            self.text = text
            self.comment_id = comment_id
            self.author = ""
            self.thread_id = "t"
            self.parent_id = ""
            self.depth = 0

    def bert_mean(texts: list[str]) -> float:
        nodes = [C(t, str(i)) for i, t in enumerate(texts)]
        specs = [
            {"thread_id": "t", "left": nodes[i], "right": nodes[j]}
            for i in range(len(nodes)) for j in range(i + 1, len(nodes))
        ]
        idf = [s["left"].text for s in specs] + [s["right"].text for s in specs]
        kw = dict(bert_score_path=DEFAULT_BERT_SCORE_PATH, model_type=DEFAULT_MODEL,
                  num_layers=None, idf=False, idf_sents=idf,
                  rescale_with_baseline=False, local_files_only=False)
        (scorer, *_r, fb) = load_bert_scorer(batch_size=args.batch_size, device=args.device, **kw)
        if fb:
            raise SystemExit("deberta-xlarge-mnli did not load; refusing a fallback model.")
        (pairs, *_r2) = score_pairs_with_device_fallback(
            scorer=scorer, pair_specs=specs, batch_size=args.batch_size,
            requested_device=args.device, fallback_used=False, **kw)
        return st.mean([p["bert_f1"] for p in pairs])

    print(f"{'setting':<22}{'thread':<32}{'n':>4}{'selfbert':>10}{'selfbleu4':>11}{'words':>7}")
    summary: dict[str, dict[str, list[float]]] = {}
    for name, per_thread in arms.items():
        summary[name] = {"bert": [], "bleu": [], "words": []}
        for tid, texts in per_thread.items():
            texts = [t for t in texts if t.strip()]
            if len(texts) < 2:
                continue
            b = bert_mean(texts)
            bl = pairwise_self_bleu_for_order([tokenize(t) for t in texts], 4)
            w = st.mean([len(t.split()) for t in texts])
            summary[name]["bert"].append(b)
            summary[name]["bleu"].append(bl)
            summary[name]["words"].append(w)
            print(f"{name:<22}{tid:<32}{len(texts):>4}{b:>10.4f}{bl:>11.4f}{w:>7.1f}")

    base = summary.get("base_T0.82")
    print(f"\n{'setting':<22}{'selfbert':>10}{'vs base':>10}{'selfbleu4':>11}{'vs base':>10}{'words':>8}")
    for name, s in summary.items():
        if not s["bert"]:
            continue
        b, bl, w = st.mean(s["bert"]), st.mean(s["bleu"]), st.mean(s["words"])
        db = f"{b - st.mean(base['bert']):+.4f}" if base and base["bert"] else "--"
        dl = f"{bl - st.mean(base['bleu']):+.4f}" if base and base["bleu"] else "--"
        print(f"{name:<22}{b:>10.4f}{db:>10}{bl:>11.4f}{dl:>10}{w:>8.1f}")
    print("\nSign and existence only. The local model is not gpt-5.4-mini, so no")
    print("magnitude here transfers; SHIPPED is listed for scale, not as a control.")
    print("Word count is printed because a knob that only shortens comments would")
    print("move both metrics for a reason that is not diversity.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--run", default=DEFAULT_RUN)
    g.add_argument("--model", default=DEFAULT_MODEL_PATH)
    g.add_argument("--threads", nargs="+", default=list(DEFAULT_THREADS))
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_generate)
    s = sub.add_parser("score")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--device", default="cpu")
    s.add_argument("--batch-size", type=int, default=16)
    s.set_defaults(func=cmd_score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
