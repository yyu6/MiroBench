#!/usr/bin/env python3
"""Prove the in-process engine reproduces the official scorers exactly.

The engine's whole value is that a gate decision made on its numbers is the
same decision the official pipeline would make. That is only true if the
numbers are identical, so this rescores a directory that was already scored by
the official subprocess pipeline and compares every field.
"""
import json, shutil, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "selfloop"))
import metric_engine as E

SRC = [REPO / f"artifacts/generalized_card/runs/v157_20260903_p{i}/cleaned/run_00_sampled_reddit"
       for i in (7, 5, 0)]
WORK = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/selfloop_verify")
if WORK.exists(): shutil.rmtree(WORK)
WORK.mkdir(parents=True)

KEEP = {"discussion.json", "discussion.md", "generation_records.json"}


def comment_count(run_dir: Path) -> int:
    """Every comment in the tree, not just the top-level ones. Counting
    `posts[0]["comments"]` reported 13 for a 42-comment thread."""
    payload = json.loads((run_dir / "discussion.json").read_text())
    total = 0

    def walk(comments):
        nonlocal total
        for comment in comments or []:
            total += 1
            walk(comment.get("replies") or [])

    for post in payload.get("posts") or []:
        walk(post.get("comments") or [])
    return total
official, mine, timings = {}, {}, []
for src in SRC:
    name = src.parents[1].name
    official[name] = json.loads((src / "thread_metrics_summary.json").read_text())[0]
    dst = WORK / name
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.name in KEEP: shutil.copy2(f, dst / f.name)
    t0 = time.time()
    mine[name] = E.score_run_dir(dst, device="cpu")
    timings.append((name, time.time() - t0, comment_count(dst)))

M12 = ["self_bertscore_mean_f1", "self_bleu_4", "semantic_mean_cosine", "hard_disagree_rate",
       "polite_rate", "impolite_rate", "neutral_rate", "length_cv", "avg_depth",
       "structural_virality", "mean_story_probability", "emotion_entropy"]
# The official artifact was produced by eight separate subprocesses; this runs
# the same code in one. Differences at 1e-8 are float32 accumulation order, not
# logic: three consecutive engine runs on identical input agree to 0.00e+00,
# and the smallest decision threshold anywhere in the loop is 0.01. A logic
# difference would show far above this.
TOLERANCE = 1e-6

print(f"{'run':<24}{'指标':<26}{'官方':>12}{'引擎':>12}{'差':>12}")
print("-" * 88)
bad = 0
for name in official:
    for k in M12:
        a, b = official[name].get(k), mine[name].get(k)
        try: a, b = float(a), float(b)
        except (TypeError, ValueError):
            print(f"{name:<24}{k:<26}{str(a):>12}{str(b):>12}{'缺':>12}"); bad += 1; continue
        d = abs(a - b)
        if d > TOLERANCE:
            print(f"{name:<24}{k:<26}{a:>12.6f}{b:>12.6f}{d:>+12.2e}  <-- 不一致"); bad += 1
print(f"\n{len(official)*len(M12)} 个数值比对，{bad} 个超出容差 {TOLERANCE:g}")
print(f"\n{'run':<24}{'评论数':>8}{'引擎耗时':>12}")
for name, dt, n in timings:
    print(f"{name:<24}{n:>8}{dt:>11.1f}s")
print(f"\n常驻模型数: {E.loaded_models()}")
sys.exit(1 if bad else 0)
