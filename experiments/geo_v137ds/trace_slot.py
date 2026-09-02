#!/usr/bin/env python3
"""Follow one thread from real comment to plan to Writer instruction to output.

  python3 experiments/geo_v137ds/trace_slot.py real_20260902 --slots 6

Each stage is judged by different evidence and they fail differently, so reading
them separately is the only way to tell whose fault a thread is. Prints, per
slot: the real comment that slot was matched to, the plan's own account of what
the comment should do, the instruction the Writer actually received, and what
the Writer returned.
"""
import argparse, glob, json, textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("--slots", type=int, default=6)
ap.add_argument("--tag", default="")
a = ap.parse_args()

files = sorted(glob.glob(str(REPO / "artifacts/generalized_card/runs/*/generated/run_00_sampled_reddit/discussion.json")))
files = [f for f in files if f.split("/")[-4].startswith(a.prefix)]
if a.tag:
    files = [f for f in files if a.tag in f]
if not files:
    raise SystemExit(f"{a.prefix}: 没有数据")

# Prefer a thread big enough that siblings exist to be alike or unalike.
best = max(files, key=lambda f: len(json.load(open(f))["posts"][0].get("generation_records") or []))
post = json.load(open(best))["posts"][0]
print(f"tag = {best.split('/')[-4]}   seed = {post.get('source_raw_post_id')}")
print(f"帖子: {str(post.get('title'))[:96]}\n")

w = lambda t, ind: textwrap.fill(str(t), 108, initial_indent=ind, subsequent_indent=ind + "  ")
for i, rec in enumerate((post.get("generation_records") or [])[: a.slots], 1):
    task = rec.get("task") or {}
    comment = rec.get("comment")
    if not isinstance(comment, dict):
        continue
    print("=" * 110)
    print(f"槽位 {i}   深度={task.get('depth')}  真实词数={task.get('real_word_count')}  "
          f"perspective={task.get('perspective_id')}  content_angle={task.get('content_angle')}")
    print("-" * 110)
    print("① Planner 计划这条评论做什么")
    for k in ("semantic_move", "local_topic", "detail_focus", "avoid_repeating"):
        v = str(task.get(k) or "").strip()
        if v:
            print(w(f"{k}: {v}", "     "))
    print(f"     标签: payload={task.get('payload_type')}  function={task.get('comment_function')}  "
          f"tone={task.get('tone_target')}  affect={task.get('affect_role')}  stance={task.get('stance')}")
    print("\n② Writer 收到的命令")
    pr = str(rec.get("prompt") or "")
    seg = pr.split("Visible discussion:")[0]
    for ln in seg.split("\n"):
        if ln.strip().startswith("-") or ln.strip().startswith("function:"):
            print(w(ln.strip(), "     "))
    print("\n③ Writer 写出来的")
    print(w(str(comment.get("content") or "").strip(), "     "))
    print()
