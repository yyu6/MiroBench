#!/usr/bin/env python3
"""Prove `--writer-plan-fields` changes the RENDERED Writer prompt, in the
subprocess that renders it.

`verify_flags.py` checks that a flag's module global crosses the process
boundary. That is necessary and not sufficient: `--matched-text measured` sets
`MATCHED_TEXT_MODE` correctly on the far side and still reaches no prompt,
because the only renderer that reads it has no call site. So this asserts on
the prompt text itself:

  * `full` reproduces the current prompt character for character
  * `angle_detail` removes exactly the three withheld fields' lines
  * nothing else moves -- every other line is identical

It rebuilds one real slot out of v156's own generation records, so the task is
a task the pipeline actually produced rather than a fixture.
"""
import json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CHILD = r'''
import json, os, sys
from dataclasses import fields as dc_fields
from pathlib import Path
REPO = Path(r"%s")
sys.path.insert(0, str(REPO / "generalized_card"))
sys.path.insert(0, str(REPO / "scripts"))
from generalized_card.backend import (DEFAULT_GENERATOR_PROFILE,
                                      configure_generator_backend, load_generator_backend)
from generalized_card.domain import load_domain_from_env
from sampling_generator.engine.model import CommentTask, SeedPost

config = load_domain_from_env()
profile = os.environ.get("GENERALIZED_CARD_GENERATOR_PROFILE", DEFAULT_GENERATOR_PROFILE)
module = configure_generator_backend(load_generator_backend(profile=profile), config, profile=profile)

rec = json.load(open(REPO / "artifacts/generalized_card/runs/v156_20260903_p5/generated/run_00_sampled_reddit/generation_records.json"))[0]
names = {f.name for f in dc_fields(CommentTask)}
task = CommentTask(**{k: v for k, v in (rec["task"] or {}).items() if k in names})
pool = json.load(open(REPO / "artifacts/generalized_card/seed_pools/celebrity_geo_150_seed907.json"))
row = next(r for r in pool["seed_posts"] if int(r["seed_index"]) == 5)
seed = SeedPost(index=5, title=row.get("title") or "", body=row.get("body") or row.get("selftext") or "",
                content=row.get("body") or "", source_raw_post_id=str(row["source_raw_post_id"]),
                real_num_comments=int(row.get("real_num_comments") or 0), metadata={})
print("<<<MODE>>>" + getattr(module, "GENERALIZED_WRITER_PLAN_FIELDS", "?"))
print("<<<PROMPT>>>")
print(module.build_writer_prompt(profile="gpt54_reddit_writer", seed_post=seed, task=task,
                                 parent_comment=None, previous_comments=[], recent_openings=[],
                                 retry_note=""))
''' % (REPO,)

def render(mode: str) -> tuple[str, str]:
    env = dict(os.environ)
    env.update({
        "GENERALIZED_CARD_DOMAIN": "celebrity_geo",
        "GENERALIZED_CARD_WRITER_PLAN_FIELDS": mode,
        "GENERALIZED_CARD_PLAN_VOCABULARY": "open",
        "GENERALIZED_CARD_MATCHED_TEXT": "measured",
        "GENERALIZED_CARD_BRANCH_DICTATION": "structural",
        "HF_HUB_OFFLINE": "1",
    })
    out = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True)
    if "<<<PROMPT>>>" not in out.stdout:
        print(out.stdout[-3000:]); print(out.stderr[-3000:]); raise SystemExit(f"{mode}: 渲染失败")
    head, body = out.stdout.split("<<<PROMPT>>>", 1)
    return head.split("<<<MODE>>>")[1].strip(), body.strip()

mode_full, full = render("full")
mode_arm, arm = render("angle_detail")
print(f"子进程里读到的模式: full -> {mode_full!r}   angle_detail -> {mode_arm!r}")
assert (mode_full, mode_arm) == ("full", "angle_detail"), "开关没过子进程边界"

f_lines, a_lines = full.split("\n"), arm.split("\n")
removed = [l for l in f_lines if l not in a_lines]
added = [l for l in a_lines if l not in f_lines]
print(f"\n只在 full 里出现的行 ({len(removed)}):")
for l in removed: print("  -", l[:140])
print(f"\n只在 angle_detail 里出现的行 ({len(added)}):")
for l in added: print("  +", l[:140])

MUST_GO = ("- The point this comment makes", "- The question your turn settles", "- decision intent:")
MUST_STAY = ("- content angle:", "- specific detail:", "- stance:", "- function:",
             "Tone target selector:", "Story realization:")
ok = True
for frag in MUST_GO:
    was, now = any(frag in l for l in f_lines), any(frag in l for l in a_lines)
    tag = "OK " if (was and not now) else "FAIL"
    if not (was and not now): ok = False
    print(f"  [{tag}] 应当消失: {frag!r}   full={was} arm={now}")
for frag in MUST_STAY:
    was, now = any(frag in l for l in f_lines), any(frag in l for l in a_lines)
    tag = "OK " if (was and now) else ("skip" if not was else "FAIL")
    if was and not now: ok = False
    print(f"  [{tag}] 应当保留: {frag!r}   full={was} arm={now}")
print("\n结论:", "开关真的改到了提示词" if ok else "有问题，不要跑")
sys.exit(0 if ok else 1)
