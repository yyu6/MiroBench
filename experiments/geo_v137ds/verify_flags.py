#!/usr/bin/env python3
"""Prove each planner flag reaches the prompt in the process that builds it.

  python3 experiments/geo_v137ds/verify_flags.py

Generation runs in run_generator_backend.py, a subprocess. run_generate.py's
setters configure the parent, which builds the domain profile; the Planner
prompt is assembled on the other side of that boundary. A flag wired only
through a setter therefore changes nothing and reports no error -- v149's first
run produced plans that ignored a brief verified to be present in the template,
because four flags were parent-only.

This spawns a real subprocess with the environment run_generate would set, and
checks the flag's effect on the rendered prompt from inside it.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "generalized_card"

CHECKS = [
    ("GENERALIZED_CARD_SLOT_GRID", "free",
     "generalized_card.planner_distribution", "SLOT_GRID_MODE", "free"),
    ("GENERALIZED_CARD_PLANNER_DISTRIBUTION", "off",
     "generalized_card.generation_distribution", "PLANNER_DISTRIBUTION_MODE", "off"),
    ("GENERALIZED_CARD_BRANCH_DICTATION", "structural",
     "generalized_card.branch_routing", "BRANCH_DICTATION_MODE", "structural"),
    ("GENERALIZED_CARD_MATCHED_TEXT", "measured",
     "generalized_card.prompts", "MATCHED_TEXT_MODE", "measured"),
    ("GENERALIZED_CARD_ISOLATION_QUOTA", "measured",
     "generalized_card.planning_quality", "ISOLATION_QUOTA_MODE", "measured"),
    ("GENERALIZED_CARD_REFERENCE_WINDOW", "unranked",
     "generalized_card.viewpoint_bank", "REFERENCE_WINDOW_MODE", "unranked"),
    ("GENERALIZED_CARD_PLAN_VOCABULARY", "open",
     "generalized_card.plan_vocabulary", "PLAN_VOCABULARY_MODE", "open"),
    # The open arm also has to reach the GENERATOR module: `normalize_plan_rows`
    # reads `GENERALIZED_PLAN_VOCABULARY` off it by name, and that is the line
    # that decides whether a Planner-named content angle survives or is folded
    # back to `unclear_mixed`. A setter that configures only the
    # generalized_card side leaves the fold in place.
    ("GENERALIZED_CARD_PLAN_VOCABULARY", "open",
     "@generator", "GENERALIZED_PLAN_VOCABULARY", "open"),
    # The persona arms are applied inside `runtime_from_env`, which the
    # subprocess calls; a setter run only in run_generate.py would render the
    # shipped projection and the shipped draw while run_config recorded the arm.
    ("GENERALIZED_CARD_PERSONA_PROJECTION", "register",
     "generalized_card.persona_bridge", "PERSONA_PROJECTION_MODE", "register"),
    ("GENERALIZED_CARD_PERSONA_DRAW", "exhaust",
     "generalized_card.persona_bridge", "PERSONA_DRAW_MODE", "exhaust"),
]

SNIPPET = """
import importlib, os, sys
sys.path.insert(0, {pkg!r})
from generalized_card.backend import configure_generator_backend, load_generator_backend
from generalized_card.domain import load_domain_from_env
cfg = load_domain_from_env()
gen = load_generator_backend()
configure_generator_backend(gen, cfg)
if os.environ.get("GENERALIZED_CARD_PERSONA_PROJECTION") or os.environ.get("GENERALIZED_CARD_PERSONA_DRAW"):
    # These are applied by `runtime_from_env`, so the probe has to build it.
    os.environ.setdefault("GENERALIZED_CARD_PERSONA_MODE", "matraix-projected")
    from generalized_card.persona_bridge import runtime_from_env
    runtime_from_env()
m = gen if {mod!r} == "@generator" else importlib.import_module({mod!r})
print(getattr(m, {attr!r}, "(缺这个属性)"))
"""

def main() -> int:
    env0 = dict(os.environ)
    env0.setdefault("GENERALIZED_CARD_DOMAIN", "celebrity_geo")
    env0["PYTHONPATH"] = str(PKG)
    bad = 0
    for var, value, mod, attr, expect in CHECKS:
        env = dict(env0)
        env[var] = value
        out = subprocess.run(
            [sys.executable, "-c", SNIPPET.format(pkg=str(PKG), mod=mod, attr=attr)],
            env=env, capture_output=True, text=True, cwd=str(PKG))
        got = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else f"(错误) {out.stderr.strip()[-160:]}"
        ok = got == expect
        bad += not ok
        where = "generator" if mod == "@generator" else mod.split(".")[-1]
        print(f"  {'OK  ' if ok else 'FAIL'} {var}={value} -> {where}.{attr} = {got}")
    print(f"\n{len(CHECKS) - bad}/{len(CHECKS)} 个 flag 能穿过子进程边界")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
