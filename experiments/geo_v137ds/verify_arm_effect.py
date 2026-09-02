#!/usr/bin/env python3
"""Did an arm actually change the artifact, or is the run inert?

  python3 experiments/geo_v137ds/verify_arm_effect.py v152probe_20260902 a5dsfit_20260902

Reads the properties each arm claims to move, off the generated run itself
rather than off run_config. `run_config` records what was REQUESTED; three arms
in this project recorded a flag and generated nothing new (v143obs's inert
share parameters, the six arms that never crossed the subprocess boundary, the
canonicalizer that would have folded every named lens back to `seed_local`).
The only proof an arm ran is a measured difference in what it produced.
"""
import collections
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MARKER = re.compile(r'persona-id="([^"]+)"')
INTENT_FALLBACK = "one seed-grounded local move"
FROZEN_LENS = re.compile(r"^P\d\d$")


def load(prefix):
    out = []
    for d in sorted((REPO / "artifacts/generalized_card/runs").glob(f"{prefix}_p*")):
        f = d / "generated/run_00_sampled_reddit/generation_records.json"
        if f.exists():
            out.append((d.name, json.load(open(f))))
    return out


def _run_persona_config(prefix):
    for d in sorted((REPO / "artifacts/generalized_card/runs").glob(f"{prefix}_p*")):
        f = d / "run_config.json"
        if not f.exists():
            continue
        raw = json.load(open(f))
        cfg = dict(raw.get("persona_conditioning") or {})
        if not cfg.get("mode") or cfg.get("mode") == "none":
            continue
        # Runs before the setters were moved ahead of `build_runtime` recorded
        # the shipped defaults inside `persona_conditioning` while the top-level
        # fields held what was actually requested. Prefer the top level, which
        # is written straight from argv.
        for key, top in (("projection", "persona_projection"), ("draw", "persona_draw")):
            if raw.get(top):
                if cfg.get(key) and cfg[key] != raw[top]:
                    print(f"  警告: run_config 内外不一致 ({key}: "
                          f"{cfg[key]} vs {raw[top]})；按顶层的 {raw[top]} 重建")
                cfg[key] = raw[top]
        return cfg
    return {}


def _projected(cfg, persona_ids):
    """The dimensions the projection actually kept, per persona."""
    if not cfg or not persona_ids:
        return {}
    sys.path.insert(0, str(REPO / "generalized_card"))
    import generalized_card.persona_bridge as PB

    runtime = _runtime_for(cfg)
    out = {}
    for pid in persona_ids:
        persona = runtime._personas_by_id.get(pid)
        if persona is None:
            continue
        out[pid] = PB._project_dimensions(
            persona.dimensions,
            expertise_dimensions=runtime.expertise_dimensions,
            projection=runtime.projection,
        )
    return out


def _runtime_for(cfg):
    sys.path.insert(0, str(REPO / "generalized_card"))
    import generalized_card.persona_bridge as PB

    PB.set_persona_projection(cfg.get("projection", "default"))
    PB.set_persona_draw(cfg.get("draw", "replace"))
    return PB.build_runtime(
        mode=cfg["mode"],
        matraix_root=Path(cfg["matraix_root"]),
        dataset_dir=Path(cfg["dataset_dir"]),
        assignment_seed=int(cfg.get("assignment_seed", 42)),
        expertise_dimensions=tuple(cfg.get("expertise_dimensions") or ()),
    )


def _render(cfg, persona_ids):
    """Render each recorded persona_id exactly as the run's own config did."""
    if not cfg or not persona_ids:
        return {}
    runtime = _runtime_for(cfg)
    out = {}
    for pid in persona_ids:
        try:
            out[pid] = runtime.assignment_for_id(pid).system_prompt
        except KeyError:
            pass
    return out


def report(prefix):
    runs = load(prefix)
    if not runs:
        print(f"\n{prefix}: 没有生成产物")
        return
    tasks, prompts, personas = [], [], []
    for _, recs in runs:
        for r in recs:
            tasks.append(r["task"])
            prompts.append(r.get("prompt") or "")
            m = MARKER.search(r.get("prompt") or "")
            if m:
                personas.append(m.group(1))
    n = len(tasks)
    print(f"\n{'='*70}\n{prefix}   {len(runs)} thread, {n} slot\n{'='*70}")

    def share(pred):
        return sum(1 for t in tasks if pred(t)) / max(n, 1)

    reply = [t for t in tasks if t.get("local_parent_task_id") not in (None, "")]
    root = [t for t in tasks if t.get("local_parent_task_id") in (None, "")]
    print("--- plan_vocabulary open 应当移动的三项 ---")
    print(f"  perspective_id = seed_local        {share(lambda t: str(t.get('perspective_id'))=='seed_local'):>6.0%}"
          f"   (顶层 {sum(1 for t in root if str(t.get('perspective_id'))=='seed_local')/max(len(root),1):.0%}"
          f" / reply {sum(1 for t in reply if str(t.get('perspective_id'))=='seed_local')/max(len(reply),1):.0%})")
    print(f"  perspective_id 是冻结的 P##         {share(lambda t: bool(FROZEN_LENS.match(str(t.get('perspective_id') or '')))):>6.0%}")
    print(f"  content_angle = unclear_mixed      {share(lambda t: str(t.get('content_angle'))=='unclear_mixed'):>6.0%}"
          f"   (顶层 {sum(1 for t in root if str(t.get('content_angle'))=='unclear_mixed')/max(len(root),1):.0%}"
          f" / reply {sum(1 for t in reply if str(t.get('content_angle'))=='unclear_mixed')/max(len(reply),1):.0%})")
    print(f"  domain_intent 停在兜底串            {share(lambda t: str(t.get('domain_intent') or '').strip()==INTENT_FALLBACK):>6.0%}"
          f"   (顶层 {sum(1 for t in root if str(t.get('domain_intent') or '').strip()==INTENT_FALLBACK)/max(len(root),1):.0%}"
          f" / reply {sum(1 for t in reply if str(t.get('domain_intent') or '').strip()==INTENT_FALLBACK)/max(len(reply),1):.0%})")
    lens = collections.Counter(str(t.get("perspective_id") or "") for t in tasks)
    print(f"  不同 lens 数                        {len(lens):>6}   最高频占 {lens.most_common(1)[0][1]/max(n,1):.0%}")
    for v, c in lens.most_common(6):
        print(f"      {c:>4}  {v[:60]}")

    print("\n--- persona 两个 arm 应当移动的项 ---")
    if not personas:
        print("  (没有 persona marker)")
    else:
        print(f"  不同 persona_id                    {len(set(personas)):>6} / {len(personas)} 个 slot"
              f"  ({len(set(personas))/len(personas):.0%})")
        # The identity itself is NOT in the recorded prompt. `inject_persona_system`
        # moves the marker into a system message at call time and that message is
        # not persisted, so the only way to see what the Writer was told is to
        # render the recorded persona_id through the same runtime. Grepping the
        # stored prompt reports 0% for a run where every slot carried an
        # identity -- which is what a first version of this script did.
        cfg = _run_persona_config(prefix)
        rendered = _render(cfg, set(personas))
        if rendered:
            axes = ("english_proficiency", "multilingualism", "neurotype",
                    "skill_writing", "skill_storytelling")
            # Check the PROJECTION, not the rendered text. The official
            # template relabels fields -- `skill_writing` renders as
            # "Skill: Writing", `tone_expected` as "Expected tone" -- so
            # grepping `axis.replace("_", " ")` reports 0% for axes that are
            # present on 284 and 192 of 400 personas. Guessing a field's
            # display form is how this script twice measured its own regex
            # instead of the artifact.
            projected = _projected(cfg, set(personas))
            hit = collections.Counter()
            for pid in personas:
                dims = projected.get(pid) or {}
                for axis in axes:
                    if dims.get(axis):
                        hit[axis] += 1
            print(f"  writer 收到的身份里含语域轴 (按 slot):")
            for axis in axes:
                print(f"      {axis:<22}{hit[axis]/len(personas):>6.0%}")
            print(f"  不同身份 / slot                    "
                  f"{len({rendered.get(p,p) for p in personas})}/{len(personas)}"
                  f"  ({len({rendered.get(p,p) for p in personas})/len(personas):.0%})")
        else:
            print("  (无法重建身份：run_config 没有 persona 配置)")


for prefix in sys.argv[1:] or ["v152probe_20260902"]:
    report(prefix)
