#!/usr/bin/env python3
"""Does persona selection actually fit the slot, and hold across an author?

  python3 experiments/geo_v137ds/check_persona_fit.py

Under `--speaker-identity matched` every slot has a speaker, and the previous
selection path answered that by drawing from the whole eligible population with
no compatibility scoring at all -- so `_compatibility_score`, which matches a
persona's expertise, trust level and emotional state against the slot's speaker
role and tone, never ran. This checks the two properties that were in tension:
whether the chosen persona scores above an average one, and whether an author
holding several slots keeps a single persona.
"""
import os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "generalized_card"))
os.environ.setdefault("GENERALIZED_CARD_DOMAIN", "celebrity_geo")

from generalized_card.domain import load_domain_from_env  # noqa: E402
from generalized_card import persona_bridge as pb  # noqa: E402

cfg = load_domain_from_env()
rt = pb.MatraixPersonaRuntime(
    mode=pb.MODE_PROJECTED,
    matraix_root=REPO / "third_party" / "MatrAIx-Persona-8B",
    dataset_dir=REPO / "third_party" / "MatrAIx-Persona-8B" / "persona" / "datasets" / "matraix-persona-dev-sample",
    assignment_seed=42,
    expertise_dimensions=cfg.persona_expertise_dimensions,
)
if not getattr(rt, "enabled", False):
    raise SystemExit("persona runtime 未启用")

pool = list(rt._eligible)
print(f"eligible personas: {len(pool)}\n")

ROLES = ["advisor", "confused_asker", "jokester", "ranter", "contrarian",
         "datapoint_only", "gratitude_reply", "side_observer"]
print(f"{'slot 角色':<18}{'选中得分':>9}{'全体均分':>9}{'最高分':>8}{'高于均值':>9}")
import statistics
for i, role in enumerate(ROLES):
    task = {"speaker_role": role, "voice": "blunt", "tone_target": "impolite",
            "local_task_id": i}
    a = rt.assign(seed_index=0, task=task, speaker_id=f"u{i}")
    chosen = rt._personas_by_id[a.persona_id]
    scores = [pb._compatibility_score(p.dimensions, task, rt.expertise_dimensions) for p in pool]
    cs = pb._compatibility_score(chosen.dimensions, task, rt.expertise_dimensions)
    print(f"  {role:<16}{cs:>9}{statistics.mean(scores):>9.2f}{max(scores):>8}"
          f"{'  是' if cs > statistics.mean(scores) else '  否':>9}")

print("\n同一作者持有多个槽位时是否保持同一 persona：")
ids = []
for k, role in enumerate(["ranter", "advisor", "jokester"]):
    t = {"speaker_role": role, "voice": "blunt", "tone_target": "impolite", "local_task_id": 90 + k}
    ids.append(rt.assign(seed_index=0, task=t, speaker_id="same_author").persona_id)
print(f"  三个不同角色的槽位 -> {len(set(ids))} 个 persona  ({'一致' if len(set(ids)) == 1 else '不一致'})")
