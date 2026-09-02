"""The persona layer's three defects, each with the measurement that found it.

None of these produced an error. A projection that never names an axis renders
without it; a draw with replacement returns a valid persona that happens to be
someone else's; a provenance pass that replays instead of reading returns a
plausible persona_id that was never used. Only a test that asserts the measured
property catches any of them.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from generalized_card import persona_bridge as PB

REPO = Path(__file__).resolve().parents[2]
MATRAIX = REPO / "third_party" / "MatrAIx-Persona-8B"
DEV = MATRAIX / "persona" / "datasets" / "matraix-persona-dev-sample"
DIMS = ("fam_film_studies", "ind_entertainment")


class _Task(dict):
    def __getattr__(self, key):
        return self.get(key, "")


def _runtime(dataset: Path) -> PB.MatraixPersonaRuntime:
    return PB.build_runtime(
        mode="matraix-projected",
        matraix_root=MATRAIX,
        dataset_dir=dataset,
        assignment_seed=42,
        expertise_dimensions=DIMS,
    )


def _task(i: int) -> _Task:
    return _Task(
        speaker_role="side_observer",
        voice="casual_neutral",
        tone_shape="neutral_fact",
        local_task_id=i,
        comment_id=i,
    )


class ProjectionArmTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dims = {
            "register": "Colloquial",
            "english_proficiency": "Fluent (C1-C2)",
            "multilingualism": "Bilingual",
            "neurotype": "ADHD",
            "skill_writing": "Advanced",
            "skill_storytelling": "Intermediate",
            "tone_expected": "Blunt",
            "dominant_trait": "High curiosity",
            "emotional_state": "Frustrated",
            "tech_savviness": "High",
            "trust_level": "Verifying",
            "expertise_gap": "Peer-level",
            "decision_style": "Analytical",
            "risk_tolerance": "Risk-seeking",
            "intent": "Debate",
            "query_complexity": "Complex",
            "time_pressure": "None",
            "urbanicity": "Suburban",
            "socioeconomic_band": "Low income",
            "lstyle_commute_mode": "Bicycle",
            "lstyle_work_schedule": "Night shift",
            "trait_adaptability": "High",
        }

    def tearDown(self) -> None:
        PB.set_persona_projection("default")

    def test_default_mode_starves_the_register_axes(self) -> None:
        """The shipped behaviour, asserted so the arm's premise stays honest."""
        PB.set_persona_projection("default")
        out = PB._project_dimensions(self.dims, expertise_dimensions=DIMS)
        self.assertEqual(PB._MAX_PROJECTED_DIMENSIONS, len(out))
        for axis in (
            "english_proficiency",
            "multilingualism",
            "neurotype",
            "urbanicity",
            "socioeconomic_band",
        ):
            self.assertNotIn(axis, out)

    def test_register_mode_renders_the_writing_axes(self) -> None:
        PB.set_persona_projection("register")
        out = PB._project_dimensions(self.dims, expertise_dimensions=DIMS)
        for axis in (
            "register",
            "english_proficiency",
            "multilingualism",
            "neurotype",
            "skill_writing",
            "skill_storytelling",
            # The two the selector stratifies on. Without them the set is
            # spread along axes the Writer never sees.
            "urbanicity",
            "socioeconomic_band",
        ):
            self.assertIn(axis, out, f"{axis} must reach the Writer under register")

    def test_register_mode_spends_lifestyle_last(self) -> None:
        """Commute mode took 19 of 123 personas' ten slots and says nothing
        about how a comment is written."""
        PB.set_persona_projection("register")
        out = PB._project_dimensions(self.dims, expertise_dimensions=DIMS)
        self.assertNotIn("lstyle_commute_mode", out)
        self.assertNotIn("lstyle_work_schedule", out)

    def test_register_mode_stays_under_the_dilution_cap(self) -> None:
        """v67 failed by letting the identity outweigh the task."""
        PB.set_persona_projection("register")
        runtime = _runtime(DEV)
        longest = max(
            len(runtime.assignment_for_id(p.persona_id).system_prompt)
            for p in runtime._eligible
        )
        self.assertLess(longest, PB._MAX_SYSTEM_PROMPT_CHARS)


class DrawArmTest(unittest.TestCase):
    def tearDown(self) -> None:
        PB.set_persona_draw("replace")
        PB.set_persona_projection("default")

    def _draw(self, mode: str, speakers: int = 30) -> list[str]:
        PB.set_persona_draw(mode)
        runtime = _runtime(DEV)
        return [
            runtime.assign(
                seed_index=7, task=_task(i + 1), speaker_id=f"S{i + 1:03d}"
            ).persona_id
            for i in range(speakers)
        ]

    def test_replace_collides_and_exhaust_does_not(self) -> None:
        replace = self._draw("replace")
        exhaust = self._draw("exhaust")
        self.assertLess(len(set(replace)), 30, "replace must show the collision")
        self.assertEqual(30, len(set(exhaust)))

    def test_exhaust_is_deterministic_across_runtimes(self) -> None:
        self.assertEqual(self._draw("exhaust"), self._draw("exhaust"))

    def test_a_speaker_keeps_one_persona(self) -> None:
        PB.set_persona_draw("exhaust")
        runtime = _runtime(DEV)
        first = runtime.assign(seed_index=7, task=_task(1), speaker_id="S001")
        again = runtime.assign(seed_index=7, task=_task(50), speaker_id="S001")
        self.assertEqual(first.persona_id, again.persona_id)

    def test_threads_do_not_exhaust_each_other(self) -> None:
        PB.set_persona_draw("exhaust")
        runtime = _runtime(DEV)
        for seed in (7, 8):
            got = {
                runtime.assign(
                    seed_index=seed, task=_task(i + 1), speaker_id=f"S{i + 1:03d}"
                ).persona_id
                for i in range(30)
            }
            self.assertEqual(30, len(got), f"thread {seed} lost variety")

    def test_exhaust_falls_back_when_the_band_runs_out(self) -> None:
        """More speakers than candidates must repeat, not raise."""
        got = self._draw("exhaust", speakers=400)
        self.assertEqual(400, len(got))


class ProvenanceTest(unittest.TestCase):
    """Reading the marker, not replaying the assignment.

    Replaying was wrong for 43 of 1,092 comments across three paid runs,
    because the per-speaker cache scores a speaker on their first slot and the
    two traversals order slots differently.
    """

    def test_recorded_ids_come_from_the_prompt(self) -> None:
        post = {
            "generation_records": [
                {
                    "comment": {"comment_id": 4},
                    "prompt": '<generalized-card-matraix persona-id="0091" '
                    'seed-index="0" task-id="4"/>\nWrite one comment.',
                },
                {"comment": {"comment_id": 5}, "prompt": "no marker here"},
            ]
        }
        self.assertEqual({4: "0091"}, PB.recorded_persona_ids(post))

    def test_missing_records_yield_nothing_rather_than_guessing(self) -> None:
        self.assertEqual({}, PB.recorded_persona_ids({}))


if __name__ == "__main__":
    unittest.main()


class RuntimeCapturesItsOwnModeTest(unittest.TestCase):
    """An instance must not change behaviour when a global moves under it.

    `public_config()` renders every eligible persona to report length
    statistics, filling `_system_cache`. When the projection was read per call,
    a setter that ran after construction left the cache holding identities
    built under the previous projection while `run_config` reported the new
    one. The v152 probe recorded `persona_projection: register` at the top
    level and `persona_conditioning.projection: default` inside it, from the
    same run, and neither the manifest nor any artifact could say which one the
    Writer actually saw.
    """

    def tearDown(self) -> None:
        PB.set_persona_projection("default")
        PB.set_persona_draw("replace")

    def test_instance_keeps_its_construction_time_modes(self) -> None:
        PB.set_persona_projection("register")
        PB.set_persona_draw("exhaust")
        runtime = _runtime(DEV)
        PB.set_persona_projection("default")
        PB.set_persona_draw("replace")
        self.assertEqual("register", runtime.projection)
        self.assertEqual("exhaust", runtime.draw)
        self.assertEqual("register", runtime.public_config()["projection"])
        self.assertEqual("exhaust", runtime.public_config()["draw"])

    def test_rendering_follows_the_instance_not_the_global(self) -> None:
        PB.set_persona_projection("register")
        runtime = _runtime(DEV)
        PB.set_persona_projection("default")
        rendered = runtime.assignment_for_id(
            runtime._eligible[0].persona_id
        ).system_prompt
        plain = _runtime(DEV)  # built under `default`
        self.assertNotEqual(
            plain.assignment_for_id(plain._eligible[0].persona_id).system_prompt,
            rendered,
        )

    def test_draw_follows_the_instance_not_the_global(self) -> None:
        PB.set_persona_draw("exhaust")
        runtime = _runtime(DEV)
        PB.set_persona_draw("replace")
        got = {
            runtime.assign(
                seed_index=3, task=_task(i + 1), speaker_id=f"S{i + 1:03d}"
            ).persona_id
            for i in range(30)
        }
        self.assertEqual(30, len(got))
