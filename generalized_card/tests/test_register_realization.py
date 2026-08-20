"""Tests for the drawn realization of the tone class the plan assigned.

The defect this arm addresses is a realization gap, so the tests care about two
things above all: the draw reproduces the measured share, and the rule reaches
the slots the plan marked and no others. A cue that fires on every slot would
move the tone marginal, which already matches real text.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card import register_realization as rr  # noqa: E402


def profile(shares: dict[str, dict[str, float]], *, tone: str = "polite") -> dict:
    """One register's bands. `tones` maps a label to its own band table."""

    return {
        "available": True,
        "tone_classes": list(rr.TARGET_TONES),
        "tones": {
            tone: {
                band: {"sample_count": 100, "shares": dict(values)}
                for band, values in shares.items()
            }
        },
    }


def multi(per_tone: dict[str, dict[str, dict[str, float]]]) -> dict:
    """Several registers at once, each with its own bands and shares."""

    return {
        "available": True,
        "tone_classes": list(rr.TARGET_TONES),
        "tones": {
            tone: {
                band: {"sample_count": 100, "shares": dict(values)}
                for band, values in bands.items()
            }
            for tone, bands in per_tone.items()
        },
    }


class ProfileBuildTest(unittest.TestCase):
    """The profile must be conditioned on the classifier's own label."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, thread_id: str, comments: list[tuple[str, str]]) -> None:
        product = self.root / thread_id
        product.mkdir(parents=True, exist_ok=True)
        (product / "politeness_results.json").write_text(
            json.dumps(
                {
                    "threads": [
                        {
                            "thread_id": thread_id,
                            "comments": [
                                {"pred_label": label, "text": text}
                                for label, text in comments
                            ],
                        }
                    ]
                }
            )
        )

    def test_every_register_is_measured_separately(self) -> None:
        """v101: all four labels, each at its own rate.

        v99 measured `polite` only, and the v100 gate showed three of the four
        moves were at exactly zero on every other register while real comments of
        those registers carry them.
        """

        warm = ("polite", "this is a really good little body " * 6)
        blunt = ("impolite", "my copy is junk and the price is a joke " * 6)
        self._write("t1", [warm] * 250 + [blunt] * 250)
        built = rr.build_register_profile(self.root, reference_thread_ids=["t1"])
        self.assertTrue(built["available"])
        self.assertEqual(built["sample_count"], 500)
        self.assertEqual(built["tone_sample_counts"]["polite"], 250)
        self.assertEqual(built["tone_sample_counts"]["impolite"], 250)
        warm_band = next(iter(built["tones"]["polite"].values()))
        blunt_band = next(iter(built["tones"]["impolite"].values()))
        # each register measured from its own comments, not pooled
        self.assertGreater(warm_band["shares"]["plain_verdict"], 0.9)
        self.assertEqual(blunt_band["shares"]["plain_verdict"], 0.0)
        self.assertGreater(blunt_band["shares"]["own_thing"], 0.9)

    def test_a_register_with_no_comments_is_absent_rather_than_empty(self) -> None:
        self._write("t1", [("polite", "a really good body " * 12)] * 300)
        built = rr.build_register_profile(self.root, reference_thread_ids=["t1"])
        self.assertIn("polite", built["tones"])
        self.assertNotIn("impolite", built["tones"])

    def test_labels_outside_the_four_registers_are_ignored(self) -> None:
        self._write("t1", [("polite", "a really good body " * 12)] * 300
                    + [("garbage_label", "x " * 40)] * 300)
        built = rr.build_register_profile(self.root, reference_thread_ids=["t1"])
        self.assertEqual(built["sample_count"], 300)
        self.assertEqual(list(built["tones"]), ["polite"])

    def test_threads_outside_the_reference_set_are_skipped(self) -> None:
        self._write("keep", [("polite", "a really good body " * 12)] * 250)
        self._write("drop", [("polite", "my lovely lens " * 12)] * 250)
        built = rr.build_register_profile(self.root, reference_thread_ids=["keep"])
        self.assertEqual(built["sample_count"], 250)

    def test_below_the_sample_floor_reports_unavailable(self) -> None:
        self._write("t1", [("polite", "good " * 20)] * 30)
        built = rr.build_register_profile(self.root, reference_thread_ids=["t1"])
        self.assertFalse(built["available"])
        self.assertEqual(built["tones"], {})

    def test_a_missing_directory_is_unavailable_not_an_error(self) -> None:
        built = rr.build_register_profile(
            self.root / "nope", reference_thread_ids=["t1"]
        )
        self.assertFalse(built["available"])


class DrawTest(unittest.TestCase):
    def test_a_zero_share_never_draws(self) -> None:
        prof = profile({"medium": {"plain_verdict": 0.0}})
        self.assertFalse(
            rr.slot_uses_move(prof, slot_key="a", move="plain_verdict", word_count=40,
                              tone_class="polite")
        )

    def test_a_full_share_always_draws(self) -> None:
        prof = profile({"medium": {"plain_verdict": 1.0}})
        for key in ("a", "b", "c", "d"):
            self.assertTrue(
                rr.slot_uses_move(
                    prof, slot_key=key, move="plain_verdict", word_count=40,
                    tone_class="polite",
                )
            )

    def test_the_draw_is_deterministic_for_a_slot(self) -> None:
        prof = profile({"medium": {"plain_verdict": 0.5}})
        first = rr.slot_uses_move(
            prof, slot_key="seed:7", move="plain_verdict", word_count=40,
            tone_class="polite",
        )
        for _ in range(5):
            self.assertEqual(
                first,
                rr.slot_uses_move(
                    prof, slot_key="seed:7", move="plain_verdict", word_count=40,
                    tone_class="polite",
                ),
            )

    def test_the_draw_reproduces_the_measured_share(self) -> None:
        for share in (0.14, 0.4, 0.61, 0.86):
            prof = profile({"medium": {"plain_verdict": share}})
            drawn = sum(
                rr.slot_uses_move(
                    prof, slot_key=f"seed:{index}", move="plain_verdict",
                    word_count=40, tone_class="polite",
                )
                for index in range(4000)
            )
            self.assertAlmostEqual(drawn / 4000, share, delta=0.02)

    def test_moves_draw_independently(self) -> None:
        prof = profile({"medium": {"plain_verdict": 0.5, "own_thing": 0.5}})
        both = sum(
            rr.slot_uses_move(prof, slot_key=f"s{i}", move="plain_verdict",
                              word_count=40, tone_class="polite")
            == rr.slot_uses_move(prof, slot_key=f"s{i}", move="own_thing",
                                 word_count=40, tone_class="polite")
            for i in range(2000)
        )
        # Independent draws agree about half the time; a shared key would be 1.0.
        self.assertLess(both / 2000, 0.6)

    def test_the_draw_is_namespaced_away_from_the_rhythm_draw(self) -> None:
        """A slot drawing a rhythm habit must not thereby draw a register move."""

        from generalized_card import sentence_rhythm as sr

        rhythm = {
            "available": True,
            "bands": {
                "medium": {
                    "sample_count": 100,
                    "median_words_per_sentence": 14,
                    "median_sentences": 3,
                    "shares": {"digit": 0.5},
                }
            },
        }
        reg = profile({"medium": {"plain_verdict": 0.5}})
        agree = sum(
            sr.slot_uses_habit(rhythm, slot_key=f"s{i}", habit="digit", word_count=40)
            == rr.slot_uses_move(
                reg, slot_key=f"s{i}", move="plain_verdict", word_count=40,
                tone_class="polite",
            )
            for i in range(2000)
        )
        self.assertLess(agree / 2000, 0.6)

    def test_bands_draw_at_their_own_share(self) -> None:
        prof = profile(
            {"micro": {"own_thing": 0.06}, "essay": {"own_thing": 0.63}}
        )
        micro = sum(
            rr.slot_uses_move(prof, slot_key=f"s{i}", move="own_thing", word_count=5,
                              tone_class="polite")
            for i in range(3000)
        )
        essay = sum(
            rr.slot_uses_move(prof, slot_key=f"s{i}", move="own_thing", word_count=600,
                              tone_class="polite")
            for i in range(3000)
        )
        self.assertAlmostEqual(micro / 3000, 0.06, delta=0.02)
        self.assertAlmostEqual(essay / 3000, 0.63, delta=0.02)

    def test_an_unknown_move_never_draws(self) -> None:
        prof = profile({"medium": {"plain_verdict": 1.0}})
        self.assertFalse(
            rr.slot_uses_move(prof, slot_key="a", move="not_a_move", word_count=40,
                              tone_class="polite")
        )


class GuidanceTest(unittest.TestCase):
    def tearDown(self) -> None:
        rr.set_register_realization("measured")
        rr.set_active_register_profile({})

    def test_each_register_draws_at_its_own_measured_rate(self) -> None:
        """v101: the rate is the assigned register's, not one shared table."""

        prof = multi({
            "polite": {"medium": {"plain_verdict": 1.0}},
            "impolite": {"medium": {"plain_verdict": 0.0, "own_thing": 1.0}},
        })
        warm = rr.register_guidance(prof, slot_key="a", word_count=40, tone_class="polite")
        blunt = rr.register_guidance(prof, slot_key="a", word_count=40, tone_class="impolite")
        self.assertIn("plainly good", warm)
        # the blunt register measures 0.0 for that move and 1.0 for possession
        self.assertNotIn("plainly good", blunt)
        self.assertIn("Refer to something of your own", blunt)

    def test_a_register_the_profile_does_not_measure_gets_nothing(self) -> None:
        prof = profile({"medium": {"plain_verdict": 1.0}}, tone="polite")
        for tone in ("impolite", "neutral", "somewhat_polite"):
            self.assertEqual(
                rr.register_guidance(prof, slot_key="a", word_count=40, tone_class=tone),
                "",
            )

    def test_an_unknown_tone_gets_nothing_rather_than_a_default(self) -> None:
        prof = multi({t: {"medium": {"plain_verdict": 1.0}} for t in rr.TARGET_TONES})
        for tone in ("not_a_tone", "", None, "POLITE_ISH"):
            self.assertEqual(
                rr.register_guidance(prof, slot_key="a", word_count=40, tone_class=tone),
                "",
            )

    def test_the_rule_never_names_a_register(self) -> None:
        """Calling it "warm" would tell a blunt slot to soften."""

        prof = multi({t: {"medium": {"plain_verdict": 1.0}} for t in rr.TARGET_TONES})
        for tone in rr.TARGET_TONES:
            text = rr.register_guidance(
                prof, slot_key="a", word_count=40, tone_class=tone
            )
            self.assertNotIn("warm", text.lower())
            self.assertNotIn("polite", text.lower())

    def test_the_possessive_cue_rules_out_narration(self) -> None:
        """Real possessive comments score 0.279 mean story probability; generated
        ones scored 0.510 on the v100 gate, because the cue invited an arc."""

        spec = next(s for s in rr.REGISTER_MOVES if s["name"] == "own_thing")
        self.assertIn("plain present fact", spec["cue"])
        self.assertIn("Do not tell the story", spec["cue"])
        self.assertNotIn("ended up keeping", spec["cue"])

    def test_gratitude_is_a_measured_move_again(self) -> None:
        """v99 excluded it on a pooled 1.25x figure. Conditioned on the register
        the cue fires on, real polite micro comments thank at 0.329 and generated
        micro produced 0.100; five planned-polite micro slots on the v100 gate
        fell from 0.600 realized polite to 0.000."""

        names = rr.move_names()
        self.assertIn("gratitude", names)
        spec = next(s for s in rr.REGISTER_MOVES if s["name"] == "gratitude")
        # It must not stack a compliment on top -- that is `plain_verdict`'s job
        # and real short thanks do not do both.
        self.assertIn("Do not add a reason or a compliment", spec["cue"])

    def test_gratitude_draws_at_the_band_rate_not_a_flat_one(self) -> None:
        """It runs backwards to every other move: 0.329 at micro, 0.057 at essay."""

        prof = multi({"polite": {"micro": {"gratitude": 0.329},
                                 "essay": {"gratitude": 0.057}}})
        for band_words, expected in ((5, 0.329), (600, 0.057)):
            drawn = sum(
                rr.slot_uses_move(prof, slot_key=f"s{i}", move="gratitude",
                                  word_count=band_words, tone_class="polite")
                for i in range(4000)
            ) / 4000
            self.assertAlmostEqual(drawn, expected, delta=0.02)

    def test_the_blunt_cue_does_not_ask_the_slot_to_soften(self) -> None:
        """Real impolite comments carry these moves at 0.30 / 0.13 / 0.18."""

        verdict = next(s for s in rr.REGISTER_MOVES if s["name"] == "plain_verdict")
        # It has to be sayable inside a negative judgement, so it must concede
        # rather than instruct a positive stance.
        self.assertIn("even if your overall judgement is negative", verdict["cue"])

    def test_the_tone_check_is_case_and_space_insensitive(self) -> None:
        prof = profile({"medium": {"plain_verdict": 1.0}})
        self.assertIn(
            "Register, realized",
            rr.register_guidance(
                prof, slot_key="a", word_count=40, tone_class="  Polite "
            ),
        )

    def test_the_off_arm_renders_nothing(self) -> None:
        prof = profile({"medium": {"plain_verdict": 1.0}})
        rr.set_register_realization("off")
        self.assertEqual(
            rr.register_guidance(prof, slot_key="a", word_count=40, tone_class="polite"),
            "",
        )

    def test_no_profile_renders_nothing(self) -> None:
        for empty in (None, {}, {"available": False, "bands": {}}):
            self.assertEqual(
                rr.register_guidance(
                    empty, slot_key="a", word_count=40, tone_class="polite"
                ),
                "",
            )

    def test_a_band_with_no_drawn_move_renders_nothing(self) -> None:
        prof = profile({"medium": {"plain_verdict": 0.0, "own_thing": 0.0}})
        self.assertEqual(
            rr.register_guidance(prof, slot_key="a", word_count=40, tone_class="polite"),
            "",
        )

    def test_two_same_size_polite_slots_can_get_different_rules(self) -> None:
        prof = profile({"medium": {name: 0.5 for name in rr.move_names()}})
        rendered = {
            rr.register_guidance(
                prof, slot_key=f"seed:{index}", word_count=40, tone_class="polite"
            )
            for index in range(24)
        }
        self.assertGreater(len(rendered), 1)

    def test_no_cue_names_a_literal_phrase_to_copy(self) -> None:
        """`self_bleu_4` is a weak pass; a fixed phrase in a cue would repeat."""

        for spec in rr.REGISTER_MOVES:
            cue = spec["cue"].lower()
            for phrase in ("say \"", "write \"", "use the phrase", "exactly:"):
                self.assertNotIn(phrase, cue, spec["name"])

    def test_no_cue_carries_domain_vocabulary(self) -> None:
        """Every test runs on camera, so nothing else would catch this."""

        banned = (
            "camera", "lens", "photo", "shot", "sensor", "iso", "aperture",
            "mirrorless", "dslr", "megapixel", "zoom", "tripod", "flash",
        )
        for spec in rr.REGISTER_MOVES:
            lowered = spec["cue"].lower()
            for word in banned:
                self.assertNotIn(word, lowered, f"{spec['name']} names {word!r}")

    def test_the_active_profile_is_read_at_call_time(self) -> None:
        """A `from ... import ACTIVE_REGISTER_PROFILE` would break this."""

        self.assertEqual(
            rr.active_register_guidance(
                slot_key="a", word_count=40, tone_class="polite"
            ),
            "",
        )
        rr.set_active_register_profile(profile({"medium": {"plain_verdict": 1.0}}))
        self.assertIn(
            "Register, realized",
            rr.active_register_guidance(
                slot_key="a", word_count=40, tone_class="polite"
            ),
        )


class AuditTest(unittest.TestCase):
    def test_realized_shares_measure_what_the_cues_ask_for(self) -> None:
        comments = [
            {"content": "this is a really good body"},
            {"content": "my setup handles it"},
            {"content": "flat statement with no register at all"},
            {"content": "i love using it"},
        ]
        shares = rr.realized_move_shares(comments)
        self.assertAlmostEqual(shares["any_intensifier"], 0.25)
        self.assertAlmostEqual(shares["plain_verdict"], 0.25)
        self.assertAlmostEqual(shares["own_thing"], 0.25)
        self.assertAlmostEqual(shares["love_like"], 0.25)

    def test_realized_shares_are_empty_without_text(self) -> None:
        self.assertEqual(rr.realized_move_shares([]), {})
        self.assertEqual(rr.realized_move_shares([{"content": "  "}]), {})

    def test_every_move_pattern_matches_its_own_cue_target(self) -> None:
        """A move whose pattern cannot fire is a silently dead cue."""

        samples = {
            "any_intensifier": "that is really solid",
            "plain_verdict": "it is great",
            "own_thing": "my rig",
            "love_like": "i enjoy it",
        }
        for name, text in samples.items():
            self.assertEqual(
                rr.realized_move_shares([{"content": text}])[name], 1.0, name
            )


class ArmTest(unittest.TestCase):
    def tearDown(self) -> None:
        rr.set_register_realization("measured")

    def test_the_arm_switch_reports_its_state(self) -> None:
        self.assertTrue(rr.set_register_realization("measured"))
        self.assertFalse(rr.set_register_realization("off"))
        self.assertTrue(rr.set_register_realization(""))
        self.assertTrue(rr.set_register_realization(None))
        self.assertFalse(rr.set_register_realization("  OFF  "))


class WiringTest(unittest.TestCase):
    """The rule has to reach every Writer path, not just one."""

    def test_all_three_writer_paths_receive_the_rule(self) -> None:
        source = (PACKAGE_ROOT / "generalized_card" / "prompts.py").read_text()
        # Built once, then rendered in the shared guidance list (full + low-info)
        # and passed explicitly into the focused arm, which has been active since
        # v82.
        self.assertIn("register_rule = _register_rule(backend, task)", source)
        self.assertIn("register_rule=register_rule", source)
        self.assertEqual(source.count("            register_rule,"), 2)

    def test_the_cli_records_the_arm(self) -> None:
        source = (PACKAGE_ROOT / "scripts" / "run_generate.py").read_text()
        self.assertIn('"--register-realization"', source)
        self.assertIn('"register_realization": args.register_realization', source)
        self.assertIn(
            'env["GENERALIZED_CARD_REGISTER_REALIZATION"]', source
        )
        # In RUN_EXPERIMENT_FIELDS, so a resume cannot silently change it.
        fields = source[source.index("RUN_EXPERIMENT_FIELDS = ("):]
        self.assertIn('"register_realization"', fields[: fields.index(")")])

    def test_the_profile_is_stored_and_the_schema_bumped(self) -> None:
        source = (
            PACKAGE_ROOT / "generalized_card" / "domain_profile.py"
        ).read_text()
        self.assertIn('"register_profile": build_register_profile(', source)
        # At least 16, not exactly: a later version bumping the schema for its own
        # profile must not fail this test. The contract is that the register
        # profile is stored and the schema moved past the version that added it.
        version = int(
            re.search(r"PROFILE_SCHEMA_VERSION = (\d+)", source).group(1)
        )
        self.assertGreaterEqual(version, 16)


if __name__ == "__main__":
    unittest.main()
