"""Tests for the drawn closing move.

The defect is that the Writer ends on an abstract verdict 19x more often than
real text does. The tests care about three things: the draw reproduces the
measured share, the rule is silent on slots too short to have a closing move
separate from their body, and it reaches every Writer path.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card import closing_move as cm  # noqa: E402


def profile(shares: dict[str, dict[str, float]], *, min_words: int = 25) -> dict:
    return {
        "available": True,
        "min_words": min_words,
        "bands": {
            band: {"sample_count": 100, "median_final_words": 18, "shares": dict(v)}
            for band, v in shares.items()
        },
    }


class LastSentenceTest(unittest.TestCase):
    def test_the_final_sentence_is_the_one_measured(self) -> None:
        self.assertEqual(
            cm.last_sentence("First one. Second one. Third one."), "Third one."
        )

    def test_a_single_sentence_is_its_own_close(self) -> None:
        self.assertEqual(cm.last_sentence("Only this"), "Only this")

    def test_newlines_split_as_sentences(self) -> None:
        self.assertEqual(cm.last_sentence("para one\n\npara two"), "para two")

    def test_empty_text_has_no_close(self) -> None:
        for value in ("", "   ", None):
            self.assertEqual(cm.last_sentence(value), "")

    def test_trailing_whitespace_does_not_become_the_close(self) -> None:
        self.assertEqual(cm.last_sentence("A thing happened.   "), "A thing happened.")


class ProfileBuildTest(unittest.TestCase):
    def _threads(self, bodies: list[str]) -> list[dict]:
        return [{"comments": [{"body": b} for b in bodies]}]

    def test_shares_are_measured_on_the_last_sentence_only(self) -> None:
        # The verdict phrase sits in the BODY, not the close, so it must not count.
        body = ("what matters here is nothing at all. " * 3) + "I still have mine."
        built = cm.build_closing_profile(self._threads([body] * 300))
        self.assertTrue(built["available"])
        band = next(iter(built["bands"].values()))
        self.assertEqual(band["shares"]["abstract_verdict_close"], 0.0)
        self.assertEqual(band["shares"]["own_concrete_close"], 1.0)

    def test_a_verdict_close_is_counted(self) -> None:
        body = ("I used it for a while and it was fine. " * 4) + "That's the part that matters."
        built = cm.build_closing_profile(self._threads([body] * 300))
        band = next(iter(built["bands"].values()))
        self.assertEqual(band["shares"]["abstract_verdict_close"], 1.0)

    def test_short_comments_are_not_counted(self) -> None:
        built = cm.build_closing_profile(self._threads(["too short to have a close"] * 400))
        self.assertFalse(built["available"])
        self.assertEqual(built["sample_count"], 0)

    def test_below_the_sample_floor_is_unavailable(self) -> None:
        body = "a word " * 40
        built = cm.build_closing_profile(self._threads([body] * 30))
        self.assertFalse(built["available"])

    def test_the_floor_is_recorded_so_the_renderer_can_honour_it(self) -> None:
        body = "a word " * 40
        built = cm.build_closing_profile(self._threads([body] * 300))
        self.assertEqual(built["min_words"], 25)


class DrawTest(unittest.TestCase):
    def test_the_draw_reproduces_the_measured_share(self) -> None:
        for value in (0.014, 0.152, 0.5, 0.9):
            prof = profile({"medium": {"own_concrete_close": value}})
            drawn = sum(
                cm.slot_uses_move(
                    prof, slot_key=f"s{i}", move="own_concrete_close", word_count=40
                )
                for i in range(4000)
            )
            self.assertAlmostEqual(drawn / 4000, value, delta=0.02)

    def test_the_draw_is_deterministic(self) -> None:
        prof = profile({"medium": {"own_concrete_close": 0.5}})
        first = cm.slot_uses_move(
            prof, slot_key="seed:3", move="own_concrete_close", word_count=40
        )
        for _ in range(4):
            self.assertEqual(
                first,
                cm.slot_uses_move(
                    prof, slot_key="seed:3", move="own_concrete_close", word_count=40
                ),
            )

    def test_namespaced_away_from_the_register_and_rhythm_draws(self) -> None:
        from generalized_card import register_realization as rr
        from generalized_card import sentence_rhythm as sr

        close = profile({"medium": {"own_concrete_close": 0.5}})
        reg = {
            "available": True,
            "tones": {
                "polite": {
                    "medium": {"sample_count": 100, "shares": {"plain_verdict": 0.5}}
                }
            },
        }
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
        for other, call in (
            ("register", lambda k: rr.slot_uses_move(
                reg, slot_key=k, move="plain_verdict", word_count=40,
                tone_class="polite")),
            ("rhythm", lambda k: sr.slot_uses_habit(
                rhythm, slot_key=k, habit="digit", word_count=40)),
        ):
            agree = sum(
                cm.slot_uses_move(
                    close, slot_key=f"s{i}", move="own_concrete_close", word_count=40
                )
                == call(f"s{i}")
                for i in range(2000)
            )
            self.assertLess(agree / 2000, 0.6, other)

    def test_an_unknown_move_never_draws(self) -> None:
        prof = profile({"medium": {"own_concrete_close": 1.0}})
        self.assertFalse(
            cm.slot_uses_move(prof, slot_key="a", move="nope", word_count=40)
        )


class GuidanceTest(unittest.TestCase):
    def tearDown(self) -> None:
        cm.set_closing_move("measured")
        cm.set_active_closing_profile({})

    def test_a_suppressed_verdict_renders_its_negative_cue(self) -> None:
        prof = profile({"medium": {"abstract_verdict_close": 0.0}})
        text = cm.closing_guidance(prof, slot_key="a", word_count=40)
        self.assertIn("Closing move:", text)
        self.assertIn("Do not end by saying what matters", text)

    def test_a_drawn_verdict_renders_nothing_for_that_move(self) -> None:
        """It has no positive cue: the point is not to ask for a verdict."""

        prof = profile({"medium": {"abstract_verdict_close": 1.0}})
        self.assertEqual(cm.closing_guidance(prof, slot_key="a", word_count=40), "")

    def test_a_drawn_concrete_close_renders_its_positive_cue(self) -> None:
        prof = profile({"medium": {"own_concrete_close": 1.0}})
        text = cm.closing_guidance(prof, slot_key="a", word_count=40)
        self.assertIn("End on something concrete of your own", text)

    def test_short_slots_are_silent(self) -> None:
        prof = profile({"micro": {"abstract_verdict_close": 0.0}}, min_words=25)
        for words in (0, 5, 12, 24):
            self.assertEqual(
                cm.closing_guidance(prof, slot_key="a", word_count=words), ""
            )
        self.assertNotEqual(
            cm.closing_guidance(
                profile({"medium": {"abstract_verdict_close": 0.0}}),
                slot_key="a",
                word_count=25,
            ),
            "",
        )

    def test_a_bad_word_count_is_silent_rather_than_raising(self) -> None:
        prof = profile({"medium": {"abstract_verdict_close": 0.0}})
        for value in (None, "", "abc", object()):
            self.assertEqual(cm.closing_guidance(prof, slot_key="a", word_count=value), "")

    def test_the_off_arm_renders_nothing(self) -> None:
        prof = profile({"medium": {"abstract_verdict_close": 0.0}})
        cm.set_closing_move("off")
        self.assertEqual(cm.closing_guidance(prof, slot_key="a", word_count=40), "")

    def test_no_profile_renders_nothing(self) -> None:
        for empty in (None, {}, {"available": False, "bands": {}}):
            self.assertEqual(cm.closing_guidance(empty, slot_key="a", word_count=40), "")

    def test_two_same_size_slots_can_get_different_rules(self) -> None:
        prof = profile({"medium": {name: 0.5 for name in cm.move_names()}})
        rendered = {
            cm.closing_guidance(prof, slot_key=f"seed:{i}", word_count=40)
            for i in range(24)
        }
        self.assertGreater(len(rendered), 1)

    def test_the_concrete_cue_rules_out_narration(self) -> None:
        """It asked for events on the v100 gate and got them: story probability
        moved from 0.8% error to 29.2%. Real closers of this kind are states."""

        spec = next(s for s in cm.CLOSING_MOVES if s["name"] == "own_concrete_close")
        self.assertIn("Do not recount what happened", spec["cue"])
        for event_word in ("how long you have had", "what it did or did not do"):
            self.assertNotIn(event_word, spec["cue"])

    def test_the_cues_carry_no_domain_vocabulary(self) -> None:
        banned = ("camera", "lens", "photo", "sensor", "iso", "shutter", "zoom")
        for spec in cm.CLOSING_MOVES:
            for cue in (spec["cue"], spec["suppress_cue"]):
                for word in banned:
                    self.assertNotIn(word, cue.lower(), spec["name"])

    def test_the_active_profile_is_read_at_call_time(self) -> None:
        self.assertEqual(
            cm.active_closing_guidance(slot_key="a", word_count=40), ""
        )
        cm.set_active_closing_profile(
            profile({"medium": {"abstract_verdict_close": 0.0}})
        )
        self.assertIn(
            "Closing move:", cm.active_closing_guidance(slot_key="a", word_count=40)
        )


class AuditTest(unittest.TestCase):
    def test_realized_shares_ignore_short_comments(self) -> None:
        long_verdict = {"content": ("word " * 30) + "That's the part that matters."}
        short = {"content": "That's the part that matters."}
        shares = cm.realized_close_shares([long_verdict, short])
        # Only the long one counts, and it closes on a verdict.
        self.assertEqual(shares["abstract_verdict_close"], 1.0)

    def test_realized_shares_are_empty_without_long_comments(self) -> None:
        self.assertEqual(cm.realized_close_shares([{"content": "tiny"}]), {})
        self.assertEqual(cm.realized_close_shares([]), {})

    def test_every_pattern_matches_the_thing_its_cue_describes(self) -> None:
        samples = {
            "own_concrete_close": ("word " * 30) + "I still have mine after 3 years.",
            "abstract_verdict_close": ("word " * 30) + "That's what actually matters.",
        }
        for name, text in samples.items():
            self.assertEqual(
                cm.realized_close_shares([{"content": text}])[name], 1.0, name
            )


class ArmTest(unittest.TestCase):
    def tearDown(self) -> None:
        cm.set_closing_move("measured")

    def test_the_arm_switch_reports_its_state(self) -> None:
        self.assertTrue(cm.set_closing_move("measured"))
        self.assertFalse(cm.set_closing_move("off"))
        self.assertFalse(cm.set_closing_move("  OFF "))
        self.assertTrue(cm.set_closing_move(None))


class WiringTest(unittest.TestCase):
    def test_all_writer_paths_receive_the_rule(self) -> None:
        source = (PACKAGE_ROOT / "generalized_card" / "prompts.py").read_text()
        self.assertIn("closing_rule = _closing_rule(backend, task)", source)
        self.assertIn("closing_rule=closing_rule", source)
        self.assertEqual(source.count("            closing_rule,"), 2)

    def test_the_cli_records_the_arm(self) -> None:
        source = (PACKAGE_ROOT / "scripts" / "run_generate.py").read_text()
        self.assertIn('"--closing-move"', source)
        self.assertIn('"closing_move": args.closing_move', source)
        self.assertIn('env["GENERALIZED_CARD_CLOSING_MOVE"]', source)
        fields = source[source.index("RUN_EXPERIMENT_FIELDS = ("):]
        self.assertIn('"closing_move"', fields[: fields.index(")")])

    def test_the_profile_is_stored_and_the_schema_bumped(self) -> None:
        source = (PACKAGE_ROOT / "generalized_card" / "domain_profile.py").read_text()
        self.assertIn('"closing_profile": build_closing_profile(', source)
        version = int(
            re.search(r"PROFILE_SCHEMA_VERSION = (\d+)", source).group(1)
        )
        self.assertGreaterEqual(version, 17)


if __name__ == "__main__":
    unittest.main()
