"""Tests for the drawn opening word.

The defect this arm addresses is a realization gap, so the tests care about
three things above all: the draw reproduces the measured distribution, the word
comes from the register the plan assigned, and an unmeasured register gets
nothing rather than a default. A word drawn from the wrong register would tell a
blunt correction slot to open on gratitude.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card import opening_move as om  # noqa: E402


def profile(per_tone: dict[str, dict[str, list[tuple[str, float]]]]) -> dict:
    """Build a profile from {tone: {opener: [(token, share), ...]}}."""

    return {
        "available": True,
        "tone_classes": list(om.TARGET_TONES),
        "drawn_openers": list(om.DRAWN_OPENERS),
        "sample_count": 1000,
        "tones": {
            tone: {
                opener: {
                    "sample_count": 100,
                    "kept_count": 100,
                    "opener_share": 0.1,
                    "tokens": [
                        {"token": token, "share": share} for token, share in tokens
                    ],
                }
                for opener, tokens in rows.items()
            }
            for tone, rows in per_tone.items()
        },
    }


class DrawTest(unittest.TestCase):
    def setUp(self) -> None:
        om.set_opening_move("measured")

    def test_the_draw_is_deterministic_for_a_slot(self) -> None:
        payload = profile({"impolite": {"discourse_marker": [("well", 0.5), ("oh", 0.5)]}})
        first = om.slot_token(
            payload, slot_key="seed:7", opener="discourse_marker", tone_class="impolite"
        )
        second = om.slot_token(
            payload, slot_key="seed:7", opener="discourse_marker", tone_class="impolite"
        )
        self.assertEqual(first, second)
        self.assertIn(first, {"well", "oh"})

    def test_the_draw_reproduces_the_measured_distribution(self) -> None:
        payload = profile(
            {
                "impolite": {
                    "discourse_marker": [("well", 0.6), ("oh", 0.3), ("lol", 0.1)]
                }
            }
        )
        counts: dict[str, int] = {}
        trials = 4000
        for index in range(trials):
            token = om.slot_token(
                payload,
                slot_key=f"seed:{index}",
                opener="discourse_marker",
                tone_class="impolite",
            )
            counts[token] = counts.get(token, 0) + 1
        for token, expected in (("well", 0.6), ("oh", 0.3), ("lol", 0.1)):
            self.assertAlmostEqual(counts.get(token, 0) / trials, expected, delta=0.02)

    def test_the_draw_is_namespaced_away_from_the_register_draw(self) -> None:
        """Two slots that draw the same opening word must not thereby share a
        register move; the namespaces are what keeps the draws independent."""

        from generalized_card import register_realization as rr

        payload = profile({"polite": {"discourse_marker": [("thanks", 0.5), ("oh", 0.5)]}})
        register = {
            "available": True,
            "tone_classes": list(rr.TARGET_TONES),
            "tones": {
                "polite": {
                    "medium": {
                        "sample_count": 100,
                        "shares": {"gratitude": 0.5},
                    }
                }
            },
        }
        agree = 0
        trials = 400
        for index in range(trials):
            key = f"seed:{index}"
            drew_thanks = (
                om.slot_token(
                    payload, slot_key=key, opener="discourse_marker", tone_class="polite"
                )
                == "thanks"
            )
            drew_gratitude = rr.slot_uses_move(
                register,
                slot_key=key,
                move="gratitude",
                word_count=40,
                tone_class="polite",
            )
            agree += drew_thanks == drew_gratitude
        # Independent draws agree about half the time; a shared key agrees always.
        self.assertLess(agree / trials, 0.62)
        self.assertGreater(agree / trials, 0.38)

    def test_registers_draw_from_their_own_vocabulary(self) -> None:
        payload = profile(
            {
                "polite": {"discourse_marker": [("thanks", 1.0)]},
                "impolite": {"discourse_marker": [("well", 1.0)]},
            }
        )
        self.assertEqual(
            om.slot_token(
                payload, slot_key="s", opener="discourse_marker", tone_class="polite"
            ),
            "thanks",
        )
        self.assertEqual(
            om.slot_token(
                payload, slot_key="s", opener="discourse_marker", tone_class="impolite"
            ),
            "well",
        )

    def test_an_unmeasured_register_draws_nothing(self) -> None:
        payload = profile({"polite": {"discourse_marker": [("thanks", 1.0)]}})
        self.assertEqual(
            om.slot_token(
                payload, slot_key="s", opener="discourse_marker", tone_class="impolite"
            ),
            "",
        )


class StanceFamilyTest(unittest.TestCase):
    """The plan assigns a stance; a drawn polarity token must not contradict it.

    On the v102 gate 2 of 10 polarity slots opened with `no` on a
    `stance=agree` plan.
    """

    def setUp(self) -> None:
        om.set_opening_move("measured")
        self.payload = profile(
            {
                "impolite": {
                    "polarity_token": [
                        ("yeah", 0.4), ("yes", 0.3), ("no", 0.2), ("nope", 0.1)
                    ]
                }
            }
        )

    def _draw(self, stance: str, trials: int = 300) -> list[str]:
        return [
            om.slot_token(
                self.payload,
                slot_key=f"s:{i}",
                opener="polarity_token",
                tone_class="impolite",
                stance=stance,
            )
            for i in range(trials)
        ]

    def test_agree_never_draws_a_negative_token(self) -> None:
        self.assertFalse(set(self._draw("agree")) & om.NEGATIVE_TOKENS)

    def test_disagree_never_draws_an_affirmative_token(self) -> None:
        self.assertFalse(set(self._draw("disagree")) & om.AFFIRMATIVE_TOKENS)

    def test_a_noncommittal_stance_keeps_the_full_draw(self) -> None:
        for stance in ("mixed", "uncertain", "neutral", "joking", ""):
            drawn = set(self._draw(stance))
            self.assertTrue(drawn & om.NEGATIVE_TOKENS, stance)
            self.assertTrue(drawn & om.AFFIRMATIVE_TOKENS, stance)

    def test_within_family_shares_stay_measured(self) -> None:
        """Restricting the family must renormalise, not reweight."""

        drawn = self._draw("agree", trials=4000)
        yeah = drawn.count("yeah") / len(drawn)
        yes = drawn.count("yes") / len(drawn)
        # 0.4 and 0.3 renormalised inside the affirmative family.
        self.assertAlmostEqual(yeah, 0.4 / 0.7, delta=0.02)
        self.assertAlmostEqual(yes, 0.3 / 0.7, delta=0.02)

    def test_only_slots_that_conflicted_change(self) -> None:
        """The correction must perturb the contradicting slots and no others.

        Restricting the family and redrawing everything inside it would move
        17-54% of stance-committed slots that were never in conflict, which
        would make a v102/v103 comparison unattributable.
        """

        changed = conflicted = 0
        for index in range(2000):
            key = f"s:{index}"
            free = om.slot_token(
                self.payload, slot_key=key, opener="polarity_token",
                tone_class="impolite",
            )
            bound = om.slot_token(
                self.payload, slot_key=key, opener="polarity_token",
                tone_class="impolite", stance="agree",
            )
            if free != bound:
                changed += 1
                self.assertIn(free, om.NEGATIVE_TOKENS, (free, bound))
                conflicted += 1
        self.assertEqual(changed, conflicted)
        self.assertGreater(changed, 0)

    def test_the_veto_redraw_keeps_the_measured_within_family_shares(self) -> None:
        """Reusing the first draw's value would pile vetoed slots onto whichever
        tokens the vetoed slice of [0,1) happens to cover."""

        drawn = self._draw("agree", trials=8000)
        yeah = drawn.count("yeah") / len(drawn)
        self.assertAlmostEqual(yeah, 0.4 / 0.7, delta=0.02)

    def test_a_register_with_no_token_of_that_family_falls_back(self) -> None:
        """Withholding would cost the slot its assigned entry type."""

        payload = profile({"polite": {"polarity_token": [("yes", 1.0)]}})
        token = om.slot_token(
            payload,
            slot_key="s",
            opener="polarity_token",
            tone_class="polite",
            stance="disagree",
        )
        self.assertEqual(token, "yes")

    def test_stance_does_not_touch_the_discourse_marker_draw(self) -> None:
        """Those words carry no polarity, so the plan has nothing to contradict."""

        payload = profile({"impolite": {"discourse_marker": [("well", 0.5), ("oh", 0.5)]}})
        free = [
            om.slot_token(payload, slot_key=f"s:{i}", opener="discourse_marker",
                          tone_class="impolite")
            for i in range(200)
        ]
        bound = [
            om.slot_token(payload, slot_key=f"s:{i}", opener="discourse_marker",
                          tone_class="impolite", stance="disagree")
            for i in range(200)
        ]
        self.assertEqual(free, bound)


class GuidanceTest(unittest.TestCase):
    def setUp(self) -> None:
        om.set_opening_move("measured")

    def test_the_clause_names_the_drawn_word(self) -> None:
        payload = profile({"impolite": {"discourse_marker": [("well", 1.0)]}})
        text = om.opening_guidance(
            payload, slot_key="s", opener="discourse_marker", tone_class="impolite"
        )
        self.assertIn('"well"', text)
        self.assertNotIn("connective", text.lower())

    def test_the_clause_honours_the_planned_stance(self) -> None:
        payload = profile(
            {"impolite": {"polarity_token": [("yeah", 0.5), ("no", 0.5)]}}
        )
        for index in range(40):
            text = om.opening_guidance(
                payload,
                slot_key=f"s:{index}",
                opener="polarity_token",
                tone_class="impolite",
                stance="agree",
            )
            self.assertNotIn('"no"', text)

    def test_a_polarity_slot_is_told_the_token_is_bare(self) -> None:
        payload = profile({"impolite": {"polarity_token": [("no", 1.0)]}})
        text = om.opening_guidance(
            payload, slot_key="s", opener="polarity_token", tone_class="impolite"
        )
        self.assertIn('"no"', text)
        self.assertIn("bare token", text)

    def test_an_undrawn_opener_type_gets_nothing(self) -> None:
        payload = profile({"impolite": {"discourse_marker": [("well", 1.0)]}})
        self.assertEqual(
            om.opening_guidance(
                payload, slot_key="s", opener="content_phrase", tone_class="impolite"
            ),
            "",
        )

    def test_an_unmeasured_register_gets_nothing(self) -> None:
        payload = profile({"polite": {"discourse_marker": [("thanks", 1.0)]}})
        self.assertEqual(
            om.opening_guidance(
                payload, slot_key="s", opener="discourse_marker", tone_class="not_a_tone"
            ),
            "",
        )

    def test_the_arm_off_renders_nothing(self) -> None:
        payload = profile({"impolite": {"discourse_marker": [("well", 1.0)]}})
        om.set_opening_move("off")
        try:
            self.assertEqual(
                om.opening_guidance(
                    payload,
                    slot_key="s",
                    opener="discourse_marker",
                    tone_class="impolite",
                ),
                "",
            )
        finally:
            om.set_opening_move("measured")

    def test_no_cue_names_domain_vocabulary(self) -> None:
        """Every test runs on camera, so nothing else would catch a leak.

        The drawn word comes from the measured table rather than from source, so
        the check is on the fixed wording around it.
        """

        payload = profile(
            {
                "impolite": {
                    "discourse_marker": [("well", 1.0)],
                    "polarity_token": [("no", 1.0)],
                }
            }
        )
        banned = (
            "camera", "lens", "sensor", "photo", "shutter", "iso", "megapixel",
            "phone", "laptop", "headphone", "canon", "sony", "nikon", "fujifilm",
        )
        for opener in om.DRAWN_OPENERS:
            text = om.opening_guidance(
                payload, slot_key="s", opener=opener, tone_class="impolite"
            ).lower()
            for term in banned:
                self.assertNotIn(term, text)


class ForbiddenTokenTest(unittest.TestCase):
    def test_the_list_is_measured_and_pooled_across_registers(self) -> None:
        payload = profile(
            {
                "polite": {"polarity_token": [("yes", 0.7), ("yeah", 0.3)]},
                "impolite": {"polarity_token": [("yeah", 0.6), ("nope", 0.4)]},
            }
        )
        tokens = om.forbidden_opening_tokens(payload)
        self.assertEqual(set(tokens), {"yes", "yeah", "nope"})
        # Ordered by pooled share, so the most common leak is named first.
        self.assertEqual(tokens[0], "yeah")

    def test_an_empty_profile_yields_no_list(self) -> None:
        self.assertEqual(om.forbidden_opening_tokens({}), ())

    def test_the_arm_off_yields_no_list(self) -> None:
        """`off` must reproduce v101, which rendered the categorical ban."""

        payload = profile({"polite": {"polarity_token": [("yes", 1.0)]}})
        om.set_opening_move("off")
        try:
            self.assertEqual(om.forbidden_opening_tokens(payload), ())
        finally:
            om.set_opening_move("measured")

    def test_the_drawn_words_are_never_on_the_forbidden_list(self) -> None:
        """A slot told to open with a word it is also told not to use is a
        contradiction, and the Writer resolves contradictions by ignoring one."""

        payload = profile(
            {
                tone: {
                    "discourse_marker": [("thanks", 0.5), ("well", 0.5)],
                    "polarity_token": [("yes", 0.5), ("yeah", 0.5)],
                }
                for tone in om.TARGET_TONES
            }
        )
        banned = set(om.forbidden_opening_tokens(payload))
        for tone in om.TARGET_TONES:
            for index in range(60):
                token = om.slot_token(
                    payload,
                    slot_key=f"s:{index}",
                    opener="discourse_marker",
                    tone_class=tone,
                )
                self.assertNotIn(token, banned)


class ProfileBuildTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

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
            ),
            encoding="utf-8",
        )

    def test_tokens_are_measured_per_register(self) -> None:
        polite = [("polite", "Thanks for that, it helped a lot.")] * 60
        impolite = [("impolite", "Well that is just not how it works at all.")] * 60
        filler = [("neutral", "The body is fine for that use.")] * 120
        self._write("t1", polite + impolite + filler)
        built = om.build_opening_profile(self.root, reference_thread_ids=["t1"])
        self.assertTrue(built["available"])
        polite_tokens = [
            row["token"]
            for row in built["tones"]["polite"]["discourse_marker"]["tokens"]
        ]
        impolite_tokens = [
            row["token"]
            for row in built["tones"]["impolite"]["discourse_marker"]["tokens"]
        ]
        self.assertEqual(polite_tokens, ["thanks"])
        self.assertEqual(impolite_tokens, ["well"])

    def test_a_rare_token_is_dropped_rather_than_cued(self) -> None:
        bulk = [("impolite", "Well that is not how it works.")] * 200
        rare = [("impolite", "Oof that is rough.")] * 2
        self._write("t1", bulk + rare)
        built = om.build_opening_profile(self.root, reference_thread_ids=["t1"])
        tokens = {
            row["token"]
            for row in built["tones"]["impolite"]["discourse_marker"]["tokens"]
        }
        self.assertIn("well", tokens)
        self.assertNotIn("oof", tokens)

    def test_threads_outside_the_reference_set_are_skipped(self) -> None:
        self._write("t1", [("impolite", "Well no.")] * 300)
        built = om.build_opening_profile(self.root, reference_thread_ids=["other"])
        self.assertFalse(built["available"])

    def test_below_the_sample_floor_reports_unavailable(self) -> None:
        self._write("t1", [("impolite", "Well no.")] * 10)
        built = om.build_opening_profile(self.root, reference_thread_ids=["t1"])
        self.assertFalse(built["available"])
        self.assertEqual(built["tones"], {})

    def test_a_missing_directory_is_unavailable_not_an_error(self) -> None:
        built = om.build_opening_profile(
            self.root / "nope", reference_thread_ids=["t1"]
        )
        self.assertFalse(built["available"])

    def test_shares_within_an_opener_sum_to_one(self) -> None:
        comments = (
            [("impolite", "Well that is not how it works.")] * 60
            + [("impolite", "Oh that is not how it works.")] * 40
            + [("neutral", "The body is fine.")] * 150
        )
        self._write("t1", comments)
        built = om.build_opening_profile(self.root, reference_thread_ids=["t1"])
        rows = built["tones"]["impolite"]["discourse_marker"]["tokens"]
        self.assertAlmostEqual(sum(row["share"] for row in rows), 1.0, places=4)


class AuditTest(unittest.TestCase):
    def test_realized_shares_report_the_entry_types_and_words(self) -> None:
        report = om.realized_opening_shares(
            [
                {"content": "Yeah, that tracks."},
                {"content": "Yep, same here."},
                {"content": "The body is what matters."},
                {"content": ""},
            ]
        )
        self.assertEqual(report["comment_count"], 3)
        self.assertAlmostEqual(report["opener_shares"]["polarity_token"], 2 / 3, places=5)
        self.assertEqual(
            report["tokens"]["polarity_token"], {"yeah": 1, "yep": 1}
        )


if __name__ == "__main__":
    unittest.main()
