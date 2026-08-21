"""Tests for `evaluative_register`.

The module ships three arms against one diagnosis (`tasks/v104-worklog.md`), so
each arm is tested for what it does, for what it must not do, and for the `off`
value reproducing v103.
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

from generalized_card import evaluative_register as ev  # noqa: E402
from generalized_card import opening_move, register_realization, sentence_rhythm  # noqa: E402


def _profile(hot_share: float = 0.5, comments: int = 4000) -> dict:
    band = {"comments": comments // 5, "positive_sentences": 400, "hot_share": hot_share}
    tone = {"comments": comments // 4, "positive_sentences": 400, "hot_share": hot_share}
    return {
        "comments": comments,
        "sentences": comments * 4,
        "positive_sentences": 2000,
        "hot_share": hot_share,
        "downtoner_tag_comment_rate": 0.004,
        "partitive_comment_rate": 0.02,
        "bands": {ev.band_of(low): dict(band) for low, _ in ev.BANDS},
        "tones": {name: dict(tone) for name in ev.TARGET_TONES},
    }


class TierClassificationTest(unittest.TestCase):
    def test_hot_and_warm_are_disjoint_vocabularies(self) -> None:
        self.assertFalse(set(ev.HOT_WORDS) & set(ev.WARM_WORDS))

    def test_the_tier_split_matches_what_was_measured(self) -> None:
        self.assertTrue(ev.is_hot("The IV is fantastic."))
        self.assertTrue(ev.is_hot("Wonderful camera."))
        self.assertFalse(ev.is_hot("That part was good."))
        self.assertFalse(ev.is_hot("Pretty useful, honestly."))
        # Both tiers count as an evaluation; only the strength differs.
        self.assertTrue(ev.is_positive_sentence("That part was good."))
        self.assertFalse(ev.is_positive_sentence("I shoot at f/8 most days."))

    def test_no_domain_vocabulary_in_either_tier(self) -> None:
        domain = {"camera", "lens", "sensor", "autofocus", "iso", "sony", "canon",
                  "headphone", "laptop", "phone", "bass", "keyboard"}
        self.assertFalse(domain & {w.lower() for w in ev.HOT_WORDS + ev.WARM_WORDS})


class DowntonerTagTest(unittest.TestCase):
    def test_fires_on_the_measured_construction(self) -> None:
        for text in ("Eye AF is good, sure.",
                     "Pretty useful, honestly.",
                     "The $200 part is nice, sure, but the rest is not.",
                     "Looks useful on the surface."):
            self.assertTrue(ev.has_downtoner_tag(text), text)

    def test_does_not_fire_on_a_fronted_adverb_or_an_ordinary_clause(self) -> None:
        # The tic is a tag that takes the sentence back *after* the evaluation.
        # A fronted "Honestly," does the opposite and real text uses it.
        for text in ("Honestly, it's a great camera.",
                     "It's great, really good glass.",
                     "Wonderful camera.",
                     "I guess I should have bought the other one."):
            self.assertFalse(ev.has_downtoner_tag(text), text)


class PartitiveTest(unittest.TestCase):
    def test_fires_on_the_measured_construction(self) -> None:
        for text in ("That part was good.", "the useful bit is the adapter",
                     "that's the part that matters", "The handy side of it."):
            self.assertTrue(ev.has_partitive(text), text)

    def test_does_not_fire_on_ordinary_use_of_the_noun(self) -> None:
        for text in ("Wonderful camera.", "I part with gear slowly.",
                     "Parts are cheap now."):
            self.assertFalse(ev.has_partitive(text), text)


class DrawTest(unittest.TestCase):
    def test_the_draw_is_stable_for_a_slot(self) -> None:
        profile = _profile(0.5)
        first = ev.slot_uses_hot_tier(profile, slot_key="s:1", tone_class="polite", word_count=60)
        for _ in range(5):
            self.assertIs(
                first,
                ev.slot_uses_hot_tier(profile, slot_key="s:1", tone_class="polite", word_count=60),
            )

    def test_the_draw_reproduces_the_measured_share(self) -> None:
        for share in (0.15, 0.5, 0.85):
            profile = _profile(share)
            hits = sum(
                ev.slot_uses_hot_tier(
                    profile, slot_key=f"seed:{index}", tone_class="polite", word_count=60
                )
                for index in range(4000)
            )
            self.assertAlmostEqual(hits / 4000, share, delta=0.02)

    def test_the_draw_is_independent_of_the_other_per_slot_draws(self) -> None:
        """A slot drawing a hot tier must not also be the slot drawing a habit.

        Against the real sibling draws, not against this module's own hash with
        a different string: the point is that the *shipped* per-slot mechanisms
        stay uncorrelated on the same slot key.
        """

        profile = _profile(0.5)
        keys = [f"seed:{index}" for index in range(2000)]
        tier = [
            ev.slot_uses_hot_tier(profile, slot_key=key, tone_class="polite", word_count=60)
            for key in keys
        ]
        bucket = register_realization.structure_bucket(60)
        register = {
            "tones": {
                "polite": {
                    bucket: {"shares": {name: 0.5 for name in register_realization.move_names()}}
                }
            }
        }
        siblings = {
            "register_realization": [
                register_realization.slot_uses_move(
                    register,
                    slot_key=key,
                    move=register_realization.move_names()[0],
                    word_count=60,
                    tone_class="polite",
                )
                for key in keys
            ],
            "sentence_rhythm": [
                sentence_rhythm.slot_uses_habit(
                    {"bands": {sentence_rhythm.structure_bucket(60): {"shares": {"exclamation": 0.5}}}},
                    slot_key=key,
                    habit="exclamation",
                    word_count=60,
                )
                for key in keys
            ],
            "opening_move": [
                opening_move.slot_token(
                    {
                        "tones": {
                            "polite": {
                                "polarity_token": {
                                    "tokens": [
                                        {"token": "yes", "share": 0.5},
                                        {"token": "no", "share": 0.5},
                                    ]
                                }
                            }
                        }
                    },
                    slot_key=key,
                    opener="polarity_token",
                    tone_class="polite",
                )
                == "yes"
                for key in keys
            ],
        }
        for name, other in siblings.items():
            if len(set(other)) < 2:
                continue
            agree = sum(a == b for a, b in zip(tier, other)) / len(tier)
            self.assertLess(abs(agree - 0.5), 0.06, name)

    def test_named_words_vary_by_slot_and_stay_in_the_hot_tier(self) -> None:
        drawn = {ev.slot_words(f"seed:{index}") for index in range(200)}
        self.assertGreater(len(drawn), 5)
        for words in drawn:
            self.assertEqual(len(words), ev._CUE_WORDS)
            for word in words:
                self.assertIn(word, ev.HOT_WORDS)


class FallbackTest(unittest.TestCase):
    def test_a_thin_register_falls_back_to_the_band_then_to_pooled(self) -> None:
        profile = _profile(0.5)
        profile["tones"]["neutral"]["positive_sentences"] = 3
        profile["bands"][ev.band_of(60)]["hot_share"] = 0.31
        self.assertAlmostEqual(
            ev.hot_share(profile, tone_class="neutral", word_count=60), 0.31
        )
        for row in profile["bands"].values():
            row["positive_sentences"] = 3
        profile["hot_share"] = 0.22
        self.assertAlmostEqual(
            ev.hot_share(profile, tone_class="neutral", word_count=60), 0.22
        )

    def test_a_domain_below_the_sample_floor_gets_no_tier_cue(self) -> None:
        profile = _profile(0.5, comments=10)
        self.assertEqual(ev.hot_share(profile, tone_class="polite", word_count=60), 0.0)
        self.assertFalse(
            ev.slot_uses_hot_tier(profile, slot_key="s:1", tone_class="polite", word_count=60)
        )


class GuidanceTest(unittest.TestCase):
    def setUp(self) -> None:
        ev.set_evaluation_tier("measured")
        ev.set_downtoner_tag("suppress")
        ev.set_partitive_reference("suppress")

    def tearDown(self) -> None:
        self.setUp()

    def test_the_tier_rule_is_conditional_on_an_evaluation_happening(self) -> None:
        """The Planner owns whether a slot praises anything; this must not."""

        profile = _profile(1.0)
        line = ev.evaluative_guidance(
            profile, slot_key="s:1", tone_class="polite", word_count=60
        )
        self.assertIn("If this comment rates something positively", line)
        for word in ("must praise", "always praise", "add a compliment"):
            self.assertNotIn(word, line)

    def test_the_rule_names_concrete_words_rather_than_a_category(self) -> None:
        profile = _profile(1.0)
        line = ev.evaluative_guidance(
            profile, slot_key="s:7", tone_class="polite", word_count=60
        )
        self.assertTrue(any(word in line for word in ev.HOT_WORDS), line)

    def test_each_arm_off_removes_only_its_own_rule(self) -> None:
        profile = _profile(1.0)
        full = ev.evaluative_guidance(profile, slot_key="s:1", tone_class="polite", word_count=60)
        self.assertIn("full strength", full)
        self.assertIn("takes the sentence back", full)
        self.assertIn("slice of it", full)

        ev.set_evaluation_tier("off")
        line = ev.evaluative_guidance(profile, slot_key="s:1", tone_class="polite", word_count=60)
        self.assertNotIn("full strength", line)
        self.assertIn("takes the sentence back", line)

        ev.set_downtoner_tag("off")
        line = ev.evaluative_guidance(profile, slot_key="s:1", tone_class="polite", word_count=60)
        self.assertNotIn("takes the sentence back", line)
        self.assertIn("slice of it", line)

        ev.set_partitive_reference("off")
        self.assertEqual(
            ev.evaluative_guidance(profile, slot_key="s:1", tone_class="polite", word_count=60),
            "",
        )

    def test_all_three_off_reproduces_v103_exactly(self) -> None:
        ev.set_evaluation_tier("off")
        ev.set_downtoner_tag("off")
        ev.set_partitive_reference("off")
        for tone in ev.TARGET_TONES:
            for words in (5, 40, 200):
                self.assertEqual(
                    ev.evaluative_guidance(
                        _profile(1.0), slot_key="s:1", tone_class=tone, word_count=words
                    ),
                    "",
                )

    def test_the_cue_carries_no_domain_vocabulary(self) -> None:
        line = ev.evaluative_guidance(
            _profile(1.0), slot_key="s:3", tone_class="polite", word_count=60
        )
        for token in ("camera", "lens", "sensor", "headphone", "laptop", "phone"):
            self.assertNotIn(token, line.lower())


class ProfileBuildTest(unittest.TestCase):
    def test_the_profile_is_built_from_reference_threads_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            product = root / "product_a"
            product.mkdir()
            payload = {
                "threads": [
                    {
                        "thread_id": "keep",
                        "comments": [
                            {"pred_label": "polite", "text": "Wonderful camera. I use it daily."},
                            {"pred_label": "impolite", "text": "That part was good, sure."},
                        ],
                    },
                    {
                        "thread_id": "drop",
                        "comments": [
                            {"pred_label": "polite", "text": "Fantastic. " * 40},
                        ],
                    },
                ]
            }
            (product / "politeness_results.json").write_text(json.dumps(payload))
            built = ev.build_evaluative_profile(root, reference_thread_ids=["keep"])
        self.assertEqual(built["comments"], 2)
        # One hot sentence ("Wonderful camera.") and one warm ("That part was
        # good, sure."), so the hot share of positive sentences is one half.
        self.assertAlmostEqual(built["hot_share"], 0.5)
        self.assertAlmostEqual(built["downtoner_tag_comment_rate"], 0.5)
        self.assertAlmostEqual(built["partitive_comment_rate"], 0.5)
        self.assertIn("polite", built["tones"])


class AuditTest(unittest.TestCase):
    def test_the_audit_measures_what_the_profile_measures(self) -> None:
        comments = [
            {"content": "Wonderful camera. I have used it for years."},
            {"content": "Eye AF is good, sure. That part was the useful bit."},
        ]
        audit = ev.realized_evaluative_shares(comments)
        # "Wonderful" is hot; "good" and "useful" are warm, so one of three.
        self.assertAlmostEqual(audit["hot_share_of_positive"], 1 / 3)
        self.assertGreater(audit["downtoner_tag_per_1k_sentences"], 0.0)
        self.assertAlmostEqual(audit["partitive_comment_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
