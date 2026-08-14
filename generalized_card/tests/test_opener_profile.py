from __future__ import annotations

import unittest
from collections import Counter

from generalized_card.opener_profile import (
    OPENER_TYPES,
    build_opener_profile,
    classify_opener,
    scaled_opener_counts,
)
from generalized_card.planner_distribution import build_slot_distribution_schedule


class OpenerProfileTest(unittest.TestCase):
    def test_categories_are_grammatical_not_topical(self) -> None:
        # The same classifier has to serve every domain, so nothing here may
        # depend on subject matter.
        cases = {
            "Yeah, that's the one.": "polarity_token",
            "I switched last year and never looked back.": "first_person",
            "> the flange distance is different\nNot really.": "quote",
            "The mount is the part that changes.": "noun_phrase",
            "If you shoot indoors it matters.": "conditional",
            "How long does the battery last?": "question",
            "You should check the firmware first.": "address",
            "Try the adapter before buying glass.": "imperative",
            "Honestly it took a week.": "discourse_marker",
            "https://example.com/guide": "link",
            "Battery life drops once the screen stays on.": "content_phrase",
        }
        for text, expected in cases.items():
            self.assertEqual(classify_opener(text), expected, text)

    def test_profile_needs_enough_samples(self) -> None:
        thin = build_opener_profile([{"comments": [{"body": "Yeah."}]}])
        self.assertFalse(thin["available"])

    def test_profile_measures_shares(self) -> None:
        threads = [
            {"comments": [{"body": "Yeah, sure."} for _ in range(100)]
             + [{"body": "I use it daily."} for _ in range(100)]
             + [{"body": "Battery drains fast."} for _ in range(100)]}
        ]
        profile = build_opener_profile(threads)
        self.assertTrue(profile["available"])
        self.assertAlmostEqual(profile["shares"]["polarity_token"], 1 / 3, places=2)
        self.assertAlmostEqual(profile["shares"]["first_person"], 1 / 3, places=2)

    def test_counts_scale_to_thread_size(self) -> None:
        profile = {"available": True, "shares": {name: 1 / len(OPENER_TYPES) for name in OPENER_TYPES}}
        counts = scaled_opener_counts(profile, 22)
        self.assertEqual(sum(counts.values()), 22)

    def test_schedule_assigns_every_slot_an_opener(self) -> None:
        # The Writer defaulted 23% of comments to a bare agreement token against
        # a real 4%, so the entry grammar is a scheduled contract.
        profile = {
            "available": True,
            "shares": {
                "content_phrase": 0.42, "first_person": 0.20, "noun_phrase": 0.10,
                "discourse_marker": 0.07, "polarity_token": 0.05, "question": 0.04,
                "quote": 0.03, "conditional": 0.03, "address": 0.02,
                "imperative": 0.02, "link": 0.02,
            },
        }
        comments = [{"body": "word " * (8 + index * 2), "depth": index % 3} for index in range(40)]
        schedule = build_slot_distribution_schedule(
            template={
                "comment_count": 40, "story_count": 2, "polite_rate": 0.3,
                "impolite_rate": 0.4, "neutral_rate": 0.2, "somewhat_polite_rate": 0.1,
                "dominant_emotion_counts": {"neutral": 30, "approval": 10},
            },
            comments=comments,
            total_comments=40,
            opener_profile=profile,
        )
        assigned = Counter(
            value["opener_type"]
            for value in schedule["assignments"].values()
            if value.get("opener_type")
        )
        self.assertEqual(sum(assigned.values()), 40)
        # The dominant real category must dominate the schedule too.
        self.assertEqual(assigned.most_common(1)[0][0], "content_phrase")
        self.assertLess(assigned["polarity_token"], assigned["content_phrase"])

    def test_unwritable_entry_grammars_are_never_scheduled(self) -> None:
        # Ranking alone left every type assignable to every slot, so `question`
        # went to 23 slots and `imperative` to 10 across 520, and none of them
        # was ever realized: a slot that answers something cannot open by asking
        # it. The quota is re-spent on writable types so no slot is left to the
        # Writer's own default opening.
        profile = {
            "available": True,
            "shares": {
                "content_phrase": 0.40, "first_person": 0.20, "noun_phrase": 0.10,
                "discourse_marker": 0.06, "polarity_token": 0.05, "question": 0.05,
                "quote": 0.04, "conditional": 0.03, "address": 0.03,
                "imperative": 0.02, "link": 0.02,
            },
        }
        # Every slot is a long top-level comment: nothing to quote, nobody to
        # address, and far too long to be mostly a question.
        comments = [{"body": "word " * 90, "depth": 0} for _ in range(30)]
        schedule = build_slot_distribution_schedule(
            template={
                "comment_count": 30, "story_count": 1, "polite_rate": 0.3,
                "impolite_rate": 0.3, "neutral_rate": 0.3, "somewhat_polite_rate": 0.1,
                "dominant_emotion_counts": {"neutral": 30},
            },
            comments=comments,
            total_comments=30,
            opener_profile=profile,
        )
        assigned = Counter(
            value["opener_type"]
            for value in schedule["assignments"].values()
            if value.get("opener_type")
        )
        for unwritable in ("question", "quote", "address"):
            self.assertEqual(assigned[unwritable], 0, unwritable)
        self.assertEqual(sum(assigned.values()), 30)
        self.assertIn("question", schedule["unassigned_opener_types"])

    def test_no_opener_profile_leaves_slots_unconstrained(self) -> None:
        schedule = build_slot_distribution_schedule(
            template={"comment_count": 5, "story_count": 0, "polite_rate": 1.0,
                      "impolite_rate": 0.0, "neutral_rate": 0.0, "somewhat_polite_rate": 0.0},
            comments=[{"body": "word " * 20, "depth": 0} for _ in range(5)],
            total_comments=5,
            opener_profile=None,
        )
        self.assertFalse(
            any("opener_type" in value for value in schedule["assignments"].values())
        )


if __name__ == "__main__":
    unittest.main()
