from __future__ import annotations

import unittest

from generalized_card.entity_inventory import (
    build_entity_inventory,
    slot_equipment_options,
)


BRANDS = ("Canon", "Sony", "Nikon", "Sigma")


def thread(*bodies: str) -> dict:
    return {"comments": [{"body": body} for body in bodies]}


class EntityInventoryTest(unittest.TestCase):
    def test_learns_designators_by_brand_adjacency_then_counts_bare_uses(self) -> None:
        # Real writers name a model without repeating its brand, so a
        # brand-adjacency-only count would miss most real usage.
        inventory = build_entity_inventory(
            [
                thread(
                    "I shoot a Canon 5D and love it.",
                    "picked up a Canon 5D used",
                    "the 5D is still fine honestly",
                    "my 5D died last year",
                )
            ],
            brand_terms=BRANDS,
            min_occurrences=1,
        )
        self.assertTrue(inventory["available"])
        terms = {row["term"]: row["count"] for row in inventory["terms"]}
        self.assertEqual(terms, {"Canon 5D": 4})
        self.assertEqual(inventory["terms"][0]["branded_mentions"], 2)

    def test_rejects_bare_numbers_and_product_lines(self) -> None:
        # "Canon EOS" is a line, and a number after a brand is a focal length,
        # ISO, or price. Counting every later occurrence of those would swamp the
        # inventory with the very concentration it exists to remove.
        inventory = build_entity_inventory(
            [
                thread(
                    "Canon EOS bodies are fine. Canon 100 dollars is cheap.",
                    "ISO 100 is clean and 100 percent usable",
                    "Canon USA support was slow",
                )
            ],
            brand_terms=BRANDS,
            min_occurrences=1,
        )
        self.assertFalse(
            any(
                row["designator"].isalpha() or row["designator"].isdigit()
                for row in inventory["terms"]
            ),
            inventory["terms"],
        )

    def test_designator_resolves_to_its_most_frequent_brand(self) -> None:
        inventory = build_entity_inventory(
            [
                thread(
                    "the Sigma 50mm is sharp",
                    "my Sigma 50mm again",
                    "another Sigma 50mm here",
                    "a Canon 50mm once",
                )
            ],
            brand_terms=BRANDS,
            min_occurrences=1,
        )
        terms = [row for row in inventory["terms"] if row["designator"] == "50mm"]
        self.assertEqual(terms[0]["brand"], "Sigma")

    def test_slot_options_rotate_so_entity_mass_spreads(self) -> None:
        inventory = {
            "available": True,
            "terms": [{"term": f"Brand {index}D", "count": 5} for index in range(12)],
        }
        first = slot_equipment_options(inventory, slot_index=0, limit=3)
        second = slot_equipment_options(inventory, slot_index=1, limit=3)
        third = slot_equipment_options(inventory, slot_index=2, limit=3)
        self.assertEqual(len(first), 3)
        self.assertFalse(set(first) & set(second))
        self.assertFalse(set(second) & set(third))

    def test_slot_options_exclude_entities_already_visible_in_the_slot(self) -> None:
        inventory = {
            "available": True,
            "terms": [
                {"term": "Canon 6D", "count": 9},
                {"term": "Nikon D750", "count": 8},
                {"term": "Sony 55mm", "count": 7},
            ],
        }
        options = slot_equipment_options(
            inventory, slot_index=0, limit=3, excluded=["6D", "D750"]
        )
        self.assertEqual(options, ["Sony 55mm"])

    def test_letter_initial_designators_are_one_token(self) -> None:
        # A pattern that stopped a letter-initial token at its first digit split
        # every such designator in half and then rejected both halves, so only
        # digit-initial names like "5D" survived. Three of four configured
        # domains name their products letter-first.
        inventory = build_entity_inventory(
            [
                thread(
                    "the Sony XM5 is comfortable",
                    "my Sony XM5 again",
                    "XM5 all day",
                    "grabbed a Sony a7III body",
                    "a Sony a7III too",
                    "the a7III handles low light",
                )
            ],
            brand_terms=("Sony",),
            min_occurrences=1,
        )
        designators = {row["designator"] for row in inventory["terms"]}
        self.assertIn("XM5", designators)
        self.assertIn("a7III", designators)

    def test_bare_specification_values_are_filtered(self) -> None:
        # "4GB" follows a brand once by accident and then appears many times on
        # its own; a real designator is brand-associated in a fair share of uses.
        bodies = ["a Dell 4GB stick maybe"] + ["4GB is not enough"] * 40
        inventory = build_entity_inventory(
            [thread(*bodies)], brand_terms=("Dell",), min_occurrences=1
        )
        self.assertNotIn(
            "4gb", {row["designator"].lower() for row in inventory["terms"]}
        )

    def test_unavailable_without_configured_brands(self) -> None:
        inventory = build_entity_inventory([thread("a 5D somewhere")], brand_terms=())
        self.assertFalse(inventory["available"])
        self.assertEqual(slot_equipment_options(inventory, slot_index=0), [])


if __name__ == "__main__":
    unittest.main()
