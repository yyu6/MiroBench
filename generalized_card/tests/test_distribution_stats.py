from __future__ import annotations

import unittest

import pandas as pd

from generalized_card.distribution_stats import (
    cliffs_delta,
    distribution_stats,
    evaluate_group_vs_real,
)


class DistributionStatsTest(unittest.TestCase):
    def test_identical_distributions_have_zero_distance(self) -> None:
        result = distribution_stats([0.0, 1.0], [0.0, 1.0])
        self.assertEqual(result["mwu_p_value"], 1.0)
        self.assertEqual(result["ks_p_value"], 1.0)
        self.assertEqual(result["cliffs_delta"], 0.0)
        self.assertEqual(result["wasserstein_distance"], 0.0)

    def test_cliffs_delta_direction_is_candidate_minus_real(self) -> None:
        self.assertEqual(cliffs_delta([2.0, 3.0], [0.0, 1.0]), 1.0)
        self.assertEqual(cliffs_delta([0.0, 1.0], [2.0, 3.0]), -1.0)

    def test_group_report_keeps_formal_distance_fields(self) -> None:
        report = evaluate_group_vs_real(
            pd.DataFrame({"metric": [0.0, 1.0, float("nan")]}),
            pd.DataFrame({"metric": [0.0, 1.0]}),
            ["metric", "missing"],
        )
        self.assertEqual(report["metric"]["direction"], "similar")
        self.assertEqual(report["metric"]["quantile_error"], 0.0)
        self.assertEqual(report["metric"]["real_mean"], 0.5)
        self.assertEqual(report["missing"]["mwu_p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
