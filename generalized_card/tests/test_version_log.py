from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_version_log.py"
SPEC = importlib.util.spec_from_file_location("generalized_card_version_log", SCRIPT)
assert SPEC and SPEC.loader
VERSION_LOG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERSION_LOG)


def test_n1_metrics_are_not_counted_as_pvalue_passes() -> None:
    metrics = {
        metric: {
            "mwu_p_value": 1.0,
            "ks_p_value": 1.0,
            "inferential_status": "DESCRIPTIVE",
        }
        for metric in VERSION_LOG.METRICS
    }
    assert VERSION_LOG._pass_summary(metrics) == "descriptive"


def test_inferential_metrics_still_count_both_tests() -> None:
    metrics = {
        VERSION_LOG.METRICS[0]: {"mwu_p_value": 0.2, "ks_p_value": 0.3},
        VERSION_LOG.METRICS[1]: {"mwu_p_value": 0.2, "ks_p_value": 0.01},
    }
    assert VERSION_LOG._pass_summary(metrics) == "1"
