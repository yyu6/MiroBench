from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "reddit_multidomain_baselines"
PORTABLE = EXPERIMENT / "portable_inputs"
EXPECTED_DOMAINS = {
    "camera",
    "celebrity",
    "cellphone",
    "credit_cards",
    "game",
    "headphones",
    "health_issue",
    "laptop",
    "movies",
    "news",
    "sports",
    "tv_series",
}
EXPECTED_CORE_METRICS = {
    "self_bleu_4",
    "self_bertscore_mean_f1",
    "semantic_mean_cosine",
    "mean_story_probability",
    "emotion_entropy",
    "impolite_rate",
    "neutral_rate",
    "polite_rate",
    "avg_depth",
    "hard_disagree_rate",
    "structural_virality",
    "length_cv",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portable_input_manifest_and_seed_counts() -> None:
    manifest = json.loads((PORTABLE / "portable_inputs_manifest.json").read_text())
    assert set(manifest["domains"]) == EXPECTED_DOMAINS
    for record in manifest["files"]:
        path = PORTABLE / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    for domain in EXPECTED_DOMAINS:
        pool = json.loads((PORTABLE / "seed_pools" / f"{domain}.json").read_text())
        reference = json.loads(
            (PORTABLE / "real_reference" / domain / "reference_manifest.json").read_text()
        )
        assert len(pool["seed_posts"]) == 150
        assert reference["seed_count"] == 150
        assert reference["run_count"] == 30

    credit_reference = json.loads(
        (
            PORTABLE
            / "real_reference"
            / "credit_cards"
            / "reference_manifest.json"
        ).read_text()
    )
    assert credit_reference["real_comment_count"] == 4288


def test_portable_references_do_not_expose_reddit_authors_or_local_paths() -> None:
    author_pattern = re.compile(r"^(?:reddit_user_[0-9a-f]{12}|\[deleted\])$")
    for path in (PORTABLE / "real_reference").rglob("discussion.json"):
        text = path.read_text()
        assert "/Users/" not in text
        assert re.search(r"(?<![A-Za-z0-9_])/?u/[A-Za-z0-9_-]{2,32}", text, re.I) is None
        payload = json.loads(text)
        for author in collect_values(payload, "author"):
            assert author_pattern.fullmatch(str(author)), (path, author)


def test_evaluation_core_metric_whitelist_matches_five_families() -> None:
    sys.path.insert(0, str(EXPERIMENT / "scripts"))
    module = load_module(
        "reddit_multidomain_evaluate",
        EXPERIMENT / "scripts" / "evaluate.py",
    )
    assert module.CORE_METRICS == EXPECTED_CORE_METRICS


def test_real_vs_real_sanity_uses_exact_12_metric_contract() -> None:
    module = load_module(
        "reddit_multidomain_real_sanity",
        EXPERIMENT / "scripts" / "real_vs_real_sanity.py",
    )
    assert len(module.METRICS) == 12
    assert set(module.METRICS) == EXPECTED_CORE_METRICS
    assert [family for family, _ in module.METRIC_FAMILIES] == [
        "Uniformity",
        "Expression",
        "Tone",
        "Interaction",
        "Form",
    ]


def test_real_vs_real_sanity_sampling_contracts() -> None:
    module = load_module(
        "reddit_multidomain_real_sanity_sampling",
        EXPERIMENT / "scripts" / "real_vs_real_sanity.py",
    )
    left, right = module.sample_indices(
        150, 150, "bootstrap", np.random.default_rng(7)
    )
    assert len(left) == len(right) == 150
    assert len(set(left)) < 150 or len(set(right)) < 150

    left, right = module.sample_indices(
        300, 150, "disjoint", np.random.default_rng(7)
    )
    assert len(left) == len(right) == 150
    assert not (set(left) & set(right))


def test_evaluation_summary_keeps_matched_pair_and_two_sample_rows(tmp_path: Path) -> None:
    sys.path.insert(0, str(EXPERIMENT / "scripts"))
    module = load_module(
        "reddit_multidomain_evaluate_summary_merge",
        EXPERIMENT / "scripts" / "evaluate.py",
    )
    path = tmp_path / "evaluation_summary.csv"
    identity = {
        "baseline": "geo",
        "model": "gpt-5.4-mini",
        "domain": "camera",
        "metric": "avg_depth",
    }
    module.write_csv(path, [{**identity, "test": "matched_pair", "real_mean": 2.2}])

    kept, merged = module.merge_evaluation_summary(
        path,
        [{**identity, "test": "two_sample", "real_mean": 1.3}],
    )

    assert kept == 1
    assert len(merged) == 2
    assert {row["test"] for row in merged} == {"matched_pair", "two_sample"}


def test_evaluation_summary_normalizes_legacy_blank_two_sample_rows(tmp_path: Path) -> None:
    sys.path.insert(0, str(EXPERIMENT / "scripts"))
    module = load_module(
        "reddit_multidomain_evaluate_legacy_summary_merge",
        EXPERIMENT / "scripts" / "evaluate.py",
    )
    path = tmp_path / "evaluation_summary.csv"
    identity = {
        "baseline": "oasis",
        "model": "gpt-5.4-mini",
        "domain": "camera",
        "metric": "avg_depth",
    }
    module.write_csv(path, [{**identity, "test": "", "real_mean": 1.0}])

    kept, merged = module.merge_evaluation_summary(
        path,
        [{**identity, "test": "two_sample", "real_mean": 2.0}],
    )

    assert kept == 0
    assert len(merged) == 1
    assert merged[0]["test"] == "two_sample"
    assert merged[0]["real_mean"] == 2.0


def test_evaluation_summary_drops_retired_diagnostic_metrics(tmp_path: Path) -> None:
    sys.path.insert(0, str(EXPERIMENT / "scripts"))
    module = load_module(
        "reddit_multidomain_evaluate_retired_metrics",
        EXPERIMENT / "scripts" / "evaluate.py",
    )
    path = tmp_path / "evaluation_summary.csv"
    base = {
        "baseline": "oasis",
        "model": "gpt-4o-mini",
        "domain": "camera",
        "test": "two_sample",
    }
    module.write_csv(
        path,
        [
            {**base, "metric": "avg_depth", "real_mean": 1.2},
            {**base, "metric": "length_std", "real_mean": 9.9},
        ],
    )
    kept, merged = module.merge_evaluation_summary(
        path,
        [{**base, "metric": "self_bleu_4", "real_mean": 0.1}],
    )
    assert kept == 1
    assert {row["metric"] for row in merged} == {"avg_depth", "self_bleu_4"}


def collect_values(value, key: str):
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                yield child
            yield from collect_values(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from collect_values(child, key)
