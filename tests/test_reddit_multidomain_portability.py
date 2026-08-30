from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "reddit_multidomain_baselines"
PORTABLE = EXPERIMENT / "portable_inputs"
EXPECTED_DOMAINS = {
    "camera",
    "celebrity",
    "cellphone",
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
    "length_std",
    "length_cv",
    "avg_depth",
    "structural_virality",
    "self_bleu_2",
    "self_bleu_3",
    "self_bleu_4",
    "self_bertscore_mean_f1",
    "semantic_mean_cosine",
    "hard_disagree_rate",
    "impolite_rate",
    "neutral_rate",
    "polite_rate",
    "mean_story_probability",
    "emotion_entropy",
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


def test_portable_references_do_not_expose_reddit_authors_or_local_paths() -> None:
    author_pattern = re.compile(r"^(?:reddit_user_[0-9a-f]{12}|\[deleted\])$")
    for path in (PORTABLE / "real_reference").rglob("discussion.json"):
        text = path.read_text()
        assert "/Users/" not in text
        assert re.search(r"(?<![A-Za-z0-9_])/?u/[A-Za-z0-9_-]{2,32}", text, re.I) is None
        payload = json.loads(text)
        for author in collect_values(payload, "author"):
            assert author_pattern.fullmatch(str(author)), (path, author)


def test_evaluation_core_metric_whitelist_matches_four_families() -> None:
    sys.path.insert(0, str(EXPERIMENT / "scripts"))
    module = load_module(
        "reddit_multidomain_evaluate",
        EXPERIMENT / "scripts" / "evaluate.py",
    )
    assert module.CORE_METRICS == EXPECTED_CORE_METRICS


def collect_values(value, key: str):
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                yield child
            yield from collect_values(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from collect_values(child, key)
