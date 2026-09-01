from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = (
    ROOT
    / "experiments"
    / "reddit_multidomain_baselines"
    / "config"
    / "models.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracker_bills_gemini_thinking_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _load_module(
        "provider_cost_token_tracker", ROOT / "scripts" / "token_usage_tracker.py"
    )
    log_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("TOKEN_USAGE_LOG_JSONL", str(log_path))
    monkeypatch.setenv("TOKEN_PRICE_INPUT_PER_1M", "0.30")
    monkeypatch.setenv("TOKEN_PRICE_CACHED_INPUT_PER_1M", "0.03")
    monkeypatch.setenv("TOKEN_PRICE_OUTPUT_PER_1M", "2.50")

    tracker.record_openai_usage(
        {
            "id": "gemini-cost-test",
            "usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 10},
                "completion_tokens": 20,
                "total_tokens": 170,
            },
        },
        model="gemini-2.5-flash",
        component="test",
    )

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["completion_tokens"] == 20
    assert record["reasoning_tokens"] == 50
    assert record["billable_output_tokens"] == 70
    assert record["estimated_cost_usd"] == pytest.approx(
        (90 * 0.30 + 10 * 0.03 + 70 * 2.50) / 1_000_000
    )


def test_tracker_does_not_double_bill_openai_reasoning_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _load_module(
        "provider_cost_openai_tracker", ROOT / "scripts" / "token_usage_tracker.py"
    )
    log_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("TOKEN_USAGE_LOG_JSONL", str(log_path))
    monkeypatch.setenv("TOKEN_PRICE_INPUT_PER_1M", "0.75")
    monkeypatch.setenv("TOKEN_PRICE_CACHED_INPUT_PER_1M", "0.075")
    monkeypatch.setenv("TOKEN_PRICE_OUTPUT_PER_1M", "4.50")

    tracker.record_openai_usage(
        {
            "id": "openai-cost-test",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 10},
                "total_tokens": 120,
            },
        },
        model="gpt-5.4-mini",
        component="test",
    )

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["reasoning_tokens"] == 10
    assert record["billable_output_tokens"] == 20
    assert record["estimated_cost_usd"] == pytest.approx(
        (100 * 0.75 + 20 * 4.50) / 1_000_000
    )


def test_tracker_reads_deepseek_top_level_cache_hit_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _load_module(
        "provider_cost_deepseek_tracker", ROOT / "scripts" / "token_usage_tracker.py"
    )
    log_path = tmp_path / "usage.jsonl"
    monkeypatch.setenv("TOKEN_USAGE_LOG_JSONL", str(log_path))
    monkeypatch.setenv("TOKEN_PRICE_INPUT_PER_1M", "0.44")
    monkeypatch.setenv("TOKEN_PRICE_CACHED_INPUT_PER_1M", "0.014")
    monkeypatch.setenv("TOKEN_PRICE_OUTPUT_PER_1M", "1.32")

    tracker.record_openai_usage(
        {
            "id": "deepseek-cost-test",
            "usage": {
                "prompt_tokens": 100,
                "prompt_cache_hit_tokens": 80,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        },
        model="deepseek-v4-flash",
        component="test",
    )

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["cached_prompt_tokens"] == 80
    assert record["billable_output_tokens"] == 20
    assert record["estimated_cost_usd"] == pytest.approx(
        (20 * 0.44 + 80 * 0.014 + 20 * 1.32) / 1_000_000
    )


def test_summary_reprices_old_gemini_logs_with_inferred_thinking_tokens(
    tmp_path: Path,
) -> None:
    common = _load_module(
        "provider_cost_common",
        ROOT / "experiments" / "reddit_multidomain_baselines" / "scripts" / "common.py",
    )
    log_path = tmp_path / "old_gemini_usage.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "model": "gemini-2.5-flash",
                "component": "test",
                "prompt_tokens": 100,
                "cached_prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 170,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    model_spec = {
        "pricing": {
            "input_per_million_usd": 0.30,
            "cached_input_per_million_usd": 0.03,
            "output_per_million_usd": 2.50,
        }
    }

    summary = common.summarize_usage(log_path, model_spec)

    assert summary["reasoning_tokens"] == 50
    assert summary["billable_output_tokens"] == 70
    assert summary["estimated_cost_usd"] == pytest.approx(
        (90 * 0.30 + 10 * 0.03 + 70 * 2.50) / 1_000_000
    )


def test_old_non_gemini_logs_keep_completion_as_billable_output() -> None:
    common = _load_module(
        "provider_cost_non_gemini_common",
        ROOT / "experiments" / "reddit_multidomain_baselines" / "scripts" / "common.py",
    )
    record = {
        "model": "gpt-5.4-mini",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 170,
    }
    model_spec = {
        "pricing": {
            "input_per_million_usd": 0.75,
            "cached_input_per_million_usd": 0.075,
            "output_per_million_usd": 4.50,
        }
    }

    assert common.estimate_record_cost(record, model_spec) == pytest.approx(
        (100 * 0.75 + 20 * 4.50) / 1_000_000
    )


def test_verified_standard_model_prices() -> None:
    models = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))["models"]

    assert models["gemini-2.5-flash"]["pricing"] == {
        "input_per_million_usd": 0.30,
        "cached_input_per_million_usd": 0.03,
        "output_per_million_usd": 2.50,
    }
    assert models["gpt-4o-mini"]["pricing"] == {
        "input_per_million_usd": 0.15,
        "cached_input_per_million_usd": 0.075,
        "output_per_million_usd": 0.60,
    }
    assert models["gpt-5.4-mini"]["pricing"] == {
        "input_per_million_usd": 0.75,
        "cached_input_per_million_usd": 0.075,
        "output_per_million_usd": 4.50,
    }
    deepseek = models["deepseek-v4-flash"]["pricing"]
    assert deepseek["peak_hours_utc"] == [1, 2, 3, 6, 7, 8, 9]
    assert deepseek["tiers"]["peak"] == {
        "input_per_million_usd": 0.44,
        "cached_input_per_million_usd": 0.014,
        "output_per_million_usd": 1.32,
    }
    assert deepseek["tiers"]["off_peak"] == {
        "input_per_million_usd": 0.22,
        "cached_input_per_million_usd": 0.007,
        "output_per_million_usd": 0.66,
    }
