from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "experiments" / "reddit_multidomain_baselines" / "scripts"
SYNTHPAI_OVERRIDE = (
    ROOT
    / "experiments"
    / "reddit_multidomain_baselines"
    / "vendor_overrides"
    / "SynthPAI"
    / "src"
    / "models"
    / "open_ai.py"
)


def load_run_generation():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "reddit_multidomain_run_generation",
        SCRIPTS / "run_generation.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_synthpai_generator():
    spec = importlib.util.spec_from_file_location(
        "matched_synthpai_generator",
        ROOT / "scripts" / "run_synthpai_matched_seed_generator.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_gemini_synthpai_forces_one_post_per_run() -> None:
    module = load_run_generation()

    assert (
        module.effective_posts_per_run(
            baseline="synthpai", model="gemini-2.5-flash", requested=5
        )
        == 1
    )
    assert (
        module.effective_posts_per_run(
            baseline="synthpai", model="gpt-4o-mini", requested=5
        )
        == 5
    )
    assert (
        module.effective_posts_per_run(
            baseline="oasis", model="gemini-2.5-flash", requested=5
        )
        == 5
    )


def test_only_gemini_synthpai_strips_base_url_trailing_slash() -> None:
    module = load_run_generation()
    gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

    assert module.effective_base_url(
        baseline="synthpai", model="gemini-2.5-flash", configured=gemini_url
    ) == gemini_url.rstrip("/")
    assert (
        module.effective_base_url(
            baseline="oasis", model="gemini-2.5-flash", configured=gemini_url
        )
        == gemini_url
    )


def test_gemini_synthpai_removes_unsupported_frequency_penalty() -> None:
    source = SYNTHPAI_OVERRIDE.read_text(encoding="utf-8")
    gemini_branch = source.split(
        'elif "gemini" in self.config.name.lower():', maxsplit=1
    )[1].split('elif "max_tokens" not in self.config.args.keys():', maxsplit=1)[0]

    assert 'pop("frequency_penalty"' in gemini_branch
    assert 'setdefault("max_tokens", 600)' in gemini_branch
    assert 'self.config.args["reasoning_effort"] = "none"' in gemini_branch


def test_only_gemini_25_flash_job_env_disables_thinking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_run_generation()
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    model_spec = {
        "key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
    }

    gemini_env = module._job_env(
        model="gemini-2.5-flash",
        model_spec=model_spec,
        usage_path=tmp_path / "gemini.jsonl",
        baseline="synthpai",
        domain="camera",
        allow_missing_key=True,
    )
    flash_lite_env = module._job_env(
        model="gemini-2.5-flash-lite",
        model_spec=model_spec,
        usage_path=tmp_path / "flash-lite.jsonl",
        baseline="synthpai",
        domain="camera",
        allow_missing_key=True,
    )

    assert gemini_env["LLM_REASONING_EFFORT"] == "none"
    assert "LLM_REASONING_EFFORT" not in flash_lite_env


def test_profile_selection_failure_uses_empty_fallback(capsys) -> None:
    module = load_synthpai_generator()

    class RefusingThread:
        def choose_profiles(self, *args, **kwargs):
            raise RuntimeError(
                "Provider returned no message content (finish_reason=content_filter)"
            )

    selected = module._choose_profiles_safely(
        thread=RefusingThread(),
        checker_model=object(),
        profile_checker_prompt="checker",
        available_profiles={"user_1": {}},
        root_text="seed post",
        no_profiles=1,
        seed_index=21,
    )

    assert selected == {}
    output = capsys.readouterr().out
    assert "[synthpai-profile-fallback] seed=21" in output
    assert "finish_reason=content_filter" in output


def test_profile_selection_does_not_hide_unrelated_failure() -> None:
    module = load_synthpai_generator()

    class BrokenThread:
        def choose_profiles(self, *args, **kwargs):
            raise RuntimeError("database is unavailable")

    with pytest.raises(RuntimeError, match="database is unavailable"):
        module._choose_profiles_safely(
            thread=BrokenThread(),
            checker_model=object(),
            profile_checker_prompt="checker",
            available_profiles={"user_1": {}},
            root_text="seed post",
            no_profiles=1,
            seed_index=21,
        )


def test_keyboard_interrupt_is_not_recorded_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_run_generation()
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    monkeypatch.setattr(
        module,
        "_run_logged",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    args = argparse.Namespace(
        force=False,
        force_template=False,
        dry_run=False,
        max_seeds=150,
        posts_per_run=5,
        synthpai_python=Path(sys.executable),
        synthpai_config="configs/thread/thread_gpt4omini_city_country.yaml",
        seed=20260828,
        thread_retries=1,
        retry_delay=60.0,
        synthpai_min_comments_per_post=1,
        oasis_min_comments_per_post=0,
    )
    model_spec = {
        "key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "pricing_basis": "test",
        "pricing": {
            "input_per_million_usd": 0.0,
            "cached_input_per_million_usd": 0.0,
            "output_per_million_usd": 0.0,
        },
    }

    with pytest.raises(KeyboardInterrupt):
        module.run_job(
            args=args,
            output_root=tmp_path,
            baseline="synthpai",
            model="gemini-2.5-flash",
            domain="camera",
            model_spec=model_spec,
        )

    report = json.loads(
        (
            tmp_path
            / "generation"
            / "synthpai"
            / "gemini-2.5-flash"
            / "camera"
            / "generation_report.json"
        ).read_text()
    )
    assert report["status"] == "interrupted"
    assert report["posts_per_run"] == 1
    assert report["requested_posts_per_run"] == 5
    assert report["base_url"].endswith("/openai")
    assert report["configured_base_url"].endswith("/openai/")
