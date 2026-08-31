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


def test_gemini_synthpai_preserves_frequency_penalty() -> None:
    source = SYNTHPAI_OVERRIDE.read_text(encoding="utf-8")
    gemini_branch = source.split(
        'elif "gemini" in self.config.name.lower():', maxsplit=1
    )[1].split('elif "max_tokens" not in self.config.args.keys():', maxsplit=1)[0]

    assert 'pop("frequency_penalty"' not in gemini_branch
    assert 'setdefault("max_tokens", 600)' in gemini_branch


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
