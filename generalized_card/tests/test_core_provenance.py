from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

from generalized_card.core_contract import CURRENT_ACTIVE_CORE_NAMES
from generalized_card.domain import REPO_ROOT


SCRIPT = Path(__file__).parents[1] / "scripts" / "repin_core_contract.py"
SPEC = importlib.util.spec_from_file_location("generalized_card_repin", SCRIPT)
assert SPEC and SPEC.loader
REPIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPIN)


class CoreProvenanceTest(unittest.TestCase):
    def test_dynamic_runtime_sources_are_active(self) -> None:
        for name in (
            "generation_backend_runner",
            "output_audit_runner",
            "token_usage_tracker",
            "token_usage_summarizer",
        ):
            self.assertIn(name, CURRENT_ACTIVE_CORE_NAMES)

    def test_sibling_script_import_is_resolved(self) -> None:
        source = REPO_ROOT / "scripts" / "summarize_token_usage.py"
        node = ast.ImportFrom(
            module="token_usage_tracker",
            names=[ast.alias(name="price_for_model")],
            level=0,
        )
        self.assertEqual(
            REPIN.resolve_local_imports(source, node),
            [REPO_ROOT / "scripts" / "token_usage_tracker.py"],
        )

    def test_active_local_import_closure_is_fully_pinned(self) -> None:
        self.assertEqual(REPIN.unpinned_local_imports(), [])


if __name__ == "__main__":
    unittest.main()
