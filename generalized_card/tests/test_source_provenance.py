"""Tests for the pre-run check that a version is recoverable from git.

Each test builds a real throwaway repository rather than mocking `git`, because
the defect being guarded against was a wrong belief about what a git command
reports: `git ls-files` lists staged-but-uncommitted files, so the existing
"untracked active" check passed while two releases had no commit at all.
A mock would have encoded that same wrong belief.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from generalized_card import source_provenance as sp  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


class SourceProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "config", "user.email", "test@example.com")
        _git(self.repo, "config", "user.name", "test")
        (self.repo / "pkg").mkdir()
        self._write("pkg/committed.py", "x = 1\n")
        _git(self.repo, "add", "pkg/committed.py")
        _git(self.repo, "commit", "-q", "-m", "base")
        self._original_root = sp.REPO_ROOT
        sp.REPO_ROOT = self.repo
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        sp.REPO_ROOT = self._original_root
        os.environ.pop(sp.ALLOW_UNCOMMITTED_ENV, None)
        self._tmp.cleanup()

    def _write(self, relative: str, text: str) -> None:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    # --- the clean case -------------------------------------------------

    def test_committed_source_passes_and_records_the_commit(self) -> None:
        record = sp.verify_source_provenance(["pkg/committed.py"])
        self.assertEqual(record["uncommitted"], [])
        self.assertEqual(record["checked"], 1)
        self.assertEqual(record["branch"], "main")
        self.assertEqual(
            record["commit"], _git(self.repo, "rev-parse", "HEAD").strip()
        )
        self.assertNotIn("override", record)

    def test_the_recorded_commit_restores_the_exact_source(self) -> None:
        """The point of the record: it has to be enough to get the file back."""

        record = sp.verify_source_provenance(["pkg/committed.py"])
        self._write("pkg/committed.py", "x = 999\n")
        _git(self.repo, "add", "pkg/committed.py")
        _git(self.repo, "commit", "-q", "-m", "later change")
        restored = _git(
            self.repo, "show", f"{record['commit']}:pkg/committed.py"
        )
        self.assertEqual(restored, "x = 1\n")

    # --- the three ways a source can be missing from HEAD ---------------

    def test_modified_working_tree_is_uncommitted(self) -> None:
        self._write("pkg/committed.py", "x = 2\n")
        with self.assertRaises(RuntimeError) as caught:
            sp.verify_source_provenance(["pkg/committed.py"])
        self.assertIn("pkg/committed.py", str(caught.exception))

    def test_staged_but_never_committed_is_uncommitted(self) -> None:
        """The exact hole v97 and v98 fell through.

        `git ls-files` reports a staged file as tracked, so the existing
        untracked-active check saw nothing wrong.
        """

        self._write("pkg/staged.py", "y = 1\n")
        _git(self.repo, "add", "pkg/staged.py")
        self.assertIn("pkg/staged.py", _git(self.repo, "ls-files"))
        record = sp.source_provenance(["pkg/staged.py"])
        self.assertEqual(record["uncommitted"], ["pkg/staged.py"])
        with self.assertRaises(RuntimeError):
            sp.verify_source_provenance(["pkg/staged.py"])

    def test_untracked_is_uncommitted(self) -> None:
        self._write("pkg/untracked.py", "z = 1\n")
        record = sp.source_provenance(["pkg/untracked.py"])
        self.assertEqual(record["uncommitted"], ["pkg/untracked.py"])

    def test_deleted_source_is_uncommitted(self) -> None:
        (self.repo / "pkg" / "committed.py").unlink()
        record = sp.source_provenance(["pkg/committed.py"])
        self.assertEqual(record["uncommitted"], ["pkg/committed.py"])

    # --- reporting ------------------------------------------------------

    def test_only_the_listed_paths_are_checked(self) -> None:
        self._write("pkg/unrelated.py", "w = 1\n")
        record = sp.verify_source_provenance(["pkg/committed.py"])
        self.assertEqual(record["uncommitted"], [])

    def test_every_uncommitted_kind_is_reported_together(self) -> None:
        self._write("pkg/committed.py", "x = 2\n")
        self._write("pkg/staged.py", "y = 1\n")
        _git(self.repo, "add", "pkg/staged.py")
        self._write("pkg/untracked.py", "z = 1\n")
        record = sp.source_provenance(
            ["pkg/committed.py", "pkg/staged.py", "pkg/untracked.py"]
        )
        self.assertEqual(
            record["uncommitted"],
            ["pkg/committed.py", "pkg/staged.py", "pkg/untracked.py"],
        )

    def test_the_error_names_the_override_and_the_commit(self) -> None:
        self._write("pkg/committed.py", "x = 2\n")
        with self.assertRaises(RuntimeError) as caught:
            sp.verify_source_provenance(["pkg/committed.py"])
        message = str(caught.exception)
        self.assertIn(sp.ALLOW_UNCOMMITTED_ENV, message)
        self.assertIn("not be reproducible", message)

    def test_long_lists_are_truncated_with_a_remainder_count(self) -> None:
        names = [f"pkg/f{index:02d}.py" for index in range(25)]
        for name in names:
            self._write(name, "v = 1\n")
        with self.assertRaises(RuntimeError) as caught:
            sp.verify_source_provenance(names)
        self.assertIn("and 5 more", str(caught.exception))

    # --- the override ---------------------------------------------------

    def test_override_allows_the_run_and_is_recorded(self) -> None:
        self._write("pkg/committed.py", "x = 2\n")
        os.environ[sp.ALLOW_UNCOMMITTED_ENV] = "1"
        record = sp.verify_source_provenance(["pkg/committed.py"])
        self.assertTrue(record["override"])
        self.assertEqual(record["uncommitted"], ["pkg/committed.py"])

    def test_any_other_override_value_still_raises(self) -> None:
        self._write("pkg/committed.py", "x = 2\n")
        for value in ("0", "true", "yes", ""):
            os.environ[sp.ALLOW_UNCOMMITTED_ENV] = value
            with self.assertRaises(RuntimeError):
                sp.verify_source_provenance(["pkg/committed.py"])

    # --- degradation ----------------------------------------------------

    def test_no_repository_reports_unknown_rather_than_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as bare:
            sp.REPO_ROOT = Path(bare)
            record = sp.source_provenance(["pkg/committed.py"])
        self.assertEqual(record["commit"], "unknown")
        self.assertEqual(record["uncommitted"], [])

    def test_empty_path_list_is_a_clean_no_op(self) -> None:
        record = sp.verify_source_provenance([])
        self.assertEqual(record["checked"], 0)
        self.assertEqual(record["uncommitted"], [])

    def test_blank_paths_are_dropped(self) -> None:
        record = sp.source_provenance(["", "pkg/committed.py", ""])
        self.assertEqual(record["checked"], 1)


class RunGenerateWiringTest(unittest.TestCase):
    """The check has to be on the active path, not merely importable."""

    def test_run_generate_verifies_before_recording(self) -> None:
        source = (PACKAGE_ROOT / "scripts" / "run_generate.py").read_text()
        self.assertIn("verify_source_provenance", source)
        self.assertIn('"source_provenance": source_record', source)
        verify_at = source.index("source_record = verify_source_provenance")
        seed_at = source.index("if not seed_pool.exists():")
        self.assertLess(
            verify_at,
            seed_at,
            "the provenance gate must run before any run setup work",
        )

    def test_every_pinned_generation_source_is_checked(self) -> None:
        from generalized_card.core_contract import (
            CORE_FILES,
            CURRENT_GENERATION_CORE_NAMES,
            version_source_paths,
        )

        paths = version_source_paths(CURRENT_GENERATION_CORE_NAMES)
        self.assertEqual(len(paths), len(set(paths)))
        for name in CURRENT_GENERATION_CORE_NAMES:
            self.assertIn(CORE_FILES[name][0], paths)
        record = sp.source_provenance(paths)
        self.assertEqual(record["checked"], len(set(paths)))

    def test_the_contract_itself_is_checked(self) -> None:
        """It cannot hold its own hash, so provenance is the only check it gets."""

        from generalized_card.core_contract import (
            CONTRACT_RELATIVE_PATH,
            CORE_FILES,
            CURRENT_GENERATION_CORE_NAMES,
            version_source_paths,
        )

        self.assertNotIn(
            CONTRACT_RELATIVE_PATH,
            {entry[0] for entry in CORE_FILES.values()},
        )
        self.assertIn(
            CONTRACT_RELATIVE_PATH,
            version_source_paths(CURRENT_GENERATION_CORE_NAMES),
        )

    def test_this_repository_is_currently_reproducible(self) -> None:
        """A live guard: if this fails, the working tree has an unshipped version."""

        from generalized_card.core_contract import (
            CURRENT_GENERATION_CORE_NAMES,
            version_source_paths,
        )

        record = sp.source_provenance(
            version_source_paths(CURRENT_GENERATION_CORE_NAMES)
        )
        self.assertEqual(
            record["uncommitted"],
            [],
            "commit the current version before generating; see "
            "docs/ORIENTATION.md section 8",
        )


if __name__ == "__main__":
    unittest.main()
