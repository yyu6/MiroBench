"""Refuse to spend a paid run on sources that git cannot give back.

The project already had four provenance mechanisms and none of them stored the
source tree. `run_config.json` records the arms and the policy string,
`core_contract.verify_core_contract` pins every active source's SHA-256 and
raises on drift, `HISTORICAL_GENERATION_POLICY_VERSIONS` records released policy
strings, and `repin_core_contract.py` refuses to pin a file that `git ls-files`
does not know about.

That last check is the near miss. `git ls-files` lists *tracked* files, and a
staged-but-never-committed file is tracked. v97 and v98 both shipped that way:
`untracked active: 0` was true the whole time while `git log` on
`sentence_rhythm.py` was empty, so two releases -- including the one whose N=10
result was being quoted as the project's state -- had no recoverable source
tree. A hash answers "has this drifted?"; it never answers "can this be
recovered?"

So this module asks the second question, at the only moment where the answer
still matters: before the API calls. It compares each pinned source's
working-tree content against the same path in `HEAD`, and it records the commit
in `run_config.json` so the chain runs artifact -> commit -> sources with no
searching.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Iterable

from .domain import REPO_ROOT


ALLOW_UNCOMMITTED_ENV = "GENERALIZED_CARD_ALLOW_UNCOMMITTED_SOURCE"


def source_provenance(paths: Iterable[str]) -> dict[str, Any]:
    """Report the commit these sources came from and which ones are not in it.

    `uncommitted` covers all three ways a source can be absent from `HEAD`:
    modified in the working tree, staged but not committed, and untracked. The
    first two come from `git diff HEAD`, which compares the working tree to the
    commit rather than to the index, and the third from `ls-files --others`.
    """

    relative = sorted({str(path) for path in paths if str(path)})
    return {
        "commit": _git("rev-parse", "HEAD") or "unknown",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "uncommitted": sorted(
            set(_git_lines("diff", "--name-only", "HEAD", "--", *relative))
            | set(
                _git_lines(
                    "ls-files", "--others", "--exclude-standard", "--", *relative
                )
            )
        ),
        "checked": len(relative),
    }


def verify_source_provenance(paths: Iterable[str]) -> dict[str, Any]:
    """Return the provenance record, raising unless every source is committed.

    The override is an environment variable rather than a CLI flag so it cannot
    be set by accident in a long generation command, and it is recorded in the
    returned record so a run made without provenance says so in its own
    `run_config.json` instead of looking like every other run.
    """

    record = source_provenance(paths)
    if not record["uncommitted"]:
        return record
    record["override"] = os.environ.get(ALLOW_UNCOMMITTED_ENV) == "1"
    if record["override"]:
        return record
    listed = "\n  ".join(record["uncommitted"][:20])
    extra = (
        f"\n  ... and {len(record['uncommitted']) - 20} more"
        if len(record["uncommitted"]) > 20
        else ""
    )
    raise RuntimeError(
        f"{len(record['uncommitted'])} pinned source(s) are not in commit "
        f"{record['commit'][:8]}, so this run would not be reproducible:\n  "
        f"{listed}{extra}\n"
        "Commit the version before generating -- the pinned hashes prove these "
        "files have not drifted, but only a commit stores them. To generate "
        "anyway and accept that the sources are unrecoverable, set "
        f"{ALLOW_UNCOMMITTED_ENV}=1; the override is recorded in run_config.json."
    )


def _git(*args: str) -> str:
    return _run(args).strip()


def _git_lines(*args: str) -> list[str]:
    return [line for line in _run(args).splitlines() if line.strip()]


def _run(args: Iterable[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return completed.stdout if completed.returncode == 0 else ""
