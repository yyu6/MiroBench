#!/usr/bin/env python3
"""Download and verify the official StanceRel disagreement checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

from common import REPO_ROOT


DOWNLOAD_URL = "https://drive.google.com/file/d/11YSO_BOpYCDR08FyxjpX3xi7M1O2LmRK/view"
MODEL_SHA256 = "419da4ca6232b994fbdf7e097ab79b4364a7a1c1291af88fead12637e714e658"
DEFAULT_TARGET = REPO_ROOT / "Stance_Rel" / "RoBERT_rel_1.5e-05"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.target.expanduser().resolve()
    model = target / "pytorch_model.bin"
    if model.is_file() and sha256(model) == MODEL_SHA256:
        print(f"[stance-ready] {target}")
        return
    if args.verify_only:
        raise SystemExit(f"StanceRel checkpoint missing or invalid: {target}")

    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("Install gdown or run setup.sh before downloading StanceRel.") from exc

    with tempfile.TemporaryDirectory(prefix="mirobench_stance_") as temp:
        temp_dir = Path(temp)
        archive = temp_dir / "stance_rel_download"
        result = gdown.download(url=DOWNLOAD_URL, output=str(archive), fuzzy=True, quiet=False)
        if not result or not archive.exists():
            raise SystemExit("Official StanceRel checkpoint download failed.")
        extracted = temp_dir / "extracted"
        extracted.mkdir()
        unpack(archive, extracted)
        candidates = sorted(extracted.rglob("pytorch_model.bin"))
        source_model = next((path for path in candidates if sha256(path) == MODEL_SHA256), None)
        if source_model is None:
            found = ", ".join(str(path.relative_to(extracted)) for path in candidates[:5])
            raise SystemExit(f"Downloaded StanceRel archive failed checksum validation. Found: {found}")
        target.mkdir(parents=True, exist_ok=True)
        for path in source_model.parent.iterdir():
            if path.is_file():
                shutil.copy2(path, target / path.name)

    if sha256(model) != MODEL_SHA256:
        raise SystemExit(f"StanceRel install verification failed: {model}")
    print(f"[stance-installed] {target}")


def unpack(archive: Path, output: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(output)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            handle.extractall(output)
        return
    raise SystemExit(f"Unsupported StanceRel archive format: {archive}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
