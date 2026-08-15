"""Validate and copy retained evidence into the built Python package."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class EvidenceBuildError(RuntimeError):
    """Raised when the retained evidence inventory cannot be packaged safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retained_entries(source_root: Path) -> tuple[tuple[Path, str], ...]:
    """Return the unique, safe retained-file inventory declared by the manifest."""

    manifest_path = source_root / "results_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceBuildError("Cannot read the retained evidence manifest") from exc
    retained = payload.get("retained_files") if isinstance(payload, dict) else None
    if not isinstance(retained, list) or not retained:
        raise EvidenceBuildError("The retained evidence manifest has no files")

    entries: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for item in retained:
        if not isinstance(item, dict):
            raise EvidenceBuildError("A retained evidence entry is not an object")
        relative = Path(str(item.get("path", "")))
        expected_hash = str(item.get("sha256", ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise EvidenceBuildError("A retained evidence path is unsafe")
        if relative in seen:
            raise EvidenceBuildError(f"Duplicate retained evidence path: {relative.as_posix()}")
        if not SHA256_PATTERN.fullmatch(expected_hash):
            raise EvidenceBuildError(f"Invalid retained evidence hash: {relative.as_posix()}")
        source = source_root / relative
        if not source.is_file() or _sha256(source) != expected_hash:
            raise EvidenceBuildError(f"Retained evidence hash differs: {relative.as_posix()}")
        seen.add(relative)
        entries.append((relative, expected_hash))
    return tuple(entries)


def copy_retained_evidence(source_root: Path, package_root: Path) -> tuple[Path, ...]:
    """Copy the verified manifest and every retained file into a build directory."""

    evidence_root = package_root / "_evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    copied = [Path("results_manifest.json")]
    shutil.copyfile(source_root / "results_manifest.json", evidence_root / "results_manifest.json")
    for relative, _expected_hash in retained_entries(source_root):
        destination = evidence_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, destination)
        copied.append(relative)
    return tuple(copied)
