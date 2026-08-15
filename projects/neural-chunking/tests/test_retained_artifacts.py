"""Integrity checks for retained result and figure artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "SHA256SUMS"
ARTIFACT_DIRECTORIES = (
    PROJECT_ROOT / "artifacts" / "figures",
    PROJECT_ROOT / "artifacts" / "results",
)


def _sha256(path: Path) -> str:
    """Hash a file without loading the complete artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_retained_artifact_manifest_is_complete_and_valid() -> None:
    """Every retained artifact should be listed once and match its pinned digest."""
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        MANIFEST_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        digest, separator, relative_path = line.partition("  ")
        assert separator, f"manifest line {line_number} must use two-space separation"
        assert len(digest) == 64 and digest == digest.lower()
        assert set(digest) <= set("0123456789abcdef")
        relative = Path(relative_path)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert relative_path not in entries, f"duplicate manifest path: {relative_path}"
        entries[relative_path] = digest

    retained_paths = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for directory in ARTIFACT_DIRECTORIES
        for path in directory.rglob("*")
        if path.is_file()
    }
    assert set(entries) == retained_paths
    for relative_path, expected_digest in entries.items():
        artifact_path = PROJECT_ROOT / relative_path
        assert not artifact_path.is_symlink()
        assert _sha256(artifact_path) == expected_digest


def test_retained_metrics_use_plain_run_status() -> None:
    """The public result record uses a concise status and does not imply provenance."""
    import json

    metrics_path = PROJECT_ROOT / "artifacts" / "results" / "retained_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["status"] == "retained_run"
    assert "historical" not in json.dumps(metrics).lower()
