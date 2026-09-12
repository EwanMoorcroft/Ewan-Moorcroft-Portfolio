"""Validate output identities and publish staged files with rollback on I/O errors."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from .errors import DataContractError


def _same_file(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and left.samefile(right)
    )


def validate_output_paths(outputs: Sequence[Path], protected: Sequence[Path] = ()) -> None:
    """Reject path, symlink and hardlink aliases before changing any output."""
    for index, output in enumerate(outputs):
        if output.exists() and not output.is_file():
            raise DataContractError("Output destination must be a regular file")
        if any(_same_file(output, source) for source in protected):
            raise DataContractError("Output paths cannot overwrite an artifact, config, or input")
        if any(_same_file(output, other) for other in outputs[:index]):
            raise DataContractError("Output paths must be different files")


def _stage(destination: Path, raw: bytes) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".fpl-", delete=False
    ) as handle:
        staged = Path(handle.name)
        try:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
    return staged


def publish_files(outputs: Mapping[Path, bytes]) -> None:
    """Stage all files first, atomically replace each, and roll back ordinary failures.

    Mapping order is publication order: publish an integrity manifest last. Two
    arbitrary paths cannot be replaced in a single filesystem transaction; readers
    must verify the manifest hash and retry if they observe a concurrent update.
    """
    validate_output_paths(list(outputs))
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        for destination, raw in outputs.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged[destination] = _stage(destination, raw)
            if destination.exists():
                backups[destination] = _stage(destination, destination.read_bytes())
        for destination, source in staged.items():
            os.replace(source, destination)
            replaced.append(destination)
    except BaseException:
        for destination in reversed(replaced):
            if destination in backups:
                os.replace(backups.pop(destination), destination)
            else:
                destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary in [*staged.values(), *backups.values()]:
            temporary.unlink(missing_ok=True)
