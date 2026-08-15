"""Shared immutable record types."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
EXACT_GROUP_PREFIX = "sha256-"


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """Derived metadata for one source image."""

    relative_path: str
    label: str
    byte_size: int
    sha256: str
    difference_hash: str
    width: int
    height: int
    color_mode: str
    has_exif: bool = False
    exact_group_id: str = ""

    def with_exact_group(self, exact_group_id: str) -> ImageRecord:
        return replace(self, exact_group_id=exact_group_id)


def canonical_exact_group_id(sha256: str) -> str:
    """Return the only valid automatic group identifier for one digest."""

    if SHA256_PATTERN.fullmatch(sha256) is None:
        raise ValueError("sha256 must be exactly 64 lowercase hexadecimal characters")
    return f"{EXACT_GROUP_PREFIX}{sha256}"


def validate_exact_identity(records: Iterable[ImageRecord]) -> list[ImageRecord]:
    """Fail closed unless every record uses the canonical digest-derived group."""

    items = list(records)
    for index, record in enumerate(items):
        expected_group = canonical_exact_group_id(record.sha256)
        if record.exact_group_id != expected_group:
            raise ValueError(
                "Exact identity invariant failed for record "
                f"{index} ({record.relative_path!r}): exact_group_id must be {expected_group!r}"
            )
    return items


MANIFEST_FIELDS = (
    "relative_path",
    "label",
    "byte_size",
    "sha256",
    "difference_hash",
    "width",
    "height",
    "color_mode",
    "has_exif",
    "exact_group_id",
)
