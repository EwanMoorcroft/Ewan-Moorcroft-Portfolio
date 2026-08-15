"""Shared immutable record types."""

from __future__ import annotations

from dataclasses import dataclass, replace


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
    group_id: str = ""

    def with_group(self, group_id: str) -> ImageRecord:
        return replace(self, group_id=group_id)


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
    "group_id",
)
