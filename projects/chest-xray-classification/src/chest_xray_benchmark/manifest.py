"""Image inventory, integrity checks and duplicate grouping."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .records import (
    MANIFEST_FIELDS,
    ImageRecord,
    canonical_exact_group_id,
    validate_exact_identity,
)
from .spec import DatasetSpec

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


class ImageInspectionError(ValueError):
    """Raised when Pillow cannot safely decode or inspect an image file."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path) -> tuple[str, int, int, str, bool]:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Pillow is required to inspect images") from exc

    try:
        with Image.open(path) as source:
            source.verify()
        with Image.open(path) as source:
            width, height = source.size
            color_mode = source.mode
            has_exif = bool(source.getexif())
            image = ImageOps.exif_transpose(source).convert("L")
            image = image.resize((9, 8), Image.Resampling.LANCZOS)
            if hasattr(image, "get_flattened_data"):
                values = list(image.get_flattened_data())
            else:  # Pillow before 12
                values = list(image.getdata())
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageInspectionError(f"Pillow could not inspect {path.name}") from exc
    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            bits = (bits << 1) | int(values[offset + column] > values[offset + column + 1])
    return f"{bits:016x}", width, height, color_mode, has_exif


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def scan_dataset(
    data_root: Path,
    spec: DatasetSpec,
) -> tuple[list[ImageRecord], list[dict[str, str]], list[str]]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    root = data_root.resolve(strict=True)
    alias_map = spec.alias_to_class
    visible_directories = sorted(
        entry.name
        for entry in data_root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
    unexpected_directories = [name for name in visible_directories if name not in alias_map]
    records: list[ImageRecord] = []
    errors: list[dict[str, str]] = []

    for alias, label in sorted(alias_map.items()):
        class_candidate = root / alias
        if not class_candidate.is_dir():
            continue
        class_root = class_candidate.resolve(strict=True)
        if not class_root.is_relative_to(root):
            raise ValueError(f"Dataset alias resolves outside the data root: {alias!r}")
        paths: list[Path] = []
        for candidate in class_root.rglob("*"):
            if candidate.is_symlink():
                try:
                    resolved_candidate = candidate.resolve(strict=True)
                except OSError as exc:
                    raise ValueError(
                        f"Dataset contains an unresolved symlink: {candidate}"
                    ) from exc
                if not resolved_candidate.is_relative_to(root):
                    raise ValueError(f"Dataset symlink resolves outside the data root: {candidate}")
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                resolved_candidate = candidate.resolve(strict=True)
                if not resolved_candidate.is_relative_to(root):
                    raise ValueError(f"Dataset image resolves outside the data root: {candidate}")
                paths.append(candidate)
        paths.sort()
        for path in paths:
            relative_path = path.relative_to(root).as_posix()
            try:
                visual_hash, width, height, color_mode, has_exif = difference_hash(path)
                records.append(
                    ImageRecord(
                        relative_path=relative_path,
                        label=label,
                        byte_size=path.stat().st_size,
                        sha256=sha256_file(path),
                        difference_hash=visual_hash,
                        width=width,
                        height=height,
                        color_mode=color_mode,
                        has_exif=has_exif,
                    )
                )
            except (ImageInspectionError, OSError) as exc:
                errors.append(
                    {
                        "relative_path": relative_path,
                        "error_type": type(exc).__name__,
                        "message": "image could not be inspected",
                    }
                )
    return records, errors, unexpected_directories


def group_exact_duplicates(records: Iterable[ImageRecord]) -> list[ImageRecord]:
    """Assign split groups using cryptographic identity only.

    Difference hashes are deliberately excluded from automatic grouping. A
    perceptual hash is a useful review signal, but it is not an identity key:
    pairwise similarities can form long transitive chains whose endpoints are
    not visually similar.
    """

    return [record.with_exact_group(canonical_exact_group_id(record.sha256)) for record in records]


def visual_review_candidates(
    records: Iterable[ImageRecord],
    max_hamming: int = 2,
) -> list[dict[str, Any]]:
    """Return direct perceptual-hash pairs for manual review.

    Each pair is evaluated independently. Candidates never alter
    ``exact_group_id`` and therefore never become split constraints without a
    separately reviewed identity source.
    """

    if not 0 <= max_hamming <= 64:
        raise ValueError("max_hamming must be between 0 and 64")
    items = sorted(records, key=lambda item: item.relative_path)
    candidates: list[dict[str, Any]] = []
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            if left.sha256 == right.sha256:
                continue
            if (left.width, left.height) != (right.width, right.height):
                continue
            distance = hamming_distance(left.difference_hash, right.difference_hash)
            if distance <= max_hamming:
                candidates.append(
                    {
                        "left_path": left.relative_path,
                        "right_path": right.relative_path,
                        "left_label": left.label,
                        "right_label": right.label,
                        "hamming_distance": distance,
                    }
                )
    return candidates


def verification_report(
    records: Iterable[ImageRecord],
    spec: DatasetSpec,
    errors: list[dict[str, str]] | None = None,
    unexpected_directories: list[str] | None = None,
    visual_review_hamming: int = 2,
) -> dict[str, Any]:
    items = list(records)
    errors = errors or []
    unexpected_directories = unexpected_directories or []
    counts = Counter(record.label for record in items)
    by_digest: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in items:
        by_digest[record.sha256].append(record)

    exact_groups = [
        {
            "sha256": digest,
            "labels": sorted({record.label for record in members}),
            "paths": sorted(record.relative_path for record in members),
        }
        for digest, members in sorted(by_digest.items())
        if len(members) > 1
    ]
    cross_label_exact_groups = [group for group in exact_groups if len(group["labels"]) > 1]
    review_candidates = visual_review_candidates(items, visual_review_hamming)
    cross_label_review_candidates = [
        candidate
        for candidate in review_candidates
        if candidate["left_label"] != candidate["right_label"]
    ]
    expected_counts = spec.expected_counts
    observed_counts = {name: counts.get(name, 0) for name in spec.class_names}
    identity_matches = (
        len(items) == spec.expected_total
        and observed_counts == expected_counts
        and not errors
        and not unexpected_directories
    )
    exact_identity_split_ready = identity_matches and not cross_label_exact_groups
    dimensions = Counter(f"{record.width}x{record.height}" for record in items)
    return {
        "dataset": {"name": spec.name, "version": spec.version, "doi": spec.doi},
        "identity_matches_spec": identity_matches,
        "expected_total": spec.expected_total,
        "observed_total": len(items),
        "expected_class_counts": expected_counts,
        "observed_class_counts": observed_counts,
        "automatic_grouping_policy": "sha256_exact_identity_only",
        "visual_hash_policy": "direct_pair_review_candidates_only",
        "visual_review_hamming_threshold": visual_review_hamming,
        "exact_duplicate_group_count": len(exact_groups),
        "visual_review_candidate_pair_count": len(review_candidates),
        "cross_label_exact_group_count": len(cross_label_exact_groups),
        "cross_label_visual_review_candidate_pair_count": len(cross_label_review_candidates),
        "manual_visual_review_required": bool(review_candidates),
        "exact_identity_split_ready": exact_identity_split_ready,
        "images_with_embedded_metadata": sum(record.has_exif for record in items),
        "dimensions": dict(sorted(dimensions.items())),
        "unexpected_directories": unexpected_directories,
        "unreadable_images": errors,
        "exact_duplicate_groups": exact_groups,
        "visual_review_candidate_pairs": review_candidates,
        "cross_label_exact_groups": cross_label_exact_groups,
        "cross_label_visual_review_candidate_pairs": cross_label_review_candidates,
    }


def write_manifest(records: Iterable[ImageRecord], path: Path) -> None:
    items = validate_exact_identity(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in sorted(items, key=lambda item: item.relative_path):
            writer.writerow(
                {
                    "relative_path": record.relative_path,
                    "label": record.label,
                    "byte_size": record.byte_size,
                    "sha256": record.sha256,
                    "difference_hash": record.difference_hash,
                    "width": record.width,
                    "height": record.height,
                    "color_mode": record.color_mode,
                    "has_exif": int(record.has_exif),
                    "exact_group_id": record.exact_group_id,
                }
            )


def read_manifest(path: Path) -> list[ImageRecord]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Manifest contains duplicate column names")
        missing = sorted(set(MANIFEST_FIELDS) - set(fields))
        if missing:
            if "group_id" in fields and "exact_group_id" in missing:
                raise ValueError(
                    "Legacy manifest schema is not accepted; regenerate it with the verify command"
                )
            raise ValueError("Manifest is missing required columns: " + ", ".join(missing))
        rows = list(reader)
    records = [
        ImageRecord(
            relative_path=row["relative_path"],
            label=row["label"],
            byte_size=int(row["byte_size"]),
            sha256=row["sha256"],
            difference_hash=row["difference_hash"],
            width=int(row["width"]),
            height=int(row["height"]),
            color_mode=row["color_mode"],
            has_exif=row["has_exif"] in {"1", "true", "True"},
            exact_group_id=row["exact_group_id"],
        )
        for row in rows
    ]
    return validate_exact_identity(records)
