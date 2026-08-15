"""Image inventory, integrity checks and duplicate grouping."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .records import MANIFEST_FIELDS, ImageRecord
from .spec import DatasetSpec

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


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
        class_root = data_root / alias
        if not class_root.is_dir():
            continue
        paths = sorted(
            path
            for path in class_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for path in paths:
            relative_path = path.relative_to(data_root).as_posix()
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
            except Exception as exc:  # keep the inventory usable for diagnosis
                errors.append(
                    {
                        "relative_path": relative_path,
                        "error_type": type(exc).__name__,
                        "message": "image could not be inspected",
                    }
                )
    return records, errors, unexpected_directories


def group_duplicates(
    records: Iterable[ImageRecord],
    near_hamming: int = 2,
) -> list[ImageRecord]:
    if not 0 <= near_hamming <= 64:
        raise ValueError("near_hamming must be between 0 and 64")
    items = list(records)
    sets = DisjointSet(len(items))
    by_digest: dict[str, int] = {}
    for index, record in enumerate(items):
        previous = by_digest.setdefault(record.sha256, index)
        sets.union(index, previous)

    for left in range(len(items)):
        left_record = items[left]
        for right in range(left + 1, len(items)):
            right_record = items[right]
            if left_record.sha256 == right_record.sha256:
                continue
            if (left_record.width, left_record.height) != (right_record.width, right_record.height):
                continue
            if (
                hamming_distance(left_record.difference_hash, right_record.difference_hash)
                <= near_hamming
            ):
                sets.union(left, right)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(items)):
        members[sets.find(index)].append(index)

    group_ids: dict[int, str] = {}
    for root, indices in members.items():
        signature = "\n".join(sorted({items[index].sha256 for index in indices}))
        digest = hashlib.sha256(signature.encode("ascii")).hexdigest()[:20]
        group_ids[root] = f"g-{digest}"

    return [record.with_group(group_ids[sets.find(index)]) for index, record in enumerate(items)]


def verification_report(
    records: Iterable[ImageRecord],
    spec: DatasetSpec,
    errors: list[dict[str, str]] | None = None,
    unexpected_directories: list[str] | None = None,
    near_hamming: int = 2,
) -> dict[str, Any]:
    items = list(records)
    errors = errors or []
    unexpected_directories = unexpected_directories or []
    counts = Counter(record.label for record in items)
    by_digest: dict[str, list[ImageRecord]] = defaultdict(list)
    by_group: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in items:
        by_digest[record.sha256].append(record)
        by_group[record.group_id].append(record)

    exact_groups = [
        {
            "sha256": digest,
            "labels": sorted({record.label for record in members}),
            "paths": sorted(record.relative_path for record in members),
        }
        for digest, members in sorted(by_digest.items())
        if len(members) > 1
    ]
    near_groups = [
        {
            "group_id": group_id,
            "labels": sorted({record.label for record in members}),
            "paths": sorted(record.relative_path for record in members),
            "distinct_sha256": len({record.sha256 for record in members}),
        }
        for group_id, members in sorted(by_group.items())
        if len({record.sha256 for record in members}) > 1
    ]
    cross_label_exact_groups = [group for group in exact_groups if len(group["labels"]) > 1]
    cross_label_groups = [group for group in near_groups if len(group["labels"]) > 1]
    expected_counts = spec.expected_counts
    observed_counts = {name: counts.get(name, 0) for name in spec.class_names}
    identity_matches = (
        len(items) == spec.expected_total
        and observed_counts == expected_counts
        and not errors
        and not unexpected_directories
    )
    dimensions = Counter(f"{record.width}x{record.height}" for record in items)
    return {
        "dataset": {"name": spec.name, "version": spec.version, "doi": spec.doi},
        "identity_matches_spec": identity_matches,
        "expected_total": spec.expected_total,
        "observed_total": len(items),
        "expected_class_counts": expected_counts,
        "observed_class_counts": observed_counts,
        "near_hamming_threshold": near_hamming,
        "exact_duplicate_group_count": len(exact_groups),
        "visual_near_duplicate_group_count": len(near_groups),
        "cross_label_exact_group_count": len(cross_label_exact_groups),
        "cross_label_visual_group_count": len(cross_label_groups),
        "images_with_embedded_metadata": sum(record.has_exif for record in items),
        "dimensions": dict(sorted(dimensions.items())),
        "unexpected_directories": unexpected_directories,
        "unreadable_images": errors,
        "exact_duplicate_groups": exact_groups,
        "visual_near_duplicate_groups": near_groups,
        "cross_label_exact_groups": cross_label_exact_groups,
        "cross_label_visual_groups": cross_label_groups,
    }


def write_manifest(records: Iterable[ImageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.relative_path):
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
                    "group_id": record.group_id,
                }
            )


def read_manifest(path: Path) -> list[ImageRecord]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [
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
            group_id=row["group_id"],
        )
        for row in rows
    ]
