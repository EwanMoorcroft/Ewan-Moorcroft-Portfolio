"""Deterministic group-level partitioning and leakage checks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from numbers import Real
from pathlib import Path
from typing import Any

from .records import (
    MANIFEST_FIELDS,
    ImageRecord,
    canonical_exact_group_id,
    validate_exact_identity,
)

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_FIELDS = (*MANIFEST_FIELDS, "split")


def _validate_fractions(fractions: dict[str, float]) -> None:
    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(f"Fractions must define {SPLIT_NAMES}")
    values = list(fractions.values())
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or value <= 0
        for value in values
    ):
        raise ValueError("Fractions must be finite and positive")
    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError("Fractions must sum to one")


def _validate_partition_map(
    records: Iterable[ImageRecord],
    partition_map: dict[str, str],
) -> list[ImageRecord]:
    items = validate_exact_identity(records)
    if not items:
        raise ValueError("At least one image record is required")
    expected_groups = {record.exact_group_id for record in items}
    provided_groups = set(partition_map)
    if provided_groups != expected_groups:
        missing = sorted(expected_groups - provided_groups)
        extra = sorted(provided_groups - expected_groups)
        raise ValueError(f"Partition map group mismatch; missing={missing}, extra={extra}")
    invalid_splits = sorted(set(partition_map.values()) - set(SPLIT_NAMES))
    if invalid_splits:
        raise ValueError("Partition map contains unknown splits: " + ", ".join(invalid_splits))
    return items


def _stable_rank(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def assign_exact_groups(
    records: Iterable[ImageRecord],
    seed: int = 534,
    fractions: dict[str, float] | None = None,
) -> dict[str, str]:
    items = validate_exact_identity(records)
    fractions = (
        {"train": 0.70, "validation": 0.15, "test": 0.15} if fractions is None else fractions
    )
    _validate_fractions(fractions)

    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in items:
        grouped[record.exact_group_id].append(record)
    if len(grouped) < len(SPLIT_NAMES):
        raise ValueError("At least three exact-identity groups are required")
    cross_label_groups = [
        group_id
        for group_id, members in grouped.items()
        if len({member.label for member in members}) > 1
    ]
    if cross_label_groups:
        raise ValueError("Exact copies with conflicting labels require review before splitting")

    class_totals = Counter(record.label for record in items)
    total = len(items)
    targets = {
        split: {label: class_totals[label] * fractions[split] for label in sorted(class_totals)}
        for split in SPLIT_NAMES
    }
    target_totals = {split: total * fractions[split] for split in SPLIT_NAMES}
    current = {split: Counter() for split in SPLIT_NAMES}
    current_totals = Counter()

    group_items = []
    for group_id, members in grouped.items():
        counts = Counter(record.label for record in members)
        rarity = max((1.0 / class_totals[label] for label in counts), default=0.0)
        group_items.append((group_id, members, counts, rarity))
    group_items.sort(key=lambda item: (-len(item[1]), -item[3], _stable_rank(seed, item[0])))

    partition_map: dict[str, str] = {}
    for group_id, members, group_counts, _ in group_items:
        candidate_scores: list[tuple[float, int, str]] = []
        for split in SPLIT_NAMES:
            score = 0.0
            for scored_split in SPLIT_NAMES:
                for label in sorted(class_totals):
                    count = current[scored_split][label]
                    if scored_split == split:
                        count += group_counts[label]
                    target = targets[scored_split][label]
                    score += ((count - target) ** 2) / max(target, 1.0)
                    if count > target:
                        score += 2.0 * ((count - target) ** 2) / max(target, 1.0)
                count_total = current_totals[scored_split]
                if scored_split == split:
                    count_total += len(members)
                target_total = target_totals[scored_split]
                score += 0.25 * ((count_total - target_total) ** 2) / max(target_total, 1.0)
            candidate_scores.append((score, _stable_rank(seed, f"{group_id}:{split}"), split))
        chosen = min(candidate_scores)[2]
        partition_map[group_id] = chosen
        current[chosen].update(group_counts)
        current_totals[chosen] += len(members)

    if set(partition_map.values()) != set(SPLIT_NAMES):
        raise RuntimeError("Group allocation produced an empty partition")
    return partition_map


def split_digest(records: Iterable[ImageRecord], partition_map: dict[str, str]) -> str:
    items = _validate_partition_map(records, partition_map)
    lines = [
        "\t".join(
            (
                record.relative_path,
                record.label,
                record.sha256,
                record.exact_group_id,
                partition_map[record.exact_group_id],
            )
        )
        for record in sorted(items, key=lambda item: item.relative_path)
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def audit_partition_map(
    records: Iterable[ImageRecord],
    partition_map: dict[str, str],
    seed: int,
    fractions: dict[str, float],
) -> dict[str, Any]:
    _validate_fractions(fractions)
    items = _validate_partition_map(records, partition_map)
    digest_splits: dict[str, set[str]] = defaultdict(set)
    exact_group_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[str, Counter[str]] = {name: Counter() for name in SPLIT_NAMES}
    split_group_counts = Counter()
    for record in items:
        split = partition_map[record.exact_group_id]
        digest_splits[record.sha256].add(split)
        exact_group_splits[record.exact_group_id].add(split)
        split_counts[split][record.label] += 1
    for _group_id, split in partition_map.items():
        split_group_counts[split] += 1

    digest_violations = sorted(
        digest for digest, splits in digest_splits.items() if len(splits) > 1
    )
    exact_group_violations = sorted(
        group for group, splits in exact_group_splits.items() if len(splits) > 1
    )
    all_labels = sorted({record.label for record in items})
    missing_classes = {
        split: sorted(set(all_labels) - set(split_counts[split])) for split in SPLIT_NAMES
    }
    leakage_passed = not digest_violations and not exact_group_violations
    class_coverage_passed = not any(missing_classes.values())
    return {
        "seed": seed,
        "fractions": fractions,
        "manifest_digest": split_digest(items, partition_map),
        "image_count": len(items),
        "exact_identity_group_count": len(partition_map),
        "partitions": {
            split: {
                "images": sum(split_counts[split].values()),
                "exact_identity_groups": split_group_counts[split],
                "class_counts": dict(sorted(split_counts[split].items())),
            }
            for split in SPLIT_NAMES
        },
        "sha256_cross_partition_violations": digest_violations,
        "exact_identity_group_cross_partition_violations": exact_group_violations,
        "missing_classes": missing_classes,
        "leakage_checks_passed": leakage_passed,
        "class_coverage_passed": class_coverage_passed,
        "split_ready": leakage_passed and class_coverage_passed,
    }


def write_splits(records: Iterable[ImageRecord], partition_map: dict[str, str], path: Path) -> None:
    items = _validate_partition_map(records, partition_map)
    split_labels: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    for record in items:
        split_labels[partition_map[record.exact_group_id]].add(record.label)
    all_labels = {record.label for record in items}
    missing_classes = {split: sorted(all_labels - split_labels[split]) for split in SPLIT_NAMES}
    if any(missing_classes.values()):
        raise ValueError(f"Partition map is missing classes: {missing_classes}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SPLIT_FIELDS)
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
                    "split": partition_map[record.exact_group_id],
                }
            )


def audit_split_rows(rows: Iterable[dict[str, str]]) -> dict[str, Any]:
    """Validate a split CSV before any training or evaluation consumer uses it."""

    items = list(rows)
    if not items:
        raise ValueError("Split file cannot be empty")
    relative_paths: set[str] = set()
    digest_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    split_labels: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}

    for index, row in enumerate(items, start=2):
        relative_path = row.get("relative_path", "")
        label = row.get("label", "")
        digest = row.get("sha256", "")
        group_id = row.get("exact_group_id", "")
        split = row.get("split", "")
        if not relative_path or not label:
            raise ValueError(f"Split row {index} requires a relative path and label")
        if relative_path in relative_paths:
            raise ValueError(f"Split file repeats relative path {relative_path!r}")
        relative_paths.add(relative_path)
        expected_group = canonical_exact_group_id(digest)
        if group_id != expected_group:
            raise ValueError(
                f"Split row {index} violates exact identity: expected {expected_group!r}"
            )
        if split not in SPLIT_NAMES:
            raise ValueError(f"Split row {index} contains unknown split {split!r}")
        digest_splits[digest].add(split)
        group_splits[group_id].add(split)
        split_labels[split].add(label)

    digest_violations = sorted(
        digest for digest, partitions in digest_splits.items() if len(partitions) > 1
    )
    group_violations = sorted(
        group for group, partitions in group_splits.items() if len(partitions) > 1
    )
    if digest_violations or group_violations:
        raise ValueError(
            "Split file contains cross-partition exact copies: "
            f"sha256={digest_violations}, exact_group_id={group_violations}"
        )
    observed_splits = {row["split"] for row in items}
    if observed_splits != set(SPLIT_NAMES):
        raise ValueError(f"Split file must contain all partitions: {SPLIT_NAMES}")
    all_labels = {row["label"] for row in items}
    missing_classes = {split: sorted(all_labels - split_labels[split]) for split in SPLIT_NAMES}
    if any(missing_classes.values()):
        raise ValueError(f"Split file is missing classes: {missing_classes}")
    return {
        "rows": len(items),
        "exact_identity_groups": len(group_splits),
        "leakage_checks_passed": True,
        "class_coverage_passed": True,
    }


def read_split_rows(path: Path, split: str | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Split file contains duplicate column names")
        missing = sorted(set(SPLIT_FIELDS) - set(fields))
        if missing:
            raise ValueError("Split file is missing required columns: " + ", ".join(missing))
        rows = list(reader)
    audit_split_rows(rows)
    if split is not None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split: {split}")
        rows = [row for row in rows if row["split"] == split]
    return rows


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
