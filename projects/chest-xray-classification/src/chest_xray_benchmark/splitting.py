"""Deterministic group-level partitioning and leakage checks."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .records import MANIFEST_FIELDS, ImageRecord

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_FIELDS = (*MANIFEST_FIELDS, "split")


def _stable_rank(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def assign_groups(
    records: Iterable[ImageRecord],
    seed: int = 534,
    fractions: dict[str, float] | None = None,
) -> dict[str, str]:
    items = list(records)
    fractions = fractions or {"train": 0.70, "validation": 0.15, "test": 0.15}
    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(f"Fractions must define {SPLIT_NAMES}")
    if abs(sum(fractions.values()) - 1.0) > 1e-9 or any(value <= 0 for value in fractions.values()):
        raise ValueError("Fractions must be positive and sum to one")
    if any(not record.group_id for record in items):
        raise ValueError("Every image must have a duplicate group before splitting")

    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in items:
        grouped[record.group_id].append(record)
    if len(grouped) < len(SPLIT_NAMES):
        raise ValueError("At least three duplicate groups are required")

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
    lines = [
        "\t".join(
            (
                record.relative_path,
                record.label,
                record.sha256,
                record.group_id,
                partition_map[record.group_id],
            )
        )
        for record in sorted(records, key=lambda item: item.relative_path)
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def audit_partition_map(
    records: Iterable[ImageRecord],
    partition_map: dict[str, str],
    seed: int,
    fractions: dict[str, float],
) -> dict[str, Any]:
    items = list(records)
    digest_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[str, Counter[str]] = {name: Counter() for name in SPLIT_NAMES}
    split_group_counts = Counter()
    for record in items:
        split = partition_map[record.group_id]
        digest_splits[record.sha256].add(split)
        group_splits[record.group_id].add(split)
        split_counts[split][record.label] += 1
    for _group_id, split in partition_map.items():
        split_group_counts[split] += 1

    digest_violations = sorted(
        digest for digest, splits in digest_splits.items() if len(splits) > 1
    )
    group_violations = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    all_labels = sorted({record.label for record in items})
    missing_classes = {
        split: sorted(set(all_labels) - set(split_counts[split])) for split in SPLIT_NAMES
    }
    leakage_passed = not digest_violations and not group_violations
    class_coverage_passed = not any(missing_classes.values())
    return {
        "seed": seed,
        "fractions": fractions,
        "manifest_digest": split_digest(items, partition_map),
        "image_count": len(items),
        "duplicate_group_count": len(partition_map),
        "partitions": {
            split: {
                "images": sum(split_counts[split].values()),
                "groups": split_group_counts[split],
                "class_counts": dict(sorted(split_counts[split].items())),
            }
            for split in SPLIT_NAMES
        },
        "sha256_cross_partition_violations": digest_violations,
        "duplicate_group_cross_partition_violations": group_violations,
        "missing_classes": missing_classes,
        "leakage_checks_passed": leakage_passed,
        "class_coverage_passed": class_coverage_passed,
        "split_ready": leakage_passed and class_coverage_passed,
    }


def write_splits(records: Iterable[ImageRecord], partition_map: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SPLIT_FIELDS)
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
                    "split": partition_map[record.group_id],
                }
            )


def read_split_rows(path: Path, split: str | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if split is not None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split: {split}")
        rows = [row for row in rows if row["split"] == split]
    return rows


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
