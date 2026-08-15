from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

from chest_xray_benchmark.cli import main
from chest_xray_benchmark.manifest import write_manifest
from chest_xray_benchmark.records import ImageRecord
from chest_xray_benchmark.splitting import (
    SPLIT_FIELDS,
    assign_exact_groups,
    audit_partition_map,
    read_split_rows,
)


def make_record(label: str, index: int, copy: int = 0) -> ImageRecord:
    digest = f"{label}:{index}".encode().hex().ljust(64, "0")[:64]
    return ImageRecord(
        relative_path=f"{label}/{index}-{copy}.jpg",
        label=label,
        byte_size=10,
        sha256=digest,
        difference_hash=f"{index:016x}",
        width=299,
        height=299,
        color_mode="RGB",
        exact_group_id=f"sha256-{digest}",
    )


class SplittingTests(unittest.TestCase):
    def setUp(self) -> None:
        labels = ("Normal", "Lung Opacity", "Viral Pneumonia")
        self.records = [make_record(label, index) for label in labels for index in range(30)]
        self.records.append(make_record("Viral Pneumonia", 0, copy=1))
        self.fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}

    def test_group_split_is_deterministic_and_leakage_free(self) -> None:
        first = assign_exact_groups(self.records, seed=534, fractions=self.fractions)
        second = assign_exact_groups(self.records, seed=534, fractions=self.fractions)
        self.assertEqual(first, second)
        summary = audit_partition_map(self.records, first, 534, self.fractions)
        self.assertTrue(summary["leakage_checks_passed"])
        self.assertTrue(summary["class_coverage_passed"])
        self.assertTrue(summary["split_ready"])
        self.assertEqual(summary["image_count"], len(self.records))
        self.assertEqual(set(first.values()), {"train", "validation", "test"})
        duplicate_group = self.records[-1].exact_group_id
        self.assertEqual(
            sum(record.exact_group_id == duplicate_group for record in self.records), 2
        )
        self.assertIn(duplicate_group, first)

    def test_different_seed_changes_at_least_one_group(self) -> None:
        first = assign_exact_groups(self.records, seed=534, fractions=self.fractions)
        second = assign_exact_groups(self.records, seed=535, fractions=self.fractions)
        self.assertNotEqual(first, second)

    def test_cross_label_exact_copies_must_be_reviewed(self) -> None:
        conflicting = list(self.records)
        conflicting[1] = ImageRecord(
            relative_path=conflicting[1].relative_path,
            label="Lung Opacity",
            byte_size=conflicting[1].byte_size,
            sha256=conflicting[0].sha256,
            difference_hash=conflicting[1].difference_hash,
            width=conflicting[1].width,
            height=conflicting[1].height,
            color_mode=conflicting[1].color_mode,
            exact_group_id=conflicting[0].exact_group_id,
        )
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            assign_exact_groups(conflicting, seed=534, fractions=self.fractions)

    def test_same_digest_with_different_group_ids_is_rejected(self) -> None:
        corrupted = list(self.records)
        corrupted[-1] = replace(corrupted[-1], exact_group_id=f"sha256-{'f' * 64}")
        with self.assertRaisesRegex(ValueError, "Exact identity invariant failed"):
            assign_exact_groups(corrupted, seed=534, fractions=self.fractions)

    def test_different_digests_with_same_group_id_are_rejected(self) -> None:
        corrupted = list(self.records)
        corrupted[1] = replace(corrupted[1], exact_group_id=corrupted[0].exact_group_id)
        with self.assertRaisesRegex(ValueError, "Exact identity invariant failed"):
            assign_exact_groups(corrupted, seed=534, fractions=self.fractions)

    def test_tampered_cross_partition_split_is_rejected(self) -> None:
        partition_map = assign_exact_groups(self.records, seed=534, fractions=self.fractions)
        rows = []
        for record in self.records:
            rows.append(
                {
                    "relative_path": record.relative_path,
                    "label": record.label,
                    "byte_size": record.byte_size,
                    "sha256": record.sha256,
                    "difference_hash": record.difference_hash,
                    "width": record.width,
                    "height": record.height,
                    "color_mode": record.color_mode,
                    "has_exif": 0,
                    "exact_group_id": record.exact_group_id,
                    "split": partition_map[record.exact_group_id],
                }
            )
        duplicate_group = self.records[-1].exact_group_id
        duplicate_rows = [row for row in rows if row["exact_group_id"] == duplicate_group]
        self.assertEqual(len(duplicate_rows), 2)
        duplicate_rows[1]["split"] = "test" if duplicate_rows[0]["split"] != "test" else "train"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=SPLIT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "cross-partition exact copies"):
                read_split_rows(path)

    def test_cli_does_not_write_split_when_class_coverage_fails(self) -> None:
        records = [
            make_record(label, index)
            for label in ("Normal", "Lung Opacity", "Viral Pneumonia")
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.csv"
            split_path = root / "splits.csv"
            summary_path = root / "summary.json"
            write_manifest(records, manifest_path)
            with redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "split",
                        "--manifest",
                        str(manifest_path),
                        "--output",
                        str(split_path),
                        "--summary-out",
                        str(summary_path),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertTrue(summary_path.is_file())
            self.assertFalse(split_path.exists())


if __name__ == "__main__":
    unittest.main()
