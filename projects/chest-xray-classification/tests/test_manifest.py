from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from chest_xray_benchmark.manifest import (
    group_exact_duplicates,
    read_manifest,
    visual_review_candidates,
    write_manifest,
)
from chest_xray_benchmark.records import MANIFEST_FIELDS, ImageRecord


def record(path: str, label: str, digest: str, visual_hash: str) -> ImageRecord:
    return ImageRecord(
        relative_path=path,
        label=label,
        byte_size=10,
        sha256=digest,
        difference_hash=visual_hash,
        width=299,
        height=299,
        color_mode="RGB",
    )


class ManifestTests(unittest.TestCase):
    def test_only_exact_copies_share_an_automatic_group(self) -> None:
        items = [
            record("Normal/1.jpg", "Normal", "a" * 64, "0000000000000000"),
            record("Normal/2.jpg", "Normal", "a" * 64, "0000000000000000"),
            record("Opacity/3.jpg", "Lung Opacity", "b" * 64, "0000000000000001"),
            record("Pneumonia/4.jpg", "Viral Pneumonia", "c" * 64, "ffffffffffffffff"),
        ]
        grouped = group_exact_duplicates(items)
        self.assertEqual(grouped[0].exact_group_id, grouped[1].exact_group_id)
        self.assertNotEqual(grouped[0].exact_group_id, grouped[2].exact_group_id)

    def test_visual_hash_pairs_are_review_candidates_without_transitive_grouping(self) -> None:
        items = [
            record("Normal/1.jpg", "Normal", "a" * 64, "0000000000000000"),
            record("Normal/2.jpg", "Normal", "b" * 64, "0000000000000001"),
            record("Normal/3.jpg", "Normal", "c" * 64, "0000000000000003"),
        ]
        grouped = group_exact_duplicates(items)
        self.assertEqual(len({item.exact_group_id for item in grouped}), 3)
        candidates = visual_review_candidates(grouped, max_hamming=1)
        self.assertEqual(
            [(item["left_path"], item["right_path"]) for item in candidates],
            [("Normal/1.jpg", "Normal/2.jpg"), ("Normal/2.jpg", "Normal/3.jpg")],
        )

    def test_manifest_round_trip(self) -> None:
        items = group_exact_duplicates(
            [record("Normal/1.jpg", "Normal", "d" * 64, "1234567890abcdef")]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            write_manifest(items, path)
            self.assertEqual(read_manifest(path), items)

    def test_noncanonical_digest_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "64 lowercase hexadecimal"):
            group_exact_duplicates([record("Normal/1.jpg", "Normal", "A" * 64, "1234567890abcdef")])

    def test_noncanonical_group_is_rejected_when_manifest_is_read(self) -> None:
        item = record("Normal/1.jpg", "Normal", "d" * 64, "1234567890abcdef")
        row = {
            "relative_path": item.relative_path,
            "label": item.label,
            "byte_size": item.byte_size,
            "sha256": item.sha256,
            "difference_hash": item.difference_hash,
            "width": item.width,
            "height": item.height,
            "color_mode": item.color_mode,
            "has_exif": 0,
            "exact_group_id": f"sha256-{'e' * 64}",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "Exact identity invariant failed"):
                read_manifest(path)


if __name__ == "__main__":
    unittest.main()
