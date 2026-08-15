from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chest_xray_benchmark.manifest import group_duplicates, read_manifest, write_manifest
from chest_xray_benchmark.records import ImageRecord


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
    def test_exact_and_visual_duplicates_share_one_group(self) -> None:
        items = [
            record("Normal/1.jpg", "Normal", "a" * 64, "0000000000000000"),
            record("Normal/2.jpg", "Normal", "a" * 64, "0000000000000000"),
            record("Opacity/3.jpg", "Lung Opacity", "b" * 64, "0000000000000001"),
            record("Pneumonia/4.jpg", "Viral Pneumonia", "c" * 64, "ffffffffffffffff"),
        ]
        grouped = group_duplicates(items, near_hamming=2)
        self.assertEqual(grouped[0].group_id, grouped[1].group_id)
        self.assertEqual(grouped[0].group_id, grouped[2].group_id)
        self.assertNotEqual(grouped[0].group_id, grouped[3].group_id)

    def test_manifest_round_trip(self) -> None:
        items = group_duplicates(
            [record("Normal/1.jpg", "Normal", "d" * 64, "1234567890abcdef")],
            near_hamming=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            write_manifest(items, path)
            self.assertEqual(read_manifest(path), items)


if __name__ == "__main__":
    unittest.main()
