from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chest_xray_benchmark.manifest import (
    group_exact_duplicates,
    scan_dataset,
    verification_report,
)
from chest_xray_benchmark.spec import ClassSpec, DatasetSpec


class VerifierTests(unittest.TestCase):
    def test_tiny_image_tree_matches_its_spec(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "raw"
            for label, value in (("A", 32), ("B", 128), ("C", 224)):
                class_root = data_root / label
                class_root.mkdir(parents=True)
                for index in range(2):
                    Image.new("L", (16, 16), color=value + index).save(class_root / f"{index}.png")
            payload = {
                "name": "fixture",
                "version": 1,
                "doi": "fixture-doi",
                "expected_total": 6,
                "classes": {
                    label: {"expected_count": 2, "directory_aliases": [label]}
                    for label in ("A", "B", "C")
                },
            }
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(payload), encoding="utf-8")
            spec = DatasetSpec.load(spec_path)
            records, errors, unexpected = scan_dataset(data_root, spec)
            grouped = group_exact_duplicates(records)
            report = verification_report(
                grouped,
                spec,
                errors,
                unexpected,
                visual_review_hamming=0,
            )
            self.assertTrue(report["identity_matches_spec"])
            self.assertTrue(report["exact_identity_split_ready"])
            self.assertEqual(report["automatic_grouping_policy"], "sha256_exact_identity_only")
            self.assertEqual(report["visual_hash_policy"], "direct_pair_review_candidates_only")
            self.assertEqual(report["observed_total"], 6)
            self.assertEqual(report["unreadable_images"], [])

    def test_verifier_rejects_symlinked_directory_outside_data_root(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "raw"
            class_root = data_root / "A"
            outside = root / "outside"
            class_root.mkdir(parents=True)
            outside.mkdir()
            Image.new("L", (16, 16), color=64).save(outside / "external.png")
            (class_root / "linked").symlink_to(outside, target_is_directory=True)
            spec = DatasetSpec(
                name="fixture",
                version=1,
                doi="fixture",
                expected_total=1,
                classes=(ClassSpec("A", 1, ("A",)),),
            )
            with self.assertRaisesRegex(ValueError, "symlink resolves outside"):
                scan_dataset(data_root, spec)

    def test_verifier_records_expected_io_failure_but_propagates_programming_error(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "raw"
            class_root = data_root / "A"
            class_root.mkdir(parents=True)
            Image.new("L", (16, 16), color=64).save(class_root / "image.png")
            spec = DatasetSpec(
                name="fixture",
                version=1,
                doi="fixture",
                expected_total=1,
                classes=(ClassSpec("A", 1, ("A",)),),
            )
            with patch(
                "chest_xray_benchmark.manifest.difference_hash",
                side_effect=OSError("unreadable"),
            ):
                records, errors, _ = scan_dataset(data_root, spec)
            self.assertEqual(records, [])
            self.assertEqual(errors[0]["error_type"], "OSError")

            with (
                patch(
                    "chest_xray_benchmark.manifest.difference_hash",
                    side_effect=RuntimeError("implementation defect"),
                ),
                self.assertRaisesRegex(RuntimeError, "implementation defect"),
            ):
                scan_dataset(data_root, spec)


if __name__ == "__main__":
    unittest.main()
