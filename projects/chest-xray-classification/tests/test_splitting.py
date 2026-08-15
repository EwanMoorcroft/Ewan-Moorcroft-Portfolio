from __future__ import annotations

import unittest

from chest_xray_benchmark.records import ImageRecord
from chest_xray_benchmark.splitting import assign_groups, audit_partition_map


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
        group_id=f"g-{label}-{index}",
    )


class SplittingTests(unittest.TestCase):
    def setUp(self) -> None:
        labels = ("Normal", "Lung Opacity", "Viral Pneumonia")
        self.records = [make_record(label, index) for label in labels for index in range(30)]
        self.records.append(make_record("Viral Pneumonia", 0, copy=1))
        self.fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}

    def test_group_split_is_deterministic_and_leakage_free(self) -> None:
        first = assign_groups(self.records, seed=534, fractions=self.fractions)
        second = assign_groups(self.records, seed=534, fractions=self.fractions)
        self.assertEqual(first, second)
        summary = audit_partition_map(self.records, first, 534, self.fractions)
        self.assertTrue(summary["leakage_checks_passed"])
        self.assertTrue(summary["class_coverage_passed"])
        self.assertTrue(summary["split_ready"])
        self.assertEqual(summary["image_count"], len(self.records))
        self.assertEqual(set(first.values()), {"train", "validation", "test"})
        self.assertEqual(first["g-Viral Pneumonia-0"], first["g-Viral Pneumonia-0"])

    def test_different_seed_changes_at_least_one_group(self) -> None:
        first = assign_groups(self.records, seed=534, fractions=self.fractions)
        second = assign_groups(self.records, seed=535, fractions=self.fractions)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
