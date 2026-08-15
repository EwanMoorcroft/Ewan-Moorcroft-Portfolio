from __future__ import annotations

import unittest
from pathlib import Path

from chest_xray_benchmark.spec import DatasetSpec


class SpecTests(unittest.TestCase):
    def test_public_spec_has_expected_identity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = DatasetSpec.load(root / "data" / "dataset-spec.json")
        self.assertEqual(spec.doi, "10.17632/p5rm59k7ph.1")
        self.assertEqual(spec.expected_total, 3475)
        self.assertEqual(
            spec.expected_counts,
            {"Normal": 1250, "Lung Opacity": 1125, "Viral Pneumonia": 1100},
        )
        self.assertEqual(spec.alias_to_class["Opacity"], "Lung Opacity")
        self.assertEqual(spec.alias_to_class["Pneumonia"], "Viral Pneumonia")


if __name__ == "__main__":
    unittest.main()
