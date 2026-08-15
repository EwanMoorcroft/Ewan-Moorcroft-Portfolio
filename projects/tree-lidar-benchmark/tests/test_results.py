from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_support import copy_retained_evidence, retained_entries
from tree_lidar_benchmark.verification import load_overall_rows, verify_project

ROOT = Path(__file__).resolve().parents[1]


class ResultTests(unittest.TestCase):
    def test_full_retained_evidence_verifies(self) -> None:
        report = verify_project(ROOT)
        self.assertEqual(report.status, "verified")
        self.assertEqual(report.protocol, "for_instance_pointwise_v2")
        self.assertEqual(report.per_plot_rows, 132)
        self.assertEqual(report.by_site_rows, 60)
        self.assertEqual(report.overall_rows, 12)
        self.assertEqual(report.result_identities, 12)
        self.assertEqual(report.aggregate_values_checked, 1152)
        self.assertEqual(report.retained_hashes_checked, 7)

    def test_each_method_has_two_distinct_routes(self) -> None:
        rows = load_overall_rows(ROOT)
        routes: dict[str, set[str]] = {}
        for row in rows:
            routes.setdefault(row["method"], set()).add(row["variant"])
        self.assertEqual(len(routes), 6)
        self.assertTrue(
            all(values == {"published_default", "development_tuned"} for values in routes.values())
        )

    def test_headline_row_matches_manifest(self) -> None:
        payload = json.loads((ROOT / "results_manifest.json").read_text(encoding="utf-8"))
        headline = payload["headline_result"]
        rows = load_overall_rows(ROOT)
        selected = next(
            row
            for row in rows
            if row["method"] == "forestformer3d" and row["variant"] == "development_tuned"
        )
        self.assertEqual(int(selected["true_positives"]), headline["true_positives"])
        self.assertEqual(int(selected["false_positives"]), headline["false_positives"])
        self.assertEqual(int(selected["false_negatives"]), headline["false_negatives"])
        self.assertEqual(float(selected["micro_f1"]), headline["micro_f1"])

    def test_wheel_inventory_is_manifest_complete(self) -> None:
        payload = json.loads((ROOT / "results_manifest.json").read_text(encoding="utf-8"))
        declared = {(item["path"], item["sha256"]) for item in payload["retained_files"]}
        build_entries = {(path.as_posix(), digest) for path, digest in retained_entries(ROOT)}
        self.assertEqual(build_entries, declared)
        self.assertEqual(len(build_entries), 7)

    def test_built_evidence_is_complete_and_byte_identical(self) -> None:
        retained = retained_entries(ROOT)
        expected = {Path("results_manifest.json"), *(path for path, _digest in retained)}
        with tempfile.TemporaryDirectory() as directory:
            package_root = Path(directory)
            copied = set(copy_retained_evidence(ROOT, package_root))
            self.assertEqual(copied, expected)
            for relative in expected:
                self.assertEqual(
                    (package_root / "_evidence" / relative).read_bytes(),
                    (ROOT / relative).read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
