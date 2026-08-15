from __future__ import annotations

import unittest

from tree_lidar_benchmark.evaluator import (
    PROTOCOL_ID,
    evaluate_aligned_labels,
    maximum_cardinality_threshold_matching,
    precision_recall_f1,
)


class EvaluatorTests(unittest.TestCase):
    def test_maximum_cardinality_recovers_two_pairs(self) -> None:
        pairs = maximum_cardinality_threshold_matching(
            [[0.90, 0.80], [0.85, 0.00]],
            0.50,
        )
        self.assertEqual(pairs, [(0, 1), (1, 0)])

    def test_exact_boundary_half_remains_eligible(self) -> None:
        result = evaluate_aligned_labels(
            [10, 10],
            [1, 0],
            [4, 3],
            source_row_index=[0, 1],
        )
        self.assertEqual(result.protocol, PROTOCOL_ID)
        self.assertEqual(result.excluded_prediction_instances, 0)
        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.false_positives, 0)
        self.assertEqual(result.false_negatives, 0)
        self.assertEqual(result.matches[0].iou, 0.5)

    def test_strict_boundary_majority_excludes_complete_instance(self) -> None:
        result = evaluate_aligned_labels(
            [10, 10, 10],
            [1, 0, 0],
            [4, 3, 3],
            source_row_index=[0, 1, 2],
        )
        self.assertEqual(result.excluded_prediction_instances, 1)
        self.assertEqual(result.predicted_instances, 0)
        self.assertEqual(result.reference_instances, 1)
        self.assertEqual(result.true_positives, 0)
        self.assertEqual(result.false_positives, 0)
        self.assertEqual(result.false_negatives, 1)

    def test_prediction_background_points_reduce_iou(self) -> None:
        result = evaluate_aligned_labels(
            [10, 10, 10],
            [1, 0, 0],
            [4, 2, 2],
            source_row_index=[0, 1, 2],
        )
        self.assertEqual(result.evaluated_point_count, 3)
        self.assertEqual(result.true_positives, 0)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_negatives, 1)

    def test_source_row_index_is_exact(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_row_index"):
            evaluate_aligned_labels(
                [10, 10],
                [1, 1],
                [4, 4],
                source_row_index=[1, 0],
            )

    def test_non_positive_reference_ids_are_never_instances(self) -> None:
        result = evaluate_aligned_labels(
            [10, 11, 12],
            [0, -1, -2],
            [4, 5, 6],
            source_row_index=[0, 1, 2],
        )
        self.assertEqual(result.reference_instances, 0)
        self.assertEqual(result.false_negatives, 0)
        self.assertEqual(result.predicted_instances, 3)
        self.assertEqual(result.false_positives, 3)

    def test_metric_changing_options_cannot_claim_v2(self) -> None:
        overrides = (
            {"reference_classes": (4, 5)},
            {"ignored_reference_ids": (0, -1, 999)},
            {"ignored_prediction_ids": (0, -1, 999)},
            {"boundary_background_class": 2},
            {"boundary_majority_threshold": 0.75},
            {"iou_threshold": 0.75},
        )
        for override in overrides:
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, PROTOCOL_ID):
                evaluate_aligned_labels(
                    [10],
                    [1],
                    [4],
                    source_row_index=[0],
                    **override,
                )

    def test_count_metrics(self) -> None:
        self.assertEqual(precision_recall_f1(3, 1, 2), (0.75, 0.6, 2 * 3 / 9))
        self.assertEqual(precision_recall_f1(0, 0, 0), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
