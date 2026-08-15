"""Tree LiDAR instance-segmentation evaluation and result verification."""

from tree_lidar_benchmark.evaluator import (
    EvaluationResult,
    Match,
    evaluate_aligned_labels,
    maximum_cardinality_threshold_matching,
    precision_recall_f1,
)
from tree_lidar_benchmark.verification import VerificationReport, verify_project

__all__ = [
    "EvaluationResult",
    "Match",
    "VerificationReport",
    "evaluate_aligned_labels",
    "maximum_cardinality_threshold_matching",
    "precision_recall_f1",
    "verify_project",
]

__version__ = "1.0.0"
