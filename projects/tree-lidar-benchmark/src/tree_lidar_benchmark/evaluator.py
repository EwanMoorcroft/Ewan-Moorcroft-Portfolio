"""Dependency-free implementation of the aligned point-wise scoring contract."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

PROTOCOL_ID = "for_instance_pointwise_v2"
MATCHING_POLICY = "maximum_cardinality_one_to_one"
EVALUATION_MASK = "union_of_reference_tree_and_eligible_prediction_points"


@dataclass(frozen=True)
class Match:
    """One accepted predicted/reference instance pair."""

    prediction_id: int
    reference_id: int
    iou: float


@dataclass(frozen=True)
class EvaluationResult:
    """Counts and metrics for one source-row-aligned point cloud."""

    protocol: str
    point_count: int
    evaluated_point_count: int
    predicted_instances: int
    reference_instances: int
    excluded_prediction_instances: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    median_matched_iou: float
    matches: tuple[Match, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""

        return asdict(self)


def _integer_labels(values: Sequence[object], name: str) -> list[int]:
    labels: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name}[{index}] must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name}[{index}] must be finite")
        rounded = round(numeric)
        if numeric != rounded:
            raise ValueError(f"{name}[{index}] must be integral")
        labels.append(int(rounded))
    return labels


def _validate_source_rows(values: Sequence[object], size: int) -> None:
    if len(values) != size:
        raise ValueError("Prediction, reference, classification and index lengths differ")
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"source_row_index[{index}] must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric != index:
            raise ValueError("source_row_index must equal 0..n-1")


def precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> tuple[float, float, float]:
    """Compute precision, recall and F1 from non-negative integer counts."""

    counts = (true_positives, false_positives, false_negatives)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
        raise TypeError("TP, FP and FN must be integers")
    if any(value < 0 for value in counts):
        raise ValueError("TP, FP and FN cannot be negative")
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    f1_denominator = 2 * true_positives + false_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * true_positives / f1_denominator if f1_denominator else 0.0
    return float(precision), float(recall), float(f1)


def maximum_cardinality_threshold_matching(
    matrix: Sequence[Sequence[float]],
    threshold: float,
) -> list[tuple[int, int]]:
    """Return a deterministic maximum-cardinality matching above a threshold."""

    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("IoU threshold must be in the interval (0, 1]")
    rows = [list(row) for row in matrix]
    width = len(rows[0]) if rows else 0
    if any(len(row) != width for row in rows):
        raise ValueError("IoU matrix must be rectangular")
    for row in rows:
        for score in row:
            if not isinstance(score, Real) or not math.isfinite(float(score)):
                raise ValueError("IoU matrix must contain finite numbers")
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError("IoU values must be in the interval [0, 1]")

    candidates: list[list[int]] = []
    for row in rows:
        valid = [column for column, score in enumerate(row) if score >= threshold]
        candidates.append(sorted(valid, key=lambda column: (-row[column], column)))

    reference_owner: dict[int, int] = {}

    def assign(prediction: int, seen: set[int]) -> bool:
        for reference in candidates[prediction]:
            if reference in seen:
                continue
            seen.add(reference)
            owner = reference_owner.get(reference)
            if owner is None or assign(owner, seen):
                reference_owner[reference] = prediction
                return True
        return False

    order = sorted(
        range(len(rows)),
        key=lambda row_index: (
            -(max(rows[row_index]) if width else 0.0),
            row_index,
        ),
    )
    for prediction in order:
        assign(prediction, set())
    return sorted((prediction, reference) for reference, prediction in reference_owner.items())


def evaluate_aligned_labels(
    predicted_tree_id: Sequence[object],
    reference_tree_id: Sequence[object],
    classification: Sequence[object],
    *,
    source_row_index: Sequence[object],
    reference_classes: tuple[int, ...] = (4, 5, 6),
    ignored_reference_ids: tuple[int, ...] = (0, -1),
    ignored_prediction_ids: tuple[int, ...] = (0, -1),
    boundary_background_class: int = 3,
    boundary_majority_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> EvaluationResult:
    """Evaluate aligned labels under ``for_instance_pointwise_v2``.

    ``source_row_index`` is required and must equal ``0..n-1``. A complete
    predicted instance is excluded before matching only when strictly more than
    half of its aligned points carry semantic class 3. Candidate pairs require
    IoU of at least 0.5 by default, followed by deterministic one-to-one
    maximum-cardinality matching.
    """

    predicted = _integer_labels(predicted_tree_id, "predicted_tree_id")
    reference = _integer_labels(reference_tree_id, "reference_tree_id")
    semantic = _integer_labels(classification, "classification")
    size = len(predicted)
    if not (size == len(reference) == len(semantic)):
        raise ValueError("Prediction, reference, classification and index lengths differ")
    _validate_source_rows(source_row_index, size)
    if not math.isfinite(boundary_majority_threshold) or not (
        0.0 <= boundary_majority_threshold <= 1.0
    ):
        raise ValueError("boundary_majority_threshold must be in [0, 1]")

    reference_class_set = set(reference_classes)
    ignored_reference_set = set(ignored_reference_ids)
    ignored_prediction_set = set(ignored_prediction_ids)

    instance_sizes: dict[int, int] = {}
    boundary_sizes: dict[int, int] = {}
    for index, prediction_id in enumerate(predicted):
        if prediction_id in ignored_prediction_set:
            continue
        instance_sizes[prediction_id] = instance_sizes.get(prediction_id, 0) + 1
        if semantic[index] == boundary_background_class:
            boundary_sizes[prediction_id] = boundary_sizes.get(prediction_id, 0) + 1
    excluded = {
        prediction_id
        for prediction_id, count in instance_sizes.items()
        if boundary_sizes.get(prediction_id, 0) / count > boundary_majority_threshold
    }

    prediction_sizes: dict[int, int] = {}
    reference_sizes: dict[int, int] = {}
    intersections: dict[tuple[int, int], int] = {}
    evaluated_point_count = 0
    for index in range(size):
        prediction_id = predicted[index]
        reference_id = reference[index]
        prediction_active = (
            prediction_id not in ignored_prediction_set and prediction_id not in excluded
        )
        reference_active = (
            semantic[index] in reference_class_set and reference_id not in ignored_reference_set
        )
        if prediction_active or reference_active:
            evaluated_point_count += 1
        if prediction_active:
            prediction_sizes[prediction_id] = prediction_sizes.get(prediction_id, 0) + 1
        if reference_active:
            reference_sizes[reference_id] = reference_sizes.get(reference_id, 0) + 1
        if prediction_active and reference_active:
            key = (prediction_id, reference_id)
            intersections[key] = intersections.get(key, 0) + 1

    prediction_ids = sorted(prediction_sizes)
    reference_ids = sorted(reference_sizes)

    scores: list[list[float]] = []
    for prediction_id in prediction_ids:
        score_row: list[float] = []
        for reference_id in reference_ids:
            intersection = intersections.get((prediction_id, reference_id), 0)
            union = prediction_sizes[prediction_id] + reference_sizes[reference_id] - intersection
            score_row.append(intersection / union if union else 0.0)
        scores.append(score_row)

    pairs = maximum_cardinality_threshold_matching(scores, iou_threshold)
    matches = tuple(
        Match(
            prediction_id=prediction_ids[prediction_index],
            reference_id=reference_ids[reference_index],
            iou=scores[prediction_index][reference_index],
        )
        for prediction_index, reference_index in pairs
    )
    true_positives = len(matches)
    false_positives = len(prediction_ids) - true_positives
    false_negatives = len(reference_ids) - true_positives
    precision, recall, f1 = precision_recall_f1(true_positives, false_positives, false_negatives)
    matched_ious = [match.iou for match in matches]
    return EvaluationResult(
        protocol=PROTOCOL_ID,
        point_count=size,
        evaluated_point_count=evaluated_point_count,
        predicted_instances=len(prediction_ids),
        reference_instances=len(reference_ids),
        excluded_prediction_instances=len(excluded),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_matched_iou=(math.fsum(matched_ious) / len(matched_ious) if matched_ious else 0.0),
        median_matched_iou=(float(statistics.median(matched_ious)) if matched_ious else 0.0),
        matches=matches,
    )
