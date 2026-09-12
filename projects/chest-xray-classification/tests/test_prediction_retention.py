"""Retained prediction rows must reproduce counts and calibration bins."""

import csv

import numpy as np
import pytest

from chest_xray_benchmark import evaluation


def test_retained_probabilities_have_safe_ids_and_reconstruct_error_counts(tmp_path):
    rows = [{"sha256": "a" * 64}, {"sha256": "b" * 64}]
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
    result = evaluation.retain_predictions(
        rows,
        np.array([0, 2]),
        probabilities,
        tmp_path / "test-predictions.csv",
    )
    saved = list(csv.DictReader((tmp_path / "test-predictions.csv").open()))
    assert [row["image_sha256"] for row in saved] == ["a" * 64, "b" * 64]
    assert [int(row["true_label_index"]) for row in saved] == [0, 2]
    assert float(saved[1]["probability_2"]) == 0.2
    assert result["error_count"] == 1
    occupied = [row for row in result["reliability_bins"] if row["count"]]
    assert sum(row["count"] for row in occupied) == 2
    assert sum(row["correct_count"] for row in occupied) == 1
    assert result["mean_confidence_incorrect"] == pytest.approx(0.7)


def test_unweighted_evaluation_refuses_inconsistent_loss():
    with pytest.raises(RuntimeError, match="cross entropy"):
        evaluation.verify_evaluation_loss(0.3967259, 0.6555965)
    evaluation.verify_evaluation_loss(0.65559653, 0.65559654)
