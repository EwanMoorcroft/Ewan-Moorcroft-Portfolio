"""Classification and calibration metrics."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any


def expected_calibration_error(
    labels,
    probabilities,
    bins: int = 15,
) -> float:
    import numpy as np

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in pairwise(edges):
        in_bin = (confidences > lower) & (confidences <= upper)
        if not np.any(in_bin):
            continue
        accuracy = correct[in_bin].mean()
        confidence = confidences[in_bin].mean()
        value += float(in_bin.mean() * abs(accuracy - confidence))
    return value


def classification_metrics(
    labels,
    probabilities,
    label_names: Sequence[str],
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        log_loss,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=1)
    indices = np.arange(len(label_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=indices,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=indices,
        average="macro",
        zero_division=0,
    )
    one_hot = np.eye(len(label_names), dtype=np.float64)[labels]
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(label_names)
    }
    return {
        "image_count": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "matthews_correlation_coefficient": float(matthews_corrcoef(labels, predictions)),
        "log_loss": float(log_loss(labels, probabilities, labels=indices)),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "expected_calibration_error_15_bins": expected_calibration_error(labels, probabilities),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=indices).tolist(),
        "per_class": per_class,
    }
