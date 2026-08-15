"""Regression and within-gameweek ranking metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .errors import DataContractError


@dataclass(frozen=True)
class MetricBundle:
    """Combined point-forecast and ranking quality measures."""

    mae: float
    rmse: float
    r2: float
    spearman: float
    ndcg_at_k: float
    top_k_overlap: float
    top_k: int
    rows: int
    gameweeks: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "spearman": self.spearman,
            f"ndcg_at_{self.top_k}": self.ndcg_at_k,
            f"top_{self.top_k}_overlap": self.top_k_overlap,
            "rows": self.rows,
            "gameweeks": self.gameweeks,
        }


def _arrays(actual: Iterable[float], predicted: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    try:
        y_true = np.asarray(list(actual), dtype=float)
        y_pred = np.asarray(list(predicted), dtype=float)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DataContractError(
            "Actual and predicted values must be finite numeric vectors"
        ) from exc
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise DataContractError("Actual and predicted values must be equal-length vectors")
    if len(y_true) == 0:
        raise DataContractError("Metric inputs cannot be empty")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise DataContractError("Metric inputs must be finite")
    return y_true, y_pred


def _gameweek_groups(values: Iterable[int], expected_length: int) -> np.ndarray:
    materialized = list(values)
    if len(materialized) != expected_length:
        raise DataContractError("target_gameweeks must align with predictions")
    groups: list[int] = []
    for index, value in enumerate(materialized):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise DataContractError(f"target_gameweeks[{index}] must be an integer")
        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise DataContractError(
                f"target_gameweeks[{index}] must be a positive integer"
            ) from exc
        if not math.isfinite(numeric) or numeric != int(numeric) or numeric < 1:
            raise DataContractError(f"target_gameweeks[{index}] must be a positive integer")
        groups.append(int(numeric))
    return np.asarray(groups, dtype=int)


def _positive_k(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise DataContractError(f"{field} must be a positive integer")
    return int(value)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered)) * float(np.dot(right_centered, right_centered))
    )
    if denominator == 0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def spearman_correlation(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Calculate Spearman correlation with average ranks for ties."""

    y_true, y_pred = _arrays(actual, predicted)
    true_ranks = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    pred_ranks = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    return _pearson(true_ranks, pred_ranks)


def ndcg_at_k(actual: Iterable[float], predicted: Iterable[float], k: int) -> float:
    """Calculate normalized discounted gain using non-negative FPL points."""

    y_true, y_pred = _arrays(actual, predicted)
    limit = min(_positive_k(k, "k"), len(y_true))
    relevance = np.clip(y_true, 0.0, 50.0)
    discounts = np.log2(np.arange(2, limit + 2, dtype=float))

    predicted_order = np.argsort(-y_pred, kind="stable")[:limit]
    ideal_order = np.argsort(-relevance, kind="stable")[:limit]
    predicted_gain = np.power(2.0, relevance[predicted_order]) - 1.0
    ideal_gain = np.power(2.0, relevance[ideal_order]) - 1.0
    dcg = float(np.sum(predicted_gain / discounts))
    ideal_dcg = float(np.sum(ideal_gain / discounts))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def top_k_overlap(actual: Iterable[float], predicted: Iterable[float], k: int) -> float:
    """Measure overlap between predicted and observed top-k player sets."""

    y_true, y_pred = _arrays(actual, predicted)
    limit = min(_positive_k(k, "k"), len(y_true))
    actual_top = set(np.argsort(-y_true, kind="stable")[:limit].tolist())
    predicted_top = set(np.argsort(-y_pred, kind="stable")[:limit].tolist())
    return len(actual_top & predicted_top) / limit


def evaluate_predictions(
    actual: Iterable[float],
    predicted: Iterable[float],
    target_gameweeks: Iterable[int],
    *,
    top_k: int,
) -> MetricBundle:
    """Evaluate point error globally and ranking quality within each gameweek."""

    y_true, y_pred = _arrays(actual, predicted)
    groups = _gameweek_groups(target_gameweeks, len(y_true))
    validated_top_k = _positive_k(top_k, "top_k")

    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    total_variation = float(np.sum(np.square(y_true - y_true.mean())))
    r2 = 1.0 - float(np.sum(np.square(residual))) / total_variation if total_variation > 0 else 0.0

    spearman_values: list[float] = []
    ndcg_values: list[float] = []
    overlap_values: list[float] = []
    unique_groups = sorted(np.unique(groups).tolist())
    for gameweek in unique_groups:
        mask = groups == gameweek
        spearman_values.append(spearman_correlation(y_true[mask], y_pred[mask]))
        ndcg_values.append(ndcg_at_k(y_true[mask], y_pred[mask], validated_top_k))
        overlap_values.append(top_k_overlap(y_true[mask], y_pred[mask], validated_top_k))

    return MetricBundle(
        mae=mae,
        rmse=rmse,
        r2=r2,
        spearman=float(np.mean(spearman_values)),
        ndcg_at_k=float(np.mean(ndcg_values)),
        top_k_overlap=float(np.mean(overlap_values)),
        top_k=validated_top_k,
        rows=len(y_true),
        gameweeks=len(unique_groups),
    )
