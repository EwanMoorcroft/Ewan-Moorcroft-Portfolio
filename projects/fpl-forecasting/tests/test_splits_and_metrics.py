from __future__ import annotations

import numpy as np
import pytest

from fpl_forecasting.errors import DataContractError, SplitError
from fpl_forecasting.features import build_training_frame
from fpl_forecasting.metrics import (
    evaluate_predictions,
    ndcg_at_k,
    spearman_correlation,
    top_k_overlap,
)
from fpl_forecasting.splits import rolling_origin_splits


def test_rolling_splits_are_strictly_chronological(synthetic_snapshots) -> None:
    frame = build_training_frame(synthetic_snapshots)
    folds = list(
        rolling_origin_splits(
            frame,
            minimum_train_gameweeks=4,
            test_gameweeks_per_fold=1,
            step=1,
        )
    )
    assert len(folds) == 5
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert set(fold.train_indices).isdisjoint(set(fold.test_indices))
        assert (
            frame.iloc[fold.train_indices]["target_gw"].max()
            < frame.iloc[fold.test_indices]["target_gw"].min()
        )


def test_rolling_splits_reject_fractional_gameweek_boundaries(synthetic_snapshots) -> None:
    frame = build_training_frame(synthetic_snapshots).astype({"target_gw": "float64"})
    frame.loc[0, "target_gw"] = 2.5
    with pytest.raises(SplitError, match="positive integers"):
        list(
            rolling_origin_splits(
                frame,
                minimum_train_gameweeks=4,
                test_gameweeks_per_fold=1,
                step=1,
            )
        )


def test_rolling_splits_reject_invalid_window_parameters(synthetic_snapshots) -> None:
    frame = build_training_frame(synthetic_snapshots)
    with pytest.raises(SplitError, match="must be an integer"):
        list(
            rolling_origin_splits(
                frame,
                minimum_train_gameweeks=4.5,
                test_gameweeks_per_fold=1,
                step=1,
            )
        )
    with pytest.raises(SplitError, match="overlapping test windows"):
        list(
            rolling_origin_splits(
                frame,
                minimum_train_gameweeks=4,
                test_gameweeks_per_fold=2,
                step=1,
            )
        )


def test_perfect_predictions_have_perfect_metrics() -> None:
    actual = [1.0, 4.0, 2.0, 8.0]
    predicted = actual.copy()
    bundle = evaluate_predictions(actual, predicted, [6, 6, 6, 6], top_k=2)
    assert bundle.mae == 0.0
    assert bundle.rmse == 0.0
    assert bundle.r2 == 1.0
    assert bundle.spearman == 1.0
    assert bundle.ndcg_at_k == 1.0
    assert bundle.top_k_overlap == 1.0


def test_ranking_helpers_on_reversed_order() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = actual[::-1]
    assert spearman_correlation(actual, predicted) == -1.0
    assert 0.0 <= ndcg_at_k(actual, predicted, 3) < 1.0
    assert top_k_overlap(actual, predicted, 2) == 0.0


def test_ranking_metrics_group_before_macro_averaging() -> None:
    actual = [1.0, 2.0, 1.0, 2.0, 3.0, 4.0]
    predicted = [1.0, 2.0, 4.0, 3.0, 2.0, 1.0]
    bundle = evaluate_predictions(actual, predicted, [6, 6, 7, 7, 7, 7], top_k=2)
    assert bundle.spearman == 0.0
    assert bundle.gameweeks == 2


def test_ranking_groups_reject_fractional_gameweeks() -> None:
    with pytest.raises(DataContractError, match="positive integer"):
        evaluate_predictions([1.0, 2.0], [1.0, 2.0], [6.1, 6.9], top_k=1)


def test_ranking_helpers_reject_fractional_cutoffs() -> None:
    with pytest.raises(DataContractError, match="positive integer"):
        ndcg_at_k([1.0, 2.0], [1.0, 2.0], 1.5)
    with pytest.raises(DataContractError, match="positive integer"):
        top_k_overlap([1.0, 2.0], [1.0, 2.0], 1.5)
    with pytest.raises(DataContractError, match="positive integer"):
        evaluate_predictions([1.0, 2.0], [1.0, 2.0], [6, 6], top_k=1.5)


def test_metrics_and_splits_reject_overflowing_integers_cleanly(
    synthetic_snapshots,
) -> None:
    enormous = 10**10_000
    with pytest.raises(DataContractError, match="finite numeric vectors"):
        evaluate_predictions([enormous], [1], [1], top_k=1)
    with pytest.raises(DataContractError, match="positive integer"):
        evaluate_predictions([1], [1], [enormous], top_k=1)

    frame = build_training_frame(synthetic_snapshots).astype({"target_gw": object})
    frame.loc[0, "target_gw"] = enormous
    with pytest.raises(SplitError, match="finite"):
        list(
            rolling_origin_splits(
                frame,
                minimum_train_gameweeks=4,
                test_gameweeks_per_fold=1,
                step=1,
            )
        )
