"""Rolling-origin comparison of baselines and ridge regression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProtocolConfig
from .features import FEATURE_COLUMNS, TARGET_COLUMN, assert_leak_free_frame
from .metrics import evaluate_predictions
from .models import fit_ridge
from .splits import rolling_origin_splits

MODEL_NAMES: tuple[str, ...] = (
    "last_gameweek",
    "rolling_3_mean",
    "training_mean",
    "ridge_regression",
)


@dataclass(frozen=True)
class EvaluationResult:
    """Serializable report plus row-level out-of-fold predictions."""

    report: dict[str, Any]
    predictions: pd.DataFrame

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    def write_predictions(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.predictions.to_csv(destination, index=False)
        return destination


def _prediction_block(
    test: pd.DataFrame,
    *,
    fold_number: int,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    block = test.loc[:, ["player_id", "as_of_gw", "target_gw", TARGET_COLUMN]].copy()
    block.insert(0, "fold", fold_number)
    block = block.rename(columns={TARGET_COLUMN: "actual_points"})
    for name in MODEL_NAMES:
        block[name] = np.asarray(predictions[name], dtype=float)
    return block


def evaluate_models(frame: pd.DataFrame, config: ProtocolConfig | None = None) -> EvaluationResult:
    """Evaluate every candidate only on gameweeks later than its training data."""

    protocol = config or ProtocolConfig()
    assert_leak_free_frame(frame)
    folds = list(
        rolling_origin_splits(
            frame,
            minimum_train_gameweeks=protocol.minimum_train_gameweeks,
            test_gameweeks_per_fold=protocol.test_gameweeks_per_fold,
            step=protocol.split_step,
        )
    )

    prediction_blocks: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in folds:
        train = frame.iloc[fold.train_indices].copy()
        test = frame.iloc[fold.test_indices].copy()
        if int(train["target_gw"].max()) >= int(test["target_gw"].min()):
            raise RuntimeError("Chronological boundary check failed")

        ridge = fit_ridge(train, alpha=protocol.ridge_alpha)
        x_test = test.loc[:, list(FEATURE_COLUMNS)]
        candidates = {
            "last_gameweek": test["lag_1_points"].to_numpy(dtype=float),
            "rolling_3_mean": test["rolling_3_points_mean"].to_numpy(dtype=float),
            "training_mean": np.full(len(test), float(train[TARGET_COLUMN].mean()), dtype=float),
            "ridge_regression": ridge.predict(x_test),
        }
        prediction_blocks.append(
            _prediction_block(test, fold_number=fold.number, predictions=candidates)
        )

        fold_metrics: dict[str, Any] = {}
        for name in MODEL_NAMES:
            fold_metrics[name] = evaluate_predictions(
                test[TARGET_COLUMN],
                candidates[name],
                test["target_gw"],
                top_k=protocol.ranking_top_k,
            ).to_dict()
        fold_reports.append(
            {
                "fold": fold.number,
                "train_gameweek_start": fold.train_gameweeks[0],
                "train_gameweek_end": fold.train_gameweeks[-1],
                "test_gameweek_start": fold.test_gameweeks[0],
                "test_gameweek_end": fold.test_gameweeks[-1],
                "train_rows": len(train),
                "test_rows": len(test),
                "metrics": fold_metrics,
            }
        )

    out_of_fold = pd.concat(prediction_blocks, ignore_index=True)
    duplicate_keys = out_of_fold.duplicated(["player_id", "target_gw"], keep=False)
    if bool(duplicate_keys.any()):
        raise RuntimeError("Rolling test windows produced duplicate out-of-fold rows")

    aggregate_metrics: dict[str, Any] = {}
    for name in MODEL_NAMES:
        aggregate_metrics[name] = evaluate_predictions(
            out_of_fold["actual_points"],
            out_of_fold[name],
            out_of_fold["target_gw"],
            top_k=protocol.ranking_top_k,
        ).to_dict()

    report: dict[str, Any] = {
        "report_format": "fpl-rolling-evaluation-v1",
        "protocol": protocol.to_dict(),
        "feature_contract": list(FEATURE_COLUMNS),
        "data": {
            "rows": len(frame),
            "players": int(frame["player_id"].nunique()),
            "target_gameweek_start": int(frame["target_gw"].min()),
            "target_gameweek_end": int(frame["target_gw"].max()),
            "evaluated_gameweek_start": int(out_of_fold["target_gw"].min()),
            "evaluated_gameweek_end": int(out_of_fold["target_gw"].max()),
            "evaluated_rows": len(out_of_fold),
            "folds": len(folds),
        },
        "models": aggregate_metrics,
        "folds": fold_reports,
    }
    return EvaluationResult(report=report, predictions=out_of_fold)
