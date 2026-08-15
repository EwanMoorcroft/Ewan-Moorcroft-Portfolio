from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import fpl_forecasting.evaluation as evaluation_api
from fpl_forecasting.config import ProtocolConfig
from fpl_forecasting.errors import ArtifactError
from fpl_forecasting.evaluation import MODEL_NAMES, evaluate_models
from fpl_forecasting.features import FEATURE_COLUMNS, TARGET_COLUMN, build_training_frame
from fpl_forecasting.models import FittedRidge, fit_ridge

SEASON_ID = "2024-25"


def _deployment_fit(frame, *, alpha: float = 3.0) -> FittedRidge:
    protocol = ProtocolConfig()
    return fit_ridge(
        frame,
        alpha=alpha,
        season_id=SEASON_ID,
        protocol_config_sha256=protocol.sha256(),
    )


def test_end_to_end_evaluation_is_out_of_fold(synthetic_snapshots) -> None:
    frame = build_training_frame(synthetic_snapshots)
    result = evaluate_models(frame, ProtocolConfig(ranking_top_k=5))
    assert set(result.report["models"]) == set(MODEL_NAMES)
    assert result.report["data"]["folds"] == 5
    assert result.report["data"]["evaluated_gameweek_start"] == 6
    assert result.report["data"]["evaluated_gameweek_end"] == 10
    assert not result.predictions.duplicated(["player_id", "target_gw"]).any()
    for fold in result.report["folds"]:
        assert fold["train_gameweek_end"] < fold["test_gameweek_start"]
    for name in MODEL_NAMES:
        assert np.isfinite(result.predictions[name]).all()


def test_safe_json_artifact_round_trip(synthetic_snapshots, tmp_path: Path) -> None:
    frame = build_training_frame(synthetic_snapshots)
    fitted = _deployment_fit(frame)
    path = fitted.save(tmp_path / "ridge.json")
    loaded = FittedRidge.load(path)
    x = frame.loc[:, list(FEATURE_COLUMNS)].head(12)
    np.testing.assert_allclose(fitted.predict(x), loaded.predict(x), rtol=0, atol=1e-12)
    assert loaded.feature_names == FEATURE_COLUMNS
    assert loaded.training_rows == len(frame)
    assert loaded.season_id == SEASON_ID


def test_fold_preprocessing_is_fitted_only_on_training_rows(
    synthetic_snapshots,
    monkeypatch,
) -> None:
    frame = build_training_frame(synthetic_snapshots)
    captured = []
    original_fit = evaluation_api.fit_ridge

    def recording_fit(train, *, alpha):
        model = original_fit(train, alpha=alpha)
        captured.append((train.copy(), model))
        return model

    monkeypatch.setattr(evaluation_api, "fit_ridge", recording_fit)
    result = evaluation_api.evaluate_models(frame, ProtocolConfig(ranking_top_k=5))
    assert len(captured) == len(result.report["folds"])

    for fold_report, (train, model) in zip(result.report["folds"], captured, strict=True):
        assert int(train["target_gw"].max()) == fold_report["train_gameweek_end"]
        assert model.trained_through_target_gw < fold_report["test_gameweek_start"]
        expected_means = train.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float).mean(axis=0)
        np.testing.assert_allclose(model.feature_means, expected_means, rtol=0, atol=0)


def test_future_value_injection_cannot_change_earlier_predictions(synthetic_snapshots) -> None:
    frame = build_training_frame(synthetic_snapshots)
    protocol = ProtocolConfig(ranking_top_k=5)
    baseline = evaluate_models(frame, protocol).predictions

    poisoned = frame.copy()
    final_gameweek = int(poisoned["target_gw"].max())
    final_rows = poisoned["target_gw"] == final_gameweek
    poisoned.loc[final_rows, list(FEATURE_COLUMNS)] = 1_000_000.0
    poisoned.loc[final_rows, TARGET_COLUMN] = 1_000_000.0
    challenged = evaluate_models(poisoned, protocol).predictions

    earlier = baseline["target_gw"] < final_gameweek
    for name in MODEL_NAMES:
        np.testing.assert_allclose(
            baseline.loc[earlier, name],
            challenged.loc[earlier, name],
            rtol=0,
            atol=0,
        )
    assert not np.array_equal(
        baseline.loc[~earlier, "ridge_regression"],
        challenged.loc[~earlier, "ridge_regression"],
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_rows", 12.5),
        ("trained_through_target_gw", 8.5),
        ("alpha", "3.0"),
    ],
)
def test_artifact_rejects_coerced_scalar_fields(
    synthetic_snapshots,
    field: str,
    value: object,
) -> None:
    payload = _deployment_fit(build_training_frame(synthetic_snapshots)).to_dict()
    payload[field] = value
    with pytest.raises(ArtifactError):
        FittedRidge.from_dict(payload)


def test_artifact_rejects_non_numeric_vectors_and_unknown_fields(synthetic_snapshots) -> None:
    payload = _deployment_fit(build_training_frame(synthetic_snapshots)).to_dict()
    payload["coefficients"][0] = "0.0"
    with pytest.raises(ArtifactError, match="must be numeric"):
        FittedRidge.from_dict(payload)

    payload = _deployment_fit(build_training_frame(synthetic_snapshots)).to_dict()
    payload["unexpected"] = "ignored metadata"
    with pytest.raises(ArtifactError, match="fields differ"):
        FittedRidge.from_dict(payload)


def test_artifact_rejects_duplicate_json_fields(
    synthetic_snapshots,
    tmp_path: Path,
) -> None:
    payload = _deployment_fit(build_training_frame(synthetic_snapshots)).to_dict()
    serialized = json.dumps(payload)
    serialized = serialized.replace(
        '"training_rows":',
        '"training_rows": 1, "training_rows":',
        1,
    )
    path = tmp_path / "duplicate-field.json"
    path.write_text(serialized, encoding="utf-8")
    with pytest.raises(ArtifactError, match="duplicate field"):
        FittedRidge.load(path)


@pytest.mark.parametrize("alpha", [True, float("inf"), "3.0"])
def test_direct_fit_rejects_invalid_alpha(synthetic_snapshots, alpha: object) -> None:
    frame = build_training_frame(synthetic_snapshots)
    with pytest.raises(ArtifactError):
        fit_ridge(frame, alpha=alpha)


def test_unspecified_identity_cannot_be_saved(synthetic_snapshots, tmp_path: Path) -> None:
    model = fit_ridge(build_training_frame(synthetic_snapshots), alpha=3.0)
    with pytest.raises(ArtifactError, match="requires season_id"):
        model.save(tmp_path / "unsafe.json")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_format", "fpl-ridge-json-v1", "Unsupported artifact format"),
        ("season_id", "2024-26", "consecutive"),
        ("feature_schema_sha256", "0" * 64, "schema hash"),
        ("protocol_config_sha256", True, "SHA-256"),
        pytest.param(
            "intercept",
            10**10_000,
            "must be finite",
            id="overflowing-intercept",
        ),
    ],
)
def test_artifact_v2_rejects_incompatible_identity(
    synthetic_snapshots,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _deployment_fit(build_training_frame(synthetic_snapshots)).to_dict()
    payload[field] = value
    with pytest.raises(ArtifactError, match=message):
        FittedRidge.from_dict(payload)
