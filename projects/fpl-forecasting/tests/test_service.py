from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

import fpl_forecasting.service as service_api
from fpl_forecasting.config import ProtocolConfig
from fpl_forecasting.contracts import canonical_json_bytes, sha256_bytes
from fpl_forecasting.data import load_gameweeks
from fpl_forecasting.errors import DataContractError
from fpl_forecasting.features import (
    FEATURE_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    build_training_frame,
)
from fpl_forecasting.models import ARTIFACT_FORMAT, FittedRidge, fit_ridge
from fpl_forecasting.service import MAX_PLAYER_IDS, create_app
from fpl_forecasting.storage import (
    BatchRequest,
    ModelRegistration,
    OperationalStore,
    StorageError,
    StoredPrediction,
)

SEASON_ID = "2024-25"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _deployment(
    synthetic_directory: Path,
    tmp_path: Path,
    *,
    with_batch: bool = False,
):
    protocol = ProtocolConfig()
    snapshots = load_gameweeks(synthetic_directory, expected_season_id=SEASON_ID)
    training = build_training_frame(snapshots)
    model = fit_ridge(
        training,
        alpha=protocol.ridge_alpha,
        season_id=SEASON_ID,
        protocol_config_sha256=protocol.sha256(),
    )
    artifact_path = model.save(tmp_path / "private-config" / "ridge.json")
    artifact_sha256 = sha256_bytes(artifact_path.read_bytes())
    database_path = tmp_path / "private-config" / "forecasts.duckdb"
    writable_store = OperationalStore(database_path)
    registration = ModelRegistration(
        season_id=SEASON_ID,
        artifact_format=ARTIFACT_FORMAT,
        artifact_sha256=artifact_sha256,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        protocol_config_sha256=protocol.sha256(),
        trained_through_target_gw=model.trained_through_target_gw,
        training_rows=model.training_rows,
    )
    with writable_store.writer() as writer:
        writer.register_model(registration)
        if with_batch:
            writer.persist_batch(
                BatchRequest(
                    model_id=registration.model_id,
                    season_id=SEASON_ID,
                    as_of_gw=10,
                    target_gw=11,
                    source_snapshot_sha256=SHA_A,
                    forecast_frame_sha256=SHA_B,
                    feature_schema_sha256=FEATURE_SCHEMA_SHA256,
                    protocol_config_sha256=protocol.sha256(),
                    predictions_file_sha256=SHA_A,
                    manifest_file_sha256=SHA_B,
                ),
                [
                    StoredPrediction(player_id=2, prediction=7.5, rank=1),
                    StoredPrediction(player_id=1, prediction=6.0, rank=2),
                ],
            )
    app = create_app(
        artifact_path=artifact_path,
        gameweek_dir=synthetic_directory,
        database_path=database_path,
        season_id=SEASON_ID,
        expected_as_of_gw=10,
        completion_status="completed",
    )
    return app, writable_store, model


def test_health_and_model_metadata_do_not_expose_sensitive_runtime_state(
    synthetic_directory: Path,
    tmp_path: Path,
) -> None:
    app, _store, model = _deployment(synthetic_directory, tmp_path)
    with TestClient(app) as client:
        health = client.get("/health")
        metadata = client.get("/model")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model": "ready",
        "database": "ready",
        "schema_version": 1,
    }
    assert metadata.status_code == 200
    body = metadata.json()
    assert body["artifact_format"] == ARTIFACT_FORMAT
    assert body["model_id"] == app.state.runtime.model_registration.model_id
    assert body["season_id"] == SEASON_ID
    assert body["trained_through_target_gw"] == 10
    serialized = metadata.text
    assert str(tmp_path) not in serialized
    assert "coefficients" not in serialized
    assert "feature_means" not in serialized
    assert str(model.coefficients[0]) not in serialized


def test_predict_scores_shared_core_in_memory_without_database_mutation(
    synthetic_directory: Path,
    tmp_path: Path,
) -> None:
    app, store, _model = _deployment(synthetic_directory, tmp_path)
    before = store.counts()
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"season_id": SEASON_ID, "as_of_gw": 10, "player_ids": [1, 2]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["target_gw"] == 11
    assert body["population"] == 16
    assert {item["player_id"] for item in body["predictions"]} == {1, 2}
    assert all(
        set(item) == {"rank", "player_id", "predicted_points"} for item in body["predictions"]
    )
    assert store.counts() == before


@pytest.mark.parametrize(
    "payload",
    [
        {"season_id": SEASON_ID, "as_of_gw": True, "player_ids": [1]},
        {"season_id": SEASON_ID, "as_of_gw": 10, "player_ids": [1, 1]},
        {"season_id": "2024-26", "as_of_gw": 10, "player_ids": [1]},
        {"season_id": SEASON_ID, "as_of_gw": 10, "player_ids": [1], "path": "/tmp"},
        {
            "season_id": SEASON_ID,
            "as_of_gw": 10,
            "player_ids": list(range(1, MAX_PLAYER_IDS + 2)),
        },
    ],
)
def test_predict_request_is_strict_and_bounded(
    synthetic_directory: Path,
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    app, _store, _model = _deployment(synthetic_directory, tmp_path)
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "Request validation failed"}


def test_predict_rejects_wrong_boundary_and_unknown_player(
    synthetic_directory: Path,
    tmp_path: Path,
) -> None:
    app, _store, _model = _deployment(synthetic_directory, tmp_path)
    with TestClient(app) as client:
        mismatch = client.post(
            "/predict",
            json={"season_id": SEASON_ID, "as_of_gw": 9, "player_ids": [1]},
        )
        unknown = client.post(
            "/predict",
            json={"season_id": SEASON_ID, "as_of_gw": 10, "player_ids": [999999]},
        )
    assert mismatch.status_code == 409
    assert unknown.status_code == 404


def test_predict_rejects_oversized_body_before_parsing(
    synthetic_directory: Path,
    tmp_path: Path,
) -> None:
    app, _store, _model = _deployment(synthetic_directory, tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            content=b"{" + (b"x" * 20_000) + b"}",
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body is too large"}


def test_stored_prediction_read_is_bounded_and_http_surface_is_read_only(
    synthetic_directory: Path,
    tmp_path: Path,
) -> None:
    app, _store, _model = _deployment(synthetic_directory, tmp_path, with_batch=True)
    route_methods = {(route.path, tuple(sorted(route.methods or ()))) for route in app.routes}
    assert route_methods == {
        ("/health", ("GET",)),
        ("/metrics", ("GET",)),
        ("/model", ("GET",)),
        ("/predict", ("POST",)),
        ("/predictions/{season_id}/{target_gw}", ("GET",)),
    }
    with TestClient(app) as client:
        response = client.get(f"/predictions/{SEASON_ID}/11?limit=1")
        invalid_limit = client.get(f"/predictions/{SEASON_ID}/11?limit=501")
    assert response.status_code == 200
    body = response.json()
    assert body["total_predictions"] == 2
    assert body["returned_predictions"] == 1
    assert body["predictions"] == [{"player_id": 2, "predicted_points": 7.5, "rank": 1}]
    assert body["predictions_file_sha256"] == SHA_A
    assert body["manifest_file_sha256"] == SHA_B
    assert invalid_limit.status_code == 422


def test_stored_reads_are_bound_to_the_configured_model(
    synthetic_directory: Path,
    tmp_path: Path,
) -> None:
    app, store, _model = _deployment(synthetic_directory, tmp_path, with_batch=True)
    configured = app.state.runtime.model_registration
    other_model = replace(configured, artifact_sha256="c" * 64)
    with store.writer() as writer:
        writer.register_model(other_model)
        writer.persist_batch(
            BatchRequest(
                model_id=other_model.model_id,
                season_id=SEASON_ID,
                as_of_gw=10,
                target_gw=11,
                source_snapshot_sha256="d" * 64,
                forecast_frame_sha256="e" * 64,
                feature_schema_sha256=FEATURE_SCHEMA_SHA256,
                protocol_config_sha256=configured.protocol_config_sha256,
                predictions_file_sha256="f" * 64,
                manifest_file_sha256="0" * 64,
            ),
            [
                StoredPrediction(player_id=99, prediction=100.0, rank=1),
                StoredPrediction(player_id=98, prediction=90.0, rank=2),
            ],
        )
    with TestClient(app) as client:
        metadata = client.get("/model")
        response = client.get(f"/predictions/{SEASON_ID}/11")
    assert metadata.json()["model_id"] == configured.model_id
    assert response.status_code == 200
    assert response.json()["model_id"] == configured.model_id
    assert [row["player_id"] for row in response.json()["predictions"]] == [2, 1]


def test_service_startup_requires_exact_model_registration(
    synthetic_directory: Path,
    tmp_path: Path,
) -> None:
    _app, _store, _model = _deployment(synthetic_directory, tmp_path)
    unregistered_database = tmp_path / "unregistered.duckdb"
    OperationalStore(unregistered_database)
    with pytest.raises(StorageError, match="not registered"):
        create_app(
            artifact_path=tmp_path / "private-config" / "ridge.json",
            gameweek_dir=synthetic_directory,
            database_path=unregistered_database,
            season_id=SEASON_ID,
            expected_as_of_gw=10,
            completion_status="completed",
        )


def test_service_wraps_database_failure_as_sanitized_503(
    synthetic_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _store, _model = _deployment(synthetic_directory, tmp_path, with_batch=True)

    def fail_connect():
        raise duckdb.IOException(f"private DB path: {tmp_path}")

    monkeypatch.setattr(app.state.runtime.store, "_connect", fail_connect)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(f"/predictions/{SEASON_ID}/11")
    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction store unavailable"}
    assert str(tmp_path) not in response.text


def test_service_bounds_configured_artifact_file(
    synthetic_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app, _store, _model = _deployment(synthetic_directory, tmp_path)
    monkeypatch.setattr(service_api, "MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(DataContractError, match="exceeds the supported size"):
        create_app(
            artifact_path=tmp_path / "private-config" / "ridge.json",
            gameweek_dir=synthetic_directory,
            database_path=tmp_path / "private-config" / "forecasts.duckdb",
            season_id=SEASON_ID,
            expected_as_of_gw=10,
            completion_status="completed",
        )


def test_service_rejects_cross_season_snapshots(
    synthetic_directory: Path,
    tmp_path: Path,
) -> None:
    _app, _store, _model = _deployment(synthetic_directory, tmp_path)
    for snapshot in synthetic_directory.glob("gameweek-*.json"):
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        payload["season_id"] = "2025-26"
        snapshot.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(DataContractError, match="season"):
        create_app(
            artifact_path=tmp_path / "private-config" / "ridge.json",
            gameweek_dir=synthetic_directory,
            database_path=tmp_path / "private-config" / "forecasts.duckdb",
            season_id=SEASON_ID,
            expected_as_of_gw=10,
            completion_status="completed",
        )


def test_metrics_use_only_fixed_labels(
    synthetic_directory: Path,
    tmp_path: Path,
) -> None:
    app, _store, _model = _deployment(synthetic_directory, tmp_path)
    with TestClient(app) as client:
        client.get("/not-a-real-player/123456")
        client.post(
            "/predict",
            json={"season_id": SEASON_ID, "as_of_gw": 10, "player_ids": [1, 2]},
        )
        response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert 'endpoint="unmatched"' in body
    assert 'endpoint="/predict"' in body
    assert "fpl_predictions_scored_total 2" in body
    assert SEASON_ID not in body
    assert "123456" not in body
    assert str(tmp_path) not in body


def test_unexpected_scoring_error_is_sanitized(
    synthetic_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, _store, _model = _deployment(synthetic_directory, tmp_path)

    def fail_with_private_path(_self, _values):
        raise RuntimeError(f"failure at {tmp_path / 'secret' / 'artifact.json'}")

    monkeypatch.setattr(FittedRidge, "predict", fail_with_private_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/predict",
            json={"season_id": SEASON_ID, "as_of_gw": 10, "player_ids": [1]},
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal service error"}
    assert str(tmp_path) not in response.text
