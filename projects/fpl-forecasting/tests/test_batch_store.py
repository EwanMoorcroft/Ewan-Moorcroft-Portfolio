from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import fpl_forecasting.batch_store as batch_store_api
from fpl_forecasting.cli import main
from fpl_forecasting.contracts import canonical_json_bytes, sha256_bytes
from fpl_forecasting.service import create_app
from fpl_forecasting.storage import OperationalStore

SEASON_ID = "2024-25"


def _generate(
    gameweeks: Path,
    artifact: Path,
    predictions: Path,
    manifest: Path,
) -> None:
    assert (
        main(
            [
                "train",
                "--gameweek-dir",
                str(gameweeks),
                "--artifact",
                str(artifact),
                "--season",
                SEASON_ID,
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "predict",
                "--gameweek-dir",
                str(gameweeks),
                "--artifact",
                str(artifact),
                "--season",
                SEASON_ID,
                "--expected-as-of-gw",
                "10",
                "--completion-status",
                "completed",
                "--predictions",
                str(predictions),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )


def _store(
    database: Path,
    gameweeks: Path,
    artifact: Path,
    predictions: Path,
    manifest: Path,
) -> int:
    return main(
        [
            "store-batch",
            "--database",
            str(database),
            "--gameweek-dir",
            str(gameweeks),
            "--artifact",
            str(artifact),
            "--predictions",
            str(predictions),
            "--manifest",
            str(manifest),
        ]
    )


def test_store_batch_registers_replays_and_starts_read_service(
    synthetic_directory: Path,
    tmp_path: Path,
    capsys,
) -> None:
    artifact = tmp_path / "ridge.json"
    predictions = tmp_path / "predictions.json"
    manifest = tmp_path / "manifest.json"
    database = tmp_path / "forecasts.duckdb"
    _generate(synthetic_directory, artifact, predictions, manifest)
    capsys.readouterr()

    assert _store(database, synthetic_directory, artifact, predictions, manifest) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["command"] == "store-batch"
    assert first["predictions"] == 16
    assert first["replayed"] is False

    assert _store(database, synthetic_directory, artifact, predictions, manifest) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["run_id"] == first["run_id"]
    assert second["idempotency_key"] == first["idempotency_key"]
    assert second["replayed"] is True
    assert OperationalStore(database).counts() == {
        "models": 1,
        "runs": 1,
        "predictions": 16,
    }

    app = create_app(
        artifact_path=artifact,
        gameweek_dir=synthetic_directory,
        database_path=database,
        season_id=SEASON_ID,
        expected_as_of_gw=10,
        completion_status="completed",
    )
    with TestClient(app) as client:
        response = client.get(f"/predictions/{SEASON_ID}/11?limit=2")
    assert response.status_code == 200
    response_body = response.json()
    assert response_body["total_predictions"] == 16
    assert response_body["returned_predictions"] == 2
    assert response_body["predictions_file_sha256"] == sha256_bytes(predictions.read_bytes())
    assert response_body["manifest_file_sha256"] == sha256_bytes(manifest.read_bytes())


def test_store_batch_rejects_changed_predictions_before_database_write(
    synthetic_directory: Path,
    tmp_path: Path,
    capsys,
) -> None:
    artifact = tmp_path / "ridge.json"
    predictions = tmp_path / "predictions.json"
    manifest = tmp_path / "manifest.json"
    database = tmp_path / "must-not-exist.duckdb"
    _generate(synthetic_directory, artifact, predictions, manifest)
    capsys.readouterr()
    predictions.write_bytes(predictions.read_bytes() + b" ")

    assert _store(database, synthetic_directory, artifact, predictions, manifest) == 2
    assert "do not exactly match" in capsys.readouterr().err
    assert not database.exists()


def test_store_batch_rejects_changed_manifest_and_source_before_database_write(
    synthetic_directory: Path,
    tmp_path: Path,
    capsys,
) -> None:
    artifact = tmp_path / "ridge.json"
    predictions = tmp_path / "predictions.json"
    manifest = tmp_path / "manifest.json"
    database = tmp_path / "must-not-exist.duckdb"
    _generate(synthetic_directory, artifact, predictions, manifest)
    capsys.readouterr()

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["hashes"]["input_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    assert _store(database, synthetic_directory, artifact, predictions, manifest) == 2
    assert "does not exactly authenticate" in capsys.readouterr().err
    assert not database.exists()


def test_store_batch_cannot_overwrite_an_evidence_file(
    synthetic_directory: Path,
    tmp_path: Path,
    capsys,
) -> None:
    artifact = tmp_path / "ridge.json"
    predictions = tmp_path / "predictions.json"
    manifest = tmp_path / "manifest.json"
    _generate(synthetic_directory, artifact, predictions, manifest)
    capsys.readouterr()
    artifact_before = artifact.read_bytes()

    assert _store(artifact, synthetic_directory, artifact, predictions, manifest) == 2
    assert "cannot overwrite" in capsys.readouterr().err
    assert artifact.read_bytes() == artifact_before


def test_store_batch_rejects_real_source_tampering_before_database_write(
    synthetic_directory: Path,
    tmp_path: Path,
    capsys,
) -> None:
    artifact = tmp_path / "ridge.json"
    predictions = tmp_path / "predictions.json"
    manifest = tmp_path / "manifest.json"
    database = tmp_path / "must-not-exist.duckdb"
    _generate(synthetic_directory, artifact, predictions, manifest)
    capsys.readouterr()
    latest = synthetic_directory / "gameweek-10.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    payload["elements"][0]["stats"]["total_points"] += 1
    latest.write_bytes(canonical_json_bytes(payload))

    assert _store(database, synthetic_directory, artifact, predictions, manifest) == 2
    assert "do not exactly match" in capsys.readouterr().err
    assert not database.exists()


def test_store_batch_rejects_real_artifact_tampering_before_database_write(
    synthetic_directory: Path,
    tmp_path: Path,
    capsys,
) -> None:
    artifact = tmp_path / "ridge.json"
    predictions = tmp_path / "predictions.json"
    manifest = tmp_path / "manifest.json"
    database = tmp_path / "must-not-exist.duckdb"
    _generate(synthetic_directory, artifact, predictions, manifest)
    capsys.readouterr()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["coefficients"][0] += 1.0
    artifact.write_bytes(canonical_json_bytes(payload))

    assert _store(database, synthetic_directory, artifact, predictions, manifest) == 2
    assert "do not exactly match" in capsys.readouterr().err
    assert not database.exists()


def test_store_batch_bounds_predictions_file_before_database_write(
    synthetic_directory: Path,
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    artifact = tmp_path / "ridge.json"
    predictions = tmp_path / "predictions.json"
    manifest = tmp_path / "manifest.json"
    database = tmp_path / "must-not-exist.duckdb"
    _generate(synthetic_directory, artifact, predictions, manifest)
    capsys.readouterr()
    monkeypatch.setattr(batch_store_api, "MAX_PREDICTIONS_FILE_BYTES", 1)

    assert _store(database, synthetic_directory, artifact, predictions, manifest) == 2
    assert "exceeds the supported size" in capsys.readouterr().err
    assert not database.exists()


def test_store_batch_rejects_cross_season_snapshot_before_database_write(
    synthetic_directory: Path,
    tmp_path: Path,
    capsys,
) -> None:
    artifact = tmp_path / "ridge.json"
    predictions = tmp_path / "predictions.json"
    manifest = tmp_path / "manifest.json"
    database = tmp_path / "must-not-exist.duckdb"
    _generate(synthetic_directory, artifact, predictions, manifest)
    capsys.readouterr()
    snapshot = synthetic_directory / "gameweek-10.json"
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["season_id"] = "2025-26"
    snapshot.write_bytes(canonical_json_bytes(payload))

    assert _store(database, synthetic_directory, artifact, predictions, manifest) == 2
    assert "season" in capsys.readouterr().err
    assert not database.exists()
