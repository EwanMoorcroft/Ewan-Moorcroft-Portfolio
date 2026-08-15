from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fpl_forecasting.cli import main
from fpl_forecasting.features import FEATURE_SCHEMA_SHA256
from fpl_forecasting.models import ARTIFACT_FORMAT
from fpl_forecasting.operational import MANIFEST_FORMAT, PREDICTION_FORMAT
from fpl_forecasting.synthetic import write_synthetic_gameweeks

SEASON_ID = "2024-25"


def _train(gameweeks: Path, artifact: Path) -> int:
    return main(
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


def _predict(
    gameweeks: Path,
    artifact: Path,
    predictions: Path,
    manifest: Path,
    *,
    season: str = SEASON_ID,
    expected_as_of_gw: int = 10,
    config: Path | None = None,
) -> int:
    arguments = [
        "predict",
        "--gameweek-dir",
        str(gameweeks),
        "--artifact",
        str(artifact),
        "--season",
        season,
        "--expected-as-of-gw",
        str(expected_as_of_gw),
        "--completion-status",
        "completed",
        "--predictions",
        str(predictions),
        "--manifest",
        str(manifest),
    ]
    if config is not None:
        arguments.extend(["--config", str(config)])
    return main(arguments)


def test_cli_train_and_predict_write_ranked_canonical_outputs(
    synthetic_directory: Path,
    tmp_path: Path,
    capsys,
) -> None:
    artifact = tmp_path / "ridge.json"
    predictions_path = tmp_path / "predictions.json"
    manifest_path = tmp_path / "manifest.json"

    assert _train(synthetic_directory, artifact) == 0
    train_summary = json.loads(capsys.readouterr().out)
    assert train_summary["artifact_format"] == ARTIFACT_FORMAT
    assert train_summary["season_id"] == SEASON_ID

    assert _predict(synthetic_directory, artifact, predictions_path, manifest_path) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["as_of_gw"] == 10
    assert summary["target_gw"] == 11
    assert summary["caller_declared_completion_status"] == "completed"

    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert predictions["prediction_format"] == PREDICTION_FORMAT
    assert predictions["caller_declared_completion"]["status"] == "completed"
    assert len(predictions["predictions"]) == 16
    assert [row["rank"] for row in predictions["predictions"]] == list(range(1, 17))
    observed_order = [
        (-row["predicted_points"], row["player_id"]) for row in predictions["predictions"]
    ]
    assert observed_order == sorted(observed_order)

    assert manifest["manifest_format"] == MANIFEST_FORMAT
    assert (
        manifest["hashes"]["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    assert (
        manifest["hashes"]["predictions_sha256"]
        == hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    )
    assert manifest["hashes"]["feature_schema_sha256"] == FEATURE_SCHEMA_SHA256
    assert manifest["hashes"]["input_sha256"] == manifest["inputs"]["aggregate_sha256"]
    assert manifest["artifact"]["file"] == artifact.name
    assert manifest["predictions"]["file"] == predictions_path.name
    assert all("/" not in entry["file"] for entry in manifest["inputs"]["files"])

    first_prediction_bytes = predictions_path.read_bytes()
    first_manifest_bytes = manifest_path.read_bytes()
    assert _predict(synthetic_directory, artifact, predictions_path, manifest_path) == 0
    capsys.readouterr()
    assert predictions_path.read_bytes() == first_prediction_bytes
    assert manifest_path.read_bytes() == first_manifest_bytes


def test_predict_rejects_stale_and_extra_as_of_declarations(
    synthetic_directory: Path,
    tmp_path: Path,
    capsys,
) -> None:
    artifact = tmp_path / "ridge.json"
    assert _train(synthetic_directory, artifact) == 0
    capsys.readouterr()

    assert (
        _predict(
            synthetic_directory,
            artifact,
            tmp_path / "stale-predictions.json",
            tmp_path / "stale-manifest.json",
            expected_as_of_gw=11,
        )
        == 2
    )
    assert "sequence is stale" in capsys.readouterr().err

    assert (
        _predict(
            synthetic_directory,
            artifact,
            tmp_path / "ahead-predictions.json",
            tmp_path / "ahead-manifest.json",
            expected_as_of_gw=9,
        )
        == 2
    )
    assert "extra or future" in capsys.readouterr().err


def test_predict_rejects_season_config_and_model_boundary_mismatches(
    tmp_path: Path,
    capsys,
) -> None:
    short = tmp_path / "short"
    long = tmp_path / "long"
    write_synthetic_gameweeks(short, gameweeks=8, players=12, seed=4)
    write_synthetic_gameweeks(long, gameweeks=10, players=12, seed=4)
    short_artifact = tmp_path / "short-model.json"
    long_artifact = tmp_path / "long-model.json"
    assert _train(short, short_artifact) == 0
    capsys.readouterr()
    assert _train(long, long_artifact) == 0
    capsys.readouterr()

    assert (
        _predict(
            long,
            short_artifact,
            tmp_path / "stale.json",
            tmp_path / "stale-manifest.json",
        )
        == 2
    )
    assert "Artifact is stale" in capsys.readouterr().err

    assert (
        _predict(
            short,
            long_artifact,
            tmp_path / "ahead.json",
            tmp_path / "ahead-manifest.json",
            expected_as_of_gw=8,
        )
        == 2
    )
    assert "Artifact is ahead" in capsys.readouterr().err

    assert (
        _predict(
            long,
            long_artifact,
            tmp_path / "season.json",
            tmp_path / "season-manifest.json",
            season="2025-26",
        )
        == 2
    )
    assert "Snapshot season mismatch" in capsys.readouterr().err

    changed_config = tmp_path / "changed-config.json"
    changed_config.write_text('{"ridge_alpha":3.0}', encoding="utf-8")
    assert (
        _predict(
            long,
            long_artifact,
            tmp_path / "config.json",
            tmp_path / "config-manifest.json",
            config=changed_config,
        )
        == 2
    )
    assert "config hash" in capsys.readouterr().err


def test_train_and_predict_reject_mislabeled_snapshot_seasons(
    tmp_path: Path,
    capsys,
) -> None:
    correct = tmp_path / "correct-season"
    wrong = tmp_path / "wrong-season"
    write_synthetic_gameweeks(
        correct,
        gameweeks=10,
        players=12,
        seed=8,
        season_id=SEASON_ID,
    )
    write_synthetic_gameweeks(
        wrong,
        gameweeks=10,
        players=12,
        seed=8,
        season_id="2025-26",
    )
    artifact = tmp_path / "ridge.json"

    assert _train(wrong, artifact) == 2
    assert "Snapshot season mismatch" in capsys.readouterr().err
    assert not artifact.exists()

    assert _train(correct, artifact) == 0
    capsys.readouterr()
    assert (
        _predict(
            wrong,
            artifact,
            tmp_path / "predictions.json",
            tmp_path / "manifest.json",
        )
        == 2
    )
    assert "Snapshot season mismatch" in capsys.readouterr().err
