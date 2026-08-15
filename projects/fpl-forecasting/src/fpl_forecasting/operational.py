"""Fail-closed assembly of ranked next-gameweek batch forecasts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from .config import ProtocolConfig
from .contracts import sha256_bytes, validate_season_id
from .data import GameweekSnapshot
from .errors import ArtifactError, DataContractError
from .features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    assert_forecast_frame,
)
from .models import ARTIFACT_FORMAT, FittedRidge

PREDICTION_FORMAT = "fpl-ranked-predictions-v1"
MANIFEST_FORMAT = "fpl-operational-forecast-manifest-v1"
COMPLETION_STATUS = "completed"


def validate_expected_as_of(snapshots: Sequence[GameweekSnapshot], expected_as_of_gw: Any) -> int:
    """Require the caller's as-of boundary to equal the latest supplied snapshot."""

    if (
        isinstance(expected_as_of_gw, bool)
        or not isinstance(expected_as_of_gw, Integral)
        or int(expected_as_of_gw) < 1
    ):
        raise DataContractError("expected_as_of_gw must be a positive integer")
    if not snapshots:
        raise DataContractError("No completed snapshots were supplied")
    expected = int(expected_as_of_gw)
    latest = max(snapshot.gameweek for snapshot in snapshots)
    if latest < expected:
        raise DataContractError(
            f"Snapshot sequence is stale: expected as-of GW{expected}, latest is GW{latest}"
        )
    if latest > expected:
        raise DataContractError(
            "Snapshot sequence contains an extra or future as-of boundary: "
            f"expected GW{expected}, latest is GW{latest}"
        )
    return latest


def validate_deployment_model(
    model: FittedRidge,
    *,
    season_id: str,
    config_sha256: str,
    as_of_gw: int,
) -> None:
    """Bind a deployment model to season, config, schema, and forecast boundary."""

    canonical_season = validate_season_id(season_id)
    if model.season_id != canonical_season:
        raise ArtifactError(
            f"Artifact season mismatch: expected {canonical_season}, found {model.season_id}"
        )
    if model.protocol_config_sha256 != config_sha256:
        raise ArtifactError("Artifact protocol config hash does not match the effective config")
    if (
        model.feature_schema_version != FEATURE_SCHEMA_VERSION
        or model.feature_schema_sha256 != FEATURE_SCHEMA_SHA256
    ):
        raise ArtifactError("Artifact feature schema does not match this release")
    if model.trained_through_target_gw < as_of_gw:
        raise ArtifactError(
            "Artifact is stale: trained through target GW"
            f"{model.trained_through_target_gw}, forecast as-of is GW{as_of_gw}"
        )
    if model.trained_through_target_gw > as_of_gw:
        raise ArtifactError(
            "Artifact is ahead of the forecast boundary: trained through target GW"
            f"{model.trained_through_target_gw}, forecast as-of is GW{as_of_gw}"
        )


def ranked_prediction_payload(
    frame,
    model: FittedRidge,
    *,
    season_id: str,
    completion_status: str,
) -> dict[str, object]:
    """Score and deterministically rank the latest-snapshot player population."""

    assert_forecast_frame(frame)
    canonical_season = validate_season_id(season_id)
    if completion_status != COMPLETION_STATUS:
        raise DataContractError(
            "completion_status must be the explicit caller declaration 'completed'"
        )
    predictions = model.predict(frame.loc[:, list(FEATURE_COLUMNS)])
    if not all(math.isfinite(float(value)) for value in predictions):
        raise ArtifactError("Model produced non-finite predictions")

    records = [
        {
            "player_id": int(player_id),
            "predicted_points": float(prediction),
        }
        for player_id, prediction in zip(frame["player_id"], predictions, strict=True)
    ]
    records.sort(key=lambda item: (-item["predicted_points"], item["player_id"]))
    ranked = [{"rank": rank, **record} for rank, record in enumerate(records, start=1)]
    as_of_gw = int(frame["as_of_gw"].iloc[0])
    target_gw = int(frame["target_gw"].iloc[0])
    return {
        "prediction_format": PREDICTION_FORMAT,
        "season_id": canonical_season,
        "as_of_gw": as_of_gw,
        "target_gw": target_gw,
        "caller_declared_completion": {
            "status": completion_status,
            "scope": "all supplied gameweek snapshots through as_of_gw",
            "independent_source_verification": False,
        },
        "ranking_rule": "predicted_points descending, then player_id ascending",
        "predictions": ranked,
    }


def snapshot_input_audit(
    snapshots: Sequence[GameweekSnapshot],
) -> tuple[list[dict[str, object]], str]:
    """Hash exact snapshot bytes and their ordered aggregate identity."""

    files: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    for snapshot in sorted(snapshots, key=lambda item: item.gameweek):
        raw = snapshot.source.read_bytes()
        digest = sha256_bytes(raw)
        name = snapshot.source.name
        aggregate.update(f"{snapshot.gameweek}\0{name}\0{len(raw)}\0{digest}\n".encode())
        files.append(
            {
                "gameweek": snapshot.gameweek,
                "file": name,
                "bytes": len(raw),
                "sha256": digest,
            }
        )
    return files, aggregate.hexdigest()


def operational_manifest(
    *,
    season_id: str,
    as_of_gw: int,
    completion_status: str,
    protocol: ProtocolConfig,
    model: FittedRidge,
    artifact_path: str | Path,
    artifact_sha256: str,
    input_files: list[dict[str, object]],
    input_sha256: str,
    predictions_path: str | Path,
    predictions_sha256: str,
    prediction_rows: int,
) -> dict[str, object]:
    """Create the canonical audit manifest without clocks or machine-local paths."""

    canonical_season = validate_season_id(season_id)
    if completion_status != COMPLETION_STATUS:
        raise DataContractError("Manifest requires caller-declared completed inputs")
    return {
        "manifest_format": MANIFEST_FORMAT,
        "season_id": canonical_season,
        "as_of_gw": as_of_gw,
        "target_gw": as_of_gw + 1,
        "caller_declared_completion": {
            "status": completion_status,
            "scope": "all supplied gameweek snapshots through as_of_gw",
            "independent_source_verification": False,
        },
        "hashes": {
            "artifact_sha256": artifact_sha256,
            "input_sha256": input_sha256,
            "config_sha256": protocol.sha256(),
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "predictions_sha256": predictions_sha256,
        },
        "artifact": {
            "file": Path(artifact_path).name,
            "artifact_format": ARTIFACT_FORMAT,
            "season_id": model.season_id,
            "training_rows": model.training_rows,
            "trained_through_target_gw": model.trained_through_target_gw,
        },
        "inputs": {
            "files": input_files,
            "file_count": len(input_files),
            "aggregate_sha256": input_sha256,
            "hash_rule": (
                "sha256 of gameweek, NUL, file name, NUL, byte count, NUL, "
                "file sha256, LF in gameweek order"
            ),
        },
        "config": {
            "effective": protocol.to_dict(),
            "sha256": protocol.sha256(),
        },
        "feature_schema": {
            "version": FEATURE_SCHEMA_VERSION,
            "sha256": FEATURE_SCHEMA_SHA256,
        },
        "predictions": {
            "file": Path(predictions_path).name,
            "prediction_format": PREDICTION_FORMAT,
            "rows": prediction_rows,
            "sha256": predictions_sha256,
            "ranking_rule": "predicted_points descending, then player_id ascending",
        },
    }
