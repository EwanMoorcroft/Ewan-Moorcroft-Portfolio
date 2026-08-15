"""Verify generated forecast evidence before one transactional DuckDB write."""

from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProtocolConfig
from .contracts import (
    canonical_json_bytes,
    canonical_json_sha256,
    object_without_duplicate_keys,
    reject_nonfinite_json_constant,
    sha256_bytes,
    validate_season_id,
)
from .data import load_gameweeks
from .errors import DataContractError
from .features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    IDENTIFIER_COLUMNS,
    assert_forecast_frame,
    build_forecast_frame,
)
from .models import ARTIFACT_FORMAT, FittedRidge
from .operational import (
    COMPLETION_STATUS,
    MANIFEST_FORMAT,
    operational_manifest,
    ranked_prediction_payload,
    snapshot_input_audit,
    validate_deployment_model,
    validate_expected_as_of,
)
from .storage import (
    MAX_ARTIFACT_BYTES,
    MAX_MANIFEST_FILE_BYTES,
    MAX_PREDICTIONS_FILE_BYTES,
    BatchRequest,
    BatchWriteResult,
    ModelRegistration,
    OperationalStore,
    StoredPrediction,
    read_bounded_bytes,
)


def forecast_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash the validated ordered scoring frame with explicit scalar types."""

    assert_forecast_frame(frame)
    rows: list[dict[str, int | float]] = []
    for values in frame.itertuples(index=False, name=None):
        raw = dict(zip(frame.columns, values, strict=True))
        row: dict[str, int | float] = {column: int(raw[column]) for column in IDENTIFIER_COLUMNS}
        row.update({column: float(raw[column]) for column in FEATURE_COLUMNS})
        rows.append(row)
    return canonical_json_sha256(
        {
            "contract": "fpl-forecast-frame-v1",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "columns": [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS],
            "rows": rows,
        }
    )


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_without_duplicate_keys,
            parse_constant=reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise DataContractError(f"{label} root must be a JSON object")
    return payload


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DataContractError(f"Manifest {field} must be a JSON object")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise DataContractError(f"Manifest {field} must be a positive integer")
    return int(value)


def _protocol_from_manifest(manifest: Mapping[str, Any]) -> ProtocolConfig:
    config = _mapping(manifest.get("config"), field="config")
    effective = _mapping(config.get("effective"), field="config.effective")
    expected_fields = set(ProtocolConfig.__dataclass_fields__)
    if set(effective) != expected_fields:
        raise DataContractError("Manifest effective config fields do not match this release")
    try:
        return ProtocolConfig(**effective)
    except TypeError as exc:
        raise DataContractError("Manifest effective config is invalid") from exc


def store_generated_batch(
    *,
    database_path: str | Path,
    gameweek_dir: str | Path,
    artifact_path: str | Path,
    predictions_path: str | Path,
    manifest_path: str | Path,
) -> BatchWriteResult:
    """Rebuild and authenticate a generated forecast before persisting it."""

    database = Path(database_path).resolve()
    artifact = Path(artifact_path).resolve()
    predictions = Path(predictions_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    protected = {artifact, predictions, manifest_file}
    protected.update(path.resolve() for path in Path(gameweek_dir).glob("gameweek-*.json"))
    if database in protected:
        raise DataContractError("Database path cannot overwrite forecast evidence or inputs")

    manifest_bytes = read_bounded_bytes(
        manifest_file,
        max_bytes=MAX_MANIFEST_FILE_BYTES,
        label="manifest",
    )
    manifest = _strict_json_object(manifest_bytes, label="Manifest")
    if manifest.get("manifest_format") != MANIFEST_FORMAT:
        raise DataContractError("Manifest format is unsupported")

    season_id = validate_season_id(manifest.get("season_id"))
    as_of_gw = _positive_integer(manifest.get("as_of_gw"), field="as_of_gw")
    target_gw = _positive_integer(manifest.get("target_gw"), field="target_gw")
    if target_gw != as_of_gw + 1:
        raise DataContractError("Manifest target_gw must be exactly one after as_of_gw")
    completion = _mapping(
        manifest.get("caller_declared_completion"),
        field="caller_declared_completion",
    )
    if completion.get("status") != COMPLETION_STATUS:
        raise DataContractError("Manifest does not declare completed inputs")

    protocol = _protocol_from_manifest(manifest)
    snapshots = load_gameweeks(
        gameweek_dir,
        minimum_adjacent_coverage=protocol.minimum_adjacent_player_coverage,
        expected_season_id=season_id,
    )
    validate_expected_as_of(snapshots, as_of_gw)
    frame = build_forecast_frame(
        snapshots,
        minimum_adjacent_coverage=protocol.minimum_adjacent_player_coverage,
    )
    artifact_bytes = read_bounded_bytes(
        artifact,
        max_bytes=MAX_ARTIFACT_BYTES,
        label="model artifact",
    )
    model = FittedRidge.from_json_bytes(artifact_bytes)
    validate_deployment_model(
        model,
        season_id=season_id,
        config_sha256=protocol.sha256(),
        as_of_gw=as_of_gw,
    )

    expected_predictions = ranked_prediction_payload(
        frame,
        model,
        season_id=season_id,
        completion_status=COMPLETION_STATUS,
    )
    expected_prediction_bytes = canonical_json_bytes(expected_predictions)
    actual_prediction_bytes = read_bounded_bytes(
        predictions,
        max_bytes=MAX_PREDICTIONS_FILE_BYTES,
        label="predictions",
    )
    if actual_prediction_bytes != expected_prediction_bytes:
        raise DataContractError(
            "Predictions do not exactly match the configured artifact and snapshots"
        )

    input_files, input_sha256 = snapshot_input_audit(snapshots)
    expected_manifest = operational_manifest(
        season_id=season_id,
        as_of_gw=as_of_gw,
        completion_status=COMPLETION_STATUS,
        protocol=protocol,
        model=model,
        artifact_path=artifact,
        artifact_sha256=sha256_bytes(artifact_bytes),
        input_files=input_files,
        input_sha256=input_sha256,
        predictions_path=predictions,
        predictions_sha256=sha256_bytes(actual_prediction_bytes),
        prediction_rows=len(frame),
    )
    if manifest_bytes != canonical_json_bytes(expected_manifest):
        raise DataContractError(
            "Manifest does not exactly authenticate the configured forecast evidence"
        )

    registration = ModelRegistration(
        season_id=season_id,
        artifact_format=ARTIFACT_FORMAT,
        artifact_sha256=sha256_bytes(artifact_bytes),
        feature_schema_version=model.feature_schema_version,
        feature_schema_sha256=model.feature_schema_sha256,
        protocol_config_sha256=protocol.sha256(),
        trained_through_target_gw=model.trained_through_target_gw,
        training_rows=model.training_rows,
    )
    request = BatchRequest(
        model_id=registration.model_id,
        season_id=season_id,
        as_of_gw=as_of_gw,
        target_gw=target_gw,
        source_snapshot_sha256=input_sha256,
        forecast_frame_sha256=forecast_frame_sha256(frame),
        feature_schema_sha256=model.feature_schema_sha256,
        protocol_config_sha256=protocol.sha256(),
        predictions_file_sha256=sha256_bytes(actual_prediction_bytes),
        manifest_file_sha256=sha256_bytes(manifest_bytes),
    )
    stored_predictions = [
        StoredPrediction(
            player_id=int(row["player_id"]),
            prediction=float(row["predicted_points"]),
            rank=int(row["rank"]),
        )
        for row in expected_predictions["predictions"]
    ]
    store = OperationalStore(database)
    with store.writer() as writer:
        writer.register_model(registration)
        return writer.persist_batch(request, stored_predictions)
