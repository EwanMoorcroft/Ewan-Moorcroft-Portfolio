"""Ridge regression and a portable, non-pickle artifact format."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .contracts import canonical_json_bytes, validate_season_id
from .errors import ArtifactError, DataContractError, LeakageError
from .features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    feature_matrix,
)

ARTIFACT_FORMAT = "fpl-ridge-json-v2"
_ARTIFACT_FIELDS = {
    "artifact_format",
    "season_id",
    "feature_schema_version",
    "feature_schema_sha256",
    "protocol_config_sha256",
    "feature_names",
    "feature_means",
    "feature_scales",
    "coefficients",
    "intercept",
    "alpha",
    "training_rows",
    "trained_through_target_gw",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _artifact_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ArtifactError(f"Artifact {field} must be numeric")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ArtifactError(f"Artifact {field} must be finite") from exc
    if not math.isfinite(number):
        raise ArtifactError(f"Artifact {field} must be finite")
    return number


def _artifact_vector(payload: Mapping[str, Any], field: str) -> tuple[float, ...]:
    raw = payload[field]
    if not isinstance(raw, (list, tuple)):
        raise ArtifactError(f"Artifact {field} must be a numeric array")
    return tuple(_artifact_number(value, field) for value in raw)


def _artifact_positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ArtifactError(f"Artifact {field} must be a positive integer")
    return int(value)


def _artifact_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ArtifactError(f"Artifact {field} must be a lowercase SHA-256 digest")
    return value


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ArtifactError(f"Artifact contains duplicate field: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite_constant(value: str) -> None:
    raise ArtifactError(f"Artifact number must be finite: {value}")


@dataclass(frozen=True)
class FittedRidge:
    """Standardized ridge parameters that can be evaluated without pickle."""

    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    alpha: float
    training_rows: int
    trained_through_target_gw: int
    season_id: str | None = None
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    feature_schema_sha256: str = FEATURE_SCHEMA_SHA256
    protocol_config_sha256: str | None = None

    def predict(self, values: pd.DataFrame) -> np.ndarray:
        if (
            self.feature_schema_version != FEATURE_SCHEMA_VERSION
            or self.feature_schema_sha256 != FEATURE_SCHEMA_SHA256
        ):
            raise ArtifactError("Model feature schema does not match this release")
        if not values.columns.is_unique:
            raise LeakageError("Prediction frame columns must be unique")
        missing = sorted(set(self.feature_names) - set(values.columns))
        if missing:
            raise LeakageError("Prediction frame is missing: " + ", ".join(missing))
        for feature in self.feature_names:
            invalid = values[feature].map(
                lambda value: isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            )
            if bool(invalid.any()):
                raise LeakageError("Prediction features must be numeric and cannot be boolean")
        matrix = values.loc[:, list(self.feature_names)].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise LeakageError("Prediction features must be finite")
        means = np.asarray(self.feature_means, dtype=float)
        scales = np.asarray(self.feature_scales, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        standardized = (matrix - means) / scales
        predictions = standardized @ coefficients + self.intercept
        if not np.isfinite(predictions).all():
            raise ArtifactError("Model produced non-finite predictions")
        return predictions

    def to_dict(self) -> dict[str, Any]:
        if self.season_id is None or self.protocol_config_sha256 is None:
            raise ArtifactError("Deployment artifact requires season_id and protocol_config_sha256")
        try:
            season_id = validate_season_id(self.season_id)
        except DataContractError as exc:
            raise ArtifactError(str(exc)) from exc
        protocol_hash = _artifact_sha256(self.protocol_config_sha256, "protocol_config_sha256")
        if (
            self.feature_schema_version != FEATURE_SCHEMA_VERSION
            or self.feature_schema_sha256 != FEATURE_SCHEMA_SHA256
        ):
            raise ArtifactError("Model feature schema does not match this release")
        return {
            "artifact_format": ARTIFACT_FORMAT,
            "season_id": season_id,
            "feature_schema_version": self.feature_schema_version,
            "feature_schema_sha256": self.feature_schema_sha256,
            "protocol_config_sha256": protocol_hash,
            "feature_names": list(self.feature_names),
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "alpha": self.alpha,
            "training_rows": self.training_rows,
            "trained_through_target_gw": self.trained_through_target_gw,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        encoded = canonical_json_bytes(self.to_dict())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)
        return destination

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FittedRidge:
        if not isinstance(payload, Mapping):
            raise ArtifactError("Artifact root must be a JSON object")
        if payload.get("artifact_format") != ARTIFACT_FORMAT:
            raise ArtifactError("Unsupported artifact format")
        if set(payload) != _ARTIFACT_FIELDS:
            missing = sorted(_ARTIFACT_FIELDS - set(payload))
            unexpected = sorted(set(payload) - _ARTIFACT_FIELDS)
            raise ArtifactError(
                f"Artifact fields differ; missing={missing}, unexpected={unexpected}"
            )
        raw_feature_names = payload["feature_names"]
        if not isinstance(raw_feature_names, (list, tuple)) or any(
            not isinstance(name, str) for name in raw_feature_names
        ):
            raise ArtifactError("Artifact feature_names must be a string array")
        feature_names = tuple(raw_feature_names)
        feature_means = _artifact_vector(payload, "feature_means")
        feature_scales = _artifact_vector(payload, "feature_scales")
        coefficients = _artifact_vector(payload, "coefficients")
        intercept = _artifact_number(payload["intercept"], "intercept")
        alpha = _artifact_number(payload["alpha"], "alpha")
        training_rows = _artifact_positive_integer(payload["training_rows"], "training_rows")
        trained_through = _artifact_positive_integer(
            payload["trained_through_target_gw"], "trained_through_target_gw"
        )
        try:
            season_id = validate_season_id(payload["season_id"])
        except DataContractError as exc:
            raise ArtifactError(str(exc)) from exc
        if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ArtifactError("Artifact feature schema version does not match this release")
        schema_hash = _artifact_sha256(payload["feature_schema_sha256"], "feature_schema_sha256")
        if schema_hash != FEATURE_SCHEMA_SHA256:
            raise ArtifactError("Artifact feature schema hash does not match this release")
        protocol_hash = _artifact_sha256(
            payload["protocol_config_sha256"], "protocol_config_sha256"
        )

        expected = len(FEATURE_COLUMNS)
        lengths = {
            len(feature_names),
            len(feature_means),
            len(feature_scales),
            len(coefficients),
        }
        if lengths != {expected}:
            raise ArtifactError("Artifact feature vectors have incompatible lengths")
        if feature_names != FEATURE_COLUMNS:
            raise ArtifactError("Artifact feature contract does not match this release")
        if any(scale <= 0 for scale in feature_scales):
            raise ArtifactError("Artifact feature scales must be positive")
        if alpha < 0:
            raise ArtifactError("Artifact training metadata is invalid")

        return cls(
            feature_names=feature_names,
            feature_means=feature_means,
            feature_scales=feature_scales,
            coefficients=coefficients,
            intercept=intercept,
            alpha=alpha,
            training_rows=training_rows,
            trained_through_target_gw=trained_through,
            season_id=season_id,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_schema_sha256=schema_hash,
            protocol_config_sha256=protocol_hash,
        )

    @classmethod
    def load(cls, path: str | Path) -> FittedRidge:
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            raise ArtifactError("Could not read ridge artifact") from exc
        return cls.from_json_bytes(raw)

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> FittedRidge:
        """Load an artifact from the exact bytes that can also be hashed."""

        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_nonfinite_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactError("Could not read ridge artifact") from exc
        except ValueError as exc:
            raise ArtifactError("Artifact contains an invalid numeric value") from exc
        if not isinstance(payload, dict):
            raise ArtifactError("Artifact root must be a JSON object")
        return cls.from_dict(payload)


def fit_ridge(
    frame: pd.DataFrame,
    *,
    alpha: float,
    season_id: str | None = None,
    protocol_config_sha256: str | None = None,
) -> FittedRidge:
    """Fit standardized ridge regression using the approved feature contract."""

    validated_alpha = _artifact_number(alpha, "alpha")
    if validated_alpha < 0:
        raise ArtifactError("Ridge alpha cannot be negative")
    if (season_id is None) != (protocol_config_sha256 is None):
        raise ArtifactError("season_id and protocol_config_sha256 must be supplied together")
    validated_season: str | None = None
    validated_protocol_hash: str | None = None
    if season_id is not None and protocol_config_sha256 is not None:
        try:
            validated_season = validate_season_id(season_id)
        except DataContractError as exc:
            raise ArtifactError(str(exc)) from exc
        validated_protocol_hash = _artifact_sha256(protocol_config_sha256, "protocol_config_sha256")
    x, y = feature_matrix(frame)
    matrix = x.to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    standardized = (matrix - means) / scales

    estimator = Ridge(alpha=validated_alpha, fit_intercept=True)
    estimator.fit(standardized, y.to_numpy(dtype=float))
    return FittedRidge(
        feature_names=FEATURE_COLUMNS,
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in estimator.coef_),
        intercept=float(estimator.intercept_),
        alpha=validated_alpha,
        training_rows=len(frame),
        trained_through_target_gw=int(frame["target_gw"].max()),
        season_id=validated_season,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        protocol_config_sha256=validated_protocol_hash,
    )
