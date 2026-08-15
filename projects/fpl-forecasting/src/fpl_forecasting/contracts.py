"""Canonical identifiers and deterministic hashing shared by operational paths."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .errors import DataContractError

_SEASON_ID_PATTERN = re.compile(r"^(?P<start>[0-9]{4})-(?P<end>[0-9]{2})$")


def validate_season_id(value: Any) -> str:
    """Return a canonical FPL season identifier such as ``2024-25``."""

    if not isinstance(value, str):
        raise DataContractError("season_id must be a string in YYYY-YY form")
    match = _SEASON_ID_PATTERN.fullmatch(value)
    if match is None:
        raise DataContractError("season_id must use canonical YYYY-YY form")
    start = int(match.group("start"))
    end = int(match.group("end"))
    if start < 1992 or end != (start + 1) % 100:
        raise DataContractError("season_id must identify consecutive FPL years")
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically and reject non-finite numbers."""

    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise DataContractError("Payload cannot be represented as canonical JSON") from exc
    return encoded.encode("utf-8") + b"\n"


def canonical_json_sha256(payload: Any) -> str:
    """Hash the canonical JSON representation used by this project."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest for raw bytes."""

    return hashlib.sha256(payload).hexdigest()


def write_canonical_json(path: str | Path, payload: Any) -> Path:
    """Write deterministic JSON only after the complete payload validates."""

    destination = Path(path)
    encoded = canonical_json_bytes(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)
    return destination


def object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys at every depth."""

    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise DataContractError(f"JSON object contains duplicate key: {key}")
        payload[key] = value
    return payload


def reject_nonfinite_json_constant(value: str) -> None:
    """Reject Python JSON decoder extensions such as NaN and Infinity."""

    raise DataContractError(f"JSON number must be finite: {value}")
