"""Configuration values for data checks, evaluation, and regression."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .contracts import (
    canonical_json_sha256,
    object_without_duplicate_keys,
    reject_nonfinite_json_constant,
)
from .errors import DataContractError


def _finite_real(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DataContractError(f"{field} must be numeric")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DataContractError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise DataContractError(f"{field} must be finite")
    return number


@dataclass(frozen=True)
class ProtocolConfig:
    """Small, explicit protocol shared by the CLI and Python API."""

    minimum_train_gameweeks: int = 4
    test_gameweeks_per_fold: int = 1
    split_step: int = 1
    minimum_adjacent_player_coverage: float = 0.90
    ranking_top_k: int = 10
    ridge_alpha: float = 4.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        integer_fields = {
            "minimum_train_gameweeks": self.minimum_train_gameweeks,
            "test_gameweeks_per_fold": self.test_gameweeks_per_fold,
            "split_step": self.split_step,
            "ranking_top_k": self.ranking_top_k,
            "random_seed": self.random_seed,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise DataContractError(f"{name} must be an integer")
        if self.minimum_train_gameweeks < 2:
            raise DataContractError("minimum_train_gameweeks must be at least 2")
        if self.test_gameweeks_per_fold < 1:
            raise DataContractError("test_gameweeks_per_fold must be positive")
        if self.split_step < 1:
            raise DataContractError("split_step must be positive")
        if self.split_step < self.test_gameweeks_per_fold:
            raise DataContractError("split_step must be at least test_gameweeks_per_fold")
        coverage = _finite_real(
            self.minimum_adjacent_player_coverage,
            "minimum_adjacent_player_coverage",
        )
        if not 0 < coverage <= 1:
            raise DataContractError(
                "minimum_adjacent_player_coverage must be in the interval (0, 1]"
            )
        if self.ranking_top_k < 1:
            raise DataContractError("ranking_top_k must be positive")
        alpha = _finite_real(self.ridge_alpha, "ridge_alpha")
        if alpha < 0:
            raise DataContractError("ridge_alpha must be finite and non-negative")
        if self.random_seed < 0:
            raise DataContractError("random_seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        """Hash the validated effective configuration, independent of input key order."""

        return canonical_json_sha256(self.to_dict())

    @classmethod
    def from_json(cls, path: str | Path) -> ProtocolConfig:
        try:
            payload = json.loads(
                Path(path).read_text(encoding="utf-8"),
                object_pairs_hook=object_without_duplicate_keys,
                parse_constant=reject_nonfinite_json_constant,
            )
        except json.JSONDecodeError as exc:
            raise DataContractError(f"Invalid configuration JSON: {exc.msg}") from exc
        except ValueError as exc:
            raise DataContractError("Invalid numeric value in configuration JSON") from exc
        if not isinstance(payload, dict):
            raise DataContractError("Configuration must be a JSON object")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise DataContractError("Unknown configuration keys: " + ", ".join(unknown))
        return cls(**payload)
