"""Chronological next-gameweek forecasting for Fantasy Premier League."""

from .config import ProtocolConfig
from .contracts import validate_season_id
from .data import SNAPSHOT_FORMAT, GameweekSnapshot, completed_snapshot_payload, load_gameweeks
from .evaluation import EvaluationResult, evaluate_models
from .features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    build_forecast_frame,
    build_training_frame,
)

__all__ = [
    "EvaluationResult",
    "FEATURE_COLUMNS",
    "FEATURE_SCHEMA_SHA256",
    "FEATURE_SCHEMA_VERSION",
    "GameweekSnapshot",
    "ProtocolConfig",
    "SNAPSHOT_FORMAT",
    "build_forecast_frame",
    "build_training_frame",
    "completed_snapshot_payload",
    "evaluate_models",
    "load_gameweeks",
    "validate_season_id",
]

__version__ = "2.0.0"
