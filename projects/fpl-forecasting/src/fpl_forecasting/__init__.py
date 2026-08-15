"""Chronological next-gameweek forecasting for Fantasy Premier League."""

from .config import ProtocolConfig
from .data import GameweekSnapshot, load_gameweeks
from .evaluation import EvaluationResult, evaluate_models
from .features import FEATURE_COLUMNS, build_training_frame

__all__ = [
    "EvaluationResult",
    "FEATURE_COLUMNS",
    "GameweekSnapshot",
    "ProtocolConfig",
    "build_training_frame",
    "evaluate_models",
    "load_gameweeks",
]

__version__ = "1.0.0"
