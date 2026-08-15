"""Build strictly as-of features for next-gameweek targets."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .data import GameweekSnapshot, PlayerWeek, validate_sequence
from .errors import DataContractError, LeakageError

FEATURE_COLUMNS: tuple[str, ...] = (
    "lag_1_points",
    "lag_1_minutes",
    "lag_1_started",
    "lag_1_ict_index",
    "lag_1_influence",
    "lag_1_creativity",
    "lag_1_threat",
    "rolling_3_points_mean",
    "rolling_3_minutes_mean",
    "rolling_3_starts_rate",
    "rolling_3_ict_mean",
    "rolling_3_goals_mean",
    "rolling_3_assists_mean",
    "rolling_5_points_mean",
    "season_points_sum",
    "season_minutes_sum",
    "season_appearances",
    "season_points_per_appearance",
)

TARGET_COLUMN = "target_next_gw_points"
IDENTIFIER_COLUMNS: tuple[str, ...] = ("player_id", "as_of_gw", "target_gw")
_FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "target",
    "future",
    "next_gw",
    "post_deadline",
)


def _mean(records: Sequence[PlayerWeek], field: str) -> float:
    return float(sum(getattr(record, field) for record in records) / len(records))


def _row_features(history: Sequence[PlayerWeek]) -> dict[str, float]:
    if not history:
        raise DataContractError("Feature history cannot be empty")
    latest = history[-1]
    last_three = history[-3:]
    last_five = history[-5:]
    season_points = float(sum(record.total_points for record in history))
    season_minutes = float(sum(record.minutes for record in history))
    appearances = int(sum(record.minutes > 0 for record in history))

    return {
        "lag_1_points": latest.total_points,
        "lag_1_minutes": latest.minutes,
        "lag_1_started": float(latest.starts > 0),
        "lag_1_ict_index": latest.ict_index,
        "lag_1_influence": latest.influence,
        "lag_1_creativity": latest.creativity,
        "lag_1_threat": latest.threat,
        "rolling_3_points_mean": _mean(last_three, "total_points"),
        "rolling_3_minutes_mean": _mean(last_three, "minutes"),
        "rolling_3_starts_rate": float(
            sum(record.starts > 0 for record in last_three) / len(last_three)
        ),
        "rolling_3_ict_mean": _mean(last_three, "ict_index"),
        "rolling_3_goals_mean": _mean(last_three, "goals_scored"),
        "rolling_3_assists_mean": _mean(last_three, "assists"),
        "rolling_5_points_mean": _mean(last_five, "total_points"),
        "season_points_sum": season_points,
        "season_minutes_sum": season_minutes,
        "season_appearances": float(appearances),
        "season_points_per_appearance": season_points / max(appearances, 1),
    }


def build_training_frame(
    snapshots: Iterable[GameweekSnapshot],
    *,
    minimum_adjacent_coverage: float = 0.90,
) -> pd.DataFrame:
    """Create rows whose features stop at t and whose target comes from t+1.

    The final supplied gameweek is used only as a target. It never creates an
    unlabeled row. Players absent from either side of an adjacent pair are not
    assigned an invented zero target.
    """

    ordered = sorted(snapshots, key=lambda item: item.gameweek)
    validate_sequence(ordered, minimum_adjacent_coverage=minimum_adjacent_coverage)
    history_by_player: dict[int, list[PlayerWeek]] = {}
    rows: list[dict[str, float | int]] = []

    for current, following in zip(ordered, ordered[1:], strict=False):
        for player_id, observed in current.players.items():
            history_by_player.setdefault(player_id, []).append(observed)

        eligible = sorted(set(current.players) & set(following.players))
        for player_id in eligible:
            history = history_by_player[player_id]
            row: dict[str, float | int] = {
                "player_id": player_id,
                "as_of_gw": current.gameweek,
                "target_gw": following.gameweek,
                TARGET_COLUMN: following.players[player_id].total_points,
            }
            row.update(_row_features(history))
            rows.append(row)

    if not rows:
        raise DataContractError("No completed next-gameweek targets could be built")
    frame = pd.DataFrame.from_records(rows)
    ordered_columns = [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN]
    frame = frame.loc[:, ordered_columns].sort_values(["target_gw", "player_id"], kind="stable")
    frame = frame.reset_index(drop=True)
    assert_leak_free_frame(frame)
    return frame


def assert_leak_free_frame(frame: pd.DataFrame) -> None:
    """Enforce the feature allow-list and the one-step as-of relationship."""

    required = {*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LeakageError("Frame is missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise LeakageError("Evaluation frame cannot be empty")

    identifiers = frame.loc[:, list(IDENTIFIER_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if identifiers.isna().any().any():
        raise LeakageError("Player and gameweek boundary columns must be numeric")
    identifier_values = identifiers.to_numpy(dtype=float)
    if not np.isfinite(identifier_values).all():
        raise LeakageError("Player and gameweek boundary columns must be finite")
    if not np.equal(identifier_values, np.floor(identifier_values)).all():
        raise LeakageError("Player and gameweek boundary columns must be integers")
    if not (identifier_values > 0).all():
        raise LeakageError("Player and gameweek boundary columns must be positive")

    as_of = identifiers["as_of_gw"]
    target_gw = identifiers["target_gw"]
    if not bool((target_gw == as_of + 1).all()):
        raise LeakageError("Every target must be exactly one gameweek after its features")
    if bool(identifiers.duplicated(["player_id", "target_gw"]).any()):
        raise LeakageError("Each player and target gameweek pair must be unique")

    exempt = {*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN}
    for column in frame.columns:
        if column in exempt:
            continue
        lowered = str(column).lower()
        if any(token in lowered for token in _FORBIDDEN_FEATURE_TOKENS):
            raise LeakageError(f"Unexpected future-facing column: {column}")

    for feature in FEATURE_COLUMNS:
        lowered = feature.lower()
        if any(token in lowered for token in _FORBIDDEN_FEATURE_TOKENS):
            raise LeakageError(f"Feature name crosses the as-of boundary: {feature}")

    numeric = frame.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise LeakageError("Non-numeric or missing values in: " + ", ".join(bad))
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise LeakageError("Feature and target values must be finite")

    if any(not math.isfinite(float(value)) for value in frame[TARGET_COLUMN]):
        raise LeakageError("Targets must be finite")


def feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return only the approved model inputs and the next-gameweek target."""

    assert_leak_free_frame(frame)
    x = frame.loc[:, list(FEATURE_COLUMNS)].astype(float).copy()
    y = frame[TARGET_COLUMN].astype(float).copy()
    if set(x.columns) != set(FEATURE_COLUMNS):
        raise LeakageError("Unexpected feature selection")
    return x, y
