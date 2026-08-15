"""Read and validate completed FPL live-gameweek snapshots."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .contracts import (
    object_without_duplicate_keys,
    reject_nonfinite_json_constant,
    validate_season_id,
)
from .errors import (
    DataContractError,
    DuplicateGameweekError,
    EmptyGameweekError,
    NonConsecutiveGameweeksError,
)

_GAMEWEEK_PATTERN = re.compile(r"gameweek-(\d+)(?:-|\.)")
SNAPSHOT_FORMAT = "fpl-completed-gameweek-v1"
_SNAPSHOT_FIELDS = {"snapshot_format", "season_id", "gameweek", "elements"}
_OPTIONAL_STATS: dict[str, float] = {
    "starts": 0.0,
    "ict_index": 0.0,
    "influence": 0.0,
    "creativity": 0.0,
    "threat": 0.0,
    "goals_scored": 0.0,
    "assists": 0.0,
    "clean_sheets": 0.0,
    "saves": 0.0,
    "bonus": 0.0,
}


@dataclass(frozen=True)
class PlayerWeek:
    """One player's observed statistics for one completed gameweek."""

    player_id: int
    total_points: float
    minutes: float
    starts: float
    ict_index: float
    influence: float
    creativity: float
    threat: float
    goals_scored: float
    assists: float
    clean_sheets: float
    saves: float
    bonus: float


@dataclass(frozen=True)
class GameweekSnapshot:
    """Validated player records for a single completed gameweek."""

    gameweek: int
    season_id: str
    players: Mapping[int, PlayerWeek]
    source: Path


@dataclass(frozen=True)
class SequenceSummary:
    """Compact validation summary suitable for CLI output."""

    season_id: str
    gameweek_start: int
    gameweek_end: int
    gameweeks: int
    minimum_players: int
    maximum_players: int
    minimum_adjacent_coverage: float

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "season_id": self.season_id,
            "gameweek_start": self.gameweek_start,
            "gameweek_end": self.gameweek_end,
            "gameweeks": self.gameweeks,
            "minimum_players": self.minimum_players,
            "maximum_players": self.maximum_players,
            "minimum_adjacent_coverage": self.minimum_adjacent_coverage,
        }


def gameweek_from_filename(path: str | Path) -> int:
    """Extract a positive gameweek number from a snapshot filename."""

    name = Path(path).name
    match = _GAMEWEEK_PATTERN.search(name)
    if match is None:
        raise DataContractError(f"Snapshot filename must contain 'gameweek-<number>': {name}")
    gameweek = int(match.group(1))
    if gameweek < 1:
        raise DataContractError(f"Gameweek must be positive: {name}")
    return gameweek


def _finite_number(value: Any, *, field: str, context: str) -> float:
    if isinstance(value, bool):
        raise DataContractError(f"{context}: {field} cannot be boolean")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DataContractError(f"{context}: {field} must be numeric") from exc
    if not math.isfinite(number):
        raise DataContractError(f"{context}: {field} must be finite")
    return number


def _parse_player(record: Any, *, gameweek: int, index: int) -> PlayerWeek:
    context = f"gameweek {gameweek}, player record {index}"
    if not isinstance(record, dict):
        raise DataContractError(f"{context}: record must be a JSON object")
    if "id" not in record:
        raise DataContractError(f"{context}: missing player id")
    player_id_number = _finite_number(record["id"], field="id", context=context)
    player_id = int(player_id_number)
    if player_id_number != player_id or player_id < 1:
        raise DataContractError(f"{context}: id must be a positive integer")

    stats = record.get("stats")
    if not isinstance(stats, dict):
        raise DataContractError(f"{context}: stats must be a JSON object")
    for required in ("total_points", "minutes"):
        if required not in stats:
            raise DataContractError(f"{context}: missing required stat {required}")

    values = {
        "total_points": _finite_number(
            stats["total_points"], field="total_points", context=context
        ),
        "minutes": _finite_number(stats["minutes"], field="minutes", context=context),
    }
    for field, default in _OPTIONAL_STATS.items():
        raw = stats.get(field, default)
        if raw is None or raw == "":
            raw = default
        values[field] = _finite_number(raw, field=field, context=context)

    for non_negative in (
        "minutes",
        "starts",
        "ict_index",
        "influence",
        "creativity",
        "threat",
        "goals_scored",
        "assists",
        "clean_sheets",
        "saves",
        "bonus",
    ):
        if values[non_negative] < 0:
            raise DataContractError(f"{context}: {non_negative} cannot be negative")

    return PlayerWeek(player_id=player_id, **values)


def completed_snapshot_payload(
    *,
    season_id: str,
    gameweek: int,
    elements: list[Any],
) -> dict[str, Any]:
    """Wrap raw completed-gameweek elements in the canonical season boundary."""

    season = validate_season_id(season_id)
    if isinstance(gameweek, bool) or not isinstance(gameweek, Integral) or gameweek < 1:
        raise DataContractError("snapshot gameweek must be a positive integer")
    if not isinstance(elements, list):
        raise DataContractError("snapshot elements must be a list")
    return {
        "snapshot_format": SNAPSHOT_FORMAT,
        "season_id": season,
        "gameweek": int(gameweek),
        "elements": elements,
    }


def read_gameweek(path: str | Path) -> GameweekSnapshot:
    """Read one snapshot and reject missing, malformed, or empty player data."""

    source = Path(path)
    gameweek = gameweek_from_filename(source)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=object_without_duplicate_keys,
            parse_constant=reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise DataContractError(f"Invalid JSON in {source.name}: {exc.msg}") from exc
    except ValueError as exc:
        raise DataContractError(f"Invalid numeric JSON value in {source.name}") from exc
    if not isinstance(payload, dict):
        raise DataContractError(f"{source.name}: top-level value must be a JSON object")
    if set(payload) != _SNAPSHOT_FIELDS:
        missing = sorted(_SNAPSHOT_FIELDS - set(payload))
        unexpected = sorted(set(payload) - _SNAPSHOT_FIELDS)
        raise DataContractError(
            f"{source.name}: snapshot fields differ; missing={missing}, unexpected={unexpected}"
        )
    if payload["snapshot_format"] != SNAPSHOT_FORMAT:
        raise DataContractError(f"{source.name}: unsupported snapshot_format")
    season_id = validate_season_id(payload["season_id"])
    payload_gameweek = payload["gameweek"]
    if (
        isinstance(payload_gameweek, bool)
        or not isinstance(payload_gameweek, Integral)
        or payload_gameweek < 1
    ):
        raise DataContractError(f"{source.name}: gameweek must be a positive integer")
    if int(payload_gameweek) != gameweek:
        raise DataContractError(
            f"{source.name}: payload gameweek {payload_gameweek} does not match filename"
        )
    elements = payload.get("elements")
    if not isinstance(elements, list):
        raise DataContractError(f"{source.name}: 'elements' must be a list")
    if not elements:
        raise EmptyGameweekError(f"{source.name}: empty gameweek cannot be used as completed data")

    players: dict[int, PlayerWeek] = {}
    for index, record in enumerate(elements):
        player = _parse_player(record, gameweek=gameweek, index=index)
        if player.player_id in players:
            raise DataContractError(f"{source.name}: duplicate player id {player.player_id}")
        players[player.player_id] = player
    return GameweekSnapshot(
        gameweek=gameweek,
        season_id=season_id,
        players=players,
        source=source,
    )


def validate_sequence(
    snapshots: Iterable[GameweekSnapshot],
    *,
    minimum_adjacent_coverage: float = 0.90,
    expected_season_id: str | None = None,
) -> SequenceSummary:
    """Validate uniqueness, continuity, and adjacent player coverage."""

    ordered = sorted(snapshots, key=lambda item: item.gameweek)
    if len(ordered) < 2:
        raise DataContractError("At least two completed gameweeks are required")
    try:
        coverage = float(minimum_adjacent_coverage)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DataContractError("minimum_adjacent_coverage must be in (0, 1]") from exc
    if (
        isinstance(minimum_adjacent_coverage, bool)
        or not isinstance(minimum_adjacent_coverage, Real)
        or not math.isfinite(coverage)
        or not 0 < coverage <= 1
    ):
        raise DataContractError("minimum_adjacent_coverage must be in (0, 1]")

    expected_season = (
        validate_season_id(expected_season_id) if expected_season_id is not None else None
    )
    seasons: set[str] = set()
    for snapshot in ordered:
        seasons.add(validate_season_id(snapshot.season_id))
    if len(seasons) != 1:
        raise DataContractError("Snapshot sequence must contain exactly one season_id")
    season_id = next(iter(seasons))
    if expected_season is not None and season_id != expected_season:
        raise DataContractError(
            f"Snapshot season mismatch: expected {expected_season}, found {season_id}"
        )

    seen: set[int] = set()
    for snapshot in ordered:
        if snapshot.gameweek in seen:
            raise DuplicateGameweekError(f"Duplicate gameweek {snapshot.gameweek} is not allowed")
        seen.add(snapshot.gameweek)
        if not snapshot.players:
            raise EmptyGameweekError(f"Gameweek {snapshot.gameweek} has no player records")

    coverages: list[float] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.gameweek != previous.gameweek + 1:
            raise NonConsecutiveGameweeksError(
                f"Expected gameweek {previous.gameweek + 1}, found {current.gameweek}"
            )
        overlap = len(set(previous.players) & set(current.players))
        coverage = overlap / max(len(previous.players), len(current.players))
        coverages.append(coverage)
        if coverage < minimum_adjacent_coverage:
            raise DataContractError(
                "Adjacent player coverage is too low between gameweeks "
                f"{previous.gameweek} and {current.gameweek}: {coverage:.3f}"
            )

    counts = [len(snapshot.players) for snapshot in ordered]
    return SequenceSummary(
        season_id=season_id,
        gameweek_start=ordered[0].gameweek,
        gameweek_end=ordered[-1].gameweek,
        gameweeks=len(ordered),
        minimum_players=min(counts),
        maximum_players=max(counts),
        minimum_adjacent_coverage=min(coverages),
    )


def load_gameweeks(
    directory: str | Path,
    *,
    minimum_adjacent_coverage: float = 0.90,
    expected_season_id: str | None = None,
) -> list[GameweekSnapshot]:
    """Load one unambiguous, consecutive sequence from a directory."""

    base = Path(directory)
    if not base.is_dir():
        raise DataContractError(f"Gameweek directory does not exist: {base}")
    paths = sorted(base.glob("gameweek-*.json"))
    if not paths:
        raise DataContractError(f"No gameweek JSON files found in {base}")

    by_gameweek: dict[int, Path] = {}
    for path in paths:
        gameweek = gameweek_from_filename(path)
        if gameweek in by_gameweek:
            first = by_gameweek[gameweek].name
            raise DuplicateGameweekError(
                f"Gameweek {gameweek} has multiple files: {first}, {path.name}"
            )
        by_gameweek[gameweek] = path

    snapshots = [read_gameweek(by_gameweek[gw]) for gw in sorted(by_gameweek)]
    validate_sequence(
        snapshots,
        minimum_adjacent_coverage=minimum_adjacent_coverage,
        expected_season_id=expected_season_id,
    )
    return snapshots
