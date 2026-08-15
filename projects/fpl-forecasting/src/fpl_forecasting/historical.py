"""Convert a pinned historical FPL CSV snapshot into the live-gameweek contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .errors import DataContractError

SOURCE_REPOSITORY = "https://github.com/vaastav/Fantasy-Premier-League"
_PLAYER_POSITIONS = frozenset({"GK", "DEF", "MID", "FWD"})
_KNOWN_NON_PLAYER_POSITIONS = frozenset({"AM"})
_SOURCE_TO_STAT = {
    "total_points": "total_points",
    "minutes": "minutes",
    "starts": "starts",
    "ict_index": "ict_index",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "goals_scored": "goals_scored",
    "assists": "assists",
    "clean_sheets": "clean_sheets",
    "saves": "saves",
    "bonus": "bonus",
}
_REQUIRED_COLUMNS = frozenset({"element", "position", "round", *_SOURCE_TO_STAT})


@dataclass(frozen=True)
class HistoricalImportResult:
    """Paths and audit manifest produced by one deterministic import."""

    paths: tuple[Path, ...]
    manifest: dict[str, object]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _number(raw: str | None, *, field: str, gameweek: int, row: int) -> float:
    if raw is None or raw == "":
        raise DataContractError(f"GW{gameweek} row {row}: {field} is missing")
    try:
        value = float(raw)
    except ValueError as exc:
        raise DataContractError(f"GW{gameweek} row {row}: {field} must be numeric") from exc
    if not math.isfinite(value):
        raise DataContractError(f"GW{gameweek} row {row}: {field} must be finite")
    if field != "total_points" and value < 0:
        raise DataContractError(f"GW{gameweek} row {row}: {field} cannot be negative")
    return value


def _positive_integer(raw: str | None, *, field: str, gameweek: int, row: int) -> int:
    value = _number(raw, field=field, gameweek=gameweek, row=row)
    integer = int(value)
    if value != integer or integer < 1:
        raise DataContractError(f"GW{gameweek} row {row}: {field} must be a positive integer")
    return integer


def _read_source(path: Path, *, gameweek: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    columns = set(reader.fieldnames or ())
    missing = sorted(_REQUIRED_COLUMNS - columns)
    if missing:
        raise DataContractError(f"{path.name}: missing columns: {', '.join(missing)}")

    players: dict[int, dict[str, float]] = {}
    rows = 0
    player_rows = 0
    excluded_non_player_rows = 0
    for row_number, record in enumerate(reader, start=2):
        rows += 1
        position = str(record["position"]).strip()
        if position in _KNOWN_NON_PLAYER_POSITIONS:
            excluded_non_player_rows += 1
            continue
        if position not in _PLAYER_POSITIONS:
            raise DataContractError(
                f"GW{gameweek} row {row_number}: unsupported position {position!r}"
            )

        source_round = _positive_integer(
            record["round"], field="round", gameweek=gameweek, row=row_number
        )
        if source_round != gameweek:
            raise DataContractError(
                f"GW{gameweek} row {row_number}: source round is {source_round}"
            )
        player_id = _positive_integer(
            record["element"], field="element", gameweek=gameweek, row=row_number
        )
        totals = players.setdefault(player_id, {field: 0.0 for field in _SOURCE_TO_STAT})
        for source_field, target_field in _SOURCE_TO_STAT.items():
            totals[target_field] += _number(
                record[source_field],
                field=source_field,
                gameweek=gameweek,
                row=row_number,
            )
        player_rows += 1

    if not players:
        raise DataContractError(f"{path.name}: no player rows found")

    elements = [{"id": player_id, "stats": players[player_id]} for player_id in sorted(players)]
    audit = {
        "file": path.name,
        "bytes": len(raw),
        "sha256": _sha256(raw),
        "rows": rows,
        "player_rows": player_rows,
        "players": len(players),
        "duplicate_fixture_rows": player_rows - len(players),
        "excluded_non_player_rows": excluded_non_player_rows,
    }
    return elements, audit


def import_historical_gameweeks(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    season: str,
    source_revision: str,
    gameweek_start: int,
    gameweek_end: int,
) -> HistoricalImportResult:
    """Import an explicit historical interval without filling missing player records.

    Multiple fixture rows for the same player and event are summed to match the
    event-level live-gameweek shape. Rows with the `AM` position are excluded because
    the forecast target is player points. Missing player rows remain missing.
    """

    if not season.strip():
        raise DataContractError("season cannot be empty")
    revision = source_revision.strip().lower()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise DataContractError("source_revision must be a full 40-character Git commit SHA")
    if gameweek_start < 1 or gameweek_end < gameweek_start:
        raise DataContractError("gameweek interval is invalid")

    source = Path(source_dir)
    destination = Path(output_dir)
    if not source.is_dir():
        raise DataContractError(f"Historical source directory does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    source_files: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    for gameweek in range(gameweek_start, gameweek_end + 1):
        source_path = source / f"gw{gameweek}.csv"
        if not source_path.is_file():
            raise DataContractError(f"Historical source file is missing: {source_path.name}")
        elements, audit = _read_source(source_path, gameweek=gameweek)
        aggregate.update(f"{audit['file']}\0{audit['bytes']}\0{audit['sha256']}\n".encode("ascii"))
        output_path = destination / f"gameweek-{gameweek:02d}.json"
        output_path.write_text(
            json.dumps({"elements": elements}, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        paths.append(output_path)
        source_files.append(audit)

    manifest: dict[str, object] = {
        "manifest_format": "fpl-historical-import-v1",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": revision,
        "season": season,
        "gameweek_start": gameweek_start,
        "gameweek_end": gameweek_end,
        "source_file_count": len(source_files),
        "source_files_sha256": aggregate.hexdigest(),
        "source_files_hash_rule": "sha256 of file name, NUL, byte count, NUL, file sha256, LF in gameweek order",
        "conversion": {
            "player_id": "element",
            "player_positions": sorted(_PLAYER_POSITIONS),
            "excluded_positions": sorted(_KNOWN_NON_PLAYER_POSITIONS),
            "duplicate_fixture_rows": "sum each retained numeric statistic by element and round",
            "missing_players": "remain missing; no zero rows are created",
        },
        "source_files": source_files,
    }
    return HistoricalImportResult(paths=tuple(paths), manifest=manifest)
