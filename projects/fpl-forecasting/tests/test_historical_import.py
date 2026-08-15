from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fpl_forecasting.data import read_gameweek
from fpl_forecasting.errors import DataContractError
from fpl_forecasting.historical import import_historical_gameweeks

FIELDS = (
    "element",
    "position",
    "round",
    "total_points",
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
)


def _row(player_id: int, position: str, gameweek: int, points: float) -> dict[str, object]:
    return {
        "element": player_id,
        "position": position,
        "round": gameweek,
        "total_points": points,
        "minutes": 45,
        "starts": 1,
        "ict_index": 2.5,
        "influence": 3.0,
        "creativity": 4.0,
        "threat": 5.0,
        "goals_scored": 0,
        "assists": 1,
        "clean_sheets": 0,
        "saves": 0,
        "bonus": 1,
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_historical_import_aggregates_fixtures_without_filling_absence(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    _write(
        source / "gw1.csv",
        [
            _row(11, "MID", 1, 2),
            _row(11, "MID", 1, 3),
            _row(12, "DEF", 1, 1),
            _row(900, "AM", 1, 10),
        ],
    )
    _write(source / "gw2.csv", [_row(11, "MID", 2, 4)])

    result = import_historical_gameweeks(
        source,
        output,
        season="2024-25",
        source_revision="a" * 40,
        gameweek_start=1,
        gameweek_end=2,
    )

    first = read_gameweek(output / "gameweek-01.json")
    second = read_gameweek(output / "gameweek-02.json")
    assert set(first.players) == {11, 12}
    assert set(second.players) == {11}
    assert first.players[11].total_points == 5
    assert first.players[11].minutes == 90
    assert result.manifest["source_file_count"] == 2
    assert result.manifest["season"] == "2024-25"
    assert "season_id" not in result.manifest
    audits = result.manifest["source_files"]
    assert isinstance(audits, list)
    assert audits[0]["duplicate_fixture_rows"] == 1
    assert audits[0]["excluded_non_player_rows"] == 1


def test_historical_import_rejects_unpinned_or_mismatched_input(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write(source / "gw1.csv", [_row(11, "MID", 2, 2)])

    with pytest.raises(DataContractError, match="full 40-character"):
        import_historical_gameweeks(
            source,
            tmp_path / "output",
            season="2024-25",
            source_revision="main",
            gameweek_start=1,
            gameweek_end=1,
        )

    with pytest.raises(DataContractError, match="source round is 2"):
        import_historical_gameweeks(
            source,
            tmp_path / "output",
            season="2024-25",
            source_revision="b" * 40,
            gameweek_start=1,
            gameweek_end=1,
        )


def test_historical_import_output_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write(source / "gw1.csv", [_row(12, "DEF", 1, 1), _row(11, "MID", 1, 2)])

    first = import_historical_gameweeks(
        source,
        tmp_path / "first",
        season="2024-25",
        source_revision="c" * 40,
        gameweek_start=1,
        gameweek_end=1,
    )
    second = import_historical_gameweeks(
        source,
        tmp_path / "second",
        season="2024-25",
        source_revision="c" * 40,
        gameweek_start=1,
        gameweek_end=1,
    )

    assert first.manifest == second.manifest
    first_payload = json.loads(first.paths[0].read_text(encoding="utf-8"))
    assert first_payload["snapshot_format"] == "fpl-completed-gameweek-v1"
    assert first_payload["season_id"] == "2024-25"
    assert first_payload["gameweek"] == 1
    assert [record["id"] for record in first_payload["elements"]] == [11, 12]
    assert first.paths[0].read_bytes() == second.paths[0].read_bytes()
