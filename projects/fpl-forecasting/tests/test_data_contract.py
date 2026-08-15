from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from fpl_forecasting.config import ProtocolConfig
from fpl_forecasting.contracts import validate_season_id
from fpl_forecasting.data import (
    SNAPSHOT_FORMAT,
    completed_snapshot_payload,
    load_gameweeks,
    read_gameweek,
    validate_sequence,
)
from fpl_forecasting.errors import (
    DataContractError,
    DuplicateGameweekError,
    EmptyGameweekError,
    NonConsecutiveGameweeksError,
)
from fpl_forecasting.synthetic import generate_synthetic_gameweeks


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_rejects_empty_gameweek(tmp_path: Path) -> None:
    path = tmp_path / "gameweek-1.json"
    _write(
        path,
        completed_snapshot_payload(season_id="2024-25", gameweek=1, elements=[]),
    )
    with pytest.raises(EmptyGameweekError, match="empty gameweek"):
        read_gameweek(path)


def test_rejects_duplicate_gameweek_files(tmp_path: Path) -> None:
    payload = generate_synthetic_gameweeks(gameweeks=2, players=4, seed=3)[1]
    _write(tmp_path / "gameweek-1-a.json", payload)
    _write(tmp_path / "gameweek-1-b.json", payload)
    with pytest.raises(DuplicateGameweekError, match="multiple files"):
        load_gameweeks(tmp_path)


def test_rejects_gap_in_sequence(tmp_path: Path) -> None:
    payloads = generate_synthetic_gameweeks(gameweeks=3, players=4, seed=3)
    _write(tmp_path / "gameweek-1.json", payloads[1])
    _write(tmp_path / "gameweek-3.json", payloads[3])
    with pytest.raises(NonConsecutiveGameweeksError, match="Expected gameweek 2"):
        load_gameweeks(tmp_path)


def test_rejects_low_adjacent_player_coverage(tmp_path: Path) -> None:
    payloads = generate_synthetic_gameweeks(gameweeks=2, players=10, seed=5)
    payloads[2]["elements"] = payloads[2]["elements"][:2]
    _write(tmp_path / "gameweek-1.json", payloads[1])
    _write(tmp_path / "gameweek-2.json", payloads[2])
    with pytest.raises(DataContractError, match="coverage is too low"):
        load_gameweeks(tmp_path, minimum_adjacent_coverage=0.9)


def test_valid_sequence_summary(synthetic_snapshots) -> None:
    summary = validate_sequence(synthetic_snapshots)
    assert summary.gameweek_start == 1
    assert summary.gameweek_end == 10
    assert summary.gameweeks == 10
    assert summary.minimum_players == 16
    assert summary.minimum_adjacent_coverage == 1.0
    assert summary.season_id == "2024-25"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_train_gameweeks", 4.5),
        ("test_gameweeks_per_fold", True),
        ("split_step", 1.5),
        ("ranking_top_k", 2.5),
        ("random_seed", -1),
        ("ridge_alpha", math.inf),
        pytest.param("ridge_alpha", 10**10_000, id="overflowing-ridge-alpha"),
        pytest.param(
            "minimum_adjacent_player_coverage",
            10**10_000,
            id="overflowing-coverage",
        ),
    ],
)
def test_protocol_rejects_invalid_numeric_boundaries(field: str, value: object) -> None:
    with pytest.raises(DataContractError):
        ProtocolConfig(**{field: value})


def test_rejects_duplicate_json_keys_and_nonfinite_constants(tmp_path: Path) -> None:
    duplicate = tmp_path / "gameweek-1.json"
    duplicate.write_text(
        '{"snapshot_format":"fpl-completed-gameweek-v1","season_id":"2024-25",'
        '"gameweek":1,"elements":[{"id":1,"stats":{"total_points":2,'
        '"total_points":3,"minutes":90}}]}',
        encoding="utf-8",
    )
    with pytest.raises(DataContractError, match="duplicate key: total_points"):
        read_gameweek(duplicate)

    nonfinite = tmp_path / "gameweek-2.json"
    nonfinite.write_text(
        '{"snapshot_format":"fpl-completed-gameweek-v1","season_id":"2024-25",'
        '"gameweek":2,"elements":[{"id":1,"stats":{"total_points":NaN,'
        '"minutes":90}}]}',
        encoding="utf-8",
    )
    with pytest.raises(DataContractError, match="must be finite"):
        read_gameweek(nonfinite)


def test_configuration_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"ridge_alpha":3.0,"ridge_alpha":4.0}', encoding="utf-8")
    with pytest.raises(DataContractError, match="duplicate key"):
        ProtocolConfig.from_json(path)


@pytest.mark.parametrize(
    "value",
    ["2024-26", "2024/25", " 2024-25", "2024-25 ", "24-25", True, None],
)
def test_season_id_contract_rejects_noncanonical_values(value: object) -> None:
    with pytest.raises(DataContractError):
        validate_season_id(value)


def test_season_id_contract_accepts_consecutive_years() -> None:
    assert validate_season_id("2024-25") == "2024-25"
    assert validate_season_id("1999-00") == "1999-00"


def test_snapshot_wrapper_rejects_raw_missing_unknown_and_duplicate_fields(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "gameweek-1.json"
    _write(raw, {"elements": []})
    with pytest.raises(DataContractError, match="snapshot fields differ"):
        read_gameweek(raw)

    unknown = tmp_path / "gameweek-2.json"
    payload = completed_snapshot_payload(season_id="2024-25", gameweek=2, elements=[])
    payload["downloaded_at"] = "later"
    _write(unknown, payload)
    with pytest.raises(DataContractError, match="unexpected=.*downloaded_at"):
        read_gameweek(unknown)

    duplicate = tmp_path / "gameweek-3.json"
    duplicate.write_text(
        '{"snapshot_format":"fpl-completed-gameweek-v1","season_id":"2024-25",'
        '"season_id":"2025-26","gameweek":3,"elements":[]}',
        encoding="utf-8",
    )
    with pytest.raises(DataContractError, match="duplicate key: season_id"):
        read_gameweek(duplicate)


def test_snapshot_wrapper_rejects_format_and_filename_gameweek_mismatch(tmp_path: Path) -> None:
    wrong_format = completed_snapshot_payload(season_id="2024-25", gameweek=1, elements=[])
    wrong_format["snapshot_format"] = "fpl-completed-gameweek-v2"
    _write(tmp_path / "gameweek-1.json", wrong_format)
    with pytest.raises(DataContractError, match="unsupported snapshot_format"):
        read_gameweek(tmp_path / "gameweek-1.json")

    mismatch = completed_snapshot_payload(season_id="2024-25", gameweek=7, elements=[])
    _write(tmp_path / "gameweek-2.json", mismatch)
    with pytest.raises(DataContractError, match="does not match filename"):
        read_gameweek(tmp_path / "gameweek-2.json")


def test_snapshot_sequence_rejects_mixed_or_unexpected_seasons(tmp_path: Path) -> None:
    first = generate_synthetic_gameweeks(
        gameweeks=2,
        players=4,
        seed=3,
        season_id="2024-25",
    )[1]
    second = generate_synthetic_gameweeks(
        gameweeks=2,
        players=4,
        seed=3,
        season_id="2025-26",
    )[2]
    _write(tmp_path / "gameweek-1.json", first)
    _write(tmp_path / "gameweek-2.json", second)
    with pytest.raises(DataContractError, match="exactly one season_id"):
        load_gameweeks(tmp_path)

    second["season_id"] = "2024-25"
    _write(tmp_path / "gameweek-2.json", second)
    with pytest.raises(DataContractError, match="expected 2025-26, found 2024-25"):
        load_gameweeks(tmp_path, expected_season_id="2025-26")


def test_snapshot_wrapper_constant_is_stable() -> None:
    payload = completed_snapshot_payload(season_id="2024-25", gameweek=1, elements=[])
    assert payload == {
        "snapshot_format": SNAPSHOT_FORMAT,
        "season_id": "2024-25",
        "gameweek": 1,
        "elements": [],
    }


def test_snapshot_numbers_that_overflow_float_are_rejected_cleanly(tmp_path: Path) -> None:
    enormous_integer = "1" + ("0" * 10_000)
    (tmp_path / "gameweek-1.json").write_text(
        '{"snapshot_format":"fpl-completed-gameweek-v1","season_id":"2024-25",'
        f'"gameweek":1,"elements":[{{"id":{enormous_integer},'
        '"stats":{"total_points":1,"minutes":90}}]}',
        encoding="utf-8",
    )
    with pytest.raises(DataContractError, match="Invalid numeric JSON value"):
        read_gameweek(tmp_path / "gameweek-1.json")
