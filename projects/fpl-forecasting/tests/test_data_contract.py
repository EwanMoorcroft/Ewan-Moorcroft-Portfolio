from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from fpl_forecasting.config import ProtocolConfig
from fpl_forecasting.data import load_gameweeks, read_gameweek, validate_sequence
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
    _write(path, {"elements": []})
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_train_gameweeks", 4.5),
        ("test_gameweeks_per_fold", True),
        ("split_step", 1.5),
        ("ranking_top_k", 2.5),
        ("random_seed", -1),
        ("ridge_alpha", math.inf),
    ],
)
def test_protocol_rejects_invalid_numeric_boundaries(field: str, value: object) -> None:
    with pytest.raises(DataContractError):
        ProtocolConfig(**{field: value})
