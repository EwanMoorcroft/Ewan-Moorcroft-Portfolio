from __future__ import annotations

from pathlib import Path

import pytest

from fpl_forecasting.data import GameweekSnapshot, load_gameweeks
from fpl_forecasting.synthetic import write_synthetic_gameweeks


@pytest.fixture
def synthetic_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "gameweeks"
    write_synthetic_gameweeks(directory, gameweeks=10, players=16, seed=17)
    return directory


@pytest.fixture
def synthetic_snapshots(synthetic_directory: Path) -> list[GameweekSnapshot]:
    return load_gameweeks(synthetic_directory)
