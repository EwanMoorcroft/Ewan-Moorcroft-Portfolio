"""Deterministic completed-gameweek fixtures for local checks and examples."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from .errors import DataContractError


def generate_synthetic_gameweeks(
    *, gameweeks: int = 10, players: int = 24, seed: int = 42
) -> dict[int, dict[str, Any]]:
    """Generate a small sequence with persistence, form cycles, and absences."""

    if gameweeks < 2:
        raise DataContractError("Synthetic data requires at least two gameweeks")
    if players < 4:
        raise DataContractError("Synthetic data requires at least four players")

    rng = random.Random(seed)
    previous_points = {
        player_id: 1.5 + (player_id % 7) * 0.65 for player_id in range(1, players + 1)
    }
    payloads: dict[int, dict[str, Any]] = {}

    for gameweek in range(1, gameweeks + 1):
        elements: list[dict[str, Any]] = []
        for player_id in range(1, players + 1):
            skill = 1.5 + (player_id % 7) * 0.65
            unavailable = (player_id * 5 + gameweek * 3) % 29 == 0
            minutes = 0 if unavailable else (60 if (player_id + gameweek) % 8 == 0 else 90)
            starts = int(minutes >= 60)
            cycle = math.sin((gameweek + player_id % 4) * math.pi / 3.0)
            noise = rng.gauss(0.0, 0.85)
            raw_points = 0.58 * previous_points[player_id] + 0.42 * (skill + 1.15 * cycle) + noise
            points = 0 if unavailable else int(round(max(-1.0, min(15.0, raw_points))))
            previous_points[player_id] = float(points)
            goals = max(0, int((points - 5) // 4))
            assists = max(0, int((points - 3) // 5))
            clean_sheet = int(minutes >= 60 and (player_id + gameweek) % 5 == 0)
            ict = max(0.0, points * 1.7 + minutes / 45.0 + rng.uniform(0.0, 1.2))

            elements.append(
                {
                    "id": player_id,
                    "stats": {
                        "total_points": points,
                        "minutes": minutes,
                        "starts": starts,
                        "ict_index": round(ict, 3),
                        "influence": round(max(0.0, ict * 0.45), 3),
                        "creativity": round(max(0.0, ict * 0.30), 3),
                        "threat": round(max(0.0, ict * 0.25), 3),
                        "goals_scored": goals,
                        "assists": assists,
                        "clean_sheets": clean_sheet,
                        "saves": int(player_id % 11 == 0 and minutes > 0) * (gameweek % 5),
                        "bonus": max(0, min(3, points // 4)),
                    },
                }
            )
        payloads[gameweek] = {"elements": elements}
    return payloads


def write_synthetic_gameweeks(
    directory: str | Path,
    *,
    gameweeks: int = 10,
    players: int = 24,
    seed: int = 42,
) -> list[Path]:
    """Write deterministic fixtures using the same JSON shape as FPL live data."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    payloads = generate_synthetic_gameweeks(gameweeks=gameweeks, players=players, seed=seed)
    paths: list[Path] = []
    for gameweek, payload in payloads.items():
        path = destination / f"gameweek-{gameweek:02d}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
    return paths
