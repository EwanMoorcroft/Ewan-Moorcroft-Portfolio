from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fpl_forecasting.contracts import canonical_json_sha256
from fpl_forecasting.data import GameweekSnapshot, load_gameweeks
from fpl_forecasting.errors import LeakageError
from fpl_forecasting.features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_SHA256,
    assert_forecast_frame,
    assert_leak_free_frame,
    build_forecast_frame,
    build_training_frame,
    feature_matrix,
    feature_schema_payload,
)
from fpl_forecasting.synthetic import write_synthetic_gameweeks


def test_builds_only_completed_next_gameweek_targets(synthetic_snapshots) -> None:
    frame = build_training_frame(synthetic_snapshots)
    assert len(frame) == 9 * 16
    assert frame["as_of_gw"].min() == 1
    assert frame["as_of_gw"].max() == 9
    assert frame["target_gw"].min() == 2
    assert frame["target_gw"].max() == 10
    assert bool((frame["target_gw"] == frame["as_of_gw"] + 1).all())
    assert list(feature_matrix(frame)[0].columns) == list(FEATURE_COLUMNS)


def test_future_changes_do_not_change_earlier_features(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_synthetic_gameweeks(first, gameweeks=7, players=12, seed=29)
    write_synthetic_gameweeks(second, gameweeks=7, players=12, seed=29)

    final_path = second / "gameweek-07.json"
    payload = json.loads(final_path.read_text(encoding="utf-8"))
    for element in payload["elements"]:
        element["stats"]["total_points"] += 50
        element["stats"]["ict_index"] += 100
    final_path.write_text(json.dumps(payload), encoding="utf-8")

    frame_a = build_training_frame(load_gameweeks(first))
    frame_b = build_training_frame(load_gameweeks(second))
    earlier_a = frame_a[frame_a["target_gw"] <= 6].reset_index(drop=True)
    earlier_b = frame_b[frame_b["target_gw"] <= 6].reset_index(drop=True)
    pd.testing.assert_frame_equal(earlier_a, earlier_b)

    final_a = frame_a[frame_a["target_gw"] == 7]
    final_b = frame_b[frame_b["target_gw"] == 7]
    pd.testing.assert_frame_equal(
        final_a.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True),
        final_b.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True),
    )
    assert not final_a["target_next_gw_points"].equals(final_b["target_next_gw_points"])


def test_boundary_guard_rejects_misaligned_target(synthetic_snapshots) -> None:
    frame = build_training_frame(synthetic_snapshots)
    frame.loc[0, "target_gw"] = frame.loc[0, "as_of_gw"]
    with pytest.raises(LeakageError, match="exactly one gameweek"):
        assert_leak_free_frame(frame)


def test_boundary_guard_rejects_fractional_or_duplicate_identifiers(
    synthetic_snapshots,
) -> None:
    frame = build_training_frame(synthetic_snapshots)
    fractional = frame.astype({"as_of_gw": "float64", "target_gw": "float64"})
    fractional.loc[0, "as_of_gw"] = 1.5
    fractional.loc[0, "target_gw"] = 2.5
    with pytest.raises(LeakageError, match="must be integers"):
        assert_leak_free_frame(fractional)

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(LeakageError, match="must be unique"):
        assert_leak_free_frame(duplicated)


def test_boundary_guard_rejects_future_facing_extra_column(
    synthetic_snapshots,
) -> None:
    frame = build_training_frame(synthetic_snapshots)
    frame["future_points"] = 0.0
    with pytest.raises(LeakageError, match="future-facing"):
        assert_leak_free_frame(frame)


def test_feature_schema_hash_covers_order_and_semantics() -> None:
    schema = feature_schema_payload()
    ordered = schema["ordered_features"]
    assert isinstance(ordered, list)
    assert [item["name"] for item in ordered] == list(FEATURE_COLUMNS)
    assert canonical_json_sha256(schema) == FEATURE_SCHEMA_SHA256

    changed = feature_schema_payload()
    changed["as_of_meaning"] = "after target gameweek"
    assert canonical_json_sha256(changed) != FEATURE_SCHEMA_SHA256


def test_forecast_frame_uses_all_history_and_only_latest_players(
    synthetic_snapshots,
) -> None:
    latest = synthetic_snapshots[-1]
    latest_players = dict(latest.players)
    latest_players.pop(1)
    challenged = [
        *synthetic_snapshots[:-1],
        GameweekSnapshot(
            gameweek=latest.gameweek,
            season_id=latest.season_id,
            players=latest_players,
            source=latest.source,
        ),
    ]

    frame = build_forecast_frame(challenged)
    assert set(frame["player_id"]) == set(latest_players)
    assert set(frame["as_of_gw"]) == {latest.gameweek}
    assert set(frame["target_gw"]) == {latest.gameweek + 1}
    expected_sum = sum(snapshot.players[2].total_points for snapshot in challenged)
    observed = frame.loc[frame["player_id"] == 2, "season_points_sum"].item()
    assert observed == expected_sum


def test_frame_guards_reject_boolean_numeric_values(synthetic_snapshots) -> None:
    training = build_training_frame(synthetic_snapshots).astype(object)
    training.loc[0, FEATURE_COLUMNS[0]] = True
    with pytest.raises(LeakageError, match="cannot be boolean"):
        assert_leak_free_frame(training)

    forecast = build_forecast_frame(synthetic_snapshots).astype(object)
    forecast.loc[0, "player_id"] = True
    with pytest.raises(LeakageError, match="cannot be boolean"):
        assert_forecast_frame(forecast)
