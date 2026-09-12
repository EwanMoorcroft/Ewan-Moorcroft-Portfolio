from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest

from fpl_forecasting.cli import main
from fpl_forecasting.config import ProtocolConfig
from fpl_forecasting.data import load_gameweeks, read_gameweek
from fpl_forecasting.errors import DataContractError, LeakageError
from fpl_forecasting.features import assert_leak_free_frame, build_training_frame
from fpl_forecasting.operational import snapshot_input_audit
from fpl_forecasting.service import RequestBodyLimitMiddleware
from fpl_forecasting.storage import StoredPrediction


def train(directory, artifact):
    return main(
        [
            "train",
            "--gameweek-dir",
            str(directory),
            "--artifact",
            str(artifact),
            "--season",
            "2024-25",
        ]
    )


def predict(directory, artifact, predictions, manifest):
    return main(
        [
            "predict",
            "--gameweek-dir",
            str(directory),
            "--artifact",
            str(artifact),
            "--season",
            "2024-25",
            "--expected-as-of-gw",
            "10",
            "--completion-status",
            "completed",
            "--predictions",
            str(predictions),
            "--manifest",
            str(manifest),
        ]
    )


@pytest.mark.parametrize("mutation", ["truncate", "replace"])
def test_prediction_rejects_changed_history(synthetic_directory, tmp_path, mutation):
    artifact = tmp_path / "ridge.json"
    assert train(synthetic_directory, artifact) == 0
    if mutation == "truncate":
        for path in sorted(synthetic_directory.glob("*.json"))[:8]:
            path.unlink()
    else:
        path = sorted(synthetic_directory.glob("*.json"))[0]
        payload = json.loads(path.read_text())
        payload["elements"][0]["stats"]["total_points"] += 1
        path.write_text(json.dumps(payload))
    predictions = tmp_path / "predictions.json"
    assert predict(synthetic_directory, artifact, predictions, tmp_path / "manifest.json") == 2
    assert not predictions.exists()


def test_snapshot_audit_hashes_parsed_bytes(synthetic_directory):
    snapshots = load_gameweeks(synthetic_directory)
    original = snapshots[0].source.read_bytes()
    snapshots[0].source.write_bytes(b"changed after parsing")
    files, _ = snapshot_input_audit(snapshots)
    assert files[0]["sha256"] == hashlib.sha256(original).hexdigest()
    assert files[0]["bytes"] == len(original)


def test_equivalent_numeric_configs_have_one_hash():
    assert ProtocolConfig(ridge_alpha=4, minimum_adjacent_player_coverage=1).sha256() == (
        ProtocolConfig(ridge_alpha=4.0, minimum_adjacent_player_coverage=1.0).sha256()
    )


@pytest.mark.parametrize("invalid_id", [1.0, "1", 2_147_483_648, 9_007_199_254_740_993])
def test_snapshot_rejects_non_integer_or_out_of_range_ids(synthetic_directory, invalid_id):
    path = sorted(synthetic_directory.glob("*.json"))[0]
    payload = json.loads(path.read_text())
    payload["elements"][0]["id"] = invalid_id
    path.write_text(json.dumps(payload))
    with pytest.raises(DataContractError):
        read_gameweek(path)


def test_chunked_oversize_rejects_without_reading_more():
    sent = []
    reads = 0

    async def receive():
        nonlocal reads
        reads += 1
        if reads > 1:
            raise AssertionError("Oversized stream was drained")
        return {"type": "http.request", "body": b"12345", "more_body": True}

    async def send(message):
        sent.append(message)

    async def downstream(*args):
        raise AssertionError("Oversized body reached the application")

    middleware = RequestBodyLimitMiddleware(downstream, max_bytes=4)
    asyncio.run(middleware({"type": "http", "method": "POST", "headers": []}, receive, send))
    assert sent[0]["status"] == 413


@pytest.mark.parametrize("command", ["build", "train", "evaluate"])
def test_cli_output_hardlink_cannot_overwrite_input(synthetic_directory, tmp_path, command):
    source = sorted(synthetic_directory.glob("*.json"))[0]
    original = source.read_bytes()
    output = tmp_path / "alias.json"
    os.link(source, output)
    option = {"build": "--output", "train": "--artifact", "evaluate": "--report-json"}[command]
    args = [command, "--gameweek-dir", str(synthetic_directory), option, str(output)]
    if command == "train":
        args.extend(["--season", "2024-25"])
    assert main(args) == 2
    assert source.read_bytes() == original


def test_failed_paired_prediction_write_preserves_existing_output(synthetic_directory, tmp_path):
    artifact = tmp_path / "ridge.json"
    assert train(synthetic_directory, artifact) == 0
    predictions = tmp_path / "predictions.json"
    predictions.write_bytes(b"previous output")
    manifest = tmp_path / "manifest-is-directory"
    manifest.mkdir()
    assert predict(synthetic_directory, artifact, predictions, manifest) == 2
    assert predictions.read_bytes() == b"previous output"


def test_pair_rolls_back_if_second_replace_fails(synthetic_directory, tmp_path, monkeypatch):
    import fpl_forecasting.output as output_api

    artifact = tmp_path / "ridge.json"
    assert train(synthetic_directory, artifact) == 0
    predictions = tmp_path / "predictions.json"
    manifest = tmp_path / "manifest.json"
    predictions.write_bytes(b"old predictions")
    manifest.write_bytes(b"old manifest")
    real_replace = os.replace
    failed = False

    def fail_manifest_once(source, destination):
        nonlocal failed
        if Path(destination) == manifest and not failed:
            failed = True
            raise OSError("simulated interrupted publication")
        return real_replace(source, destination)

    monkeypatch.setattr(output_api.os, "replace", fail_manifest_once)
    assert predict(synthetic_directory, artifact, predictions, manifest) == 2
    assert predictions.read_bytes() == b"old predictions"
    assert manifest.read_bytes() == b"old manifest"
    assert not list(tmp_path.glob(".fpl-*"))


def test_feature_frame_rejects_float_player_ids(synthetic_snapshots):
    frame = build_training_frame(synthetic_snapshots)
    frame["player_id"] = frame["player_id"].astype(float)
    with pytest.raises(LeakageError):
        assert_leak_free_frame(frame)


def test_storage_rejects_player_id_outside_database_bound():
    with pytest.raises(DataContractError):
        StoredPrediction(player_id=2_147_483_648, prediction=1.0, rank=1)
