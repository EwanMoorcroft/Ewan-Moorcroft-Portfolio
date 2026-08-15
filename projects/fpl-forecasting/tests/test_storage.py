"""Adversarial tests for local immutable operational persistence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest

from fpl_forecasting.errors import DataContractError
from fpl_forecasting.storage import (
    BatchRequest,
    ModelRegistration,
    OperationalStore,
    StorageConflictError,
    StorageError,
    StoredPrediction,
    StoreWriter,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


@pytest.fixture
def model() -> ModelRegistration:
    return ModelRegistration(
        season_id="2024-25",
        artifact_format="fpl-ridge-json-v2",
        artifact_sha256=SHA_A,
        feature_schema_version="fpl-feature-schema-v1",
        feature_schema_sha256=SHA_B,
        protocol_config_sha256=SHA_C,
        trained_through_target_gw=10,
        training_rows=250,
    )


@pytest.fixture
def batch(model: ModelRegistration) -> BatchRequest:
    return BatchRequest(
        model_id=model.model_id,
        season_id=model.season_id,
        as_of_gw=10,
        target_gw=11,
        source_snapshot_sha256=SHA_D,
        forecast_frame_sha256=SHA_E,
        feature_schema_sha256=model.feature_schema_sha256,
        protocol_config_sha256=model.protocol_config_sha256,
        predictions_file_sha256=SHA_A,
        manifest_file_sha256=SHA_B,
    )


@pytest.fixture
def predictions() -> tuple[StoredPrediction, ...]:
    return (
        StoredPrediction(player_id=7, prediction=6.5, rank=1),
        StoredPrediction(player_id=12, prediction=4.25, rank=2),
    )


def _store(tmp_path: Path) -> OperationalStore:
    return OperationalStore(tmp_path / "operations.duckdb")


def _register(store: OperationalStore, model: ModelRegistration) -> None:
    with store.writer() as writer:
        writer.register_model(model)


def test_migration_and_registration_are_idempotent(
    tmp_path: Path, model: ModelRegistration
) -> None:
    store = _store(tmp_path)
    assert store.health() == {"status": "ready", "schema_version": 1}
    with store.writer() as writer:
        first = writer.register_model(model)
        second = writer.register_model(model)
    assert first == second == model.model_id
    assert store.counts() == {"models": 1, "runs": 0, "predictions": 0}


def test_identical_batch_replay_does_not_duplicate_rows(
    tmp_path: Path,
    model: ModelRegistration,
    batch: BatchRequest,
    predictions: tuple[StoredPrediction, ...],
) -> None:
    store = _store(tmp_path)
    _register(store, model)
    with store.writer() as writer:
        first = writer.persist_batch(batch, predictions)
    with store.writer() as writer:
        replay = writer.persist_batch(batch, predictions)
    assert first.replayed is False
    assert replay.replayed is True
    assert replay.run_id == first.run_id
    assert store.counts() == {"models": 1, "runs": 1, "predictions": 2}


@pytest.mark.parametrize(
    "changed",
    [
        {"source_snapshot_sha256": SHA_A},
        {"forecast_frame_sha256": SHA_A},
        {"as_of_gw": 11, "target_gw": 12},
    ],
)
def test_changed_request_components_produce_distinct_runs(
    tmp_path: Path,
    model: ModelRegistration,
    batch: BatchRequest,
    predictions: tuple[StoredPrediction, ...],
    changed: dict[str, object],
) -> None:
    store = _store(tmp_path)
    _register(store, model)
    other = replace(batch, **changed)
    assert other.idempotency_key != batch.idempotency_key
    with store.writer() as writer:
        first = writer.persist_batch(batch, predictions)
        second = writer.persist_batch(other, predictions)
    assert first.run_id != second.run_id
    assert store.counts() == {"models": 1, "runs": 2, "predictions": 4}


def test_same_request_with_changed_output_is_a_conflict(
    tmp_path: Path,
    model: ModelRegistration,
    batch: BatchRequest,
    predictions: tuple[StoredPrediction, ...],
) -> None:
    store = _store(tmp_path)
    _register(store, model)
    with store.writer() as writer:
        writer.persist_batch(batch, predictions)
    changed = (replace(predictions[0], prediction=99.0), predictions[1])
    with pytest.raises(StorageConflictError, match="different batch result"):
        with store.writer() as writer:
            writer.persist_batch(batch, changed)
    assert store.counts() == {"models": 1, "runs": 1, "predictions": 2}


@pytest.mark.parametrize(
    "changed",
    [
        {"predictions_file_sha256": SHA_C},
        {"manifest_file_sha256": SHA_D},
    ],
)
def test_same_causal_request_with_changed_evidence_hash_is_a_conflict(
    tmp_path: Path,
    model: ModelRegistration,
    batch: BatchRequest,
    predictions: tuple[StoredPrediction, ...],
    changed: dict[str, str],
) -> None:
    store = _store(tmp_path)
    _register(store, model)
    altered_evidence = replace(batch, **changed)
    assert altered_evidence.idempotency_key == batch.idempotency_key
    with store.writer() as writer:
        writer.persist_batch(batch, predictions)
    with pytest.raises(StorageConflictError, match="different batch result"):
        with store.writer() as writer:
            writer.persist_batch(altered_evidence, predictions)


def test_mid_write_failure_rolls_back_run_and_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: ModelRegistration,
    batch: BatchRequest,
    predictions: tuple[StoredPrediction, ...],
) -> None:
    store = _store(tmp_path)
    _register(store, model)
    original = StoreWriter._insert_prediction
    writes = 0

    def fail_on_second(writer: StoreWriter, run_id: str, prediction: StoredPrediction) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise RuntimeError("injected failure")
        original(writer, run_id, prediction)

    monkeypatch.setattr(StoreWriter, "_insert_prediction", fail_on_second)
    with pytest.raises(RuntimeError, match="injected failure"):
        with store.writer() as writer:
            writer.persist_batch(batch, predictions)
    assert store.counts() == {"models": 1, "runs": 0, "predictions": 0}


def test_nested_writer_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with store.writer():
        with pytest.raises(StorageConflictError, match="writer transaction"):
            with store.writer():
                pass


def test_second_store_object_cannot_open_parallel_writer(tmp_path: Path) -> None:
    database = tmp_path / "operational.duckdb"
    first = OperationalStore(database)
    second = OperationalStore(database)
    with first.writer():
        with pytest.raises(StorageConflictError, match="already active"):
            with second.writer():
                pass


def test_concurrent_initialization_is_serialized_per_database(tmp_path: Path) -> None:
    database = tmp_path / "concurrent.duckdb"

    def initialize(_index: int) -> dict[str, int | str]:
        return OperationalStore(database).health()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(initialize, range(8)))
    assert results == [{"status": "ready", "schema_version": 1}] * 8


def test_connect_failure_is_wrapped_and_does_not_leak_writer_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    original_connect = store._connect
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise duckdb.IOException("private database detail")
        return original_connect()

    monkeypatch.setattr(store, "_connect", fail_once)
    with pytest.raises(StorageError, match="write failed") as captured:
        with store.writer():
            pass
    assert "private database detail" not in str(captured.value)
    with store.writer():
        pass


@pytest.mark.parametrize(
    "unranked",
    [
        (
            StoredPrediction(player_id=7, prediction=4.0, rank=1),
            StoredPrediction(player_id=12, prediction=5.0, rank=2),
        ),
        (
            StoredPrediction(player_id=12, prediction=5.0, rank=1),
            StoredPrediction(player_id=7, prediction=5.0, rank=2),
        ),
    ],
)
def test_direct_storage_rejects_incorrect_prediction_ranking(
    tmp_path: Path,
    model: ModelRegistration,
    batch: BatchRequest,
    unranked: tuple[StoredPrediction, ...],
) -> None:
    store = _store(tmp_path)
    _register(store, model)
    with pytest.raises(DataContractError, match="prediction descending"):
        with store.writer() as writer:
            writer.persist_batch(batch, unranked)


def test_latest_predictions_are_ranked_and_bounded(
    tmp_path: Path,
    model: ModelRegistration,
    batch: BatchRequest,
    predictions: tuple[StoredPrediction, ...],
) -> None:
    store = _store(tmp_path)
    _register(store, model)
    with store.writer() as writer:
        writer.persist_batch(batch, predictions)
    result = store.latest_predictions(
        model_id=model.model_id,
        season_id="2024-25",
        target_gw=11,
        limit=1,
    )
    assert result is not None
    assert result.run_id == batch.run_id
    assert result.total_predictions == 2
    assert result.predictions_file_sha256 == SHA_A
    assert result.manifest_file_sha256 == SHA_B
    assert result.predictions == (predictions[0],)
    assert (
        store.latest_predictions(
            model_id=model.model_id,
            season_id="2024-25",
            target_gw=12,
            limit=10,
        )
        is None
    )
