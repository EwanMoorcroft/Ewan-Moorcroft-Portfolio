"""Tests for checkpoint identity and the explicit test boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neural_chunking import training
from neural_chunking.training import (
    CHECKPOINT_FILENAME,
    SELECTION_CONTRACT_ID,
    SELECTION_RECORD_FILENAME,
    TrainingConfig,
    evaluate_pipeline,
    file_digest,
    train_pipeline,
)


def write_fixture(path: Path) -> None:
    """Write enough one-token sentences for stable non-empty hash partitions."""

    path.write_text(
        "".join(f"token-{index} O\n\n" for index in range(200)),
        encoding="utf-8",
    )


@pytest.fixture
def selected_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Create a real one-epoch validation-selected checkpoint and its record."""

    monkeypatch.setattr(training, "select_device", lambda: torch.device("cpu"))
    data_path = tmp_path / "chunks.txt"
    output_dir = tmp_path / "run"
    write_fixture(data_path)
    train_pipeline(
        data_path,
        output_dir,
        TrainingConfig(
            architecture="bilstm",
            batch_size=64,
            epochs=1,
            patience=1,
            embedding_size=8,
            hidden_size=8,
            dropout=0.0,
        ),
    )
    return {
        "data": data_path,
        "checkpoint": output_dir / CHECKPOINT_FILENAME,
        "selection": output_dir / SELECTION_RECORD_FILENAME,
        "result": tmp_path / "test-results.json",
    }


def test_training_persists_selection_and_checkpoint_identity(
    selected_run: dict[str, Path],
) -> None:
    record = json.loads(selected_run["selection"].read_text(encoding="utf-8"))

    assert record["selection"]["contract"] == SELECTION_CONTRACT_ID
    assert record["selection"]["best_epoch"] == 1
    assert record["selection"]["metric"] == "validation_span_f1"
    assert record["checkpoint"]["file"] == CHECKPOINT_FILENAME
    assert record["checkpoint"]["sha256"] == file_digest(selected_run["checkpoint"])
    assert record["test_partition_used"] is False


def test_training_rejects_over_length_transformer_before_model_indexing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "chunks.txt"
    write_fixture(data_path)
    with data_path.open("a", encoding="utf-8") as stream:
        stream.write("one O\ntwo O\nthree O\n\n")
    monkeypatch.setattr(
        training,
        "build_model",
        lambda *args, **kwargs: pytest.fail("model must not be built"),
    )

    with pytest.raises(ValueError, match="3 tokens; configured maximum is 2"):
        train_pipeline(
            data_path,
            tmp_path / "run",
            TrainingConfig(
                architecture="transformer",
                epochs=1,
                transformer_maximum_length=2,
            ),
        )


def test_evaluation_requires_the_selection_data_identity(
    selected_run: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changed input must fail before any checkpoint deserialisation."""

    selected_run["data"].write_text(
        selected_run["data"].read_text(encoding="utf-8") + "changed O\n\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        training.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("checkpoint must not be loaded"),
    )
    with pytest.raises(ValueError, match="input data differs"):
        evaluate_pipeline(
            selected_run["data"],
            selected_run["checkpoint"],
            selected_run["result"],
        )


def test_evaluation_rejects_tampered_checkpoint_before_loading(
    selected_run: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint bytes must match the selected artifact before torch.load runs."""

    with selected_run["checkpoint"].open("ab") as stream:
        stream.write(b"tampered")
    monkeypatch.setattr(
        training.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("checkpoint must not be loaded"),
    )
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        evaluate_pipeline(
            selected_run["data"],
            selected_run["checkpoint"],
            selected_run["result"],
        )


def test_evaluation_rejects_changed_selection_contract_before_loading(
    selected_run: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown validation-selection rule must fail before deserialisation."""

    record = json.loads(selected_run["selection"].read_text(encoding="utf-8"))
    record["selection"]["contract"] = "arbitrary-selection"
    selected_run["selection"].write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        training.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("checkpoint must not be loaded"),
    )
    with pytest.raises(ValueError, match="selection contract"):
        evaluate_pipeline(
            selected_run["data"],
            selected_run["checkpoint"],
            selected_run["result"],
        )


def test_evaluation_rejects_non_finite_selected_metric(
    selected_run: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupted validation metric must fail before checkpoint deserialisation."""

    record = json.loads(selected_run["selection"].read_text(encoding="utf-8"))
    record["selection"]["validation_metrics"]["span_precision"] = float("inf")
    selected_run["selection"].write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        training.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("checkpoint must not be loaded"),
    )
    with pytest.raises(ValueError, match="finite numbers"):
        evaluate_pipeline(
            selected_run["data"],
            selected_run["checkpoint"],
            selected_run["result"],
        )


def test_evaluation_binds_checkpoint_payload_to_selection_record(
    selected_run: dict[str, Path],
) -> None:
    """A rehashed checkpoint with different selection metadata is still rejected."""

    payload = torch.load(selected_run["checkpoint"], map_location="cpu", weights_only=True)
    payload["selection"] = {**payload["selection"], "best_epoch": 999}
    torch.save(payload, selected_run["checkpoint"])
    record = json.loads(selected_run["selection"].read_text(encoding="utf-8"))
    record["checkpoint"]["sha256"] = file_digest(selected_run["checkpoint"])
    selected_run["selection"].write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint selection contract"):
        evaluate_pipeline(
            selected_run["data"],
            selected_run["checkpoint"],
            selected_run["result"],
        )


def test_selected_checkpoint_evaluation_records_provenance(
    selected_run: dict[str, Path],
) -> None:
    """A genuine selected checkpoint should produce scoped, bound test evidence."""

    result = evaluate_pipeline(
        selected_run["data"],
        selected_run["checkpoint"],
        selected_run["result"],
    )
    assert result["status"] == "fresh_test_evaluation"
    assert result["selection"]["contract"] == SELECTION_CONTRACT_ID
    assert result["checkpoint"]["sha256"] == file_digest(selected_run["checkpoint"])
    assert result["checkpoint"]["selection_metadata_file"] == SELECTION_RECORD_FILENAME
    assert result["config"]["architecture"] == "bilstm"
    assert result["test_partition_used"] is True
    assert selected_run["result"].is_file()
