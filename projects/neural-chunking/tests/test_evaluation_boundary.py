"""Tests for checkpoint identity and the explicit test boundary."""

from __future__ import annotations

import json
import os
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
        corpus_metadata={
            "name": "synthetic test fixture",
            "version": "1",
            "licence": "test-generated",
        },
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
    assert result["corpus"]["name"] == "synthetic test fixture"
    prediction_path = selected_run["result"].with_name(result["predictions"]["file"])
    assert result["predictions"]["sha256"] == file_digest(prediction_path)
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    assert len(predictions) == result["split_sentences"]["test"]
    assert all(row["expected_tags"] == ["O"] for row in predictions)
    assert all(len(row["predicted_tags"]) == 1 for row in predictions)
    assert all("tokens" not in row and len(row["sentence_sha256"]) == 64 for row in predictions)


def test_derived_prediction_name_cannot_overwrite_corpus(selected_run):
    source = selected_run["result"].with_name("test-results-predictions.jsonl")
    original = selected_run["data"].read_bytes()
    source.write_bytes(original)
    with pytest.raises(ValueError, match="overwrite"):
        evaluate_pipeline(source, selected_run["checkpoint"], selected_run["result"])
    assert source.read_bytes() == original
    assert not selected_run["result"].exists()


@pytest.mark.parametrize("source_key", ["data", "checkpoint", "selection"])
@pytest.mark.parametrize("output_kind", ["report", "predictions"])
@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_output_aliases_cannot_modify_selected_inputs(
    selected_run, source_key, output_kind, alias_kind
):
    source = selected_run[source_key]
    original = source.read_bytes()
    report = selected_run["result"]
    destination = (
        report if output_kind == "report" else report.with_name("test-results-predictions.jsonl")
    )
    if alias_kind == "symlink":
        destination.symlink_to(source)
    else:
        os.link(source, destination)
    with pytest.raises(ValueError, match="overwrite"):
        evaluate_pipeline(selected_run["data"], selected_run["checkpoint"], report)
    assert source.read_bytes() == original


@pytest.mark.parametrize("existing", [True, False])
def test_second_output_failure_rolls_back_prediction_pair(selected_run, monkeypatch, existing):
    report = selected_run["result"]
    predictions = report.with_name("test-results-predictions.jsonl")
    if existing:
        report.write_bytes(b"old report")
        predictions.write_bytes(b"old predictions")
    real_replace = os.replace
    failed = False

    def fail_report_once(source, destination):
        nonlocal failed
        if Path(destination) == report and not failed:
            failed = True
            raise OSError("simulated publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_report_once)
    with pytest.raises(OSError, match="simulated publication failure"):
        evaluate_pipeline(selected_run["data"], selected_run["checkpoint"], report)
    if existing:
        assert report.read_bytes() == b"old report"
        assert predictions.read_bytes() == b"old predictions"
    else:
        assert not report.exists()
        assert not predictions.exists()


def test_report_and_predictions_reconstruct_metrics_including_absent_training_label(
    tmp_path, monkeypatch
):
    from sklearn.metrics import accuracy_score, f1_score

    from neural_chunking.data import read_sentences, split_by_content_hash

    data_path = tmp_path / "corpus.txt"
    write_fixture(data_path)
    training_token = split_by_content_hash(read_sentences(data_path))["train"][0].tokens[0]
    data_path.write_text(
        data_path.read_text().replace(f"{training_token} O\n", f"{training_token} B-NP\n")
    )
    original_build = training.build_model

    def always_predict_o(config, vocabulary_size, label_count):
        model = original_build(config, vocabulary_size, label_count)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.classifier.bias[1] = 10.0  # Alphabetical vocabulary is B-NP, O.
        return model

    monkeypatch.setattr(training, "build_model", always_predict_o)
    monkeypatch.setattr(training, "select_device", lambda: torch.device("cpu"))
    output_dir = tmp_path / "run"
    train_pipeline(
        data_path,
        output_dir,
        TrainingConfig(
            epochs=1,
            patience=1,
            embedding_size=8,
            hidden_size=8,
            batch_size=64,
            learning_rate=1e-12,
            dropout=0.0,
        ),
    )
    report = tmp_path / "report.json"
    evaluate_pipeline(data_path, output_dir / CHECKPOINT_FILENAME, report)
    saved = json.loads(report.read_text())
    predictions = [
        json.loads(line)
        for line in (tmp_path / saved["predictions"]["file"]).read_text().splitlines()
    ]
    gold = [tag for row in predictions for tag in row["expected_tags"]]
    predicted = [tag for row in predictions for tag in row["predicted_tags"]]
    assert set(gold) == set(predicted) == {"O"}
    assert saved["label_order"] == ["B-NP", "O"]
    zero_division = saved["metric_conventions"]["zero_division"]
    assert saved["test"]["token_macro_f1"] == 0.5
    for average, field in [("macro", "token_macro_f1"), ("weighted", "token_weighted_f1")]:
        score = f1_score(
            gold,
            predicted,
            labels=saved["label_order"],
            average=average,
            zero_division=zero_division,
        )
        assert score == saved["test"][field]
    assert accuracy_score(gold, predicted) == saved["test"]["token_accuracy"]
    # MCC is explicitly zero for a single observed gold/predicted label.
    assert saved["metric_conventions"]["single_observed_label_mcc"] == 0.0
    assert saved["test"]["token_matthews_correlation"] == 0.0
    for field in ("span_precision", "span_recall", "span_f1"):
        assert saved["test"][field] == saved["metric_conventions"]["empty_span_score"] == 0.0
