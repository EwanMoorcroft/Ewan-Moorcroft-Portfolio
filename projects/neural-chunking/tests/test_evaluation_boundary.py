"""Tests for checkpoint identity and the explicit test boundary."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from neural_chunking.data import build_vocabularies, read_sentences, split_by_content_hash
from neural_chunking.training import (
    TrainingConfig,
    build_model,
    evaluate_pipeline,
    file_digest,
)


def write_fixture(path: Path) -> None:
    """Write enough one-token sentences for stable non-empty hash partitions."""
    path.write_text(
        "".join(f"token-{index} O\n\n" for index in range(200)),
        encoding="utf-8",
    )


def checkpoint_for(data_path: Path, checkpoint_path: Path) -> None:
    """Create an untrained tensor checkpoint that follows the public format."""
    config = TrainingConfig(
        architecture="bilstm",
        batch_size=32,
        epochs=1,
        embedding_size=8,
        hidden_size=8,
        dropout=0.0,
    )
    splits = split_by_content_hash(read_sentences(data_path), seed=config.seed)
    tokens, labels = build_vocabularies(splits["train"])
    model = build_model(config, len(tokens), len(labels))
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "token_values": list(tokens.index_to_value),
            "label_values": list(labels.index_to_value),
            "data_file_sha256": file_digest(data_path),
        },
        checkpoint_path,
    )


def test_evaluation_requires_the_selection_data_identity(tmp_path: Path) -> None:
    """Evaluation should reject any data file changed after model selection."""
    data_path = tmp_path / "chunks.txt"
    checkpoint_path = tmp_path / "chunker.pt"
    write_fixture(data_path)
    checkpoint_for(data_path, checkpoint_path)
    data_path.write_text(
        data_path.read_text(encoding="utf-8") + "changed O\n\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="input data differs"):
        evaluate_pipeline(data_path, checkpoint_path, tmp_path / "result.json")


def test_evaluation_writes_only_test_evidence(tmp_path: Path) -> None:
    """A frozen expected-format checkpoint should produce scoped test metrics."""
    data_path = tmp_path / "chunks.txt"
    checkpoint_path = tmp_path / "chunker.pt"
    output_path = tmp_path / "result.json"
    write_fixture(data_path)
    checkpoint_for(data_path, checkpoint_path)
    result = evaluate_pipeline(data_path, checkpoint_path, output_path)
    assert result["status"] == "fresh_test_evaluation"
    assert result["test"]["token_accuracy"] == 1.0
    assert result["test"]["token_matthews_correlation"] == 0.0
    assert output_path.is_file()
