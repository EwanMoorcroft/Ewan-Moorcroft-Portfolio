from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torchvision

from chest_xray_benchmark.evaluation import evaluate_checkpoint
from chest_xray_benchmark.image_data import LABEL_ORDER, build_transforms
from chest_xray_benchmark.metrics import classification_metrics
from chest_xray_benchmark.modeling import build_resnet18
from chest_xray_benchmark.training import file_digest, predict_probabilities


def test_training_dependencies_and_current_model_api_are_available() -> None:
    """Exercise the optional training imports and model construction without fitting a model."""

    network = build_resnet18(class_count=3, dropout=0.2, pretrained=False)
    train_transform, evaluation_transform = build_transforms(image_size=64, rotation_degrees=2.0)
    network.eval()
    with torch.inference_mode():
        logits = network(torch.zeros((1, 3, 64, 64), dtype=torch.float32))
    probabilities = np.eye(3, dtype=np.float64)
    metrics = classification_metrics(np.arange(3), probabilities, ("A", "B", "C"))

    assert isinstance(network, torch.nn.Module)
    assert network.fc[-1].out_features == 3
    assert logits.shape == (1, 3)
    assert len(train_transform.transforms) == 5
    assert len(evaluation_transform.transforms) == 4
    assert metrics["accuracy"] == 1.0
    assert torch.__version__
    assert torchvision.__version__


def test_predict_probabilities_copies_reused_tensor_storage() -> None:
    images = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.int64)
    logits = torch.tensor([[4.0, 1.0, 0.0], [0.0, 4.0, 1.0]])
    probability_buffer = torch.empty((2, 3), dtype=torch.float32)
    probability_batches = iter(
        (
            torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
            torch.tensor([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]),
        )
    )

    class FixedNetwork(torch.nn.Module):
        def forward(self, batch):
            return logits[: len(batch)]

    def loader():
        yield images, labels
        labels.copy_(torch.tensor([2, 0]))
        yield images, labels

    def reuse_probability_storage(_logits, dim):
        assert dim == 1
        probability_buffer.copy_(next(probability_batches))
        return probability_buffer

    with patch("torch.softmax", side_effect=reuse_probability_storage):
        observed_labels, observed_probabilities, _ = predict_probabilities(
            FixedNetwork(),
            loader(),
            torch.nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )

    np.testing.assert_array_equal(observed_labels, np.array([0, 1, 2, 0]))
    np.testing.assert_allclose(
        observed_probabilities,
        np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.8, 0.1, 0.1],
            ]
        ),
    )


def test_classification_metrics_rejects_out_of_range_labels() -> None:
    with pytest.raises(ValueError, match="outside the configured label range"):
        classification_metrics(
            np.array([0, 20]),
            np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]),
            ("A", "B", "C"),
        )


def test_evaluation_rejects_checkpoint_mismatch_before_torch_load() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        splits_path = root / "splits.csv"
        checkpoint_path = root / "selected.pt"
        metadata_path = root / "run-metadata.json"
        splits_path.write_text("fixture split bytes\n", encoding="utf-8")
        checkpoint_path.write_bytes(b"different checkpoint bytes")
        metadata_path.write_text(
            json.dumps(
                {
                    "label_order": list(LABEL_ORDER),
                    "split_file_sha256": file_digest(splits_path),
                    "checkpoint_file_sha256": "0" * 64,
                }
            ),
            encoding="utf-8",
        )

        with (
            patch("torch.load") as mocked_load,
            pytest.raises(ValueError, match="differs from the state selected"),
        ):
            evaluate_checkpoint(
                root,
                splits_path,
                checkpoint_path,
                metadata_path,
                root / "metrics.json",
            )
        mocked_load.assert_not_called()
