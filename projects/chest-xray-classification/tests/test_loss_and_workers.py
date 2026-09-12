"""Regression tests for weighted epoch reporting and spawn-based image workers."""

import hashlib

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor

from chest_xray_benchmark.image_data import build_image_dataset
from chest_xray_benchmark.training import _fit_epoch, predict_probabilities


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("fitting", [False, True])
def test_weighted_epoch_loss_is_independent_of_batch_boundaries(batch_size, fitting):
    # Weighted NLL numerator = -log(.8) - 9log(.2) - log(.6) - 9log(.7).
    probabilities = torch.tensor([[0.8, 0.2], [0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(probabilities.log(), labels), batch_size=batch_size)
    network = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        network.weight.copy_(torch.eye(2))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 9.0]))
    expected = (-np.log(0.8) - 9 * np.log(0.2) - np.log(0.6) - 9 * np.log(0.7)) / 20
    if fitting:
        loss = _fit_epoch(
            network,
            loader,
            criterion,
            torch.optim.SGD(network.parameters(), lr=0),
            torch.device("cpu"),
        )
    else:
        _, _, loss = predict_probabilities(network, loader, criterion, torch.device("cpu"))
    assert loss == pytest.approx(expected, rel=1e-6)


def test_image_dataset_runs_in_spawn_worker(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)
    raw = image_path.read_bytes()
    dataset = build_image_dataset(
        [
            {
                "relative_path": "image.png",
                "label": "Normal",
                "byte_size": str(len(raw)),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
        tmp_path,
        ToTensor(),
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=1, multiprocessing_context="spawn")
    batches = list(loader)
    assert len(batches) == 1
    images, labels = batches[0]
    assert labels.tolist() == [0]
    assert images.shape == (1, 3, 8, 8)
    assert images[0, :, 0, 0].tolist() == [1.0, 0.0, 0.0]
