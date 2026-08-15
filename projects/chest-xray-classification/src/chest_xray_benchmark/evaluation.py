"""Explicit test-partition evaluation."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

from .image_data import LABEL_ORDER, build_image_dataset, build_transforms
from .metrics import classification_metrics
from .modeling import build_resnet18
from .records import SHA256_PATTERN
from .settings import RunSettings
from .splitting import read_split_rows, write_json
from .training import (
    file_digest,
    make_loader,
    predict_probabilities,
    seed_everything,
    select_device,
)


def read_verified_checkpoint(metadata: dict[str, Any], checkpoint_path: Path) -> tuple[bytes, str]:
    """Read once and require the exact state selected during training."""

    expected = metadata.get("checkpoint_file_sha256")
    if not isinstance(expected, str) or SHA256_PATTERN.fullmatch(expected) is None:
        raise ValueError("Run metadata does not contain a valid selected-checkpoint SHA-256")
    checkpoint_bytes = checkpoint_path.read_bytes()
    observed = hashlib.sha256(checkpoint_bytes).hexdigest()
    if observed != expected:
        raise ValueError("Checkpoint differs from the state selected during training")
    return checkpoint_bytes, observed


def evaluate_checkpoint(
    data_root: Path,
    splits_path: Path,
    checkpoint_path: Path,
    metadata_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    import torch
    from torch import nn

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if tuple(metadata["label_order"]) != LABEL_ORDER:
        raise ValueError("Checkpoint label order does not match the benchmark contract")
    current_split_digest = file_digest(splits_path)
    if metadata["split_file_sha256"] != current_split_digest:
        raise ValueError("The split file differs from the one used for model selection")
    checkpoint_bytes, checkpoint_digest = read_verified_checkpoint(metadata, checkpoint_path)

    settings = RunSettings(**metadata["settings"])
    settings.validate()
    seed_everything(settings.seed)
    device = select_device(settings.device)
    test_rows = read_split_rows(splits_path, "test")
    if not test_rows:
        raise ValueError("Test partition must be non-empty")
    _, evaluation_transform = build_transforms(settings.image_size, settings.rotation_degrees)
    test_data = build_image_dataset(test_rows, data_root, evaluation_transform)
    test_loader = make_loader(
        test_data,
        settings.batch_size,
        False,
        settings.num_workers,
        settings.seed + 2,
        device.type,
    )

    network = build_resnet18(len(LABEL_ORDER), settings.dropout, pretrained=False)
    state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    network.load_state_dict(state, strict=True)
    network.to(device)
    loss_function = nn.CrossEntropyLoss()
    labels, probabilities, test_loss = predict_probabilities(
        network,
        test_loader,
        loss_function,
        device,
    )
    metrics = classification_metrics(labels, probabilities, LABEL_ORDER)
    payload = {
        "evidence_status": "fresh grouped-split test evaluation",
        "label_order": list(LABEL_ORDER),
        "split_file_sha256": current_split_digest,
        "checkpoint_file_sha256": checkpoint_digest,
        "seed": settings.seed,
        "test_loss": test_loss,
        "metrics": metrics,
    }
    write_json(payload, output_path)
    return payload
