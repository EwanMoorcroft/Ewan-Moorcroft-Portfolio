"""Train-only fitting and validation-based model selection."""

from __future__ import annotations

import hashlib
import json
import platform
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .image_data import LABEL_ORDER, build_image_dataset, build_transforms
from .metrics import classification_metrics
from .modeling import build_resnet18
from .settings import RunSettings
from .splitting import read_split_rows, write_json


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_: int) -> None:
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def select_device(requested: str):
    import torch

    if requested == "auto":
        if torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, seed: int, device_type: str):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device_type == "cuda",
        worker_init_fn=seed_worker if workers else None,
        generator=generator,
        persistent_workers=workers > 0,
    )


def _fit_epoch(network, loader, loss_function, optimizer, device) -> float:
    network.train()
    total_loss = 0.0
    total_images = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = network(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        total_images += len(labels)
    return total_loss / max(total_images, 1)


def predict_probabilities(network, loader, loss_function, device):
    import numpy as np
    import torch

    network.eval()
    labels_out = []
    probabilities_out = []
    total_loss = 0.0
    total_images = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = network(images)
            loss = loss_function(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            labels_out.append(labels.cpu().numpy())
            probabilities_out.append(probabilities.cpu().numpy())
            total_loss += float(loss.detach()) * len(labels)
            total_images += len(labels)
    return (
        np.concatenate(labels_out),
        np.concatenate(probabilities_out),
        total_loss / max(total_images, 1),
    )


def _class_weights(rows: list[dict[str, str]]):
    import torch

    counts = Counter(row["label"] for row in rows)
    missing = sorted(set(LABEL_ORDER) - set(counts))
    if missing:
        raise ValueError(f"Train partition is missing classes: {', '.join(missing)}")
    total = sum(counts.values())
    values = [total / (len(LABEL_ORDER) * counts[label]) for label in LABEL_ORDER]
    return torch.tensor(values, dtype=torch.float32), counts


def _cpu_state(network) -> dict[str, Any]:
    return {name: value.detach().cpu() for name, value in network.state_dict().items()}


def train_model(
    data_root: Path,
    splits_path: Path,
    settings_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import numpy
    import PIL
    import sklearn
    import torch
    import torchvision
    from torch import nn

    settings = RunSettings.load(settings_path)
    seed_everything(settings.seed)
    device = select_device(settings.device)
    train_rows = read_split_rows(splits_path, "train")
    validation_rows = read_split_rows(splits_path, "validation")
    if not train_rows or not validation_rows:
        raise ValueError("Train and validation partitions must be non-empty")

    train_transform, evaluation_transform = build_transforms(
        settings.image_size,
        settings.rotation_degrees,
    )
    train_data = build_image_dataset(train_rows, data_root, train_transform)
    validation_data = build_image_dataset(validation_rows, data_root, evaluation_transform)
    train_loader = make_loader(
        train_data,
        settings.batch_size,
        True,
        settings.num_workers,
        settings.seed,
        device.type,
    )
    validation_loader = make_loader(
        validation_data,
        settings.batch_size,
        False,
        settings.num_workers,
        settings.seed + 1,
        device.type,
    )

    network = build_resnet18(len(LABEL_ORDER), settings.dropout, settings.pretrained).to(device)
    weight_values, train_counts = _class_weights(train_rows)
    loss_function = nn.CrossEntropyLoss(weight=weight_values.to(device))
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best-state-dict.pt"
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, settings.epochs + 1):
        train_loss = _fit_epoch(network, train_loader, loss_function, optimizer, device)
        labels, probabilities, validation_loss = predict_probabilities(
            network,
            validation_loader,
            loss_function,
            device,
        )
        metrics = classification_metrics(labels, probabilities, LABEL_ORDER)
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_macro_f1": metrics["macro_f1"],
            "validation_balanced_accuracy": metrics["balanced_accuracy"],
            "validation_matthews_correlation_coefficient": metrics[
                "matthews_correlation_coefficient"
            ],
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, sort_keys=True))
        if metrics["macro_f1"] > best_f1 + 1e-12:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(_cpu_state(network), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= settings.early_stopping_patience:
                break

    split_sha256 = file_digest(splits_path)
    config_sha256 = file_digest(settings_path)
    metadata = {
        "evidence_status": "fresh training and validation",
        "test_partition_used": False,
        "architecture": "resnet18",
        "label_order": list(LABEL_ORDER),
        "selection_metric": "validation_macro_f1",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "settings": settings.to_dict(),
        "train_class_counts": dict(sorted(train_counts.items())),
        "validation_image_count": len(validation_rows),
        "split_file_sha256": split_sha256,
        "config_file_sha256": config_sha256,
        "checkpoint_file": checkpoint_path.name,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": numpy.__version__,
        "pillow_version": PIL.__version__,
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "device_type": device.type,
    }
    write_json({"epochs": history}, output_dir / "history.json")
    write_json(metadata, output_dir / "run-metadata.json")
    return metadata
