"""Explicit test-partition evaluation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
from itertools import pairwise
from pathlib import Path
from typing import Any

from .image_data import LABEL_ORDER, build_image_dataset, build_transforms, resolve_image_path
from .metrics import classification_metrics
from .modeling import build_resnet18
from .output import publish_files, validate_output_paths
from .records import SHA256_PATTERN
from .settings import RunSettings
from .splitting import parse_split_rows
from .training import (
    make_loader,
    predict_probabilities,
    seed_everything,
    select_device,
)


def verify_evaluation_loss(cross_entropy: float, probability_log_loss: float) -> None:
    """Refuse an internally inconsistent unweighted result before publishing outputs."""
    if not math.isclose(cross_entropy, probability_log_loss, rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError(
            "Unweighted cross entropy disagrees with probability log loss; "
            "check device/numerical behavior and rerun on CPU before retaining this evaluation"
        )


def _prediction_artifact(rows, labels, probabilities, path: Path) -> tuple[bytes, dict[str, Any]]:
    """Serialize aligned probabilities before changing any output file."""
    import numpy as np

    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correct = predictions == labels
    fields = [
        "row_id",
        "image_sha256",
        "true_label_index",
        "predicted_label_index",
        *[f"probability_{index}" for index in range(len(LABEL_ORDER))],
    ]
    with io.StringIO(newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, (row, label, prediction, values) in enumerate(
            zip(rows, labels, predictions, probabilities, strict=True),
            start=1,
        ):
            writer.writerow(
                {
                    "row_id": f"test-{index:04d}",
                    "image_sha256": row["sha256"],
                    "true_label_index": int(label),
                    "predicted_label_index": int(prediction),
                    **{f"probability_{i}": float(value) for i, value in enumerate(values)},
                }
            )
        raw = stream.getvalue().encode("utf-8")
    edges = np.linspace(0, 1, 16)
    bins = []
    for lower, upper in pairwise(edges):
        mask = (confidences > lower) & (confidences <= upper)
        count = int(mask.sum())
        bins.append(
            {
                "lower_exclusive": float(lower),
                "upper_inclusive": float(upper),
                "count": count,
                "correct_count": int(correct[mask].sum()),
                "mean_confidence": float(confidences[mask].mean()) if count else None,
                "accuracy": float(correct[mask].mean()) if count else None,
            }
        )
    return raw, {
        "predictions_file": path.name,
        "predictions_file_sha256": hashlib.sha256(raw).hexdigest(),
        "id_policy": "sequential test-row ID plus source image SHA-256; no paths or patient identifiers",
        "error_count": int((~correct).sum()),
        "mean_confidence_correct": float(confidences[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(confidences[~correct].mean())
        if (~correct).any()
        else None,
        "reliability_bins": bins,
    }


def retain_predictions(rows, labels, probabilities, path: Path) -> dict[str, Any]:
    """Persist a standalone prediction artifact after complete serialization."""
    raw, summary = _prediction_artifact(rows, labels, probabilities, path)
    publish_files({path: raw})
    return summary


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
    predictions_path = output_path.with_name(output_path.stem + "-predictions.csv")
    protected = [splits_path, checkpoint_path, metadata_path]
    validate_output_paths([predictions_path, output_path], protected)

    import numpy
    import PIL
    import sklearn
    import torch
    import torchvision
    from torch import nn

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if tuple(metadata["label_order"]) != LABEL_ORDER:
        raise ValueError("Checkpoint label order does not match the benchmark contract")
    split_bytes = splits_path.read_bytes()
    current_split_digest = hashlib.sha256(split_bytes).hexdigest()
    if metadata["split_file_sha256"] != current_split_digest:
        raise ValueError("The split file differs from the one used for model selection")
    checkpoint_bytes, checkpoint_digest = read_verified_checkpoint(metadata, checkpoint_path)

    settings = RunSettings(**metadata["settings"])
    settings.validate()
    seed_everything(settings.seed)
    device = select_device(settings.device)
    split_rows = parse_split_rows(split_bytes)
    protected.extend(resolve_image_path(data_root, row["relative_path"]) for row in split_rows)
    validate_output_paths([predictions_path, output_path], protected)
    test_rows = [row for row in split_rows if row["split"] == "test"]
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
    verify_evaluation_loss(test_loss, metrics["log_loss"])
    prediction_bytes, retained = _prediction_artifact(
        test_rows,
        labels,
        probabilities,
        predictions_path,
    )
    payload = {
        "evidence_status": "fresh grouped-split test evaluation",
        "label_order": list(LABEL_ORDER),
        "split_file_sha256": current_split_digest,
        "checkpoint_file_sha256": checkpoint_digest,
        "seed": settings.seed,
        "test_loss": test_loss,
        "loss_definition": "unweighted mean cross entropy from logits",
        "device_type": device.type,
        "torch_version": torch.__version__,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": numpy.__version__,
            "pillow": PIL.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "prediction_analysis": retained,
        "metrics": metrics,
    }
    report_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    publish_files({predictions_path: prediction_bytes, output_path: report_bytes}, protected)
    return payload
