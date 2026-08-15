"""Training and evaluation orchestration for neural chunkers."""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader

from neural_chunking.data import (
    ChunkDataset,
    Sentence,
    Vocabulary,
    build_vocabularies,
    collate_sentences,
    read_sentences,
    split_by_content_hash,
)
from neural_chunking.metrics import span_metrics
from neural_chunking.models import BiLSTMChunker, TransformerChunker

LABEL_IGNORE_INDEX = -100
CHECKPOINT_SCHEMA_VERSION = 2
SELECTION_RECORD_SCHEMA_VERSION = 2
SELECTION_CONTRACT_ID = "validation_span_f1_max_v1"
SELECTION_METRIC = "validation_span_f1"
CHECKPOINT_FILENAME = "chunker_checkpoint.pt"
SELECTION_RECORD_FILENAME = "training-results.json"
TRANSFORMER_DEFAULT_MAXIMUM_LENGTH = 512
VALIDATION_METRIC_NAMES = {
    "token_accuracy",
    "token_macro_f1",
    "token_weighted_f1",
    "token_matthews_correlation",
    "span_precision",
    "span_recall",
    "span_f1",
}


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for a deterministic neural chunking run."""

    seed: int = 534
    architecture: str = "bilstm"
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 2e-3
    patience: int = 4
    embedding_size: int = 128
    hidden_size: int = 128
    dropout: float = 0.3
    transformer_maximum_length: int = TRANSFORMER_DEFAULT_MAXIMUM_LENGTH

    def validate(self) -> None:
        """Reject invalid or unsupported training settings."""
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.architecture not in {"bilstm", "transformer"}:
            raise ValueError("architecture must be bilstm or transformer")
        if (
            min(
                self.batch_size,
                self.epochs,
                self.patience,
                self.embedding_size,
                self.hidden_size,
                self.transformer_maximum_length,
            )
            < 1
        ):
            raise ValueError(
                "batch, epoch, patience, model sizes, and length limit must be positive"
            )
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def select_device() -> torch.device:
    """Use Apple Metal acceleration when available and otherwise use CPU."""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable data and model state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def file_digest(path: Path) -> str:
    """Return the SHA-256 digest of an input data file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(
    config: TrainingConfig,
    vocabulary_size: int,
    label_count: int,
) -> nn.Module:
    """Create the selected chunker architecture."""
    if config.architecture == "bilstm":
        return BiLSTMChunker(
            vocabulary_size,
            label_count,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
        )
    return TransformerChunker(
        vocabulary_size,
        label_count,
        embedding_size=config.embedding_size,
        feedforward_size=config.hidden_size * 2,
        dropout=config.dropout,
        maximum_length=config.transformer_maximum_length,
    )


def _validate_transformer_lengths(
    sentences: Sequence[Sentence],
    config: TrainingConfig,
) -> None:
    """Reject over-length Transformer inputs before vocabulary or model indexing."""

    if config.architecture != "transformer":
        return
    longest = max(len(sentence.tokens) for sentence in sentences)
    if longest > config.transformer_maximum_length:
        raise ValueError(
            "Transformer input contains "
            f"{longest} tokens; configured maximum is {config.transformer_maximum_length}"
        )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    optimiser: torch.optim.Optimizer | None,
) -> tuple[float, list[int], list[int], list[list[int]], list[list[int]]]:
    """Run one training or evaluation epoch and return aligned predictions."""
    model.train(optimiser is not None)
    total_loss = 0.0
    valid_tokens = 0
    expected_flat: list[int] = []
    predicted_flat: list[int] = []
    expected_rows: list[list[int]] = []
    predicted_rows: list[list[int]] = []
    for tokens, labels, mask in loader:
        tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
        with torch.set_grad_enabled(optimiser is not None):
            logits = model(tokens, mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            if optimiser is not None:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimiser.step()
        predictions = logits.argmax(dim=-1)
        batch_valid_tokens = int(mask.sum().item())
        total_loss += float(loss.item()) * batch_valid_tokens
        valid_tokens += batch_valid_tokens
        for expected_row, predicted_row, valid_row in zip(labels, predictions, mask, strict=True):
            expected = expected_row[valid_row].detach().cpu().tolist()
            predicted = predicted_row[valid_row].detach().cpu().tolist()
            expected_rows.append(expected)
            predicted_rows.append(predicted)
            expected_flat.extend(expected)
            predicted_flat.extend(predicted)
    return (
        total_loss / max(valid_tokens, 1),
        expected_flat,
        predicted_flat,
        expected_rows,
        predicted_rows,
    )


def evaluate_predictions(
    expected_flat: list[int],
    predicted_flat: list[int],
    expected_rows: list[list[int]],
    predicted_rows: list[list[int]],
    labels: Vocabulary,
) -> dict[str, float]:
    """Calculate token metrics and exact-span BIO metrics."""
    expected_tags = [labels.decode(row) for row in expected_rows]
    predicted_tags = [labels.decode(row) for row in predicted_rows]
    spans = span_metrics(expected_tags, predicted_tags)
    label_indices = list(range(len(labels)))
    distinct_indices = set(expected_flat) | set(predicted_flat)
    matthews = (
        float(matthews_corrcoef(expected_flat, predicted_flat))
        if len(distinct_indices) > 1
        else 0.0
    )
    return {
        "token_accuracy": float(accuracy_score(expected_flat, predicted_flat)),
        "token_macro_f1": float(
            f1_score(
                expected_flat,
                predicted_flat,
                labels=label_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "token_weighted_f1": float(
            f1_score(
                expected_flat,
                predicted_flat,
                labels=label_indices,
                average="weighted",
                zero_division=0,
            )
        ),
        "token_matthews_correlation": matthews,
        "span_precision": spans.precision,
        "span_recall": spans.recall,
        "span_f1": spans.f1,
    }


def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy model tensors so later CPU updates cannot mutate the selected state."""
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _collate_for(
    token_vocabulary: Vocabulary,
    label_vocabulary: Vocabulary,
):
    """Create padding behavior for token inputs and ignored label targets."""
    if token_vocabulary.pad_index is None:
        raise ValueError("token vocabulary must define a padding index")
    if label_vocabulary.pad_index is not None:
        raise ValueError("label vocabulary must not expose padding as a prediction class")
    return partial(
        collate_sentences,
        token_pad_index=token_vocabulary.pad_index,
        label_pad_index=LABEL_IGNORE_INDEX,
    )


def _loader(
    rows,
    token_vocabulary: Vocabulary,
    label_vocabulary: Vocabulary,
    config: TrainingConfig,
    shuffle: bool,
    seed_offset: int,
) -> DataLoader:
    """Build a deterministically seeded sentence loader."""
    return DataLoader(
        ChunkDataset(rows, token_vocabulary, label_vocabulary),
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=_collate_for(token_vocabulary, label_vocabulary),
        generator=torch.Generator().manual_seed(config.seed + seed_offset),
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= set("0123456789abcdef")
    )


def _load_selection_record(path: Path) -> tuple[dict[str, Any], TrainingConfig, str]:
    """Read and validate the external selection contract before checkpoint loading."""

    raw = path.read_bytes()
    record_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("selection metadata is not valid UTF-8 JSON") from error
    if not isinstance(record, dict):
        raise ValueError("selection metadata must be a JSON object")
    if record.get("schema_version") != SELECTION_RECORD_SCHEMA_VERSION:
        raise ValueError("selection metadata schema is unsupported")
    if record.get("status") != "fresh_train_validation_run":
        raise ValueError("selection metadata does not describe a training/validation run")
    if record.get("test_partition_used") is not False:
        raise ValueError("selection metadata must confirm that test data was not used")

    raw_config = record.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("selection metadata config must be an object")
    try:
        config = TrainingConfig(**raw_config)
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("selection metadata contains an invalid training config") from error
    if asdict(config) != raw_config:
        raise ValueError("selection metadata config is not canonical")

    selection = record.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("selection metadata must include a selection object")
    fixed_selection = {
        "contract": SELECTION_CONTRACT_ID,
        "metric": SELECTION_METRIC,
        "mode": "max",
    }
    for name, expected in fixed_selection.items():
        if selection.get(name) != expected:
            raise ValueError(f"selection contract has invalid {name}")
    best_epoch = selection.get("best_epoch")
    best_value = selection.get("best_value")
    epochs_completed = selection.get("epochs_completed")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch < 1:
        raise ValueError("selection best_epoch must be a positive integer")
    if (
        isinstance(epochs_completed, bool)
        or not isinstance(epochs_completed, int)
        or epochs_completed < best_epoch
    ):
        raise ValueError("selection epochs_completed must include best_epoch")
    if (
        isinstance(best_value, bool)
        or not isinstance(best_value, (int, float))
        or not math.isfinite(float(best_value))
        or not 0.0 <= float(best_value) <= 1.0
    ):
        raise ValueError("selection best_value must be a finite score in [0, 1]")

    validation_metrics = selection.get("validation_metrics")
    if (
        not isinstance(validation_metrics, dict)
        or set(validation_metrics) != VALIDATION_METRIC_NAMES
    ):
        raise ValueError("selected validation metrics are incomplete")
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        for score in validation_metrics.values()
    ):
        raise ValueError("selected validation metrics must be finite numbers")
    if validation_metrics.get("span_f1") != best_value:
        raise ValueError("selected validation metrics do not match best_value")
    history = record.get("history")
    if not isinstance(history, list) or len(history) != epochs_completed:
        raise ValueError("selection history does not match epochs_completed")
    expected_epochs = list(range(1, epochs_completed + 1))
    observed_epochs = [row.get("epoch") if isinstance(row, dict) else None for row in history]
    if observed_epochs != expected_epochs:
        raise ValueError("selection history epochs must be consecutive")
    scores = [row.get(SELECTION_METRIC) for row in history]
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        for score in scores
    ):
        raise ValueError("selection history contains an invalid metric")
    maximum = max(float(score) for score in scores)
    selected_epoch = next(index for index, score in enumerate(scores, start=1) if score == maximum)
    if best_epoch != selected_epoch or float(best_value) != maximum:
        raise ValueError("selection metadata does not identify the first maximum validation score")
    if record.get("selection_metric") != SELECTION_METRIC:
        raise ValueError("selection_metric does not match the selection contract")
    if record.get("best_epoch") != best_epoch:
        raise ValueError("top-level best_epoch does not match the selection contract")
    if record.get("best_validation_span_f1") != best_value:
        raise ValueError("top-level best validation score does not match the selection contract")

    data_sha256 = record.get("data_file_sha256")
    if not _is_sha256(data_sha256):
        raise ValueError("selection metadata data_file_sha256 is invalid")
    split_sentences = record.get("split_sentences")
    if not isinstance(split_sentences, dict) or set(split_sentences) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("selection metadata split counts are incomplete")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in split_sentences.values()
    ):
        raise ValueError("selection metadata split counts must be positive integers")

    checkpoint = record.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("selection metadata must include checkpoint identity")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("selection metadata checkpoint schema is unsupported")
    checkpoint_file = checkpoint.get("file")
    if (
        not isinstance(checkpoint_file, str)
        or not checkpoint_file
        or Path(checkpoint_file).name != checkpoint_file
    ):
        raise ValueError("selection metadata checkpoint file must be a basename")
    if not _is_sha256(checkpoint.get("sha256")):
        raise ValueError("selection metadata checkpoint SHA-256 is invalid")
    return record, config, record_sha256


def _validate_checkpoint_payload(payload: object, record: dict[str, Any]) -> dict[str, Any]:
    """Bind the deserialised tensors to the already verified selection record."""

    required = {
        "schema_version",
        "artifact_name",
        "model_state",
        "config",
        "token_values",
        "label_values",
        "data_file_sha256",
        "selection",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("checkpoint does not match the expected selected-chunker format")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema is unsupported")
    if payload["artifact_name"] != record["checkpoint"]["file"]:
        raise ValueError("checkpoint filename identity differs from selection metadata")
    if payload["config"] != record["config"]:
        raise ValueError("checkpoint config differs from selection metadata")
    if payload["selection"] != record["selection"]:
        raise ValueError("checkpoint selection contract differs from selection metadata")
    if payload["data_file_sha256"] != record["data_file_sha256"]:
        raise ValueError("checkpoint data identity differs from selection metadata")
    if not isinstance(payload["model_state"], dict) or not payload["model_state"]:
        raise ValueError("checkpoint model_state must be a non-empty mapping")
    for name in ("token_values", "label_values"):
        values = payload[name]
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) for value in values)
        ):
            raise ValueError(f"checkpoint {name} must be a non-empty string list")
    return payload


def train_pipeline(data_path: Path, output_dir: Path, config: TrainingConfig) -> dict[str, Any]:
    """Train and select with validation data without evaluating the test split."""
    config.validate()
    set_seed(config.seed)
    data_sha256 = file_digest(data_path)
    sentences = read_sentences(data_path)
    if file_digest(data_path) != data_sha256:
        raise ValueError("input data changed while it was being read")
    _validate_transformer_lengths(sentences, config)
    splits = split_by_content_hash(sentences, seed=config.seed)
    token_vocabulary, label_vocabulary = build_vocabularies(splits["train"])
    loaders = {
        "train": _loader(
            splits["train"],
            token_vocabulary,
            label_vocabulary,
            config,
            shuffle=True,
            seed_offset=0,
        ),
        "validation": _loader(
            splits["validation"],
            token_vocabulary,
            label_vocabulary,
            config,
            shuffle=False,
            seed_offset=1,
        ),
    }
    device = select_device()
    model = build_model(config, len(token_vocabulary), len(label_vocabulary)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=LABEL_IGNORE_INDEX)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_f1 = -1.0
    best_validation_loss = math.inf
    best_validation_metrics: dict[str, float] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, *_ = _run_epoch(model, loaders["train"], criterion, device, optimiser)
        validation = _run_epoch(model, loaders["validation"], criterion, device, None)
        validation_metrics = evaluate_predictions(*validation[1:], label_vocabulary)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation[0],
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
        )
        if validation_metrics["span_f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["span_f1"]
            best_validation_loss = validation[0]
            best_validation_metrics = dict(validation_metrics)
            best_epoch = epoch
            best_state = _snapshot_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    if best_state is None or best_validation_metrics is None:
        raise RuntimeError("training did not produce a selectable model")
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "contract": SELECTION_CONTRACT_ID,
        "metric": SELECTION_METRIC,
        "mode": "max",
        "best_epoch": best_epoch,
        "best_value": best_validation_f1,
        "validation_loss": best_validation_loss,
        "validation_metrics": best_validation_metrics,
        "epochs_completed": len(history),
    }
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "artifact_name": CHECKPOINT_FILENAME,
            "model_state": best_state,
            "config": asdict(config),
            "token_values": list(token_vocabulary.index_to_value),
            "label_values": list(label_vocabulary.index_to_value),
            "data_file_sha256": data_sha256,
            "selection": selection,
        },
        checkpoint_path,
    )
    checkpoint_sha256 = file_digest(checkpoint_path)
    result = {
        "schema_version": SELECTION_RECORD_SCHEMA_VERSION,
        "status": "fresh_train_validation_run",
        "test_partition_used": False,
        "config": asdict(config),
        "data_file_sha256": data_sha256,
        "split_sentences": {name: len(rows) for name, rows in splits.items()},
        "selection": selection,
        "selection_metric": SELECTION_METRIC,
        "best_epoch": best_epoch,
        "best_validation_span_f1": best_validation_f1,
        "checkpoint": {
            "file": CHECKPOINT_FILENAME,
            "sha256": checkpoint_sha256,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
        },
        "history": history,
    }
    (output_dir / SELECTION_RECORD_FILENAME).write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def evaluate_pipeline(
    data_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    selection_metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen, validation-selected checkpoint on the test split."""
    selection_path = selection_metadata_path or checkpoint_path.with_name(SELECTION_RECORD_FILENAME)
    record, config, selection_record_sha256 = _load_selection_record(selection_path)
    if checkpoint_path.name != record["checkpoint"]["file"]:
        raise ValueError("checkpoint filename does not match selection metadata")
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint_sha256 != record["checkpoint"]["sha256"]:
        raise ValueError("checkpoint SHA-256 differs from selection metadata")
    current_digest = file_digest(data_path)
    if record["data_file_sha256"] != current_digest:
        raise ValueError("input data differs from the file used for model selection")

    set_seed(config.seed)
    sentences = read_sentences(data_path)
    if file_digest(data_path) != current_digest:
        raise ValueError("input data changed while it was being read")
    _validate_transformer_lengths(sentences, config)
    payload = _validate_checkpoint_payload(
        torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True),
        record,
    )
    splits = split_by_content_hash(sentences, seed=config.seed)
    token_vocabulary = Vocabulary.from_index_values(payload["token_values"])
    label_vocabulary = Vocabulary.from_index_values(payload["label_values"])
    test_loader = _loader(
        splits["test"],
        token_vocabulary,
        label_vocabulary,
        config,
        shuffle=False,
        seed_offset=2,
    )
    device = select_device()
    model = build_model(config, len(token_vocabulary), len(label_vocabulary)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    criterion = nn.CrossEntropyLoss(ignore_index=LABEL_IGNORE_INDEX)
    test = _run_epoch(model, test_loader, criterion, device, None)
    result = {
        "schema_version": SELECTION_RECORD_SCHEMA_VERSION,
        "status": "fresh_test_evaluation",
        "test_partition_used": True,
        "config": asdict(config),
        "data_file_sha256": current_digest,
        "split_sentences": {name: len(rows) for name, rows in splits.items()},
        "selection": record["selection"],
        "checkpoint": {
            "file": checkpoint_path.name,
            "sha256": checkpoint_sha256,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "selection_metadata_file": selection_path.name,
            "selection_metadata_sha256": selection_record_sha256,
        },
        "test": {"loss": test[0], **evaluate_predictions(*test[1:], label_vocabulary)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
