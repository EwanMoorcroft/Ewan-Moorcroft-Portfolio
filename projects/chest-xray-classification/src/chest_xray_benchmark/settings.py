"""Validated run settings."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunSettings:
    seed: int = 534
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 20
    early_stopping_patience: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.30
    rotation_degrees: float = 5.0
    pretrained: bool = True
    device: str = "auto"

    @classmethod
    def load(cls, path: Path) -> RunSettings:
        payload: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(unknown)}")
        settings = cls(**payload)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.image_size < 64 or self.batch_size < 1:
            raise ValueError("image_size and batch_size are too small")
        if self.num_workers < 0 or self.epochs < 1 or self.early_stopping_patience < 1:
            raise ValueError("worker, epoch and patience values must be valid")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings must be non-negative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.rotation_degrees <= 15:
            raise ValueError("rotation_degrees must be in [0, 15]")
        if self.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps or cuda")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
