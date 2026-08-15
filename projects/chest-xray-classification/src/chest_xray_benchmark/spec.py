"""Dataset identity contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ClassSpec:
    name: str
    expected_count: int
    directory_aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    version: int
    doi: str
    expected_total: int
    classes: tuple[ClassSpec, ...]

    @classmethod
    def load(cls, path: Path) -> DatasetSpec:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        classes = tuple(
            ClassSpec(
                name=name,
                expected_count=int(values["expected_count"]),
                directory_aliases=tuple(values["directory_aliases"]),
            )
            for name, values in payload["classes"].items()
        )
        expected_total = int(payload["expected_total"])
        if sum(item.expected_count for item in classes) != expected_total:
            raise ValueError("Class counts do not sum to expected_total")
        aliases = [alias for item in classes for alias in item.directory_aliases]
        if len(aliases) != len(set(aliases)):
            raise ValueError("Directory aliases must be unique")
        return cls(
            name=str(payload["name"]),
            version=int(payload["version"]),
            doi=str(payload["doi"]),
            expected_total=expected_total,
            classes=classes,
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.classes)

    @property
    def expected_counts(self) -> dict[str, int]:
        return {item.name: item.expected_count for item in self.classes}

    @property
    def alias_to_class(self) -> dict[str, str]:
        return {alias: item.name for item in self.classes for alias in item.directory_aliases}
