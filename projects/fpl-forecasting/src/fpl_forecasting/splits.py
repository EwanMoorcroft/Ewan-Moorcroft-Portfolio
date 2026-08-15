"""Expanding-window splits that preserve gameweek order."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from .errors import SplitError


@dataclass(frozen=True)
class RollingFold:
    """Row indices and gameweek boundaries for one rolling-origin fold."""

    number: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_gameweeks: tuple[int, ...]
    test_gameweeks: tuple[int, ...]

    @property
    def train_end(self) -> int:
        return self.train_gameweeks[-1]

    @property
    def test_start(self) -> int:
        return self.test_gameweeks[0]


def rolling_origin_splits(
    frame: pd.DataFrame,
    *,
    minimum_train_gameweeks: int,
    test_gameweeks_per_fold: int = 1,
    step: int = 1,
) -> Iterator[RollingFold]:
    """Yield expanding train windows followed by untouched future windows."""

    if "target_gw" not in frame.columns:
        raise SplitError("Frame must contain target_gw")
    parameters = {
        "minimum_train_gameweeks": minimum_train_gameweeks,
        "test_gameweeks_per_fold": test_gameweeks_per_fold,
        "step": step,
    }
    for name, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise SplitError(f"{name} must be an integer")
    if minimum_train_gameweeks < 2:
        raise SplitError("minimum_train_gameweeks must be at least 2")
    if test_gameweeks_per_fold < 1 or step < 1:
        raise SplitError("test_gameweeks_per_fold and step must be positive")
    if step < test_gameweeks_per_fold:
        raise SplitError("step must prevent overlapping test windows")

    values = pd.to_numeric(frame["target_gw"], errors="coerce")
    if values.isna().any():
        raise SplitError("target_gw must be numeric")
    numeric_values = values.to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise SplitError("target_gw must be finite")
    if (
        not np.equal(numeric_values, np.floor(numeric_values)).all()
        or not (numeric_values > 0).all()
    ):
        raise SplitError("target_gw must contain positive integers")
    gameweeks = tuple(sorted(int(value) for value in values.unique()))
    required = minimum_train_gameweeks + test_gameweeks_per_fold
    if len(gameweeks) < required:
        raise SplitError(f"Need at least {required} target gameweeks, found {len(gameweeks)}")

    fold_number = 0
    test_offset = minimum_train_gameweeks
    while test_offset + test_gameweeks_per_fold <= len(gameweeks):
        train_gameweeks = gameweeks[:test_offset]
        test_gameweeks = gameweeks[test_offset : test_offset + test_gameweeks_per_fold]
        train_mask = values.isin(train_gameweeks).to_numpy()
        test_mask = values.isin(test_gameweeks).to_numpy()
        train_indices = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        if train_indices.size == 0 or test_indices.size == 0:
            raise SplitError("A chronological fold has no rows")
        if max(train_gameweeks) >= min(test_gameweeks):
            raise SplitError("Training gameweeks must precede test gameweeks")
        yield RollingFold(
            number=fold_number,
            train_indices=train_indices,
            test_indices=test_indices,
            train_gameweeks=train_gameweeks,
            test_gameweeks=test_gameweeks,
        )
        fold_number += 1
        test_offset += step
