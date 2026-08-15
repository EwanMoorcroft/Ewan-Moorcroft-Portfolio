"""Spatial weighting and deterministic inference tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liverpool_accessibility.spatial import morans_i


def test_moran_matches_hand_calculated_chain() -> None:
    """The observed statistic should match the explicit row-standardized formula."""
    edges = pd.DataFrame(
        [("A", "B"), ("B", "C"), ("C", "D")],
        columns=["area_code_a", "area_code_b"],
    )
    result = morans_i(["A", "B", "C", "D"], [1.0, 2.0, 4.0, 8.0], edges, permutations=0)
    values = np.asarray([1.0, 2.0, 4.0, 8.0])
    centred = values - values.mean()
    weights = np.asarray(
        [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]],
        dtype=float,
    )
    expected = (
        4 / weights.sum() * (weights * np.outer(centred, centred)).sum() / (centred @ centred)
    )
    assert result.statistic == pytest.approx(expected)


def test_moran_permutation_result_is_repeatable() -> None:
    """A fixed seed must reproduce the same pseudo-p-value."""
    edges = pd.DataFrame(
        [("A", "B"), ("B", "C"), ("C", "D")],
        columns=["area_code_a", "area_code_b"],
    )
    first = morans_i(["A", "B", "C", "D"], [1.0, 2.0, 4.0, 8.0], edges, permutations=99, seed=7)
    second = morans_i(["A", "B", "C", "D"], [1.0, 2.0, 4.0, 8.0], edges, permutations=99, seed=7)
    assert first == second
