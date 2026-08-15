"""Transparent spatial weighting and autocorrelation routines."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd

from .contracts import DataContractError


@dataclass(frozen=True)
class MoranResult:
    """Observed Moran statistic and deterministic permutation evidence."""

    statistic: float
    pseudo_p_value: float
    permutations: int
    seed: int


def queen_edges(boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return sorted undirected Queen-contiguity edges without self-links."""
    records: list[tuple[str, str]] = []
    rows = list(boundaries[["MSOA21CD", "geometry"]].itertuples(index=False, name=None))
    for index, (left_code, left_geometry) in enumerate(rows):
        for right_code, right_geometry in rows[index + 1 :]:
            if left_geometry.touches(right_geometry):
                records.append((str(left_code), str(right_code)))
    return (
        pd.DataFrame(records, columns=["area_code_a", "area_code_b"])
        .sort_values(["area_code_a", "area_code_b"])
        .reset_index(drop=True)
    )


def _row_standardized_matrix(codes: list[str], edges: pd.DataFrame) -> np.ndarray:
    index = {code: position for position, code in enumerate(codes)}
    if len(index) != len(codes):
        raise DataContractError("Moran area codes must be unique")
    matrix = np.zeros((len(codes), len(codes)), dtype=float)
    for row in edges.itertuples(index=False):
        left, right = str(row.area_code_a), str(row.area_code_b)
        if left == right or left not in index or right not in index:
            raise DataContractError("spatial edge contains an invalid endpoint")
        matrix[index[left], index[right]] = 1.0
        matrix[index[right], index[left]] = 1.0
    degrees = matrix.sum(axis=1)
    if (degrees == 0).any():
        islands = [codes[position] for position in np.flatnonzero(degrees == 0)]
        raise DataContractError(f"spatial weights contain islands: {', '.join(islands)}")
    return matrix / degrees[:, None]


def morans_i(
    codes: Iterable[str],
    values: Iterable[float],
    edges: pd.DataFrame,
    *,
    permutations: int = 999,
    seed: int = 2026,
) -> MoranResult:
    """Compute row-standardized Moran's I with a fixed two-sided permutation test."""
    code_list = [str(code) for code in codes]
    vector = np.asarray(list(values), dtype=float)
    if len(code_list) != len(vector) or len(vector) < 3:
        raise DataContractError("Moran inputs must contain at least three aligned areas")
    if not np.isfinite(vector).all() or np.allclose(vector, vector[0]):
        raise DataContractError("Moran values must be finite and non-constant")
    if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 0:
        raise DataContractError("permutations must be a non-negative integer")
    weights = _row_standardized_matrix(code_list, edges)
    centred = vector - vector.mean()
    denominator = float(centred @ centred)

    def statistic(candidate: np.ndarray) -> float:
        return float(
            (len(candidate) / weights.sum())
            * ((weights * np.outer(candidate, candidate)).sum() / denominator)
        )

    observed = statistic(centred)
    expected = -1.0 / (len(vector) - 1)
    if permutations == 0:
        return MoranResult(observed, float("nan"), permutations, seed)
    generator = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted = generator.permutation(centred)
        if abs(statistic(permuted) - expected) >= abs(observed - expected):
            exceedances += 1
    return MoranResult(observed, (exceedances + 1) / (permutations + 1), permutations, seed)
