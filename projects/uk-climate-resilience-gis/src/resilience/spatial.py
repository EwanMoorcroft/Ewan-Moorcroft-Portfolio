"""One predeclared global spatial-association test and a topology sensitivity.

Queen is the primary weights definition; rook is a declared sensitivity check.
Returned quadrants describe a score and its spatial lag, not significant local
clusters. No local p-values or multiple-testing claims are produced.
"""

from __future__ import annotations

from numbers import Integral

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
import shapely


NULL = (
    "Conditional on the complete finite-score cohort, its fixed score multiset, "
    "and the fixed spatial weights matrix, all assignments of score labels to "
    "eligible neighbourhoods are exchangeable. Each draw uniformly permutes "
    "those labels without replacement. Isolated units remain in that population "
    "with zero-weight rows. This is a spatial-randomness null, not a causal test."
)
TAIL_TOLERANCE = 1e-12
AXIS_TOLERANCE = 1e-12


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _centred(values: np.ndarray) -> np.ndarray:
    if not len(values):
        return values.copy()
    scale = np.max(np.abs(values))
    scaled = values / scale if scale else values.copy()
    return scaled - scaled.mean()


def global_moran(values: np.ndarray, weights, permutations: int, seed: int,
                 chunk_size: int = 128) -> dict:
    """Calculate Global Moran's I for a supplied nonnegative zero-diagonal W.

    W is used as supplied; the spatial wrapper row-standardises its adjacency.
    I = (n/S0) * (z.T @ W @ z) / (z.T @ z), where z is centred. Under total
    label permutation with a zero diagonal, E[I] = -1/(n-1). The two-sided
    pseudo p-value is (1 + number of simulated absolute deviations from E[I]
    at least as large as the observed deviation) / (permutations + 1).

    Two dense chunks of at most ``chunk_size * n`` elements are sufficient;
    only the one-dimensional statistic distribution is retained. A chunk size
    does not change the random-permutation stream. Undefined tests return None
    and an explicit status, never fabricated zero statistics or p-values.
    """
    permutations = _integer(permutations, "permutations", 1)
    seed = _integer(seed, "seed", 0)
    chunk_size = _integer(chunk_size, "chunk_size", 1)
    try:
        values = np.asarray(values, dtype=float)
        weights = sparse.csr_matrix(weights, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Values and weights must be numeric") from exc
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("values must be one finite numeric vector")
    n = len(values)
    if weights.shape != (n, n):
        raise ValueError("Spatial weights must be square and aligned with values")
    if not np.isfinite(weights.data).all() or (weights.data < 0).any():
        raise ValueError("Spatial weights must be finite and nonnegative")
    if np.any(weights.diagonal() != 0):
        raise ValueError("The specified randomization expectation requires zero self-weights")
    weights.eliminate_zeros()
    weights.sort_indices()
    s0 = float(weights.sum())
    result = {
        "status": "ok", "n": n, "s0": s0,
        "statistic": None, "expected": None,
        "permutation_mean": None, "permutation_sd": None,
        "p_two_sided": None, "extreme_permutations": None,
        "permutations_requested": permutations, "permutations_run": 0,
        "seed": seed, "generator": "PCG64",
        "permutation_statistics": [],
        "null": NULL,
        "statistic_definition": "(n / S0) * sum_ij(w_ij * z_i * z_j) / sum_i(z_i**2), with centred z",
        "expectation_definition": "-1 / (n - 1), for zero-diagonal weights under total label permutation",
        "p_definition": "(1 + count(abs(I_permuted - E[I]) >= abs(I_observed - E[I]))) / (B + 1)",
        "tail_equality_tolerance": {"absolute": TAIL_TOLERANCE, "relative": TAIL_TOLERANCE},
        "permutation_sd_ddof": 1,
    }
    if n < 2:
        result["status"] = "insufficient_units"
        return result
    if s0 == 0:
        result["status"] = "no_neighbours"
        return result
    centred = _centred(values)
    denominator = float(centred @ centred)
    if denominator == 0:
        result["status"] = "zero_variance"
        return result

    multiplier = n / s0 / denominator
    observed = float(multiplier * (centred @ (weights @ centred)))
    expected = -1 / (n - 1)
    rng = np.random.default_rng(seed)
    distribution = np.empty(permutations)
    for start in range(0, permutations, chunk_size):
        size = min(chunk_size, permutations - start)
        draws = np.vstack([rng.permutation(centred) for _ in range(size)])
        lag = (weights @ draws.T).T
        distribution[start:start + size] = multiplier * np.einsum("ij,ij->i", draws, lag)
    distance = np.abs(distribution - expected)
    observed_distance = abs(observed - expected)
    # Include numerical ties: round-off must not turn an attainable identical
    # statistic into a falsely more extreme observation.
    extreme = (distance > observed_distance) | np.isclose(
        distance, observed_distance, atol=TAIL_TOLERANCE, rtol=TAIL_TOLERANCE)
    count = int(extreme.sum())
    result.update({
        "statistic": observed, "expected": expected,
        "permutation_mean": float(distribution.mean()),
        "permutation_sd": float(distribution.std(ddof=1)) if permutations > 1 else None,
        "p_two_sided": (count + 1) / (permutations + 1),
        "extreme_permutations": count, "permutations_run": permutations,
        "generator": type(rng.bit_generator).__name__,
        "permutation_statistics": distribution.tolist(),
    })
    return result


def _contiguity(geometries: np.ndarray) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Exact boundary contacts, with no snapping, buffering or island links."""
    n = len(geometries)
    if not n:
        empty = sparse.csr_matrix((0, 0), dtype=float)
        return empty, empty.copy()
    tree = shapely.STRtree(geometries)
    pairs = tree.query(geometries, predicate="intersects")
    left, right = pairs[:, pairs[0] < pairs[1]]
    if len(left):
        touches = shapely.touches(geometries[left], geometries[right])
        if not touches.all():
            raise ValueError("Scored neighbourhood interiors overlap; repair the source topology before spatial analysis")
        shared_boundaries = shapely.intersection(
            shapely.boundary(geometries[left]), shapely.boundary(geometries[right]))
        rook = shapely.length(shared_boundaries) > 0
    else:
        rook = np.zeros(0, dtype=bool)

    def matrix(a, b):
        row = np.concatenate([a, b])
        column = np.concatenate([b, a])
        return sparse.csr_matrix((np.ones(len(row)), (row, column)), shape=(n, n))

    return matrix(left, right), matrix(left[rook], right[rook])


def _row_standardise(adjacency: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros(len(degree))
    np.divide(1, degree, out=inverse, where=degree > 0)
    return (sparse.diags(inverse) @ adjacency).tocsr(), degree


def spatial_diagnostics(neighbourhoods: gpd.GeoDataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Return keyed descriptive quadrants and primary/sensitivity global tests.

    Geometry must be valid, nonempty polygonal geometry for each finite scored
    and, if supplied, model-eligible row. It is never silently repaired. Excluded
    rows remain in the returned table with null spatial outputs. All operations
    use a stable ID order, while output preserves the input order and index.

    Isolated units remain in n, centring, variance and the permutation cohort;
    their zero-weight rows contribute nothing to S0 or the spatial numerator.
    Their displayed spatial lag is null. Exact mean-axis values have a null
    quadrant: high and low are strict comparisons, within the stated tolerance.
    """
    try:
        permutations = _integer(config["moran_permutations"], "moran_permutations", 1)
        seed = _integer(config["seed"], "seed", 0)
    except (KeyError, TypeError) as exc:
        raise ValueError("Configuration requires moran_permutations and seed") from exc
    if not isinstance(neighbourhoods, gpd.GeoDataFrame) or neighbourhoods.columns.duplicated().any():
        raise ValueError("Input must be a GeoDataFrame with unique column names")
    if any(column not in neighbourhoods for column in ["lsoa21cd", "score_balanced"]):
        raise ValueError("Input requires lsoa21cd and score_balanced")
    try:
        all_geometries = neighbourhoods.geometry.to_numpy()
    except AttributeError as exc:
        raise ValueError("Input requires an active geometry column") from exc
    ids = neighbourhoods["lsoa21cd"].to_numpy()
    if any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in ids):
        raise ValueError("lsoa21cd must contain nonempty, whitespace-free strings")
    if neighbourhoods["lsoa21cd"].duplicated().any():
        raise ValueError("lsoa21cd must be unique")
    scores = pd.to_numeric(neighbourhoods["score_balanced"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan)
    eligible = np.isfinite(scores)
    if "model_eligible" in neighbourhoods:
        eligible &= neighbourhoods["model_eligible"].eq(True).fillna(False).to_numpy(dtype=bool)
    positions = np.flatnonzero(eligible)
    positions = positions[np.argsort(ids[positions], kind="stable")]
    cohort_ids = ids[positions].tolist()
    cohort_scores = scores[positions]
    geometries = all_geometries[positions]
    valid_geometry = (
        ~shapely.is_missing(geometries) & ~shapely.is_empty(geometries)
        & shapely.is_valid(geometries)
        & np.isin(shapely.get_type_id(geometries), [3, 6])
    )
    if not valid_geometry.all():
        invalid_ids = [area for area, valid in zip(cohort_ids, valid_geometry) if not valid]
        raise ValueError("Scored units require valid nonempty Polygon/MultiPolygon geometry: "
                         + ", ".join(invalid_ids[:10]))

    queen_adjacency, rook_adjacency = _contiguity(geometries)
    queen, queen_degree = _row_standardise(queen_adjacency)
    rook, rook_degree = _row_standardise(rook_adjacency)
    n = len(positions)

    def test(weights, adjacency, degree, definition):
        output = global_moran(cohort_scores, weights, permutations, seed)
        output.update({
            "weights_definition": definition,
            "standardisation": "Non-isolated row weights sum to one; island rows sum to zero; diagonal zero",
            "island_ids": [area for area, count in zip(cohort_ids, degree) if count == 0],
            "island_count": int(np.count_nonzero(degree == 0)),
            "undirected_edges": int(adjacency.nnz // 2),
            "neighbour_counts": {area: int(count) for area, count in zip(cohort_ids, degree)},
            "row_weight_sums": {area: float(value) for area, value in zip(
                cohort_ids, np.asarray(weights.sum(axis=1)).ravel())},
            "id_order": cohort_ids,
        })
        return output

    primary = test(queen, queen_adjacency, queen_degree,
                   "Queen: exact shared polygon boundary point or line; no buffers, snapping or added links")
    sensitivity = test(rook, rook_adjacency, rook_degree,
                       "Rook: exact shared polygon boundary of strictly positive length; no buffers, snapping or added links")
    centred = _centred(cohort_scores)
    variance = float(np.mean(centred ** 2)) if n else 0
    z = centred / np.sqrt(variance) if variance > 0 else np.full(n, np.nan)
    lag = np.asarray(queen @ z).ravel()
    islands = queen_degree == 0
    lag[islands] = np.nan
    quadrants = np.full(n, None, dtype=object)
    quadrants[islands] = "isolated"
    finite_lag = np.isfinite(z) & np.isfinite(lag) & ~islands
    on_axis = finite_lag & ((np.abs(z) <= AXIS_TOLERANCE) | (np.abs(lag) <= AXIS_TOLERANCE))
    strict = finite_lag & ~on_axis
    for label, condition in {
        "HH": (z > 0) & (lag > 0), "HL": (z > 0) & (lag < 0),
        "LH": (z < 0) & (lag > 0), "LL": (z < 0) & (lag < 0),
    }.items():
        quadrants[strict & condition] = label

    table = pd.DataFrame({"lsoa21cd": ids, "spatial_eligible": eligible}, index=neighbourhoods.index)
    for name, values in [("z_score", z), ("spatial_lag_z", lag)]:
        output = np.full(len(neighbourhoods), np.nan)
        output[positions] = values
        table[name] = output
    quadrant_output = np.full(len(neighbourhoods), None, dtype=object)
    quadrant_output[positions] = quadrants
    table["quadrant"] = quadrant_output
    diagnostics = {
        "cohort": {
            "input_count": len(neighbourhoods), "eligible_count": n,
            "excluded_count": len(neighbourhoods) - n,
            "eligible_ids_sorted": cohort_ids,
            "excluded_ids_sorted": sorted(ids[~eligible].tolist()),
            "rule": "Finite score_balanced and, where supplied, model_eligible=True",
            "island_rule": "Retain islands in n, centring, variance and label permutations; zero-weight rows and no added links",
            "crs": str(neighbourhoods.crs) if neighbourhoods.crs is not None else None,
        },
        "primary": primary,
        "rook_sensitivity": sensitivity,
        "sensitivity_rule": "Predeclared rook check on the identical cohort using the same seeded permutation stream; no selection of the smaller p-value",
        "inference_scope": "One primary global test; rook is a sensitivity analysis. No local p-values, local significance labels, or FDR claims",
        "quadrants": {
            "weights": "Primary row-standardised queen",
            "z_definition": "Population-standardised score within the complete spatial cohort (ddof=0)",
            "lag_definition": "Weighted mean of neighbouring z scores; null for islands",
            "interpretation": "Descriptive sign of z_score and spatial_lag_z relative to the cohort mean, not a local significance result",
            "axis_rule": "Null quadrant if z_score or its spatial lag is within the stated tolerance of zero",
            "axis_tolerance": AXIS_TOLERANCE,
            "axis_count": int(on_axis.sum()),
            "counts": {label: int(np.count_nonzero(quadrants == label))
                       for label in ["HH", "HL", "LH", "LL", "isolated"]},
        },
    }
    return table, diagnostics
