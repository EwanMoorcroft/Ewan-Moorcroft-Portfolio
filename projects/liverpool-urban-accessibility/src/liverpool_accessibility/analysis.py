"""Public-data transformation for Liverpool commuting-accessibility evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from .contracts import (
    DataContractError,
    read_boundaries,
    read_centroids,
    validate_flow_header,
    validate_metrics,
)
from .evidence import verify_source_manifest
from .spatial import queen_edges

WGS84_TO_BNG = (
    "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
    "+step +proj=push +v_3 +step +proj=cart +ellps=WGS84 "
    "+step +inv +proj=helmert +x=446.448 +y=-125.157 +z=542.06 "
    "+rx=0.15 +ry=0.247 +rz=0.842 +s=-20.489 +convention=position_vector "
    "+step +inv +proj=cart +ellps=airy +step +proj=pop +v_3 "
    "+step +proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 "
    "+x_0=400000 +y_0=-100000 +ellps=airy"
)


@dataclass(frozen=True)
class PreparedEvidence:
    """Canonical derived tables written by one preparation run."""

    source_contract: str
    evidence_scope: str
    metrics: pd.DataFrame
    edges: pd.DataFrame
    boundaries: gpd.GeoDataFrame
    centroids: gpd.GeoDataFrame


def _sql_path(path: str | Path) -> str:
    resolved = Path(path).resolve(strict=True).as_posix()
    return "'" + resolved.replace("'", "''") + "'"


def _codes_table(boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
    return boundaries[["MSOA21CD"]].rename(columns={"MSOA21CD": "area_code"})


def _audit_flow_rows(connection: duckdb.DuckDBPyConnection, flow_path: str) -> None:
    """Reject malformed counts, duplicate keys, and inconsistent Liverpool origins."""
    invalid_rows = connection.execute(
        f"""
        SELECT count(*)
        FROM read_csv_auto({flow_path}, header = true, sample_size = -1)
        WHERE "Middle layer Super Output Areas code" IS NULL
           OR "MSOA of workplace code" IS NULL
           OR try_cast("Place of work indicator (4 categories) code" AS INTEGER) IS NULL
           OR "Count" IS NULL
           OR NOT regexp_full_match(CAST("Count" AS VARCHAR), '[0-9]+')
        """
    ).fetchone()[0]
    if invalid_rows:
        raise DataContractError(f"flow input contains {invalid_rows} invalid row(s)")

    duplicate_keys = connection.execute(
        f"""
        SELECT count(*)
        FROM (
            SELECT
                "Middle layer Super Output Areas code",
                "MSOA of workplace code",
                "Place of work indicator (4 categories) code",
                count(*) AS rows_per_key
            FROM read_csv_auto({flow_path}, header = true, sample_size = -1)
            GROUP BY 1, 2, 3
            HAVING rows_per_key > 1
        )
        """
    ).fetchone()[0]
    if duplicate_keys:
        raise DataContractError(f"flow input contains {duplicate_keys} duplicate key(s)")

    inconsistent_origins = connection.execute(
        f"""
        SELECT count(*)
        FROM (
            SELECT
                f."Middle layer Super Output Areas code" AS origin_code,
                count(DISTINCT f."Middle layer Super Output Areas label") AS labels,
                count(DISTINCT CASE
                    WHEN try_cast(f."Place of work indicator (4 categories) code" AS INTEGER)
                         IN (1, 2, 3)
                    THEN try_cast(f."Place of work indicator (4 categories) code" AS INTEGER)
                END) AS required_indicators
            FROM read_csv_auto({flow_path}, header = true, sample_size = -1) AS f
            INNER JOIN area_codes AS a
              ON CAST(f."Middle layer Super Output Areas code" AS VARCHAR) = a.area_code
            GROUP BY origin_code
            HAVING labels != 1 OR required_indicators != 3
        )
        """
    ).fetchone()[0]
    if inconsistent_origins:
        raise DataContractError(
            f"flow input contains {inconsistent_origins} inconsistent Liverpool origin(s)"
        )


def _origin_aggregates(connection: duckdb.DuckDBPyConnection, flow_path: str) -> pd.DataFrame:
    return connection.execute(
        f"""
        WITH flows AS (
            SELECT
                CAST("Middle layer Super Output Areas code" AS VARCHAR) AS origin_code,
                CAST("Middle layer Super Output Areas label" AS VARCHAR) AS origin_name,
                CAST("MSOA of workplace code" AS VARCHAR) AS destination_code,
                CAST("Place of work indicator (4 categories) code" AS INTEGER) AS indicator,
                CAST("Count" AS BIGINT) AS flow_count
            FROM read_csv_auto({flow_path}, header = true, sample_size = -1)
        )
        SELECT
            f.origin_code AS area_code,
            min(f.origin_name) AS area_name,
            sum(CASE WHEN f.indicator IN (1, 2, 3) THEN f.flow_count ELSE 0 END) AS employed_total,
            sum(CASE WHEN f.indicator = 1 THEN f.flow_count ELSE 0 END) AS home_or_no_fixed,
            sum(CASE WHEN f.indicator = 2 THEN f.flow_count ELSE 0 END) AS other_workplace,
            sum(CASE WHEN f.indicator = 3 THEN f.flow_count ELSE 0 END) AS fixed_workplace,
            sum(CASE WHEN f.indicator = 3 AND d.area_code IS NOT NULL THEN f.flow_count ELSE 0 END) AS local_fixed,
            sum(CASE WHEN f.indicator = 3 AND f.destination_code = f.origin_code THEN f.flow_count ELSE 0 END) AS same_area_fixed
        FROM flows AS f
        INNER JOIN area_codes AS o ON f.origin_code = o.area_code
        LEFT JOIN area_codes AS d ON f.destination_code = d.area_code
        GROUP BY f.origin_code
        ORDER BY f.origin_code
        """
    ).fetchdf()


def _destination_flow_mass(connection: duckdb.DuckDBPyConnection, flow_path: str) -> pd.DataFrame:
    return connection.execute(
        f"""
        WITH flows AS (
            SELECT
                CAST("MSOA of workplace code" AS VARCHAR) AS destination_code,
                CAST("Place of work indicator (4 categories) code" AS INTEGER) AS indicator,
                CAST("Count" AS BIGINT) AS flow_count
            FROM read_csv_auto({flow_path}, header = true, sample_size = -1)
        )
        SELECT
            d.area_code,
            coalesce(sum(f.flow_count), 0) AS inbound_fixed_workers
        FROM area_codes AS d
        LEFT JOIN flows AS f
          ON f.destination_code = d.area_code AND f.indicator = 3
        GROUP BY d.area_code
        ORDER BY d.area_code
        """
    ).fetchdf()


def _local_flows(connection: duckdb.DuckDBPyConnection, flow_path: str) -> pd.DataFrame:
    return connection.execute(
        f"""
        WITH flows AS (
            SELECT
                CAST("Middle layer Super Output Areas code" AS VARCHAR) AS origin_code,
                CAST("MSOA of workplace code" AS VARCHAR) AS destination_code,
                CAST("Place of work indicator (4 categories) code" AS INTEGER) AS indicator,
                CAST("Count" AS BIGINT) AS flow_count
            FROM read_csv_auto({flow_path}, header = true, sample_size = -1)
        )
        SELECT f.origin_code, f.destination_code, f.flow_count
        FROM flows AS f
        INNER JOIN area_codes AS o ON f.origin_code = o.area_code
        INNER JOIN area_codes AS d ON f.destination_code = d.area_code
        WHERE f.indicator = 3
        ORDER BY f.origin_code, f.destination_code
        """
    ).fetchdf()


def _distance_matrix(centroids: gpd.GeoDataFrame) -> tuple[list[str], np.ndarray]:
    longitude = centroids.geometry.x.to_numpy(dtype=float)
    latitude = centroids.geometry.y.to_numpy(dtype=float)
    transformer = Transformer.from_pipeline(WGS84_TO_BNG)
    east, north = transformer.transform(longitude, latitude)
    coordinates = np.column_stack([east, north])
    if not np.isfinite(coordinates).all():
        raise DataContractError("EPSG:4326 to EPSG:27700 transformation returned non-finite values")
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    return centroids["MSOA21CD"].astype(str).tolist(), np.sqrt(np.square(delta).sum(axis=2))


def _add_accessibility(
    metrics: pd.DataFrame,
    local_flows: pd.DataFrame,
    centroids: gpd.GeoDataFrame,
) -> pd.DataFrame:
    codes, distances = _distance_matrix(centroids)
    positions = {code: index for index, code in enumerate(codes)}
    aligned = metrics.set_index("area_code").reindex(codes)
    if aligned.isna().any().any():
        raise DataContractError("flow and boundary area identities do not align")
    flow_mass = aligned["inbound_fixed_workers"].to_numpy(dtype=float)
    for decay_km in (3, 5, 10):
        aligned[f"accessibility_{decay_km}km"] = (
            np.exp(-distances / (decay_km * 1000.0)) @ flow_mass
        )

    weighted_distance = {code: 0.0 for code in codes}
    weighted_count = {code: 0 for code in codes}
    for row in local_flows.itertuples(index=False):
        origin = str(row.origin_code)
        destination = str(row.destination_code)
        count = int(row.flow_count)
        weighted_distance[origin] += distances[positions[origin], positions[destination]] * count
        weighted_count[origin] += count
    aligned["mean_local_commute_km"] = [
        weighted_distance[code] / weighted_count[code] / 1000.0 if weighted_count[code] else 0.0
        for code in codes
    ]
    return aligned.reset_index()


def prepare_evidence(
    flow_csv: str | Path,
    boundaries_path: str | Path,
    centroids_path: str | Path,
    *,
    source_manifest: dict[str, object] | None = None,
) -> PreparedEvidence:
    """Transform authenticated official inputs or explicitly fictional fixture inputs."""
    if source_manifest is None:
        raise DataContractError("an authenticated source manifest is required for preparation")
    source_identity = verify_source_manifest(
        source_manifest,
        {
            "flow_csv": flow_csv,
            "boundaries": boundaries_path,
            "centroids": centroids_path,
        },
    )
    validate_flow_header(flow_csv)
    boundaries = read_boundaries(boundaries_path)
    centroids = read_centroids(
        centroids_path,
        expected_codes=set(boundaries["MSOA21CD"].astype(str)),
    )
    connection = duckdb.connect()
    try:
        connection.register("area_codes", _codes_table(boundaries))
        path_literal = _sql_path(flow_csv)
        _audit_flow_rows(connection, path_literal)
        metrics = _origin_aggregates(connection, path_literal)
        destination_mass = _destination_flow_mass(connection, path_literal)
        local_flows = _local_flows(connection, path_literal)
    finally:
        connection.close()

    if len(metrics) != len(boundaries):
        raise DataContractError("not every boundary area has a flow origin row")
    metrics = metrics.merge(destination_mass, on="area_code", validate="one_to_one")
    count_columns = [
        "employed_total",
        "home_or_no_fixed",
        "other_workplace",
        "fixed_workplace",
        "local_fixed",
        "same_area_fixed",
        "inbound_fixed_workers",
    ]
    metrics[count_columns] = metrics[count_columns].astype("int64")
    metrics["outside_liverpool_fixed"] = metrics["fixed_workplace"] - metrics["local_fixed"]
    metrics["local_retention_share"] = metrics["local_fixed"] / metrics["fixed_workplace"]
    metrics["home_or_no_fixed_share"] = metrics["home_or_no_fixed"] / metrics["employed_total"]
    metrics = _add_accessibility(metrics, local_flows, centroids)
    order = [
        "area_code",
        "area_name",
        "employed_total",
        "home_or_no_fixed",
        "other_workplace",
        "fixed_workplace",
        "local_fixed",
        "same_area_fixed",
        "outside_liverpool_fixed",
        "local_retention_share",
        "home_or_no_fixed_share",
        "inbound_fixed_workers",
        "accessibility_3km",
        "accessibility_5km",
        "accessibility_10km",
        "mean_local_commute_km",
    ]
    metrics = metrics[order].sort_values("area_code").reset_index(drop=True)
    validate_metrics(metrics)
    edges = queen_edges(boundaries)
    if not set(metrics["area_code"]).issubset(
        set(edges["area_code_a"]) | set(edges["area_code_b"])
    ):
        raise DataContractError("Queen-contiguity graph contains at least one island")
    return PreparedEvidence(
        source_contract=str(source_identity["contract"]),
        evidence_scope=str(source_identity["evidence_scope"]),
        metrics=metrics,
        edges=edges,
        boundaries=boundaries,
        centroids=centroids,
    )


def write_prepared(evidence: PreparedEvidence, output_dir: str | Path) -> list[Path]:
    """Write stable analysis inputs."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "area-metrics.csv"
    edges_path = destination / "spatial-edges.csv"
    boundaries_path = destination / "liverpool-msoa-boundaries.geojson"
    centroids_path = destination / "liverpool-msoa-centroids.geojson"
    evidence.metrics.to_csv(metrics_path, index=False, float_format="%.12g", lineterminator="\n")
    evidence.edges.to_csv(edges_path, index=False, lineterminator="\n")
    evidence.boundaries.to_file(boundaries_path, driver="GeoJSON")
    evidence.centroids.to_file(centroids_path, driver="GeoJSON")
    return [metrics_path, edges_path, boundaries_path, centroids_path]
