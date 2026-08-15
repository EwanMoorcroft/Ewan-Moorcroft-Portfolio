"""Fail-closed contracts for public spatial and flow inputs."""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

AREA_CODE = re.compile(r"E020\d{5}")
FLOW_COLUMNS = {
    "Middle layer Super Output Areas code",
    "Middle layer Super Output Areas label",
    "MSOA of workplace code",
    "MSOA of workplace label",
    "Place of work indicator (4 categories) code",
    "Place of work indicator (4 categories) label",
    "Count",
}
AREA_COLUMNS = {"MSOA21CD", "MSOA21NM", "geometry"}
CENTROID_COLUMNS = {"MSOA21CD", "geometry"}
METRIC_COLUMNS = {
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
}


class DataContractError(ValueError):
    """Raised when a source or derived table violates its declared contract."""


def validate_flow_header(path: str | Path) -> None:
    """Require the exact Census flow fields used by the transformation."""
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(FLOW_COLUMNS - columns)
    if missing:
        raise DataContractError(f"flow input is missing columns: {', '.join(missing)}")


def read_boundaries(path: str | Path) -> gpd.GeoDataFrame:
    """Read and validate one EPSG:4326 Liverpool MSOA boundary layer."""
    frame = gpd.read_file(path)
    missing = sorted(AREA_COLUMNS - set(frame.columns))
    if missing:
        raise DataContractError(f"boundary input is missing columns: {', '.join(missing)}")
    if frame.crs is None or frame.crs.to_epsg() != 4326:
        raise DataContractError("boundary input must declare EPSG:4326")
    if frame.empty:
        raise DataContractError("boundary input is empty")
    if frame["MSOA21CD"].duplicated().any():
        raise DataContractError("boundary area codes must be unique")
    if not frame["MSOA21CD"].map(lambda value: bool(AREA_CODE.fullmatch(str(value)))).all():
        raise DataContractError("boundary input contains an invalid MSOA code")
    if frame.geometry.is_empty.any() or frame.geometry.isna().any():
        raise DataContractError("boundary input contains empty geometry")
    if not frame.geometry.is_valid.all():
        raise DataContractError("boundary input contains invalid geometry")
    if not frame.geometry.geom_type.isin({"Polygon", "MultiPolygon"}).all():
        raise DataContractError("boundary input must contain polygon geometry")
    return (
        frame[["MSOA21CD", "MSOA21NM", "geometry"]].sort_values("MSOA21CD").reset_index(drop=True)
    )


def read_centroids(path: str | Path, *, expected_codes: set[str]) -> gpd.GeoDataFrame:
    """Read population-weighted centroids and require exact boundary alignment."""
    frame = gpd.read_file(path)
    missing = sorted(CENTROID_COLUMNS - set(frame.columns))
    if missing:
        raise DataContractError(f"centroid input is missing columns: {', '.join(missing)}")
    if frame.crs is None or frame.crs.to_epsg() != 4326:
        raise DataContractError("centroid input must declare EPSG:4326")
    if (
        frame["MSOA21CD"].duplicated().any()
        or frame.geometry.is_empty.any()
        or frame.geometry.isna().any()
    ):
        raise DataContractError("centroids must contain unique non-empty points")
    if not frame.geometry.geom_type.eq("Point").all():
        raise DataContractError("centroid input must contain point geometry")
    codes = set(frame["MSOA21CD"].astype(str))
    if codes != expected_codes:
        raise DataContractError("centroid and boundary area identities do not align")
    return frame[["MSOA21CD", "geometry"]].sort_values("MSOA21CD").reset_index(drop=True)


def validate_metrics(frame: pd.DataFrame) -> None:
    """Validate the compact area-level table used by models and verification."""
    missing = sorted(METRIC_COLUMNS - set(frame.columns))
    if missing:
        raise DataContractError(f"area metrics are missing columns: {', '.join(missing)}")
    if frame.empty or frame["area_code"].duplicated().any():
        raise DataContractError("area metrics must contain unique rows")
    if not frame["area_code"].map(lambda value: bool(AREA_CODE.fullmatch(str(value)))).all():
        raise DataContractError("area metrics contain an invalid MSOA code")
    numeric = sorted(METRIC_COLUMNS - {"area_code", "area_name"})
    numeric_frame = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if (numeric_frame.isna() & frame[numeric].notna()).any().any():
        raise DataContractError("area metrics contain a non-numeric value")
    if not np.isfinite(numeric_frame.to_numpy(dtype=float)).all():
        raise DataContractError("area metrics contain a non-finite value")
    if (numeric_frame < 0).any().any():
        raise DataContractError("area metrics contain a negative value")
    if (numeric_frame[["employed_total", "fixed_workplace"]] <= 0).any().any():
        raise DataContractError("area metric denominators must be positive")
    if not (
        numeric_frame["employed_total"]
        == numeric_frame["home_or_no_fixed"]
        + numeric_frame["other_workplace"]
        + numeric_frame["fixed_workplace"]
    ).all():
        raise DataContractError("employed category reconciliation failed")
    if (numeric_frame["local_fixed"] > numeric_frame["fixed_workplace"]).any():
        raise DataContractError("local fixed-workplace counts exceed their exposure")
    if (numeric_frame["same_area_fixed"] > numeric_frame["local_fixed"]).any():
        raise DataContractError("same-area counts exceed local fixed-workplace counts")
    if not (
        numeric_frame["outside_liverpool_fixed"]
        == numeric_frame["fixed_workplace"] - numeric_frame["local_fixed"]
    ).all():
        raise DataContractError("fixed-workplace flow conservation failed")
    expected_local_share = numeric_frame["local_fixed"] / numeric_frame["fixed_workplace"]
    expected_combined_share = numeric_frame["home_or_no_fixed"] / numeric_frame["employed_total"]
    if not np.allclose(
        numeric_frame["local_retention_share"], expected_local_share, atol=1e-10, rtol=0
    ):
        raise DataContractError("local-retention share identity failed")
    if not np.allclose(
        numeric_frame["home_or_no_fixed_share"], expected_combined_share, atol=1e-10, rtol=0
    ):
        raise DataContractError("combined-category share identity failed")
    if not (
        (numeric_frame["accessibility_3km"] <= numeric_frame["accessibility_5km"])
        & (numeric_frame["accessibility_5km"] <= numeric_frame["accessibility_10km"])
    ).all():
        raise DataContractError("accessibility decay sensitivity is not monotonic")
