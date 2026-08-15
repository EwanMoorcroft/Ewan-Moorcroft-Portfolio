"""Input and derived-data contract tests."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from liverpool_accessibility.analysis import prepare_evidence
from liverpool_accessibility.contracts import DataContractError, read_boundaries, validate_metrics
from liverpool_accessibility.fixtures import write_fixture


def test_boundaries_reject_missing_crs(tmp_path, monkeypatch) -> None:
    """Metric operations must never proceed on a layer without declared coordinates."""
    frame = gpd.GeoDataFrame(
        [{"MSOA21CD": "E02090001", "MSOA21NM": "Fictional", "geometry": box(0, 0, 1, 1)}]
    )
    monkeypatch.setattr(gpd, "read_file", lambda _: frame)
    with pytest.raises(DataContractError, match="EPSG:4326"):
        read_boundaries(tmp_path / "boundaries.geojson")


def test_metrics_reject_broken_flow_conservation() -> None:
    """Outside and local flows must sum to the fixed-workplace exposure."""
    frame = pd.DataFrame(
        [
            {
                "area_code": "E02090001",
                "area_name": "Fictional",
                "employed_total": 100,
                "home_or_no_fixed": 20,
                "other_workplace": 2,
                "fixed_workplace": 78,
                "local_fixed": 50,
                "same_area_fixed": 20,
                "outside_liverpool_fixed": 10,
                "local_retention_share": 0.64,
                "home_or_no_fixed_share": 0.2,
                "inbound_fixed_workers": 80,
                "accessibility_3km": 60,
                "accessibility_5km": 70,
                "accessibility_10km": 75,
                "mean_local_commute_km": 2,
            }
        ]
    )
    with pytest.raises(DataContractError, match="conservation"):
        validate_metrics(frame)


def test_metrics_reject_invented_share() -> None:
    """Stored ratios must be derived from their retained numerator and denominator."""
    frame = pd.DataFrame(
        [
            {
                "area_code": "E02090001",
                "area_name": "Fictional",
                "employed_total": 100,
                "home_or_no_fixed": 20,
                "other_workplace": 2,
                "fixed_workplace": 78,
                "local_fixed": 50,
                "same_area_fixed": 20,
                "outside_liverpool_fixed": 28,
                "local_retention_share": 0.99,
                "home_or_no_fixed_share": 0.2,
                "inbound_fixed_workers": 80,
                "accessibility_3km": 60,
                "accessibility_5km": 70,
                "accessibility_10km": 75,
                "mean_local_commute_km": 2,
            }
        ]
    )
    with pytest.raises(DataContractError, match="share identity"):
        validate_metrics(frame)


def test_flow_input_rejects_negative_count(tmp_path) -> None:
    """A negative count must fail before any aggregate is written."""
    flows, boundaries, centroids = write_fixture(tmp_path / "source", size=3)
    frame = pd.read_csv(flows)
    frame.loc[0, "Count"] = -1
    frame.to_csv(flows, index=False)
    with pytest.raises(DataContractError, match="invalid row"):
        prepare_evidence(flows, boundaries, centroids)


def test_flow_input_rejects_duplicate_key(tmp_path) -> None:
    """One logical origin-destination-indicator key may appear only once."""
    flows, boundaries, centroids = write_fixture(tmp_path / "source", size=3)
    frame = pd.read_csv(flows)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(flows, index=False)
    with pytest.raises(DataContractError, match="duplicate key"):
        prepare_evidence(flows, boundaries, centroids)
