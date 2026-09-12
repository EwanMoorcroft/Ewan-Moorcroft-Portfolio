"""The verifier must reject self-consistent hashes attached to invalid claims."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from liverpool_accessibility.contracts import DataContractError
from liverpool_accessibility.evidence import sha256_file, verify_evidence

RETAINED = Path(__file__).resolve().parents[1] / "reports" / "retained"


def _copy(tmp_path: Path) -> Path:
    return Path(shutil.copytree(RETAINED, tmp_path / "evidence"))


def _rehash(root: Path, name: str) -> None:
    path = root / "results-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][name] = sha256_file(root / name)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.6837067160378045", True])
def test_invalid_result_scalars_fail_closed(tmp_path: Path, value: object) -> None:
    root = _copy(tmp_path)
    path = root / "results.json"
    result = json.loads(path.read_text())
    result["local_retention_share"] = value
    path.write_text(json.dumps(result))
    _rehash(root, path.name)
    with pytest.raises(DataContractError):
        verify_evidence(root)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1, True, "0.1"])
def test_invalid_verification_tolerance_is_rejected(tolerance: object) -> None:
    with pytest.raises(DataContractError, match="tolerance"):
        verify_evidence(RETAINED, tolerance=tolerance)


def test_nan_cannot_hide_a_changed_coefficient(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    path = root / "model-coefficients.csv"
    coefficients = pd.read_csv(path)
    coefficients.loc[0, "estimate"] = float("nan")
    coefficients.loc[1, "estimate"] = 999
    coefficients.to_csv(path, index=False)
    _rehash(root, path.name)
    with pytest.raises(DataContractError, match="coefficient"):
        verify_evidence(root)


def test_permutation_metadata_must_reconstruct(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    path = root / "results.json"
    result = json.loads(path.read_text())
    result["moran"]["permutations"] = 0
    path.write_text(json.dumps(result))
    _rehash(root, path.name)
    with pytest.raises(DataContractError, match="permutations"):
        verify_evidence(root)


def test_graph_must_reconstruct_from_polygons(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    path = root / "spatial-edges.csv"
    edges = pd.read_csv(path)
    pd.concat([edges, edges.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    _rehash(root, path.name)
    with pytest.raises(DataContractError, match="graph"):
        verify_evidence(root)


def test_accessibility_must_reconstruct_from_centroids(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    path = root / "liverpool-msoa-centroids.geojson"
    geometry = json.loads(path.read_text())
    geometry["features"][0]["geometry"]["coordinates"][0] += 0.01
    path.write_text(json.dumps(geometry))
    _rehash(root, path.name)
    with pytest.raises(DataContractError, match="accessibility"):
        verify_evidence(root)


@pytest.mark.parametrize(
    "field,value",
    [("passed", False), ("areas", 999), ("binomial_maximum_robust_se_difference", float("nan"))],
)
def test_r_validation_metadata_is_checked(tmp_path: Path, field: str, value: object) -> None:
    root = _copy(tmp_path)
    path = root / "r-validation.json"
    result = json.loads(path.read_text())
    result[field] = value
    path.write_text(json.dumps(result))
    _rehash(root, path.name)
    with pytest.raises(DataContractError, match="R validation"):
        verify_evidence(root)
