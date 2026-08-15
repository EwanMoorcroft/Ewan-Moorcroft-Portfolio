"""Maintenance tests for the public Liverpool evidence pack."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from liverpool_accessibility.evidence import (
    OFFICIAL_ANALYSIS_AREA,
    OFFICIAL_NATIONAL_FLOW_IDENTITY,
    OFFICIAL_SOURCE_CONTRACT,
    verify_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports" / "retained"


def test_official_source_gate_matches_the_retained_manifest() -> None:
    """Pinned full-source identity must remain aligned with the retained rebuild contract."""
    manifest = json.loads((EVIDENCE / "source-manifest.json").read_text(encoding="utf-8"))
    flow_source = next(source for source in manifest["sources"] if source["role"] == "flow_csv")
    assert manifest["contract"] == OFFICIAL_SOURCE_CONTRACT
    assert manifest["analysis_area"] == OFFICIAL_ANALYSIS_AREA
    for field, expected in OFFICIAL_NATIONAL_FLOW_IDENTITY.items():
        assert flow_source[field] == expected


def test_retained_evidence_reconstructs() -> None:
    """Every public numerical claim should survive independent reconstruction."""
    summary = verify_evidence(EVIDENCE)
    assert summary == {
        "contract": "liverpool-retained-evidence-v1",
        "files_verified": 8,
        "areas": 61,
        "aggregates_verified": 22,
        "coefficient_values_verified": 50,
    }
    results = json.loads((EVIDENCE / "results.json").read_text(encoding="utf-8"))
    assert results["local_fixed_total"] == 84_567
    assert results["fixed_workplace_total"] == 123_689
    assert results["local_retention_share"] == pytest.approx(0.6837067160378045)
    assert results["moran"]["statistic"] == pytest.approx(0.4901355503899729)
    assert results["poisson"]["pearson_dispersion"] > 1.5
    assert results["negative_binomial_nb2"]["aic"] < results["poisson"]["aic"]
    assert 0 <= results["binomial_sensitivity"]["predicted_rate_min"]
    assert results["binomial_sensitivity"]["predicted_rate_max"] <= 1


def test_r_validation_passes() -> None:
    """The retained R comparison must match all 61 areas within its tolerance."""
    result = json.loads((EVIDENCE / "r-validation.json").read_text(encoding="utf-8"))
    assert result["contract"] == "liverpool-r-validation-v1"
    assert result["areas"] == 61
    assert result["passed"] is True
    assert result["moran_difference"] <= 1e-12
    assert result["maximum_coefficient_difference"] <= 1e-7


def test_public_pngs_have_no_text_metadata() -> None:
    """Generated figures must not expose software, timestamps, or local paths."""
    for path in sorted((ROOT / "assets").glob("*.png")):
        with Image.open(path) as image:
            assert set(image.info).issubset({"dpi"})
