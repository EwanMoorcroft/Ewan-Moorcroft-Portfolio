"""End-to-end fictional-data and retained-evidence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from liverpool_accessibility.cli import main
from liverpool_accessibility.contracts import DataContractError
from liverpool_accessibility.evidence import verify_evidence, verify_source_manifest
from liverpool_accessibility.fixtures import fixture_source_manifest, write_fixture


def test_fixture_runs_full_pipeline(tmp_path: Path, capsys) -> None:
    """A network-free grid should exercise preparation, analysis, figures, and verification."""
    output = tmp_path / "run"
    assert main(["fixture", "--output-dir", str(output), "--size", "5"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["areas"] == 25
    assert summary["files_verified"] == 7
    assert summary["aggregates_verified"] == 22
    assert (output / "figures" / "local-retention-map.png").is_file()


def test_prepared_fixture_conserves_every_origin(tmp_path: Path, capsys) -> None:
    """Local and outside counts should reconcile for every fictional area."""
    output = tmp_path / "run"
    assert main(["fixture", "--output-dir", str(output)]) == 0
    capsys.readouterr()
    metrics = pd.read_csv(output / "evidence" / "area-metrics.csv")
    assert (metrics["fixed_workplace"] > metrics["local_fixed"]).all()
    assert (metrics["outside_liverpool_fixed"] > 0).all()
    assert (metrics["same_area_fixed"] <= metrics["local_fixed"]).all()


def test_verifier_rejects_tampering(tmp_path: Path, capsys) -> None:
    """Changing a retained table must fail before its claims are reconstructed."""
    output = tmp_path / "run"
    assert main(["fixture", "--output-dir", str(output)]) == 0
    capsys.readouterr()
    path = output / "evidence" / "area-metrics.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace("Fictional area 01", "Changed area"),
        encoding="utf-8",
    )
    with pytest.raises(DataContractError, match="hash mismatch"):
        verify_evidence(output / "evidence")


def test_source_manifest_binds_exact_input_files(tmp_path: Path) -> None:
    """A plausible manifest must not be reusable with different input bytes."""
    flows, boundaries, centroids = write_fixture(tmp_path / "source", size=3)
    manifest = fixture_source_manifest(flows, boundaries, centroids)
    paths = {"flow_csv": flows, "boundaries": boundaries, "centroids": centroids}
    assert verify_source_manifest(manifest, paths)["files_verified"] == 3
    flows.write_text(flows.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="identity mismatch: flow_csv"):
        verify_source_manifest(manifest, paths)
