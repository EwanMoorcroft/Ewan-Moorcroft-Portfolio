"""Deterministic recruiter-facing figures derived from retained evidence."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .contracts import DataContractError, validate_metrics
from .evidence import FIXTURE_EVIDENCE_SCOPE, FIXTURE_SOURCE_CONTRACT, OFFICIAL_SOURCE_CONTRACT

PLOT_METADATA = {"Software": None, "Creation Time": None}


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})


def write_figures(evidence_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Render one map and one analytical relationship from retained tables."""
    root = Path(evidence_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(root / "area-metrics.csv")
    validate_metrics(metrics)
    source = json.loads((root / "source-manifest.json").read_text())
    if source.get("contract") == FIXTURE_SOURCE_CONTRACT:
        evidence_scope = FIXTURE_EVIDENCE_SCOPE
        map_title = "Fictional grid: local workplace-flow retention"
        local_label = "fictional study area"
    elif source.get("contract") == OFFICIAL_SOURCE_CONTRACT:
        evidence_scope = "Liverpool Census 2021 area-level analysis"
        map_title = "Liverpool Census 2021 local workplace-flow retention"
        local_label = "Liverpool"
    else:
        raise DataContractError("figure source must identify official or fictional evidence")
    boundaries = gpd.read_file(root / "liverpool-msoa-boundaries.geojson")
    mapped = boundaries.merge(
        metrics,
        left_on="MSOA21CD",
        right_on="area_code",
        validate="one_to_one",
    )
    _style()

    map_path = output / "local-retention-map.png"
    figure, axis = plt.subplots(figsize=(9.2, 8.2), constrained_layout=True)
    mapped.plot(
        column="local_retention_share",
        cmap="viridis",
        linewidth=0.35,
        edgecolor="white",
        legend=True,
        legend_kwds={"label": f"Share of fixed-workplace flows staying in {local_label}"},
        ax=axis,
    )
    axis.set_title(map_title)
    axis.set_axis_off()
    figure.savefig(map_path, dpi=180, metadata=PLOT_METADATA)
    plt.close(figure)

    relationship_path = output / "accessibility-relationship.png"
    figure, axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    points = axis.scatter(
        metrics["accessibility_5km"],
        metrics["local_retention_share"],
        c=metrics["home_or_no_fixed_share"],
        cmap="magma",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )
    axis.set(
        title=f"Destination-flow proximity and workplace-flow retention\n{evidence_scope}",
        xlabel=f"Gravity-style access to {local_label} flow mass (5 km decay)",
        ylabel="Local fixed-workplace flow share",
    )
    figure.colorbar(points, ax=axis, label="Home or no-fixed-place share")
    figure.savefig(relationship_path, dpi=180, metadata=PLOT_METADATA)
    plt.close(figure)

    coefficients = pd.read_csv(root / "model-coefficients.csv")
    slopes = (
        coefficients.loc[
            (coefficients["family"] == "binomial_sensitivity") & (coefficients["term"] != "const")
        ]
        .set_index("term")
        .loc[["log_accessibility_5km", "home_or_no_fixed_share"]]
    )
    coefficient_path = output / "binomial-area-uncertainty.png"
    figure, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    axis.errorbar(
        slopes["estimate"],
        [1, 0],
        xerr=[
            slopes["estimate"] - slopes["confidence_low"],
            slopes["confidence_high"] - slopes["estimate"],
        ],
        fmt="o",
        color="#205d75",
        capsize=5,
    )
    axis.axvline(0, color="#6b7280", linewidth=1, linestyle="--")
    axis.set(
        yticks=[1, 0],
        yticklabels=["Log accessibility (5 km)", "Home or no-fixed-place share"],
        xlabel=(
            "Log-odds coefficient per predictor standard deviation\n"
            "Descriptive associations; intervals do not adjust for spatial dependence."
        ),
        title=f"Grouped-binomial estimates and area-level HC0 95% intervals\n{evidence_scope}",
        ylim=(-0.6, 1.6),
    )
    axis.xaxis.label.set_fontsize(10)
    figure.savefig(coefficient_path, dpi=180, metadata=PLOT_METADATA)
    plt.close(figure)

    summary_path = output / "figure-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "figures": [map_path.name, relationship_path.name, coefficient_path.name],
                "source": "supplied area metrics, grouped-binomial coefficients and boundaries",
                "evidence_scope": evidence_scope,
                "map_title": map_title,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return [map_path, relationship_path, coefficient_path, summary_path]
