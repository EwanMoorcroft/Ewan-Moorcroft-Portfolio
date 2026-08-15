"""Deterministic recruiter-facing figures derived from retained evidence."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .contracts import validate_metrics

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
        legend_kwds={"label": "Share of fixed-workplace flows staying in Liverpool"},
        ax=axis,
    )
    axis.set_title("Liverpool Census 2021 local workplace-flow retention")
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
        title="Destination-flow proximity and workplace-flow retention",
        xlabel="Gravity-style access to Liverpool fixed-workplace flow mass (5 km decay)",
        ylabel="Local fixed-workplace flow share",
    )
    figure.colorbar(points, ax=axis, label="Home or no-fixed-place share")
    figure.savefig(relationship_path, dpi=180, metadata=PLOT_METADATA)
    plt.close(figure)

    summary_path = output / "figure-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "figures": [map_path.name, relationship_path.name],
                "source": "reports/retained area metrics and 2021 boundaries",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return [map_path, relationship_path, summary_path]
