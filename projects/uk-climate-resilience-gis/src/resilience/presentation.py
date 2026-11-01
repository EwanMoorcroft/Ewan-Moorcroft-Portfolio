"""Source-backed figures and a self-contained, offline neighbourhood explorer.

The GeoPackage and methodological diagnostics are mandatory inputs. Display
geometry is simplified independently; analytical geometries are never changed.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from .io import ROOT

CHARCOAL = "#243335"
TEAL = "#117d83"
MUTED = "#6e7c7e"
GREY = "#d4dcdd"
WARM = "#ad643d"
DIMENSION_LABELS = ["Flood\nextent", "Surface\nheat", "Green-space\naccess deficit", "Income\ndeprivation", "Age-sensitive\npopulation"]
SCENARIOS = ("balanced", "hazard", "equity")
FRACTIONS = {"flood_exposure", "access_deficit", "income_rate", "age_sensitive_share",
             "case_network_deficit", "top20_frequency", "heat_coverage", "rank_span_fraction"}
REQUIRED = {
    "lsoa21cd", "name", "borough", "population", "flood_exposure", "heat_anomaly",
    "heat_coverage", "access_deficit", "income_rate", "age_sensitive_share",
    "case_network_deficit", "top20_frequency", "rank_p05", "rank_p95",
    "rank_span_fraction", "robust_priority", "model_eligible", "priority_class", "follow_up",
    "z_score", "spatial_lag_z", "quadrant",
    *["score_" + scenario for scenario in SCENARIOS],
    *["rank_" + scenario for scenario in SCENARIOS],
    *["dim_" + name for name in ("flood", "heat", "access", "income", "age")],
}


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_value(value, unit: str = "number", digits: int = 1) -> str:
    """Use the same explicit missing and fractional-rate conventions as the map."""
    number = _number(value)
    if number is None:
        return "Not available"
    if unit == "percent":
        return f"{number * 100:.{digits}f}%"
    if unit == "count":
        return f"{number:,.0f}"
    number = 0.0 if round(number, digits) == 0 else number
    if unit == "celsius":
        return f"{'+' if number > 0 else ''}{number:.{digits}f} °C"
    return f"{number:,.{digits}f}"


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is None or value is pd.NA:
        return None
    return str(value) if not isinstance(value, str) else value


def json_for_script(value) -> str:
    """Escape script delimiters as well as invalid JSON numeric values."""
    return (json.dumps(_plain(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def metric_scale(field: str, values) -> list[float]:
    if field in FRACTIONS:
        return [0.0, 1.0]
    if field == "heat_anomaly":
        finite = [abs(number) for value in values if (number := _number(value)) is not None]
        limit = max(0.5, math.ceil(max(finite, default=0.5) * 2) / 2)
        return [-limit, limit]
    return [0.0, 100.0]


def svg_path(geometry, west: float, north: float, tolerance: float = 15) -> str:
    """Preserve polygon holes while projecting BNG coordinates to an SVG plane."""
    geometry = geometry.simplify(tolerance, preserve_topology=True) if tolerance else geometry
    polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    parts = []
    for polygon in polygons:
        for ring in [polygon.exterior, *polygon.interiors]:
            points = [(x - west, north - y) for x, y, *_ in ring.coords]
            parts.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in points) + "Z")
    return "".join(parts)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.edgecolor": "#aebcbe", "axes.labelcolor": CHARCOAL,
        "text.color": CHARCOAL, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "grid.color": "#e5ebeb", "grid.linewidth": 0.65, "svg.fonttype": "none",
    })


def _save_figure(fig, stem: str, destination: Path) -> list[Path]:
    paths = []
    for extension in ("svg", "png"):
        path = destination / f"{stem}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.22)
        paths.append(path)
    plt.close(fig)
    return paths


def _scenario_figure(frame, model):
    eligible = frame[frame.model_eligible.eq(True)].copy()
    n = len(eligible)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    fig.suptitle("How do policy scenarios change neighbourhood ranks?", x=0.07, ha="left", fontsize=17, y=0.99)
    for axis, scenario, label in zip(axes, ("hazard", "equity"), ("Hazard emphasis", "Equity emphasis")):
        stable = eligible.robust_priority.eq(True)
        axis.scatter(eligible.rank_balanced, eligible["rank_" + scenario], s=11, c=GREY, alpha=0.8, linewidths=0)
        axis.scatter(eligible.loc[stable, "rank_balanced"], eligible.loc[stable, "rank_" + scenario],
                     s=13, c=TEAL, alpha=0.85, linewidths=0, label="Stable priority under tested weights")
        axis.plot([1, n], [1, n], color=MUTED, lw=0.9, ls="--", zorder=0)
        axis.set(xlim=(0.5, n + 0.5), ylim=(n + 0.5, 0.5), xlabel="Balanced rank (1 = highest priority)", ylabel=f"{label} rank")
        rho = model["scenario_rank_spearman"]["balanced"][scenario]
        axis.set_title(f"{label}  ·  Spearman ρ = {format_value(rho, digits=3)}", loc="left", pad=12)
        axis.grid(alpha=0.6, zorder=-1)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=9, markerscale=1.5)
    fig.text(0.07, 0.01, f"{n:,} complete-case LSOAs. Average ranks retain ties. The dashed line indicates unchanged rank.", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.24, wspace=0.28)
    return fig


def _sensitivity_figure(frame, model):
    eligible = frame[frame.model_eligible.eq(True)].copy()
    selected = eligible.sort_values(["rank_balanced", "lsoa21cd"]).head(20).iloc[::-1]
    experiment = model["weight_experiment"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [0.9, 1.6]})
    fig.suptitle("How sensitive are priorities to indicator weights?", x=0.065, ha="left", fontsize=17, y=0.985)
    axes[0].hist(100 * eligible.top20_frequency, bins=np.linspace(0, 100, 21), color=TEAL, edgecolor="white", linewidth=0.8)
    axes[0].set(xlabel="Selection frequency in the top group (%)", ylabel="Neighbourhoods", xlim=(0, 100))
    axes[0].set_title("All eligible neighbourhoods", loc="left", pad=12)
    axes[0].grid(axis="y", alpha=0.65)
    axes[0].set_axisbelow(True)
    for y, (_, row) in enumerate(selected.iterrows()):
        colour = TEAL
        axes[1].plot([row.rank_p05, row.rank_p95], [y, y], color=colour, lw=2)
        axes[1].scatter(row.rank_balanced, y, s=28, color=colour, edgecolor="white", lw=0.5, zorder=3)
    axes[1].set_yticks(range(len(selected)), [str(name) for name in selected["name"]], fontsize=9)
    axes[1].set_xlabel("Rank (1 = highest priority)")
    axes[1].set_title("Leading 20 by balanced rank · 5th–95th percentiles", loc="left", pad=12)
    axes[1].grid(axis="x", alpha=0.6)
    axes[1].set_xlim(left=0)
    overlap = model["alternative_minmax"]["balanced"]["top_jaccard"]
    fig.text(0.065, 0.076, f"{experiment['samples']:,} weight draws; top {100 * model['top_group']['fraction']:.0f}% group. Lines show sensitivity ranges; dots show balanced ranks.", fontsize=10)
    fig.text(0.065, 0.050, f"Changing midrank to min-max normalisation gives balanced top-group Jaccard overlap {format_value(overlap, digits=3)}.", fontsize=9)
    fig.text(0.065, 0.025, "Weight stability is not robustness to every modelling choice, evidence quality, or a confidence interval.", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.89, bottom=0.16, wspace=0.58)
    return fig


def _correlation_figure(model, config):
    indicators = config["indicators"]
    values = np.array([[model["indicator_spearman"][left][right] for right in indicators] for left in indicators], dtype=float)
    cmap = LinearSegmentedColormap.from_list("association", [TEAL, "#f6f6f2", WARM])
    cmap.set_bad(GREY)
    fig, axis = plt.subplots(figsize=(8.5, 7.4))
    fig.suptitle("Which screening dimensions move together?", x=0.06, ha="left", fontsize=17, y=0.985)
    image = axis.imshow(np.ma.masked_invalid(values), vmin=-1, vmax=1, cmap=cmap)
    axis.set_xticks(range(5), ["Flood\nextent", "Surface\nheat", "Access\ndeficit", "Income\ndeprivation", "Age-sensitive\npopulation"])
    axis.set_yticks(range(5), [label.replace("\n", " ") for label in DIMENSION_LABELS])
    axis.tick_params(length=0, pad=9)
    for (row, col), value in np.ndenumerate(values):
        axis.text(col, row, f"{value:.2f}" if np.isfinite(value) else "NA", ha="center", va="center",
                  fontsize=12, color="white" if np.isfinite(value) and abs(value) > 0.7 else CHARCOAL)
    bar = fig.colorbar(image, ax=axis, shrink=0.75, pad=0.04, ticks=[-1, -0.5, 0, 0.5, 1])
    bar.set_label("Spearman rank correlation (ρ)")
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.text(0.06, 0.035, "Same complete-case LSOA cohort as the model. NA denotes an undefined correlation, including constant indicators.", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.15)
    return fig


def _moran_figure(frame, spatial):
    cohort = frame[frame.model_eligible.eq(True)]
    finite = np.isfinite(pd.to_numeric(cohort.z_score, errors="coerce")) & np.isfinite(pd.to_numeric(cohort.spatial_lag_z, errors="coerce"))
    points = cohort.loc[finite]
    primary = spatial["primary"]
    fig, axis = plt.subplots(figsize=(10.4, 7.3))
    fig.suptitle("Do neighbouring areas have similar screening scores?", x=0.075, ha="left", fontsize=17, y=0.985)
    axis.scatter(points.z_score, points.spatial_lag_z, s=14, c=TEAL, alpha=0.4, linewidths=0)
    axis.axhline(0, color=MUTED, lw=0.9)
    axis.axvline(0, color=MUTED, lw=0.9)
    axis.set(xlabel="Balanced score · standard deviations from cohort mean", ylabel="Neighbouring scores · row-standardised queen spatial lag")
    axis.grid(alpha=0.45)
    labels = [(0.025, 0.96, "Lower score / higher neighbours", "left"),
              (0.975, 0.96, "Higher score / higher neighbours", "right"),
              (0.025, 0.04, "Lower score / lower neighbours", "left"),
              (0.975, 0.04, "Higher score / lower neighbours", "right")]
    for x, y, label, alignment in labels:
        axis.text(x, y, label, transform=axis.transAxes, va="center", ha=alignment, fontsize=9,
                  color=MUTED, bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2})
    if primary["status"] == "ok":
        caption = (f"Global Moran's I = {format_value(primary['statistic'], digits=3)}; "
                   f"two-sided permutation p = {format_value(primary['p_two_sided'], digits=4)}; "
                   f"{primary['permutations_run']:,} permutations.")
    else:
        caption = "Global statistic unavailable: " + str(primary["status"]).replace("_", " ") + "."
    fig.text(0.075, 0.065, caption, fontsize=10)
    fig.text(0.075, 0.034, f"{len(points):,} displayed areas with a defined spatial lag. Quadrants are descriptive, not significant local clusters.", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.105, right=0.975, top=0.90, bottom=0.19)
    return fig


def _validated_frame(frame, model):
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError("Presentation input is missing columns: " + ", ".join(missing))
    if frame.empty or frame.lsoa21cd.isna().any() or frame.lsoa21cd.duplicated().any():
        raise ValueError("Presentation input requires unique non-null neighbourhood identifiers")
    if frame.crs is None:
        raise ValueError("Presentation geometry has no CRS")
    frame = frame.to_crs(27700).copy()
    if (~frame.geometry.is_valid | frame.geometry.is_empty | frame.geometry.isna()).any():
        raise ValueError("Presentation geometry must be valid and nonempty")
    if not frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("Presentation geometry must contain polygons")
    eligible = frame.model_eligible.eq(True)
    if not eligible.any():
        raise ValueError("No eligible neighbourhoods are available for the requested model figures")
    if set(frame.loc[eligible, "lsoa21cd"]) != set(model["cohort"]["eligible_ids_sorted"]):
        raise ValueError("Model diagnostics and GeoPackage have different eligible cohorts")
    for field in FRACTIONS:
        values = pd.to_numeric(frame[field], errors="coerce").dropna()
        if ((values < 0) | (values > 1) | ~np.isfinite(values)).any():
            raise ValueError(f"Invalid fractional presentation field: {field}")
    for field in ["score_" + s for s in SCENARIOS] + ["rank_p05", "rank_p95", "top20_frequency"]:
        if not np.isfinite(pd.to_numeric(frame.loc[eligible, field], errors="coerce")).all():
            raise ValueError(f"Eligible rows have missing or nonfinite {field}")
    if (frame.loc[eligible, "rank_p05"] > frame.loc[eligible, "rank_p95"]).any():
        raise ValueError("Sensitivity rank percentiles are reversed")
    return frame


def build_presentations(config: dict) -> dict:
    """Build eight figure files and one offline HTML from completed real inputs."""
    package = ROOT / "data/processed/resilience.gpkg"
    model_path = ROOT / "outputs/qa/model.json"
    spatial_path = ROOT / "outputs/qa/spatial.json"
    for path in (package, model_path, spatial_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required real presentation input is not ready: {path}")
    model = json.loads(model_path.read_text())
    model = model.get("exactfit_model", model)
    spatial = json.loads(spatial_path.read_text())
    spatial = spatial.get("spatial_diagnostics", spatial)
    frame = _validated_frame(gpd.read_file(package, layer="neighbourhoods"), model)
    if set(spatial["cohort"]["eligible_ids_sorted"]) != set(model["cohort"]["eligible_ids_sorted"]):
        raise ValueError("Spatial diagnostics and model use different cohorts")
    _style()
    destination = ROOT / "outputs/figures"
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem, figure in [
        ("scenario_ranks", lambda: _scenario_figure(frame, model)),
        ("weight_sensitivity", lambda: _sensitivity_figure(frame, model)),
        ("indicator_correlations", lambda: _correlation_figure(model, config)),
        ("moran_scatter", lambda: _moran_figure(frame, spatial)),
    ]:
        paths.extend(_save_figure(figure(), stem, destination))
    output = ROOT / "outputs/portfolio/interactive.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_html(frame, config, model), encoding="utf-8")
    return {"figures": paths, "interactive": output, "neighbourhoods": len(frame)}


def _render_html(frame, config, model) -> str:
    frame = frame.sort_values(["borough", "name", "lsoa21cd"]).reset_index(drop=True)
    west, south, east, north = map(float, frame.total_bounds)
    width, height = east - west, north - south
    rows, paths = [], []
    optional = [name for name in ("model_missing_indicators", "case_network_coverage") if name in frame]
    for index, row in frame.iterrows():
        record = row[sorted(REQUIRED) + optional].to_dict()
        left, bottom, right, top = row.geometry.bounds
        record["bounds"] = [left - west, north - top, right - left, top - bottom]
        record["model_eligible"] = bool(row.model_eligible == True)
        record["robust_priority"] = bool(row.robust_priority == True)
        rows.append(record)
        label = html.escape(f"{row['name']}, {row.borough}, {row.lsoa21cd}", quote=True)
        paths.append(f'<path data-i="{index}" d="{svg_path(row.geometry, west, north)}" '
                     f'fill-rule="evenodd" tabindex="{-1}" role="button" aria-label="{label}"/>')
    metrics = [
        {"field": "score_balanced", "label": "Priority · balanced", "unit": "score", "title": "Balanced screening score",
         "note": "Index 0–100. Greater values indicate higher relative priority across five equally weighted dimensions."},
        {"field": "score_hazard", "label": "Priority · hazard emphasis", "unit": "score", "title": "Hazard-emphasis screening score",
         "note": "Index 0–100. This scenario gives greater weight to flood extent and surface heat."},
        {"field": "score_equity", "label": "Priority · equity emphasis", "unit": "score", "title": "Equity-emphasis screening score",
         "note": "Index 0–100. This scenario gives greater weight to income deprivation and age sensitivity."},
        {"field": "flood_exposure", "label": "Flood · area overlap", "unit": "percent", "title": "Neighbourhood area in the defended flood scenario",
         "note": "Whole LSOA polygon area, including retained inland water, within the present-day 1% river / 0.5% sea extent. Defences are assumed to function. Not residents or properties exposed."},
        {"field": "heat_anomaly", "label": "Heat · surface anomaly", "unit": "celsius", "title": "Summer surface-temperature anomaly",
         "note": "Five selected summer days in 2021–2025. Degrees Celsius relative to the study's clear-sky reference, not a continuous climate series. Land-surface temperature is not air temperature; grey indicates insufficient usable evidence."},
        {"field": "access_deficit", "label": "Access · household deficit", "unit": "percent", "title": "Households without nearby green-space access",
         "note": "Official household share without a 1 km walking route to green space of at least 10 ha. Corrected Defra 2024 statistics, aggregated from OA household counts."},
        {"field": "top20_frequency", "label": "Sensitivity · selection frequency", "unit": "percent", "title": "Frequency of selection in the top group",
         "note": "Share of tested weight draws selecting an area in the configured top group. This measures sensitivity to weights, not evidence quality or statistical confidence."},
        {"field": "case_network_deficit", "label": "Case study · park walking deficit", "unit": "percent", "title": "Modelled residents without nearby park access",
         "note": "Manchester–Salford only: OA-population-weighted share beyond 1 km or with no mapped route to a public-park entrance. Parks are at least 10 ha. Network snapping does not guarantee a route. This differs from the official household measure."},
    ]
    for metric in metrics:
        metric["domain"] = metric_scale(metric["field"], frame[metric["field"]])
    data = {
        "rows": rows, "metrics": metrics,
        "view": [-0.025 * width, -0.025 * height, width * 1.05, height * 1.05],
        "samples": model["weight_experiment"]["samples"],
        "topFraction": model["top_group"]["fraction"],
        "stableThreshold": model["robustness"]["frequency_threshold"],
        "normalisationJaccard": model.get("alternative_minmax", {}).get("balanced", {}).get("top_jaccard"),
        "minimumHeatCoverage": config["minimum_lsoa_heat_coverage"],
    }
    boroughs = "".join(f'<option value="{html.escape(str(value), quote=True)}">{html.escape(str(value))}</option>'
                        for value in sorted(frame.borough.unique()))
    return (HTML_TEMPLATE.replace("__DATA__", json_for_script(data))
            .replace("__PATHS__", "".join(paths)).replace("__BOROUGHS__", boroughs))


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en-GB">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="An evidence-led Greater Manchester neighbourhood screening study and Manchester–Salford walking-access case study.">
<title>Greater Manchester · Neighbourhood adaptation priorities</title>
<style>
:root{color-scheme:light;--ink:#243335;--muted:#66777a;--teal:#117d83;--line:#dce5e5;--paper:#f6f8f7;--missing:#cbd2d4;--warm:#ad643d}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:15px;line-height:1.5}button,input,select{font:inherit}a{color:#086a70;text-underline-offset:3px}button,select,input{min-height:42px;border:1px solid #aebfc1;border-radius:5px;background:white;color:var(--ink)}button{cursor:pointer;padding:7px 13px}button:hover{background:#eef5f4}button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #0b848c;outline-offset:3px}input,select{padding:8px 11px;width:100%}h1,h2,h3,p{margin-top:0}h1{font-size:clamp(27px,3vw,42px);line-height:1.12;letter-spacing:-.035em;margin-bottom:13px;font-weight:720}h2{font-size:21px;letter-spacing:-.02em;margin-bottom:7px}h3{font-size:14px;margin:21px 0 9px}p{margin-bottom:12px}.wrap{max-width:1540px;margin:auto;padding:0 34px}.masthead{padding:31px 0 25px;background:#fff;border-top:5px solid var(--teal);border-bottom:1px solid var(--line)}.eyebrow{font-size:11px;letter-spacing:.15em;text-transform:uppercase;font-weight:750;color:var(--teal);margin-bottom:12px}.intro{max-width:810px;color:var(--muted);font-size:16px}.summary{display:flex;flex-wrap:wrap;gap:15px 38px;margin-top:21px}.summary div{display:flex;align-items:baseline;gap:8px}.summary strong{font-size:24px;letter-spacing:-.03em}.summary span{font-size:13px;color:var(--muted)}main{padding-top:23px;padding-bottom:30px}.controls{display:grid;grid-template-columns:1.1fr .9fr 1.2fr;gap:16px;margin-bottom:18px}.control label{font-size:12px;font-weight:700;display:block;margin-bottom:6px}.workspace{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(320px,1fr);gap:20px;align-items:start}.map-panel,.detail{background:#fff;border:1px solid var(--line);border-radius:7px;overflow:hidden}.map-heading{padding:18px 20px 13px;border-bottom:1px solid var(--line)}.map-heading h2{font-size:18px}.note,.small{font-size:12px;color:var(--muted)}#metric-note{min-height:48px;max-width:780px;margin-bottom:8px}.legend{display:grid;grid-template-columns:minmax(160px,270px) 1fr;gap:14px;align-items:center;margin-top:12px}.legend-scale{max-width:270px}.legend-bar{height:9px;border-radius:2px;background:#ddd}.legend-labels{display:flex;justify-content:space-between;font-size:11px;margin-top:4px;font-variant-numeric:tabular-nums}.missing-key{font-size:11px;color:var(--muted);display:flex;gap:6px;align-items:center}.missing-key i{width:12px;height:12px;display:inline-block;background:var(--missing);border:1px solid #9ca9ab}.map-stage{position:relative;height:620px;min-height:430px;background:#fbfcfb;overflow:hidden}.map-stage svg{width:100%;height:100%;display:block;touch-action:none;cursor:grab}.map-stage svg.dragging{cursor:grabbing}.map-stage path{stroke:#fff;stroke-width:.45px;vector-effect:non-scaling-stroke;cursor:pointer;transition:fill .12s}.map-stage path:hover{stroke:var(--ink);stroke-width:1.7px}.map-stage path:focus-visible,.map-stage path.selected{outline:none;stroke:#182e31;stroke-width:2.4px}.map-buttons{position:absolute;right:13px;top:13px;display:flex;gap:5px;z-index:2}.map-buttons button{min-width:36px;min-height:35px;padding:3px 10px;font-size:18px;box-shadow:0 1px 3px #24333512}.map-buttons .fit{font-size:12px;font-weight:650}.north{position:absolute;left:17px;top:16px;text-align:center;font-size:11px;font-weight:750;color:#516165;pointer-events:none}.north:before{content:"↑";display:block;font-size:27px;line-height:29px}.scale{position:absolute;left:17px;bottom:16px;pointer-events:none;font-size:11px;color:#516165;background:#ffffffd9;padding:6px 8px}.scale-line{height:6px;border:1px solid #516165;border-top:0;width:70px}.tooltip{position:absolute;left:17px;top:64px;max-width:min(340px,85%);z-index:3;background:#243335f2;color:white;padding:10px 13px;border-radius:5px;box-shadow:0 4px 15px #24333518;pointer-events:none;font-size:12px}.tooltip strong{display:block;font-size:13px;margin-bottom:3px}.tooltip[hidden]{display:none}.map-bottom{padding:11px 18px;border-top:1px solid var(--line);font-size:12px;color:var(--muted);display:flex;justify-content:space-between;gap:10px}.detail{padding:20px;position:sticky;top:18px}.choose label{font-size:12px;font-weight:700;display:block;margin-bottom:6px}.choose{padding-bottom:19px;margin-bottom:18px;border-bottom:1px solid var(--line)}.empty{padding:10px 0 18px;color:var(--muted)}.empty h2{color:var(--ink)}.identity{color:var(--muted);font-size:12px;margin-bottom:12px}.tag{display:inline-block;font-size:11px;padding:4px 8px;background:#e7f3f1;color:#0c686f;border-radius:3px;font-weight:700;margin-bottom:13px}.tag.missing{background:#eceff0;color:#526164}.score-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:5px 0 18px}.score-grid div{border-bottom:3px solid #e2eeee;padding:5px 0 9px}.score-grid b{font-size:22px;display:block;font-variant-numeric:tabular-nums}.score-grid small{display:block;color:var(--muted);font-size:10px}.score-grid span{display:block;color:var(--muted);font-size:11px}.metrics{margin:0}.metrics>div{display:flex;justify-content:space-between;gap:20px;padding:8px 0;border-bottom:1px solid #edf1f1}.metrics dt{font-size:12px;color:#536669}.metrics dd{font-size:13px;font-weight:650;text-align:right;margin:0;font-variant-numeric:tabular-nums;white-space:nowrap}.explanation{font-size:12px;color:var(--muted);margin:9px 0 0}.evidence{margin-top:16px;padding:13px 14px;background:#f3f6f5;border-left:3px solid #a7b9bc}.evidence strong{display:block;font-size:12px;margin-bottom:4px}.evidence p{font-size:12px;margin:0;color:#526568}.follow{font-size:13px;border-top:1px solid var(--line);padding-top:14px;margin-top:18px}.method-note{margin-top:20px;max-width:1100px;color:var(--muted);font-size:12px}.method-note strong{color:var(--ink)}footer{border-top:1px solid var(--line);padding:23px 0 28px;background:#fff;font-size:11px;color:var(--muted)}footer p{max-width:1240px;margin-bottom:9px}.source-links{display:flex;flex-wrap:wrap;gap:5px 16px;margin:8px 0 12px}noscript{display:block;padding:20px;color:#7b452a;background:#fff0df}[hidden]{display:none!important}
main.wrap{padding-top:23px;padding-bottom:30px}
@media(max-width:1050px){.wrap{padding:0 22px}.workspace{grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr)}.map-stage{height:590px}.legend{grid-template-columns:minmax(130px,1fr) 1fr}}
@media(max-width:820px){.workspace{grid-template-columns:1fr}.detail{position:static}.map-stage{height:58vh;min-height:400px}.controls{grid-template-columns:1fr 1fr}.control:last-child{grid-column:1/-1}.map-heading h2{font-size:17px}.summary{gap:10px 25px}}
@media(max-width:480px){.wrap{padding:0 15px}.masthead{padding-top:24px}.controls{grid-template-columns:1fr;gap:11px}.control:last-child{grid-column:auto}.intro{font-size:14px}.summary strong{font-size:21px}.summary span{font-size:11px}.map-heading,.detail{padding:16px}.map-stage{min-height:380px}.legend{grid-template-columns:1fr}.missing-key{margin-top:-4px}.map-bottom{display:block}.map-bottom span{display:block}.metrics dt{max-width:65%}.metrics dd{white-space:normal}.score-grid b{font-size:20px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<header class="masthead"><div class="wrap">
<p class="eyebrow">Simulated commission · Greater Manchester</p>
<h1>Neighbourhood adaptation priorities</h1>
<p class="intro">Explore how flood extent, surface heat, green-space access and social vulnerability combine across the region. Use the evidence to guide further investigation, rather than individual property or health decisions.</p>
<div class="summary" aria-label="Regional study coverage"><div><strong id="total-count">—</strong><span>neighbourhoods</span></div><div><strong id="eligible-count">—</strong><span>with all five indicators</span></div><div><strong id="stable-count">—</strong><span>stable priorities under tested weights</span></div></div>
</div></header>
<main class="wrap">
<noscript>This self-contained explorer needs JavaScript for its map controls. The study's static figures remain available separately.</noscript>
<section class="controls" aria-label="Map filters">
<div class="control"><label for="metric">View a measure</label><select id="metric"></select></div>
<div class="control"><label for="borough">Borough</label><select id="borough"><option value="">All Greater Manchester</option>__BOROUGHS__</select></div>
<div class="control"><label for="search">Find a neighbourhood or LSOA code</label><input id="search" type="search" placeholder="For example, Salford or E010…" autocomplete="off"></div>
</section>
<div class="workspace">
<section class="map-panel" aria-labelledby="map-title">
<div class="map-heading"><h2 id="map-title"></h2><p class="note" id="metric-note"></p>
<div class="legend"><div class="legend-scale"><div id="legend-bar" class="legend-bar"></div><div class="legend-labels"><span id="legend-low"></span><span id="legend-mid"></span><span id="legend-high"></span></div></div><div class="missing-key"><i></i><span>Not available / not assessed</span></div></div></div>
<div class="map-stage"><svg id="map" xmlns="http://www.w3.org/2000/svg" role="group" aria-labelledby="map-title" aria-describedby="map-help"><g id="features">__PATHS__</g></svg>
<div class="north" title="Grid north">N</div><div class="map-buttons"><button id="zoom-in" aria-label="Zoom in">+</button><button id="zoom-out" aria-label="Zoom out">−</button><button id="fit" class="fit">Fit view</button></div>
<div id="tooltip" class="tooltip" role="tooltip" hidden><strong id="tip-name"></strong><span id="tip-value"></span></div><div class="scale"><div class="scale-line" id="scale-line"></div><span id="scale-label"></span></div></div>
<div class="map-bottom"><span id="filter-count" aria-live="polite"></span><span id="map-help">Drag to pan · use + / − to zoom · arrow keys browse areas</span></div>
</section>
<aside class="detail" aria-label="Neighbourhood details">
<div class="choose"><label for="neighbourhood">Inspect a neighbourhood</label><select id="neighbourhood"><option value="">Choose from the visible areas</option></select></div>
<div id="empty" class="empty"><h2>Read the local evidence</h2><p>Select an area on the map or from the list. Its raw measures, scenario ranks and evidence limits will appear here.</p><p class="small">Keyboard: focus a map area, use arrow keys to browse, then press Enter or Space to inspect.</p></div>
<article id="record" hidden><h2 id="detail-name"></h2><p id="detail-meta" class="identity"></p><span id="detail-tag" class="tag"></span>
<p class="small">Scenario scores · relative index 0–100</p><div class="score-grid"><div><small>BALANCED</small><b id="score-balanced"></b><span id="rank-balanced"></span></div><div><small>HAZARD</small><b id="score-hazard"></b><span id="rank-hazard"></span></div><div><small>EQUITY</small><b id="score-equity"></b><span id="rank-equity"></span></div></div>
<h3>What the inputs show</h3><dl id="raw-values" class="metrics"></dl><p class="explanation">Percentages have different denominators: neighbourhood area, households or residents. The case-study route measure is separate from official household access.</p>
<div class="evidence"><strong id="evidence-title"></strong><p id="evidence-text"></p></div>
<h3>How priorities respond to weights</h3><dl id="sensitivity-values" class="metrics"></dl><p class="explanation">Rank ranges are 5th–95th percentiles across tested weights, not confidence intervals. Stability does not certify source quality.</p>
<p id="follow-up" class="follow"></p></article>
</aside>
</div>
<p class="method-note"><strong>Reading the map.</strong> Scales remain fixed across borough and search filters. Composite scores compare the same complete-evidence cohort; grey composite areas are unassessed. <span id="stability-rule"></span> Raw inputs can remain visible where no composite score is assigned.</p><p class="method-note" id="normalisation-note"></p>
</main>
<footer><div class="wrap"><p><strong>Portfolio study · simulated commission, not an official Greater Manchester assessment.</strong> Sources span 2021–2026 and do not describe one common observation date. Flooding covers the selected river/sea scenario, not surface-water flooding. Satellite heat describes the land surface. Walking access is modelled, not field verified.</p>
<nav class="source-links" aria-label="Data sources"><a href="https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/adhocs/15432ts007acensus2021agebyfiveyearagebands">ONS · Census and geography</a><a href="https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025">MHCLG · deprivation</a><a href="https://www.data.gov.uk/dataset/1589b7ec-2300-4028-9199-aa7cac4b0f68/rivers-and-sea-defended-and-undefended-flood-risk-extents-present-day">Environment Agency · flood extent</a><a href="https://doi.org/10.5066/P9OGBGM6">USGS · Landsat</a><a href="https://www.gov.uk/government/statistics/access-to-green-space-in-england/access-to-green-space-in-england">Defra · household access</a><a href="https://docs.os.uk/os-downloads/products/land-and-terrain-portfolio/os-open-greenspace/os-open-greenspace-overview">OS · greenspace</a><a href="https://www.openstreetmap.org/copyright">OpenStreetMap · walking network</a></nav>
<p>Contains National Statistics, MHCLG and Defra data © Crown copyright; contains Ordnance Survey data © Crown copyright and database right 2026. © Environment Agency copyright and/or database right 2025. All rights reserved. These inputs are used under the Open Government Licence. Landsat imagery courtesy of the U.S. Geological Survey, public domain. © OpenStreetMap contributors, ODbL 1.0.</p></div></footer>
<script id="study-data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('study-data').textContent),rows=data.rows;
const $=id=>document.getElementById(id),svg=$('map'),paths=Array.from(document.querySelectorAll('#features path'));
let selected=null,visible=rows.map((_,i)=>i),view=data.view.slice(),drag=null,moved=false;
const finite=value=>typeof value==='number'&&Number.isFinite(value);
function fmt(value,unit='number'){
 if(!finite(value))return 'Not available';
 if(unit==='percent')return (value*100).toFixed(1)+'%';
 if(unit==='count')return Math.round(value).toLocaleString('en-GB');
 if(unit==='rank')return value.toLocaleString('en-GB',{maximumFractionDigits:1});
 if(unit==='celsius'){const n=Math.abs(value)<.05?0:value;return (n>0?'+':'')+n.toFixed(1)+' °C';}
 return value.toFixed(1);
}
function activeMetric(){return data.metrics.find(m=>m.field===$('metric').value);}
function interpolate(a,b,t){return '#'+a.map((v,i)=>Math.round(v+(b[i]-v)*t).toString(16).padStart(2,'0')).join('');}
function colour(value,metric){
 if(!finite(value))return '#cbd2d4';
 const [low,high]=metric.domain,t=Math.min(1,Math.max(0,(value-low)/(high-low)));
 if(metric.field==='heat_anomaly')return t<.5?interpolate([17,125,131],[246,246,242],t*2):interpolate([246,246,242],[173,100,61],(t-.5)*2);
 return interpolate([236,245,242],[10,100,108],t);
}
function metricValue(row,metric){
 if((metric.field.startsWith('score_')||metric.field==='top20_frequency')&&!row.model_eligible)return null;
 return row[metric.field];
}
function updateLegend(){
 const m=activeMetric();$('map-title').textContent=m.title;$('metric-note').textContent=m.note;
 $('legend-low').textContent=fmt(m.domain[0],m.unit);$('legend-mid').textContent=fmt((m.domain[0]+m.domain[1])/2,m.unit);$('legend-high').textContent=fmt(m.domain[1],m.unit);
 $('legend-bar').style.background=m.field==='heat_anomaly'?'linear-gradient(90deg,#117d83,#f6f6f2,#ad643d)':'linear-gradient(90deg,#ecf5f2,#0a646c)';
 paths.forEach((path,i)=>{path.style.fill=colour(metricValue(rows[i],m),m);});
 $('tooltip').hidden=true;
}
function showTooltip(i){const m=activeMetric();$('tip-name').textContent=rows[i].name+' · '+rows[i].borough;$('tip-value').textContent=m.title+': '+fmt(metricValue(rows[i],m),m.unit);$('tooltip').hidden=false;}
function setRows(target,values){
 target.replaceChildren();
 values.forEach(([label,value])=>{const line=document.createElement('div'),term=document.createElement('dt'),description=document.createElement('dd');term.textContent=label;description.textContent=value;line.append(term,description);target.append(line);});
}
function inspect(i,focus=false){
 if(i===null){selected=null;$('record').hidden=true;$('empty').hidden=false;$('neighbourhood').value='';paths.forEach(p=>{p.classList.remove('selected');p.setAttribute('aria-pressed','false');});return;}
 selected=i;const r=rows[i];$('record').hidden=false;$('empty').hidden=true;$('neighbourhood').value=String(i);
 $('detail-name').textContent=r.name;$('detail-meta').textContent=r.borough+' · '+r.lsoa21cd+' · '+fmt(r.population,'count')+' Census 2021 residents';
 $('detail-tag').textContent=r.model_eligible?(r.priority_class||'Assessed').replaceAll('_',' '):'Not assessed for composite priority';$('detail-tag').className='tag'+(r.model_eligible?'':' missing');
 ['balanced','hazard','equity'].forEach(s=>{$('score-'+s).textContent=r.model_eligible?fmt(r['score_'+s]):'—';$('rank-'+s).textContent=r.model_eligible?'Rank '+fmt(r['rank_'+s],'rank')+' of '+fmt(rows.filter(d=>d.model_eligible).length,'count'):'Not assessed';});
 setRows($('raw-values'),[
  ['Neighbourhood flood overlap',fmt(r.flood_exposure,'percent')],['Surface heat anomaly',fmt(r.heat_anomaly,'celsius')],
  ['Household green-space deficit',fmt(r.access_deficit,'percent')],['Income deprivation rate',fmt(r.income_rate,'percent')],
  ['Residents aged 0–4 or 65+',fmt(r.age_sensitive_share,'percent')],['Case: >1 km or no mapped park route',fmt(r.case_network_deficit,'percent')]
 ]);
 $('evidence-title').textContent=r.model_eligible?'Five indicators available for scoring':'Evidence incomplete for scoring';
 const missing=r.model_missing_indicators?String(r.model_missing_indicators).replaceAll('_',' '):'one or more screening inputs';
 $('evidence-text').textContent='Usable heat coverage: '+fmt(r.heat_coverage,'percent')+'. '+(r.model_eligible?'Completeness does not remove source limitations.':'Missing or inadequate '+missing+'. No weights are redistributed around missing evidence.');
 setRows($('sensitivity-values'),[
  ['Top-group selection frequency',r.model_eligible?fmt(r.top20_frequency,'percent'):'Unassessed'],
  ['Rank range across weight draws',r.model_eligible?fmt(r.rank_p05,'rank')+'–'+fmt(r.rank_p95,'rank'):'Unassessed'],
  ['Stable priority under tested weights',r.model_eligible?(r.robust_priority?'Yes':'No'):'Unassessed']
 ]);
 $('follow-up').textContent=r.follow_up?String(r.follow_up):'Use the evidence and its limitations to identify the next local investigation.';
 paths.forEach((path,j)=>{path.classList.toggle('selected',i===j);path.setAttribute('tabindex',j===i?'0':'-1');path.setAttribute('aria-pressed',j===i?'true':'false');});
 if(focus)paths[i].focus();
}
function fitVisible(){
 if(!visible.length){setView(data.view.slice());return;}
 const boxes=visible.map(i=>rows[i].bounds),left=Math.min(...boxes.map(b=>b[0])),top=Math.min(...boxes.map(b=>b[1]));
 const right=Math.max(...boxes.map(b=>b[0]+b[2])),bottom=Math.max(...boxes.map(b=>b[1]+b[3]));
 const w=Math.max(1500,right-left),h=Math.max(1500,bottom-top),cx=(left+right)/2,cy=(top+bottom)/2;
 setView([cx-w*.55,cy-h*.55,w*1.1,h*1.1]);
}
function filterRows(fit=false){
 const borough=$('borough').value,query=$('search').value.trim().toLocaleLowerCase();
 visible=[];rows.forEach((r,i)=>{const match=(!borough||r.borough===borough)&&(!query||[r.name,r.lsoa21cd,r.borough].join(' ').toLocaleLowerCase().includes(query));paths[i].style.display=match?'':'none';if(match)visible.push(i);});
 $('filter-count').textContent=fmt(visible.length,'count')+' of '+fmt(rows.length,'count')+' neighbourhoods shown';
 $('neighbourhood').replaceChildren(new Option(visible.length?'Choose from the visible areas':'No matching neighbourhoods',''));
 visible.forEach(i=>$('neighbourhood').append(new Option(rows[i].name+' · '+rows[i].lsoa21cd,String(i))));
 if(selected!==null&&!visible.includes(selected))inspect(null);else if(selected!==null)$('neighbourhood').value=String(selected);
 paths.forEach((p,i)=>p.setAttribute('tabindex',i===(selected??visible[0])?'0':'-1'));
 $('tooltip').hidden=true;if(fit)fitVisible();
}
function updateScale(){
 const transform=svg.getScreenCTM();if(!transform)return;
 const px=Math.abs(transform.a),target=100/px,power=10**Math.floor(Math.log10(target)),ratio=target/power;
 const metres=(ratio>=5?5:ratio>=2?2:1)*power;
 $('scale-line').style.width=(metres*px)+'px';$('scale-label').textContent=metres>=1000?(metres/1000).toLocaleString('en-GB')+' km':metres.toLocaleString('en-GB')+' m';
}
function setView(next){view=next;svg.setAttribute('viewBox',view.join(' '));updateScale();}
function zoom(factor,centre){const newWidth=view[2]*factor;if(newWidth<data.view[2]/100||newWidth>data.view[2]*1.5)return;const [cx,cy]=centre||[view[0]+view[2]/2,view[1]+view[3]/2];setView([cx+(view[0]-cx)*factor,cy+(view[1]-cy)*factor,newWidth,view[3]*factor]);}
function point(event){const p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;const q=p.matrixTransform(svg.getScreenCTM().inverse());return [q.x,q.y];}
data.metrics.forEach(m=>$('metric').append(new Option(m.label,m.field)));
$('total-count').textContent=fmt(rows.length,'count');$('eligible-count').textContent=fmt(rows.filter(r=>r.model_eligible).length,'count');$('stable-count').textContent=fmt(rows.filter(r=>r.model_eligible&&r.robust_priority).length,'count');
$('stability-rule').textContent='A stable priority belongs to the top '+fmt(data.topFraction,'percent')+' group in all three scenarios and in at least '+fmt(data.stableThreshold,'percent')+' of '+fmt(data.samples,'count')+' weight draws. Ties can enlarge the top group.';
$('normalisation-note').textContent=finite(data.normalisationJaccard)?'Weight stability is conditional on the chosen model. Switching midrank to min-max normalisation gives a balanced top-group Jaccard overlap of '+data.normalisationJaccard.toFixed(3)+' (1 means identical groups). This is a reason to review component evidence before acting.':'Weight stability is conditional on the chosen normalisation and model; it does not establish stability to every modelling choice.';
$('metric').addEventListener('change',updateLegend);$('borough').addEventListener('change',()=>filterRows(true));$('search').addEventListener('input',()=>filterRows(false));
$('neighbourhood').addEventListener('change',event=>inspect(event.target.value===''?null:Number(event.target.value)));
$('zoom-in').addEventListener('click',()=>zoom(.8));$('zoom-out').addEventListener('click',()=>zoom(1.25));$('fit').addEventListener('click',fitVisible);
paths.forEach((path,i)=>{
 path.addEventListener('pointerenter',()=>{if(!drag)showTooltip(i);});path.addEventListener('pointerleave',()=>{$('tooltip').hidden=true;});
 path.addEventListener('focus',()=>showTooltip(i));path.addEventListener('blur',()=>{$('tooltip').hidden=true;});
 path.addEventListener('keydown',event=>{
  if(event.key==='Enter'||event.key===' '){event.preventDefault();inspect(i);return;}
  let next=null,position=visible.indexOf(i);
  if(['ArrowRight','ArrowDown'].includes(event.key))next=visible[(position+1)%visible.length];
  if(['ArrowLeft','ArrowUp'].includes(event.key))next=visible[(position-1+visible.length)%visible.length];
  if(event.key==='Home')next=visible[0];if(event.key==='End')next=visible[visible.length-1];
  if(next!==null&&next!==undefined){event.preventDefault();paths.forEach((p,j)=>p.setAttribute('tabindex',j===next?'0':'-1'));paths[next].focus();}
 });
});
svg.addEventListener('pointerdown',event=>{if(event.button!==0)return;const transform=svg.getScreenCTM();drag={x:event.clientX,y:event.clientY,view:view.slice(),scale:transform.a,index:event.target.dataset.i};moved=false;svg.setPointerCapture(event.pointerId);svg.classList.add('dragging');$('tooltip').hidden=true;});
svg.addEventListener('pointermove',event=>{if(!drag)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y;if(Math.hypot(dx,dy)>3)moved=true;setView([drag.view[0]-dx/drag.scale,drag.view[1]-dy/drag.scale,drag.view[2],drag.view[3]]);});
svg.addEventListener('pointerup',event=>{if(!drag)return;const index=drag.index;if(!moved&&index!==undefined)inspect(Number(index));drag=null;svg.classList.remove('dragging');if(svg.hasPointerCapture(event.pointerId))svg.releasePointerCapture(event.pointerId);});
svg.addEventListener('pointercancel',()=>{drag=null;svg.classList.remove('dragging');});
svg.addEventListener('wheel',event=>{event.preventDefault();zoom(event.deltaY>0?1.14:.88,point(event));},{passive:false});
new ResizeObserver(updateScale).observe(svg);
updateLegend();filterRows();setView(data.view.slice());
</script>
</body></html>'''
