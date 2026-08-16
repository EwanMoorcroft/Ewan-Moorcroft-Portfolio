"""Build the public FPL results walkthrough from the retained evaluation report.

The builder reads one committed JSON report. It does not download data, train a
model, fit new parameters, or create new predictions. Keeping the chart and
table generated from the same source prevents the presentation numbers from
drifting away from the report used by the project.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "retained" / "2024-25-gw01-15-evaluation.json"
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"

MODEL_ORDER = (
    ("last_gameweek", "Last gameweek", False),
    ("rolling_3_mean", "Three-observation mean", False),
    ("training_mean", "Training-window mean", False),
    ("ridge_regression", "Ridge regression", True),
)


def load_report() -> dict[str, object]:
    """Load the single committed report used by every walkthrough artifact."""

    with REPORT.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict) or not isinstance(report.get("models"), dict):
        raise ValueError("The retained report must contain a models mapping")
    return report


def model_rows(report: dict[str, object]) -> list[dict[str, object]]:
    """Return the report's aggregate model rows in presentation order."""

    models = report["models"]
    if not isinstance(models, dict):
        raise ValueError("The retained report models value must be a mapping")

    rows: list[dict[str, object]] = []
    for key, label, is_ridge in MODEL_ORDER:
        metrics = models.get(key)
        if not isinstance(metrics, dict):
            raise ValueError(f"Missing aggregate metrics for {key}")
        required = ("mae", "ndcg_at_10", "rmse", "spearman", "r2", "top_10_overlap")
        if any(name not in metrics for name in required):
            raise ValueError(f"Incomplete aggregate metrics for {key}")
        rows.append(
            {
                "candidate": label,
                "model_key": key,
                "mae": metrics["mae"],
                "ndcg_at_10": metrics["ndcg_at_10"],
                "rmse": metrics["rmse"],
                "spearman": metrics["spearman"],
                "r2": metrics["r2"],
                "top_10_overlap": metrics["top_10_overlap"],
                "rows": metrics.get("rows", ""),
                "gameweeks": metrics.get("gameweeks", ""),
                "is_ridge": is_ridge,
            }
        )
    return rows


def write_results(rows: list[dict[str, object]]) -> None:
    """Write the exact aggregate values used by the chart."""

    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = (
        "candidate",
        "model_key",
        "mae",
        "ndcg_at_10",
        "rmse",
        "spearman",
        "r2",
        "top_10_overlap",
        "rows",
        "gameweeks",
    )
    with (RESULTS / "model-comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def svg_text(x: float, y: float, value: str, *, size: int = 14, anchor: str = "start") -> str:
    """Return a small escaped SVG text element."""

    return (
        f'<text x="{x:g}" y="{y:g}" font-family="Arial, sans-serif" '
        f'font-size="{size}px" text-anchor="{anchor}" fill="#17202a">'
        f"{html.escape(value)}</text>"
    )


def chart_panel(
    rows: list[dict[str, object]],
    *,
    title: str,
    metric: str,
    maximum: float,
    left: float,
    top: float,
    width: float,
    height: float,
    lower_is_better: bool,
) -> list[str]:
    """Create one deterministic bar-chart panel."""

    colours = ["#7286a0", "#5d9a8b", "#b28b62", "#2f6f8e"]
    bottom = top + height
    parts = [svg_text(left, top - 35, title, size=18)]
    for tick in range(5):
        value = maximum * tick / 4
        y = bottom - height * tick / 4
        parts.append(
            f'<line x1="{left:g}" y1="{y:g}" x2="{left + width:g}" y2="{y:g}" '
            'stroke="#d9e1e8" stroke-width="1"/>'
        )
        parts.append(svg_text(left - 12, y + 5, f"{value:.2f}", size=11, anchor="end"))

    bar_width = 62
    gap = (width - bar_width * len(rows)) / (len(rows) + 1)
    for index, row in enumerate(rows):
        value = float(row[metric])
        bar_height = height * value / maximum if maximum else 0
        x = left + gap * (index + 1) + bar_width * index
        y = bottom - bar_height
        colour = colours[index]
        parts.append(
            f'<rect x="{x:g}" y="{y:g}" width="{bar_width:g}" height="{bar_height:g}" '
            f'fill="{colour}" rx="4"/>'
        )
        parts.append(svg_text(x + bar_width / 2, y - 10, f"{value:.3f}", size=11, anchor="middle"))
        label = str(row["candidate"])
        # Two lines keep the labels legible without relying on an external chart library.
        label_lines = {
            "Last gameweek": ("Last", "gameweek"),
            "Three-observation mean": ("Three-observation", "mean"),
            "Training-window mean": ("Training-window", "mean"),
            "Ridge regression": ("Ridge", "regression"),
        }
        first, second = label_lines[label]
        parts.append(svg_text(x + bar_width / 2, bottom + 27, first, size=11, anchor="middle"))
        if second:
            parts.append(svg_text(x + bar_width / 2, bottom + 43, second, size=11, anchor="middle"))

    direction = "lower is better" if lower_is_better else "higher is better"
    parts.append(svg_text(left, bottom + 78, direction, size=12))
    return parts


def write_svg(report: dict[str, object], rows: list[dict[str, object]]) -> None:
    """Write the two-panel model comparison figure."""

    data = report.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("The retained report data value must be a mapping")
    width, height = 1240, 700
    parts = [
        svg_text(42, 44, "FPL next-gameweek model comparison", size=26),
        svg_text(
            42,
            76,
            "Expanding-window evaluation over GW6 to GW15, using 6,684 player-gameweek rows",
            size=15,
        ),
        svg_text(
            42,
            101,
            f"{data.get('folds', 10)} chronological folds | metrics from the retained public-data report",
            size=13,
        ),
    ]
    parts.extend(
        chart_panel(
            rows,
            title="Mean absolute error (MAE)",
            metric="mae",
            maximum=1.65,
            left=90,
            top=195,
            width=500,
            height=300,
            lower_is_better=True,
        )
    )
    parts.extend(
        chart_panel(
            rows,
            title="Ranking quality (NDCG@10)",
            metric="ndcg_at_10",
            maximum=0.30,
            left=650,
            top=195,
            width=500,
            height=300,
            lower_is_better=False,
        )
    )
    parts.extend(
        [
            svg_text(
                42,
                625,
                "Ridge is the only learned model; the other bars are simple baselines.",
                size=13,
            ),
            svg_text(42, 650, "Source: reports/retained/2024-25-gw01-15-evaluation.json", size=12),
        ]
    )
    document = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="#ffffff">'
        "</rect>" + "".join(parts) + "</svg>\n"
    )
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "model-comparison.svg").write_text(document, encoding="utf-8")


def main() -> None:
    report = load_report()
    rows = model_rows(report)
    write_results(rows)
    write_svg(report, rows)


if __name__ == "__main__":
    main()
