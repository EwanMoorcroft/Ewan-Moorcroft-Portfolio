"""Build small, public-facing X-ray walkthrough artifacts from retained evidence.

This script deliberately reads only committed JSON evidence. It does not access
the image dataset, download files, train a model, or create new model results.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"


def load_json(name: str) -> dict:
    with (EVIDENCE / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_csv(name: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def svg_text(x: float, y: float, value: str, *, size: int = 14, anchor: str = "start") -> str:
    return (
        f'<text x="{x:g}" y="{y:g}" font-family="Arial, sans-serif" '
        f'font-size="{size}px" text-anchor="{anchor}" fill="#17202a">'
        f"{html.escape(value)}</text>"
    )


def write_svg(name: str, content: str, width: int, height: int) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    document = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        f"{content}</svg>\n"
    )
    (ASSETS / name).write_text(document, encoding="utf-8")


def composition_svg(dataset: dict) -> None:
    classes = dataset["classes"]
    colours = ["#3264a8", "#d97732", "#4f8a5b"]
    width, height = 900, 520
    left, chart_top, chart_width, chart_height = 100, 145, 690, 270
    maximum = max(item["expected_count"] for item in classes.values())
    parts = [
        svg_text(45, 42, "Chest X-ray dataset composition", size=26),
        svg_text(45, 72, "Expected counts from the repository data contract", size=16),
        svg_text(45, 98, "Not a fresh scan of the excluded image files", size=14),
        f'<line x1="{left}" y1="{chart_top + chart_height}" x2="{left + chart_width}" '
        f'y2="{chart_top + chart_height}" stroke="#34495e" stroke-width="2"/>',
    ]
    for tick in range(0, maximum + 1, 250):
        y = chart_top + chart_height - (tick / maximum) * chart_height
        parts.append(
            f'<line x1="{left}" y1="{y:g}" x2="{left + chart_width}" y2="{y:g}" '
            'stroke="#d9e1e8" stroke-width="1"/>'
        )
        parts.append(svg_text(left - 12, y + 5, f"{tick:,}", size=12, anchor="end"))
    bar_width = 150
    gap = 75
    for index, (label, item) in enumerate(classes.items()):
        x = left + gap + index * (bar_width + gap)
        bar_height = item["expected_count"] / maximum * chart_height
        y = chart_top + chart_height - bar_height
        parts.append(
            f'<rect x="{x}" y="{y:g}" width="{bar_width}" height="{bar_height:g}" '
            f'fill="{colours[index]}" rx="4"/>'
        )
        parts.append(
            svg_text(x + bar_width / 2, y - 10, f"{item['expected_count']:,}", anchor="middle")
        )
        parts.append(
            svg_text(x + bar_width / 2, chart_top + chart_height + 30, label, anchor="middle")
        )
    parts.append(svg_text(45, height - 22, "Source: data/dataset-spec.json", size=12))
    write_svg("dataset-composition.svg", "".join(parts), width, height)


def model_results_svg(results: dict) -> None:
    width, height = 1100, 650
    parts = [
        svg_text(42, 42, "Chest X-ray model results", size=26),
        svg_text(42, 72, "Earlier run - not reproduced here", size=16),
        svg_text(
            42,
            98,
            "The earlier image-level split had five exact duplicate pairs identified afterwards",
            size=14,
        ),
    ]
    overall = [
        ("Accuracy", results["test"]["accuracy"]),
        ("Balanced accuracy", results["test"]["balanced_accuracy"]),
        ("Macro F1", results["test"]["macro_f1"]),
        ("MCC", results["test"]["matthews_correlation_coefficient"]),
    ]
    classes = results["test"]["per_class"]
    left_x, top, panel_w, panel_h = 75, 155, 410, 330
    right_x = 610
    for panel_x, title in ((left_x, "Aggregate test metrics"), (right_x, "Per-class F1")):
        parts.append(svg_text(panel_x, top - 28, title, size=18))
        parts.append(
            f'<line x1="{panel_x}" y1="{top + panel_h}" x2="{panel_x + panel_w}" '
            f'y2="{top + panel_h}" stroke="#34495e" stroke-width="2"/>'
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + panel_h - tick * panel_h
            parts.append(
                f'<line x1="{panel_x}" y1="{y:g}" x2="{panel_x + panel_w}" y2="{y:g}" '
                'stroke="#d9e1e8" stroke-width="1"/>'
            )
            parts.append(svg_text(panel_x - 10, y + 5, f"{tick:.2f}", size=11, anchor="end"))
    colours = ["#3264a8", "#d97732", "#4f8a5b", "#8556a8"]
    bar_width = 62
    gap = 34
    for index, (label, value) in enumerate(overall):
        x = left_x + 28 + index * (bar_width + gap)
        bar_height = value * panel_h
        y = top + panel_h - bar_height
        parts.append(
            f'<rect x="{x}" y="{y:g}" width="{bar_width}" height="{bar_height:g}" '
            f'fill="{colours[index]}" rx="3"/>'
        )
        parts.append(svg_text(x + bar_width / 2, y - 9, f"{value:.4f}", size=11, anchor="middle"))
        parts.append(
            svg_text(x + bar_width / 2, top + panel_h + 28, label, size=11, anchor="middle")
        )
    class_colours = ["#3264a8", "#d97732", "#4f8a5b"]
    for index, (label, values) in enumerate(classes.items()):
        value = values["f1"]
        x = right_x + 48 + index * 116
        bar_height = value * panel_h
        y = top + panel_h - bar_height
        parts.append(
            f'<rect x="{x}" y="{y:g}" width="{bar_width}" height="{bar_height:g}" '
            f'fill="{class_colours[index]}" rx="3"/>'
        )
        parts.append(svg_text(x + bar_width / 2, y - 9, f"{value:.4f}", size=11, anchor="middle"))
        parts.append(
            svg_text(x + bar_width / 2, top + panel_h + 28, label, size=11, anchor="middle")
        )
    parts.append(svg_text(42, height - 35, "Source: evidence/retained-results.json", size=12))
    write_svg("model-results.svg", "".join(parts), width, height)


def main() -> None:
    retained_results = load_json("retained-results.json")
    spec = json.loads((ROOT / "data" / "dataset-spec.json").read_text(encoding="utf-8"))
    duplicates = load_json("known-exact-duplicates.json")
    if retained_results.get("reproduced_in_this_repository") is not False:
        raise ValueError("Retained evidence must remain explicitly unreproduced")
    if retained_results.get("evidence_status") != "retained earlier run":
        raise ValueError("Unexpected retained evidence status")
    write_csv(
        "overall-metrics.csv",
        ["metric", "value", "image_count", "evidence_status"],
        [
            {
                "metric": label,
                "value": value,
                "image_count": retained_results["test"]["image_count"],
                "evidence_status": retained_results["evidence_status"],
            }
            for label, value in (
                ("accuracy", retained_results["test"]["accuracy"]),
                ("balanced_accuracy", retained_results["test"]["balanced_accuracy"]),
                ("macro_f1", retained_results["test"]["macro_f1"]),
                (
                    "matthews_correlation_coefficient",
                    retained_results["test"]["matthews_correlation_coefficient"],
                ),
            )
        ],
    )
    write_csv(
        "per-class-metrics.csv",
        ["class", "precision", "recall", "f1", "support", "evidence_status"],
        [
            {"class": label, **values, "evidence_status": retained_results["evidence_status"]}
            for label, values in retained_results["test"]["per_class"].items()
        ],
    )
    write_csv(
        "duplicate-audit.csv",
        ["class", "left_path", "right_path", "sha256", "evidence_status"],
        [
            {
                "class": duplicates["class"],
                "left_path": pair["members"][0],
                "right_path": pair["members"][1],
                "sha256": pair["sha256"],
                "evidence_status": duplicates["evidence_status"],
            }
            for pair in duplicates["duplicate_groups"]
        ],
    )
    composition_svg(spec)
    model_results_svg(retained_results)


if __name__ == "__main__":
    main()
