"""Compare retained gameweek aggregates without fitting models or reconstructing rows."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .contracts import canonical_json_bytes, sha256_bytes
from .output import publish_files, validate_output_paths

_METRICS = {
    "mae": False,
    "rmse": False,
    "spearman": True,
    "ndcg_at_10": True,
    "top_10_overlap": True,
    "r2": True,
}


def compare_retained_folds(report: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return paired, equally weighted gameweek differences and descriptive sensitivity."""
    folds = report["folds"]
    if len(folds) < 2:
        raise ValueError("At least two retained single-gameweek folds are required")
    seen: set[int] = set()
    baselines = sorted(set(folds[0]["metrics"]) - {"ridge_regression"})
    metrics = sorted(set(folds[0]["metrics"]["ridge_regression"]) & set(_METRICS))
    if not baselines or not metrics:
        raise ValueError("Paired baseline and ridge metrics are required")
    rows: list[dict[str, Any]] = []
    values: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in metrics} for name in baselines
    }
    for fold in sorted(folds, key=lambda item: item["test_gameweek_start"]):
        gw = fold["test_gameweek_start"]
        if gw != fold["test_gameweek_end"] or gw in seen:
            raise ValueError("Retained folds must represent distinct single gameweeks")
        seen.add(gw)
        for baseline in baselines:
            for metric in metrics:
                ridge = float(fold["metrics"]["ridge_regression"][metric])
                other = float(fold["metrics"][baseline][metric])
                if not math.isfinite(ridge) or not math.isfinite(other):
                    raise ValueError("Retained metrics must be finite")
                delta = ridge - other
                values[baseline][metric].append(delta)
                rows.append(
                    {
                        "gameweek": gw,
                        "rows": fold["test_rows"],
                        "baseline": baseline,
                        "metric": metric,
                        "ridge": ridge,
                        "baseline_value": other,
                        "ridge_minus_baseline": delta,
                    }
                )
    comparisons: dict[str, Any] = {}
    for baseline, metric_values in values.items():
        comparisons[baseline] = {}
        for metric, deltas in metric_values.items():
            leave_one_out = [
                statistics.mean(deltas[:index] + deltas[index + 1 :])
                for index in range(len(deltas))
            ]
            higher_better = _METRICS[metric]
            wins = sum(value > 0 if higher_better else value < 0 for value in deltas)
            ties = deltas.count(0)
            comparisons[baseline][metric] = {
                "gameweeks": len(deltas),
                "mean_ridge_minus_baseline": statistics.mean(deltas),
                "sample_standard_deviation": statistics.stdev(deltas),
                "minimum": min(deltas),
                "maximum": max(deltas),
                "ridge_wins": wins,
                "ties": ties,
                "ridge_losses": len(deltas) - wins - ties,
                "leave_one_gameweek_out_mean_range": [min(leave_one_out), max(leave_one_out)],
                "better_difference": "positive" if higher_better else "negative",
            }
    return {
        "analysis_format": "fpl-retained-fold-comparison-v1",
        "unit": "one completed gameweek; each retained fold has equal weight",
        "fold_count": len(folds),
        "test_gameweek_range": [min(seen), max(seen)],
        "comparisons": comparisons,
        "limitations": [
            "Derived only from retained fold aggregates; no source rows, predictions or new fits.",
            "Standard deviations and leave-one-gameweek-out ranges describe variation, not confidence intervals.",
            "Temporal folds can share training history; their independence must not be assumed.",
            "Equal-gameweek mean deltas differ from pooled row-weighted error metrics.",
            "No player, position, minutes, blank/double-gameweek or injury subgroup analysis is possible from these aggregates.",
            "Source coverage and importer checks require the source manifest and cannot be established from fold metrics alone.",
        ],
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    summary, rows = compare_retained_folds(json.loads(raw))
    summary["source_report_file"] = args.report.name
    summary["source_report_sha256"] = sha256_bytes(raw)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_path = args.output_dir / "per-gameweek.csv"
    summary_path = args.output_dir / "comparison.json"
    validate_output_paths([csv_path, summary_path], [args.report])
    publish_files(
        {csv_path: stream.getvalue().encode(), summary_path: canonical_json_bytes(summary)}
    )


if __name__ == "__main__":
    main()
