from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from fpl_forecasting.retained import compare_retained_folds


def small_report():
    return {
        "folds": [
            {
                "test_gameweek_start": 6,
                "test_gameweek_end": 6,
                "test_rows": 10,
                "metrics": {
                    "ridge_regression": {"mae": 1.0, "rmse": 2.0},
                    "rolling_3_mean": {"mae": 2.0, "rmse": 3.0},
                },
            },
            {
                "test_gameweek_start": 7,
                "test_gameweek_end": 7,
                "test_rows": 20,
                "metrics": {
                    "ridge_regression": {"mae": 3.0, "rmse": 4.0},
                    "rolling_3_mean": {"mae": 2.0, "rmse": 5.0},
                },
            },
        ]
    }


def test_paired_fold_summary_preserves_sign_and_equal_week_weight():
    report = small_report()
    original = copy.deepcopy(report)
    summary, rows = compare_retained_folds(report)
    assert summary["fold_count"] == 2
    assert summary["test_gameweek_range"] == [6, 7]
    metric = summary["comparisons"]["rolling_3_mean"]["mae"]
    assert metric["mean_ridge_minus_baseline"] == 0
    assert metric["minimum"] == -1
    assert metric["maximum"] == 1
    assert metric["ridge_wins"] == metric["ridge_losses"] == 1
    assert metric["leave_one_gameweek_out_mean_range"] == [-1.0, 1.0]
    assert [row["ridge_minus_baseline"] for row in rows if row["metric"] == "mae"] == [-1, 1]
    assert report == original


@pytest.mark.parametrize("mutation", ["duplicate", "nonfinite", "multiweek"])
def test_comparison_rejects_unusable_fold_evidence(mutation):
    report = small_report()
    if mutation == "duplicate":
        report["folds"].append(copy.deepcopy(report["folds"][0]))
    elif mutation == "nonfinite":
        report["folds"][0]["metrics"]["ridge_regression"]["mae"] = float("nan")
    else:
        report["folds"][0]["test_gameweek_end"] = 7
    with pytest.raises(ValueError):
        compare_retained_folds(report)


def test_committed_derived_summary_is_bound_to_unchanged_retained_bytes():
    root = Path(__file__).resolve().parents[1]
    report_path = root / "reports/retained/2024-25-gw01-15-evaluation.json"
    summary = json.loads((root / "results/retained-folds/comparison.json").read_text())
    assert summary["source_report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    calculated, _ = compare_retained_folds(json.loads(report_path.read_bytes()))
    assert summary["comparisons"] == calculated["comparisons"]
