from __future__ import annotations

import json
from pathlib import Path

from fpl_forecasting.cli import main


def test_cli_synthetic_validate_and_evaluate(tmp_path: Path, capsys) -> None:
    gameweeks = tmp_path / "gameweeks"
    assert (
        main(
            [
                "synthetic",
                "--output-dir",
                str(gameweeks),
                "--gameweeks",
                "8",
                "--players",
                "12",
                "--seed",
                "9",
                "--season",
                "2024-25",
            ]
        )
        == 0
    )
    generated = json.loads(capsys.readouterr().out)
    assert generated["season_id"] == "2024-25"

    assert main(["validate", "--gameweek-dir", str(gameweeks)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["gameweeks"] == 8

    report_json = tmp_path / "report.json"
    predictions_csv = tmp_path / "predictions.csv"
    report_html = tmp_path / "report.html"
    assert (
        main(
            [
                "evaluate",
                "--gameweek-dir",
                str(gameweeks),
                "--report-json",
                str(report_json),
                "--predictions-csv",
                str(predictions_csv),
                "--report-html",
                str(report_html),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert report_json.is_file()
    assert predictions_csv.is_file()
    assert report_html.is_file()
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["report_format"] == "fpl-rolling-evaluation-v1"
    assert "ridge_regression" in report["models"]
