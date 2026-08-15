from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETAINED_ROOT = PROJECT_ROOT / "reports" / "retained"
EVALUATION_PATH = RETAINED_ROOT / "2024-25-gw01-15-evaluation.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_retained_evaluation_provenance_and_fold_contract() -> None:
    evaluation = _load_json(EVALUATION_PATH)
    evidence = evaluation["evidence"]

    manifest_name = evidence["source_manifest_file"]
    assert Path(manifest_name).name == manifest_name
    manifest_path = RETAINED_ROOT / manifest_name
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_json(manifest_path)

    assert hashlib.sha256(manifest_bytes).hexdigest() == evidence["source_manifest_sha256"]

    aggregate = hashlib.sha256()
    source_files = manifest["source_files"]
    assert len(source_files) == manifest["source_file_count"] == 15
    for source_file in source_files:
        line = f"{source_file['file']}\0{source_file['bytes']}\0{source_file['sha256']}\n"
        aggregate.update(line.encode("ascii"))
    assert aggregate.hexdigest() == manifest["source_files_sha256"]
    assert manifest["source_files_sha256"] == evidence["source_files_sha256"]

    data = evaluation["data"]
    folds = evaluation["folds"]
    assert len(folds) == data["folds"] == 10
    assert data["evaluated_gameweek_start"] == 6
    assert data["evaluated_gameweek_end"] == 15
    assert [fold["test_gameweek_start"] for fold in folds] == list(range(6, 16))
    assert [fold["test_gameweek_end"] for fold in folds] == list(range(6, 16))
    assert sum(fold["test_rows"] for fold in folds) == data["evaluated_rows"] == 6684

    model_names = set(evaluation["models"])
    for fold in folds:
        assert fold["train_gameweek_start"] == data["target_gameweek_start"] == 2
        assert fold["train_gameweek_end"] < fold["test_gameweek_start"]
        assert fold["train_gameweek_end"] + 1 == fold["test_gameweek_start"]
        assert set(fold["metrics"]) == model_names
        for metrics in fold["metrics"].values():
            assert metrics["rows"] == fold["test_rows"]
            assert metrics["gameweeks"] == 1

    for metrics in evaluation["models"].values():
        assert metrics["rows"] == data["evaluated_rows"] == 6684
        assert metrics["gameweeks"] == data["folds"] == 10
