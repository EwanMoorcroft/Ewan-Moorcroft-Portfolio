from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from chest_xray_benchmark.metrics import classification_metrics


def test_grouped_result_evidence_is_compact_and_internally_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence_path = root / "evidence" / "retained-results.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    spec = json.loads((root / "data" / "dataset-spec.json").read_text(encoding="utf-8"))

    assert evidence["evidence_status"] == "retained grouped-split evaluation"
    assert evidence["reproduced_in_this_repository"] is True
    assert evidence["dataset"]["doi"] == spec["doi"]
    assert evidence["dataset"]["image_count"] == spec["expected_total"] == 3475
    assert evidence["split"]["split_ready"] is True
    assert evidence["split"]["leakage_checks_passed"] is True
    assert evidence["split"]["sha256_cross_partition_violation_count"] == 0
    assert sum(part["images"] for part in evidence["split"]["partitions"].values()) == 3475

    test = evidence["test"]
    assert test["macro_f1"] == 0.8097125109045282
    assert sum(sum(row) for row in test["confusion_matrix"]) == test["image_count"] == 522
    assert sum(values["support"] for values in test["per_class"].values()) == 522

    digest_pattern = re.compile(r"[0-9a-f]{64}")
    for name in ("split_file_sha256", "config_file_sha256", "checkpoint_file_sha256"):
        assert digest_pattern.fullmatch(evidence["provenance"][name])

    serialized = evidence_path.read_text(encoding="utf-8")
    assert "visual_review_candidate_pairs" not in serialized
    assert evidence_path.stat().st_size < 20_000


def test_corrected_test_metrics_reconstruct_from_retained_probabilities():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads((root / "evidence/retained-results.json").read_text())
    revision = evidence["reevaluation"]
    prediction_path = root / revision["predictions"]
    assert (
        hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        == revision["predictions_file_sha256"]
    )
    with prediction_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len({row["row_id"] for row in rows}) == len(rows) == 522
    labels = np.array([int(row["true_label_index"]) for row in rows])
    probabilities = np.array(
        [[float(row[f"probability_{index}"]) for index in range(3)] for row in rows]
    )
    recomputed = classification_metrics(
        labels, probabilities, evidence["provenance"]["label_order"]
    )
    assert recomputed["confusion_matrix"] == evidence["test"]["confusion_matrix"]
    for metric in (
        "macro_f1",
        "log_loss",
        "multiclass_brier_score",
        "expected_calibration_error_15_bins",
    ):
        assert recomputed[metric] == pytest.approx(evidence["test"][metric], abs=1e-12)
    assert evidence["test"]["loss"] == pytest.approx(recomputed["log_loss"], abs=1e-6)
