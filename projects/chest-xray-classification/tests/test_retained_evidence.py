from __future__ import annotations

import json
import re
from pathlib import Path


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
