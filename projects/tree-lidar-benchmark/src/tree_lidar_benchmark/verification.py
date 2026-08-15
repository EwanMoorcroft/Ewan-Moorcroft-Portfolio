"""Integrity checks for retained benchmark tables and visual evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

from tree_lidar_benchmark.evaluator import (
    EVALUATION_MASK,
    MATCHING_POLICY,
    PROTOCOL_ID,
    precision_recall_f1,
)

IOU_THRESHOLD = 0.5
EXPECTED_ROWS = {"per_plot": 132, "by_site": 60, "overall": 12}
EXPECTED_METHODS = {
    "forestformer3d",
    "forainet",
    "segmentanytree",
    "tls2trees",
    "treelearn",
    "treex",
}
EXPECTED_VARIANTS = {"published_default", "development_tuned"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

IDENTITY_FIELDS = (
    "method",
    "variant",
    "run_id",
    "training_mode",
    "checkpoint_identity",
    "checkpoint_sha256",
    "split",
)
PER_PLOT_FIELDS = (
    *IDENTITY_FIELDS,
    "site",
    "plot_id",
    "point_count",
    "evaluated_point_count",
    "predicted_instances",
    "reference_instances",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "mean_matched_iou",
    "median_matched_iou",
    "evaluation_protocol",
    "matching_policy",
    "evaluation_mask",
    "iou_threshold",
    "source_metrics_sha256",
)
INTEGER_AGGREGATE_FIELDS = (
    "plots",
    "point_count",
    "evaluated_point_count",
    "predicted_instances",
    "reference_instances",
    "true_positives",
    "false_positives",
    "false_negatives",
)
FLOAT_AGGREGATE_FIELDS = (
    "mean_plot_precision",
    "mean_plot_recall",
    "mean_plot_f1",
    "median_plot_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "mean_of_plot_mean_matched_iou",
)
BY_SITE_FIELDS = (
    *IDENTITY_FIELDS,
    "site",
    *INTEGER_AGGREGATE_FIELDS,
    *FLOAT_AGGREGATE_FIELDS,
)
OVERALL_FIELDS = tuple(field for field in BY_SITE_FIELDS if field != "site")

PLOT_INVENTORY: Mapping[str, tuple[str, int, int]] = {
    "CULS/plot_2_annotated": ("CULS", 3_946_098, 20),
    "NIBIO/plot_17_annotated": ("NIBIO", 6_890_118, 30),
    "NIBIO/plot_18_annotated": ("NIBIO", 6_915_118, 27),
    "NIBIO/plot_1_annotated": ("NIBIO", 5_000_698, 37),
    "NIBIO/plot_22_annotated": ("NIBIO", 6_366_607, 20),
    "NIBIO/plot_23_annotated": ("NIBIO", 6_163_377, 28),
    "NIBIO/plot_5_annotated": ("NIBIO", 5_223_631, 19),
    "RMIT/test": ("RMIT", 357_435, 64),
    "SCION/plot_31_annotated": ("SCION", 2_977_537, 25),
    "SCION/plot_61_annotated": ("SCION", 3_589_254, 18),
    "TUWIEN/test": ("TUWIEN", 2_280_049, 35),
}


class IntegrityError(ValueError):
    """Raised when retained evidence differs from its declared contract."""


@dataclass(frozen=True)
class VerificationReport:
    """Compact summary returned after all integrity checks pass."""

    status: str
    protocol: str
    per_plot_rows: int
    by_site_rows: int
    overall_rows: int
    result_identities: int
    aggregate_values_checked: int
    retained_hashes_checked: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready report."""

        return asdict(self)


EvidencePath = Path | Traversable


def _has_evidence(root: EvidencePath) -> bool:
    required = (
        root / "results_manifest.json",
        root / "data" / "route_manifest.json",
        root / "data" / "benchmark_per_plot.csv",
        root / "data" / "benchmark_by_site.csv",
        root / "data" / "benchmark_overall.csv",
    )
    return all(path.is_file() for path in required)


def _project_root(root: Path | None) -> EvidencePath:
    if root is not None:
        return root.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if _has_evidence(repository_root):
        return repository_root
    packaged_root = files("tree_lidar_benchmark") / "_evidence"
    if _has_evidence(packaged_root):
        return packaged_root
    raise IntegrityError("Cannot locate retained benchmark evidence")


def _read_csv(path: EvidencePath, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != tuple(expected_fields):
                raise IntegrityError(f"Unexpected CSV header: {path.name}")
            rows: list[dict[str, str]] = []
            for row in reader:
                if not row or not any((value or "").strip() for value in row.values()):
                    continue
                if None in row or any(value is None for value in row.values()):
                    raise IntegrityError(f"Malformed CSV record: {path.name}")
                rows.append({field: value.strip() for field, value in row.items()})
    except (OSError, UnicodeError, csv.Error) as exc:
        raise IntegrityError(f"Cannot read {path.name}: {exc}") from exc
    return rows


def _integer(value: str, field: str) -> int:
    if not re.fullmatch(r"[0-9]+", value):
        raise IntegrityError(f"Invalid non-negative integer in {field}: {value!r}")
    return int(value)


def _number(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise IntegrityError(f"Invalid number in {field}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise IntegrityError(f"Non-finite number in {field}")
    return parsed


def _sha256(path: EvidencePath) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: EvidencePath) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"Cannot read {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntegrityError(f"Expected a JSON object: {path.name}")
    return payload


def _route_specs(path: EvidencePath) -> dict[tuple[str, str], dict[str, str]]:
    payload = _load_json(path)
    expected_protocol = {
        "evaluation_mask": EVALUATION_MASK,
        "id": PROTOCOL_ID,
        "iou_threshold": IOU_THRESHOLD,
        "matching_policy": MATCHING_POLICY,
    }
    if payload.get("schema_version") != 2 or payload.get("protocol") != expected_protocol:
        raise IntegrityError("Route manifest protocol differs")
    if payload.get("expected_rows") != EXPECTED_ROWS:
        raise IntegrityError("Route manifest row counts differ")
    if payload.get("legacy_protocols_included") != []:
        raise IntegrityError("Earlier protocol rows are not permitted")
    if payload.get("registered_without_current_result") != []:
        raise IntegrityError("A declared route has no retained result")
    identities = payload.get("identities")
    if not isinstance(identities, list) or len(identities) != 12:
        raise IntegrityError("Route manifest must declare twelve identities")

    specs: dict[tuple[str, str], dict[str, str]] = {}
    for raw in identities:
        if not isinstance(raw, dict):
            raise IntegrityError("Route identity must be an object")
        key = (str(raw.get("method", "")), str(raw.get("variant", "")))
        if key in specs:
            raise IntegrityError(f"Duplicate route identity: {key}")
        if raw.get("evaluation_protocol") != PROTOCOL_ID:
            raise IntegrityError(f"Protocol drift in route identity: {key}")
        if raw.get("matching_policy") != MATCHING_POLICY:
            raise IntegrityError(f"Matching drift in route identity: {key}")
        if raw.get("evaluation_mask") != EVALUATION_MASK:
            raise IntegrityError(f"Mask drift in route identity: {key}")
        checkpoint_hash = raw.get("checkpoint_sha256")
        specs[key] = {
            "run_id": str(raw.get("run_id", "")),
            "training_mode": str(raw.get("training_mode", "")),
            "checkpoint_identity": str(raw.get("checkpoint_identity", "")),
            "checkpoint_sha256": "" if checkpoint_hash is None else str(checkpoint_hash),
        }
    return specs


def _validate_per_plot(
    rows: Sequence[dict[str, str]],
    specs: Mapping[tuple[str, str], Mapping[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    if len(rows) != EXPECTED_ROWS["per_plot"]:
        raise IntegrityError("Per-plot row count differs")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["method"], row["variant"])
        if key not in specs:
            raise IntegrityError(f"Unsupported result identity: {key}")
        expected_identity = specs[key]
        for field in ("run_id", "training_mode", "checkpoint_identity", "checkpoint_sha256"):
            if row[field] != expected_identity[field]:
                raise IntegrityError(f"Identity drift for {key} in {field}")
        if row["split"] != "test":
            raise IntegrityError(f"Non-held-out row in {key}")
        if (
            row["evaluation_protocol"] != PROTOCOL_ID
            or row["matching_policy"] != MATCHING_POLICY
            or row["evaluation_mask"] != EVALUATION_MASK
            or _number(row["iou_threshold"], "iou_threshold") != IOU_THRESHOLD
        ):
            raise IntegrityError(f"Scoring contract drift for {key}")
        plot_id = row["plot_id"]
        expected_plot = PLOT_INVENTORY.get(plot_id)
        if expected_plot is None:
            raise IntegrityError(f"Unknown plot: {plot_id}")
        point_count = _integer(row["point_count"], "point_count")
        references = _integer(row["reference_instances"], "reference_instances")
        if (row["site"], point_count, references) != expected_plot:
            raise IntegrityError(f"Inventory drift for {plot_id}")
        evaluated = _integer(row["evaluated_point_count"], "evaluated_point_count")
        predicted = _integer(row["predicted_instances"], "predicted_instances")
        tp = _integer(row["true_positives"], "true_positives")
        fp = _integer(row["false_positives"], "false_positives")
        fn = _integer(row["false_negatives"], "false_negatives")
        if evaluated > point_count:
            raise IntegrityError(f"Evaluated point count exceeds source count for {plot_id}")
        if predicted != tp + fp or references != tp + fn:
            raise IntegrityError(f"Instance count identity differs for {key}, {plot_id}")
        expected_metrics = precision_recall_f1(tp, fp, fn)
        for field, expected in zip(("precision", "recall", "f1"), expected_metrics, strict=True):
            if not math.isclose(_number(row[field], field), expected, rel_tol=0.0, abs_tol=1e-15):
                raise IntegrityError(f"Metric drift for {key}, {plot_id}, {field}")
        mean_iou = _number(row["mean_matched_iou"], "mean_matched_iou")
        if not 0.0 <= mean_iou <= 1.0 or (tp and mean_iou < IOU_THRESHOLD):
            raise IntegrityError(f"Matched IoU differs for {key}, {plot_id}")
        if not SHA256_PATTERN.fullmatch(row["source_metrics_sha256"]):
            raise IntegrityError(f"Invalid source metric hash for {key}, {plot_id}")
        unique = (key[0], key[1], plot_id)
        if unique in seen:
            raise IntegrityError(f"Duplicate plot row: {unique}")
        seen.add(unique)
        grouped[key].append(row)

    if set(grouped) != set(specs):
        raise IntegrityError("Per-plot identities differ from the route manifest")
    if {key[0] for key in grouped} != EXPECTED_METHODS:
        raise IntegrityError("Method inventory differs")
    if {key[1] for key in grouped} != EXPECTED_VARIANTS:
        raise IntegrityError("Route inventory differs")
    expected_plots = set(PLOT_INVENTORY)
    for key, group in grouped.items():
        if len(group) != 11 or {row["plot_id"] for row in group} != expected_plots:
            raise IntegrityError(f"Held-out plot coverage differs for {key}")
        if sum(_integer(row["point_count"], "point_count") for row in group) != 49_709_922:
            raise IntegrityError(f"Source point total differs for {key}")
        if sum(_integer(row["reference_instances"], "reference_instances") for row in group) != 323:
            raise IntegrityError(f"Reference total differs for {key}")
    return dict(grouped)


def _mean(values: Iterable[float]) -> float:
    materialised = list(values)
    return math.fsum(materialised) / len(materialised)


def _aggregate(rows: Sequence[dict[str, str]]) -> dict[str, object]:
    first = rows[0]
    tp = sum(_integer(row["true_positives"], "true_positives") for row in rows)
    fp = sum(_integer(row["false_positives"], "false_positives") for row in rows)
    fn = sum(_integer(row["false_negatives"], "false_negatives") for row in rows)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    result: dict[str, object] = {field: first[field] for field in IDENTITY_FIELDS}
    result.update(
        {
            "plots": len(rows),
            "point_count": sum(_integer(row["point_count"], "point_count") for row in rows),
            "evaluated_point_count": sum(
                _integer(row["evaluated_point_count"], "evaluated_point_count") for row in rows
            ),
            "predicted_instances": sum(
                _integer(row["predicted_instances"], "predicted_instances") for row in rows
            ),
            "reference_instances": sum(
                _integer(row["reference_instances"], "reference_instances") for row in rows
            ),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "mean_plot_precision": _mean(_number(row["precision"], "precision") for row in rows),
            "mean_plot_recall": _mean(_number(row["recall"], "recall") for row in rows),
            "mean_plot_f1": _mean(_number(row["f1"], "f1") for row in rows),
            "median_plot_f1": float(statistics.median(_number(row["f1"], "f1") for row in rows)),
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
            "mean_of_plot_mean_matched_iou": _mean(
                _number(row["mean_matched_iou"], "mean_matched_iou") for row in rows
            ),
        }
    )
    return result


def _expected_aggregates(
    grouped: Mapping[tuple[str, str], Sequence[dict[str, str]]],
) -> tuple[dict[tuple[str, str, str], dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    by_site: dict[tuple[str, str, str], dict[str, object]] = {}
    overall: dict[tuple[str, str], dict[str, object]] = {}
    for key, rows in grouped.items():
        overall[key] = _aggregate(rows)
        for site in sorted({row["site"] for row in rows}):
            by_site[(key[0], key[1], site)] = _aggregate(
                [row for row in rows if row["site"] == site]
            )
    return by_site, overall


def _verify_aggregate_table(
    rows: Sequence[dict[str, str]],
    expected: Mapping[tuple[str, ...], Mapping[str, object]],
    *,
    include_site: bool,
) -> int:
    observed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = (
            (row["method"], row["variant"], row["site"])
            if include_site
            else (row["method"], row["variant"])
        )
        if key in observed:
            raise IntegrityError(f"Duplicate aggregate row: {key}")
        observed[key] = row
    if set(observed) != set(expected):
        raise IntegrityError("Aggregate identity inventory differs")

    checks = 0
    for key, wanted in expected.items():
        row = observed[key]
        for field in IDENTITY_FIELDS:
            if row[field] != wanted[field]:
                raise IntegrityError(f"Aggregate identity drift for {key}, {field}")
        for field in INTEGER_AGGREGATE_FIELDS:
            if _integer(row[field], field) != wanted[field]:
                raise IntegrityError(f"Aggregate integer drift for {key}, {field}")
            checks += 1
        for field in FLOAT_AGGREGATE_FIELDS:
            if _number(row[field], field) != wanted[field]:
                raise IntegrityError(f"Aggregate metric drift for {key}, {field}")
            checks += 1
    return checks


def _verify_retained_hashes(root: EvidencePath) -> int:
    payload = _load_json(root / "results_manifest.json")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("id") != PROTOCOL_ID:
        raise IntegrityError("Results manifest protocol differs")
    if payload.get("row_counts") != EXPECTED_ROWS:
        raise IntegrityError("Results manifest row counts differ")
    retained = payload.get("retained_files")
    if not isinstance(retained, list) or not retained:
        raise IntegrityError("Results manifest has no retained files")
    checked = 0
    for item in retained:
        if not isinstance(item, dict):
            raise IntegrityError("Retained file entry must be an object")
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise IntegrityError("Retained file path must be project-relative")
        expected_hash = str(item.get("sha256", ""))
        if not SHA256_PATTERN.fullmatch(expected_hash):
            raise IntegrityError(f"Invalid retained hash: {relative}")
        path = root / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            raise IntegrityError(f"Retained file hash differs: {relative}")
        checked += 1
    return checked


def load_overall_rows(root: Path | None = None) -> list[dict[str, str]]:
    """Load retained overall rows after strict header validation."""

    project_root = _project_root(root)
    return _read_csv(project_root / "data" / "benchmark_overall.csv", OVERALL_FIELDS)


def verify_project(root: Path | None = None) -> VerificationReport:
    """Verify hashes, route identity, plot rows, and both aggregate tables."""

    project_root = _project_root(root)
    data_root = project_root / "data"
    specs = _route_specs(data_root / "route_manifest.json")
    per_plot = _read_csv(data_root / "benchmark_per_plot.csv", PER_PLOT_FIELDS)
    grouped = _validate_per_plot(per_plot, specs)
    expected_by_site, expected_overall = _expected_aggregates(grouped)
    by_site = _read_csv(data_root / "benchmark_by_site.csv", BY_SITE_FIELDS)
    overall = _read_csv(data_root / "benchmark_overall.csv", OVERALL_FIELDS)
    if len(by_site) != EXPECTED_ROWS["by_site"] or len(overall) != EXPECTED_ROWS["overall"]:
        raise IntegrityError("Aggregate row count differs")
    aggregate_checks = _verify_aggregate_table(
        by_site, expected_by_site, include_site=True
    ) + _verify_aggregate_table(overall, expected_overall, include_site=False)
    retained_checks = _verify_retained_hashes(project_root)
    return VerificationReport(
        status="verified",
        protocol=PROTOCOL_ID,
        per_plot_rows=len(per_plot),
        by_site_rows=len(by_site),
        overall_rows=len(overall),
        result_identities=len(grouped),
        aggregate_values_checked=aggregate_checks,
        retained_hashes_checked=retained_checks,
    )
