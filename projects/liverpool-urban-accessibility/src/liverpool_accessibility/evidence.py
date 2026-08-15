"""Retained-result creation and independent verification."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from .contracts import DataContractError, validate_metrics
from .modelling import CountModelResult, fit_count_models
from .spatial import MoranResult, morans_i

CORE_FILES = (
    "area-metrics.csv",
    "spatial-edges.csv",
    "liverpool-msoa-boundaries.geojson",
    "liverpool-msoa-centroids.geojson",
    "model-coefficients.csv",
    "results.json",
    "source-manifest.json",
)
OPTIONAL_FILES = ("r-validation.json",)
SOURCE_ROLES = {"flow_csv", "boundaries", "centroids"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
OFFICIAL_SOURCE_CONTRACT = "liverpool-source-manifest-v1"
FIXTURE_SOURCE_CONTRACT = "liverpool-fixture-source-manifest-v1"
FIXTURE_EVIDENCE_SCOPE = "fictional deterministic integration fixture"
OFFICIAL_ANALYSIS_AREA = {
    "area_code": "E08000012",
    "area_name": "Liverpool",
    "geography": "2021 Middle layer Super Output Areas",
    "msoa_count": 61,
}
OFFICIAL_NATIONAL_FLOW_IDENTITY = {
    "archive_bytes": 80_621_150,
    "archive_sha256": "9e32ababfd9f77e353411d399e463f812942440df115a2d6c296c74dbeea70d7",
    "archive_url": "https://www.nomisweb.co.uk/output/census/2021/odwp01ew.zip",
    "bytes": 197_658_221,
    "file": "ODWP01EW_MSOA.csv",
    "sha256": "8af475023e18227fdcee3ac4a547d6549b1fcb87138bd93d177a7a698d1a10dd",
}


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_manifest(
    payload: dict[str, object], role_paths: dict[str, str | Path]
) -> dict[str, object]:
    """Bind each source-manifest identity to the exact file supplied for preparation."""
    contract = payload.get("contract")
    if contract not in {OFFICIAL_SOURCE_CONTRACT, FIXTURE_SOURCE_CONTRACT}:
        raise DataContractError("unsupported source-manifest contract")
    if set(role_paths) != SOURCE_ROLES:
        raise DataContractError("source paths must cover flow, boundary, and centroid roles")
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise DataContractError("source manifest must contain a source list")
    by_role: dict[str, dict[str, object]] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise DataContractError("source manifest contains a non-object source")
        role = source.get("role")
        if role not in SOURCE_ROLES or role in by_role:
            raise DataContractError("source manifest roles must be exact and unique")
        by_role[str(role)] = source
    if set(by_role) != SOURCE_ROLES:
        raise DataContractError("source manifest roles are incomplete")

    for role, supplied_path in role_paths.items():
        path = Path(supplied_path)
        record = by_role[role]
        expected_hash = record.get("sha256")
        expected_bytes = record.get("bytes")
        if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
            raise DataContractError(f"source manifest has an invalid SHA-256 for {role}")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
        ):
            raise DataContractError(f"source manifest has invalid bytes for {role}")
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_hash:
            raise DataContractError(f"source identity mismatch: {role}")

    if contract == OFFICIAL_SOURCE_CONTRACT:
        if payload.get("analysis_area") != OFFICIAL_ANALYSIS_AREA:
            raise DataContractError("official source manifest has an invalid analysis area")
        flow_record = by_role["flow_csv"]
        if any(
            flow_record.get(field) != expected
            for field, expected in OFFICIAL_NATIONAL_FLOW_IDENTITY.items()
        ):
            raise DataContractError(
                "official source manifest does not authenticate the complete national OD source"
            )
        evidence_scope = "official national OD source with Liverpool analysis geography"
    else:
        if payload.get("evidence_scope") != FIXTURE_EVIDENCE_SCOPE:
            raise DataContractError("fixture source manifest has an invalid evidence scope")
        evidence_scope = FIXTURE_EVIDENCE_SCOPE
    return {
        "contract": contract,
        "evidence_scope": evidence_scope,
        "files_verified": len(SOURCE_ROLES),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compute_results(
    metrics: pd.DataFrame, edges: pd.DataFrame
) -> tuple[
    dict[str, object],
    tuple[CountModelResult, CountModelResult, CountModelResult],
    MoranResult,
]:
    """Recompute every aggregate published by the project."""
    validate_metrics(metrics)
    spatial = morans_i(
        metrics["area_code"],
        metrics["local_retention_share"],
        edges,
        permutations=9999,
        seed=2026,
    )
    poisson, negative_binomial, binomial = fit_count_models(metrics)
    fixed_total = int(metrics["fixed_workplace"].sum())
    local_total = int(metrics["local_fixed"].sum())
    employed_total = int(metrics["employed_total"].sum())
    home_total = int(metrics["home_or_no_fixed"].sum())
    results: dict[str, object] = {
        "analysis_contract": "liverpool-commuting-access-v1",
        "geography": "2021 Middle layer Super Output Areas",
        "areas": len(metrics),
        "employed_total": employed_total,
        "fixed_workplace_total": fixed_total,
        "local_fixed_total": local_total,
        "local_retention_share": local_total / fixed_total,
        "home_or_no_fixed_total": home_total,
        "home_or_no_fixed_share": home_total / employed_total,
        "moran": {
            "variable": "area local-retention share",
            "weights": "row-standardized Queen contiguity",
            "statistic": spatial.statistic,
            "pseudo_p_value_two_sided": spatial.pseudo_p_value,
            "permutations": spatial.permutations,
            "seed": spatial.seed,
        },
        "poisson": {
            "outcome": "local fixed-workplace count",
            "exposure": "all fixed-workplace workers from the origin area",
            "observations": poisson.observations,
            "aic": poisson.aic,
            "log_likelihood": poisson.log_likelihood,
            "pearson_dispersion": poisson.pearson_dispersion,
            "predicted_rate_min": poisson.predicted_rate_min,
            "predicted_rate_max": poisson.predicted_rate_max,
            "uncertainty": "HC0 robust standard errors; descriptive associations only",
        },
        "negative_binomial_nb2": {
            "observations": negative_binomial.observations,
            "aic": negative_binomial.aic,
            "log_likelihood": negative_binomial.log_likelihood,
            "alpha": negative_binomial.alpha,
            "converged": negative_binomial.converged,
            "predicted_rate_min": negative_binomial.predicted_rate_min,
            "predicted_rate_max": negative_binomial.predicted_rate_max,
            "uncertainty": "HC0 robust standard errors; sensitivity to Poisson overdispersion",
        },
        "binomial_sensitivity": {
            "observations": binomial.observations,
            "log_likelihood": binomial.log_likelihood,
            "predicted_rate_min": binomial.predicted_rate_min,
            "predicted_rate_max": binomial.predicted_rate_max,
            "uncertainty": "bounded-response sensitivity; descriptive associations only",
        },
        "interpretation_boundary": (
            "Census 2021 workplace flows were collected during pandemic disruption; "
            "results are ecological, descriptive, and not current travel demand."
        ),
    }
    return results, (poisson, negative_binomial, binomial), spatial


def write_results(evidence_dir: str | Path) -> list[Path]:
    """Write computed result tables and the core evidence manifest."""
    root = Path(evidence_dir)
    metrics = pd.read_csv(root / "area-metrics.csv")
    edges = pd.read_csv(root / "spatial-edges.csv")
    results, models, _ = compute_results(metrics, edges)
    coefficients_path = root / "model-coefficients.csv"
    results_path = root / "results.json"
    coefficients = pd.concat(
        [model.coefficients.assign(family=model.family) for model in models],
        ignore_index=True,
    )
    coefficients = coefficients[
        [
            "family",
            "term",
            "estimate",
            "robust_se",
            "p_value",
            "confidence_low",
            "confidence_high",
        ]
    ]
    coefficients.to_csv(coefficients_path, index=False, float_format="%.12g", lineterminator="\n")
    _write_json(results_path, results)
    missing = [name for name in CORE_FILES if not (root / name).is_file()]
    if missing:
        raise DataContractError(f"cannot create manifest; missing: {', '.join(missing)}")
    inventory = CORE_FILES + tuple(name for name in OPTIONAL_FILES if (root / name).is_file())
    manifest = {
        "contract": "liverpool-retained-evidence-v1",
        "files": {name: sha256_file(root / name) for name in inventory},
    }
    manifest_path = root / "results-manifest.json"
    _write_json(manifest_path, manifest)
    return [coefficients_path, results_path, manifest_path]


def verify_evidence(evidence_dir: str | Path, *, tolerance: float = 1e-9) -> dict[str, object]:
    """Check hashes and independently reconstruct every retained numerical result."""
    root = Path(evidence_dir)
    manifest = json.loads((root / "results-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != "liverpool-retained-evidence-v1":
        raise DataContractError("unsupported retained-evidence contract")
    expected_files = manifest.get("files")
    if (
        not isinstance(expected_files, dict)
        or not set(CORE_FILES).issubset(expected_files)
        or not set(expected_files).issubset(set(CORE_FILES) | set(OPTIONAL_FILES))
    ):
        raise DataContractError("retained-evidence file inventory is incomplete")
    for name, expected_hash in expected_files.items():
        if sha256_file(root / name) != expected_hash:
            raise DataContractError(f"retained file hash mismatch: {name}")

    metrics = pd.read_csv(root / "area-metrics.csv")
    edges = pd.read_csv(root / "spatial-edges.csv")
    reconstructed, models, _ = compute_results(metrics, edges)
    retained = json.loads((root / "results.json").read_text(encoding="utf-8"))
    for key in (
        "areas",
        "employed_total",
        "fixed_workplace_total",
        "local_fixed_total",
        "home_or_no_fixed_total",
    ):
        if retained.get(key) != reconstructed[key]:
            raise DataContractError(f"retained result mismatch: {key}")
    for key in ("local_retention_share", "home_or_no_fixed_share"):
        if abs(float(retained[key]) - float(reconstructed[key])) > tolerance:
            raise DataContractError(f"retained result mismatch: {key}")
    for section, keys in {
        "moran": ("statistic", "pseudo_p_value_two_sided"),
        "poisson": (
            "aic",
            "log_likelihood",
            "pearson_dispersion",
            "predicted_rate_min",
            "predicted_rate_max",
        ),
        "negative_binomial_nb2": (
            "aic",
            "log_likelihood",
            "alpha",
            "predicted_rate_min",
            "predicted_rate_max",
        ),
        "binomial_sensitivity": (
            "log_likelihood",
            "predicted_rate_min",
            "predicted_rate_max",
        ),
    }.items():
        for key in keys:
            if abs(float(retained[section][key]) - float(reconstructed[section][key])) > tolerance:
                raise DataContractError(f"retained result mismatch: {section}.{key}")

    reconstructed_coefficients = pd.concat(
        [model.coefficients.assign(family=model.family) for model in models],
        ignore_index=True,
    )
    retained_coefficients = pd.read_csv(root / "model-coefficients.csv")
    if list(retained_coefficients["term"]) != list(reconstructed_coefficients["term"]):
        raise DataContractError("retained coefficient terms do not align")
    if list(retained_coefficients["family"]) != list(reconstructed_coefficients["family"]):
        raise DataContractError("retained coefficient families do not align")
    numeric_columns = [
        column for column in reconstructed_coefficients.columns if column not in {"term", "family"}
    ]
    difference = retained_coefficients[numeric_columns].to_numpy(
        dtype=float
    ) - reconstructed_coefficients[numeric_columns].to_numpy(dtype=float)
    if abs(difference).max() > tolerance:
        raise DataContractError("retained model coefficients do not reconstruct")
    return {
        "contract": manifest["contract"],
        "files_verified": len(expected_files),
        "areas": len(metrics),
        "aggregates_verified": 22,
        "coefficient_values_verified": int(reconstructed_coefficients[numeric_columns].size),
    }
