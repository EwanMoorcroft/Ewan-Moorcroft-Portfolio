"""Acquire the bounded, authoritative inputs used by the regional study.

Raw downloads and exact API parameters are retained with checksums.  No
credentials or national polygon downloads are required.  Geography coverage is
checked against the independent IMD table before a source is accepted.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

from .io import ROOT, cached_json, download, get_json, record_source, save_json

LOG = logging.getLogger(__name__)
RAW = ROOT / "data/raw"


def source_catalogue() -> dict:
    return json.loads((ROOT / "config/sources.json").read_text())["sources"]


def region_imd_codes(rows: Iterable[dict], boroughs: list[str]) -> list[str]:
    """Select by exact LAD name and reject incomplete or duplicated geography."""
    requested = set(boroughs)
    seen_boroughs, codes = set(), []
    for row in rows:
        borough = row["Local Authority District name (2024)"]
        if borough in requested:
            code = row["LSOA code (2021)"]
            if not re.fullmatch(r"E\d{8}", code):
                raise ValueError(f"Invalid LSOA 2021 code: {code!r}")
            codes.append(code)
            seen_boroughs.add(borough)
    if requested - seen_boroughs:
        raise ValueError(f"Missing IMD boroughs: {sorted(requested - seen_boroughs)}")
    if len(codes) != len(set(codes)):
        raise ValueError("Duplicate LSOA codes in selected IMD rows")
    if not codes:
        raise ValueError("No LSOAs selected")
    return sorted(codes)


def combine_arcgis_pages(pages: list[dict], code_field: str, expected_codes: set[str]) -> dict:
    """Fail closed on partial pages, duplicate IDs and incomplete spatial joins."""
    features, codes = [], []
    for page in pages:
        if "error" in page or page.get("type") != "FeatureCollection":
            raise ValueError(f"ArcGIS returned an invalid feature response: {page.get('error', page.get('type'))}")
        if page.get("exceededTransferLimit") or page.get("properties", {}).get("exceededTransferLimit"):
            raise ValueError("ArcGIS transfer limit exceeded; reduce the request size")
        for feature in page.get("features", []):
            if not feature.get("geometry"):
                raise ValueError("Missing feature geometry")
            code = feature.get("properties", {}).get(code_field)
            if not code:
                raise ValueError(f"Missing {code_field}")
            codes.append(code)
            features.append(feature)
    if len(codes) != len(set(codes)):
        raise ValueError("Duplicate geographic codes across ArcGIS pages")
    actual = set(codes)
    if actual != expected_codes:
        raise ValueError(
            f"Geography coverage mismatch: {len(expected_codes - actual)} missing; "
            f"{len(actual - expected_codes)} unexpected"
        )
    return {"type": "FeatureCollection", "features": features}


def _metadata(source: dict) -> dict:
    return {key: source[key] for key in ("title", "publisher", "metadata_url", "version", "licence", "attribution") if key in source}


def _sql_strings(values: Iterable[str]) -> str:
    return ",".join("'" + value.replace("'", "''") + "'" for value in values)


def _verify_csv(path: Path, required_fields: set[str]) -> None:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required_fields.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected CSV schema in {path.name}: {reader.fieldnames}")
        if next(reader, None) is None:
            raise ValueError(f"Empty CSV: {path.name}")


def _geojson_by_codes(source: dict, codes: list[str], field: str, target: Path) -> Path:
    pages, queries = [], []
    # The ONS centroid endpoint returns HTTP 404 for longer query URLs even
    # though the same query succeeds below its approximately 2 KB limit.
    for start in range(0, len(codes), 100):
        selected = codes[start:start + 100]
        params = {"where": f"{field} IN ({_sql_strings(selected)})", "outFields": "*", "outSR": 4326,
                  "returnGeometry": "true", "f": "geojson", "resultRecordCount": 2000}
        part = target.parent / "pages" / f"{target.stem}_{start:05d}.geojson"
        page = cached_json(source["url"], part, params, **_metadata(source))
        combine_arcgis_pages([page], field, set(selected))
        pages.append(page)
        queries.append(str(part.relative_to(ROOT)))
    combined = combine_arcgis_pages(pages, field, set(codes))
    save_json(target, combined)
    record_source(target, source["url"], **_metadata(source), derivation="Concatenated verified raw API pages without geometry modification", input_files=queries)
    return target


def acquire_population(config: dict, sources: dict | None = None) -> dict:
    """Acquire age bins and case-study OA origins, retaining their exact lookup."""
    sources = sources or source_catalogue()
    paths = {}
    for source_key, output_key, filename in [
        ("census_lsoa_age", "census_lsoa_csv", "census2021_ts007a_lsoa.csv"),
        ("census_oa_age", "census_oa_csv", "census2021_ts007a_oa_northwest.csv"),
    ]:
        source = sources[source_key]
        path = download(source["url"], RAW / "ons" / filename, **_metadata(source))
        _verify_csv(path, {source["code_field"], "Age (B) (18 categories) Code", "Count"})
        paths[output_key] = path

    source = sources["oa_lookup"]
    where = f"{source['lad_name_field']} IN ({_sql_strings(config['case_study_boroughs'])})"
    count_response = cached_json(source["url"], RAW / "ons/pages/oa_lookup_case_count.json",
                                 {"where": where, "returnCountOnly": "true", "f": "json"}, **_metadata(source))
    if "count" not in count_response or count_response["count"] <= 0:
        raise ValueError(f"Cannot determine Output Area lookup coverage: {count_response}")
    expected_count = count_response["count"]
    records, page_files = [], []
    for start in range(0, expected_count, 1000):
        params = {"where": where, "returnGeometry": "false", "outFields": "*", "orderByFields": "OA21CD",
                  "f": "json", "resultOffset": start, "resultRecordCount": 1000}
        part = RAW / "ons/pages" / f"oa_lookup_case_{start:05d}.json"
        response = cached_json(source["url"], part, params, **_metadata(source))
        if "error" in response:
            raise ValueError(f"ArcGIS lookup query failed: {response['error']}")
        records.extend(feature["attributes"] for feature in response.get("features", []))
        page_files.append(str(part.relative_to(ROOT)))
    oa_codes = sorted(record["OA21CD"] for record in records)
    if len(records) != expected_count or len(set(oa_codes)) != expected_count:
        raise ValueError("Incomplete or duplicate OA lookup rows")
    lookup_path = RAW / "ons/oa_lookup_case.json"
    save_json(lookup_path, {"records": records})
    record_source(lookup_path, source["url"], params={"where": where}, **_metadata(source), input_files=page_files, row_count=len(records))
    paths["oa_lookup"] = lookup_path
    paths["oa_centroids"] = _geojson_by_codes(sources["oa_centroids"], oa_codes, "OA21CD", RAW / "ons/oa_centroids_case.geojson")
    return paths


def acquire_core(config: dict) -> dict:
    """Return source paths after verifying region coverage and archive integrity."""
    sources = source_catalogue()
    paths = {}
    source = sources["imd2025"]
    imd_path = download(source["url"], RAW / "mhclg/imd2025_file7.csv", **_metadata(source))
    with imd_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    codes = region_imd_codes(rows, config["boroughs"])
    paths["imd_csv"] = imd_path

    source = sources["lsoa_boundaries"]
    where = " OR ".join(f"LSOA21NM LIKE '{name.replace(chr(39), chr(39) * 2)} %'" for name in config["boroughs"])
    params = {"where": where, "outFields": "LSOA21CD,LSOA21NM", "outSR": 4326,
              "returnGeometry": "true", "f": "geojson", "resultRecordCount": 2000}
    boundary_path = RAW / "ons/region_lsoas.geojson"
    boundary = cached_json(source["url"], boundary_path, params, **_metadata(source))
    combine_arcgis_pages([boundary], "LSOA21CD", set(codes))
    paths["region_lsoas"] = boundary_path
    LOG.info("Verified %s region LSOAs against the independent IMD table", len(codes))
    save_json(RAW / "core_paths.json", paths)

    source = sources["greenspace_access"]
    path = download(source["url"], RAW / "defra/greenspace_access_2024_corrected2025.ods", **_metadata(source))
    with zipfile.ZipFile(path) as archive:
        if archive.read("mimetype") != b"application/vnd.oasis.opendocument.spreadsheet":
            raise ValueError("Greenspace-access file is not an ODS workbook")
        if archive.testzip() is not None:
            raise ValueError("Greenspace-access workbook archive is corrupt")
    paths["green_access_ods"] = path

    source = sources["os_greenspace"]
    product = cached_json(source["url"], RAW / "os/greenspace_product.json", **_metadata(source))
    if product["version"] != source["version"]:
        raise ValueError(f"OS Greenspace release changed from pinned {source['version']} to {product['version']}; review source version before updating")
    listing = cached_json(source["downloads_url"], RAW / "os/greenspace_downloads.json", **_metadata(source))
    candidates = {item["area"]: item for item in listing if item["format"] == source["format"] and item["area"] in source["tiles"]}
    if set(candidates) != set(source["tiles"]):
        raise ValueError("An expected OS greenspace tile is unavailable")
    paths["os_greenspace_zips"] = []
    for tile in source["tiles"]:
        item = candidates[tile]
        path = download(item["url"], RAW / "os" / item["fileName"], **_metadata(source), tile=tile, upstream_md5=item["md5"])
        if path.stat().st_size != item["size"] or hashlib.md5(path.read_bytes()).hexdigest() != item["md5"]:
            raise ValueError(f"OS tile checksum or size mismatch: {tile}")
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Corrupt OS tile archive: {tile}")
        paths["os_greenspace_zips"].append(path)
    save_json(RAW / "core_paths.json", paths)
    paths.update(acquire_population(config, sources))
    save_json(RAW / "core_paths.json", paths)
    return paths


if __name__ == "__main__":
    from .io import configuration
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(acquire_core(configuration()), indent=2, default=str))
