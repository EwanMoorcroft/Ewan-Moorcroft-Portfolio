"""Current Environment Agency scenario extents and defensible area overlays."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS

from .io import ROOT, record_source, save_json, session, sha256

LOG = logging.getLogger(__name__)
FLOOD_API = "https://environment.data.gov.uk/spatialdata/rivers-and-sea-defended-and-undefended-flood-risk-extents-present-day/ogc/features/v1/collections"
COLLECTIONS = {
    "defended_1pct": "Rivers_1in100_Sea_1in200_defended_extents",
    "defended_01pct": "Rivers_1in1000_Sea_1in1000_defended_extents",
}
SOURCE_PAGE = "https://www.data.gov.uk/dataset/1589b7ec-2300-4028-9199-aa7cac4b0f68/rivers-and-sea-defended-and-undefended-flood-risk-extents-present-day"
MAX_PAGES = 8
MAX_FLOOD_RAW_BYTES = 1_250_000_000


class _JSONStream:
    """Decode one feature at a time, without materialising a whole large page."""
    def __init__(self, stream, chunk_size):
        self.stream, self.chunk_size = stream, chunk_size
        self.buffer, self.position, self.eof = "", 0, False
        self.decoder = json.JSONDecoder()

    def fill(self):
        self.buffer = self.buffer[self.position:]
        self.position = 0
        block = self.stream.read(self.chunk_size)
        self.eof = not block
        self.buffer += block
        if len(self.buffer) > 64 * 1024 * 1024:
            raise ValueError("A GeoJSON value exceeds the bounded 64 MiB streaming limit")

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ""
            self.fill()

    def expect(self, character):
        if self.peek() != character:
            raise ValueError(f"Malformed or truncated GeoJSON: expected {character!r}")
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as exc:
                if self.eof:
                    raise ValueError("Malformed or truncated GeoJSON value") from exc
                self.fill()
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                if end < len(self.buffer) and self.buffer[end] not in " \t\r\n,}]":
                    if self.eof:
                        raise ValueError("Malformed GeoJSON number")
                    self.fill()
                    continue
            self.position = end
            return value


def inspect_flood_page(path: Path, chunk_size: int = 1024 * 1024) -> dict:
    """Audit page counts/IDs while retaining only one decoded feature at a time."""
    path = Path(path)
    metadata, identifiers, fingerprints = {}, [], []
    found_features = False
    with path.open(encoding="utf-8") as stream:
        reader = _JSONStream(stream, chunk_size)
        reader.expect("{")
        while reader.peek() != "}":
            key = reader.value()
            if not isinstance(key, str) or key in metadata or (key == "features" and found_features):
                raise ValueError("Malformed or duplicate GeoJSON root key")
            reader.expect(":")
            if key == "features":
                found_features = True
                reader.expect("[")
                while reader.peek() != "]":
                    feature = reader.value()
                    if not isinstance(feature, dict) or feature.get("type") != "Feature":
                        raise ValueError("Flood feature array contains a non-feature")
                    identifier = feature.get("id")
                    geometry = feature.get("geometry")
                    if identifier is None or not str(identifier):
                        raise ValueError("Flood features require stable source IDs")
                    if not isinstance(geometry, dict) or geometry.get("type") not in ("Polygon", "MultiPolygon"):
                        raise ValueError("Flood feature is missing polygon geometry")
                    identifiers.append(str(identifier))
                    canonical = json.dumps(feature, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    fingerprints.append(hashlib.sha256(canonical.encode()).hexdigest())
                    if reader.peek() == ",":
                        reader.expect(",")
                    elif reader.peek() != "]":
                        raise ValueError("Malformed GeoJSON feature array")
                reader.expect("]")
            else:
                metadata[key] = reader.value()
            if reader.peek() == ",":
                reader.expect(",")
            elif reader.peek() != "}":
                raise ValueError("Malformed GeoJSON root object")
        reader.expect("}")
        if reader.peek():
            raise ValueError("Unexpected content after GeoJSON object")
    if not found_features or metadata.get("type") != "FeatureCollection":
        raise ValueError("Response is not a GeoJSON FeatureCollection")
    matched, returned = metadata.get("numberMatched"), metadata.get("numberReturned")
    if (not isinstance(matched, int) or isinstance(matched, bool) or matched < 0
            or not isinstance(returned, int) or isinstance(returned, bool) or returned != len(identifiers)):
        raise ValueError("numberMatched/numberReturned must reconcile with actual returned features")
    declared_crs = metadata.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    if CRS.from_user_input(declared_crs).to_epsg() != 4326:
        raise ValueError("Flood source contract requires longitude/latitude EPSG:4326 GeoJSON")
    next_links = [link.get("href") for link in metadata.get("links", []) if link.get("rel") == "next"]
    if len(next_links) > 1 or any(not isinstance(url, str) or not url for url in next_links):
        raise ValueError("Ambiguous or malformed next-page link")
    return {
        "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
        "number_matched": matched, "number_returned": returned,
        "next_url": next_links[0] if next_links else None,
        "source_timestamp": metadata.get("timeStamp"), "response_bbox": metadata.get("bbox"),
        "source_crs": declared_crs, "feature_ids": identifiers,
        "feature_fingerprints": fingerprints,
    }


def _audit_page_infos(infos: list[dict]) -> dict:
    if not infos or len(infos) > MAX_PAGES:
        raise ValueError("Flood acquisition has no pages or exceeds its page bound")
    matched, seen, duplicates = infos[0]["number_matched"], {}, 0
    for position, page in enumerate(infos):
        if page["number_matched"] != matched:
            raise ValueError("numberMatched changed between source pages")
        mask = []
        for identifier, fingerprint in zip(page["feature_ids"], page["feature_fingerprints"]):
            if identifier in seen and seen[identifier] != fingerprint:
                raise ValueError(f"Conflicting changed content for duplicate flood feature {identifier}")
            new = identifier not in seen
            mask.append(new)
            duplicates += not new
            seen[identifier] = fingerprint
        page["new_feature_mask"] = mask
        if position < len(infos) - 1 and not page["next_url"]:
            raise ValueError("Unexpected extra flood page after an end-of-collection response")
    if infos[-1]["next_url"]:
        raise ValueError("Flood acquisition is incomplete: next page has not been acquired")
    if len(seen) != matched:
        raise ValueError(f"Flood unique feature counts do not reconcile: {len(seen)} of {matched}; records missing")
    return {"number_matched": matched, "features_returned": sum(p["number_returned"] for p in infos),
            "unique_features": len(seen), "duplicate_features": int(duplicates), "pages": infos}


def audit_flood_pages(paths: list[Path]) -> dict:
    """Reject an incomplete chain and deduplicate only identical source records."""
    return _audit_page_infos([inspect_flood_page(path) for path in paths])


def _next_url(url: str, collection: str, bounds, expected_offset: int) -> str:
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    if (parsed.scheme != "https" or parsed.hostname != "environment.data.gov.uk"
            or not parsed.path.endswith(f"/collections/{collection}/items")):
        raise ValueError("Unexpected flood pagination destination or collection")
    try:
        bbox = [float(value) for value in query["bbox"][0].split(",")]
        offset = int(query["startIndex"][0])
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError("Flood continuation must retain bbox and startIndex") from exc
    if len(bbox) != 4 or not np.allclose(bbox, bounds, rtol=0, atol=1e-9) or offset != expected_offset:
        raise ValueError("Flood continuation changed bbox or skipped/repeated source positions")
    return url


def _download_page(url, path, params, collection, allow_download=True):
    sidecar = path.with_name(path.name + ".source.json")
    if path.exists():
        if not sidecar.exists():
            raise ValueError(f"Cached flood page lacks provenance: {path}")
        source = json.loads(sidecar.read_text())
        if source.get("source_url") != url or source.get("parameters") != params:
            # Allow harmless projection/serialization rounding of the bbox.
            old = source.get("parameters") or {}
            same_bbox = params and old.keys() == params.keys()
            if same_bbox:
                same_bbox = all(old[key] == params[key] for key in params if key != "bbox")
                same_bbox = same_bbox and np.allclose(
                    list(map(float, old["bbox"].split(","))),
                    list(map(float, params["bbox"].split(","))), rtol=0, atol=1e-9)
            if source.get("source_url") != url or not same_bbox:
                raise ValueError("Cached flood page belongs to a different query")
        if source.get("sha256") != sha256(path):
            raise ValueError("Cached flood page checksum changed")
        return
    if not allow_download:
        raise ValueError(f"Missing flood continuation {path}; run acquisition first")
    used = sum(p.stat().st_size for p in path.parent.glob("defended*.geojson"))
    remaining = MAX_FLOOD_RAW_BYTES - used
    if remaining <= 0:
        raise ValueError("Flood raw-data byte budget exhausted")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        with session() as client, client.get(url, params=params, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                received = 0
                for block in response.iter_content(1024 * 1024):
                    received += len(block)
                    if received > remaining:
                        raise ValueError("Flood response exceeds the bounded raw-data budget")
                    stream.write(block)
            expected = response.headers.get("Content-Length")
            if expected and not response.headers.get("Content-Encoding") and received != int(expected):
                raise ValueError("Truncated flood download")
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    record_source(path, url, params=params, publisher="Environment Agency", licence="OGL v3",
                  scenario=collection, source_page=SOURCE_PAGE)


def _page_chain(label, bounds, acquire: bool) -> dict:
    collection = COLLECTIONS[label]
    url = FLOOD_API + "/" + collection + "/items"
    params = {"f": "json", "bbox": ",".join(map(str, bounds)), "limit": 10000}
    pages, returned = [], 0
    for page_number in range(1, MAX_PAGES + 1):
        name = label + ("" if page_number == 1 else f".page{page_number:03d}") + ".geojson"
        path = ROOT / "data/raw/ea" / name
        _download_page(url, path, params, collection, allow_download=acquire)
        info = inspect_flood_page(path)
        info["source_url"], info["parameters"] = url, params
        if pages and info["number_matched"] != pages[0]["number_matched"]:
            raise ValueError("numberMatched changed during acquisition")
        pages.append(info)
        returned += info["number_returned"]
        LOG.info("Flood %s page %d: %d returned / %d matched", label, page_number,
                 info["number_returned"], info["number_matched"])
        if not info["next_url"]:
            return _audit_page_infos(pages)
        url = _next_url(info["next_url"], collection, bounds, returned)
        params = None
    raise ValueError("Flood pagination exceeds the bounded page count")


def acquire_flood(boundaries) -> dict[str, Path]:
    """Cache and verify complete immutable page chains; return first-page paths."""
    bounds = boundaries.to_crs(4326).total_bounds
    result = {}
    for label in COLLECTIONS:
        audit = _page_chain(label, bounds, acquire=True)
        result[label] = Path(audit["pages"][0]["path"])
    return result


def _polygonal(geometry):
    if geometry.is_empty:
        return geometry
    if geometry.geom_type in ("Polygon", "MultiPolygon"):
        return geometry
    parts = [_polygonal(part) for part in getattr(geometry, "geoms", [])]
    return shapely.union_all([part for part in parts if not part.is_empty])


def _local_intersections(areas, hazard):
    """Union only features intersecting each area, after exact local clipping."""
    geometries = hazard.geometry.to_numpy()
    tree = shapely.STRtree(geometries)
    pieces = []
    for geometry in areas.geometry:
        candidates = tree.query(geometry, predicate="intersects")
        if not len(candidates):
            pieces.append(shapely.GeometryCollection())
        else:
            intersections = shapely.intersection(geometries[candidates], geometry)
            pieces.append(_polygonal(shapely.union_all(intersections)))
    return pieces


def area_overlap(areas: gpd.GeoDataFrame, hazard: gpd.GeoDataFrame, key="lsoa21cd") -> pd.Series:
    """Union the hazard first: overlapping features must not double count land."""
    if areas.crs is None or not areas.crs.is_projected or areas.crs.to_epsg() != 27700:
        raise ValueError("Area calculations require the declared British National Grid CRS")
    if areas[key].duplicated().any():
        raise ValueError("Duplicated area identifier")
    if (~areas.geometry.is_valid).any() or areas.geometry.is_empty.any() or (areas.area <= 0).any():
        raise ValueError("Neighbourhood geometries must be valid polygons of positive area")
    hazard = hazard.to_crs(areas.crs)
    hazard = hazard.copy()
    hazard.geometry = hazard.geometry.make_valid()
    fractions = shapely.area(_local_intersections(areas, hazard)) / areas.area.to_numpy()
    return pd.Series(np.clip(fractions, 0, 1), index=areas[key], name="flood_exposure")


def process_flood(boundaries: gpd.GeoDataFrame) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Process each source page once in memory, preserving exact local overlap."""
    study = boundaries.to_crs(27700)
    if study.lsoa21cd.duplicated().any() or (~study.geometry.is_valid).any() or (study.area <= 0).any():
        raise ValueError("Study requires unique IDs and valid positive-area neighbourhood polygons")
    bounds = study.to_crs(4326).total_bounds
    query_box = shapely.box(*bounds)
    tables, clips, source_audits = [], [], {}
    for label in COLLECTIONS:
        audit = _page_chain(label, bounds, acquire=False)
        by_area = [[] for _ in range(len(study))]
        vertices, repaired, bbox_hits, geometric_hits, returned = 0, 0, 0, 0, 0
        for page in audit["pages"]:
            LOG.info("Reading and clipping flood %s: %s", label, Path(page["path"]).name)
            raw = gpd.read_file(page["path"])
            if len(raw) != page["number_returned"] or raw.crs.to_epsg() != 4326:
                raise ValueError("GDAL page count/CRS differs from the verified source contract")
            if "id" in raw and raw["id"].astype(str).tolist() != page["feature_ids"]:
                raise ValueError("GDAL feature order differs from source IDs")
            returned += len(raw)
            bbox_hits += int(shapely.intersects(shapely.envelope(raw.geometry.to_numpy()), query_box).sum())
            geometric_hits += int(raw.geometry.intersects(query_box).sum())
            vertices += int(shapely.get_num_coordinates(raw.geometry.to_numpy()).sum())
            raw = raw.loc[np.asarray(page["new_feature_mask"], dtype=bool)].copy().to_crs(27700)
            repaired += int((~raw.geometry.is_valid).sum())
            raw.geometry = raw.geometry.make_valid()
            local = _local_intersections(study, raw)
            for index, geometry in enumerate(local):
                if not geometry.is_empty:
                    by_area[index].append(geometry)
            del raw, local
        geometries = [_polygonal(shapely.union_all(parts)) for parts in by_area]
        area_m2 = shapely.area(geometries)
        fractions = area_m2 / study.area.to_numpy()
        if np.any(~np.isfinite(fractions)) or np.any(fractions < -1e-10) or np.any(fractions > 1 + 1e-10):
            raise ValueError("Flood polygon-area fractions are invalid after exact local union")
        fraction = pd.Series(np.clip(fractions, 0, 1), index=study.lsoa21cd, name="flood_exposure")
        tables.append(fraction.rename("flood_exposure" if label == "defended_1pct" else "flood_01pct"))
        clipped = gpd.GeoDataFrame({"lsoa21cd": study.lsoa21cd.to_numpy(), "hazard": label},
                                   geometry=geometries, crs=27700)
        clips.append(clipped.loc[area_m2 > 0].copy())
        source_audits[label] = {
            key: value for key, value in audit.items() if key != "pages"
        }
        source_audits[label].update({
            "pages": [{key: (str(Path(value).relative_to(ROOT)) if key == "path" else value)
                       for key, value in page.items()
                       if key not in ("feature_ids", "feature_fingerprints", "new_feature_mask")}
                      for page in audit["pages"]],
            "source_coordinate_count": vertices, "invalid_source_geometries_repaired": repaired,
            "returned_features_tested": returned, "feature_bbox_intersects_query": bbox_hits,
            "feature_geometry_intersects_query": geometric_hits,
            "features_not_intersecting_query": returned - geometric_hits,
            "bbox_interpretation": "Returned polygons retain their full coordinates and include a small measured spill outside the query; exact neighbourhood intersections define the local output",
            "positive_neighbourhoods": int(np.count_nonzero(area_m2 > 0)),
            "local_union_area_km2": float(area_m2.sum() / 1e6),
            "max_polygon_area_fraction": float(fractions.max()),
        })
    table = pd.concat(tables, axis=1).reset_index()
    save_json(ROOT / "outputs/qa/flood.json", {
        "crs": "EPSG:27700", "primary_indicator": "Fraction of LSOA polygon area intersecting the present-day defended 1% river / 0.5% sea AEP scenario",
        "sensitivity_indicator": "Present-day defended 0.1% AEP extent",
        "area_denominator": "Full ONS neighbourhood polygon area, including any inland water retained in the boundary; no residential-land or address weighting",
        "population_interpretation": "No address-level or population exposure count is inferred from polygon-area overlap",
        "defence_assumption": "Defences operate as intended; asset failure and removal are not modelled",
        "neighbourhoods": len(table), "overlap_positive": int((table.flood_exposure > 0).sum()),
        "query_bbox_epsg4326": bounds.tolist(), "sources": source_audits,
        "raw_data_byte_bound": MAX_FLOOD_RAW_BYTES, "maximum_pages_per_scenario": MAX_PAGES,
        "exact_geometry": "Each source page is read sequentially; source intersections are unioned inside each neighbourhood; no simplification enters the metric",
        "drawing_filter": "Use hazard=defended_1pct for the primary map; defended_01pct is the separate sensitivity extent",
        "snapshot_limit": "Page timestamps and hashes are retained; sequential public responses are not a transactional snapshot",
    })
    return table, gpd.GeoDataFrame(pd.concat(clips, ignore_index=True), crs=27700)
