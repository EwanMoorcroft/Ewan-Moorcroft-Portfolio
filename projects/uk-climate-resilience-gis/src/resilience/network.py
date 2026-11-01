"""Entrance-based walking distances on an explicitly filtered OSM graph."""
from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from pyproj import Transformer
import shapely
from shapely.geometry import LineString

from .io import ROOT, record_source, session

LOG = logging.getLogger(__name__)


def walk_allowed(tags: dict) -> bool:
    highway = tags.get("highway", "")
    if not highway or highway in {"motorway", "motorway_link", "trunk", "trunk_link", "construction", "proposed", "raceway", "platform"}:
        return False
    if tags.get("foot") in {"no", "private"} or tags.get("area") == "yes":
        return False
    explicit = tags.get("foot") in {"yes", "designated", "permissive"}
    if (tags.get("access") in {"no", "private"} or highway == "cycleway") and not explicit:
        return False
    return True


def acquire_network(case_boundaries, buffer_m: float) -> Path:
    path = ROOT / "data/raw/osm/walking_network.json.gz"
    if path.exists():
        return path
    study = gpd.GeoSeries([case_boundaries.to_crs(27700).geometry.union_all().buffer(buffer_m)], crs=27700)
    west, south, east, north = study.to_crs(4326).total_bounds
    query = (f'[out:json][timeout:240];way["highway"]["highway"!~"motorway|trunk|construction|proposed|raceway"]'
             f'({south},{west},{north},{east});(._;>;);out body;')
    errors = []
    for endpoint in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"):
        try:
            LOG.info("Requesting buffered walking network from %s", endpoint)
            with session() as client:
                response = client.post(endpoint, data={"data": query}, timeout=(30, 300))
                response.raise_for_status()
                document = response.json()
            if document.get("remark") or len(document.get("elements", [])) < 1000:
                raise ValueError(f"Incomplete Overpass response: {document.get('remark', 'too few elements')}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix(".part").open("wb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                compressed.write(response.content)
            path.with_suffix(".part").replace(path)
            record_source(path, endpoint, params={"query": query}, publisher="OpenStreetMap contributors", licence="ODbL 1.0",
                          snapshot=document.get("osm3s", {}).get("timestamp_osm_base"))
            return path
        except Exception as error:
            errors.append(f"{endpoint}: {error}")
    raise RuntimeError("No complete walking network acquired: " + "; ".join(errors))


def build_graph(document: dict) -> tuple[nx.DiGraph, gpd.GeoDataFrame]:
    nodes = {element["id"]: (element["lon"], element["lat"]) for element in document["elements"] if element["type"] == "node"}
    blocked=set()
    for element in document['elements']:
        tags=element.get('tags',{})
        if element['type']=='node' and (tags.get('foot') in {'no','private'} or tags.get('locked')=='yes'
             or (tags.get('access') in {'no','private'} and tags.get('foot') not in {'yes','designated','permissive'})):
            blocked.add(element['id'])
    identifiers = list(nodes)
    xy = np.array([nodes[key] for key in identifiers])
    x, y = Transformer.from_crs(4326, 27700, always_xy=True).transform(xy[:, 0], xy[:, 1])
    coordinates = dict(zip(identifiers, zip(x, y)))
    graph = nx.DiGraph()
    segments = {}
    for way in document["elements"]:
        if way["type"] != "way" or not walk_allowed(way.get("tags", {})):
            continue
        tags = way["tags"]
        foot_direction = tags.get("oneway:foot", "no")
        for u, v in zip(way["nodes"][:-1], way["nodes"][1:]):
            if u not in coordinates or v not in coordinates:
                raise ValueError("Overpass omitted nodes required by a walking way")
            if u == v:
                continue
            if u in blocked or v in blocked:
                continue
            line = LineString([coordinates[u], coordinates[v]])
            length = line.length
            if length == 0:
                continue
            forward, backward = foot_direction != "-1", foot_direction not in {"yes", "1", "true"}
            if forward:
                graph.add_edge(u, v, weight=length)
            if backward:
                graph.add_edge(v, u, weight=length)
            # One geometric segment; multiple ways between the same node IDs share it.
            a, b = sorted((u, v))
            key = (a, b)
            if key not in segments:
                segments[key] = {"u": a, "v": b, "length": length,
                                 "geometry": LineString([coordinates[a], coordinates[b]])}
    for item in segments.values():
        item["forward"] = graph.has_edge(item["u"], item["v"])
        item["backward"] = graph.has_edge(item["v"], item["u"])
    edges = gpd.GeoDataFrame(list(segments.values()), crs=27700)
    return graph, edges


def _snap(points: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, threshold: float) -> gpd.GeoDataFrame:
    left = points.to_crs(27700).reset_index(drop=True).copy()
    left["point_order"] = np.arange(len(left))
    snapped = gpd.sjoin_nearest(left, edges, how="left", max_distance=threshold, distance_col="snap_m")
    snapped = snapped.sort_values(["point_order", "index_right"]).drop_duplicates("point_order").set_index("point_order")
    good = snapped.index_right.notna()
    snapped["along"] = np.nan
    snapped.loc[good, "along"] = shapely.line_locate_point(
        edges.geometry.loc[snapped.loc[good, "index_right"].astype(int)].to_numpy(), snapped.loc[good].geometry.to_numpy())
    return snapped


def access_distances(graph: nx.DiGraph, edges: gpd.GeoDataFrame, origins: gpd.GeoDataFrame,
                     entrances: gpd.GeoDataFrame, max_origin_snap_m: float,
                     max_entrance_snap_m: float) -> gpd.GeoDataFrame:
    """Connector + shortest route, preserving topology and same-edge shortcuts."""
    targets = _snap(entrances[["geometry"]], edges, max_entrance_snap_m)
    targets = targets[targets.index_right.notna()]
    if targets.empty:
        raise ValueError("No eligible entrance could be connected to the walking graph")
    seeds = {}
    targets_by_edge = {}
    for target in targets.itertuples():
        u, v = int(target.u), int(target.v)
        if target.forward:
            seeds[u] = min(seeds.get(u, np.inf), target.along + target.snap_m)
        if target.backward:
            seeds[v] = min(seeds.get(v, np.inf), target.length - target.along + target.snap_m)
        targets_by_edge.setdefault(int(target.index_right), []).append((target.along, target.snap_m))
    reverse = graph.reverse(copy=True)
    root = "__greenspace_destinations__"
    for node, distance in seeds.items():
        reverse.add_edge(root, node, weight=float(distance))
    distances = nx.single_source_dijkstra_path_length(reverse, root, weight="weight")
    snapped = _snap(origins, edges, max_origin_snap_m)
    values = []
    for origin in snapped.itertuples():
        if np.isnan(origin.index_right):
            values.append(np.nan)
            continue
        choices = []
        if origin.backward:
            choices.append(distances.get(int(origin.u), np.inf) + origin.along)
        if origin.forward:
            choices.append(distances.get(int(origin.v), np.inf) + origin.length - origin.along)
        for along, gap in targets_by_edge.get(int(origin.index_right), []):
            if (along >= origin.along and origin.forward) or (along <= origin.along and origin.backward):
                choices.append(abs(along-origin.along) + gap)
        values.append(min(choices, default=np.inf) + origin.snap_m)
    result = origins.reset_index(drop=True).copy()
    result["snap_m"] = snapped["snap_m"].to_numpy()
    result["known"] = snapped.index_right.notna().to_numpy()
    result["distance_m"] = values
    result["reachable"] = np.isfinite(values) & result.known
    result.attrs["entrances_total"] = len(entrances)
    result.attrs["entrances_connected"] = len(targets)
    result.attrs["nodes"] = graph.number_of_nodes()
    result.attrs["edges"] = graph.number_of_edges()
    return result
