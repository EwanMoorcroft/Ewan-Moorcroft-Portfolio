"""Deterministic fictional inputs for offline integration tests and demos."""

from __future__ import annotations

import csv
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point, box

from .evidence import sha256_file

FLOW_HEADER = [
    "Middle layer Super Output Areas code",
    "Middle layer Super Output Areas label",
    "MSOA of workplace code",
    "MSOA of workplace label",
    "Place of work indicator (4 categories) code",
    "Place of work indicator (4 categories) label",
    "Count",
]


def write_fixture(output_dir: str | Path, *, size: int = 4) -> tuple[Path, Path, Path]:
    """Write a square fictional geography and complete positive flow table."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 2 or size > 8:
        raise ValueError("fixture size must be an integer from 2 to 8")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    centroid_records: list[dict[str, object]] = []
    codes: list[str] = []
    for row in range(size):
        for column in range(size):
            index = row * size + column + 1
            code = f"E0209{index:04d}"
            codes.append(code)
            west = -3.10 + column * 0.015
            south = 53.35 + row * 0.01
            records.append(
                {
                    "MSOA21CD": code,
                    "MSOA21NM": f"Fictional area {index:02d}",
                    "geometry": box(west, south, west + 0.015, south + 0.01),
                }
            )
            centroid_records.append(
                {"MSOA21CD": code, "geometry": Point(west + 0.0075, south + 0.005)}
            )
    boundaries = gpd.GeoDataFrame(records, crs=4326)
    boundary_path = destination / "boundaries.geojson"
    boundaries.to_file(boundary_path, driver="GeoJSON")
    centroids = gpd.GeoDataFrame(centroid_records, crs=4326)
    centroid_path = destination / "centroids.geojson"
    centroids[["MSOA21CD", "geometry"]].to_file(centroid_path, driver="GeoJSON")

    flow_path = destination / "flows.csv"
    with flow_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(FLOW_HEADER)
        for origin_index, origin in enumerate(codes):
            origin_name = f"Fictional area {origin_index + 1:02d}"
            origin_effect = (0.3, 2.2, 0.7, 3.0, 1.1)[(origin_index * 3) % 5]
            writer.writerow(
                [
                    origin,
                    origin_name,
                    "-8",
                    "No fixed place",
                    "1",
                    "Home or no fixed place",
                    round((40 + origin_index % 3) * origin_effect),
                ]
            )
            writer.writerow(
                [origin, origin_name, "-8", "Outside the UK", "2", "Other workplace", 2]
            )
            for destination_index, destination_code in enumerate(codes):
                destination_name = f"Fictional area {destination_index + 1:02d}"
                distance = abs(origin_index // size - destination_index // size) + abs(
                    origin_index % size - destination_index % size
                )
                flow_count = max(
                    2,
                    round((48 - 7 * distance + (destination_index % 3)) * origin_effect),
                )
                writer.writerow(
                    [
                        origin,
                        origin_name,
                        destination_code,
                        destination_name,
                        "3",
                        "Working at a fixed workplace",
                        flow_count,
                    ]
                )
            writer.writerow(
                [
                    origin,
                    origin_name,
                    "E02089999",
                    "Fictional external area",
                    "3",
                    "Working at a fixed workplace",
                    (80, 900, 250, 1300, 450)[(origin_index * 2) % 5] + (origin_index // 5) * 30,
                ]
            )
    return flow_path, boundary_path, centroid_path


def fixture_source_manifest(
    flow_path: Path, boundary_path: Path, centroid_path: Path
) -> dict[str, object]:
    """Return the identity record for generated fictional inputs."""
    return {
        "contract": "liverpool-fixture-source-manifest-v1",
        "evidence_scope": "fictional deterministic integration fixture",
        "sources": [
            {
                "name": "fictional flows",
                "path_role": "generated test input",
                "role": "flow_csv",
                "sha256": sha256_file(flow_path),
                "bytes": flow_path.stat().st_size,
            },
            {
                "name": "fictional boundaries",
                "path_role": "generated test input",
                "role": "boundaries",
                "sha256": sha256_file(boundary_path),
                "bytes": boundary_path.stat().st_size,
            },
            {
                "name": "fictional population-weighted centroids",
                "path_role": "generated test input",
                "role": "centroids",
                "sha256": sha256_file(centroid_path),
                "bytes": centroid_path.stat().st_size,
            },
        ],
    }
