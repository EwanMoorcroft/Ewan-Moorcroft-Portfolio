"""Landsat surface-temperature and vegetation evidence, with explicit QA."""
from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds

from .io import ROOT, cached_json, get_json, record_source, save_json

LOG = logging.getLogger(__name__)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def temperature_celsius(digital_number: np.ndarray) -> np.ndarray:
    result = digital_number.astype("float32") * 0.00341802 + 149.0 - 273.15
    return np.where(digital_number > 0, result, np.nan)


def clear_land_mask(qa_pixel: np.ndarray, qa_radsat: np.ndarray) -> np.ndarray:
    # Fill, dilated cloud, cirrus, cloud, shadow, snow and water (USGS QA_PIXEL).
    excluded_bits = 1 | 2 | 4 | 8 | 16 | 32 | 128
    return ((qa_pixel & excluded_bits) == 0) & (qa_radsat == 0)


def vegetation_index(nir_dn: np.ndarray, red_dn: np.ndarray) -> np.ndarray:
    nir = nir_dn.astype("float32") * 0.0000275 - 0.2
    red = red_dn.astype("float32") * 0.0000275 - 0.2
    denominator = nir + red
    good = (nir_dn > 0) & (red_dn > 0) & (nir >= 0) & (red >= 0) & (denominator > 0)
    result = np.full(nir.shape, np.nan, dtype="float32")
    np.divide(nir - red, denominator, out=result, where=good)
    return result


def search_scenes(bounds_wgs84: tuple, config: dict) -> list[dict]:
    candidates = []
    for year in config["landsat_years"]:
        params = {"collections": "landsat-c2-l2", "bbox": ",".join(map(str, bounds_wgs84)),
                  "datetime": f"{year}-06-01T00:00:00Z/{year}-08-31T23:59:59Z", "limit": 1000}
        document = cached_json(STAC + "/search", ROOT / f"data/raw/landsat/search_{year}.json", params,
                               publisher="USGS / Microsoft Planetary Computer", licence="USGS public domain")
        if any(link.get("rel") == "next" for link in document.get("links", [])):
            raise ValueError("Landsat search was paginated; refine the study search before selecting scenes")
        # One WRS footprint covering the region prevents same-day adjacent-scene double counting.
        selected = [item for item in document["features"]
                    if "_203023_" in item["id"] and "lwir11" in item["assets"]
                    and item["properties"].get("eo:cloud_cover", 100) <= config["landsat_max_scene_cloud_pct"]]
        selected.sort(key=lambda item: (item["properties"].get("eo:cloud_cover", 100), item["id"]))
        candidates.extend(selected[:config.get("landsat_candidates_per_year", 5)])
    if len(candidates) < config["minimum_heat_observations"]:
        raise ValueError("Too few qualifying summer Landsat scenes; revise a documented temporal design")
    return sorted(candidates, key=lambda item: item["id"])


def subset_asset(item: dict, band: str, bounds_bng: tuple) -> Path:
    path = ROOT / "data/raw/landsat" / item["id"] / f"{band}.tif"
    if path.exists():
        return path
    href = item["assets"][band]["href"]
    signed = get_json("https://planetarycomputer.microsoft.com/api/sas/v1/sign", {"href": href})["href"]
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Window-reading %s %s", item["id"], band)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                      GDAL_HTTP_MAX_RETRY=4, GDAL_HTTP_RETRY_DELAY=2):
        with rasterio.open(signed) as source:
            bbox = transform_bounds("EPSG:27700", source.crs, *bounds_bng, densify_pts=21)
            window = from_bounds(*bbox, transform=source.transform).round_offsets().round_lengths()
            data = source.read(1, window=window, boundless=True, fill_value=0)
            profile = source.profile.copy()
            profile.update(driver="GTiff", height=data.shape[0], width=data.shape[1],
                           transform=source.window_transform(window), compress="deflate", tiled=True,
                           blockxsize=256, blockysize=256, nodata=0)
            with rasterio.open(path.with_suffix(".part.tif"), "w", **profile) as target:
                target.write(data, 1)
    path.with_suffix(".part.tif").replace(path)
    record_source(path, href, publisher="USGS", licence="Public domain", scene=item["id"], band=band,
                  operation="Lossless pixel-aligned study-area window; original DN values", bounds_bng=bounds_bng)
    return path


def acquire_landsat(boundaries, config: dict) -> list[dict]:
    boundaries = boundaries.to_crs(27700)
    bounds = tuple(boundaries.to_crs(27700).total_bounds)
    scenes = search_scenes(transform_bounds("EPSG:27700", "EPSG:4326", *bounds), config)
    assessed = []
    for item in scenes:
        arrays = {}
        for band in ("qa_pixel", "qa_radsat", "cdist"):
            path = subset_asset(item, band, bounds)
            with rasterio.open(path) as source:
                arrays[band] = source.read(1)
                native_transform, native_crs = source.transform, source.crs
        region = geometry_mask(boundaries.to_crs(native_crs).geometry, out_shape=arrays['qa_pixel'].shape,
                               transform=native_transform, invert=True)
        clear = clear_land_mask(arrays['qa_pixel'], arrays['qa_radsat'])
        clear &= arrays['cdist'] * .01 >= config['landsat_cloud_distance_km']
        fraction = float(clear[region].mean())
        assessed.append({'id':item['id'], 'date':item['properties']['datetime'],
                         'scene_cloud_percent':item['properties']['eo:cloud_cover'], 'region_clear_fraction':fraction})
        LOG.info('Study-area clear fraction %.3f for %s', fraction, item['id'])
    by_id = {x['id']:x for x in scenes}
    eligible = sorted([x for x in assessed if x['region_clear_fraction'] >= config['minimum_scene_region_coverage']],
                      key=lambda x:(-x['region_clear_fraction'],x['id']))
    counts, chosen = {}, []
    for row in eligible:
        year = row['date'][:4]
        if counts.get(year,0) < 2 and len(chosen) < config['landsat_max_dates']:
            chosen.append(row);counts[year]=counts.get(year,0)+1
    if len(chosen) < config['minimum_heat_observations'] or len(counts) < config['minimum_heat_summers']:
        raise ValueError('Insufficient well-observed summer dates after regional QA')
    for row in chosen:
        item = by_id[row['id']]
        for band in ('lwir11', 'red', 'nir08'):
            subset_asset(item, band, bounds)
    save_json(ROOT/'metadata/landsat_selection.json', {'rule':'June-August 2021-2025; WRS203/023; five lowest scene-cloud candidates per year; configured minimum clear study land; highest regional coverage up to eight dates, at most two per year',
        'candidate_assessment':assessed, 'scenes':chosen, 'config':{key:value for key,value in config.items() if key.startswith('landsat_') or key.startswith('minimum_') or key=='raster_resolution_m'}})
    return [by_id[row['id']] for row in chosen]


def write_raster(path: Path, data: np.ndarray, transform, *, nodata=-9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                       dtype="float32", crs="EPSG:27700", transform=transform, nodata=nodata,
                       compress="deflate", tiled=True, blockxsize=256, blockysize=256) as target:
        target.write(np.where(np.isfinite(data), data, nodata).astype("float32"), 1)
        target.build_overviews([2, 4, 8], Resampling.average)
        target.update_tags(AREA_OR_POINT="Area")


def process_landsat(boundaries, config: dict) -> dict:
    boundaries = boundaries.to_crs(27700)
    bounds = tuple(boundaries.total_bounds)
    resolution = config["raster_resolution_m"]
    left, bottom = np.floor(np.array(bounds[:2]) / resolution) * resolution
    right, top = np.ceil(np.array(bounds[2:]) / resolution) * resolution
    transform = from_origin(left, top, resolution, resolution)
    shape = (int(round((top-bottom)/resolution)), int(round((right-left)/resolution)))
    region = geometry_mask(boundaries.geometry, out_shape=shape, transform=transform, invert=True)
    scenes = search_scenes(transform_bounds("EPSG:27700", "EPSG:4326", *bounds), config)
    import json
    selected = {row['id'] for row in json.loads((ROOT/'metadata/landsat_selection.json').read_text())['scenes']}
    scenes = [item for item in scenes if item['id'] in selected]
    temperatures, anomalies, vegetation, scene_reports, years = [], [], [], [], []
    for item in scenes:
        raw = ROOT / "data/raw/landsat" / item["id"]
        arrays = {}
        for band in ("qa_pixel", "qa_radsat", "lwir11", "cdist", "red", "nir08"):
            with rasterio.open(raw / f"{band}.tif") as source:
                arrays[band] = source.read(1)
                native_transform, native_crs = source.transform, source.crs
        clear = clear_land_mask(arrays["qa_pixel"], arrays["qa_radsat"])
        clear &= arrays["cdist"] * 0.01 >= config["landsat_cloud_distance_km"]
        clear &= arrays['lwir11']>0
        temp = np.where(clear, temperature_celsius(arrays["lwir11"]), np.nan)
        ndvi = np.where(clear, vegetation_index(arrays["nir08"], arrays["red"]), np.nan)
        projected = []
        for values in (temp, ndvi):
            dest = np.full(shape, np.nan, dtype="float32")
            reproject(values, dest, src_transform=native_transform, src_crs=native_crs, src_nodata=np.nan,
                      dst_transform=transform, dst_crs="EPSG:27700", dst_nodata=np.nan, resampling=Resampling.average)
            dest[~region] = np.nan
            projected.append(dest)
        temperature, ndvi_grid = projected
        valid_fraction=np.zeros(shape,dtype='float32')
        reproject(clear.astype('float32'),valid_fraction,src_transform=native_transform,src_crs=native_crs,
                  dst_transform=transform,dst_crs='EPSG:27700',dst_nodata=-9999,resampling=Resampling.average)
        temperature[valid_fraction<config['minimum_heat_cell_valid_fraction']]=np.nan
        ndvi_grid[valid_fraction<config['minimum_heat_cell_valid_fraction']]=np.nan
        coverage = float(np.isfinite(temperature[region]).mean())
        median = float(np.nanmedian(temperature[region]))
        scene_reports.append({"id": item["id"], "date": item["properties"]["datetime"],
                              "valid_region_fraction": coverage, "regional_median_c": median})
        temperatures.append(temperature)
        anomalies.append(temperature - median)
        vegetation.append(ndvi_grid)
        years.append(int(item["properties"]["datetime"][:4]))
    stack = np.stack(anomalies)
    count = np.isfinite(stack).sum(axis=0)
    year_count = sum(np.any(np.isfinite(stack[np.array(years) == year]), axis=0) for year in sorted(set(years)))
    sufficient = (count >= config["minimum_heat_observations"]) & (year_count >= config['minimum_heat_summers']) & region
    # Compare each date over a fixed shared valid reference area, avoiding shifts
    # in the reference median when clouds hide different parts of the region.
    common = np.all(np.isfinite(np.stack(temperatures)), axis=0) & region
    if common[region].mean() < config['minimum_reference_land_fraction']:
        raise ValueError('Shared clear reference land is too small for comparable date anomalies')
    for index, temperature in enumerate(temperatures):
        reference = float(np.median(temperature[common]))
        scene_reports[index]['common_reference_median_c'] = reference
        anomalies[index] = temperature-reference
    stack = np.stack(anomalies)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        products = {"heat_anomaly": np.nanmedian(stack, axis=0),
                    "surface_temperature": np.nanmedian(np.stack(temperatures), axis=0),
                    "ndvi": np.nanmedian(np.stack(vegetation), axis=0)}
        for year in sorted(set(years)):
            products[f"heat_{year}"] = np.nanmedian(stack[np.array(years) == year], axis=0)
    for name, values in products.items():
        if name in {"heat_anomaly", "surface_temperature"}:
            values[~sufficient] = np.nan
        write_raster(ROOT / f"data/processed/{name}.tif", values, transform)
    write_raster(ROOT / "data/processed/heat_observations.tif", np.where(region, count, np.nan), transform)
    report = {"scenes": scene_reports, "resolution_m": resolution,
              "minimum_dates": config["minimum_heat_observations"], "required_summers": config['minimum_heat_summers'],
              "common_reference_cell_fraction": float(common[region].mean()),
              "adequate_region_cell_fraction": float(sufficient[region].mean()),
              "interpretation": "Median clear-sky daytime land-surface anomaly; dates centred on the same common valid study-region land. Not air temperature or climate projection."}
    save_json(ROOT / "outputs/qa/landsat.json", report)
    return report
