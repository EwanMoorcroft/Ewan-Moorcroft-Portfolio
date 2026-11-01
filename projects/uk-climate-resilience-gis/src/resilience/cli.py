"""Stage-oriented, logged entrypoint; use scripts/run_python.sh -m resilience."""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .io import ROOT, configuration, save_json

LOG = logging.getLogger(__name__)
STAGES = ('setup', 'download', 'process', 'analyse', 'build-qgis', 'export', 'validate', 'all')


def _portable(value):
    if isinstance(value, Path):
        return str(value.relative_to(ROOT)) if value.is_relative_to(ROOT) else value.name
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    return value


def setup(config):
    """Verify the existing runtime without altering the user's installation."""
    modules = ('numpy', 'pandas', 'geopandas', 'shapely', 'rasterio', 'scipy',
               'requests', 'pyarrow', 'pyogrio', 'matplotlib', 'qgis.core')
    for name in modules:
        importlib.import_module(name)
    from qgis.core import Qgis
    return {'python': sys.version.split()[0], 'qgis': Qgis.QGIS_VERSION, 'imports': list(modules)}


def download(config):
    from .acquire import acquire_core
    from .preprocess import prepare_neighbourhoods
    from .hazards import acquire_flood
    from .raster import acquire_landsat
    from .network import acquire_network
    paths = acquire_core(config)
    boundaries = prepare_neighbourhoods(config)
    acquire_flood(boundaries)
    scenes = acquire_landsat(boundaries, config)
    case = boundaries.loc[boundaries.borough.isin(config['case_study_boroughs'])]
    acquire_network(case, config['network_buffer_m'])
    return {'core_sources': len(paths), 'selected_landsat_dates': len(scenes)}


def process(config):
    from .preprocess import prepare_neighbourhoods, prepare_greenspaces, prepare_origins, process_access
    from .raster import process_landsat
    from .hazards import process_flood
    from .analysis import assemble_indicators
    boundaries = prepare_neighbourhoods(config)
    parks, entrances = prepare_greenspaces(boundaries, config)
    origins = prepare_origins()
    process_landsat(boundaries, config)
    flood, extent = process_flood(boundaries)
    # Persist the explicit contracts also when these functions are called independently.
    flood.to_parquet(ROOT/'data/interim/flood_indicators.parquet', index=False)
    extent.to_parquet(ROOT/'data/interim/flood_extent.parquet', index=False)
    _, network = process_access(origins, entrances, config)
    network.to_parquet(ROOT/'data/interim/network_indicators.parquet', index=False)
    indicators = assemble_indicators(config)
    return {'neighbourhoods': len(indicators), 'public_parks': len(parks), 'origins': len(origins)}


def analyse(config):
    from .analysis import run_analysis
    return run_analysis(config)


def build_qgis(config):
    subprocess.run([sys.executable, str(ROOT/'scripts/qgis/build_project.py'), '--build'], check=True, cwd=ROOT)
    return {'project': 'qgis/project/gm_climate_resilience.qgz'}


def export(config):
    from .presentation import build_presentations
    subprocess.run([sys.executable, str(ROOT/'scripts/qgis/build_project.py'), '--export'], check=True, cwd=ROOT)
    return build_presentations(config)


def validate(config):
    from .validation import validate_project
    return validate_project(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    args = parser.parse_args()
    log_dir = ROOT/'outputs/logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir/f'{args.stage}.log')])
    config = configuration()
    stages = STAGES[:-1] if args.stage == 'all' else [args.stage]
    for stage in stages:
        started = time.monotonic()
        LOG.info('Starting %s', stage)
        try:
            result = _portable(globals()[stage.replace('-', '_')](config))
        except Exception:
            LOG.exception('Stage %s failed; preserve the raw cache and inspect the logged source/validation error', stage)
            raise
        elapsed = time.monotonic() - started
        save_json(ROOT/f'metadata/stages/{stage}.json', {'stage': stage, 'status': 'passed',
                  'completed_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': round(elapsed, 2), 'result': result})
        LOG.info('Completed %s in %.1f seconds', stage, elapsed)
        print(json.dumps(result, default=str, indent=2))
