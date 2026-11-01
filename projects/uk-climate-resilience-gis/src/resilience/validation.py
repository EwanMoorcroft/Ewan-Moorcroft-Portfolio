"""Validate the delivered real-data contracts and emit an inspectable QA record."""
from __future__ import annotations

import gc
import json
import math
import sys
import zipfile
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from scipy.spatial import cKDTree

from .io import ROOT, save_json, sha256
from .model import fit_model


def validate_project(config: dict) -> dict:
    """Keep QGIS alive until every validation-local wrapper is released."""
    from qgis.core import QgsApplication, QgsProject

    app = QgsApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QgsApplication([], False)
        app.initQgis()
    try:
        return _validate_project(config)
    finally:
        if owns_app:
            # The inner frame (project, feature and provider wrappers) must be
            # destroyed before QGIS tears down its C++ registries and caches.
            QgsProject.instance().clear()
            gc.collect()
            app.exitQgis()


def _validate_project(config: dict) -> dict:
    checks = []

    def check(name, condition, detail=None):
        checks.append({'check': name, 'passed': bool(condition), 'detail': detail})

    package = ROOT/'data/processed/resilience.gpkg'
    layers = {name: gpd.read_file(package, layer=name) for name in
              ['neighbourhoods', 'boroughs', 'greenspaces', 'entrances', 'case_origins', 'flood_extent']}
    for name, frame in layers.items():
        check(f'{name}: nonempty required layer', len(frame) > 0, len(frame))
        check(f'{name}: BNG CRS', frame.crs.to_epsg() == 27700)
        check(f'{name}: valid nonempty geometries', frame.geometry.is_valid.all() and not frame.geometry.is_empty.any())
        allowed = {'Point'} if name in {'entrances', 'case_origins'} else {'Polygon', 'MultiPolygon'}
        check(f'{name}: geometry types', set(frame.geom_type).issubset(allowed), sorted(set(frame.geom_type)))
    frame = layers['neighbourhoods']
    check('Unique complete neighbourhood IDs', frame.lsoa21cd.notna().all() and not frame.lsoa21cd.duplicated().any())
    check('Study geography snapshot', len(frame) == 1702 and set(frame.borough) == set(config['boroughs']))
    check('Census regional reconciliation', frame.population.sum() == 2867735, int(frame.population.sum()))
    extent = layers['flood_extent']
    check('Flood geometries have unique scenario-area keys', not extent.duplicated(['hazard', 'lsoa21cd']).any())
    check('Both flood scenarios persist', set(extent.hazard) == {'defended_1pct', 'defended_01pct'})
    for hazard, column in [('defended_1pct', 'flood_exposure'), ('defended_01pct', 'flood_01pct')]:
        selected = extent.loc[extent.hazard == hazard].copy()
        selected['measured_area'] = selected.geometry.area
        measured = selected.groupby('lsoa21cd').measured_area.sum().reindex(frame.lsoa21cd, fill_value=0).to_numpy()
        check(f'{hazard}: saved overlay areas match indicators', np.allclose(measured/frame.geometry.area.to_numpy(), frame[column], atol=1e-8, rtol=0))
    xmin, ymin, xmax, ymax = frame.total_bounds
    check('Plausible BNG extent', 340000 < xmin < xmax < 430000 and 380000 < ymin < ymax < 450000,
          frame.total_bounds.tolist())
    for field in ['flood_exposure', 'flood_01pct', 'access_deficit', 'income_rate', 'age_sensitive_share', 'heat_coverage']:
        check(f'{field}: complete unit interval', frame[field].notna().all() and frame[field].between(0, 1).all())
    complete = frame[config['indicators']].apply(np.isfinite).all(axis=1)
    check('Eligibility exactly matches finite evidence', np.array_equal(frame.model_eligible.astype(bool), complete))
    check('Heat coverage rule applied', frame.loc[frame.heat_coverage < config['minimum_lsoa_heat_coverage'], 'heat_anomaly'].isna().all())
    check('Missing evidence is not ranked', frame.loc[~complete, 'rank_balanced'].isna().all())
    expected, diagnostics = fit_model(frame, config)
    numeric = [f'{prefix}_{name}' for name in config['scenarios'] for prefix in ['score', 'rank']]
    numeric += ['top20_frequency', 'rank_p05', 'rank_p95']
    for field in numeric:
        check(f'{field}: seeded reproduction', np.allclose(frame[field], expected[field], equal_nan=True, rtol=0, atol=1e-10))
    check('Persistent-priority rule reproduces', np.array_equal(frame.robust_priority.astype(bool), expected.robust_priority))
    dimensions = ['dim_flood', 'dim_heat', 'dim_access', 'dim_income', 'dim_age']
    for name, weights in config['scenarios'].items():
        check(f'{name}: weights sum to one', math.isclose(sum(weights), 1, abs_tol=1e-12))
        independent = frame.loc[complete, dimensions].to_numpy() @ np.array(weights)
        check(f'{name}: direct weighted sum', np.allclose(independent, frame.loc[complete, f'score_{name}'], rtol=0, atol=1e-10))
    for raw, dimension in zip(config['indicators'], dimensions):
        order = frame.loc[complete].sort_values(raw)
        check(f'{raw}: adverse-high monotonicity', (np.diff(order[dimension]) >= -1e-12).all())

    origins = layers['case_origins']
    entrances = layers['entrances']
    check('Unique case-study origins', origins.origin_id.notna().all() and not origins.origin_id.duplicated().any())
    check('OA population reconciliation', len(origins) == 2466 and origins.population.sum() == 821878)
    check('Reachability and distance agree', np.array_equal(origins.reachable.astype(bool), origins.distance_m.notna()))
    check('All case origins snapped within tolerance', origins.known.all())
    euclidean, _ = cKDTree(np.column_stack([entrances.geometry.x, entrances.geometry.y])).query(
        np.column_stack([origins.geometry.x, origins.geometry.y]))
    reachable = origins.reachable.astype(bool).to_numpy()
    check('Network distance is at least straight-line distance', (origins.loc[reachable, 'distance_m'].to_numpy() + 1e-6 >= euclidean[reachable]).all())
    case = frame.loc[frame.case_study.astype(bool)]
    check('Walking-threshold sensitivity is monotonic', ((case.network_deficit_500m >= case.network_deficit_1000m) &
          (case.network_deficit_1000m >= case.network_deficit_1500m)).all())
    check('Case metric is absent outside its study extent', frame.loc[~frame.case_study.astype(bool), 'case_network_deficit'].isna().all())
    case_by_id = case.set_index('lsoa21cd')
    for threshold in config['walking_sensitivity_m']:
        records = {}
        for code, group in origins.groupby('lsoa21cd'):
            known = group.known.astype(bool)
            eligible_population = group.loc[known, 'population'].sum()
            coverage = eligible_population/group.population.sum()
            deficit = known & (~group.reachable.astype(bool) | (group.distance_m > threshold))
            records[code] = group.loc[deficit, 'population'].sum()/eligible_population if coverage >= .8 and eligible_population else np.nan
        independent = pd.Series(records).reindex(case_by_id.index)
        check(f'Case access {threshold}m: OA weighted reconstruction', np.allclose(independent, case_by_id[f'network_deficit_{threshold}m'], equal_nan=True, rtol=0, atol=1e-12))
    check('Case headline equals configured threshold', np.allclose(case.case_network_deficit,
          case[f"network_deficit_{config['walking_threshold_m']}m"], equal_nan=True))

    raster_info = {}
    reference = None
    for name in ['heat_anomaly', 'surface_temperature', 'ndvi', 'heat_observations']:
        with rasterio.open(ROOT/f'data/processed/{name}.tif') as source:
            values = source.read(1, masked=True)
            check(f'{name}: raster CRS/resolution/nodata', source.crs.to_epsg() == 27700 and
                  source.res == (config['raster_resolution_m'],)*2 and source.nodata == -9999)
            check(f'{name}: finite observed values', np.isfinite(values.compressed()).all() and values.count() > 0)
            grid = (source.transform, source.shape)
            reference = reference or grid
            check(f'{name}: common grid', grid == reference)
            raster_info[name] = {'shape': list(source.shape), 'valid_cells': int(values.count()),
                                 'min': float(values.min()), 'max': float(values.max())}
            if name == 'heat_anomaly':
                heat_valid = ~np.ma.getmaskarray(values)
            if name == 'heat_observations':
                check('Every heat cell has minimum observations', (values.data[heat_valid] >= config['minimum_heat_observations']).all())
    check('NDVI is a bounded vegetation index', -1 <= raster_info['ndvi']['min'] <= raster_info['ndvi']['max'] <= 1)
    contract = json.loads((ROOT/'metadata/analysis_contract.json').read_text())
    check('Configuration matches analysed snapshot', contract['configuration_sha256'] == sha256(ROOT/'config/analysis.json'))
    check('Heat raster matches analysed snapshot', contract['heat_input_sha256'] == sha256(ROOT/'data/processed/heat_anomaly.tif'))
    check('Indicator input matches analysed snapshot', contract['indicator_input_sha256'] == sha256(ROOT/'data/interim/indicators.parquet'))
    delivered = frame.set_index('lsoa21cd').sort_index()
    input_frame = gpd.read_parquet(ROOT/'data/interim/indicators.parquet').set_index('lsoa21cd').sort_index()
    check('GeoPackage uses the analysed source indicators', delivered.index.equals(input_frame.index) and
          np.allclose(delivered[config['indicators']], input_frame[config['indicators']], equal_nan=True, rtol=0, atol=1e-12))
    for label, other in [
        ('CSV', pd.read_csv(ROOT/'outputs/tables/neighbourhood_indicators.csv')),
        ('GeoParquet', gpd.read_parquet(ROOT/'data/processed/neighbourhoods.parquet')),
    ]:
        other = other.set_index('lsoa21cd').sort_index()
        fields = config['indicators'] + numeric
        check(f'{label}: delivered values match GeoPackage', delivered.index.equals(other.index) and
              np.allclose(delivered[fields], other[fields], equal_nan=True, rtol=0, atol=1e-10))
    source_records = []
    for sidecar in sorted((ROOT/'data/raw').rglob('*.source.json')):
        record = json.loads(sidecar.read_text())
        path = ROOT/record['file']
        valid = path.exists() and sha256(path) == record['sha256']
        check('Source checksum: '+record['file'], valid)
        source_records.append(record)
    check('Provenance is populated', len(source_records) > 25)
    save_json(ROOT/'metadata/provenance_manifest.json', {'sources': source_records,
              'verified_utc': datetime.now(timezone.utc).isoformat(), 'source_files': len(source_records)})

    from qgis.core import QgsApplication, QgsProject
    project = QgsProject()
    project_path = ROOT/'qgis/project/gm_climate_resilience.qgz'
    check('QGIS project reopens', project.read(str(project_path)))
    check('All QGIS layers resolve', len(project.mapLayers()) >= 12 and all(layer.isValid() for layer in project.mapLayers().values()))
    check('QGIS vector providers return readable counts', all(layer.featureCount() >= 0
          for layer in project.mapLayers().values() if hasattr(layer, 'featureCount')))
    check('Six QGIS layouts persist', len(project.layoutManager().layouts()) == 6)
    # Exercise the reusable QGIS algorithm on all real rows, independently of
    # the NumPy model implementation and without writing to the source package.
    sys.path.insert(0, str(ROOT/'scripts/qgis'))
    from resilience_processing import ResilienceProvider
    from processing.core.Processing import Processing
    import processing
    Processing.initialize()
    provider = ResilienceProvider()
    QgsApplication.processingRegistry().addProvider(provider)
    parameters = {'INPUT':str(package)+'|layername=neighbourhoods',
                  'FLOOD':'dim_flood','HEAT':'dim_heat','ACCESS':'dim_access','INCOME':'dim_income','AGE':'dim_age',
                  'WEIGHTS':'0.2,0.2,0.2,0.2,0.2','OUTPUT':'memory:'}
    scored = processing.run('resilience:weightedneighbourhoodscores', parameters)['OUTPUT']
    processing_scores = {}
    for feature in scored.getFeatures():
        if feature['score_valid']:
            processing_scores[feature['lsoa21cd']] = float(feature['weighted_score'])
    wanted = frame.loc[complete].set_index('lsoa21cd').score_balanced
    check('QGIS Processing preserves all rows', scored.featureCount() == len(frame))
    check('QGIS Processing reproduces complete-cohort scores', set(processing_scores) == set(wanted.index) and
          np.allclose(pd.Series(processing_scores).reindex(wanted.index), wanted, rtol=0, atol=1e-10))
    save_json(ROOT/'metadata/processing_validation.json', {'features':scored.featureCount(),
              'valid_scores':len(processing_scores), 'algorithm':'resilience:weightedneighbourhoodscores',
              'checks':[row for row in checks if row['check'].startswith('QGIS Processing')]})
    del scored
    QgsApplication.processingRegistry().removeProvider(provider)
    with zipfile.ZipFile(project_path) as archive:
        xml = archive.read(next(name for name in archive.namelist() if name.endswith('.qgs'))).decode()
        check('QGIS paths do not embed this workspace', str(ROOT) not in xml)
    qgis_build = json.loads((ROOT/'metadata/qgis_build.json').read_text())
    relocation = json.loads((ROOT/'metadata/qgis_relocation.json').read_text())
    check('Latest QGIS build passed', qgis_build.get('status') == 'passed')
    check('QGIS relocation passed', relocation.get('status') == 'passed')
    for record in relocation.get('copied_files', []):
        check('Relocation evidence matches current file: '+record['path'], sha256(ROOT/record['path']) == record['sha256'])
    check('Relocation verifies all six rendered layouts', len(relocation.get('layout_render_checks', [])) == 6 and
          all(row.get('pixel_identical', False) for row in relocation.get('layout_render_checks', [])))
    export_info = []
    for name in ['01_overview', '02_flood', '03_heat', '04_accessibility', '05_priorities', '06_robustness']:
        png = ROOT/f'outputs/maps/{name}.png'
        pdf = ROOT/f'outputs/maps/{name}.pdf'
        if png.exists():
            with Image.open(png) as picture:
                picture.load()
                check(f'{name}: A3 landscape PNG', picture.width > 2500 and 1.39 < picture.width/picture.height < 1.43)
                export_info.append({'map': name, 'pixels': list(picture.size), 'pdf_bytes': pdf.stat().st_size if pdf.exists() else 0})
        else:
            check(f'{name}: PNG exists', False)
        check(f'{name}: PDF readable header', pdf.exists() and pdf.read_bytes().startswith(b'%PDF') and pdf.stat().st_size > 10000)
    for name in ['scenario_ranks', 'weight_sensitivity', 'indicator_correlations', 'moran_scatter']:
        check(f'{name}: figure pair', all((ROOT/f'outputs/figures/{name}.{extension}').exists() for extension in ['png', 'svg']))
    check('Offline interactive output exists', (ROOT/'outputs/portfolio/interactive.html').exists())
    project.clear()
    report = {'status': 'passed' if all(row['passed'] for row in checks) else 'failed',
              'checked_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
              'passed_checks': sum(row['passed'] for row in checks), 'total_checks': len(checks),
              'eligible_neighbourhoods': int(complete.sum()), 'excluded_neighbourhoods': int((~complete).sum()),
              'raster_info': raster_info, 'exports': export_info,
              'limits': 'Automated integrity checks do not establish causal validity, future climate risk, or intervention effectiveness. Visual inspection is recorded separately.'}
    attested_directories = ['data/processed', 'outputs/tables', 'outputs/figures', 'outputs/portfolio']
    report['artifact_sha256'] = {
        str(path.relative_to(ROOT)): sha256(path)
        for directory in attested_directories
        for path in sorted((ROOT/directory).rglob('*'))
        if path.is_file()
    }
    save_json(ROOT/'outputs/qa/validation.json', report)
    failed = [row['check'] for row in checks if not row['passed']]
    if failed:
        raise ValueError('Validation failed: '+ '; '.join(failed))
    return {key: report[key] for key in ['status', 'passed_checks', 'total_checks', 'eligible_neighbourhoods', 'excluded_neighbourhoods']}
