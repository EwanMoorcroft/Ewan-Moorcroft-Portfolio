"""Integrate reviewed indicators, policy scenarios and decision outputs."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .io import ROOT,save_json,sha256
from .model import fit_model
from .preprocess import validated_join,zonal_raster
from .spatial import spatial_diagnostics

LOG=logging.getLogger(__name__)


def assemble_indicators(config: dict) -> gpd.GeoDataFrame:
    areas=gpd.read_parquet(ROOT/'data/interim/neighbourhoods.parquet')
    flood=pd.read_parquet(ROOT/'data/interim/flood_indicators.parquet')
    areas=validated_join(areas,flood,'lsoa21cd')
    for raster_name, field in [('heat_anomaly','heat_anomaly'),('ndvi','ndvi')]:
        stats=zonal_raster(areas,ROOT/f'data/processed/{raster_name}.tif')
        stats=stats.rename(columns={'mean':field,'coverage':('heat_coverage' if field=='heat_anomaly' else 'ndvi_coverage'),'cells':field+'_cells'})
        areas=validated_join(areas,stats,'lsoa21cd')
    areas['heat_raw_mean']=areas.heat_anomaly
    areas.loc[areas.heat_coverage<config['minimum_lsoa_heat_coverage'],'heat_anomaly']=np.nan
    network=pd.read_parquet(ROOT/'data/interim/network_indicators.parquet')
    if network.lsoa21cd.duplicated().any() or set(network.lsoa21cd)!=set(areas.loc[areas.case_study,'lsoa21cd']):
        raise ValueError('Case-study network aggregation is incomplete or duplicated')
    areas=areas.merge(network,on='lsoa21cd',how='left',validate='one_to_one')
    areas.to_parquet(ROOT/'data/interim/indicators.parquet',index=False)
    return areas


def _investigation(row) -> str:
    if not row.model_eligible:
        return 'Improve heat evidence before composite comparison'
    themes=[]
    if row.dim_flood>=80:
        themes.append('River-flood resilience and local hydraulic assessment')
    if row.dim_heat>=80:
        themes.append('Shade, cooling and overheating assessment')
    if row.dim_access>=80:
        themes.append('Large-greenspace access and provision assessment')
    if row.dim_income>=80 or row.dim_age>=80:
        themes.append('Inclusive engagement and heat-preparedness assessment')
    return '; '.join(themes) if themes else 'Review component evidence and local adaptation needs'


def _write_geopackage(layers: dict[str,gpd.GeoDataFrame]) -> None:
    path=ROOT/'data/processed/resilience.gpkg'
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('resilience.building.gpkg')
    if temporary.exists():
        temporary.unlink()
    for name, layer in layers.items():
        copy=layer.copy()
        for column in copy.select_dtypes(include=['float']).columns:
            copy[column]=copy[column].replace([np.inf,-np.inf],np.nan)
        if len(copy):
            copy.to_file(temporary,layer=name,driver='GPKG',engine='pyogrio')
    temporary.replace(path)


def run_analysis(config: dict) -> dict:
    areas=gpd.read_parquet(ROOT/'data/interim/indicators.parquet')
    model,diagnostics=fit_model(areas,config)
    save_json(ROOT/'outputs/qa/model.json',diagnostics)
    top_cut=diagnostics['scenarios']['balanced']['top_threshold']
    model['priority_class']='Other assessed neighbourhood'
    model.loc[(model.dim_flood>=95)|(model.dim_heat>=95),'priority_class']='Single-hazard review'
    model.loc[model.score_balanced>=top_cut,'priority_class']='Scenario-sensitive priority'
    model.loc[model.robust_priority,'priority_class']='Persistent priority'
    model.loc[~model.model_eligible,'priority_class']='Insufficient heat evidence'
    model['follow_up']=[_investigation(row) for row in model.itertuples()]
    spatial,spatial_report=spatial_diagnostics(model,config)
    model=validated_join(model,spatial,'lsoa21cd')
    save_json(ROOT/'outputs/qa/spatial.json',spatial_report)
    eligible=model.loc[model.model_eligible]
    # A different flood scenario is re-ranked on the identical comparison cohort.
    alternative=eligible.copy();alternative['flood_exposure']=alternative.flood_01pct
    alt,alt_report=fit_model(alternative,config)
    base_top=set(eligible.loc[eligible.score_balanced>=top_cut,'lsoa21cd'])
    alt_top=set(alt.loc[alt.score_balanced>=alt_report['scenarios']['balanced']['top_threshold'],'lsoa21cd'])
    flood_sensitivity={'cohort_n':len(eligible),'rank_spearman':float(spearmanr(eligible.score_balanced,alt.score_balanced).statistic),
        'top_group_jaccard':len(base_top&alt_top)/len(base_top|alt_top),'retained_baseline_priorities':len(base_top&alt_top),
        'interpretation':'Alternative defended 0.1% annual probability footprint; policy scenarios and complete comparison cohort unchanged'}
    save_json(ROOT/'outputs/qa/flood_sensitivity.json',flood_sensitivity)
    table_dir=ROOT/'outputs/tables';table_dir.mkdir(parents=True,exist_ok=True)
    plain=pd.DataFrame(model.drop(columns='geometry'))
    plain.to_csv(table_dir/'neighbourhood_indicators.csv',index=False)
    shortlist=plain.loc[plain.score_balanced>=top_cut].sort_values('rank_balanced')
    shortlist.to_csv(table_dir/'priority_shortlist.csv',index=False)
    plain.loc[~plain.model_eligible,['lsoa21cd','name','borough','population','heat_coverage','model_missing_indicators']].to_csv(table_dir/'evidence_gaps.csv',index=False)
    summary_rows=[]
    for borough,group in model.groupby('borough'):
        summary_rows.append({'borough':borough,'neighbourhoods':len(group),'population':int(group.population.sum()),
            'assessed_neighbourhoods':int(group.model_eligible.sum()),'assessed_population':int(group.loc[group.model_eligible,'population'].sum()),
            'excluded_neighbourhoods':int((~group.model_eligible).sum()),
            'assessed_neighbourhood_share':float(group.model_eligible.mean()),
            'persistent_priority_neighbourhoods':int(group.robust_priority.sum()),
            'residents_in_persistent_priority_areas':int(group.loc[group.robust_priority,'population'].sum()),
            'households_without_scenario5_access_share':float(1-group.households_with_access.sum()/group.households.sum())})
    pd.DataFrame(summary_rows).to_csv(table_dir/'borough_summary.csv',index=False)
    boroughs=model[['borough','geometry']].dissolve(by='borough').reset_index()
    parks=gpd.read_parquet(ROOT/'data/interim/greenspaces.parquet')
    entrances=gpd.read_parquet(ROOT/'data/interim/entrances.parquet')
    origins=gpd.read_parquet(ROOT/'data/interim/network_origins.parquet')
    flood_extent=gpd.read_parquet(ROOT/'data/interim/flood_extent.parquet')
    _write_geopackage({'neighbourhoods':model,'boroughs':boroughs,'greenspaces':parks,'entrances':entrances,
                      'case_origins':origins,'flood_extent':flood_extent})
    model.to_parquet(ROOT/'data/processed/neighbourhoods.parquet',index=False)
    summary={'study':'Greater Manchester','comparison_unit':'LSOA 2021','neighbourhoods':len(model),
        'population_census2021':int(model.population.sum()),'eligible_neighbourhoods':int(model.model_eligible.sum()),
        'eligible_population':int(model.loc[model.model_eligible,'population'].sum()),
        'excluded_neighbourhoods':int((~model.model_eligible).sum()),'excluded_population':int(model.loc[~model.model_eligible,'population'].sum()),
        'baseline_priority_count':len(shortlist),'persistent_priority_count':int(model.robust_priority.sum()),
        'residents_in_persistent_priority_areas':int(model.loc[model.robust_priority,'population'].sum()),
        'households_without_large_greenspace_access_share':float(1-model.households_with_access.sum()/model.households.sum()),
        'case_modelled_residents_beyond_1km_share':float(origins.loc[origins.known & (origins.distance_m>config['walking_threshold_m']),'population'].sum()/origins.loc[origins.known,'population'].sum()),
        'case_access_metric_definition':'Beyond the configured walking threshold or no mapped route; successfully snapped OA origins, population weighted',
        'case_origins_without_route':int((origins.known & ~origins.reachable).sum()),
        'case_residents_without_route':int(origins.loc[origins.known & ~origins.reachable,'population'].sum()),
        'case_population':int(origins.population.sum()),
        'case_deficit_if_disconnected_accessible':float(origins.loc[origins.reachable & (origins.distance_m>config['walking_threshold_m']),'population'].sum()/origins.population.sum()),
        'scenario_rank_correlations':diagnostics['scenario_rank_spearman'],
        'flood_sensitivity':flood_sensitivity,
        'moran_i':spatial_report['primary']['statistic'],'moran_p_two_sided':spatial_report['primary']['p_two_sided'],
        'top10':shortlist[['lsoa21cd','name','borough','score_balanced','rank_balanced','top20_frequency','priority_class','follow_up']].head(10).to_dict('records'),
        'claim_boundary':'Priorities identify places for local investigation. Population figures describe residents of neighbourhoods, not people predicted to experience harm.'}
    save_json(ROOT/'outputs/qa/summary.json',summary)
    pd.DataFrame([{'field':column,'dtype':str(model[column].dtype),'null_count':int(model[column].isna().sum())}
                  for column in model.columns if column!='geometry']).to_csv(ROOT/'metadata/analytical_schema.csv',index=False)
    save_json(ROOT/'metadata/analysis_contract.json', {
        'configuration_sha256':sha256(ROOT/'config/analysis.json'),
        'indicator_input_sha256':sha256(ROOT/'data/interim/indicators.parquet'),
        'heat_input_sha256':sha256(ROOT/'data/processed/heat_anomaly.tif'),
        'seed':config['seed'], 'cohort_n':len(eligible),
        'interpretation':'Numeric outputs reproduce from the cached inputs and this configuration; container file bytes may include writer metadata.'})
    return summary
