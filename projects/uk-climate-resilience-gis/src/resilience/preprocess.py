"""Validated joins, source-preserving aggregation and projected spatial data."""
from __future__ import annotations

import gzip
import json
import logging
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window

from .io import ROOT, save_json

LOG = logging.getLogger(__name__)


def validated_join(left, right, key: str):
    if left[key].isna().any() or right[key].isna().any() or left[key].duplicated().any() or right[key].duplicated().any():
        raise ValueError(f"Null or duplicated join key: {key}")
    result = left.merge(right, on=key, how="left", validate="one_to_one", indicator=True)
    if not result['_merge'].eq('both').all():
        raise ValueError(f"Unmatched source records: {result.loc[result._merge!='both',key].tolist()[:12]}")
    return result.drop(columns='_merge')


def census_age(path: Path, code_field: str, output_key: str) -> pd.DataFrame:
    table = pd.read_csv(path, usecols=[code_field,'Age (B) (18 categories) Code','Count'])
    table.columns = [output_key,'age_code','count']
    if table.duplicated([output_key,'age_code']).any() or (table['count'] < 0).any():
        raise ValueError('Duplicated or negative Census cells')
    wide = table.pivot(index=output_key, columns='age_code', values='count')
    if set(wide.columns) != set(range(1,19)) or wide.isna().any().any():
        raise ValueError('Census age categories must be 18 complete mutually exclusive bins')
    result = pd.DataFrame({'population':wide.sum(axis=1), 'children_0_4':wide[1], 'older_65_plus':wide[list(range(14,19))].sum(axis=1)})
    result['age_sensitive_share'] = (result.children_0_4+result.older_65_plus)/result.population
    return result.reset_index()


def extract_access_ods(path: Path, lsoa_codes: set[str], scenario=5) -> pd.DataFrame:
    cache = ROOT / f'data/interim/official_access_scenario{scenario}.parquet'
    if cache.exists():
        result = pd.read_parquet(cache)
        if set(result.lsoa21cd) != lsoa_codes:
            raise ValueError('Access extraction cache has a different study population')
        return result
    table_ns = '{urn:oasis:names:tc:opendocument:xmlns:table:1.0}'
    office_ns = '{urn:oasis:names:tc:opendocument:xmlns:office:1.0}'
    current = None
    rows = []
    with zipfile.ZipFile(path) as archive, archive.open('content.xml') as stream:
        for event, element in ET.iterparse(stream, events=['start','end']):
            if event=='start' and element.tag==table_ns+'table':
                current = element.get(table_ns+'name')
            if event=='end' and element.tag==table_ns+'table-row':
                if current==f'Scenario{scenario}':
                    values=[]
                    for cell in element:
                        value = cell.get(office_ns+'value')
                        if value is None:
                            value = ''.join(cell.itertext())
                        repeat = min(int(cell.get(table_ns+'number-columns-repeated','1')),10)
                        values.extend([value]*repeat)
                    if len(values)>=6 and values[1] in lsoa_codes:
                        rows.append((values[0],values[1],values[2],float(values[3]),float(values[4])))
                element.clear()
            if event=='end' and element.tag==table_ns+'table':
                element.clear()
                if current==f'Scenario{scenario}':
                    break
    result = pd.DataFrame(rows,columns=['oa21cd','lsoa21cd','msoa21cd','households','households_with_access'])
    if result.empty or result.oa21cd.duplicated().any() or set(result.lsoa21cd) != lsoa_codes:
        raise ValueError('Official access extraction has incomplete/duplicate study-area coverage')
    if (result.households<=0).any() or (result.households_with_access<0).any() or (result.households_with_access > result.households).any():
        raise ValueError('Invalid official greenspace access numerator/denominator')
    cache.parent.mkdir(parents=True,exist_ok=True)
    result.to_parquet(cache,index=False)
    return result


def prepare_neighbourhoods(config: dict) -> gpd.GeoDataFrame:
    boundary = gpd.read_file(ROOT/'data/raw/ons/region_lsoas.geojson').to_crs(27700)
    boundary = boundary.rename(columns={'LSOA21CD':'lsoa21cd','LSOA21NM':'name'})
    before_invalid = int((~boundary.is_valid).sum())
    boundary.geometry = boundary.geometry.make_valid()
    if not boundary.is_valid.all() or boundary.geometry.is_empty.any():
        raise ValueError('Invalid study-area boundary after explicit geometry repair')
    imd = pd.read_csv(ROOT/'data/raw/mhclg/imd2025_file7.csv')
    fields = {'LSOA code (2021)':'lsoa21cd','Local Authority District name (2024)':'borough',
              'Income Score (rate)':'income_rate','Index of Multiple Deprivation (IMD) Score':'imd_score',
              'Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)':'imd_decile'}
    imd = imd.rename(columns=fields)[list(fields.values())]
    joined = validated_join(boundary,imd,'lsoa21cd')
    age = census_age(ROOT/'data/raw/ons/census2021_ts007a_lsoa.csv','Lower Layer Super Output Areas Code','lsoa21cd')
    joined = validated_join(joined,age,'lsoa21cd')
    oa_access = extract_access_ods(ROOT/'data/raw/defra/greenspace_access_2024_corrected2025.ods',set(joined.lsoa21cd),config['official_greenspace_scenario'])
    access = oa_access.groupby('lsoa21cd')[['households','households_with_access']].sum().reset_index()
    access['access_deficit'] = 1-access.households_with_access/access.households
    joined = validated_join(joined,access,'lsoa21cd')
    msoa = oa_access.groupby('lsoa21cd').msoa21cd.agg(lambda s: s.unique()[0] if s.nunique()==1 else None).reset_index()
    joined = validated_join(joined,msoa,'lsoa21cd')
    joined['area_km2'] = joined.area/1e6
    joined['case_study'] = joined.borough.isin(config['case_study_boroughs'])
    if set(joined.borough) != set(config['boroughs']):
        raise ValueError('Unexpected borough coverage')
    (ROOT/'data/interim').mkdir(parents=True,exist_ok=True)
    joined.to_parquet(ROOT/'data/interim/neighbourhoods.parquet',index=False)
    save_json(ROOT/'outputs/qa/core_joins.json',{'neighbourhoods':len(joined),'geometry_repairs':before_invalid,
        'population_census2021':int(joined.population.sum()),'households_official_access':float(joined.households.sum()),
        'all_core_join_match_rates':1.0,'official_access_OAs':len(oa_access),
        'access_scenario':config['official_greenspace_scenario'],'borough_counts':joined.groupby('borough').size().to_dict()})
    return joined


def prepare_greenspaces(boundaries: gpd.GeoDataFrame, config: dict) -> tuple[gpd.GeoDataFrame,gpd.GeoDataFrame]:
    sites, points = [], []
    envelope = boundaries.geometry.union_all().buffer(config['network_buffer_m'])
    for path in sorted((ROOT/'data/raw/os').glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith('.shp')]
        for name in names:
            target = f'/vsizip/{path}/{name}'
            layer = gpd.read_file(target,bbox=envelope.bounds).to_crs(27700)
            if name.endswith('_GreenspaceSite.shp'):
                sites.append(layer)
            elif name.endswith('_AccessPoint.shp'):
                points.append(layer)
    parks = gpd.GeoDataFrame(pd.concat(sites,ignore_index=True),crs=27700).drop_duplicates('id')
    entrances = gpd.GeoDataFrame(pd.concat(points,ignore_index=True),crs=27700).drop_duplicates('id')
    parks = parks[parks['function'].eq('Public Park Or Garden') | parks['function'].eq('Public Park or Garden')].copy()
    parks.geometry = parks.geometry.make_valid()
    parks['area_ha'] = parks.area/10000  # Before clipping: eligibility uses whole-site size.
    parks['eligible'] = parks.area_ha >= config['minimum_public_park_ha']
    parks = parks.rename(columns={'id':'site_id','distName1':'name'})
    parks['name'] = parks['name'].fillna('Unnamed public park')
    parks = parks[parks.intersects(envelope)][['site_id','name','function','area_ha','eligible','geometry']]
    entrances = entrances.rename(columns={'id':'entrance_id','refToGSite':'site_id'})
    entrances = entrances[entrances.site_id.isin(parks.loc[parks.eligible,'site_id']) & entrances.accessType.str.contains('Pedestrian',na=False)].copy()
    entrances = entrances[entrances.intersects(envelope)][['entrance_id','site_id','geometry']]
    if parks.eligible.sum()==0 or entrances.empty:
        raise ValueError(f'No eligible public parks/entrances; source function schema is {pd.concat(sites)["function"].unique()}')
    parks.to_parquet(ROOT/'data/interim/greenspaces.parquet',index=False)
    entrances.to_parquet(ROOT/'data/interim/entrances.parquet',index=False)
    return parks,entrances


def prepare_origins() -> gpd.GeoDataFrame:
    points = gpd.read_file(ROOT/'data/raw/ons/oa_centroids_case.geojson').to_crs(27700)
    points = points.rename(columns={'OA21CD':'origin_id'})[['origin_id','geometry']]
    lookup = pd.DataFrame(json.loads((ROOT/'data/raw/ons/oa_lookup_case.json').read_text())['records'])
    lookup = lookup.rename(columns={'OA21CD':'origin_id','LSOA21CD':'lsoa21cd'})[['origin_id','lsoa21cd']]
    points = validated_join(points,lookup,'origin_id')
    population = census_age(ROOT/'data/raw/ons/census2021_ts007a_oa_northwest.csv','Output Areas Code','origin_id')
    points = validated_join(points,population,'origin_id')
    points.to_parquet(ROOT/'data/interim/origins.parquet',index=False)
    return points


def zonal_raster(areas: gpd.GeoDataFrame, path: Path) -> pd.DataFrame:
    records=[]
    with rasterio.open(path) as source:
        if source.crs.to_epsg()!=27700:
            raise ValueError('Raster must be aligned to British National Grid')
        for row in areas.itertuples():
            floating=from_bounds(*row.geometry.bounds,transform=source.transform)
            col0,row0=np.floor([floating.col_off,floating.row_off]).astype(int)
            col1,row1=np.ceil([floating.col_off+floating.width,floating.row_off+floating.height]).astype(int)
            window=Window(col0,row0,col1-col0,row1-row0)
            values=source.read(1,window=window,boundless=True,masked=True)
            if values.size==0:
                records.append((row.lsoa21cd,np.nan,0.,0));continue
            local=source.window_transform(window)
            yy,xx=np.indices(values.shape)
            x0=local.c+xx*local.a;y1=local.f+yy*local.e
            cells=shapely.box(x0,y1+local.e,x0+local.a,y1)
            weights=shapely.area(shapely.intersection(cells,row.geometry))
            valid=(weights>0) & ~np.ma.getmaskarray(values) & np.isfinite(values.data)
            valid_area=weights[valid].sum()
            records.append((row.lsoa21cd,float(np.average(values.data[valid],weights=weights[valid])) if valid.any() else np.nan,
                            float(np.clip(valid_area/row.geometry.area,0,1)),int((weights>0).sum())))
    return pd.DataFrame(records,columns=['lsoa21cd','mean','coverage','cells'])


def process_access(origins, entrances, config: dict) -> tuple[gpd.GeoDataFrame,pd.DataFrame]:
    from .network import build_graph,access_distances
    with gzip.open(ROOT/'data/raw/osm/walking_network.json.gz','rt') as stream:
        graph,edges=build_graph(json.load(stream))
    boundaries=gpd.read_parquet(ROOT/'data/interim/neighbourhoods.parquet')
    case=boundaries.loc[boundaries.borough.isin(config['case_study_boroughs'])].geometry.union_all().buffer(config['network_buffer_m'])
    entrances=entrances.loc[entrances.intersects(case)].copy()
    result=access_distances(graph,edges,origins,entrances,config['max_origin_snap_m'],config['max_entrance_snap_m'])
    diagnostics=dict(result.attrs)
    rows=[]
    for code, group in result.groupby('lsoa21cd'):
        known=group.known.to_numpy()
        population=group.population.to_numpy()
        denominator=float(population[known].sum())
        coverage=denominator/population.sum()
        record={'lsoa21cd':code,'network_coverage':coverage,'case_network_population':int(population.sum()),
                'case_network_deficit':float(np.sum(population[known & (group.distance_m.to_numpy()>config['walking_threshold_m'])])/denominator) if denominator and coverage>=.8 else np.nan}
        for threshold in config['walking_sensitivity_m']:
            record[f'network_deficit_{threshold}m']=float(np.sum(population[known & (group.distance_m.to_numpy()>threshold)])/denominator) if denominator and coverage>=.8 else np.nan
        rows.append(record)
    diagnostics.update(origins=len(result),known_origins=int(result.known.sum()),reachable_origins=int(result.reachable.sum()),
        population=int(result.population.sum()),known_population=int(result.loc[result.known,'population'].sum()),
        threshold_m=config['walking_threshold_m'],min_park_ha=config['minimum_public_park_ha'],
        interpretation='Modelled resident access from population-weighted Output Area centroids to OS public-park entrances; differs from official household destinations and vintage')
    save_json(ROOT/'outputs/qa/network.json',diagnostics)
    result.to_parquet(ROOT/'data/interim/network_origins.parquet',index=False)
    return result,pd.DataFrame(rows)
