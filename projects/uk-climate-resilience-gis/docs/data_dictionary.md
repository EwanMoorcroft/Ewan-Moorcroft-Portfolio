# Analytical data dictionary

The GeoPackage layer `neighbourhoods`, `data/processed/neighbourhoods.parquet` and `outputs/tables/neighbourhood_indicators.csv` contain the same analytical rows. CSV omits geometry; GeoPackage stores nonfinite distances as NULL. [The generated schema](../metadata/analytical_schema.csv) records exact field names, dtypes and null counts for the delivered snapshot.

| Field or family | Meaning and unit |
| --- | --- |
| `lsoa21cd`, `name`, `borough`, `msoa21cd` | Census 2021 small-area IDs/names; administrative context |
| `population` | Census 2021 usual residents from the LSOA age table |
| `age_sensitive_share` | (Under-5 + age-65-plus residents) / same-table population, fraction 0–1 |
| `income_rate` | IoD2025 income-deprivation rate, fraction 0–1 |
| `imd_score`, `imd_decile` | Context only; overall IMD is excluded from the weighted composite |
| `households`, `households_with_access` | Official Defra Scenario 5 aggregate counts |
| `access_deficit` | Household share lacking Scenario 5 access, 0–1; regional composite input |
| `area_km2` | Full projected LSOA polygon area, km² |
| `flood_exposure` | Primary defended scenario overlap / full LSOA polygon area, fraction |
| `flood_01pct` | Alternative defended 0.1% scenario overlap fraction |
| `heat_anomaly` | Area-weighted selected-date surface anomaly, °C; NULL below the LSOA coverage gate |
| `heat_raw_mean` | Observed area-weighted anomaly before applying the LSOA coverage gate; not an eligible model value by itself |
| `heat_coverage` | Fraction of full LSOA area with an adequate observed heat raster value |
| `heat_anomaly_cells`, `ndvi_cells` | Number of grid cells touching the LSOA, including unobserved cells; not the number of satellite dates |
| `ndvi`, `ndvi_coverage` | Contextual vegetation index and its observed LSOA area fraction |
| `case_study` | Manchester–Salford case membership |
| `case_network_population` | Sum of OA population assigned to each case LSOA |
| `network_coverage` | Population share whose centroid snapped to the modelled graph; this is not routeability |
| `case_network_deficit`, `network_deficit_1000m` | OA resident-weighted share beyond 1 km or with no mapped route to a qualifying public park; case only |
| `network_deficit_500m`, `network_deficit_1500m` | Same model and denominator at alternative distance thresholds |
| `model_eligible`, `model_missing_indicators` | Complete evidence flag and explicit exclusion reason |
| `dim_flood`, `dim_heat`, `dim_access`, `dim_income`, `dim_age` | Midrank-normalised adverse-high dimensions, 0–100 on the common eligible cohort |
| `score_balanced`, `score_hazard`, `score_equity` | Relative weighted scores, 0–100; not probabilities |
| `rank_balanced`, `rank_hazard`, `rank_equity` | Descending average ranks; 1 is highest score and ties are preserved |
| `top20_frequency` | Fraction of 2,000 configured weight draws placing the area in the inclusive highest quintile |
| `rank_p05`, `rank_p95`, `rank_span_fraction` | Empirical weight-experiment rank interval and span divided by `n−1`; not confidence limits |
| `robust_priority` | Top group in all named scenarios and ≥80% draw inclusion; false also for unassessed rows, so interpret with eligibility |
| `priority_class` | Persistent, scenario-sensitive, single-hazard review, other assessed, or insufficient evidence |
| `follow_up` | Rule-generated investigation theme based on relative components; not an approved intervention |
| `spatial_eligible`, `z_score`, `spatial_lag_z`, `quadrant` | Global-statistic cohort and descriptive centred/standardised spatial-lag diagnostics; quadrants have no local significance claim |

Other GeoPackage layers are `boroughs`, `greenspaces`, `entrances`, `case_origins` and `flood_extent`. The latter must be filtered by `hazard` (`defended_1pct` or `defended_01pct`) to avoid accidentally combining scenarios. In `case_origins`, `known` means within snap tolerance, `reachable` means a finite graph route, `snap_m` is connector distance and `distance_m` is total nearest-destination path distance. A known but unreachable origin has a NULL distance in the GeoPackage, plus `known=true`, `reachable=false`; it is retained in the deficit denominator.
