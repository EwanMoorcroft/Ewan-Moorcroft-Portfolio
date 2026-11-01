# Data sources and provenance

This study combines eight external input families for Greater Manchester screening and a Manchester–Salford walking-access case study. The [machine-readable catalogue](../metadata/data_catalogue.csv) identifies each input, its role and its limitations. Two additional flood products are listed as research-only and are excluded from the analysis.

Acquisition records dated 29 September 2026 sit beside the cached files in `data/raw/` as `*.source.json`. They retain the source URL, request parameters, UTC acquisition time, file size and SHA-256 checksum. The [source configuration](../config/sources.json) records service endpoints, versions and attribution. The cached `data/raw/core_paths.json` file locates the boundary, demographic and greenspace files; raw data are local cache material rather than repository source files. Where an API result combines pages, the component responses and their acquisition records are retained.

## Geography and population

The study uses Census 2021 LSOAs throughout. The ONS [full-resolution, coastline-clipped V10 boundary service](https://www.arcgis.com/home/item.html?id=2bbaef5230694f3abae4f9145a3a9800) supplies the 1,702 LSOAs across the ten Greater Manchester boroughs. Queries retain their original polygon geometry in WGS 84 GeoJSON. Distance and area calculations use British National Grid, **EPSG:27700**; latitude and longitude are not measurement units for those calculations.

The ONS [TS007a five-year age-band tables](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/adhocs/15432ts007acensus2021agebyfiveyearagebands) describe usual residents on **21 March 2021** and were published on 13 January 2023. The LSOA file supplies regional age proportions; the North West OA file supplies case-study routing weights. All 18 disjoint age categories are summed for each table's own denominator. Category 1 represents ages 0–4; categories 14–18 represent ages 65 and over.

The case study has 456 LSOAs and 2,466 OAs. Its population total is **821,867 in the LSOA age table** and **821,878 in the OA age table**, a difference of 11 people. Census disclosure control can create differences between separately published tables. Retaining each table's denominator avoids forcing an artificial reconciliation: LSOA age shares use LSOA totals, while OA routing summaries use OA totals. Neither count is presented as a 2026 population estimate.

The [ONS exact-fit lookup V3](https://www.arcgis.com/home/item.html?id=b9ca90c10aaa4b8d9791e9859a38ca67) links `OA21CD` to `LSOA21CD` and `MSOA21CD`. Its published title refers to December 2021, but the service fields are `LAD22CD` and `LAD22NM`; the cached case selection uses those actual fields. [ONS population-weighted centroids V4](https://www.arcgis.com/home/item.html?id=d05a5fdaf2e44615acc0b9d5b08592d6) supply the OA origin points. A single centroid represents an OA's population distribution approximately; it is neither a household address nor an observed trip origin.

## Deprivation

The [English Indices of Deprivation 2025](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025), corrected File 7, provides LSOA 2021 codes, LAD 2024 names, ranks, deciles, scores and population denominators. The cached CSV is the November 2025 correction. Income deprivation uses `Income Score (rate)`; full IMD deciles provide context. IMD rank is ordinal and is not treated as a measured interval between neighbourhoods. The release year is not the observation year of every underlying indicator; its supplied population denominators refer to mid-2022.

## Flood extent

The input is the Environment Agency's [present-day defended and undefended flood-risk extents](https://www.data.gov.uk/dataset/1589b7ec-2300-4028-9199-aa7cac4b0f68/rivers-and-sea-defended-and-undefended-flood-risk-extents-present-day), dataset `bb486190-bb9c-4eb0-a4b2-c9bd63ca86ec`. Its dataset revision is **20 May 2026**; the portal update is 14 August 2026. The two cached collections are:

| Purpose | Defended scenario | OGC collection |
| --- | --- | --- |
| Primary | 1% river / 0.5% sea annual exceedance probability | `Rivers_1in100_Sea_1in200_defended_extents` |
| Sensitivity | 0.1% river and sea annual exceedance probability | `Rivers_1in1000_Sea_1in1000_defended_extents` |

The measure is the intersected share of the **whole LSOA polygon area**, including any inland water retained in the ONS boundary. It is not a dry-land or residential-land denominator. Defences are assumed to work as designed; asset failure is excluded. This is not an estimate of exposed residents, properties, water depth or future climate risk.

The 0.1% scenario is a separate sensitivity product and is not guaranteed to contain the primary extent everywhere. In the cached overlay, five LSOAs have reductions greater than 0.0001 percentage points in the 0.1% scenario; the maximum reduction is 1.236 percentage points. The project retains this difference rather than forcing the footprints to be nested.

The separate current [Risk of Flooding from Rivers and Sea](https://environment.data.gov.uk/dataset/96ab4342-82c1-4095-87f1-0082e8d84ef1) probability-band and [Risk of Flooding from Surface Water](https://environment.data.gov.uk/dataset/b5aaa28d-6eb9-460e-8d6f-43caa71fbe0e) products were investigated through public area-of-interest requests. They are **not used** in this analysis. The selected flood indicator therefore does not include surface-water flooding and must not be described as an all-source flood measure.

## Satellite observations

[USGS Landsat 8–9 Collection 2 Level-2](https://doi.org/10.5066/P9OGBGM6) is accessed through the [Microsoft Planetary Computer STAC collection](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2). Candidate searches cover **June–August 2021–2025**. Regional quality screening selected five dates: **21 July 2021, 9 August 2022, 9 June 2023, 29 July 2024 and 17 August 2025**. These are five selected days with usable clear-sky observations, not a continuous climate series or complete summer record. Scene identifiers, candidate coverage and selection rules are retained in [landsat_selection.json](../metadata/landsat_selection.json).

Cached bands retain pixel-aligned study-area windows in their source UTM zone 30N CRS, **EPSG:32630**, and original digital numbers. [USGS product guidance](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature) supplies the temperature conversion: `kelvin = DN × 0.00341802 + 149.0`; Celsius subtracts 273.15. Cloud, shadow, snow, fill, saturation and cloud-distance information support quality screening. Missing emissivity information and retrieval artefacts can leave gaps or residual errors.

Thermal observations have a [native 100 m resolution and are distributed on a 30 m grid](https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites). The analysis grid is **100 m**. Resampling does not create finer thermal detail. These are clear-sky daytime **land-surface temperatures**, not air temperatures or personal heat exposure. The optical red and near-infrared bands support contextual NDVI.

## Green-space access

Regional screening uses corrected [Defra access-to-green-space statistics](https://www.gov.uk/government/statistics/access-to-green-space-in-england/access-to-green-space-in-england), published for 2024 and corrected in May 2025. **Scenario 5** measures access to green space of at least **10 ha within a 1 km walk**. OA records include `Household count`, `Household with access count` and Census 2021 geographic codes. LSOA rates are calculated as the sum of households with access divided by the sum of households, rather than an unweighted average of OA percentages. The workbook is about 35 MiB compressed but contains more than 1 GB of XML, so selected sheets are read as a stream.

These are official statistics in development. The corrected workbook is essential because Scenario 5 was revised. Defra's analysis uses household origins, a broader green-space inventory and a different network from this project's openly reproducible case study. The two access measures are comparable in distance and site-size thresholds, but their rates are not interchangeable.

The case study uses **OS Open Greenspace, April 2026**, with site polygons and entrance points from tiles SD, SE, SJ and SK. Downloads come directly from the [OS product API](https://api.os.uk/downloads/v1/products/OpenGreenspace); [OS documentation](https://docs.os.uk/os-downloads/products/land-and-terrain-portfolio/os-open-greenspace/os-open-greenspace-overview) describes the product. Tiles overlap at their edges, so site and entrance identifiers must be deduplicated. Eligibility is public parks and gardens of at least **10 ha**. The routing origin is the OA population-weighted centroid and aggregation uses Census OA population. The deficit includes origins beyond **1 km or with no mapped route** to an eligible entrance. Snapping an origin to the network does not establish route connectivity. A mapped entrance does not establish opening hours, unrestricted entry or step-free access.

The walking graph uses [OpenStreetMap](https://www.openstreetmap.org/copyright) ways and nodes retrieved through Overpass for Manchester–Salford and its configured buffer. The cached OSM base snapshot is **2026-09-29T16:11:18Z**, distinct from the retrieval timestamp. The saved query and source record accompany `walking_network.json.gz`. Access tags inform pedestrian filtering, but incomplete paths, barriers and access tags can affect routing. Network distances are modelled opportunities, not observed journeys or an accessibility audit.

## Reuse and temporal alignment

ONS, MHCLG, Defra, Environment Agency and OS inputs are supplied under the **Open Government Licence** with their recorded attribution. Landsat products are **USGS public domain**; the dataset citation and image credit are retained. OSM data are available under **ODbL 1.0** and require OpenStreetMap contributor attribution; the catalogue keeps that licence distinct from the project code licence. Attribution strings are recorded per row in the catalogue.

The sources intentionally describe different dates: Census 2021 population, a 2025 deprivation release with mixed underlying periods, corrected 2024 household-access statistics, selected summer satellite observations, May 2026 flood extents, April 2026 OS greenspace and a September 2026 OSM snapshot. Their combination is a neighbourhood screening dataset, not a synchronised observation of current individual circumstances. Source provenance and reproducible joins do not constitute field validation.
