# Analytical methodology

## Decision and study design

This is an independent, simulated consultancy commission: help a Greater Manchester adaptation team decide **where to commission more detailed neighbourhood investigations**. It is not an official GMCA product. The output is a transparent shortlist with evidence, alternatives and gaps. There is no outcome-labelled training dataset, intervention cost model or calibrated loss function, so a trained predictive model and claims about return on investment would be unjustified.

Regional screening covers the ten Greater Manchester boroughs: 1,702 Census 2021 LSOAs. LSOAs provide compatible public small-area population and income data without personal records. Manchester and Salford form a contiguous, cross-authority urban case study, containing 456 LSOAs and 2,466 OAs. Their shared urban network allows a more detailed investigation of access across administrative boundaries. A 2 km network buffer extends beyond both authorities; destinations across their borders are eligible.

All area and distance calculations use British National Grid (EPSG:27700). Original source representations remain in the raw cache. Boundary IDs are checked against an independent deprivation table; one-to-one joins reject missing or duplicate keys. Geometry validity, aggregate denominators and observed coverage are reported, not inferred from successful file reads. See the [source register](data_sources.md) for exact vintages and licences.

## Five distinct dimensions

Every model input is adverse-high. Rates retain their source denominator; no individual risk is inferred from an area's attributes.

| Dimension | Raw measure | Reason for inclusion |
| --- | --- | --- |
| River/sea flood footprint | Fraction of LSOA polygon area inside the present-day defended 1% river / 0.5% sea annual-probability extent | Physical hazard footprint; a separate 0.1% defended scenario tests footprint assumptions |
| Surface heat | Area-weighted mean of median selected-date surface-temperature anomalies | Relative clear-sky daytime surface heating across the study region |
| Large-greenspace access deficit | One minus official Scenario 5 household access rate | Household access to qualifying greenspace ≥10 ha within a 1 km walk |
| Income deprivation | IoD2025 Income Score, a rate | One social-resource dimension, avoiding repeated use of the full overlapping IMD composite |
| Age sensitivity | Census share aged under 5 or 65+ | An aggregate planning sensitivity proxy; age does not determine any individual's vulnerability |

Population totals support denominators and descriptive summaries. Population density, overall IMD and NDVI are not additional weighted dimensions. This avoids automatically giving an urban, deprivation or vegetation signal multiple votes. The indicator Spearman matrix exposes remaining relationships. Correlation does not prove that dimensions are interchangeable; removing each dimension and renormalising the remaining weights tests dependence on indicator choice.

The age-band choice is informed by [UKHSA heat-health guidance](https://www.gov.uk/guidance/heat-health-alert-action-card-for-health-and-social-care-providers), which identifies older adults and young children among relevant groups. The available Census bins operationalise this as under 5 and 65+. It is a coarse sensitivity dimension, not a clinical prediction, and omits other important personal and housing circumstances.

## Flood geometry

The Environment Agency OGC endpoint returns full intersecting features. A bounding-box response is not assumed to be clipped. Source-page counts, identifiers, checksums and next-page offsets are audited; the primary response contains 9,851 features and the sensitivity response 10,589 over two pages.

Each source geometry is reprojected, made valid where required, indexed and intersected with each LSOA. Local intersections are unioned before summing area, so overlapping source polygons cannot double-count land. The denominator is the LSOA's projected polygon area. Null source coverage is not silently interpreted as zero. A verified complete query permits zero for an area with no intersecting footprint.

The denominator includes any inland water retained in the ONS polygon; it is not a dry-land-only measurement. The two source scenarios are not perfectly nested: five LSOAs have a slightly lower fraction under the 0.1% extent, with the largest difference 1.236 percentage points in Trafford 020A. The sensitivity calculation preserves these source differences rather than forcing geometric containment.

These extents assume the represented defences operate as designed. They exclude defence failure, surface-water flooding and future climate allowances. The annual-probability label describes the scenario, not an expected frequency of flooding at every point. Land fraction is neither a count of flooded properties nor the share of residents exposed. In particular, no uniform-population allocation into flood polygons is performed.

## Satellite selection and thermal processing

The configured search covers June–August 2021–2025, Landsat Collection 2 Level 2, WRS path/row 203/023. Up to five lowest scene-cloud candidates per year are examined, subject to scene cloud cover ≤70%. This deterministic candidate cap is an engineering sampling decision, not an exhaustive survey of every usable observation. No dates are selected using their resulting temperature or shortlist ranks.

Per-pixel QA removes fill, dilated cloud, cirrus, cloud, shadow, snow, water and radiometric saturation. Cloud distance must be at least 0.3 km. Candidate dates require at least 60% clear study-region land; accepted dates are ordered by coverage, with at most two per year and eight total. The accepted snapshot contains five dates: **21 July 2021, 9 August 2022, 9 June 2023, 29 July 2024 and 17 August 2025**. These are selected observations, not a continuous five-year series or climatic average.

Only pixel-aligned regional windows are downloaded; original digital numbers and native UTM 30N grids are preserved. Temperature is `DN × 0.00341802 + 149 − 273.15` degrees Celsius, with DN 0 excluded. Reflectance uses the USGS scale and offset before `NDVI = (NIR − red)/(NIR + red)`; invalid reflectances are excluded. Thermal data distributed on a 30 m grid originate from approximately 100 m thermal measurements, so the analytical output is 100 m.

Masked continuous values are area-averaged into a common BNG grid. At least 80% of each target cell's source area must be valid on a date. Each date is centred on its median across **the same land cells observed on all accepted dates**, preventing shifting cloud masks from changing the comparison reference. That common reference covers 31.0% of regional land-grid cells and must exceed a configured 25% minimum. Its location and clear-sky selection may still bias the reference.

The temporal median of these anomalies is retained only where at least three observations across at least two summers exist. Five dates currently contribute one observation per summer. LSOA values use exact fractional intersections of raster cells with each polygon, rather than cell-centre or unweighted touched-cell assignment. The eligible observed area must cover at least 80% of an LSOA; otherwise heat and the composite are withheld. NDVI is contextual and may use fewer dates; its coverage is separate. Source QA and observation-count rasters remain inspectable.

The 60% date acceptance and 25% shared-reference thresholds were settled during coverage feasibility, before composite ranking. Earlier, more restrictive date screening left large regional gaps. Pixel-level QA, minimum observations and the LSOA coverage gate were retained. These choices should be tested in a follow-on thermal study; weight stability does not validate them.

## Two access measures with different denominators

For the regional composite, official Defra Scenario 5 OA numerators and denominators are summed within LSOAs. This is a household-weighted rate, not an average of OA percentages. The corrected May 2025 workbook is used.

The detailed case study uses a narrower, openly reproducible destination definition: OS Open Greenspace public parks and gardens ≥10 ha with pedestrian access points. Whole-site area determines eligibility before study clipping, with duplicate site and entrance IDs removed across OS tiles. A mapped site does not establish quality, unrestricted opening or accessible facilities.

OSM ways become directed graph edges with metric segment lengths. Shared node identifiers determine junctions; geometrically crossing bridges do not create false connections. Motorways, trunks, prohibited/private walking links and selected nonwalkable classes are excluded. Explicit pedestrian permissions and `oneway:foot` are respected. Explicitly locked or prohibited nodes are blocked. Ordinary car one-way restrictions are not assumed to restrict walking.

OA population-weighted centroids connect to their nearest eligible edge within 150 m; park entrances connect within 100 m. Exact projected positions on edges preserve travel along each partial segment. Off-network connector length is included, and same-edge trips use the direct segment where permitted. A reversed graph and one multi-source Dijkstra pass find the nearest destination, rather than routing every origin–destination pair separately.

A disconnected but successfully snapped origin is *unreachable in the model* and contributes to distance-threshold deficit; an origin outside snapping tolerance is unknown and excluded from the denominator. The latter must not be coded as either zero distance or inaccessible. Each LSOA requires ≥80% known resident coverage. Thresholds of 500, 1,000 and 1,500 m test distance sensitivity. The nominal 1 km threshold is about 15 minutes at 4 km/h, but no speed, slope, crossing-delay or mobility-specific travel-time model is claimed.

There are 821,878 residents in the OA routing table versus 821,867 in the case LSOA table. Separately published Census tables can differ because of disclosure control; each analysis retains its own denominator. The official household measure uses different origins, destination types, network and vintage. Differences between the two rates are not validation errors or a direct change over time.

## Normalisation and policy scenarios

Only LSOAs with all five finite indicators enter **one common comparison cohort**. Missing values remain missing: there is no imputation, zero substitution or per-row reweighting. Unassessed neighbourhoods remain visible on maps and in an evidence-gap table.

Each raw dimension becomes `100 × (midrank − minimum midrank)/(maximum midrank − minimum midrank)`. Ties receive the same average rank; a constant dimension is 50 and cannot differentiate areas. This limits outlier influence and respects the strong pile-up at zero flood footprint, while losing information about the magnitude of differences. Raw min–max normalisation is a reported alternative. Scores are relative within this eligible region and cannot be compared directly with another region or rerun cohort.

| Scenario | Flood | Heat | Access deficit | Income | Age |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced reference | 20% | 20% | 20% | 20% | 20% |
| Hazard emphasis | 30% | 30% | 10% | 15% | 15% |
| Equity emphasis | 15% | 15% | 15% | 30% | 25% |

Scores are weighted sums, rounded to 12 decimals to prevent floating-point artefacts splitting exact ties. Descending ranks use average ties. The priority group is the empirical upper quintile (`score ≥ 80th percentile`), including boundary ties. It can exceed exactly 20% and is a screening workload convention, not an externally validated risk threshold.

These are explicit simulated policy preferences, not fitted importance coefficients or stakeholder-approved weights. Additivity permits compensation between dimensions. The mutually exclusive `priority_class` assigns single-hazard review to assessed areas outside the balanced shortlist when their normalised flood or heat dimension is ≥95. This keeps extreme component evidence visible outside that shortlist. Baseline and persistent-priority categories take precedence. The most-adverse-dimension map describes the largest relative dimension; it does not compare physical units across hazards.

## Robustness and spatial association

With seed **20260929**, 2,000 Dirichlet weight vectors are sampled around the three scenario centres in counts 667/667/666, concentration 50. For each vector the same cohort is scored and the inclusive top quintile recalculated. Save each area's inclusion frequency and empirical 5th–95th rank range. These describe this weight experiment; they are neither posterior risk probabilities nor statistical confidence intervals.

`Persistent priority` requires both top-quintile membership in all three named scenarios and inclusion in at least 80% of sampled weight configurations. Other baseline top-quintile areas are scenario-sensitive. Pairwise rank correlations and top-group Jaccard overlaps expose disagreement. Additional comparisons use raw min–max normalisation, leave-one-dimension-out scores and the defended 0.1% flood extent, always on the same eligible cohort. High weight stability cannot compensate for source uncertainty.

One predeclared **Global Moran's I** evaluates spatial association of the balanced score. Primary weights are exact queen contiguity, row-standardised; rook shared-edge contiguity is a sensitivity check. Only the eligible cohort participates. Islands keep zero rows and remain in the permutation population; no nearest-neighbour edges are invented to repair missing heat evidence.

For centred scores `z`, `I = (n/S0)(zᵀWz)/(zᵀz)`, with expectation `−1/(n−1)` under the fixed-score label-permutation null. The two-sided pseudo-p value uses 9,999 uniformly shuffled assignments and the plus-one correction, counting absolute deviations from that expectation. This tests spatial randomness conditional on the chosen units, weights and cohort. It does not explain causes. No local significance tests are conducted: HH/LL/HL/LH quadrants are descriptive, so no local multiple-testing or hotspot-significance claim is made.

## Interpretation and reproducibility

A useful deliverable combines the shortlist, component values, stability, evidence gaps and recommended **next investigations**. Any implementation decision needs local engagement, flood/overheating expertise, access audits, feasibility and cost appraisal. The analysis does not estimate harm avoided or an optimal allocation of money.

Methods follow the transparent construction and sensitivity principles of the [OECD/JRC composite-indicator handbook](https://www.oecd.org/en/publications/handbook-on-constructing-composite-indicators-methodology-and-user-guide_9789264043466-en.html); the exact implemented contracts and tests are the authority for this project. Read [limitations](limitations.md), [reproduction](reproduction.md), and `outputs/qa/model.json`, `spatial.json`, `flood.json`, `landsat.json` and `network.json` alongside the maps.
