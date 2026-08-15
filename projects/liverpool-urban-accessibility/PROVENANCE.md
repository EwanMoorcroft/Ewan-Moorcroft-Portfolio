# Provenance

## Study scope

The retained analysis covers Liverpool local authority (`E08000012`) through its 61 Census 2021
MSOAs. Every source uses the 2021 geography identity; no statistical crosswalk is applied.

## Workplace flows

- Publisher: Office for National Statistics via Nomis
- Product: Census 2021 ODWP01EW origin-destination workplace flows
- Official release: [Nomis Census 2021 origin-destination data](https://www.nomisweb.co.uk/sources/census_2021_od)
- Archive: [ODWP01EW](https://www.nomisweb.co.uk/output/census/2021/odwp01ew.zip)
- Archive SHA-256: `9e32ababfd9f77e353411d399e463f812942440df115a2d6c296c74dbeea70d7`
- MSOA member SHA-256: `8af475023e18227fdcee3ac4a547d6549b1fcb87138bd93d177a7a698d1a10dd`
- Licence: [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)

Indicator `3` supplies fixed UK workplace flows. Indicators `1` and `2` are retained separately in
area summaries. Indicator `-8` is outside the employed denominator used here. Sparse published edges
are never silently expanded into a complete zero matrix.

## Geography

- Product: MSOA December 2021 boundaries EW BGC V3
- Catalogue: [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/datasets/6b282db29762450881ed5159259a6e4e/about)
- Retained query response SHA-256: `2fed47c4278747fcde3e5cf2008442a0572da8712499d2e92e61c62fa8dc6951`
- Product: MSOA December 2021 population-weighted centroids EW PWC V2
- Retained query response SHA-256: `18d271e247a385f5edbf4f9a948ce23e504d204bb7d226193e5c789b1ca3448d`
- Query output CRS: EPSG:4326; distance calculation CRS: EPSG:27700

Required attribution:

> Source: Office for National Statistics licensed under the Open Government Licence v.3.0.
> Contains OS data © Crown copyright and database right 2026.

The repository retains the small licensed Liverpool boundary and centroid subsets because they are
needed to reconstruct the spatial graph, distances, and map. The 197.7 MB national flow table remains
outside Git.

## Evidence identity

[`reports/retained/source-manifest.json`](reports/retained/source-manifest.json) records the exact
retrieval time, byte sizes, URLs, source hashes and transformations. The separate results manifest
hashes every canonical input and output used by the verifier, including the independent R result.
