# QGIS deliverable and engineering

Open `qgis/project/gm_climate_resilience.qgz` with the accompanying `data/processed/` directory in its original relative location. The project is built and tested with QGIS 4.2.2 on macOS arm64. It contains projected analytical layers and saved layouts; no network basemap, credentials or online service is needed to view the delivered project.

## Layer organisation

Four groups separate screening maps, source indicators/evidence, the Manchester–Salford walking case, and administrative context. The full neighbourhood attribute table retains raw indicators, coverage, scenario results, uncertainty and investigation themes. Layer clones give each analytical view its own renderer without duplicating the underlying GeoPackage table.

Evidence includes the primary and alternative flood extents, selected-date heat and vegetation rasters, observation coverage, qualifying public parks, entrances and OA origins. The primary flood view filters `hazard='defended_1pct'`; the separate 0.1% sensitivity footprint stays available for inspection. The case access map displays only qualifying public parks ≥10 ha so destination symbols agree with the routing definition.

Aliases and metadata explain units and denominator distinctions. Neighbourhood map tips show local evidence. Rule-based expressions handle most-adverse dimensions, tied maxima, missing values, stable-priority outlines and route status. Explicit classes/palettes support the analytic units, with neutral missing-evidence fills. Borough labels, scale bars, a regional case-location inset, source text and analytical notes follow one restrained A3 design. QML files in `qgis/styles/` make the styles independently inspectable.

## Six saved layouts and map themes

| Layout/file stem | Analytical story |
| --- | --- |
| `01_overview` | Which relative dimension is most adverse in each assessed neighbourhood? |
| `02_flood` | How much neighbourhood area intersects the declared defended flood footprint? |
| `03_heat` | Where are selected summer surfaces warmer or cooler relative to the shared reference? |
| `04_accessibility` | Where do modelled residents face long or unavailable mapped routes to qualifying parks? |
| `05_priorities` | Which areas score highly in the balanced scenario, and which also meet the weight-persistence rule? |
| `06_robustness` | How frequently does each area enter the top group under the sampled weights? |

All six are A3 landscape. PNGs are exported at 180 dpi (2,976 × 2,104 pixels); PDFs retain vector geometry/text where supported and use 300 dpi settings for raster content. The flood PDF is larger because it preserves detailed footprint geometry. The layouts remain editable in QGIS. The master portfolio recommendation is Map 05, with Map 04 and the sensitivity figure beside it.

## Rebuild and export

```sh
make qgis
make export
```

The focused command is:

```sh
./scripts/run_python.sh scripts/qgis/build_project.py --build --export
```

The builder creates the project, layers, expressions, styles, metadata, themes and layouts programmatically. Rule expressions are prepared and checked against real features before export. Fonts are explicitly selected; the build report records the resolved font and QGIS version. Layer validity alone is insufficient: unreadable/negative feature counts are rejected.

Repeated-build QA identified a QGIS/GDAL GeoPackage journaling interaction. Provider read-only options and a process-local `OGR_SQLITE_JOURNAL=DELETE` setting prevent a filter operation from changing the database mode while other providers retain connections. The builder records input hashes to check that rendering has not changed the analytical package.

PDFs are written to a fresh temporary destination and atomically replaced to avoid a reproduced stale-content issue on repeated exports. `scripts/validation/check_qgis_exports.py` independently checks the current PDF labels/fonts/page size and renders previews with Poppler. Visual review is retained only for identical decoded export pixels; changed layouts require fresh inspection.

QGIS writes some layout fallback source references as absolute paths even when ordinary layer data sources are relative. The builder normalises those saved layout references too. Its relocation test copies the project and referenced data to a fresh directory, validates hashes, checks that every source resolves inside the copied hierarchy, and compares all six original/copied layout renders at 72 dpi. Reports live in `metadata/qgis_build.json` and `metadata/qgis_relocation.json`.

## Reusable Processing algorithm

`scripts/qgis/resilience_processing.py` implements **Weighted neighbourhood scores (five dimensions)**. Registered programmatically, its ID is `resilience:weightedneighbourhoodscores`. It accepts five distinct, already-normalised adverse-high numeric fields and a five-number weight vector. Positive weights are normalised by their sum; an invalid active input produces a NULL score and named issue. It preserves feature count, attributes, geometry and CRS and adds `weighted_score`, `score_valid`, and `score_issue`.

This algorithm is a reusable scoring component, not the complete scientific pipeline. It does not select the comparison cohort, normalise raw data, compute ranks or perform the weight experiment. For the delivered model, use the `dim_*` fields rather than raw percentages/degrees. `make validate` runs it over all real neighbourhood rows and checks its balanced scores against the Python model.

No manual QGIS operation is required to rebuild the project or exports. To experiment with the reusable algorithm in the GUI, optionally add this file through the Processing Toolbox's Scripts menu. That optional profile installation is not performed automatically. To inspect layouts or change presentation interactively, use QGIS's Layout Manager and map themes; changing the scientific parameters belongs in `config/analysis.json` and the reproducible pipeline.
