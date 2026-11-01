# Validation and review record

The delivered snapshot was checked on **29–30 September 2026** using the installed macOS arm64 QGIS 4.2.2/Python 3.12.11 runtime. The full automated validation command exits successfully: **267/267 real-data checks passed**. The final unit/algorithm run passed **77 tests**. These are integrity and implementation checks, not independent validation of climate impacts or intervention benefits.

## Evidence and commands

| Evidence | Result | Record |
| --- | --- | --- |
| Critical transformation and algorithm tests | 77 passed; includes actual QGIS Processing execution | `outputs/qa/unit_tests.json`, `tests/` |
| Real-data contracts and source hashes | 267 passed; 141 source files verified | `outputs/qa/validation.json`, `metadata/provenance_manifest.json` |
| QGIS project | 21 valid layers, six layouts and six themes; eight saved QML styles | `metadata/qgis_build.json` |
| Relocation | All sources resolve in the copied hierarchy; all six layouts match pixel-for-pixel at 72 dpi | `metadata/qgis_relocation.json` |
| Map exports | Six A3 PNG/PDF pairs; PDF page size, embedded Arial fonts and every saved-layout text label checked | `metadata/qgis_pdf_qa.json` |
| Presentation review | All six maps, six PDF renders and four analytical figures inspected | PDF QA record and this document |
| Interactive explorer | Search, borough/measure filters, evidence details, missingness, no-results state and keyboard controls checked in a browser | `outputs/qa/browser_review.json` |

The documented `make setup`, `make download`, `make process`, `make analyse`, `make qgis`, `make export`, `make test` and `make validate` commands were exercised. Acquisition was also rerun against its retained cache. Stage completion times and durations are in `metadata/stages/`; logs are local in `outputs/logs/`. Representative cached runtimes on this machine were 273 seconds for processing, 10 seconds for analysis, 14 seconds for QGIS build, 38 seconds for export and 12 seconds for validation. These are observations, not performance guarantees.

## What was checked

Data checks cover unique Census geography, complete joins, CRS and plausible extent, valid nonempty geometries, required layer contents, indicator ranges and explicit missingness. The 1,702 LSOAs reconcile to 2,867,735 residents. The 2,466 case-study OA origins reconcile to their own published total of 821,878; the separate LSOA total differs by 11 people and is not forced to agree.

Saved flood geometries reproduce the reported polygon-area fractions for both scenarios. Flood source pagination, IDs and counts are audited, including the second page of the 0.1% response. Raster checks cover common 100 m grid, EPSG:27700, nodata, finite values, observation thresholds and bounded NDVI. Source QA additionally records acquisition/mask coverage. Core checks explicitly retain 174 heat-evidence gaps and exclude them from composite ranking.

The passing validation report fingerprints the processed data, decision tables, analytical figures and offline explorer. The review archive builder checks those hashes again, alongside the independently reviewed QGIS and map hashes, and refuses to package files changed since validation.

The balanced top-group shortlist contains 306 neighbourhoods: 153 meet the persistence rule and 153 are scenario-sensitive. The separate evidence-gap table contains 174 neighbourhoods.

Every scenario score, rank, inclusion frequency and rank interval is reproduced with the configured seed. Independent weighted sums and adverse-high ordering are checked. CSV, GeoParquet and GeoPackage values agree on keyed records; configuration, input indicators and thermal raster match the recorded analysis contract. The reusable QGIS algorithm preserves all 1,702 rows and independently reproduces the 1,528 eligible balanced scores.

Walking distances exceed the straight-line lower bound. Accessibility fractions at 500/1,000/1,500 m are independently reconstructed from OA population and route status. All origins snap within tolerance; 54 remain disconnected and are not confused with unsnapped/unknown origins. Hand-checkable network tests exercise detours, direction, connectors and same-edge trips. Spatial-statistical tests use tiny independent examples, degenerate cases and seeded permutation behaviour.

## Visual and portability review

All six maps were reviewed for complete extents, readable legends and notes, unclipped titles, label collisions, units, scales, contrast, source attribution and visible missing evidence. PDF pages were independently rendered, rather than assuming successful export guaranteed the same content as PNG. That review caught stale PDF text on a reused export target; exports now use fresh temporary PDF paths and atomic replacement. Map 02 correctly displays 42.7 km²; Map 04 displays 63%, qualifying parks and the 54 disconnected origins.

Relocation checks include both ordinary layer URIs and layout fallback references. They copy the actual project, GeoPackage and three referenced rasters, verify hashes and compare original/copied renders. They demonstrate portability of the delivered hierarchy on the tested runtime, not compatibility with every QGIS version or OS.

All four figure pairs were checked for legible labels and agreement with the numerical diagnostics. Browser inspection confirmed the narrower stacked layout, fixed scales, correct neighbourhood values and explicit unassessed states. Keyboard zoom and fit controls worked. Pointer, drag and touch interaction and other browsers were not certified. There were no captured browser warnings or errors during the checks.

## Resolved engineering issues and remaining limits

The build fixes a reproduced GeoPackage journal-mode conflict with read-only QGIS providers and a process-local `OGR_SQLITE_JOURNAL=DELETE` setting. Rendering leaves the analytical package hash unchanged. A validation shutdown fault was traced to QGIS object lifetime; validation now releases local project/provider wrappers before shutting down QGIS, and the CLI completes with exit code zero.

No field survey, independent thermal ground truth, property-level flood exposure, predictive outcome test or intervention appraisal was performed. Weight sensitivity is conditional on the indicator definitions, complete cohort and normalisation. Source vintages differ; surface-water flooding is omitted; mapped network access may differ from actual access. Read [limitations.md](limitations.md) before using the shortlist. Publishing/deployment and testing on other operating systems are outside the delivered local verification.

The review archive includes a per-member SHA-256 manifest and is checked for CRC and source/member hash agreement when built by `scripts/package_review.py`. Its processed data support immediate viewing. Full scientific revalidation needs the retained raw/intermediate cache or a fresh acquisition and processing run.
