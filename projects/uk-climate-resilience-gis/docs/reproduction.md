# Reproduce and inspect the project

## Verified environment

The delivered build uses macOS 27 arm64, Python 3.12.11 and QGIS **4.2.2**, with Qt 6.11.1, GDAL 3.13.3 and PROJ 9.8.1. Exact observed package versions are in [environment/detected.json](../environment/detected.json). Its `verification` section records initial environment smoke tests; final real-data checks are separate in `outputs/qa/validation.json` and `metadata/qgis_build.json`.

QGIS's bundled Python already supplied the required libraries. No global package installation was needed. `scripts/run_python.sh` discovers a usable QGIS application in `/Applications` or `~/Applications`, isolates the profile/cache under `work/`, and removes conflicting inherited Conda GDAL/PROJ settings. It respects an explicit `QGIS_APP` override. Arial is requested for layouts; the actual resolved font is recorded.

The launcher also fixes `OGR_SQLITE_JOURNAL=DELETE` for its own processes. In this QGIS/GDAL build, changing subset filters otherwise switched the GeoPackage to WAL while earlier providers held connections, producing unreliable feature counts. The setting avoids that observed build failure without changing the user's global environment or analytical records. The builder rejects negative provider counts and verifies copied-layout renders.

## Commands

Run from the project root:

```sh
make setup
make download
make process
make analyse
make qgis
make export
make test
make validate
```

`make all` runs setup, download, process, analyse, QGIS build, export and validation in that order. It does not run the unit tests; use `make test` as above. The equivalent stage interface is:

```sh
./scripts/run_python.sh -m resilience setup
./scripts/run_python.sh -m resilience download
./scripts/run_python.sh -m resilience process
./scripts/run_python.sh -m resilience analyse
./scripts/run_python.sh -m resilience build-qgis
./scripts/run_python.sh -m resilience export
./scripts/run_python.sh -m resilience validate
```

To select an installed macOS bundle explicitly:

```sh
QGIS_APP="/Applications/QGIS-final-4_2_2.app" make setup
```

For Linux/another environment, install a coherent QGIS/PyQGIS, GDAL, Rasterio and GeoPandas stack through your platform's QGIS or conda-forge distribution. Set `QGIS_PYTHON` to its Python executable, with QGIS plugin paths configured. The launcher preserves those external settings on nonmacOS hosts. This route and QGIS 3.44 compatibility have **not** been execution-tested. Do not install a second GDAL into a working bundled QGIS environment just to match an individual pip version.

## Caches and source changes

The first download requires public internet access and no account. Raw files and per-file source sidecars are retained in `data/raw/`, with checksums, queries and source dates. Cached acquisition and the downstream analysis use these snapshots. The large national ODS is parsed as streamed XML; flood responses are consumed page by page; Landsat imagery is downloaded through study-area windows.

`download` also prepares the core boundary/statistical join required to define requests. `process` repeats that deterministic join and produces flood, raster and network indicators. `analyse` computes policy scenarios, sensitivity, spatial statistics, GeoPackage and decision tables. `build-qgis` produces the relative-path project and styles; `export` renders layouts and creates figures plus the standalone HTML. `validate` checks the final delivered contracts and regenerates the source manifest.

Existing data are deliberately reused rather than silently refreshed. The OS release is pinned and a changed release causes a reviewable failure. Keep a copy of the complete raw snapshot for exact historical reruns; upstream APIs may stop serving a past release. To update sources or change the study region, archive the previous raw/interim/processed outputs and start a separate versioned build. Do not mix an old cache with new spatial assumptions. This is a configured Greater Manchester study, not a general-purpose nationwide downloader.

The analysis configuration lives in [config/analysis.json](../config/analysis.json). After changing analytic settings rerun the dependent processing, analysis, project and exports. The analysis contract hashes configuration and heat input, and validation rejects a stale configuration/raster. Keep raw-file SHA-256 records: stage success alone does not identify the source snapshot.

Logs are in `outputs/logs/`. `metadata/stages/` records measured stage durations. Runtime varies by hardware, cache state, service latency and competing work. The source flood responses are approximately 604 MB in total and a single ODS expands to over 1 GB of XML; full in-memory spreadsheet loading is unnecessary. Keep several GB of free space. The graph includes about 673,000 nodes and 1.48 million directed edges, so graph construction/routing is a material memory stage.

## Open and share

Open **`qgis/project/gm_climate_resilience.qgz`** in QGIS. Keep `qgis/` and `data/processed/` together in their current relative hierarchy. The prebuilt project needs no manual layer relinking or geoprocessing. Six layouts appear in Project → Layouts; their PNG/PDF exports are already in `outputs/maps/`.

Open `outputs/portfolio/interactive.html` in a modern browser. It embeds geometry and values and requires no API key, server, CDN or hosted basemap. Geometry is simplified for display; analytical intersections always use the unsimplified source geometry.

Raw/interim/processed caches and release bundles are ignored by Git. A repository clone reconstructs data through the commands above; it will not contain a working GeoPackage just because the `.qgz` is present. For review without acquisition, [download the complete review ZIP](https://github.com/EwanMoorcroft/Ewan-Moorcroft-Portfolio/releases/download/gm-climate-resilience-v1.0.0/gm_climate_resilience_review.zip) and unzip it, retaining its hierarchy and attribution. The local build creates the same file at `outputs/release/gm_climate_resilience_review.zip`. It includes source code, processed data, maps, figures, tables, documentation and `SHA256SUMS.txt`. It omits raw and intermediate caches, which must be reacquired/reprocessed for full scientific revalidation. On Macs using Desktop cloud optimisation, first keep the extracted review folder downloaded locally. A full processing or validation rerun also needs local copies of the raw and intermediate caches; cloud-only placeholders can block these commands. Do not commit the raw cache or archive to the parent portfolio repository.

After a passing validation run, recreate and integrity-check that archive with:

```sh
./scripts/run_python.sh scripts/package_review.py
```

Independent PDF checks use the existing Poppler tools (`pdfinfo`, `pdftotext`, `pdffonts`, `pdftoppm`) and Pillow:

```sh
./scripts/run_python.sh scripts/validation/check_qgis_exports.py
```

This checks current layout labels, page size and embedded fonts, renders previews under `work/qgis-qa/pdf/`, and compares export pixels with the recorded visual review. Identical pixels preserve that review. Changed pixels require inspecting all six PNGs and PDF previews; only after doing so use `--record-visual-review`. The flag records a completed inspection and does not perform one. Packaging refuses a pending visual review or changed project/export hashes. If Poppler is unavailable, the script reports the missing tool; the core QGIS project still opens normally.

## Failure handling

- Source pagination/count mismatch: stop, retain responses and inspect `hazards.py`/`acquire.py` diagnostics. Never accept the first page as the full source.
- Missing/changed schema or unmatched geography: inspect the source register and exact failed fields before revising an adapter.
- Heat coverage failure: preserve the gaps; review acquisition/QA/selection, not the ranking target.
- QGIS import or projection errors: confirm the selected runtime and avoid mixing system/Conda/QGIS libraries.
- Unexpected output or QA failure: inspect the failed check and raw provenance. Do not label the build validated until both analytical checks and visual review are complete.
