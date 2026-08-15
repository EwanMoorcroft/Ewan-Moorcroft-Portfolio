# Liverpool Urban Accessibility

An evidence-led spatial analysis of how Census 2021 workplace flows connect Liverpool's 61 Middle
layer Super Output Areas (MSOAs).

![Liverpool local workplace-flow retention](assets/local-retention-map.png)

*Share of fixed-workplace flows from each MSOA whose recorded workplace was within Liverpool,
including the origin MSOA.
Source and interpretation details are in [the figure captions](assets/CAPTIONS.md).*

## What this demonstrates

- ingestion of a 197.7 MB national origin–destination table with DuckDB;
- strict geography, geometry, CRS, count, and flow-conservation contracts;
- population-weighted centroids and metric distance calculations in British National Grid;
- gravity-style accessibility at predeclared 3, 5, and 10 km decay distances;
- Queen-contiguity spatial weights and a transparent Moran's I implementation;
- exposure-offset Poisson and overdispersion-aware NB2 count models, plus a bounded binomial
  sensitivity on identical observations;
- independent numerical validation in base R;
- retained source identities, file hashes, figures, tests, a CLI, and non-root Docker packaging.

## Retained public-data findings

**Fresh public-data analysis:** run on the completed Census 2021 ODWP01EW release, official 2021
ONS boundaries, and official population-weighted centroids. The raw national flow table is excluded;
compact derived evidence, source identities, and every published aggregate are retained.

| Finding | Retained value |
|---|---:|
| Liverpool MSOAs | 61 |
| Employed or temporarily absent population represented by indicators 1–3 | 205,768 |
| Fixed-workplace flows | 123,689 |
| Fixed-workplace flows staying within Liverpool | 84,567 (68.37%) |
| Combined working-at-home or no-fixed-place category | 81,158 (39.44% of indicators 1–3) |
| Moran's I for area local-retention share | 0.4901 |
| Two-sided permutation p-value | 0.0001 from 9,999 fixed-seed permutations |
| Poisson Pearson dispersion | 4.7468 |
| Poisson / NB2 AIC on the same 61 rows | 836.18 / 711.85 |
| Python-to-R maximum Poisson coefficient difference | `1.05e-12` |

The combined indicator is not interpreted as a home-working rate: ONS groups working mainly at or
from home with having no fixed place of work. The model is descriptive rather than predictive or
causal. Its destination-flow mass is derived from the same release and includes each origin's own
recorded flows, so the association is mechanically coupled rather than independent predictive
evidence. The 2021 Census was also collected during pandemic disruption.

## Reproduce the retained checks

Python 3.11 or newer is required.

```bash
uv sync --locked --extra test
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run liverpool-access verify --evidence-dir reports/retained
```

`requirements.txt` and `requirements-dev.txt` are exact-pin exports for environments that use pip;
`uv.lock` is the canonical cross-platform dependency record.

The verifier checks eight retained file hashes, reconstructs 22 numerical aggregates, refits all
three specifications, reruns 9,999 spatial permutations, and compares 50 coefficient values.

The full pipeline has a deterministic, network-free 5 × 5 fictional fixture:

```bash
uv run liverpool-access fixture --output-dir reports/generated/fixture
```

R is deliberately limited to one transparent cross-language check:

```bash
Rscript r/validate_results.R reports/retained reports/generated/r-validation.json
```

## Rebuild from official inputs

Download ODWP01EW from the official Nomis release page and extract only `ODWP01EW_MSOA.csv` outside
the repository. Download the two Liverpool GeoJSON queries recorded in
[`data/source-spec.json`](data/source-spec.json), then run:

```bash
liverpool-access prepare \
  --flow-csv /path/to/ODWP01EW_MSOA.csv \
  --boundaries /path/to/liverpool-boundaries.geojson \
  --centroids /path/to/liverpool-centroids.geojson \
  --source-manifest reports/retained/source-manifest.json \
  --output-dir reports/generated/rebuild
liverpool-access analyse --evidence-dir reports/generated/rebuild
liverpool-access report \
  --evidence-dir reports/generated/rebuild \
  --output-dir reports/generated/figures
```

The source manifest is an explicit trust boundary: it records URLs, sizes, SHA-256 values, retrieval
time, licences, and transformations. Preparation rejects mismatched CRS, codes, geometry, headers,
negative counts, missing areas, and spatial islands.

## Container smoke path

```bash
docker build -t liverpool-access:local .
docker run --rm liverpool-access:local --help
docker run --rm -v "$PWD:/workspace" liverpool-access:local \
  verify --evidence-dir reports/retained
```

The runtime image uses a dedicated non-root user. Public-data download is never part of an image
build or CI run.

## Project map

| Path | Purpose |
|---|---|
| `src/liverpool_accessibility/` | contracts, DuckDB transformation, spatial analysis, count models, evidence verification and CLI |
| `reports/retained/` | compact real-data evidence, source identity, hashes and R parity result |
| `assets/` | figures derived from retained evidence |
| `tests/` | deterministic contract, spatial, model and full-pipeline checks |
| `r/` | independent base-R coefficient and Moran reconstruction |
| `data/` | public source specification and redistribution boundary |
| `docs/` | methods, findings and interpretation limits |

## Interpretation boundary

This analysis measures recorded workplace-flow patterns, not individual accessibility, realised
journey quality, or present-day transport demand. It is affected by the modifiable areal unit
problem, ecological inference, centroid approximation, cell-key perturbation, missing cross-border
origins, and the exceptional Census date. Read the complete [limitations](docs/limitations.md)
before using the results.

Software is MIT licensed. Census and boundary data remain under the Open Government Licence v3.0;
full attribution and source identities are in [PROVENANCE.md](PROVENANCE.md).
