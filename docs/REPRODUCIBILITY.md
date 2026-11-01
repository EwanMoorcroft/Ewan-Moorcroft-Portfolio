# Reproducibility

## Project environments

The six projects in the shared quality workflow each have a `pyproject.toml` and `uv.lock` and can
be installed independently. Python 3.11 is their shared baseline. The [Greater Manchester climate
resilience case](../projects/uk-climate-resilience-gis/) instead uses QGIS 4.2.2 and its Python 3.12
runtime; its [reproduction guide](../projects/uk-climate-resilience-gis/docs/reproduction.md) documents
that separate environment and the source-data workflow.

For the six lockfile-based projects, a typical setup with `uv` is:

```bash
cd projects/<project-name>
uv sync --locked --extra test
uv run --no-sync pytest
```

Each project README covers any additional training or plotting packages. The FPL chart builder uses
the optional `analysis` extra. A standard virtual
environment and `pip` can be used instead, although installing directly from `pyproject.toml`
resolves a new set of versions. The lockfiles specify the tested dependency resolution for those six projects. Their CI jobs
build wheels, check installed dependencies and run command-line entry points from outside the
source folder. FPL and Liverpool have container workflow checks. The Greater Manchester case is
not in that CI matrix; its recorded local QGIS tests and real-data checks are described in its
[validation record](../projects/uk-climate-resilience-gis/docs/validation.md).

## Compute

- Unit tests run on CPU and do not require the source datasets.
- PyTorch projects select Apple Metal acceleration when available and fall back to CPU.
- CI uses small synthetic checks, not full model training or source-dataset downloads.
- Local defaults favour low worker counts and modest batch sizes for an 8 GB machine.
- Short smoke runs check that the software works; they are not model results.

## Data and outputs

| Project | Included in Git | Outside ordinary Git |
|---|---|---|
| Greater Manchester climate resilience | Code, selected maps and figures, decision tables, provenance and local validation records | Raw and intermediate caches stay local; processed data and the complete QGIS handover are provided in a separate release download |
| Tree LiDAR | Result tables, route details, evaluator and figures | Raw point clouds and prediction arrays |
| Liverpool urban accessibility | Derived area metrics, boundaries, centroids, spatial edges, models and figures | National Census workplace-flow table |
| Chest X-ray | Dataset specification, duplicate analysis, evaluation outputs and preparation code | 3,475 source images and checkpoints |
| FPL | Forecasting code, aggregate model comparison and source details | Downloaded gameweek snapshots, row predictions and fitted models |
| LunarLander | Figures and evaluation summary | Model checkpoint and new run outputs |
| Neural chunking | Model code, figures and evaluation summary | Source corpus, checkpoints and new run outputs |

The X-ray corpus is the CC BY 4.0 [Chest X-Ray V1 dataset](https://data.mendeley.com/datasets/p5rm59k7ph/1),
DOI `10.17632/p5rm59k7ph.1`.

## Determinism

Projects set Python, NumPy and PyTorch seeds where relevant and record configuration with outputs.
Hardware and dependency changes can still affect floating-point training trajectories; the project
limitations note where that matters.

## Safe persistence

The forecasting project saves coefficients and preprocessing values as JSON rather than executable
pickle data. PyTorch projects write state dictionaries and metadata for locally produced
checkpoints. Only trusted local checkpoints should be loaded.
