# Reproducibility

## Project environments

Each project has its own `pyproject.toml` and can be installed independently. A geospatial analysis
should not need the same environment as a reinforcement-learning run or an image classifier. Python
3.11 is the shared baseline.

With `uv` installed, a typical project setup is:

```bash
cd projects/<project-name>
uv sync --extra test
uv run pytest
```

Each project README covers any additional training or plotting packages. A standard virtual
environment and `pip` can be used instead.

## Compute

- Unit tests run on CPU and do not require the source datasets.
- PyTorch projects select Apple Metal acceleration when available and fall back to CPU.
- Neural training is never part of CI.
- Local defaults favour low worker counts and modest batch sizes for an 8 GB machine.
- Short smoke runs check that the software works; they are not model results.

## Data and outputs

| Project | Included | Kept local |
|---|---|---|
| Tree LiDAR | Result tables, route details, evaluator and figures | Raw point clouds and prediction arrays |
| Liverpool urban accessibility | Derived area metrics, boundaries, centroids, spatial edges, models and figures | National Census workplace-flow table |
| Chest X-ray | Dataset specification, duplicate analysis and preparation code | 3,475 source images and checkpoints |
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
