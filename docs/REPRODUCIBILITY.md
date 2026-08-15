# Reproducibility

## Environment strategy

Each project owns a small `pyproject.toml` and can be installed independently. This avoids one large
environment combining unrelated computer-vision, reinforcement-learning, and tabular dependencies.
Python 3.11 is the shared baseline.

With `uv` installed, a typical project setup is:

```bash
cd projects/<project-name>
uv sync --extra test
uv run pytest
```

Project READMEs document any additional training or plotting extra. A standard virtual environment
and `pip` can be used instead.

## Compute profiles

- Verification and unit tests run on CPU and do not require raw datasets.
- PyTorch projects select Apple Metal acceleration when available and fall back to CPU.
- Neural training is never part of CI.
- Local defaults favour low worker counts and modest batch sizes for an 8 GB machine.
- Short smoke runs validate software flow but are not presented as trained-model evidence.

## Data boundaries

| Project | Included | Kept local |
|---|---|---|
| Tree LiDAR | Aggregate v2 tables, route manifest, evaluator, figures | Raw point clouds and prediction arrays |
| Liverpool urban accessibility | Derived 61-area metrics, boundaries, centroids, spatial edges, models, figures, source and result manifests | National Census workplace-flow table and rebuild outputs |
| Chest X-ray | Source specification, duplicate evidence, preparation code | 3,475 source images and checkpoints |
| FPL | Protocol code, retained aggregate evaluation, source manifest | Downloaded gameweek snapshots, row predictions and fitted models |
| LunarLander | Retained figures and scoped metric JSON | Model checkpoint and new run outputs |
| Neural chunking | Model/evaluator code, figures, scoped metric JSON | Source corpus, checkpoints, new run outputs |
| Academic geospatial background | Public-safe skills and methods summary | Source reports, raw observations, instructional material, personal identifiers and third-party datasets |

The X-ray corpus is the CC BY 4.0 [Chest X-Ray V1 dataset](https://data.mendeley.com/datasets/p5rm59k7ph/1),
DOI `10.17632/p5rm59k7ph.1`. The project verifier checks its expected class counts before preparation.

## Determinism

Projects fix Python, NumPy, and PyTorch seeds where relevant, record configuration with outputs, and
use stable hashes for data grouping or artifact identity. Hardware and dependency changes can still
affect floating-point training trajectories, so exact reproducibility and practical repeatability are
distinguished in the project limitations.

## Safe persistence

The forecasting project saves a transparent JSON model artifact instead of executable pickle data.
PyTorch projects write state dictionaries and metadata for locally produced checkpoints. Only trusted
local checkpoints should be loaded.
