# Ewan Moorcroft Portfolio

[![Quality checks](https://github.com/EwanMoorcroft/Ewan-Moorcroft-Portfolio/actions/workflows/quality.yml/badge.svg)](https://github.com/EwanMoorcroft/Ewan-Moorcroft-Portfolio/actions/workflows/quality.yml)

MSc Data Science and Artificial Intelligence student at the University of Liverpool, with a BSc
in Geography from the same institution.

I build reproducible machine learning systems across computer vision, 3D spatial data,
forecasting, reinforcement learning, and natural-language processing. My work emphasises
defensible evaluation, data integrity, tested Python packages, and clear technical communication.

**Core stack:** Python · PyTorch · scikit-learn · pandas · NumPy · pytest · Docker · GitHub Actions

**Technical focus:** computer vision · 3D LiDAR · spatial data · temporal modelling · rigorous model evaluation

![Paired LiDAR benchmark results](projects/tree-lidar-benchmark/assets/micro_f1_paired.png)

*Held-out micro F1 across six tree-instance segmentation routes; see the
[benchmark methodology and provenance](projects/tree-lidar-benchmark/).*

## Featured work

| Project | What I built and evaluated | Technical evidence | Result status |
|---|---|---|---|
| [Tree LiDAR instance benchmark](projects/tree-lidar-benchmark/) | Compares six instance-segmentation methods on dense, source-aligned point clouds | 49.7M held-out points, deterministic bipartite matching, route-aware evaluation, 1,152 aggregate checks | **Current verified:** top development-selected micro F1 0.8436 |
| [FPL next-gameweek forecasting](projects/fpl-forecasting/) | Predicts player points using only information available before the target gameweek | Empty-period rejection, as-of features, expanding-window evaluation, regression and ranking metrics, JSON model artifacts | **Fresh historical evaluation:** ridge MAE 1.110 and Spearman 0.717 across GW6–15 |
| [Chest X-ray classification](projects/chest-xray-classification/) | Builds a three-class transfer-learning pipeline around a public image corpus | Dataset identity contract, exact-duplicate grouping, visual-similarity review candidates, calibration metrics | **Historical reference:** macro F1 0.9381; safer split awaits a new run |

## Supporting work

| Project | Distinct capability | Result status |
|---|---|---|
| [Neural chunking](projects/neural-chunking/) | BiLSTM and Transformer encoders, variable-length masking, duplicate-safe sentence splits, exact BIO span scoring | **Historical reference:** token macro F1 0.7521 from a retained BiLSTM run; retrospective span F1 unavailable |
| [LunarLander Double DQN](projects/lunar-lander-double-dqn/) | Replay memory, online/target networks, terminal masking, checkpoint lifecycle, deterministic greedy evaluation | **Historical reference:** mean reward 283.56 +/- 12.07 over 10 episodes from one retained training seed |

The status wording is deliberate. Current verified results, retained historical evidence, and
software-only validation are never presented as equivalent. See [Evidence and claim rules](docs/EVIDENCE.md).

## Capability map

| Capability | Public evidence and bounded experience |
|---|---|
| 3D computer vision and spatial data | Point-aligned LiDAR contracts, semantic eligibility rules, IoU matching, qualitative point-cloud comparison |
| GIS and geospatial analysis | [Academic experience summary](docs/GEOSPATIAL_BACKGROUND.md): QGIS, GeoPandas, Rasterio, Rasterstats, PySAL, R `sf`/`tmap`, CRS handling, spatial joins, zonal statistics, origin-destination flows and spatial autocorrelation |
| Image classification | ResNet18 transfer learning, conservative augmentation, group-aware data preparation, calibration-aware evaluation |
| Temporal machine learning | Strict as-of features, next-gameweek targets, expanding windows, error and ranking measures |
| Deep learning systems | PyTorch CNN, BiLSTM, Transformer, and Double DQN implementations with Apple Metal/CPU selection |
| Statistics and data analysis | [Academic experience summary](docs/GEOSPATIAL_BACKGROUND.md): environmental observations, data cleaning, exploratory visualisation, hypothesis tests, analysis of variance and Poisson regression |
| Relational data and SQL | [Academic experience summary](docs/GEOSPATIAL_BACKGROUND.md): relational modelling, joins, grouping, nested queries, constraints, transactions and concurrency-safe updates |
| Dataset engineering | Schema validation, hashing, duplicate containment, immutable manifests, raw-data exclusion |
| Evaluation design | Development-only selection, untouched held-out sets, leakage guards, baselines, limitations tied to each claim |
| Python engineering | Typed packages, command-line interfaces, JSON/TOML configuration, safe persistence, unit and integration tests |
| Delivery tooling | Ruff, pytest, repository safety checks, GitHub Actions, and non-root Docker packaging for the forecasting CLI |

## Education and geospatial background

My Geography background adds spatial, environmental, and decision-focused analysis to the machine
learning work above. It includes raster and vector geodata, census and urban-mobility data,
environmental observations, cartography, spatial statistics, and communicating results to
non-specialist audiences.

The [academic geospatial and analytical background](docs/GEOSPATIAL_BACKGROUND.md) page documents
this experience without publishing raw field observations, instructional material, private
identifiers, or third-party datasets with unclear redistribution terms.

## Repository navigation

```text
projects/
  tree-lidar-benchmark/         flagship 3D instance-segmentation evidence pack
  chest-xray-classification/    image-classification lifecycle and data controls
  fpl-forecasting/              corrected chronological forecasting system
  lunar-lander-double-dqn/      reinforcement-learning control system
  neural-chunking/              neural BIO sequence labelling
docs/                           cross-project evidence and reproducibility rules
scripts/                        local link and public-safety checks
tests/                          repository-tool tests
.github/workflows/              lightweight quality checks only
```

Each project is independently packaged and has its own setup, data contract, run commands, tests,
results, and limitations. Large datasets, model checkpoints, raw predictions, and local experiment
outputs are intentionally excluded.

This repository is a curated, public-safe presentation of earlier academic and research work. Its
public commit history records the portfolio assembly and hardening process, not the full original
development timeline of every project.

## Quick validation

Python 3.11 or newer is the common baseline. Project environments are separate so a lightweight
evaluator does not inherit deep-learning dependencies it does not use.

```bash
python3 scripts/check_markdown_links.py .
python3 scripts/audit_repository.py .
```

Then enter a project and follow its README. For example, the dependency-free LiDAR verifier runs as:

```bash
cd projects/tree-lidar-benchmark
PYTHONPATH=src python3 -m tree_lidar_benchmark verify
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Full environment and compute guidance is in [Reproducibility](docs/REPRODUCIBILITY.md).

## Design principles

- A metric is published only with its protocol, split, selection route, and known limitations.
- Raw or unsuitable data stays outside Git; download and validation steps replace copied datasets.
- Academic experience is summarised without copying source reports, private identifiers, or restricted data.
- Reusable logic lives in importable packages rather than depending on notebook state.
- Default workflows fit an 8 GB Apple laptop where practical; expensive experiments remain explicit.
- Lightweight CI checks code, tests, links, and public safety without training neural networks.

Code in this repository is available under the [MIT License](LICENSE). Dataset and environment
licences remain with their respective publishers.
