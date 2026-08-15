# Ewan Moorcroft Portfolio

[![Quality checks](https://github.com/EwanMoorcroft/Ewan-Moorcroft-Portfolio/actions/workflows/quality.yml/badge.svg)](https://github.com/EwanMoorcroft/Ewan-Moorcroft-Portfolio/actions/workflows/quality.yml)

I am studying for an MSc in Data Science and Artificial Intelligence at the University of
Liverpool, with completion expected in September 2026. I completed a BSc in Geography there in
June 2025.

Most of my work is in Python. I am especially interested in problems where location, time, or an
awkward data boundary matters. That has led me to spatial analysis, forecasting, statistical
modelling, deep learning, and careful evaluation. R is included where it gives a useful independent
check rather than just repeating the Python workflow.

**Tools used here:** Python, pandas, NumPy, scikit-learn, PyTorch, GeoPandas, DuckDB, SQL, R, QGIS,
pytest, Docker, and GitHub Actions.

![Liverpool workplace-flow retention](projects/liverpool-urban-accessibility/assets/local-retention-map.png)

*The map shows the share of fixed-workplace flows from each Liverpool MSOA that remained within
the city. It is one result from the [urban-accessibility analysis](projects/liverpool-urban-accessibility/).*

## Main projects

| Project | Work completed | Evidence available |
|---|---|---|
| [Liverpool urban accessibility](projects/liverpool-urban-accessibility/) | Used DuckDB, GeoPandas, spatial weights, count models, and a separate R check to analyse Census workplace flows across 61 MSOAs | Fresh public-data analysis. Local fixed-workplace retention was 68.37%; Moran's I was 0.4901 |
| [FPL next-gameweek forecasting](projects/fpl-forecasting/) | Built a chronological forecasting pipeline with strict as-of features, season-bound JSON artifacts, hashed batch evidence, transactional DuckDB storage, and a read-only local service | Fresh historical evaluation. Ridge MAE was 1.110 and Spearman correlation was 0.717 across GW6 to GW15 |
| [Tree LiDAR benchmark](projects/tree-lidar-benchmark/) | Evaluated six tree-instance segmentation methods, each with a published/default and development-selected route, on 49.7 million aligned test points | Current verified evidence. The highest development-selected micro F1 was 0.8436 |
| [Chest X-ray classification](projects/chest-xray-classification/) | Rebuilt a three-class training pipeline around an identified public dataset, exact-copy grouping, strict split checks, and calibration metrics | Historical reference only. Macro F1 was 0.9381 on an older image-level split; the safer split needs a new run |

Two smaller projects show other parts of my MSc work:

- [Neural chunking](projects/neural-chunking/) compares BiLSTM and Transformer encoders, masks
  variable-length sequences correctly, and scores exact BIO spans. Its retained token macro F1 is a
  historical result, not a new benchmark.
- [LunarLander Double DQN](projects/lunar-lander-double-dqn/) covers replay memory, online and target
  networks, terminal handling, checkpoints, and deterministic evaluation. The retained reward came
  from one training seed, which is stated plainly in the project.

## What the repository shows

The Liverpool work is the clearest link between my Geography degree and data science. It includes
CRS checks, population-weighted centroids, origin-destination data, Moran's I, regression diagnostics,
maps, SQL-style transformation in DuckDB, and numerical comparison with R. My earlier academic work
also covered QGIS, raster analysis, environmental field observations, hypothesis tests, analysis of
variance, clustering, relational databases, transactions, and locking. The
[geospatial background note](docs/GEOSPATIAL_BACKGROUND.md) explains the useful parts without
publishing restricted teaching material or unclear third-party data.

Across the machine-learning projects, I have worked with temporal splits, grouped image splits,
development and test separation, baselines, class imbalance, ranking measures, calibration, sequence
masking, reinforcement learning, and 3D matching. The subjects differ, but the recurring concern is
whether a result still holds once leakage, duplicates, incomplete data, and selection decisions are
made explicit.

The code is organised as six installable packages with command-line tools and tests. CI installs each
project independently, runs its suite, checks formatting and links, and scans the public tree for
accidental private paths, credentials, and oversized files. The forecasting and Liverpool tools
also have non-root Docker builds. FPL container CI exercises a synthetic train, predict,
transactional store, and health endpoint without downloading live data.

## Finding your way around

```text
projects/
  liverpool-urban-accessibility/ public Census data, GIS, statistics, SQL, R and Docker
  fpl-forecasting/              chronological forecasting, batch evidence and local serving
  tree-lidar-benchmark/         3D tree-instance evaluation and retained evidence
  chest-xray-classification/    image classification and duplicate-safe data preparation
  neural-chunking/              BIO sequence labelling with two neural encoders
  lunar-lander-double-dqn/      reinforcement learning for discrete control
docs/                           evidence rules, reproducibility and academic background
scripts/                        link and public-safety checks
tests/                          tests for the repository-level checks
```

Each project README gives its data source, setup, result status, and limitations. Large datasets,
checkpoints, raw predictions, and local experiment outputs are kept out of Git.

To run the repository-level checks with Python 3.11 or newer:

```bash
python3 scripts/check_markdown_links.py .
python3 scripts/audit_repository.py .
```

Project environments are separate. Enter the project you want to inspect and follow its README.
The [reproducibility note](docs/REPRODUCIBILITY.md) explains the shared conventions, while
[evidence and claim rules](docs/EVIDENCE.md) explain why a current verified result, a historical
number, and a software-only check are labelled differently.

Code is available under the [MIT License](LICENSE). Data licences remain with the original
publishers and are recorded beside each project.
