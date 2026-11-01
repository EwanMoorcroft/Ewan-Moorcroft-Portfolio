# Ewan Moorcroft Portfolio

[![Quality checks](https://github.com/EwanMoorcroft/Ewan-Moorcroft-Portfolio/actions/workflows/quality.yml/badge.svg)](https://github.com/EwanMoorcroft/Ewan-Moorcroft-Portfolio/actions/workflows/quality.yml)

I have completed an MSc in Data Science and Artificial Intelligence at the University of Liverpool
and am awaiting formal award. I graduated with a BSc in Geography in June 2025.

My main language is Python. I use it for statistical modelling, machine learning and data
engineering, often with spatial or time-dependent data. This portfolio also includes R, SQL, GIS
and Docker where they suit the problem.

[![Greater Manchester neighbourhood adaptation priorities](projects/uk-climate-resilience-gis/outputs/maps/05_priorities.png)](projects/uk-climate-resilience-gis/outputs/maps/05_priorities.png)

*The balanced screen highlights neighbourhoods for local investigation. The 153 outlined areas meet the stated persistence rule; 174 grey areas lack enough heat evidence and remain unassessed. Open the full-size map for its legend and sources.*

## Selected projects

### [Greater Manchester climate resilience](projects/uk-climate-resilience-gis/)

This independent consultancy-style case asks where an adaptation team should investigate first.
I combined flood, satellite, greenspace, deprivation and population evidence across 1,702
neighbourhoods, then tested how the shortlist changes with policy assumptions. Python and PyQGIS
produce a reviewable decision brief, six maps and an offline explorer: 153 areas meet the stated
persistence rule, while 174 remain unassessed because heat evidence is insufficient.

### [Liverpool urban accessibility](projects/liverpool-urban-accessibility/)

This project examines workplace movement within Liverpool using Census 2021 origin-destination data.
I used DuckDB, GeoPandas and spatial statistics to prepare, map and analyse the flows.
Of the fixed-workplace flows represented in the data, 68.37% remained within the city. The area
pattern was spatially clustered, with Moran's I of 0.4901. A separate R implementation checks the
main numerical results.

### [FPL next-gameweek forecasting](projects/fpl-forecasting/)

This project predicts a player's points in the following gameweek from information already
available at the forecast date. The ridge model reached an MAE of 1.110 and a Spearman correlation
of 0.717 across ten chronological test windows. It also includes a command-line workflow, DuckDB
storage, a read-only local service and a non-root Docker image.

### [Tree LiDAR benchmark](projects/tree-lidar-benchmark/)

This benchmark compares six individual-tree segmentation methods under one scoring protocol for
UAV and airborne LiDAR data. It handles point alignment, one-to-one tree matching, class-aware scoring
and two result routes per method across 49.7 million aligned points.
ForestFormer3D achieved the highest development-selected micro F1 at 0.8436.

### [Chest X-ray classification](projects/chest-xray-classification/)

A three-class image-classification project using ResNet18 transfer learning. The grouped-split run
reached macro F1 of 0.8097 on 522 test images. The data pipeline identifies byte-identical images,
keeps each exact copy group in one partition and reports calibration alongside classification metrics.

## Other work

- [Neural chunking](projects/neural-chunking/) labels noun-phrase spans with BiLSTM and Transformer
  encoders. The available BiLSTM result reached token macro F1 of 0.7521.
- [LunarLander Double DQN](projects/lunar-lander-double-dqn/) covers replay memory, online and target
  networks, checkpointing and deterministic evaluation in a reinforcement-learning environment.

Together, the projects demonstrate work with pandas, NumPy, scikit-learn, PyTorch, GeoPandas,
DuckDB, SQL, R, QGIS, pytest, Docker and GitHub Actions. My
[geospatial background](docs/GEOSPATIAL_BACKGROUND.md) gives more context on the link between the
Geography degree and current data-science work. The [reproducibility note](docs/REPRODUCIBILITY.md)
explains the shared approach to environments, data sources and saved outputs.

Each folder under [`projects/`](projects/) is self-contained, with its own README, source code and
results. The current source is published under an [evaluation-only licence](LICENSE): it may be
viewed, cited and run for professional assessment, but not reused in other work. Data and
third-party materials retain the licences recorded within the relevant project. Earlier versions
remain governed by the licence supplied with those versions.
