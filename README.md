# Ewan Moorcroft Portfolio

End-to-end machine learning systems for 3D LiDAR segmentation, medical-image research,
next-event forecasting, reinforcement learning, and neural sequence labelling. The focus is on
defensible evaluation, data integrity, reproducible Python packages, focused tests, and clear
technical communication.

![Paired LiDAR benchmark results](projects/tree-lidar-benchmark/assets/micro_f1_paired.png)

## Featured work

| Project | What it solves | Technical evidence | Result status |
|---|---|---|---|
| [Tree LiDAR instance benchmark](projects/tree-lidar-benchmark/) | Compares six instance-segmentation methods on dense, source-aligned point clouds | 49.7M held-out points, deterministic bipartite matching, route-aware evaluation, 1,152 aggregate checks | **Current verified:** top development-selected micro F1 0.8436 |
| [Chest X-ray classification](projects/chest-xray-classification/) | Builds a three-class transfer-learning pipeline around a public image corpus | Dataset identity contract, exact-duplicate grouping, visual-similarity review candidates, calibration metrics | **Historical reference:** macro F1 0.9381; safer split awaits a new run |
| [FPL next-gameweek forecasting](projects/fpl-forecasting/) | Predicts player points using only information available before the target gameweek | Empty-period rejection, as-of features, expanding-window evaluation, regression and ranking metrics, JSON model artifacts | **Fresh historical evaluation:** ridge MAE 1.110 and Spearman 0.717 across GW6–15 |

## Supporting work

| Project | Distinct capability | Result status |
|---|---|---|
| [LunarLander Double DQN](projects/lunar-lander-double-dqn/) | Replay memory, online/target networks, terminal masking, checkpoint lifecycle, deterministic greedy evaluation | **Historical reference:** mean reward 283.56 +/- 12.07 over 10 episodes from one retained training seed |
| [Neural chunking](projects/neural-chunking/) | BiLSTM and Transformer encoders, variable-length masking, duplicate-safe sentence splits, exact BIO span scoring | **Historical reference:** token macro F1 0.7521 from a retained BiLSTM run; retrospective span F1 unavailable |

The status wording is deliberate. Current verified results, retained historical evidence, and
software-only validation are never presented as equivalent. See [Evidence and claim rules](docs/EVIDENCE.md).

## Capability map

| Capability | Evidence in this repository |
|---|---|
| 3D computer vision and spatial data | Point-aligned LiDAR contracts, semantic eligibility rules, IoU matching, qualitative point-cloud comparison |
| Image classification | ResNet18 transfer learning, conservative augmentation, group-aware data preparation, calibration-aware evaluation |
| Temporal machine learning | Strict as-of features, next-gameweek targets, expanding windows, error and ranking measures |
| Deep learning systems | PyTorch CNN, BiLSTM, Transformer, and Double DQN implementations with Apple Metal/CPU selection |
| Dataset engineering | Schema validation, hashing, duplicate containment, immutable manifests, raw-data exclusion |
| Evaluation design | Development-only selection, untouched held-out sets, leakage guards, baselines, limitations tied to each claim |
| Python engineering | Typed packages, command-line interfaces, JSON/TOML configuration, safe persistence, unit and integration tests |
| Delivery tooling | Ruff, pytest, repository safety checks, GitHub Actions, and non-root Docker packaging for the forecasting CLI |

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
- Reusable logic lives in importable packages rather than depending on notebook state.
- Default workflows fit an 8 GB Apple laptop where practical; expensive experiments remain explicit.
- Lightweight CI checks code, tests, links, and public safety without training neural networks.

Code in this repository is available under the [MIT License](LICENSE). Dataset and environment
licences remain with their respective publishers.
