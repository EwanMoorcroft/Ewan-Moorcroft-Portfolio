# Tree LiDAR Instance Benchmark

This project compares six methods for identifying individual trees in terrestrial laser-scanning point clouds. Different methods produce different labels and have been developed under different settings, so the benchmark converts their aligned predictions into one common, reproducible evaluation on 11 held-out plots: 49,709,922 points and 323 reference trees.

![Published/default and development-selected micro F1](assets/micro_f1_paired.png)

## Results

All figures use `for_instance_pointwise_v2`. The two routes are intentionally kept apart: `published_default` is the published or default setup, while `development_tuned` is a checkpoint or parameter set selected with development data only.

| Method | Published/default micro F1 | Development-selected micro F1 |
|---|---:|---:|
| ForestFormer3D | 0.814 | **0.844** |
| ForAINet | 0.742 | 0.759 |
| SegmentAnyTree | 0.716 | 0.748 |
| TreeX | 0.572 | 0.567 |
| TreeLearn | 0.132 | 0.502 |
| TLS2trees | 0.000 | 0.276 |

ForestFormer3D reaches the highest micro F1 on the development-selected route in this comparison. ForestFormer3D, ForAINet, and SegmentAnyTree change only modestly between the routes, while TreeLearn and TLS2trees improve more substantially. These are route-specific descriptions, not controlled estimates of training benefit: the methods received unequal development effort. TreeX and TLS2trees use parameter selection rather than neural fine-tuning.

## What is being measured

The evaluator works on point-aligned reference and prediction labels. Reference trees are positive instance IDs on semantic classes 4, 5, or 6. A predicted instance is excluded only when class 3 makes up more than half of its aligned points; an instance at exactly 50% remains eligible.

For each plot, the scoring area is the union of valid reference-tree points and eligible prediction points. A predicted tree and a reference tree can match when their intersection over union is at least 0.50. The evaluator then finds a deterministic one-to-one set of matches that contains as many valid pairs as possible. Unmatched references are false negatives, unmatched predictions are false positives, and micro precision, recall, and F1 are calculated after those counts have been summed across plots.

Those choices are fixed by `for_instance_pointwise_v2`. The public evaluator rejects changes to the reference classes, ignored ID, boundary rule, or IoU threshold under that protocol name.

## A scored example

![ForestFormer3D reference and matched prediction](assets/forestformer3d_matched_comparison.png)

The held-out `CULS/plot_2_annotated` example uses the same colour for each accepted match. It contains 20 matches and no unmatched eligible reference or prediction instances. It was selected after scoring to show how the evaluator treats a clean plot, not as a summary of the full benchmark.

## Code and evidence

- [evaluator.py](src/tree_lidar_benchmark/evaluator.py) contains aligned-label scoring and one-to-one matching.
- [verification.py](src/tree_lidar_benchmark/verification.py) checks the result files, route identities, coverage, and aggregate reconstruction.
- [cli.py](src/tree_lidar_benchmark/cli.py) exposes `verify` and `summary`; [results_manifest.json](results_manifest.json) records the protocol and result inventory.
- [PROVENANCE.md](PROVENANCE.md) describes evidence scope, and [asset captions](assets/CAPTIONS.md) give context for the figures.

The raw point clouds and prediction arrays are not included. The published tables support deterministic aggregate verification without distributing those larger inputs.

## Run the evaluator

Python 3.11 or newer is sufficient; no third-party packages are required.

```bash
PYTHONPATH=src python3 -m tree_lidar_benchmark verify
PYTHONPATH=src python3 -m tree_lidar_benchmark summary
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The packaged wheel contains the same manifest-checked tables:

```bash
python -m pip install tree_lidar_benchmark-1.0.0-py3-none-any.whl
tree-lidar-benchmark verify
tree-lidar-benchmark summary
```

## Method and dataset sources

The test data are from the [FOR-instance dataset](https://doi.org/10.5281/zenodo.8287792). The six compared methods are documented by their original authors:

- [SegmentAnyTree](https://doi.org/10.1016/j.rse.2024.114367)
- [TreeLearn](https://doi.org/10.1016/j.ecoinf.2024.102888)
- [treeX](https://doi.org/10.48550/arXiv.2509.03633)
- [ForestFormer3D](https://arxiv.org/abs/2506.16991)
- [ForAINet](https://doi.org/10.1016/j.rse.2024.114078)
- [TLS2trees](https://doi.org/10.1111/2041-210X.14233)

The exact code revisions and checkpoint identities for the result rows are in [`data/route_manifest.json`](data/route_manifest.json).
