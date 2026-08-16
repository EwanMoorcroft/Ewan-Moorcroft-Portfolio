# Chest X-Ray Classification

This project fine-tunes ResNet18 to classify chest X-ray images as **Normal**,
**Lung Opacity** or **Viral Pneumonia**. It combines a small, public
three-class dataset with a validation-selected checkpoint and a held-out test
evaluation. The surrounding pipeline builds an image manifest, records file
identity and creates splits that keep exact copies together.

[Open the walkthrough](notebooks/chest_xray_walkthrough.ipynb) for the tables,
charts and the code behind them.

## Results

These figures come from the available ResNet18 run. It used an image-level split;
a later audit found five exact duplicate pairs, and the dataset contains no
patient identifiers. The current grouped split therefore needs a new evaluation
before it can support a new performance claim.

| Measure | Value |
| --- | ---: |
| Selected model | ResNet18 fine-tuning with dropout |
| Validation macro F1 | 0.9512 |
| Test images | 522 |
| Test accuracy | 0.9368 |
| Test balanced accuracy | 0.9376 |
| Test macro F1 | 0.9381 |
| Test Matthews correlation coefficient | 0.9051 |

![Model aggregate metrics and per-class F1](assets/model-results.svg)

The [results tables](results/) and
[saved result record](evidence/retained-results.json) contain the underlying
values, including per-class performance.

## Skills and implementation

| Area | Main code |
| --- | --- |
| Dataset contract and source metadata | [`data/dataset-spec.json`](data/dataset-spec.json), [`spec.py`](src/chest_xray_benchmark/spec.py) |
| Image inventory and integrity checks | [`manifest.py`](src/chest_xray_benchmark/manifest.py) |
| Deterministic, duplicate-aware splitting | [`splitting.py`](src/chest_xray_benchmark/splitting.py) |
| Transfer learning and training configuration | [`modeling.py`](src/chest_xray_benchmark/modeling.py), [`training.py`](src/chest_xray_benchmark/training.py), [`default.toml`](configs/default.toml) |
| Evaluation and calibration metrics | [`evaluation.py`](src/chest_xray_benchmark/evaluation.py), [`metrics.py`](src/chest_xray_benchmark/metrics.py) |
| Command-line workflow | [`cli.py`](src/chest_xray_benchmark/cli.py) |

## Why exact copies matter

When an identical image is placed in both training and test data, the model has
already seen the test example while learning. That can make a test score look
better than it should. Each image receives a SHA-256 digest, and exact matches
are kept in the same partition. Near-duplicate checks are only prompts for
review: they do not establish patient identity, which cannot be assessed from
this public collection.

## Dataset

The expected source is the public **Chest X-Ray** dataset, version 1,
DOI [`10.17632/p5rm59k7ph.1`](https://doi.org/10.17632/p5rm59k7ph.1), under
CC BY 4.0. It contains 3,475 labelled images:

| Class | Images |
| --- | ---: |
| Normal | 1,250 |
| Lung Opacity | 1,125 |
| Viral Pneumonia | 1,100 |
| **Total** | **3,475** |

![Expected class counts from the dataset contract](assets/dataset-composition.svg)

Images are not included in this repository. Download and attribute them as
described in [`data/README.md`](data/README.md).

## Prepare and train

Create an environment with Python 3.11 or newer, install the package, then
place the downloaded images in `data/raw`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,test]"

cxr-benchmark verify \
  --data-root data/raw \
  --spec data/dataset-spec.json \
  --manifest-out data/derived/image-manifest.csv \
  --report-out data/derived/verification-report.json \
  --strict

cxr-benchmark split \
  --manifest data/derived/image-manifest.csv \
  --output data/derived/splits.csv \
  --summary-out data/derived/split-summary.json \
  --seed 534
```

Train and evaluate the selected checkpoint:

```bash
cxr-benchmark train \
  --data-root data/raw \
  --splits data/derived/splits.csv \
  --config configs/default.toml \
  --output-dir outputs/resnet18

cxr-benchmark evaluate \
  --data-root data/raw \
  --splits data/derived/splits.csv \
  --checkpoint outputs/resnet18/best-state-dict.pt \
  --run-metadata outputs/resnet18/run-metadata.json \
  --output outputs/resnet18/test-metrics.json
```

The configuration uses ImageNet initialisation, dropout and light rotations.
Horizontal flips are excluded because they may reverse laterality markers. The
validation set selects the checkpoint by macro F1; the test set is evaluated
only after selection.

## Clinical limits

This is a research exercise using public images, not a medical device. It is
not suitable for diagnosis, triage, treatment or patient-facing decisions.
The labels have not been independently clinically reviewed here, and no
external population has been evaluated.
