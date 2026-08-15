# Chest X-Ray Classification

The main problem here is not choosing a larger neural network. It is knowing exactly which public
dataset is being used and stopping identical files from appearing on both sides of evaluation. The
package compares a compact convolutional network with ResNet18 transfer learning, records file
identity, and rechecks both split structure and source-image bytes before training or evaluation.

## Safety boundary

This is a research benchmark, not a medical device. It must not be used for
diagnosis, treatment, triage or patient-facing decisions. Class labels come from
the source collection and have not been independently reviewed here.

## Dataset

The expected source is **Chest X-Ray**, version 1, DOI
[`10.17632/p5rm59k7ph.1`](https://doi.org/10.17632/p5rm59k7ph.1), licensed
under CC BY 4.0. It contains 3,475 images: Normal 1,250, Lung Opacity 1,125
and Viral Pneumonia 1,100. Images are deliberately excluded from this
repository. See [data/README.md](data/README.md) for acquisition and attribution.

## Why the split is different

Patient identifiers are not supplied. A random image-level split can therefore
place exact copies on both sides of evaluation. Perceptual image hashes can
also flag visually similar pairs, but similarity is not proof that two files
belong to the same patient or acquisition.
The preparation pipeline:

1. computes SHA-256 for exact identity;
2. places exact copies into indivisible split groups;
3. computes a 64-bit difference hash and reports direct candidate pairs for
   manual review;
4. never merges perceptual-hash candidates automatically or transitively;
5. assigns exact-identity groups to train, validation and test partitions; and
6. checks that no exact digest or exact-identity group crosses a boundary before writing the split;
   training and evaluation repeat that audit and re-hash every source image before building a loader.

This prevents exact-copy leakage but cannot prove visual or patient-level
independence. Difference-hash candidates are diagnostic only: the similarity
relation can collide and is not transitive. A future release should use
reviewed acquisition identifiers or verified patient keys as additional hard
grouping constraints if the publisher makes them available.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,test]"
```

Prepare the data:

```bash
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

Train and evaluate:

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

The default model is ResNet18 with ImageNet weights, dropout and a new
three-class output layer. Augmentation uses small rotations only; horizontal
flips are excluded because they can reverse laterality markers. Model selection
uses validation macro F1. The test partition is evaluated only by the explicit
evaluation command.

On Apple silicon, `device = "auto"` selects MPS when available. The default
batch size of 32 and zero worker processes are intended to remain practical on
an 8 GB machine; reduce the batch size if memory pressure appears. The first
pretrained run may fetch ImageNet weights. Set `pretrained = false` for a fully
offline run, with the expectation that results will differ.

## Earlier result

A retained earlier run selected ResNet18 fine-tuning with dropout and recorded
validation macro F1 **0.9512**. Its test record reported accuracy **0.9368**,
balanced accuracy **0.9376**, macro F1 **0.9381** and Matthews correlation
coefficient **0.9051** across 522 images.

These are **historical, not freshly reproduced results**. That earlier split was
image-level, five exact duplicate pairs were subsequently identified, and
patient-level independence was not established. The values are retained only
as provenance in [evidence/historical-results.json](evidence/historical-results.json);
they are not the expected score of the safer split implemented here. The retained
digest audit is available in
[evidence/known-exact-duplicates.json](evidence/known-exact-duplicates.json).

## Metrics written by a new run

Evaluation writes accuracy, balanced accuracy, macro and per-class
precision/recall/F1, Matthews correlation coefficient, multiclass log loss,
Brier score, expected calibration error and the confusion matrix. Every output
includes the split-manifest digest, selected-checkpoint digest, label order, seed and run settings.

## Checks

```bash
python -m pytest
```

The focused suite checks deterministic exact-identity grouping and splitting,
non-transitive visual-review candidates, cross-partition exact-copy guards,
tampered split and source-image rejection, checkpoint binding, source-spec parsing, manifest round
trips and a synthetic CPU smoke pass over the optional training stack.

## Limitations

- Patient-level keys and acquisition-site metadata are unavailable.
- Difference hashing is a review heuristic, not proof of shared origin; it is
  never used as an automatic split identity.
- The collection is small and may contain source-specific shortcuts.
- No external population has been evaluated.
- Probability calibration and uncertainty must be reviewed on independent data.
- Any future result must state whether it came from the grouped split in this
  repository or from historical evidence.
