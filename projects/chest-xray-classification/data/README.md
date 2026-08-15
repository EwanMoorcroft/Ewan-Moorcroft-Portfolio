# Data acquisition and attribution

This repository does not redistribute the images. Download version 1 from the
[Mendeley Data record](https://data.mendeley.com/datasets/p5rm59k7ph/1), then
extract `Lung X-Ray.zip` into `data/raw/`.

Expected classes and counts:

| Class | Images |
|---|---:|
| Normal | 1,250 |
| Lung Opacity | 1,125 |
| Viral Pneumonia | 1,100 |
| Total | 3,475 |

The record identifies the Kaggle collection
[`fatemehmehrparvar/lung-disease`](https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease/data)
as its source. This is a three-class, 3,475-image collection; it is not the
5,856-image, two-class collection commonly used in other chest X-ray examples.

The data record is licensed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Retain attribution to:

> Sugianto, Dwi (2025), “Chest X-Ray”, Mendeley Data, V1,
> https://doi.org/10.17632/p5rm59k7ph.1

After extraction, run the verifier before creating any split:

```bash
cxr-benchmark verify \
  --data-root data/raw \
  --spec data/dataset-spec.json \
  --manifest-out data/derived/image-manifest.csv \
  --report-out data/derived/verification-report.json \
  --strict
```

The verifier checks identity, class counts, readability, dimensions, SHA-256
digests, embedded metadata presence and exact duplicates. It assigns automatic
split groups only from exact SHA-256 identity. A 64-bit difference hash reports
direct visual-review candidate pairs, but those candidates are never merged or
used as automatic split constraints. The output stores only relative paths and
derived metadata.

The derived CSV contract is `exact-identity-v1`: `sha256` must be exactly 64 lowercase hexadecimal
characters and `exact_group_id` must equal `sha256-<sha256>`. Legacy `group_id` manifests are
rejected and should be regenerated. The split command validates this invariant before allocation,
writes no split CSV unless leakage and class-coverage checks pass, and every training or evaluation
read repeats the cross-partition identity audit.
