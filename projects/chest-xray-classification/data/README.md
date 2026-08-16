# Data acquisition and attribution

The images are not redistributed in this repository. Download version 1 of
[Chest X-Ray on Mendeley Data](https://data.mendeley.com/datasets/p5rm59k7ph/1)
and extract `Lung X-Ray.zip` into `data/raw/`.

| Class | Expected images |
| --- | ---: |
| Normal | 1,250 |
| Lung Opacity | 1,125 |
| Viral Pneumonia | 1,100 |
| **Total** | **3,475** |

This is a three-class collection; a binary pneumonia dataset is not a substitute
for this project.

The data is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Please retain this attribution:

> Sugianto, Dwi (2025), “Chest X-Ray”, Mendeley Data, V1,
> https://doi.org/10.17632/p5rm59k7ph.1

Before splitting, verify the download:

```bash
cxr-benchmark verify \
  --data-root data/raw \
  --spec data/dataset-spec.json \
  --manifest-out data/derived/image-manifest.csv \
  --report-out data/derived/verification-report.json \
  --strict
```

This checks the expected classes, readable images and exact copied files, then
writes a manifest used by the split and training commands.
