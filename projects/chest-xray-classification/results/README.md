# X-ray results

These CSV files are deterministic extracts from the committed JSON evidence in
`evidence/`. They are included so the project can be inspected without
downloading the image collection.

The model metrics come from an earlier run, not a new run of the grouped split
implemented in this repository. The earlier experiment used
an image-level split and five exact duplicate pairs were identified afterwards.
The duplicate audit is retained for provenance and is not a substitute for
rerunning verification against the source images.

| File | Contents | Evidence status |
| --- | --- | --- |
| [overall-metrics.csv](overall-metrics.csv) | Accuracy, balanced accuracy, macro F1 and MCC | Retained earlier run |
| [per-class-metrics.csv](per-class-metrics.csv) | Precision, recall, F1 and support by class | Retained earlier run |
| [duplicate-audit.csv](duplicate-audit.csv) | Five exact duplicate pairs identified in the earlier audit | Retained digest audit |

The charts in `../assets/` are generated from the same committed evidence by
`analysis/build_walkthrough.py`. No data download, training or prediction is
performed by that script.
