# X-ray results

| File | Contents |
| --- | --- |
| [overall-metrics.csv](overall-metrics.csv) | Grouped-split accuracy, balanced accuracy, macro F1 and Matthews correlation coefficient |
| [per-class-metrics.csv](per-class-metrics.csv) | Grouped-split precision, recall, F1 and support for each class |
| [confusion-matrix.csv](confusion-matrix.csv) | Test confusion matrix in the recorded label order |
| [split-summary.csv](split-summary.csv) | Image, exact-group and class counts for each partition |
| [duplicate-audit.csv](duplicate-audit.csv) | Five exact duplicate groups kept within split boundaries |

All model tables describe the current exact-copy-grouped evaluation. Values are
generated from `evidence/retained-results.json`; raw images, checkpoints and the
full perceptual-review candidate list are not included.
