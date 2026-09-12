# Checkpoint re-evaluation and loss correction: 12 September 2026

The same validation-selected checkpoint was re-evaluated on the same 522 test
images, with no training, threshold tuning or model selection. Image bytes,
split bytes and selected-checkpoint SHA-256 were verified before inference.
The locked-environment [evaluation](checkpoint-reevaluation-locked-mps-2026-09-12.json)
is the controlling test record. Its runtime versions and device are recorded.

| Quantity | Superseded record | Verified re-evaluation |
| --- | ---: | ---: |
| Unweighted test cross entropy | 0.3967259264 | 0.6555965317 |
| Probability log loss | 0.6555965393 | 0.6555965369 |
| Test macro F1 | 0.8097125109 | 0.8097125109 |
| Correct / total | 421 / 522 | 421 / 522 |

The [original record](../evidence/superseded-retained-results-2026-08-16.json)
is preserved byte-for-byte and is superseded for test-loss reporting. Its
confusion matrix and classification scores are reproduced unchanged. The
original unweighted cross entropy disagreed with its own log loss; these should
agree within floating-point tolerance for this evaluation. The current evaluator
rejects such disagreement before writing outputs.

A diagnostic evaluation with PyTorch 2.9.1 on MPS produced another contradictory
scalar loss (0.948431834) while reproducing the original probabilities and
classification metrics. CPU evaluation of the same checkpoint in that runtime
gave coherent cross entropy/log loss (0.655596602 / 0.655596610). The locked
PyTorch 2.14.0 MPS evaluation gives the coherent values above. This establishes a
device/runtime-sensitive discrepancy in the older diagnostic runtime; it does
not establish the exact cause of the historical PyTorch 2.11.0 result. The final
evaluation also normalises only float32 rounding in probability row sums before
calculating probability metrics; this explains negligible last-digit changes.

Separately, historical weighted training/validation epoch losses averaged batch
means by image count instead of summed target weights. That reporting code is
fixed, but the past training trajectory was not rerun. The old validation loss
is therefore retained only in the superseded file and marked unavailable in the
current summary. Checkpoint selection used validation macro F1, not that loss.

## Inspectable probabilities and errors

[All 522 probability rows](checkpoint-reevaluation-locked-mps-2026-09-12-predictions.csv)
include a sequential row ID, source-image SHA-256, gold/predicted class indices
and three probabilities. Indices are **Normal, Lung Opacity, Viral Pneumonia**,
in that order. A content hash can repeat for an exact copy; the row ID is unique.
No image paths, patient identifiers, raw images or model weights are published.
The evaluation JSON pins this CSV by SHA-256 and includes all 15 reliability
bins, including empty bins, counts and mean confidence.

There are 101 errors. The largest error groups are 38 Normal images and 34 Viral
Pneumonia images predicted as Lung Opacity. Lung Opacity recall is 89.94%, but
precision is 67.86%: the model often assigns this class to other labels.
Mean confidence is 57.48% on correct predictions and 49.45% on errors. In the
60–66.67% confidence bin, 73 of 79 predictions are correct (92.41%); the
66.67–73.33% bin has 77 of 78 correct (98.72%). This retained test sample is
underconfident in those bins. ECE is 0.2473 and multiclass Brier score is 0.4013.
These are descriptive findings from one held-out sample, not evidence of
calibration on an external population. No calibration was fitted to test labels.

Derived results use the CC BY 4.0 source collection: Sugianto, Dwi (2025),
“Chest X-Ray”, Mendeley Data, V1, [doi:10.17632/p5rm59k7ph.1](https://doi.org/10.17632/p5rm59k7ph.1).
Patient-level independence and external-population performance remain unknown;
the benchmark is not suitable for clinical use.
