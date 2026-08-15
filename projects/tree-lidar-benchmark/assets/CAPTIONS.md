# Visual captions

## Published/default versus development-selected micro F1

This paired comparison shows all six methods under
`for_instance_pointwise_v2`, with published/default and development-selected
routes kept separate. Micro F1 is recomputed from TP, FP and FN aggregated over
the same 11 held-out plots and 323 reference instances. ForestFormer3D on the
development-selected route records the highest value in this fixed comparison.
The route pairs are descriptive and have unequal development effort; TreeX and
TLS2trees use parameter selection rather than neural fine-tuning.

Retained files: `micro_f1_paired.svg` and `micro_f1_paired.png`.

## ForestFormer3D matched-colour comparison

Reference instances and the unchanged ForestFormer3D development-selected
prediction for held-out plot `CULS/plot_2_annotated` use the same camera, crop,
point size, and background. Each accepted v2 one-to-one match has the same
colour in both panels. The retained output contains 20 accepted matches and no
unmatched eligible reference or prediction instances. The plot was chosen
after scoring for illustration only; it did not influence method, parameter,
threshold, or metric selection and is not claimed to represent every plot.

Retained file: `forestformer3d_matched_comparison.png`.

