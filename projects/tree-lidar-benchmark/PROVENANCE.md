# Evidence provenance

## Final/current evidence

This project carries a byte-identical evidence set from review revision
`6b927f7d12c0ab87fd29280ad6684dff23c0354a`. The retained tables contain only
`for_instance_pointwise_v2` rows. Their declared and verified shapes are 132
plot rows, 60 site rows, and 12 overall rows.

The result verifier reconstructs all site and overall aggregates from the plot
table. It compares 1,152 aggregate values and verifies the SHA-256 identity of
the three tables, route manifest, paired-route chart in two formats, and
matched-colour qualitative image. Exact hashes are recorded in
`results_manifest.json`.

The retained quality report recorded `BUILD ALL: PASS`, zero aggregate
discrepancies, and 20 accepted matches in the qualitative example.

## Interpretation

ForestFormer3D on the development-selected route has the highest micro F1 in
this fixed comparison. The route pairs are descriptive: development effort and
selection procedures differ between methods, so the differences are not
controlled estimates of training benefit.

## Exclusions

No raw point clouds, prediction arrays, environment configuration, job logs, or
private paths are included. The project verifies retained evidence offline and
does not recreate predictions.

