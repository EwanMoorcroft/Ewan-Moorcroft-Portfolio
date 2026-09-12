# Evidence provenance

## Final/current evidence

The tables contain only `for_instance_pointwise_v2` rows. Their declared and verified shapes are 132
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

The route manifest has no historical upstream commit for either TreeX route and no checkpoint
SHA-256 for the project-adapted SegmentAnyTree route. These missing identities limit full prediction
reproduction. They are not replaced with present-day revisions or checkpoints.

The small `examples/aligned-labels.json` file is fictional evaluator input. It is separate from the
canonical result tables and demonstrates source-row alignment, boundary exclusion and matching.
