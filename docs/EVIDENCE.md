# Evidence and Claim Rules

This repository uses five result labels so a reader can tell exactly what has been verified.

## Current verified

The retained inputs are present, their identity is checked, and the included software independently
reconstructs the published aggregates. The Tree LiDAR project is the only project currently carrying
this label. Its verifier checks seven artifact hashes, all route identities, every held-out plot, and
1,152 aggregate values.

## Historical reference

A result artifact from an earlier run is retained, but the improved project has not regenerated that
number. The chest X-ray, LunarLander, and neural chunking projects use this label. Each README states
the original scope and why the value must not be treated as a fresh result from the rebuilt pipeline.

## Fresh historical evaluation

The included protocol has been run on a completed, commit-pinned public dataset and the aggregate
out-of-fold report and source manifest are retained. FPL forecasting uses this label for its fixed
2024–25 GW1–15 interval. The label does not imply full-season coverage, live-season performance, or
retention of raw records, row predictions, or a fitted model.

## Fresh public-data analysis

The included methods have been executed on dated, documented public sources. Compact transformed
inputs, source identities, licences, aggregate outputs and hashes are retained, while a large raw
source may remain outside Git. The Liverpool project uses this label. Its verifier proves retained
integrity and reconstructs the Python aggregates and coefficients; a separate R script checks the
cross-language values. The label does not claim that the national source can be rebuilt offline
without downloading the identified archive.

## Rebuilt protocol

The software and tests validate a corrected method, but no claim is made about performance on the
real target data. The deterministic FPL synthetic path uses this label for protocol and plumbing
checks. The invalid models, metrics, and recommendations from the earlier zero-target construction
remain excluded.

## Claim checklist

Every numerical claim should answer:

1. Which data identity and split does it use?
2. Was model or parameter selection isolated from the held-out set?
3. Which metric definition and aggregation route produced it?
4. Is the result reproduced here, independently verified here, or only retained?
5. Which limitations change how the number should be interpreted?

Missing evidence is stated rather than inferred. In particular, this repository does not reconstruct
unavailable prediction-level outputs, invent patient identities, or rerun expensive experiments to
make a historical number appear current.

## Academic background

The geospatial and analytical background page is a verified skills summary, not a fifth model-result
label. It describes completed academic experience while keeping source reports, raw observations,
private identifiers, and data with unclear redistribution rights outside the repository. It does not
promote archived numerical results to current reproducible evidence.
