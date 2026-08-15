# Evidence and Claim Rules

This repository uses three result labels so a reader can tell exactly what has been verified.

## Current verified

The retained inputs are present, their identity is checked, and the included software independently
reconstructs the published aggregates. The Tree LiDAR project is the only project currently carrying
this label. Its verifier checks seven artifact hashes, all route identities, every held-out plot, and
1,152 aggregate values.

## Historical reference

A result artifact from an earlier run is retained, but the improved project has not regenerated that
number. The chest X-ray, LunarLander, and neural chunking projects use this label. Each README states
the original scope and why the value must not be treated as a fresh result from the rebuilt pipeline.

## Rebuilt protocol

The software and tests validate the corrected method, but no claim is made about performance on the
real target data. FPL forecasting uses this label because the earlier dataset construction created
invalid zero targets. Those models, metrics, and recommendations are excluded.

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

