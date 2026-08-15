# Data boundary

No national flow archive is stored here. The retained project includes only:

- a compact 61-row derived metrics table;
- a 151-edge Queen-contiguity table;
- the official 61-feature Liverpool boundary subset;
- the official 61-feature population-weighted centroid subset; and
- aggregate models, spatial results, source identity and file hashes.

The source and transformed tables use matching 2021 MSOA codes. Missing origin-destination pairs are
not assumed to be explicit zeros. Downloaded national files and local rebuild outputs belong under
ignored `data/raw/`, `data/interim/`, or `reports/generated/` paths.

See [`source-spec.json`](source-spec.json) for authoritative URLs and
[`../PROVENANCE.md`](../PROVENANCE.md) for attribution.
