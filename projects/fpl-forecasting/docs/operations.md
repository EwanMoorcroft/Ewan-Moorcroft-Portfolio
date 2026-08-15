# Operational contract

This layer turns one season-scoped ridge artifact into reproducible batch evidence and a
small read-only local service. It does not change the evaluation protocol or add a performance
claim.

## Batch boundary

`fpl-forecast predict` accepts a canonical season, an expected latest gameweek, and the literal
caller declaration `completed`. Every snapshot must carry that season in the exact
`fpl-completed-gameweek-v1` wrapper, and the latest wrapper and filename must match the expected
boundary. The model must have been trained through that same target gameweek. The season, effective
protocol configuration, and semantic feature schema must match the artifact.

Predictions cover every player in the latest snapshot and target exactly the following gameweek.
They are ordered by predicted points, then player ID. The prediction file and manifest use compact,
sorted-key JSON with a final line feed. No timestamp or machine-local path enters these files.

The manifest binds SHA-256 identities for:

- the exact artifact bytes;
- every exact input file and their ordered aggregate;
- the effective protocol configuration;
- the versioned semantic feature schema; and
- the canonical prediction output.

The completion declaration is not independent source verification. A non-empty snapshot can still
have been captured before the official event finished. Acquisition time and official completion
evidence belong in a separate operator log.

## Verified persistence

`fpl-forecast store-batch` does not trust the supplied output files on name alone. It reconstructs
the feature frame, predictions, and manifest from the supplied snapshots and artifact, then requires
exact canonical bytes and matching hashes. A model registration, the exact prediction and manifest
file hashes, and the prediction rows are written in one DuckDB transaction.

The idempotency key covers the model identity, season, as-of and target gameweeks, source snapshot
hash, forecast-frame hash, feature-schema hash, and protocol-configuration hash. Repeating the exact
request returns the existing run with `replayed: true`. A different output under that identity is a
conflict. This is the storage contract; the HTTP `POST /predict` route does not persist data and does
not accept an idempotency key.

DuckDB is used as a local embedded store. The supported boundary is one local process, one database,
and one writer transaction at a time. Migrations and write transactions for the same resolved
database path are serialized inside that process; readers use separate connections. This is not a
network database or a shared multi-process writer service. The HTTP process opens an already migrated
database read-only and runs with one worker.

## HTTP surface

Launch one local service with `fpl-forecast-service`. Startup validates the artifact, snapshot
sequence, database, season, as-of gameweek, completion declaration, configuration, and feature
schema before listening. It also requires the exact artifact's deterministic `model_id` to exist in
the database. The default bind address is `127.0.0.1`.

The only routes are:

- `GET /health` for model and database readiness;
- `GET /model` for the deterministic `model_id` and non-sensitive artifact and schema identities;
- `POST /predict` for in-memory scoring of 1 to 100 requested player IDs;
- `GET /predictions/{season_id}/{target_gw}?limit=100` for 1 to 500 rows from the latest stored batch
  for the configured model, including its prediction and manifest file hashes;
- `GET /metrics` for fixed-label HTTP response and returned-prediction counters.

The strict prediction body is:

```json
{"season_id":"2025-26","as_of_gw":10,"player_ids":[1,2]}
```

Player IDs must be unique positive 32-bit integers. Extra fields, an incorrect season or gameweek,
an unknown player, and request bodies above 16,384 bytes are rejected. Selected predictions retain
their rank in the full configured latest-snapshot population.

The artifact is limited to 10 MB, a prediction file to 5 MB, and a manifest to 1 MB. These limits
bound local parsing but are not substitutes for an authenticated gateway.

There are no documentation, OpenAPI, training, upload, registration, or database-write routes.
Error responses omit local paths and internal exception text. The process does not provide
authentication, TLS termination, or rate limiting, so a wider bind requires those controls in a
separate trusted gateway.

## Monitoring and model limits

Readiness proves that the configured files and database satisfy the startup contract. Metrics count
HTTP outcomes and returned predictions using fixed endpoint and status labels. They do not measure
forecast quality or feature drift.

No built-in feature-shift diagnostic is implemented. Before each new deployment, rerun the
chronological evaluation for the intended season and inspect the input distributions, missingness,
ranking stability, and residuals. The model also omits fixtures, prices, injury news, predicted
line-ups, and team or position constraints, so the service is a ranking component rather than a
squad-selection system.
