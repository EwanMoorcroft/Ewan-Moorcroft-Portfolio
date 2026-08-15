# FPL Next-Gameweek Forecasting

Fantasy Premier League data makes temporal leakage easy: a row can look valid while containing
information that was not available when the decision would have been made. This project predicts a
player's next-gameweek points and treats that time boundary as part of the data contract.

[Open the project overview](demo/index.html) for a quick visual tour.

## Question and approach

The pipeline reads canonical season-tagged snapshots converted from completed Fantasy Premier League
live-gameweek payloads. It validates their sequence, creates strictly as-of player features, and
predicts points in the next completed gameweek. Standardized ridge regression is compared with three
simple baselines under an expanding-window evaluation.

The repository contains no downloaded match data and no trained model. It retains one compact,
real-data evaluation report with an auditable source manifest. A deterministic synthetic generator
supports local checks and examples without network access.

## Guardrails

- Empty gameweeks are rejected rather than converted into zero targets; completion status must
  already have been established when each file is captured.
- Duplicate and non-consecutive gameweek files are rejected.
- Adjacent snapshots must meet a configurable player-coverage threshold.
- A row with features from gameweek `t` receives a target only from completed gameweek `t+1`.
- Players missing from either side of a pair are excluded; absence is never invented as a zero score.
- Model inputs come from a fixed numeric allow-list. Identifiers, target values, and future-facing columns cannot enter training.
- Evaluation uses expanding train windows followed by untouched later gameweeks.
- Results include MAE and RMSE for point error, plus Spearman, NDCG@K, and top-K overlap for recommendation quality.
- Saved ridge artifacts are plain JSON, not executable pickle files. A saved model artifact is bound
  to one season, one effective configuration, and one semantic feature schema.

## Historical evaluation on public data

The retained report was generated on 15 August 2026 from gameweeks 1 to 15 of the completed 2024-25
season. The source is Vaastav Anand's public FPL historical dataset at immutable revision
[`a59a43a`](https://github.com/vaastav/Fantasy-Premier-League/tree/a59a43a8343d960a58cbe7a1f9fba2d2ce431856/data/2024-25/gws).
The interval was chosen before model evaluation: it is the longest season-opening interval that
passes the pre-existing 0.90 adjacent-player coverage gate. The GW15→GW16 coverage is 0.8987, so
the pipeline rejects joining later gameweeks across that boundary.

The expanding-window comparison trained first on target gameweeks 2 to 5, then evaluated ten
chronologically held-out target gameweeks (GW6 to GW15), covering 6,684 player-gameweek rows:

| Candidate | MAE ↓ | RMSE ↓ | R² ↑ | Spearman ↑ | NDCG@10 ↑ | Top-10 overlap ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Last gameweek | 1.235 | 2.673 | -0.274 | 0.655 | 0.084 | 0.090 |
| Three-observation mean | 1.130 | 2.232 | 0.111 | 0.692 | 0.151 | 0.110 |
| Training-window mean | 1.501 | 2.368 | -0.000 | 0.000 | 0.002 | 0.020 |
| Ridge regression | **1.110** | **2.019** | **0.273** | **0.717** | **0.226** | **0.170** |

These are out-of-fold results for this fixed historical interval, not a live-season or full-season
claim. Ridge regularization and split settings were the existing defaults and were not selected on
these results. See the [evaluation report](reports/retained/2024-25-gw01-15-evaluation.json) and
[source manifest](reports/retained/2024-25-gw01-15-source-manifest.json) for exact folds, hashes,
conversion rules, and per-file checks.

The upstream repository uses an MIT licence for its software; its [licence
file](https://github.com/vaastav/Fantasy-Premier-League/blob/a59a43a8343d960a58cbe7a1f9fba2d2ce431856/LICENSE)
states that the underlying data belong to the named data providers. This repository therefore retains no
source CSVs or converted player records. Only aggregate metrics and file-level provenance are kept.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
ruff check .
ruff format --check .
```

Create a deterministic local example:

```bash
fpl-forecast synthetic \
  --output-dir scratch/synthetic-gameweeks \
  --season 2025-26 \
  --gameweeks 10 \
  --players 24 \
  --seed 42
```

Validate and evaluate it:

```bash
fpl-forecast validate \
  --gameweek-dir scratch/synthetic-gameweeks \
  --config config/default.json

fpl-forecast evaluate \
  --gameweek-dir scratch/synthetic-gameweeks \
  --config config/default.json \
  --report-json reports/generated/evaluation.json \
  --predictions-csv reports/generated/predictions.csv \
  --report-html reports/generated/evaluation.html
```

Fit a final ridge model only after reviewing the out-of-fold comparison:

```bash
fpl-forecast train \
  --gameweek-dir scratch/synthetic-gameweeks \
  --season 2025-26 \
  --config config/default.json \
  --artifact models/ridge.json
```

Score the next gameweek from the same completed sequence:

```bash
fpl-forecast predict \
  --gameweek-dir scratch/synthetic-gameweeks \
  --artifact models/ridge.json \
  --season 2025-26 \
  --expected-as-of-gw 10 \
  --completion-status completed \
  --config config/default.json \
  --predictions reports/generated/gw11-predictions.json \
  --manifest reports/generated/gw11-manifest.json
```

`--expected-as-of-gw` must equal the latest supplied file. `--completion-status completed` is a
caller declaration; see the [completion boundary](docs/data-contract.md#completion-boundary). The
command refuses a different season, effective configuration, feature schema, or artifact gameweek.
It ranks players found in the latest snapshot for gameweek `t+1`, then writes canonical JSON with a
stable player-ID tie-break.

The manifest records SHA-256 identities for the exact artifact bytes, ordered input snapshots,
effective configuration, semantic feature schema, and prediction output. It stores file names but
not machine-local paths or a clock timestamp, so the same inputs produce the same manifest bytes.

Verify and persist that generated batch in a local DuckDB database:

```bash
fpl-forecast store-batch \
  --database scratch/forecasts.duckdb \
  --gameweek-dir scratch/synthetic-gameweeks \
  --artifact models/ridge.json \
  --predictions reports/generated/gw11-predictions.json \
  --manifest reports/generated/gw11-manifest.json
```

`store-batch` rebuilds the forecast frame and expected output from the supplied snapshots and
artifact. It verifies the canonical prediction and manifest bytes before one transactional write,
including the exact hashes of both evidence files. An exact replay returns the same run and
idempotency identities with `replayed: true`; changed evidence under the same input identity is
rejected.

The optional local HTTP process starts only after that database exists:

```bash
fpl-forecast-service \
  --artifact models/ridge.json \
  --gameweek-dir scratch/synthetic-gameweeks \
  --database scratch/forecasts.duckdb \
  --season 2025-26 \
  --expected-as-of-gw 10 \
  --completion-status completed \
  --config config/default.json
```

It opens DuckDB read-only, requires the exact loaded model identity to be registered, and provides
bounded health, model-identity, in-memory prediction, model-filtered stored prediction, and metrics
endpoints. It has no training, upload, registration, or write route. See [the operational
contract](docs/operations.md) before exposing it beyond the default loopback address.

The `scratch/`, `data/`, `models/`, and generated report locations are ignored by Git.

To reproduce the retained historical import, download and extract the pinned upstream archive,
then run the adapter against its `data/2024-25/gws` directory:

```bash
fpl-forecast import-vaastav \
  --source-dir /path/to/pinned-archive/data/2024-25/gws \
  --output-dir data/interim/2024-25-gw01-15 \
  --manifest reports/generated/2024-25-gw01-15-source-manifest.json \
  --season 2024-25 \
  --source-revision a59a43a8343d960a58cbe7a1f9fba2d2ce431856 \
  --gameweek-start 1 \
  --gameweek-end 15
```

The retained manifest records the archive URL and SHA-256. The adapter cannot inspect Git history,
so its `--source-revision` value is supplied by the caller; the per-file hashes and aggregate hash are
the byte-level identity checks it actually performs. Run `validate` and `evaluate` with the same
default configuration after import. Raw and converted records remain ignored.

## Input contract

Use one file per completed gameweek, named `gameweek-<number>.json` or
`gameweek-<number>-<label>.json`. Raw FPL live payloads are not accepted directly. Each file must use
the exact completed-snapshot wrapper:

```json
{
  "snapshot_format": "fpl-completed-gameweek-v1",
  "season_id": "2025-26",
  "gameweek": 7,
  "elements": [
    {
      "id": 101,
      "stats": {
        "total_points": 6,
        "minutes": 90,
        "starts": 1,
        "ict_index": "5.4"
      }
    }
  ]
}
```

The wrapper adds the season and gameweek that are absent from the raw live payload. Completion is a
caller assertion rather than independent verification; see the [completion boundary](docs/data-contract.md#completion-boundary).
The validator rejects empty placeholders and season or gameweek mismatches. Operational training and
prediction require the same canonical season, and saved model artifacts cannot be reused across seasons.

`total_points` and `minutes` are required. Missing optional statistics are set to zero. Numeric values must be finite, player IDs must be unique within a gameweek, and the `elements` list cannot be empty. See [the full data contract](docs/data-contract.md).

## Feature contract

Every feature is available by the end of the as-of gameweek:

- latest points, minutes, start flag, ICT, influence, creativity, and threat;
- means over a player's latest three recorded rows for points, minutes, ICT, goals, and assists;
- start rate over those latest three recorded rows;
- points mean over the latest five recorded rows;
- season-to-date points, minutes, appearances, and points per appearance.

These are observation windows, not guaranteed consecutive-gameweek windows: a player who is absent
from a snapshot has no row added to their personal history. Player ID and gameweek fields remain
metadata and are never model inputs. Team, position, current price, fixture difficulty, and
availability news are absent because the live-gameweek files do not provide trustworthy historical
as-of versions.

The ordered fields, their `float64` type, window meaning, value meaning, as-of boundary, population,
and missing-observation meaning form a versioned semantic schema. Its canonical JSON SHA-256 is
stored in saved model artifacts and checked again before prediction.

## Evaluation

For each fold, all training target gameweeks precede all test target gameweeks. The window expands after each test period. The candidates are:

1. last observed gameweek points;
2. mean points over the last three completed gameweeks;
3. mean target from the current training window;
4. standardized ridge regression.

Point error alone does not describe recommendation quality, so ranking metrics are calculated separately within each target gameweek and then averaged. Full formulas and edge-case rules are in [the methodology note](docs/methodology.md).

## Scope and limitations

This project forecasts individual next-gameweek points; it does not optimize a full squad, transfers, captaincy, budget, or bench order. It does not include fixture schedules, opponent strength, injury reports, predicted line-ups, or historical price snapshots. Results depend on the supplied season and must be re-evaluated after material rule or data changes.

Synthetic results exercise the pipeline and protocol only. They are not live-season performance.
Forecasts remain uncertain and should not be treated as a guarantee.

## Files

```text
config/default.json          Protocol settings
demo/index.html              Self-contained visual overview
docs/                        Data, evaluation, model, and operational notes
reports/retained/            Compact metrics and source audit evidence
src/fpl_forecasting/         Validation, modelling, batch storage, CLI, and local service
tests/                       Deterministic contract and end-to-end checks
```

The Docker build pins the direct runtime dependencies in `requirements.txt`. CI uses
`requirements-dev.txt`, which adds pinned test tools. These files are not a hash-locked record of
every transitive package; compatible ranges remain in `pyproject.toml` for ordinary installation.

## Docker

The optional image uses a slim Python runtime and runs as an unprivileged user. Build it locally,
then keep generated evidence in a mounted scratch directory:

```bash
docker build -t fpl-forecast .
mkdir -p scratch/container
chmod 0777 scratch/container
docker run --rm -v "${PWD}/scratch/container:/work" fpl-forecast \
  synthetic --output-dir /work/gameweeks --season 2025-26 \
  --gameweeks 10 --players 24 --seed 42
docker run --rm -v "${PWD}/scratch/container:/work" fpl-forecast \
  train --gameweek-dir /work/gameweeks --artifact /work/ridge.json --season 2025-26
docker run --rm -v "${PWD}/scratch/container:/work" fpl-forecast \
  predict --gameweek-dir /work/gameweeks --artifact /work/ridge.json --season 2025-26 \
  --expected-as-of-gw 10 --completion-status completed \
  --predictions /work/predictions.json --manifest /work/manifest.json
docker run --rm -v "${PWD}/scratch/container:/work" fpl-forecast \
  store-batch --database /work/forecasts.duckdb --gameweek-dir /work/gameweeks \
  --artifact /work/ridge.json --predictions /work/predictions.json \
  --manifest /work/manifest.json
```

The permission change applies only to this disposable ignored scratch directory and lets the fixed
unprivileged container UID write the bind-mounted files. The service mount below is read-only.

Start the same read-only local service in the foreground:

```bash
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "${PWD}/scratch/container:/work:ro" \
  --entrypoint fpl-forecast-service fpl-forecast \
  --artifact /work/ridge.json --gameweek-dir /work/gameweeks \
  --database /work/forecasts.duckdb --season 2025-26 \
  --expected-as-of-gw 10 --completion-status completed \
  --host 0.0.0.0 --port 8000
```

The port is published to host loopback only. CI repeats the synthetic train, predict, store, and
health endpoint without downloading FPL data.
