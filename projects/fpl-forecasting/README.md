# FPL Next-Gameweek Forecasting

Fantasy Premier League data makes temporal leakage easy: a row can look valid while containing
information that was not available when the decision would have been made. This project predicts a
player's next-gameweek points and treats that time boundary as part of the data contract.

[Open the project overview](demo/index.html) for a quick visual tour.

## Question and approach

The pipeline reads completed Fantasy Premier League live-gameweek JSON files, validates their
sequence, creates strictly as-of player features, and predicts points in the next completed
gameweek. Standardized ridge regression is compared with three simple baselines under an
expanding-window evaluation.

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
- Saved ridge artifacts are plain JSON, not executable pickle files.

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

The upstream repository uses an MIT licence for its software, while its licence file states that
the underlying data belong to the named data providers. This repository therefore retains no
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
  --config config/default.json \
  --artifact models/ridge.json
```

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

Use one file per completed gameweek, named `gameweek-<number>.json` or `gameweek-<number>-<label>.json`. Each file must follow the public live-gameweek shape:

```json
{
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

The live-gameweek shape has no event-completion flag. Files are therefore treated as
caller-declared completed snapshots: confirm the event's official finished status before saving
each file. The validator rejects empty placeholders, but a non-empty in-progress snapshot cannot
be distinguished from a finished one using this payload alone.

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
docs/                        Data, evaluation, and model notes
reports/retained/            Compact metrics and source audit evidence
src/fpl_forecasting/         Validation, features, splits, metrics, model, CLI
tests/                       Deterministic contract and end-to-end checks
```

The Docker build pins the three direct runtime dependencies in `requirements.txt`. CI uses
`requirements-dev.txt`, which adds pinned test tools. These files are not a hash-locked record of
every transitive package; compatible ranges remain in `pyproject.toml` for ordinary installation.

## Docker

The optional image uses a slim Python runtime and runs the CLI as an unprivileged user. Build it locally, then generate the same deterministic fixture on every run:

```bash
docker build -t fpl-forecast .
docker run --rm fpl-forecast synthetic --output-dir /tmp/gameweeks --gameweeks 8 --players 12 --seed 42
```

This is a packaging smoke check, not a hosted-service claim. The temporary gameweek files are removed with the container.
