# FPL Next-Gameweek Forecasting

A compact forecasting project built around one rule: every reported prediction must be made with information that existed before its target gameweek.

[Open the project overview](demo/index.html) for a quick visual tour.

## What it does

The pipeline reads completed Fantasy Premier League live-gameweek JSON files, validates their sequence, creates strictly as-of player features, and predicts points in the next completed gameweek. It compares standardized ridge regression with three transparent baselines under an expanding-window evaluation.

The repository contains no downloaded match data and no trained model. A deterministic synthetic generator supports local checks and examples without network access.

## Evidence controls

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
- three-gameweek means for points, minutes, ICT, goals, and assists;
- three-gameweek start rate;
- five-gameweek points mean;
- season-to-date points, minutes, appearances, and points per appearance.

Player ID and gameweek fields remain metadata and are never model inputs. Team, position, current price, fixture difficulty, and availability news are deliberately absent because historical as-of versions are not provided by the live-gameweek files.

## Evaluation

For each fold, all training target gameweeks precede all test target gameweeks. The window expands after each test period. The candidates are:

1. last observed gameweek points;
2. mean points over the last three completed gameweeks;
3. mean target from the current training window;
4. standardized ridge regression.

Point error alone does not describe recommendation quality, so ranking metrics are calculated separately within each target gameweek and then averaged. Full formulas and edge-case rules are in [the methodology note](docs/methodology.md).

## Scope and limitations

This project forecasts individual next-gameweek points; it does not optimize a full squad, transfers, captaincy, budget, or bench order. It does not include fixture schedules, opponent strength, injury reports, predicted line-ups, or historical price snapshots. Results depend on the supplied season and must be re-evaluated after material rule or data changes.

Synthetic results demonstrate plumbing and protocol behavior only. They must not be presented as live-season performance. Forecasts are uncertain and should not be treated as a guarantee.

## Project map

```text
config/default.json          Protocol settings
demo/index.html              Self-contained visual overview
docs/                        Data, evaluation, and model notes
src/fpl_forecasting/         Validation, features, splits, metrics, model, CLI
tests/                       Deterministic contract and end-to-end checks
```

The tested dependency set is pinned in `requirements.txt` and `requirements-dev.txt`. Runtime behavior is also declared in `pyproject.toml`.

## Container smoke check

The optional image uses a slim Python runtime and runs the CLI as an unprivileged user. Build it locally, then generate the same deterministic fixture on every run:

```bash
docker build -t fpl-forecast .
docker run --rm fpl-forecast synthetic --output-dir /tmp/gameweeks --gameweeks 8 --players 12 --seed 42
```

This is a packaging smoke check, not a hosted-service claim. The temporary gameweek files are removed with the container.
