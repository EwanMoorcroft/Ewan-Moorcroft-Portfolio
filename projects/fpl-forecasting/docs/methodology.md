# Forecasting Methodology

## Forecast question

For player `p`, estimate FPL points in completed target gameweek `t+1` using statistics available through completed gameweek `t`.

This definition fixes both the prediction horizon and the information boundary. The code asserts `target_gw = as_of_gw + 1` for every row.

## Features

| Family | Fields | Availability |
|---|---|---|
| Latest gameweek | points, minutes, started, ICT, influence, creativity, threat | End of `t` |
| Latest three recorded player rows | mean points, mean minutes, start rate, mean ICT, mean goals, mean assists | No later than `t` |
| Latest five recorded player rows | mean points | No later than `t` |
| Season to date | points sum, minutes sum, appearances, points per appearance | No later than `t` |

At the beginning of a season, rolling windows use the history available so far. There is no padding
with future values. If a player is absent from a snapshot, that gameweek is not inserted as a zero
row; the window therefore means recent recorded observations rather than consecutive gameweeks. An
appearance is a recorded gameweek with more than zero minutes.

The feature list is constant and ordered. Training and prediction both select exactly that list. Any target, identifier, or extra future-facing field stays outside the model matrix. The deployment
contract also hashes each feature's type, window and value semantics, plus the as-of, population and
missing-observation meanings. Matching names alone are not sufficient.

## Expanding-window evaluation

Let the ordered target gameweeks be `g1 ... gn`. With a four-gameweek minimum training window and a one-gameweek test window, the first fold fits on `g1 ... g4` and predicts `g5`. The next fold fits on `g1 ... g5` and predicts `g6`. This continues until the final available target gameweek.

Test windows do not overlap. A row can appear in at most one out-of-fold result. The split code verifies that the largest training gameweek is less than the smallest test gameweek.

## Candidates

- **Last gameweek:** latest observed points.
- **Three-observation mean:** mean points over the available latest three player rows.
- **Training-window mean:** mean target among current training rows.
- **Ridge regression:** all approved features are standardized from the current training fold, then fitted with L2 regularization.

The baselines make it clear whether added model complexity improves on recent form and a broad central estimate.

## Metrics

Point metrics are pooled across all out-of-fold rows:

- **MAE:** mean absolute difference between observed and predicted points.
- **RMSE:** square root of mean squared error; larger misses receive more weight.
- **R²:** variation explained relative to the out-of-fold target mean.

Ranking metrics are calculated within each target gameweek, then averaged:

- **Spearman:** correlation between observed and predicted ranks, with average ranks for ties.
- **NDCG@K:** discounted gain of the predicted top K relative to the ideal ordering. Negative points receive zero relevance for this metric.
- **Top-K overlap:** share of the observed top-K player set recovered by the predicted top K.

Each target gameweek receives equal weight in the final ranking-metric mean, regardless of its row
count. When a target gameweek has fewer than K players, K is reduced to its player count. A
constant ranking receives Spearman zero. A gameweek with no positive relevance receives NDCG
zero. NDCG relevance is clipped to the interval 0 to 50 for numerical stability. NDCG and top-K
ties use stable input order; the pipeline sorts each gameweek by player ID before evaluation.

## Final fit

The `train` command fits ridge on every completed labeled row only after the rolling comparison has been reviewed. This final artifact has no attached performance claim; performance evidence comes from the out-of-fold report.

The artifact stores feature names, scaling values, coefficients, intercept, regularization strength,
row count, final target gameweek, canonical season, semantic schema identity, and effective
configuration identity as JSON. Loading requires the exact versioned field set, numeric parameter
types, positive integer training metadata, compatible vector lengths, fixed feature order, finite
values, and positive scales. Duplicate JSON keys are rejected.

## Operational batch scoring

The latest supplied completed gameweek is the as-of boundary. Every player present in that snapshot
gets one unlabeled row using only that season's observations through the boundary. The target
gameweek is exactly one greater. The artifact must have been trained through the same boundary and
must match the requested season, effective configuration, and semantic feature schema.

Predictions are sorted by predicted points descending and then player ID ascending. Canonical JSON
serialization and content hashes make an exact rerun distinguishable from a changed artifact,
snapshot set, configuration, schema, or output.
