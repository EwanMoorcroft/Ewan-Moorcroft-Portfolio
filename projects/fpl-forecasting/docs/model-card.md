# Ridge Forecast Model Card

## Purpose

Estimate individual player points for the next gameweek from recent and season-to-date completed-gameweek statistics.

## Intended use

- Compare a regularized linear forecast with simple recent-form baselines.
- Produce ranked player shortlists for further human review.
- Save fitted coefficients and preprocessing values in a portable JSON artifact.
- Run a season-scoped batch forecast from a caller-verified completed snapshot sequence.

## Out of scope

- Full-squad optimization under budget and position constraints.
- Transfer, captaincy, bench, or chip decisions.
- Real-time injury, suspension, fixture, opponent, and predicted-line-up signals.
- Guaranteed points or financial advice.

## Training data

One row represents a player at the end of gameweek `t`, with the observed target from completed
gameweek `t+1`. The exact wrapper records season and gameweek; raw live payloads are rejected.
Empty placeholders are also rejected. The wrapper is a caller assertion rather than proof of
official completion; see the [completion boundary](data-contract.md#completion-boundary). No real
data ships with the repository.

## Model

The estimator is ridge regression after per-feature standardization learned only from the active training window. Regularization strength is explicit in the configuration and artifact.

## Evaluation

Evaluation uses expanding train windows and later non-overlapping test gameweeks. Three baselines are reported alongside ridge. Point-error and within-gameweek ranking measures are both required.

## Known limits

The features omit fixture context, player availability, price history, team changes, and position
because the accepted input does not provide trustworthy historical as-of values for them. Player
performance and scoring rules can drift. A result from one period should not be assumed to transfer
to another without a fresh chronological run. No built-in feature-shift diagnostic is implemented.
This is a ranking aid, not a guarantee of points or squad value.

## Artifact format and checks

Artifacts are JSON numeric parameters with a versioned format. A saved model artifact includes the
season, effective-configuration SHA-256, semantic feature-schema version and SHA-256, training row
count, and final target gameweek. Prediction rejects a mismatch before scoring. The project does not
load pickle or joblib content. Generated artifacts are ignored by Git.
