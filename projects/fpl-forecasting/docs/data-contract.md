# Completed Gameweek Data Contract

## Accepted files

The reader accepts JSON filenames containing `gameweek-<positive integer>`, followed by either a hyphen or the file extension. Examples:

- `gameweek-7.json`
- `gameweek-07.json`
- `gameweek-7-20260101T120000Z.json`

Exactly one file may claim each gameweek.

## Required shape

The top-level value must be an object with a non-empty `elements` list. Every element must contain:

- a positive integer `id`;
- a `stats` object;
- finite numeric `total_points`;
- finite, non-negative numeric `minutes`.

The reader also accepts `starts`, `ict_index`, `influence`, `creativity`, `threat`, `goals_scored`, `assists`, `clean_sheets`, `saves`, and `bonus`. Missing optional values become zero. Non-negative fields cannot contain negative values. Negative `total_points` remains valid because FPL scoring can produce it.

## Completion boundary

The accepted live-gameweek payload does not contain an event-completion flag. Each file is treated
as caller-declared completed data, so official event status must be checked when the snapshot is
captured. An empty `elements` list is rejected as an obvious placeholder; a non-empty in-progress
snapshot cannot be proven complete from this file alone.

## Sequence checks

Before feature construction, the complete directory is checked for:

| Check | Reason |
|---|---|
| At least two completed gameweeks | One is needed for features and one for a target. |
| No empty `elements` list | An obvious future placeholder is not usable result data. |
| No duplicate gameweek number | File selection must be unambiguous. |
| Consecutive gameweeks | Silent gaps change rolling windows and target meaning. |
| Unique player IDs within each file | Duplicate rows would double-count one player. |
| Adjacent player coverage | A sharp roster mismatch usually indicates the wrong source or season. |

Adjacent coverage is `overlapping player IDs / larger roster size`. The default minimum is 0.90.

## Target construction

For adjacent completed gameweeks `t` and `t+1`, only players found in both files are eligible. The row uses statistics no later than `t`; its target is that player's `total_points` in `t+1`.

The last supplied gameweek contributes targets to the preceding gameweek but does not create an unlabeled prediction row. The builder does not generate rows for later gameweeks and does not replace an empty snapshot with zeros.

## Storage boundary

Real downloaded data should remain under an ignored local path such as `data/raw/`. Commit a small synthetic fixture only when it is clearly labeled and contains no copied live-season records.

## Historical CSV adapter

The optional `import-vaastav` command converts an explicit gameweek interval from a locally
extracted, commit-pinned Vaastav historical snapshot into this JSON contract. It requires a full
40-character source revision and records a SHA-256 for every input CSV.

Only `GK`, `DEF`, `MID`, and `FWD` rows are retained. Rows with the `AM` position are excluded because
the forecast target is player points. When a double gameweek contains multiple fixture rows for one
player, the retained numeric gameweek statistics are summed to match the event-level shape. Missing
players remain missing; the adapter never adds zero records. The normal sequence, coverage, feature,
and temporal-boundary checks run after conversion without exceptions or relaxed thresholds.
