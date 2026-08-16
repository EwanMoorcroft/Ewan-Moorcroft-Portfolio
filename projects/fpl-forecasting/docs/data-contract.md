# Completed Gameweek Data Contract

## Accepted files

The reader accepts JSON filenames containing `gameweek-<positive integer>`, followed by either a hyphen or the file extension. Examples:

- `gameweek-7.json`
- `gameweek-07.json`
- `gameweek-7-20260101T120000Z.json`

Exactly one file may claim each gameweek.

The filename gameweek must equal the wrapper gameweek. Every file in a sequence must use the same
canonical `YYYY-YY` season, such as `2025-26`. Operational commands check that season against the
caller request and fitted artifact.

## Required shape

The top-level object must contain exactly:

- `snapshot_format` with value `fpl-completed-gameweek-v1`;
- `season_id` as a canonical consecutive season;
- `gameweek` as a positive integer matching the filename; and
- a non-empty `elements` list.

Raw public FPL live payloads contain `elements` but do not carry this season and gameweek wrapper,
so they are rejected. A caller must verify the source event and convert it explicitly. The
`synthetic` and `import-vaastav` commands write the accepted wrapper.

Every element must contain:

- a positive integer `id`;
- a `stats` object;
- finite numeric `total_points`;
- finite, non-negative numeric `minutes`.

The reader also accepts `starts`, `ict_index`, `influence`, `creativity`, `threat`, `goals_scored`, `assists`, `clean_sheets`, `saves`, and `bonus`. Missing optional values become zero. Non-negative fields cannot contain negative values. Negative `total_points` remains valid because FPL scoring can produce it.

## Completion boundary

The raw live-gameweek payload does not contain an event-completion flag. The accepted wrapper names
the file as completed, but that remains a caller assertion. Official event status must be checked
before conversion. An empty `elements` list is rejected as an obvious placeholder; the wrapper
cannot prove that a non-empty source payload was captured after completion.

Operational prediction adds two fail-closed declarations. The caller must provide the expected
latest gameweek, which must equal the latest supplied filename, and must pass the literal completion
status `completed`. These checks prevent accidental stale or future boundaries, but they do not
query the official FPL event status.

## Sequence checks

Before feature construction, the complete directory is checked for:

| Check | Reason |
|---|---|
| At least two completed gameweeks | One is needed for features and one for a target. |
| Exact wrapper fields and format | Season and gameweek identity must be explicit. |
| One season across the sequence | Cross-season windows are not meaningful. |
| Wrapper gameweek matches filename | File selection and content identity must agree. |
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

## Operational output identity

The prediction command hashes the exact artifact bytes and each exact input file. The ordered input
aggregate includes gameweek, file name, byte count, and file SHA-256. The accompanying manifest also
binds the effective protocol configuration, semantic feature schema, and canonical prediction JSON.
Outputs cannot overwrite an input snapshot, configuration, or artifact.

The manifest intentionally omits wall-clock time and machine-local paths. Retrieval time and proof
that an event was officially complete belong in the caller's acquisition log.

## Pinned CSV adapter

The optional `import-vaastav` command converts an explicit gameweek interval from a locally
extracted, commit-pinned Vaastav season snapshot into this JSON contract. It requires a full
40-character source revision and records a SHA-256 for every input CSV.

Only `GK`, `DEF`, `MID`, and `FWD` rows are retained. Rows with the `AM` position are excluded because
the forecast target is player points. When a double gameweek contains multiple fixture rows for one
player, the retained numeric gameweek statistics are summed to match the event-level shape. Missing
players remain missing; the adapter never adds zero records. The normal sequence, coverage, feature,
and temporal-boundary checks run after conversion without exceptions or relaxed thresholds.
