# How results are labelled

A score is only useful when the reader can tell which data, split, and selection rule produced it.
The short labels below stop an older run from looking equivalent to a result that can be checked from
the files in this repository.

## Current verified

The inputs needed for aggregate checking are present and their hashes match. Included software can
recalculate the published values. Tree LiDAR is the only project using this label: its verifier checks
seven files, 12 method/route identities, every held-out plot, and 1,152 aggregate values.

## Historical reference

An artifact from an earlier run exists, but the revised pipeline has not regenerated its score. The
X-ray, LunarLander, and neural-chunking projects use this label. Their READMEs give the original scope
and explain what is missing, so the number cannot be mistaken for a fresh result.

## Fresh historical evaluation

The current method has been run on a completed, commit-pinned public dataset. FPL uses this label for
gameweeks 1 to 15 of the 2024-25 season. Its source manifest and aggregate out-of-fold report are in
Git, but the raw rows, row-level predictions, and fitted model are not. The result says nothing about
a live season or the rest of that completed season.

## Fresh public-data analysis

The analysis has been run on dated public sources with licences and source identities recorded.
Liverpool uses this label. Compact transformed inputs and outputs are included, while the much larger
national Census table must be downloaded from its publisher. The verifier rebuilds the Python
aggregates and coefficients; a small R script checks the cross-language values.

## Rebuilt protocol

Tests and fictional fixtures exercise the corrected method, but no real-data performance claim is
attached. FPL's synthetic command is an example. Older results built from empty future files are not
included.

## Before a number is published

I check five things:

1. the exact data identity and split;
2. whether selection was kept away from the held-out data;
3. the metric definition and aggregation route;
4. whether the result is recalculated here, checked from smaller artifacts, or historical only; and
5. which limitations materially change its interpretation.

If an input or prediction is unavailable, that gap is stated. I do not reconstruct missing outputs,
invent patient identities, or rerun an expensive experiment merely to make an older number look new.

The [geography background](GEOSPATIAL_BACKGROUND.md) is different. It summarises work I have done but
does not attach a model-result label to source reports or data that cannot be redistributed.
