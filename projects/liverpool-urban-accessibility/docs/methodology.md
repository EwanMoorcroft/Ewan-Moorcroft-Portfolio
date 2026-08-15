# Methodology

## Questions

1. What share of fixed-workplace flows from each Liverpool MSOA remained within Liverpool?
2. How did local retention vary spatially across the city?
3. How was retention associated with a gravity-style measure of proximity to Liverpool workplace
   destinations after accounting for the origin's fixed-workplace exposure?

The questions are descriptive. They do not estimate individual accessibility or a causal transport
effect.

## Flow reconciliation

The DuckDB transformation scans the national MSOA member once per fixed query. Liverpool origins are
joined to the exact boundary code set. Indicator `3` is divided into same-area, other-Liverpool and
outside-Liverpool destinations. Each row must satisfy:

```text
fixed workplace = local fixed workplace + outside-Liverpool fixed workplace
same area <= local fixed workplace
```

Indicators `1` and `2` remain explicit. Indicator `-8` is excluded from employed metrics. The project
does not turn absent sparse OD edges into modelled zeros.

## Spatial accessibility

Official population-weighted centroids are transformed from EPSG:4326 to British National Grid
(EPSG:27700) through the fixed EPSG operation 1314 Helmert pipeline. This avoids platform-dependent
selection of an optional grid transformation. Pairwise straight-line distances connect all Liverpool
MSOA centroids. For origin `i`, decay distance `d`, destination fixed-workplace flow mass `J`, and
distance `x`, the index is:

```text
A_i(d) = sum_j J_j exp(-x_ij / d), d in {3 km, 5 km, 10 km}
```

`J` is the total fixed-workplace inflow to each Liverpool destination from the same Census release.
The index is therefore an internally consistent descriptive opportunity measure, not an externally
validated job-access score. Population-weighted centroids reduce, but do not remove, within-area
aggregation error.

## Spatial autocorrelation

Two areas are neighbours when their 2021 polygon boundaries touch at an edge or vertex. The graph is
made symmetric, self-links are forbidden, and islands fail the contract. Weights are row-standardized.
Moran's I is calculated explicitly from centred area retention shares. The two-sided pseudo-p-value
uses 9,999 fixed-seed permutations and compares distance from the randomization expectation
`-1 / (n - 1)`.

## Count models

The Poisson and NB2 models use the same 61 areas, local fixed-workplace count as the outcome, and the log of all
fixed-workplace workers from the origin as an exposure offset. Predictors are population-standardized
`log1p` 5 km accessibility and the combined indicator-1 share.

The primary Poisson diagnostic is Pearson chi-square divided by residual degrees of freedom. Its
retained value of 4.7468 shows material overdispersion. An NB2 sensitivity model estimates dispersion
`alpha` rather than assuming Poisson equality of mean and variance. AIC comparison is valid only
because the observation set and design are identical. HC0 robust standard errors are retained, but
coefficient p-values are not headline evidence because areas remain spatially dependent and the
destination-flow mass comes from the same release.

Because local fixed-workplace count is bounded by the fixed-workplace exposure, a binomial GLM also
models the retained local proportion with frequency weights equal to that exposure. This is a
structural sensitivity check; its likelihood is not compared with the Poisson or NB2 AIC because the
response likelihood differs.

## Cross-language validation

The base-R script reads the retained metrics and edge table, recreates population standardization,
refits the Poisson GLM with its offset, and calculates observed Moran's I from the row-standardized
weight matrix. The retained check passed with a maximum coefficient difference of `1.05e-12`.
