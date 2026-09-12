# Independent R validation

The Python-derived area table and spatial edge list are small enough for an
independent base-R check. The script refits Poisson and grouped-binomial models, constructs the
area-level HC0 sandwich directly, and reconstructs observed and residual Moran statistics without
downloading data or requiring additional R packages.

```bash
Rscript r/validate_results.R reports/retained reports/generated/r-validation.json
```

The comparison uses `1e-7` coefficient and robust-SE tolerances, a `1e-12` observed-Moran tolerance,
and a `1e-10` residual-Moran tolerance. Python
remains the primary implementation; the R result is a cross-language numerical check, not a
separate source of headline findings.
