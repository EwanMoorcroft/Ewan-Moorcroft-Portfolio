# Independent R validation

The Python-derived area table and spatial edge list are deliberately small enough for an
independent base-R check. The script refits the fixed Poisson model and reconstructs observed
Moran's I without downloading data or requiring additional R packages.

```bash
Rscript r/validate_results.R reports/retained reports/generated/r-validation.json
```

The comparison uses a `1e-7` coefficient tolerance and a `1e-12` observed-Moran tolerance. Python
remains the primary implementation; the R result is a cross-language numerical check, not a
separate source of headline findings.
