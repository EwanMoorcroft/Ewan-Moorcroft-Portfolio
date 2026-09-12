# Area-level binomial uncertainty correction

The corrected `liverpool-commuting-access-v2` analysis treats each MSOA as one observation with
grouped local and outside-Liverpool worker counts. The earlier proportion model used frequency
weights equal to worker counts. Its HC0 calculation therefore repeated each area's proportion as
if it were independent worker-level information, understating uncertainty.

For standardized log accessibility, the estimate remains 0.0481494. The robust standard error
changes from 0.000536196 to 0.0261129, about 49 times larger. The corrected 95% interval is
[-0.00303089, 0.0993296], with p = 0.06520. The positive point estimate does not establish a precise
non-zero association. Grouped-binomial log likelihood also changes because the response is now
encoded as successes and failures; it should not be compared with the old weighted-proportion
likelihood or with the count-model AIC values.

Base R independently fits grouped counts and constructs the area-level sandwich from the design
matrix, trial counts and area residual scores. The maximum robust-standard-error difference from
Python is 2.58e-13. The original pack is preserved under
[`reports/superseded/`](../reports/superseded/README.md).

The city totals, observed Moran statistic and Poisson/NB2 estimates are unchanged. A new descriptive
check finds Moran's I = 0.4484 in the Poisson Pearson residuals. Ordinary HC0 errors do not account
for this spatial dependence. No fitted-residual permutation p-value or causal interpretation is
claimed. The corrected interval remains descriptive rather than spatially adjusted inference.

The verifier now rejects non-finite claims, checks result metadata, reconstructs all 151 Queen edges
from retained polygons, and rebuilds all 183 accessibility values from population-weighted centroids
and destination-flow mass. It does not reconstruct national OD aggregation or mean commute distance
without the original OD flows.
