# Findings

## City-level pattern

The retained release represents 205,768 people across workplace indicators 1 to 3. Of 123,689 flows to
a fixed UK workplace, 84,567 remained inside Liverpool: a city-wide local-retention share of 68.37%.
This describes the recorded Census 2021 pattern; it is not a target, service level, or present-day
commuting estimate.

## Spatial structure

Local-retention shares show a visible and measurable geographic pattern. Moran's I is 0.4901 under
row-standardized Queen contiguity, with a fixed-seed, two-sided permutation p-value of 0.0001 from
9,999 permutations. Neighbouring MSOAs therefore had more similar retention values than expected
under random spatial allocation of the observed shares.

This result does not explain why the pattern exists. Housing, occupation, transport, land use,
workplace composition, reporting conditions and the exceptional Census date may all contribute.

## Model diagnosis

The initial Poisson model has Pearson dispersion 4.7468, contradicting its equidispersion assumption.
The retained NB2 sensitivity fit converged with `alpha = 0.00250` and reduced AIC from 836.18 to 711.85
on the same rows. That is evidence that the NB2 likelihood describes these area counts better in
sample; it is not held-out performance or evidence of causal effects.

The bounded binomial sensitivity produces valid area probabilities and reaches the same broad
direction for the accessibility association. Its role is to check that the count-model narrative is
not an artifact of allowing an unbounded rate.

The 5 km accessibility coefficient is positive under all three specifications. Because destination
flow mass is built from the same OD release and includes each origin's own recorded flows, the
coefficient is retained for methodological transparency but is not promoted as independent
predictive evidence.
