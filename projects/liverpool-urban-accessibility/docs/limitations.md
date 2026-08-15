# Limitations

- Census Day was 21 March 2021, during pandemic restrictions, furlough and unusual working patterns.
  The figures are not current travel demand.
- ONS combines mainly working at or from home with having no fixed place of work. This project does
  not relabel that combined category as a home-working rate.
- Results are ecological area summaries. They do not describe every resident and are vulnerable to
  the modifiable areal unit problem.
- Population-weighted centroids are representative points; straight-line centroid distance is not
  route distance, time, cost, reliability, accessibility need, or journey quality.
- The accessibility index weights destination flows from employed people in the same OD release. It
  includes each origin's own flows, is not an independent job-stock measure, excludes Scotland and
  Northern Ireland origins, and is not an external predictor.
- Sparse OD data contain no published zero rows. Missing pairs are not assumed to be true zeros.
- Cell-key perturbation protects confidentiality and can create small inconsistencies between
  separately tabulated Census products.
- Origins in Scotland and Northern Ireland are not present in the England and Wales source even when
  destinations are represented.
- Queen contiguity is only one spatial relationship; it does not encode transport-network connection.
- Spatial dependence remains in model rows. Robust standard errors do not make the findings causal.
- AIC is an in-sample likelihood comparison, not evidence of deployment or unseen-area performance.
