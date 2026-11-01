# What this evidence can and cannot support

**Intended use:** compare observed neighbourhood characteristics and organise further adaptation investigation. **Unsupported uses:** predicting individual harm, pricing insurance, allocating statutory funding, certifying safe land, or claiming an intervention's financial or health benefit.

| Limitation | Consequence and treatment |
| --- | --- |
| MAUP and within-area variation | LSOA boundaries and aggregation influence ranks. Small hot surfaces, barriers and at-risk buildings may be concealed. Exact fractional raster aggregation improves measurement, not this conceptual limitation. No alternate-geography stability claim is made. |
| Ecological inference | Area income/age shares do not identify the circumstances of individuals. Population counts mean residents of an area, not exposed or harmed people. |
| Incomplete heat evidence | Unassessed LSOAs are explicitly missing from the composite and statistical comparison. Their omission changes the cohort and may create spatial islands; grey does not mean low risk. |
| Thermal sampling | Five selected cloud-screened daytime dates are not a long-term, night-time, indoor or air-temperature series. Clouds, land-cover/emissivity error and a 31% shared reference footprint can bias comparisons. |
| Raster resolution | A 30 m distributed thermal grid does not establish 30 m effective thermal detail. The 100 m analytical grid still smooths local variability. NDVI is contextual and has a different observation-completeness rule. |
| Partial flood representation | Defended river/sea extents omit surface water, defence failure and climate-change allowances. Flood area fraction is not depth, expected annual damage or population exposure. No building or critical-infrastructure exposure is claimed. |
| Mixed temporal vintages | Population 2021, selected thermal dates 2021–2025, mixed-period IoD2025, official access 2024 corrected 2025, and 2026 flood/OS/OSM data do not form one contemporaneous survey. |
| Walking-network completeness | OSM paths/access tags, assumed off-network connectors and OS entrances can be incomplete. Snapping does not prove a traversable connection through a wall or crossing. Graph disconnection means modelled inaccessibility, not confirmed real-world exclusion. |
| Mobility and greenspace quality | The route model omits slope, crossing delays, pavement conditions, wheelchair access, safety, opening hours and park quality. A universal walking threshold cannot represent everyone. |
| Buffer boundary | A 2 km buffer supports the 0.5–1.5 km threshold analysis; very long reported nearest-park distances may be truncated by the downloaded network extent. |
| Different access measures | Defra household access and the OA resident/public-park case model have different denominators, destinations, networks and vintages. They cannot be treated as a time series or interchangeable measurements. |
| Normative model choices | Indicator set, normalisation, additive compensation, weights, priority cutoff and robustness threshold are declared analyst choices. No stakeholder elicitation has taken place. |
| Sensitivity is conditional | Weight experiments, alternate normalisation, indicator omission and a flood alternative cover specified assumptions only. They do not propagate all observational uncertainty or prove the shortlist is correct. |
| Spatial significance | Global association is conditional on the cohort and contiguity graph. Neither a small permutation p-value nor descriptive HH quadrants establishes a mechanism or local statistically significant hotspot. |
| Operational validity | The workflow is tested on one macOS arm64 QGIS 4.2.2 installation. Other platforms/QGIS versions are documented routes, not verified deployments. Upstream services and licences must be rechecked for a new data release. |

The work contains no trained predictive model because an appropriate target and independent validation set are absent. The data-science contribution is reproducible acquisition, measurement design, statistical diagnostics, scenario modelling, uncertainty analysis and tested numerical/spatial code. A future predictive extension should begin with a defensible observed outcome, time-aware and spatially separated validation, and a baseline that it must improve.

Before acting on the shortlist: verify local flood and overheating evidence, audit routes and entrances, consult residents and service teams, evaluate alternative interventions, and appraise feasibility and costs. Those steps are outside this simulated commission.
