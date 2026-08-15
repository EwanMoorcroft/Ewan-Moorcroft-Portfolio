# Geography and data analysis background

I completed a BSc in Geography at the University of Liverpool in June 2025. I am now studying for
an MSc in Data Science and Artificial Intelligence there, with completion expected in September
2026. Geography is still a large part of how I approach data: I look for the spatial unit, the scale,
the way a measurement was collected, and what might have been lost when the real world became a
table.

## River Soar research

My undergraduate research examined how riparian buffers and surrounding land use related to
nutrient conditions in the River Soar. I collected and organised observations from six sites over 20
days, then combined them with river-height data used as a proxy for rainfall occurrence.

The work included field planning, repeated sampling, data cleaning, maps, charts, correlation and
group comparisons. It also forced a useful limitation into the analysis: local rainfall observations
were unavailable, so the proxy could support a cautious comparison but not a precise rainfall model.
The final report had to connect those analytical choices to practical land-management questions for
a non-specialist reader.

The raw observations and exact site locations are not public here. This is a summary of the work, not
a repackaged software project.

## GIS work

I have used QGIS for cartography, coordinate reference systems, buffers, overlays, catchments, and
choropleth maps. In Python I have worked with GeoPandas, Rasterio, Rasterstats, and PySAL; in R, with
`sf`, `tmap`, and `dplyr`. The exercises covered spatial joins, raster masking, zonal statistics,
administrative boundaries, census data, transport flows, and OpenStreetMap-derived information.

Two Liverpool analyses are worth separating because they answered different questions. One looked
at retail site-selection evidence. The other used commuting origin-destination flows, Moran's I, and
Poisson regression to examine urban movement. Their original files mixed my work with teaching
scaffolding and data whose redistribution terms were incomplete, so I did not copy them into Git.
Instead, I rebuilt the stronger ideas with official public sources in
[Liverpool urban accessibility](../projects/liverpool-urban-accessibility/).

## Statistics, databases, and presentation

My wider work includes exploratory analysis, environmental time series, hypothesis tests, analysis
of variance, regression, clustering, and data visualisation. Database study covered relational
design, joins, aggregation, nested queries, keys, constraints, transactions, and locking. The public
Liverpool project puts some of that into practice through a 197.7 MB Census flow
table transformed with DuckDB, followed by spatial analysis in Python and an independent check in R.

I have deliberately kept this page concrete. It records skills I have used, but it does not pretend
that every university analysis is independently reproducible from this repository. Reports, teaching
material, raw field data, private identifiers, and third-party datasets without clear redistribution
rights remain outside the public portfolio.
