args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: Rscript r/validate_results.R <evidence-dir> <output-json>")
}

evidence_dir <- args[[1]]
output_path <- args[[2]]
metrics <- read.csv(file.path(evidence_dir, "area-metrics.csv"), stringsAsFactors = FALSE)
edges <- read.csv(file.path(evidence_dir, "spatial-edges.csv"), stringsAsFactors = FALSE)
python_coefficients <- read.csv(
  file.path(evidence_dir, "model-coefficients.csv"), stringsAsFactors = FALSE
)
result_lines <- readLines(file.path(evidence_dir, "results.json"), warn = FALSE)

population_scale <- function(values) {
  (values - mean(values)) / sqrt(mean((values - mean(values))^2))
}

metrics$log_accessibility_5km <- population_scale(log1p(metrics$accessibility_5km))
metrics$home_or_no_fixed_share <- population_scale(metrics$home_or_no_fixed_share)

fitted <- glm(
  local_fixed ~ log_accessibility_5km + home_or_no_fixed_share,
  family = poisson(link = "log"),
  offset = log(fixed_workplace),
  data = metrics
)

codes <- metrics$area_code
positions <- setNames(seq_along(codes), codes)
weights <- matrix(0, nrow = length(codes), ncol = length(codes))
for (row in seq_len(nrow(edges))) {
  left <- positions[[edges$area_code_a[[row]]]]
  right <- positions[[edges$area_code_b[[row]]]]
  weights[left, right] <- 1
  weights[right, left] <- 1
}
degrees <- rowSums(weights)
if (any(degrees == 0)) {
  stop("spatial weights contain islands")
}
weights <- weights / degrees
centred <- metrics$local_retention_share - mean(metrics$local_retention_share)
moran <- (length(codes) / sum(weights)) *
  (sum(weights * outer(centred, centred)) / sum(centred^2))
moran_line <- grep('"statistic":', result_lines, value = TRUE)
if (length(moran_line) != 1) {
  stop("retained results must contain one Moran statistic")
}
python_moran <- as.numeric(sub('.*: *([-+0-9.eE]+),? *$', '\\1', moran_line))
moran_difference <- abs(moran - python_moran)

coefficient_names <- c("const", "log_accessibility_5km", "home_or_no_fixed_share")
r_estimates <- unname(coef(fitted))
python_poisson <- python_coefficients[python_coefficients$family == "poisson", ]
python_estimates <- python_poisson$estimate[match(coefficient_names, python_poisson$term)]
maximum_difference <- max(abs(r_estimates - python_estimates))
passed <- is.finite(moran) && moran_difference <= 1e-12 && maximum_difference <= 1e-7

json_lines <- c(
  "{",
  sprintf("  \"contract\": \"liverpool-r-validation-v1\","),
  sprintf("  \"areas\": %d,", nrow(metrics)),
  sprintf("  \"moran_statistic\": %.15g,", moran),
  sprintf("  \"moran_difference\": %.15g,", moran_difference),
  sprintf("  \"maximum_coefficient_difference\": %.15g,", maximum_difference),
  sprintf("  \"passed\": %s,", ifelse(passed, "true", "false")),
  sprintf("  \"r_version\": \"%s\"", R.version.string),
  "}"
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
writeLines(json_lines, output_path, useBytes = TRUE)
if (!passed) {
  quit(status = 1)
}
