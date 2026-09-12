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
section_number <- function(section, key) {
  start <- grep(sprintf('^  "%s": \\{', section), result_lines)
  if (length(start) != 1) stop("missing or duplicate result section")
  remaining <- result_lines[(start + 1):length(result_lines)]
  end <- grep('^  \\}', remaining)[[1]]
  field <- grep(sprintf('"%s":', key), remaining[seq_len(end)], value = TRUE)
  if (length(field) != 1) stop("missing or duplicate result number")
  value <- as.numeric(sub('.*: *([-+0-9.eE]+),? *$', '\\1', field))
  if (!is.finite(value)) stop("non-finite result number")
  value
}
python_moran <- section_number("moran", "statistic")
moran_difference <- abs(moran - python_moran)

coefficient_names <- c("const", "log_accessibility_5km", "home_or_no_fixed_share")
r_estimates <- unname(coef(fitted))
python_poisson <- python_coefficients[python_coefficients$family == "poisson", ]
python_estimates <- python_poisson$estimate[match(coefficient_names, python_poisson$term)]
maximum_difference <- max(abs(r_estimates - python_estimates))

# Grouped counts retain the MSOA as the observational unit. Construct the
# sandwich explicitly in base R, independently of statsmodels' HC0 routine.
binomial_fit <- glm(
  cbind(local_fixed, fixed_workplace - local_fixed) ~
    log_accessibility_5km + home_or_no_fixed_share,
  family = binomial(link = "logit"), data = metrics
)
x <- model.matrix(binomial_fit)
probability <- fitted(binomial_fit)
trials <- metrics$fixed_workplace
score <- x * as.vector(metrics$local_fixed - trials * probability)
bread <- solve(crossprod(x, x * as.vector(trials * probability * (1 - probability))))
robust_covariance <- bread %*% crossprod(score) %*% bread
binomial_se <- sqrt(diag(robust_covariance))
python_binomial <- python_coefficients[python_coefficients$family == "binomial_sensitivity", ]
aligned <- match(coefficient_names, python_binomial$term)
binomial_estimate_difference <- max(abs(unname(coef(binomial_fit)) - python_binomial$estimate[aligned]))
binomial_se_difference <- max(abs(binomial_se - python_binomial$robust_se[aligned]))
residual <- residuals(fitted, type = "pearson")
residual <- residual - mean(residual)
residual_moran <- (length(codes) / sum(weights)) *
  (sum(weights * outer(residual, residual)) / sum(residual^2))
residual_moran_difference <- abs(residual_moran - section_number("residual_moran", "statistic"))
differences <- c(moran_difference, maximum_difference, binomial_estimate_difference,
                 binomial_se_difference, residual_moran_difference)
passed <- all(is.finite(differences)) && fitted$converged && binomial_fit$converged &&
  moran_difference <= 1e-12 && maximum_difference <= 1e-7 &&
  binomial_estimate_difference <= 1e-7 && binomial_se_difference <= 1e-7 &&
  residual_moran_difference <= 1e-10

json_lines <- c(
  "{",
  sprintf("  \"contract\": \"liverpool-r-validation-v2\","),
  sprintf("  \"areas\": %d,", nrow(metrics)),
  sprintf("  \"moran_statistic\": %.15g,", moran),
  sprintf("  \"moran_difference\": %.15g,", moran_difference),
  sprintf("  \"maximum_coefficient_difference\": %.15g,", maximum_difference),
  sprintf("  \"binomial_maximum_coefficient_difference\": %.15g,", binomial_estimate_difference),
  sprintf("  \"binomial_maximum_robust_se_difference\": %.15g,", binomial_se_difference),
  sprintf("  \"binomial_accessibility_robust_se\": %.15g,", binomial_se[[2]]),
  sprintf("  \"residual_moran_statistic\": %.15g,", residual_moran),
  sprintf("  \"residual_moran_difference\": %.15g,", residual_moran_difference),
  sprintf("  \"passed\": %s,", ifelse(passed, "true", "false")),
  sprintf("  \"r_version\": \"%s\"", R.version.string),
  "}"
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
writeLines(json_lines, output_path, useBytes = TRUE)
if (!passed) {
  quit(status = 1)
}
