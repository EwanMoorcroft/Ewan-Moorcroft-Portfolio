"""Area-level count modelling with explicit exposure and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

from .contracts import DataContractError, validate_metrics

PREDICTORS = ("log_accessibility_5km", "home_or_no_fixed_share")


@dataclass(frozen=True)
class CountModelResult:
    """Poisson association model and overdispersion diagnostic."""

    family: str
    coefficients: pd.DataFrame
    observations: int
    log_likelihood: float
    aic: float
    pearson_dispersion: float
    alpha: float | None
    converged: bool
    predicted_rate_min: float
    predicted_rate_max: float


def model_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create the fixed, finite observation set shared by all count-model comparisons."""
    validate_metrics(metrics)
    frame = metrics.loc[metrics["fixed_workplace"] > 0].copy()
    frame["log_accessibility_5km"] = np.log1p(frame["accessibility_5km"])
    for column in PREDICTORS:
        standard_deviation = float(frame[column].std(ddof=0))
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            raise DataContractError(f"model predictor is constant: {column}")
        frame[column] = (frame[column] - frame[column].mean()) / standard_deviation
    required = ["local_fixed", "fixed_workplace", *PREDICTORS]
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise DataContractError("model frame contains non-finite values")
    return frame


def fit_poisson(metrics: pd.DataFrame) -> CountModelResult:
    """Fit a Poisson rate model with robust uncertainty and worker-count exposure."""
    frame = model_frame(metrics)
    design = sm.add_constant(frame.loc[:, PREDICTORS], has_constant="add")
    fitted = sm.GLM(
        frame["local_fixed"],
        design,
        family=sm.families.Poisson(),
        offset=np.log(frame["fixed_workplace"]),
    ).fit(cov_type="HC0")
    converged = bool(getattr(fitted, "converged", False))
    if not converged:
        raise DataContractError("Poisson fit did not converge")
    degrees = max(int(fitted.df_resid), 1)
    dispersion = float(np.square(fitted.resid_pearson).sum() / degrees)
    confidence = fitted.conf_int()
    predicted_rate = fitted.fittedvalues.to_numpy(dtype=float) / frame["fixed_workplace"].to_numpy(
        dtype=float
    )
    coefficients = pd.DataFrame(
        {
            "term": fitted.params.index,
            "estimate": fitted.params.to_numpy(dtype=float),
            "robust_se": fitted.bse.to_numpy(dtype=float),
            "p_value": fitted.pvalues.to_numpy(dtype=float),
            "confidence_low": confidence.iloc[:, 0].to_numpy(dtype=float),
            "confidence_high": confidence.iloc[:, 1].to_numpy(dtype=float),
        }
    )
    return CountModelResult(
        family="poisson",
        coefficients=coefficients,
        observations=len(frame),
        log_likelihood=float(fitted.llf),
        aic=float(fitted.aic),
        pearson_dispersion=dispersion,
        alpha=None,
        converged=converged,
        predicted_rate_min=float(predicted_rate.min()),
        predicted_rate_max=float(predicted_rate.max()),
    )


def fit_negative_binomial(metrics: pd.DataFrame) -> CountModelResult:
    """Fit an NB2 sensitivity model on the exact Poisson observation set."""
    frame = model_frame(metrics)
    design = sm.add_constant(frame.loc[:, PREDICTORS], has_constant="add")
    fitted = NegativeBinomial(
        frame["local_fixed"],
        design,
        offset=np.log(frame["fixed_workplace"]),
        loglike_method="nb2",
    ).fit(disp=False, cov_type="HC0")
    confidence = fitted.conf_int()
    coefficients = pd.DataFrame(
        {
            "term": fitted.params.index,
            "estimate": fitted.params.to_numpy(dtype=float),
            "robust_se": fitted.bse.to_numpy(dtype=float),
            "p_value": fitted.pvalues.to_numpy(dtype=float),
            "confidence_low": confidence.iloc[:, 0].to_numpy(dtype=float),
            "confidence_high": confidence.iloc[:, 1].to_numpy(dtype=float),
        }
    )
    alpha = float(fitted.params["alpha"])
    if not np.isfinite(alpha) or alpha <= 0:
        raise DataContractError("NB2 dispersion must be positive and finite")
    converged = bool(fitted.mle_retvals.get("converged", False))
    if not converged:
        raise DataContractError("NB2 fit did not converge")
    predicted_rate = fitted.predict() / frame["fixed_workplace"].to_numpy(dtype=float)
    return CountModelResult(
        family="negative_binomial_nb2",
        coefficients=coefficients,
        observations=len(frame),
        log_likelihood=float(fitted.llf),
        aic=float(fitted.aic),
        pearson_dispersion=float("nan"),
        alpha=alpha,
        converged=converged,
        predicted_rate_min=float(predicted_rate.min()),
        predicted_rate_max=float(predicted_rate.max()),
    )


def fit_binomial_sensitivity(metrics: pd.DataFrame) -> CountModelResult:
    """Fit a bounded-response sensitivity model for the local-retention proportion."""
    frame = model_frame(metrics)
    design = sm.add_constant(frame.loc[:, PREDICTORS], has_constant="add")
    exposure = frame["fixed_workplace"].to_numpy(dtype=float)
    response = frame["local_fixed"].to_numpy(dtype=float) / exposure
    fitted = sm.GLM(
        response,
        design,
        family=sm.families.Binomial(),
        freq_weights=exposure,
    ).fit(cov_type="HC0")
    converged = bool(getattr(fitted, "converged", False))
    if not converged:
        raise DataContractError("binomial sensitivity fit did not converge")
    confidence = fitted.conf_int()
    coefficients = pd.DataFrame(
        {
            "term": fitted.params.index,
            "estimate": fitted.params.to_numpy(dtype=float),
            "robust_se": fitted.bse.to_numpy(dtype=float),
            "p_value": fitted.pvalues.to_numpy(dtype=float),
            "confidence_low": confidence.iloc[:, 0].to_numpy(dtype=float),
            "confidence_high": confidence.iloc[:, 1].to_numpy(dtype=float),
        }
    )
    predicted_rate = fitted.predict(design)
    if not ((predicted_rate >= 0) & (predicted_rate <= 1)).all():
        raise DataContractError("binomial sensitivity produced an invalid probability")
    return CountModelResult(
        family="binomial_sensitivity",
        coefficients=coefficients,
        observations=len(frame),
        log_likelihood=float(fitted.llf),
        aic=float(fitted.aic),
        pearson_dispersion=float("nan"),
        alpha=None,
        converged=converged,
        predicted_rate_min=float(predicted_rate.min()),
        predicted_rate_max=float(predicted_rate.max()),
    )


def fit_count_models(
    metrics: pd.DataFrame,
) -> tuple[CountModelResult, CountModelResult, CountModelResult]:
    """Fit count specifications and a bounded sensitivity model on identical rows."""
    poisson = fit_poisson(metrics)
    negative_binomial = fit_negative_binomial(metrics)
    binomial = fit_binomial_sensitivity(metrics)
    if len({poisson.observations, negative_binomial.observations, binomial.observations}) != 1:
        raise DataContractError("model specifications do not use the same observation set")
    return poisson, negative_binomial, binomial
