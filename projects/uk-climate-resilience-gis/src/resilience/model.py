"""Transparent, complete-case neighbourhood screening and weight sensitivity.

All five inputs must be adverse-high. Scores are relative screening measures,
not estimated losses or health outcomes. Weight-experiment frequencies and rank
ranges describe this specified experiment; they are not probabilities that a
priority is correct, confidence intervals, or assessments of source quality.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import rankdata


DIMENSIONS = ("dim_flood", "dim_heat", "dim_access", "dim_income", "dim_age")
SCENARIOS = ("balanced", "hazard", "equity")
SCORE_DECIMALS = 12


def _settings(config: dict) -> tuple[list[str], dict[str, np.ndarray], float, int, float, float, int]:
    """Reject policy mistakes rather than silently repairing their weights."""
    try:
        indicators = list(config["indicators"])
        scenarios = config["scenarios"]
        fraction = float(config["top_fraction"])
        samples = config["sensitivity_samples"]
        concentration = float(config["dirichlet_concentration"])
        robust_threshold = float(config["robust_frequency_threshold"])
        seed = config["seed"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid model configuration") from exc
    if (len(indicators) != 5 or not all(isinstance(x, str) and x for x in indicators)
            or len(set(indicators)) != 5):
        raise ValueError("Exactly five distinct indicator columns are required")
    if not isinstance(scenarios, dict) or set(scenarios) != set(SCENARIOS):
        raise ValueError("Scenarios must be balanced, hazard and equity")
    weights = {}
    for name in SCENARIOS:
        try:
            weight = np.asarray(scenarios[name], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid weights for {name}") from exc
        if (weight.shape != (5,) or not np.isfinite(weight).all()
                or (weight < 0).any()
                or not np.isclose(weight.sum(), 1.0, rtol=0, atol=1e-9)):
            raise ValueError(f"Weights for {name} must be five finite nonnegative values summing to one")
        weights[name] = weight
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if isinstance(samples, bool) or not isinstance(samples, Integral) or samples < 3:
        raise ValueError("sensitivity_samples must be an integer of at least three")
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("dirichlet_concentration must be finite and positive")
    if not np.isfinite(robust_threshold) or not 0 <= robust_threshold <= 1:
        raise ValueError("robust_frequency_threshold must be in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return indicators, weights, fraction, int(samples), concentration, robust_threshold, int(seed)


def _normalise(values: np.ndarray, method: str) -> np.ndarray:
    """Map tied minima/maxima to 0/100; constant columns receive 50."""
    result = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        raw = values[:, column]
        if not len(raw):
            continue
        if np.all(raw == raw[0]):
            result[:, column] = 50.0
            continue
        if method == "midrank":
            ordered = rankdata(raw, method="average")
        else:
            # Scaling first avoids overflow for a finite but very wide raw range.
            ordered = raw / np.max(np.abs(raw))
        result[:, column] = 100 * (ordered - ordered.min()) / (ordered.max() - ordered.min())
    return result


def _scores(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # Numerical round-off must not split mathematically tied weighted scores.
    return np.round(values @ weights, SCORE_DECIMALS)


def _top(scores: np.ndarray, fraction: float) -> tuple[np.ndarray, float | None]:
    if not len(scores):
        return np.zeros(0, dtype=bool), None
    threshold = float(np.quantile(scores, 1 - fraction, method="linear"))
    return scores >= threshold, threshold


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def _correlations(vectors: dict[str, np.ndarray]) -> dict:
    return {left: {right: _spearman(a, b) for right, b in vectors.items()}
            for left, a in vectors.items()}


def _overlap(left: np.ndarray, right: np.ndarray) -> dict:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return {"intersection_count": intersection, "union_count": union,
            "jaccard": intersection / union if union else None}


def _comparison(reference: np.ndarray, alternative: np.ndarray, fraction: float) -> dict:
    reference_top, _ = _top(reference, fraction)
    alternative_top, threshold = _top(alternative, fraction)
    return {
        "assessed": bool(len(reference)),
        "spearman": _spearman(reference, alternative),
        "top_count": int(alternative_top.sum()),
        "top_threshold": threshold,
        "top_jaccard": _overlap(reference_top, alternative_top)["jaccard"],
    }


def _weight_experiment(weights: dict[str, np.ndarray], samples: int,
                       concentration: float, seed: int) -> tuple[np.ndarray, dict]:
    """Use equally sized centre groups to the extent allowed by integer counts."""
    rng = np.random.default_rng(seed)
    counts = {name: samples // len(SCENARIOS) + (i < samples % len(SCENARIOS))
              for i, name in enumerate(SCENARIOS)}
    blocks = []
    for name in SCENARIOS:
        centre = weights[name]
        active = centre > 0
        block = np.zeros((counts[name], len(centre)))
        # Zero policy weights remain zero: sample the corresponding simplex face.
        block[:, active] = rng.dirichlet(concentration * centre[active], size=counts[name])
        blocks.append(block)
    draws = np.vstack(blocks)
    return draws, {
        "seed": seed,
        "generator": type(rng.bit_generator).__name__,
        "samples": samples,
        "concentration": concentration,
        "centre_order": list(SCENARIOS),
        "centres": {name: weights[name].tolist() for name in SCENARIOS},
        "alpha_by_scenario": {name: (concentration * weights[name]).tolist() for name in SCENARIOS},
        "draws_by_scenario": counts,
        "allocation_rule": "Equal integer counts, with remainder assigned in centre_order",
        "zero_weight_rule": "Zero-weight dimensions stay zero; Dirichlet uses positive dimensions only",
        "weight_draws": draws.tolist(),
        "interpretation": "Selection frequency and rank ranges conditional on this weight experiment; not probabilities of correctness or confidence intervals",
    }


def fit_model(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Return screening columns and JSON-safe methodological diagnostics.

    The five entries of ``config['indicators']`` map, in order, to flood, heat,
    access, income and age. ``model_eligible`` requires five finite numeric
    inputs. Only this common complete-case cohort fits either normalisation or
    any scenario; no row is partly reweighted. Original rows, columns and index
    are preserved. Excluded rows have missing numerical model outputs and
    ``robust_priority=False`` (meaning unassessed when eligibility is false).

    Ranks are descending average ranks. Top groups include every score at or
    above the linear empirical (1-top_fraction) quantile, so ties can enlarge
    the group. ``top20_frequency`` retains the requested interface name but
    follows configured ``top_fraction``. Rank spans are divided by n-1 (zero
    for a singleton). Source adequacy never enters the model-stability flag.
    """
    indicators, weights, fraction, samples, concentration, robust_threshold, seed = _settings(config)
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ValueError("Input must be a DataFrame with unique column names")
    missing_columns = [name for name in ["lsoa21cd"] + indicators if name not in frame]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))
    ids = frame["lsoa21cd"].to_numpy()
    if any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in ids):
        raise ValueError("lsoa21cd must contain nonempty, whitespace-free string identifiers")
    if frame["lsoa21cd"].duplicated().any():
        raise ValueError("lsoa21cd must be unique")

    raw = np.column_stack([
        pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        for name in indicators
    ])
    finite = np.isfinite(raw)
    eligible = finite.all(axis=1)
    positions = np.flatnonzero(eligible)
    positions = positions[np.argsort(ids[positions], kind="stable")]
    cohort_ids = ids[positions].tolist()
    values = raw[positions]
    n = len(positions)
    normalized = _normalise(values, "midrank")
    result = frame.copy(deep=True)
    result["model_eligible"] = eligible
    result["model_missing_indicators"] = [
        ", ".join(name for name, valid in zip(indicators, row) if not valid) for row in finite
    ]

    def assign(column: str, cohort_values: np.ndarray) -> None:
        output = np.full(len(frame), np.nan)
        output[positions] = cohort_values
        result[column] = output

    for j, dimension in enumerate(DIMENSIONS):
        assign(dimension, normalized[:, j])
    scenario_scores = {}
    scenario_tops = {}
    scenario_diagnostics = {}
    for name in SCENARIOS:
        scores = _scores(normalized, weights[name])
        top, threshold = _top(scores, fraction)
        scenario_scores[name] = scores
        scenario_tops[name] = top
        assign("score_" + name, scores)
        assign("rank_" + name, rankdata(-scores, method="average"))
        scenario_diagnostics[name] = {
            "weights": weights[name].tolist(),
            "top_threshold": threshold,
            "top_count": int(top.sum()),
            "observed_top_fraction": float(top.mean()) if n else None,
            "ties_at_threshold": int(np.count_nonzero(scores == threshold)) if n else 0,
            "excess_over_nominal_count": max(0, int(top.sum()) - math.ceil(n * fraction)),
            "top_ids": [area for area, selected in zip(cohort_ids, top) if selected],
            "all_scores_tied": bool(n and np.all(scores == scores[0])),
        }
    scenario_top_count = np.stack(list(scenario_tops.values()), axis=1).sum(axis=1)
    assign("scenario_top_count", scenario_top_count)

    draws, experiment = _weight_experiment(weights, samples, concentration, seed)
    if n:
        sampled_scores = _scores(normalized, draws.T)
        thresholds = np.quantile(sampled_scores, 1 - fraction, axis=0, method="linear")
        sampled_top = sampled_scores >= thresholds
        frequency = sampled_top.mean(axis=1)
        sampled_ranks = rankdata(-sampled_scores, method="average", axis=0)
        p05, p95 = np.quantile(sampled_ranks, [0.05, 0.95], axis=1, method="linear")
        rank_span = (p95 - p05) / (n - 1) if n > 1 else np.zeros(n)
        experiment["top_counts_by_draw"] = sampled_top.sum(axis=0).tolist()
    else:
        frequency = p05 = p95 = rank_span = np.empty(0)
        experiment["top_counts_by_draw"] = [0] * samples
    for column, output in [
        ("top20_frequency", frequency), ("rank_p05", p05),
        ("rank_p95", p95), ("rank_span_fraction", rank_span),
    ]:
        assign(column, output)
    robust = np.zeros(len(frame), dtype=bool)
    robust[positions] = (frequency >= robust_threshold) & (scenario_top_count == len(SCENARIOS))
    result["robust_priority"] = robust

    alternative = _normalise(values, "minmax")
    alternative_diagnostics = {
        name: _comparison(scenario_scores[name], _scores(alternative, weights[name]), fraction)
        for name in SCENARIOS
    }
    leave_one_out = {}
    for j, indicator in enumerate(indicators):
        leave_one_out[indicator] = {}
        for name in SCENARIOS:
            reduced = weights[name].copy()
            reduced[j] = 0
            if reduced.sum() == 0:
                leave_one_out[indicator][name] = {
                    "assessed": False,
                    "reason": "No positive policy weight remains after omitting this dimension",
                    "spearman": None, "top_count": None,
                    "top_threshold": None, "top_jaccard": None,
                }
            else:
                reduced /= reduced.sum()
                leave_one_out[indicator][name] = _comparison(
                    scenario_scores[name], _scores(normalized, reduced), fraction)

    normalization_details = {}
    for j, name in enumerate(indicators):
        raw_column = values[:, j]
        normalization_details[name] = {
            "output_column": DIMENSIONS[j],
            "raw_min": float(raw_column.min()) if n else None,
            "raw_max": float(raw_column.max()) if n else None,
            "constant": bool(n and np.all(raw_column == raw_column[0])),
            "unique_values": int(len(np.unique(raw_column))),
        }
    diagnostics = {
        "cohort": {
            "input_count": len(frame), "eligible_count": n,
            "excluded_count": len(frame) - n,
            "eligible_ids_sorted": cohort_ids,
            "excluded_ids_sorted": sorted(ids[~eligible].tolist()),
            "rule": "Same five-finite-indicator complete cases for every fit and comparison; no per-row reweighting",
        },
        "normalization": {
            "method": "100 * (average_rank - minimum_average_rank) / (maximum_average_rank - minimum_average_rank)",
            "constant_value": 50.0,
            "dimensions": normalization_details,
            "alternative": "Raw min-max over the same complete cases, with constants at 50",
        },
        "top_group": {
            "fraction": fraction,
            "rule": "score >= linear empirical quantile(1 - top_fraction); include all ties",
            "frequency_column": "top20_frequency",
        },
        "score_rounding_decimals": SCORE_DECIMALS,
        "rank_method": "Descending average ranks; no identifier-based tie breaking",
        "scenarios": scenario_diagnostics,
        "indicator_spearman": _correlations({name: values[:, j] for j, name in enumerate(indicators)}),
        "scenario_rank_spearman": _correlations(scenario_scores),
        "scenario_top_overlap": {
            left: {right: _overlap(a, b) for right, b in scenario_tops.items()}
            for left, a in scenario_tops.items()
        },
        "weight_experiment": experiment,
        "alternative_minmax": alternative_diagnostics,
        "leave_one_dimension_out": leave_one_out,
        "robustness": {
            "frequency_threshold": robust_threshold,
            "rule": "frequency >= threshold AND top group in all three configured scenarios",
            "rank_interval": "5th and 95th empirical percentiles across configured weight draws; not confidence limits",
            "rank_span_fraction": "(rank_p95 - rank_p05) / (eligible_count - 1), or zero for a singleton",
            "source_quality_included": False,
            "unassessed_rows": "robust_priority is False when model_eligible is False",
            "all_tied_warning": "An all-tied score cannot distinguish neighbourhoods; inclusive top groups can contain the entire cohort",
        },
    }
    return result, diagnostics
