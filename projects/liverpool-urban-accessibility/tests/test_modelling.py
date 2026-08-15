"""Count-model design tests."""

from __future__ import annotations

from pathlib import Path

from liverpool_accessibility.analysis import prepare_evidence
from liverpool_accessibility.fixtures import write_fixture
from liverpool_accessibility.modelling import fit_count_models, model_frame


def test_model_uses_fixed_workplace_exposure(tmp_path: Path) -> None:
    """The fixed model must retain the declared count exposure for every row."""
    flows, boundaries, centroids = write_fixture(tmp_path / "source", size=5)
    evidence = prepare_evidence(flows, boundaries, centroids)
    frame = model_frame(evidence.metrics)
    assert len(frame) == 25
    assert (frame["fixed_workplace"] > 0).all()
    poisson, negative_binomial, binomial = fit_count_models(evidence.metrics)
    assert poisson.observations == negative_binomial.observations == binomial.observations == 25
    assert list(poisson.coefficients["term"]) == [
        "const",
        "log_accessibility_5km",
        "home_or_no_fixed_share",
    ]
    assert poisson.pearson_dispersion >= 0
    assert negative_binomial.alpha is not None and negative_binomial.alpha > 0
    assert negative_binomial.converged
    assert 0 <= binomial.predicted_rate_min <= binomial.predicted_rate_max <= 1
