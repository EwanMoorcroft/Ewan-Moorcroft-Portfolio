"""Count-model design tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import statsmodels.api as sm

from liverpool_accessibility.analysis import prepare_evidence
from liverpool_accessibility.contracts import DataContractError
from liverpool_accessibility.fixtures import fixture_source_manifest, write_fixture
from liverpool_accessibility.modelling import (
    fit_binomial_sensitivity,
    fit_count_models,
    fit_poisson,
    model_frame,
)


def _fixture_evidence(tmp_path: Path):
    flows, boundaries, centroids = write_fixture(tmp_path / "source", size=5)
    return prepare_evidence(
        flows,
        boundaries,
        centroids,
        source_manifest=fixture_source_manifest(flows, boundaries, centroids),
    )


def test_model_uses_fixed_workplace_exposure(tmp_path: Path) -> None:
    """The fixed model must retain the declared count exposure for every row."""
    evidence = _fixture_evidence(tmp_path)
    assert evidence.source_contract == "liverpool-fixture-source-manifest-v1"
    assert evidence.evidence_scope == "fictional deterministic integration fixture"
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
    assert poisson.converged
    assert negative_binomial.alpha is not None and negative_binomial.alpha > 0
    assert negative_binomial.converged
    assert binomial.converged
    assert 0 <= binomial.predicted_rate_min <= binomial.predicted_rate_max <= 1


@pytest.mark.parametrize(
    ("fit_model", "message"),
    [
        (fit_poisson, "Poisson fit did not converge"),
        (fit_binomial_sensitivity, "binomial sensitivity fit did not converge"),
    ],
)
def test_glm_models_fail_closed_on_non_convergence(
    tmp_path: Path, monkeypatch, fit_model, message: str
) -> None:
    """A statsmodels non-convergence signal must stop evidence creation."""
    evidence = _fixture_evidence(tmp_path)
    monkeypatch.setattr(
        sm.GLM,
        "fit",
        lambda self, **kwargs: SimpleNamespace(converged=False),
    )
    with pytest.raises(DataContractError, match=message):
        fit_model(evidence.metrics)
