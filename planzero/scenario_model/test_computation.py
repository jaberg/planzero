import pytest

from .computation import (
    Computation,
    build_annual_prior_shape,
    build_annual_posterior_shape,
)

from .test_scenario_model import _annual_test_model, AnnualEcho


def test_annual_shape():


    model = _annual_test_model(AnnualEcho(), n_posterior_years=5)
    comp = Computation(model, rng_key=1)

    prior_shape = build_annual_prior_shape(comp, AnnualEcho().annual_key('input_series'), ())
    comp.set_phase_posterior(
            n_mcmc=3,
            sample_sites={},
            grouped_samples={})
    posterior_shape = build_annual_posterior_shape(
            comp, AnnualEcho().annual_key('input_series'), ())
    assert prior_shape == [4]
    assert posterior_shape == [5, 3]

    posterior_shape_2 = build_annual_posterior_shape(
            comp, AnnualEcho().annual_key('input_series'), (5, 3))
    assert posterior_shape_2 == [5, 3]
