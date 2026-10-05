
import jax.random as jrandom
import pytest

from .base import (
    GroupedPosterior,
    NextCarry,
    final_carry,
    initial_carry,
)
from .computation import Computation, Model, run_mcmc
from .example_population import Example_Population


@pytest.fixture

def model():
    model = Model(
            first_year=1990,
            n_prior_years=7,
            n_posterior_years=9,
            num_warmup=30,
            num_samples=5)
    model.add_element(Example_Population())
    return model


def test_add_inference_add_vars(model):

    assert 'alpha' in model.mv.general_nd
    assert 'data' in model.mv.general_nd
    assert 'generated_data' in model.mv.general_nd

    assert initial_carry('y_curr') in model.mv.general_nd
    assert initial_carry('y_prev') in model.mv.general_nd
    assert 'y_curr' in model.mv.carry_nd
    assert 'y_prev' in model.mv.carry_nd
    assert final_carry('y_curr') in model.mv.general_nd
    assert final_carry('y_prev') in model.mv.general_nd

    assert 'mu' in model.mv.annual_nd


def test_add_inference_run_smoke(model):
    comp = Computation(
            model=model,
            rng_key=jrandom.key(123),
            )
    comp.run()


def test_add_inference_mcmc_smoke(model):
    run_mcmc(model=model, seed=123)


def test_posterior_smoke(model):
    mcmc = run_mcmc(model=model, seed=123)

    comp = Computation(
            model=model,
            rng_key=jrandom.key(123),
            )
    grouped_samples = mcmc.get_samples(group_by_chain=True)
    sample_sites = {
                'alpha': 'alpha',
                NextCarry(carry_key='y_curr'): str(NextCarry(carry_key='y_curr')),
                initial_carry('y_curr'): str(initial_carry('y_curr')),
                }
    assert set(sample_sites.values()) == set(grouped_samples)

    comp.set_phase_posterior(
            n_mcmc=5,
            sample_sites=sample_sites,
            grouped_samples=grouped_samples)
    comp.run()
