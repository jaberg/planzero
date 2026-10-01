
import jax.random as jrandom
import pytest

from .base import (
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
            n_posterior_years=9)
    model.add_element(Example_Population())
    return model


def test_add_inference_add_vars(model):

    assert 'alpha' in model.mv.general_nd
    assert 'data' in model.mv.general_nd
    assert 'generated_data' in model.mv.general_nd

    assert initial_carry('y_curr') in model.mv.general_nd
    assert initial_carry('y_prev') in model.mv.general_nd
    assert 'y_curr' in model.mv.this_carry_nd
    assert 'y_prev' in model.mv.this_carry_nd
    assert 'y_curr' in model.mv.next_carry_nd
    assert 'y_prev' in model.mv.next_carry_nd
    assert final_carry('y_curr') in model.mv.general_nd
    assert final_carry('y_prev') in model.mv.general_nd

    assert 'mu' in model.mv.this_Y_nd
    assert {} == model.mv.this_X_nd


def test_add_inference_run_smoke(model):
    comp = Computation(
            model=model,
            rng_key=jrandom.key(123),
            grouped_samples=None)
    comp.run()


def test_add_inference_mcmc_smoke(model):
    run_mcmc(model=model,
             seed=123,
             num_warmup=10,
             thinning=1,
             num_samples=10)


def test_posterior_smoke(model):
    mcmc = run_mcmc(model=model,
             seed=123,
             num_warmup=10,
             thinning=1,
             num_samples=10)

    comp = Computation(
            model=model,
            grouped_samples=mcmc.get_samples(group_by_chain=True),
            rng_key=jrandom.key(123),
            )
    comp.run()
