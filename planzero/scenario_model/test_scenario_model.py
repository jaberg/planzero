
import jax.random as jrandom
import numpy as np
import pytest

from .base import (
    ElementAnnualKey,
    ElementCarryKey,
    ElementGeneralKey,
    FinalCarry,
    InitialCarry,
    NextCarry,
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

    elem_id = 'Example_Population'

    assert ElementGeneralKey(elem_id=elem_id, name='alpha') in model.mv.general_nd
    assert ElementGeneralKey(elem_id=elem_id, name='data') in model.mv.general_nd
    assert ElementGeneralKey(elem_id=elem_id, name='generated_data') in model.mv.general_nd

    assert InitialCarry(
            carry_key=ElementCarryKey(elem_id=elem_id,
                                      name='y_curr')) in model.mv.general_nd
    assert InitialCarry(
            carry_key=ElementCarryKey(elem_id=elem_id,
                                      name='y_prev')) in model.mv.general_nd
    assert ElementCarryKey(
            elem_id=elem_id, name='y_curr') in model.mv.carry_nd
    assert ElementCarryKey(
            elem_id=elem_id, name='y_prev') in model.mv.carry_nd
    assert FinalCarry(
            carry_key=ElementCarryKey(elem_id=elem_id,
                                      name='y_curr')) in model.mv.general_nd
    assert FinalCarry(
            carry_key=ElementCarryKey(elem_id=elem_id,
                                      name='y_prev')) in model.mv.general_nd

    assert ElementAnnualKey(elem_id=elem_id, name='mu') in model.mv.annual_nd


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
    sample_sites = {}
    elem_id = 'Example_Population'
    alpha_key = ElementGeneralKey(elem_id=elem_id, name='alpha')
    y_curr_next = NextCarry(carry_key=ElementCarryKey(elem_id=elem_id, name='y_curr'))
    y_curr_init = InitialCarry(carry_key=ElementCarryKey(elem_id=elem_id, name='y_curr'))
    sample_sites = {
                alpha_key: str(alpha_key),
                y_curr_next: str(y_curr_next),
                y_curr_init: str(y_curr_init),
                }
    assert set(sample_sites.values()) == set(grouped_samples)
    generated_data_key = ElementGeneralKey(elem_id=elem_id, name='generated_data')
    assert generated_data_key not in comp._ndarray_d

    n_mcmc = 5
    comp.set_phase_posterior(
            n_mcmc=n_mcmc,
            sample_sites=sample_sites,
            grouped_samples=grouped_samples)
    comp.run()

    generated_data = comp._ndarray_d[generated_data_key]
    assert generated_data.shape == (9, n_mcmc)
    assert np.all(generated_data[0] == 1000.0)
    assert np.all(generated_data[6] == 1020.0)
    # Test that each mcmc sample is unique beyond the the first 7 steps
    assert len({float(foo) for foo in generated_data[7]}) == n_mcmc
