
import jax.numpy as jnp
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
    Subphase,
    scan_step_ii_key,
    years_key,
)
from .computation import (
    Computation,
    ModelElement,
    ScenarioModel,
    define_annual,
    run_mcmc,
)
from .example_population import Example_Population
from .workspace import Workspace


@pytest.fixture
def model():
    model = ScenarioModel(
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

    # annual variables declared on model_element_prepare are inputs to the
    # scan, and those declared on model_element_annual_step are outputs.
    assert model.mv.annual_nd[years_key].defining_subphase == Subphase.Prep
    assert model.mv.annual_nd[scan_step_ii_key].defining_subphase == Subphase.Prep
    assert model.mv.annual_nd[
            ElementAnnualKey(elem_id=elem_id, name='mu')
            ].defining_subphase == Subphase.Step


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


def _annual_test_model(element:ModelElement, n_years:int=4) -> ScenarioModel:
    model = ScenarioModel(
            first_year=2000,
            n_prior_years=n_years,
            n_posterior_years=n_years,
            num_warmup=1,
            num_samples=1)
    model.add_element(element)
    return model


class AnnualEcho(ModelElement):
    """Exercises both sides of this_annual within a scan step."""

    @define_annual('input_series', annual_shape=[])
    def model_element_prepare(self, ws:Workspace):
        ws.val.annual['input_series'] = jnp.arange(ws.n_years) * 10.0

    @define_annual('echo', annual_shape=[])
    @define_annual('scan_step_echo', annual_shape=[])
    def model_element_annual_step(self, ws:Workspace):
        # read an annual variable defined prior to the scan ...
        ws.val.this_annual['echo'] = ws.val.this_annual['input_series'] + 1
        # ... and a built-in annual variable defined prior to the scan.
        ws.val.this_annual['scan_step_echo'] = ws.val.this_annual[scan_step_ii_key]


def test_this_annual_combines_scan_inputs_and_outputs():
    n_years = 4
    model = _annual_test_model(AnnualEcho(), n_years=n_years)
    comp = Computation(model=model, rng_key=jrandom.key(1))
    comp.run()

    elem_id = 'AnnualEcho'

    def nd(name):
        return np.asarray(
                comp._ndarray_d[ElementAnnualKey(elem_id=elem_id, name=name)])

    expected_input = np.arange(n_years) * 10.0
    assert np.all(nd('input_series') == expected_input)
    assert np.all(nd('echo') == expected_input + 1)
    assert np.all(nd('scan_step_echo') == np.arange(n_years))


def test_this_annual_combines_scan_inputs_and_outputs_posterior():
    n_years = 5
    model = _annual_test_model(AnnualEcho(), n_years=n_years)
    comp = Computation(model=model, rng_key=jrandom.key(1))
    comp.set_phase_posterior(
            n_mcmc=1,
            sample_sites={},
            grouped_samples={})
    comp.run()

    elem_id = 'AnnualEcho'

    def nd(name):
        return np.asarray(
                comp._ndarray_d[ElementAnnualKey(elem_id=elem_id, name=name)])

    expected_input = np.arange(n_years) * 10.0
    assert nd('input_series').shape == (n_years, 1)
    assert nd('echo').shape == (n_years, 1)
    assert np.all(nd('input_series')[:, 0] == expected_input)
    assert np.all(nd('echo')[:, 0] == expected_input + 1)
    assert np.all(nd('scan_step_echo')[:, 0] == np.arange(n_years))


class WritesPrepareAnnualInStep(ModelElement):
    """An annual defined prior to the scan must not be written in a step."""

    @define_annual('input_series', annual_shape=[])
    def model_element_prepare(self, ws:Workspace):
        ws.val.annual['input_series'] = jnp.arange(ws.n_years)

    def model_element_annual_step(self, ws:Workspace):
        ws.val.this_annual['input_series'] = 1.0


def test_this_annual_prepare_defined_is_read_only_in_step():
    model = _annual_test_model(WritesPrepareAnnualInStep())
    comp = Computation(model=model, rng_key=jrandom.key(1))
    with pytest.raises(NotImplementedError):
        comp.run()


class DistOnPrepareAnnualInStep(ModelElement):
    """A scan-step distribution only makes sense for step-defined annuals."""

    @define_annual('input_series', annual_shape=[])
    def model_element_prepare(self, ws:Workspace):
        ws.val.annual['input_series'] = jnp.arange(ws.n_years)

    def model_element_annual_step(self, ws:Workspace):
        ws.dist.this_annual['input_series']


def test_this_annual_dist_for_prepare_defined_raises():
    model = _annual_test_model(DistOnPrepareAnnualInStep())
    comp = Computation(model=model, rng_key=jrandom.key(1))
    with pytest.raises(NotImplementedError):
        comp.run()
