
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from .base import (
    DefinitionError,
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
    access_annual,
    access_val,
    define,
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


# ---------------------------------------------------------------------------
# access_val / access_annual metadata and defaults
# ---------------------------------------------------------------------------

class AccessMetaElement(ModelElement):
    """Exercises the key forms accepted by the access decorators."""

    def __init__(self):
        self._prop_key = ElementGeneralKey(
                elem_id=self.identifier, name='prop')
        self._dict_keys = {
                'a': ElementAnnualKey(elem_id=self.identifier, name='a'),
                'b': ElementAnnualKey(elem_id=self.identifier, name='b'),
                }

    @property
    def prop_key(self):
        return self._prop_key

    @property
    def dict_keys(self):
        return self._dict_keys

    @access_val('plain', default=jnp.array(1.5))
    @access_val(prop_key)
    @access_annual('annual_plain', default=jnp.array(2.5))
    @access_annual(dict_keys)
    def model_element_postprocess(self, ws:Workspace):
        pass


def test_access_decorator_metadata():
    model = _annual_test_model(AccessMetaElement())
    elem_id = 'AccessMetaElement'

    assert len(model.mv.accesses) == 5
    by_key = {access.var_key: access for access in model.mv.accesses}

    plain = ElementGeneralKey(elem_id=elem_id, name='plain')
    assert by_key[plain].subphase == Subphase.Proc
    assert by_key[plain].elem_id == elem_id
    assert np.all(np.asarray(by_key[plain].default) == 1.5)

    prop = ElementGeneralKey(elem_id=elem_id, name='prop')
    assert prop in by_key
    assert by_key[prop].default is None

    annual_plain = ElementAnnualKey(elem_id=elem_id, name='annual_plain')
    assert by_key[annual_plain].subphase == Subphase.Proc
    assert np.all(np.asarray(by_key[annual_plain].default) == 2.5)

    assert ElementAnnualKey(elem_id=elem_id, name='a') in by_key
    assert ElementAnnualKey(elem_id=elem_id, name='b') in by_key


def test_access_decorators_check_key_kind():
    annual_key = ElementAnnualKey(elem_id='BadElement', name='x')

    class BadAccessVal(ModelElement):
        @access_val(annual_key)
        def model_element_prepare(self, ws:Workspace):
            pass

    with pytest.raises(TypeError):
        _annual_test_model(BadAccessVal())

    general_key = ElementGeneralKey(elem_id='BadElement', name='x')

    class BadAccessAnnual(ModelElement):
        @access_annual(general_key)
        def model_element_prepare(self, ws:Workspace):
            pass

    with pytest.raises(TypeError):
        _annual_test_model(BadAccessAnnual())


class GeneralDefaultReader(ModelElement):

    @access_val('missing', default=jnp.array(7.0))
    @define('out', prior_shape=[])
    def model_element_prepare(self, ws:Workspace):
        ws.val.general['out'] = ws.val.general['missing'] * 2.0


def test_access_val_default_when_undefined():
    model = _annual_test_model(GeneralDefaultReader())
    comp = Computation(model=model, rng_key=jrandom.key(1))
    comp.run()

    out_key = ElementGeneralKey(elem_id='GeneralDefaultReader', name='out')
    assert float(comp._ndarray_d[out_key]) == 14.0


class AnnualDefaultReader(ModelElement):

    @access_annual('missing_annual', default=jnp.arange(4.0))
    @define_annual('out', annual_shape=[])
    def model_element_prepare(self, ws:Workspace):
        ws.val.annual['out'] = ws.val.annual['missing_annual'] + 1.0


def test_access_annual_default_when_undefined():
    model = _annual_test_model(AnnualDefaultReader(), n_years=4)
    comp = Computation(model=model, rng_key=jrandom.key(1))
    comp.run()

    out_key = ElementAnnualKey(elem_id='AnnualDefaultReader', name='out')
    assert np.all(np.asarray(comp._ndarray_d[out_key]) == np.arange(1.0, 5.0))


class AnnualStepDefaultReader(ModelElement):

    @access_annual('missing_annual', default=jnp.array(5.0))
    @define_annual('out', annual_shape=[])
    def model_element_annual_step(self, ws:Workspace):
        ws.val.this_annual['out'] = ws.val.this_annual['missing_annual'] + 1.0


def test_access_annual_step_default_when_undefined():
    n_years = 4
    model = _annual_test_model(AnnualStepDefaultReader(), n_years=n_years)
    comp = Computation(model=model, rng_key=jrandom.key(1))
    comp.run()

    out_key = ElementAnnualKey(elem_id='AnnualStepDefaultReader', name='out')
    assert np.all(np.asarray(comp._ndarray_d[out_key]) == 6.0)


def test_access_annual_step_default_when_undefined_posterior():
    n_years = 3
    model = _annual_test_model(AnnualStepDefaultReader(), n_years=n_years)
    comp = Computation(model=model, rng_key=jrandom.key(1))
    comp.set_phase_posterior(
            n_mcmc=2,
            sample_sites={},
            grouped_samples={})
    comp.run()

    out_key = ElementAnnualKey(elem_id='AnnualStepDefaultReader', name='out')
    out = np.asarray(comp._ndarray_d[out_key])
    assert out.shape == (n_years, 1)
    assert np.all(out == 6.0)


class DistDefaultReader(ModelElement):
    """A default is for value reads; dist reads of undefined keys still fail."""

    @access_val('missing', default=jnp.array(1.0))
    def model_element_prepare(self, ws:Workspace):
        ws.dist.general['missing']


def test_access_default_does_not_apply_to_dist_reads():
    model = _annual_test_model(DistDefaultReader())
    comp = Computation(model=model, rng_key=jrandom.key(1))
    with pytest.raises(KeyError):
        comp.run()


# ---------------------------------------------------------------------------
# Defaults for one key must agree across access locations
# ---------------------------------------------------------------------------

_shared_default_key = ElementGeneralKey(elem_id='DefaultUserA', name='shared')


class DefaultUserA(ModelElement):

    @access_val(_shared_default_key, default=jnp.array(5.0))
    @define('out', prior_shape=[])
    def model_element_prepare(self, ws:Workspace):
        ws.val.general['out'] = ws.val.general[_shared_default_key] + 0.0


class DefaultUserB(ModelElement):

    @access_val(_shared_default_key, default=jnp.array([5.0, 5.0]))
    @define('out', prior_shape=[2])
    def model_element_postprocess(self, ws:Workspace):
        ws.val.general['out'] = ws.val.general[_shared_default_key] + 0.0


def test_default_consistency_allows_broadcast_equal():
    model = _annual_test_model(DefaultUserA())
    # scalar and shape-(2,) defaults broadcast to (2,) and are equal
    model.add_element(DefaultUserB())
    comp = Computation(model=model, rng_key=jrandom.key(1))
    comp.run()

    a_out = ElementGeneralKey(elem_id='DefaultUserA', name='out')
    b_out = ElementGeneralKey(elem_id='DefaultUserB', name='out')
    assert float(comp._ndarray_d[a_out]) == 5.0
    b_val = np.asarray(comp._ndarray_d[b_out])
    assert b_val.shape == (2,)
    assert np.all(b_val == 5.0)


_shared_bad_default_key = ElementGeneralKey(elem_id='BadDefaultA', name='shared')


class BadDefaultA(ModelElement):

    @access_val(_shared_bad_default_key, default=jnp.array(5.0))
    def model_element_prepare(self, ws:Workspace):
        pass


class BadDefaultB(ModelElement):

    @access_val(_shared_bad_default_key, default=jnp.array(6.0))
    def model_element_prepare(self, ws:Workspace):
        pass


def test_default_consistency_rejects_unequal_values():
    model = _annual_test_model(BadDefaultA())
    with pytest.raises(DefinitionError):
        model.add_element(BadDefaultB())


_shape_bad_default_key = ElementGeneralKey(elem_id='ShapeBadDefaultA', name='shared')


class ShapeBadDefaultA(ModelElement):

    @access_val(_shape_bad_default_key, default=jnp.zeros(2))
    def model_element_prepare(self, ws:Workspace):
        pass


class ShapeBadDefaultB(ModelElement):

    @access_val(_shape_bad_default_key, default=jnp.zeros(3))
    def model_element_prepare(self, ws:Workspace):
        pass


def test_default_consistency_rejects_incompatible_shapes():
    model = _annual_test_model(ShapeBadDefaultA())
    with pytest.raises(DefinitionError):
        model.add_element(ShapeBadDefaultB())


_nan_default_key = ElementGeneralKey(elem_id='NanDefaultA', name='shared')


class NanDefaultA(ModelElement):

    @access_val(_nan_default_key, default=jnp.array(jnp.nan))
    def model_element_prepare(self, ws:Workspace):
        pass


class NanDefaultB(ModelElement):

    @access_val(_nan_default_key, default=jnp.array([jnp.nan, jnp.nan]))
    def model_element_prepare(self, ws:Workspace):
        pass


def test_default_consistency_nan_equal():
    model = _annual_test_model(NanDefaultA())
    model.add_element(NanDefaultB())


# ---------------------------------------------------------------------------
# Dependency ordering
# ---------------------------------------------------------------------------

_prep_shared_key = ElementGeneralKey(elem_id='PrepKeyProducer', name='shared')


class PrepKeyProducer(ModelElement):

    @define(_prep_shared_key, prior_shape=[])
    def model_element_prepare(self, ws:Workspace):
        ws.val.general[_prep_shared_key] = jnp.array(42.0)


class PrepKeyConsumer(ModelElement):

    @access_val(_prep_shared_key, default=jnp.array(99.0))
    @define('out', prior_shape=[])
    def model_element_prepare(self, ws:Workspace):
        ws.val.general['out'] = ws.val.general[_prep_shared_key] * 1.0


def test_prep_dependency_order_and_default_ignored():
    model = _annual_test_model(PrepKeyConsumer())
    model.add_element(PrepKeyProducer())

    assert model.element_order(Subphase.Prep) == [
            'ModelBuiltIns', 'PrepKeyProducer', 'PrepKeyConsumer']

    comp = Computation(model=model, rng_key=jrandom.key(1))
    comp.run()
    out_key = ElementGeneralKey(elem_id='PrepKeyConsumer', name='out')
    assert float(comp._ndarray_d[out_key]) == 42.0


_step_shared_key = ElementAnnualKey(elem_id='StepKeyProducer', name='shared')


class StepKeyProducer(ModelElement):

    @define_annual(_step_shared_key, annual_shape=[])
    def model_element_annual_step(self, ws:Workspace):
        ws.val.this_annual[_step_shared_key] = 2.0


class StepKeyConsumer(ModelElement):

    @access_annual(_step_shared_key)
    @define_annual('out', annual_shape=[])
    def model_element_annual_step(self, ws:Workspace):
        ws.val.this_annual['out'] = ws.val.this_annual[_step_shared_key] + 1.0


def test_step_dependency_order():
    n_years = 4
    model = _annual_test_model(StepKeyConsumer(), n_years=n_years)
    model.add_element(StepKeyProducer())

    assert model.element_order(Subphase.Step) == [
            'ModelBuiltIns', 'StepKeyProducer', 'StepKeyConsumer']

    comp = Computation(model=model, rng_key=jrandom.key(1))
    comp.run()
    out_key = ElementAnnualKey(elem_id='StepKeyConsumer', name='out')
    assert np.all(np.asarray(comp._ndarray_d[out_key]) == 3.0)


_proc_shared_key = ElementGeneralKey(elem_id='ProcKeyProducer', name='shared')


class ProcKeyProducer(ModelElement):

    @define(_proc_shared_key, prior_shape=[])
    def model_element_postprocess(self, ws:Workspace):
        ws.val.general[_proc_shared_key] = jnp.array(7.0)


class ProcKeyConsumer(ModelElement):

    @access_val(_proc_shared_key)
    @define('out', prior_shape=[])
    def model_element_postprocess(self, ws:Workspace):
        ws.val.general['out'] = ws.val.general[_proc_shared_key] * 2.0


def test_proc_dependency_order():
    model = _annual_test_model(ProcKeyConsumer())
    model.add_element(ProcKeyProducer())

    assert model.element_order(Subphase.Proc) == [
            'ModelBuiltIns', 'ProcKeyProducer', 'ProcKeyConsumer']

    comp = Computation(model=model, rng_key=jrandom.key(1))
    comp.run()
    out_key = ElementGeneralKey(elem_id='ProcKeyConsumer', name='out')
    assert float(comp._ndarray_d[out_key]) == 14.0


_cycle_a_key = ElementGeneralKey(elem_id='CycleA', name='a')
_cycle_b_key = ElementGeneralKey(elem_id='CycleB', name='b')


class CycleA(ModelElement):

    @access_val(_cycle_b_key)
    @define(_cycle_a_key, prior_shape=[])
    def model_element_prepare(self, ws:Workspace):
        pass


class CycleB(ModelElement):

    @access_val(_cycle_a_key)
    @define(_cycle_b_key, prior_shape=[])
    def model_element_prepare(self, ws:Workspace):
        pass


def test_dependency_cycle_raises():
    model = _annual_test_model(CycleA())
    model.add_element(CycleB())
    with pytest.raises(DefinitionError):
        model.element_order(Subphase.Prep)


_dup_key = ElementGeneralKey(elem_id='DupA', name='same')


class DupA(ModelElement):

    @define(_dup_key, prior_shape=[])
    def model_element_prepare(self, ws:Workspace):
        pass


class DupB(ModelElement):

    @define(_dup_key, prior_shape=[])
    def model_element_postprocess(self, ws:Workspace):
        pass


def test_duplicate_key_definition_raises():
    model = _annual_test_model(DupA())
    with pytest.raises(DefinitionError):
        model.add_element(DupB())
