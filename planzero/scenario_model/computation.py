# this resolves circular type references
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jrandom
import numpyro
from jax.typing import ArrayLike
from numpyro.contrib.control_flow import scan as numpyro_scan
from numpyro.distributions.distribution import Distribution
from pydantic import BaseModel, computed_field

from .base import (
    InitialCarry,
    ModellingPhase,
    ModelVariables,
    NdarrayDefinitionMetadata,
    NdarrayDim,
    NdarrayType,
    NdarrayVariableMetadata,
    Subphase,
    VarKey,
    final_carry,
    initial_carry,
    next_carry,
    years_dim,
)


class WorkSpacePrepInference_Dist_Attr:

    wsd: WorkSpacePrepInference_Dist

    def __init__(self, wsd:WorkSpacePrepInference_Dist):
        self.wsd = wsd

    @property
    def ic(self) -> InferenceComputation:
        return self.wsd.ws.ic


class WorkSpacePrepInference_Dist_General(WorkSpacePrepInference_Dist_Attr):
    """object to represent `ws.dist.general`"""

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.ic.storage_dist[item] = value

        obs_var_key = self.ic.model.mv.general_nd[item].definition_metadata.observation
        if obs_var_key:
            obs = self.ic.storage_nd[obs_var_key]
        else:
            obs = None

        self.wsd.ws.ic.storage_nd[item] = numpyro.sample(
                self.ic.model.mv.sample_sites[item],
                fn=value,
                rng_key=self.ic._split_rng_key(),
                obs=obs,
                )


class WorkSpacePrepInference_Dist_InitialCarry(WorkSpacePrepInference_Dist_Attr):
    """object to represent `ws.dist.initial_carry`"""

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct

        self.ic.storage_dist[initial_carry(item)] = value

        self.wsd.ws.ic.storage_nd[initial_carry(item)] = numpyro.sample(
                self.ic.model.mv.sample_sites[initial_carry(item)],
                fn=value,
                rng_key=self.ic._split_rng_key(),
                )

class WorkSpacePrepInference_Val_Attr:

    wsv: WorkSpacePrepInference_Val

    def __init__(self, wsv:WorkSpacePrepInference_Val):
        self.wsv = wsv

    @property
    def ic(self) -> InferenceComputation:
        return self.wsv.ws.ic


class WorkSpacePrepInference_Val_General(WorkSpacePrepInference_Val_Attr):
    """object to represent `ws.val.general`"""

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.ic.storage_nd[item] = value


class WorkSpacePrepInference_Val_InitialCarry(WorkSpacePrepInference_Val_Attr):
    """object to represent `ws.val.initial_carry`"""

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.ic.storage_nd[initial_carry(item)] = value


class WorkSpacePrepInference_Val_AnnualX(WorkSpacePrepInference_Val_Attr):
    """object to represent `ws.val.annual_X`"""

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.ic.storage_nd[item] = value


class WorkSpaceStepInference_Dist_Attr:

    wsd: WorkSpaceStepInference_Dist

    def __init__(self, wsd:WorkSpaceStepInference_Dist):
        self.wsd = wsd

    @property
    def ic(self) -> InferenceComputation:
        return self.wsd.ws.ic

    @property
    def scan_storage(self) -> ScanStorage:
        return self.wsd.ws.scan_storage


class WorkSpaceStepInference_Dist_ThisY(WorkSpaceStepInference_Dist_Attr):
    """object to represent `ws.dist.this_Y`"""

    def __getitem__(self, item:VarKey) -> Distribution:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        raise NotImplementedError()


    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        raise NotImplementedError()


class WorkSpaceStepInference_Dist_NextCarry(WorkSpaceStepInference_Dist_Attr):
    """object to represent `ws.dist.next_carry`"""

    def __getitem__(self, item:VarKey) -> Distribution:
        return self.scan_storage.next_carry_dist_d[item]

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.scan_storage.next_carry_dist_d[item] = value

        self.scan_storage.next_carry_d[item] = numpyro.sample(
                self.ic.model.mv.sample_sites[next_carry(item)],
                fn=value,
                rng_key=self.scan_storage._split_rng_key(),
                )


class WorkSpaceStepInference_Val_Attr:

    wsv: WorkSpaceStepInference_Val

    def __init__(self, wsv:WorkSpaceStepInference_Val):
        self.wsv = wsv

    @property
    def ic(self) -> InferenceComputation:
        return self.wsv.ws.ic

    @property
    def scan_storage(self) -> ScanStorage:
        return self.wsv.ws.scan_storage


class WorkSpaceStepInference_Val_General(WorkSpaceStepInference_Val_Attr):
    """object to represent `ws.val.general`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.ic.storage_nd[item]


class WorkSpaceStepInference_Val_ThisCarry(WorkSpaceStepInference_Val_Attr):
    """object to represent `ws.val.this_carry`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.scan_storage.this_carry_d[item]


class WorkSpaceStepInference_Val_NextCarry(WorkSpaceStepInference_Val_Attr):
    """object to represent `ws.val.next_carry`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        raise NotImplementedError()


    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.scan_storage.next_carry_d[item] = value


class WorkSpaceStepInference_Val_ThisX(WorkSpaceStepInference_Val_Attr):
    """object to represent `ws.val.this_X`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        raise NotImplementedError()


class WorkSpaceStepInference_Val_ThisY(WorkSpaceStepInference_Val_Attr):
    """object to represent `ws.val.this_Y`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        raise NotImplementedError()

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.scan_storage.this_Y_d[item] = value


class WorkSpacePrepInference_Attr:

    ws: WorkSpace_Prep

    def __init__(self, ws:WorkSpace_Prep):
        self.ws = ws


class WorkSpacePrepInference_Dist(WorkSpacePrepInference_Attr):
    """object to represent `ws.dist`"""

    @property
    def general(self) -> WorkSpacePrepInference_Dist_General:
        return WorkSpacePrepInference_Dist_General(self)

    @property
    def initial_carry(self) -> WorkSpacePrepInference_Dist_InitialCarry:
        return WorkSpacePrepInference_Dist_InitialCarry(self)


class WorkSpacePrepInference_Val(WorkSpacePrepInference_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpacePrepInference_Val_General:
        return WorkSpacePrepInference_Val_General(self)

    @property
    def initial_carry(self) -> WorkSpacePrepInference_Val_InitialCarry:
        return WorkSpacePrepInference_Val_InitialCarry(self)

    @property
    def annual_X(self) -> WorkSpacePrepInference_Val_AnnualX:
        return WorkSpacePrepInference_Val_AnnualX(self)


class WorkSpaceStepInference_Attr:

    ws: WorkSpace_Step

    def __init__(self, ws:WorkSpace_Step):
        self.ws = ws


class WorkSpaceStepInference_Dist(WorkSpaceStepInference_Attr):
    """object to represent `ws.dist`"""

    @property
    def this_Y(self) -> WorkSpaceStepInference_Dist_ThisY:
        return WorkSpaceStepInference_Dist_ThisY(self)

    @property
    def next_carry(self) -> WorkSpaceStepInference_Dist_NextCarry:
        return WorkSpaceStepInference_Dist_NextCarry(self)


class WorkSpaceStepInference_Val(WorkSpaceStepInference_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpaceStepInference_Val_General:
        return WorkSpaceStepInference_Val_General(self)

    @property
    def this_carry(self) -> WorkSpaceStepInference_Val_ThisCarry:
        return WorkSpaceStepInference_Val_ThisCarry(self)

    @property
    def next_carry(self) -> WorkSpaceStepInference_Val_NextCarry:
        return WorkSpaceStepInference_Val_NextCarry(self)

    @property
    def this_X(self) -> WorkSpaceStepInference_Val_ThisX:
        return WorkSpaceStepInference_Val_ThisX(self)

    @property
    def this_Y(self) -> WorkSpaceStepInference_Val_ThisY:
        return WorkSpaceStepInference_Val_ThisY(self)


class WorkSpacePostInference_Attr:

    ws: WorkSpace_Post

    def __init__(self, ws:WorkSpace_Post):
        self.ws = ws


class WorkSpacePostInference_Dist_Attr:

    wsd: WorkSpacePostInference_Dist

    def __init__(self, wsd:WorkSpacePostInference_Dist):
        self.wsd = wsd

    @property
    def ic(self) -> InferenceComputation:
        return self.wsd.ws.ic


class WorkSpacePostInference_Dist_General(WorkSpacePostInference_Dist_Attr):
    """object to represent `ws.dist.general`"""

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # check if the element, subphase defines item
        # check that the shape is correct
        #
        self.ic.storage_dist[item] = value

        # define the numpyro sample as well
        obs_var_key = self.ic.model.mv.general_nd[item].definition_metadata.observation
        if obs_var_key:
            obs = self.ic.storage_nd[obs_var_key]
        else:
            obs = None

        self.ic.storage_nd[item] = numpyro.sample(
                self.ic.model.mv.sample_sites[item],
                fn=value,
                rng_key=self.ic._split_rng_key(),
                obs=obs)


class WorkSpacePostInference_Dist(WorkSpacePostInference_Attr):
    """object to represent `ws.dist`"""

    @property
    def general(self) -> WorkSpacePostInference_Dist_General:
        return WorkSpacePostInference_Dist_General(self)


class WorkSpacePostInference_Val_Attr:

    wsv: WorkSpacePostInference_Val

    def __init__(self, wsv:WorkSpacePostInference_Val):
        self.wsv = wsv

    @property
    def ic(self) -> InferenceComputation:
        return self.wsv.ws.ic


class WorkSpacePostInference_Val_General(WorkSpacePostInference_Val_Attr):
    """object to represent `ws.val.general`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.ic.storage_nd[item]


class WorkSpacePostInference_Val_AnnualY(WorkSpacePostInference_Val_Attr):
    """object to represent `ws.val.annual_Y`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.ic.storage_nd[item]


class WorkSpacePostInference_Val(WorkSpacePostInference_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpacePostInference_Val_General:
        return WorkSpacePostInference_Val_General(self)

    @property
    def annual_Y(self) -> WorkSpacePostInference_Val_AnnualY:
        return WorkSpacePostInference_Val_AnnualY(self)


class WorkSpacePostInference_Obs(WorkSpacePostInference_Attr):
    """object to represent `ws.obs`"""

    @property
    def general(self) -> WorkSpacePostInference_Obs_General:
        return WorkSpacePostInference_Obs_General(self)


class WorkSpacePostInference_ObsValid(WorkSpacePostInference_Attr):
    """object to represent `ws.obs_valid`"""

    @property
    def general(self) -> WorkSpacePostInference_ObsValid_General:
        return WorkSpacePostInference_ObsValid_General(self)


class WorkSpace:

    year_0: int
    n_years: int
    years: jnp.ndarray
    phase: ModellingPhase
    subphase: Subphase
    comp: Computation

    def __init__(self, ic:InferenceComputation):
        super().__init__()
        self.comp = comp

    def __init__(
            self,
            year_0:int,
            n_years:int,
            phase:ModellingPhase,
            subphase:Subphase):
        self.year_0 = year_0
        self.n_years = n_years
        assert n_years >= 0
        self.years = jnp.arange(year_0, year_0 + n_years)
        self.phase = phase
        self.subphase = subphase


class WorkSpace_Prep(WorkSpace):

    @property
    def dist(self) -> WorkSpacePrepInference_Dist:
        return WorkSpacePrepInference_Dist(self)

    @property
    def val(self) -> WorkSpacePrepInference_Val:
        return WorkSpacePrepInference_Val(self)


class WorkSpace_Step(WorkSpace):

    scan_storage: ScanStorage

    def __init__(self, ic:InferenceComputation, scan_storage:ScanStorage):
        super().__init__(ic=ic)
        self.scan_storage = scan_storage

    @property
    def dist(self) -> WorkSpaceStepInference_Dist:
        return WorkSpaceStepInference_Dist(self)

    @property
    def val(self) -> WorkSpaceStepInference_Val:
        return WorkSpaceStepInference_Val(self)


class WorkSpace_Proc(WorkSpace):

    @property
    def dist(self) -> WorkSpacePostInference_Dist:
        return WorkSpacePostInference_Dist(self)

    @property
    def val(self) -> WorkSpacePostInference_Val:
        return WorkSpacePostInference_Val(self)

    @property
    def obs(self) -> WorkSpacePostInference_Obs:
        return WorkSpacePostInference_Obs(self)

    @property
    def obs_valid(self) -> WorkSpacePostInference_ObsValid:
        return WorkSpacePostInference_ObsValid(self)


class ScanStorage:

    this_carry_d: dict[VarKey, ArrayLike]
    this_X_d: dict[VarKey, ArrayLike]
    next_carry_d: dict[VarKey, ArrayLike]
    this_Y_d: dict[VarKey, ArrayLike]

    next_carry_dist_d: dict[VarKey, Distribution]
    this_Y_dist_d: dict[VarKey, Distribution]

    def __init__(self, this_carry_d, this_X_d):
        self.this_carry_d = this_carry_d
        self.this_X_d = this_X_d

        self.next_carry_d = {}
        self.this_Y_d = {}

        self.next_carry_dist_d = {}
        self.this_Y_dist_d = {}

    def carry_rng(self):
        if '__rng_key' not in self.next_carry_d:
            self.next_carry_d['__rng_key'] = self.this_carry_d['__rng_key']

    def _split_rng_key(self):
        self.next_carry_d['__rng_key'], rval = jrandom.split(
                self.next_carry_d['__rng_key'])
        return rval


class Computation:

    model: Model

    storage_dist: dict[VarKey, Distribution]
    storage_nd: dict[VarKey, ArrayLike]
    storage_obj: dict[VarKey, object]
    rng_key: ArrayLike

    def __init__(self, model:Model, seed:int):
        self.model = model

        self.storage_dist = {}
        self.storage_nd = {}
        self.storage_obj = {}
        self.rng_key = jrandom.key(seed=seed)

    def _split_rng_key(self) -> ArrayLike:
        self.rng_key, key = jrandom.split(self.rng_key)
        return key

    def _initial_carry_d(self):
        initial_carry_d = {
                var_key.carry_key: val
                for var_key, val in self.storage_nd.items()
                if isinstance(var_key, InitialCarry)
                }
        return initial_carry_d

    def _X_d(self):
        # I'm not sure what heuristic / policy to use here.
        # First try: all var_keys in general_nd whose first shape dim
        # is the inference_years_dim.
        rval = {
                var_key: self.storage_nd[var_key]
                for var_key, nvm in self.model.mv.general_nd.items()
                if (
                    nvm.definition_metadata.value_type.shape
                    and (
                        nvm.definition_metadata.value_type.shape[0]
                        in {g_inference_years_dim, posterior_years_dim})
                    and var_key in self.storage_nd)
                }
        return rval

    def run_once(self):
        inference_elements = self.model._inference_elements
        for element_id, inference_element in inference_elements.items():
            try:
                ws = WorkSpace_Prep(ic=self)
                inference_element.annual_scan_prep_fn(ws)
            except Exception as err:
                err.add_note(f'element_id={element_id}')
                raise


        def scan_step(this_carry_d, this_X_d):
            scan_storage = ScanStorage(this_carry_d, this_X_d)
            scan_storage.carry_rng()
            for element_id, inference_element in inference_elements.items():
                try:
                    ws = WorkSpace_Step(
                            ic=self,
                            scan_storage=scan_storage)
                    inference_element.annual_scan_step_fn(ws)
                except Exception as err:
                    err.add_note(f'element_id={element_id}')
                    raise
            return scan_storage.next_carry_d, scan_storage.this_Y_d

        initial_carry_d = self._initial_carry_d()
        initial_carry_d['__rng_key'] = self._split_rng_key()
        X_d = self._X_d()

        final_carry_d, Y_d = numpyro_scan(scan_step, initial_carry_d, X_d)

        # TODO: check for collisions
        self.storage_nd.update(Y_d)
        self.rng_key = final_carry_d['__rng_key']

        for element_id, inference_element in inference_elements.items():
            try:
                ws = WorkSpace_Post(ic=self)
                inference_element.annual_scan_post_fn(ws)
            except Exception as err:
                err.add_note(f'element_id={element_id}')
                raise

def run_mcmc(
        model:Model,
        seed:int,
        num_warmup:int,
        thinning:int,
        num_samples:int,
        ):
    from numpyro.infer import MCMC, NUTS
    def trace_fn():
        # the seed value is ignored
        # when running via MCMC
        obj = Computation(model=model, seed=1)
        obj.run_once()

    rng_key = jrandom.key(seed=seed)
    mcmc = MCMC(NUTS(trace_fn),
                num_warmup=num_warmup,
                thinning=thinning,
                num_samples=num_samples)
    mcmc.run(rng_key=rng_key)
    mcmc.print_summary()
    grouped_samples = mcmc.get_samples(group_by_chain=True)
    return mcmc


def subphase_from_f(f):
    if f.__name__ == 'model_element_prepare':
        return Subphase.Prep
    elif f.__name__ == 'model_element_annual_step':
        return Subphase.Step
    elif f.__name__ == 'model_element_postprocess':
        return Subphase.Proc
    else:
        raise NotImplementedError(f)

_deco_attr_define = 'model_element_define'

def define(
        var_key:VarKey, *,
        sampled:bool,
        shape:list[int|NdarrayDim],
        dtype:str='float64',
        ):
    def deco(f):
        if not hasattr(f, _deco_attr_define):
            setattr(f, _deco_attr_define, {})
        assert var_key not in f.model_element_define
        ndm = NdarrayDefinitionMetadata(
                value_type=NdarrayType(
                    shape=shape,
                    dtype=dtype),
                sampled=sampled,
                subphase=subphase_from_f(f))
        f.model_element_define[var_key] = ndm
        return f
    return deco


_deco_attr_annual = 'model_element_annual_step'

def define_annual(
        var_key:VarKey, *,
        sampled:bool,
        annual_shape:list[int|NdarrayDim],
        dtype:str='float64',
        ):
    def deco(f):
        if not hasattr(f, _deco_attr_annual):
            setattr(f, _deco_attr_annual, {})
        assert var_key not in f.model_element_annual_step
        ndm = NdarrayDefinitionMetadata(
                value_type=NdarrayType(
                    shape=[years_dim] + list(annual_shape),
                    dtype=dtype),
                sampled=sampled,
                subphase=subphase_from_f(f))
        f.model_element_annual_step[var_key] = ndm
        return f
    return deco

_deco_attr_carry = 'model_element_define_carry'

def define_carry(
        var_key:VarKey, *,
        initial_sampled:bool,
        next_sampled:bool,
        shape:list[int|NdarrayDim],
        dtype:str='float64',
        ):
    def deco(f):
        if not hasattr(f, _deco_attr_carry):
            setattr(f, _deco_attr_carry, {})
        assert var_key not in f.model_element_define_carry
        initial_ndm = NdarrayDefinitionMetadata(
                value_type=NdarrayType(
                    shape=shape,
                    dtype=dtype),
                sampled=initial_sampled,
                subphase=subphase_from_f(f))
        next_ndm = NdarrayDefinitionMetadata(
                value_type=NdarrayType(
                    shape=shape,
                    dtype=dtype),
                sampled=next_sampled,
                subphase=subphase_from_f(f))
        f.model_element_define_carry[var_key] = (
                initial_ndm, next_ndm)
        return f
    return deco



class ModelElement(BaseModel):
    """
    Inherit from this to define a model
    """

    _identifier: str|None = None

    @computed_field
    def identifier(self) -> str:
        # must be unique within a model
        return self._identifier or self.__class__.__name__

    def model_element_prepare(self, ws:WorkSpace_Prep):
        pass

    def model_element_annual_step(self, ws:WorkSpace_Step):
        pass

    def model_element_postprocess(self, ws:WorkSpace_Proc):
        pass


class Model:
    """
    Try to make the elements work mostly as a set.
    Try to make this work like a bag of elements that self-assemble.
    That way, documentation of the model can be created by documenting
    the elements.
    I'm afraid that if there is central structuring of the model, it will
    just be impossible to understand.
    Discourage thinking of elements as attributes of the model that can
    be accessed individually.
    """

    model_elements: dict[str, ModelElement]
    mv: ModelVariables

    def __init__(self, first_year:int, n_prior_years:int, n_posterior_years:int):
        self.model_elements = {}
        self.mv = ModelVariables(
                year_0=first_year,
                n_prior_years=n_prior_years,
                n_posterior_years=n_posterior_years,
                )

    def _add_define_d(self, element_id, defining_subphase, define_d):
        for var_key, ndm in define_d.items():
            self.mv.general_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=ndm,
                    defining_element_id=element_id,
                    defining_subphase=defining_subphase)
            if ndm.sampled:
                assert var_key not in self.mv.sample_sites
                self.mv.sample_sites[var_key] = str(var_key)

    def _add_define_annual_d(self, element_id, defining_subphase, annual_d):
        for var_key, ndm in annual_d.items():
            self.mv.general_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=ndm,
                    defining_element_id=element_id,
                    defining_subphase=defining_subphase)
            if ndm.sampled:
                assert var_key not in self.mv.sample_sites
                self.mv.sample_sites[var_key] = str(var_key)

            print(element_id, defining_subphase, var_key, ndm)

            if defining_subphase in (Subphase.Prep, Subphase.Step):
                this_ndm = NdarrayVariableMetadata(
                        definition_metadata=NdarrayDefinitionMetadata(
                            value_type=NdarrayType(
                                shape=ndm.value_type.shape[1:],
                                dtype=ndm.value_type.dtype),
                            sampled=ndm.sampled,
                            subphase=defining_subphase,
                            ),
                        defining_element_id=element_id,
                        defining_subphase=defining_subphase)
                if defining_subphase == Subphase.Prep:
                    self.mv.this_X_nd[var_key] = this_ndm
                else:
                    self.mv.this_Y_nd[var_key] = this_ndm

    def _add_define_carry_d(self, element_id, carry_d):
        for var_key, (initial_ndm, next_ndm) in carry_d.items():

            initial_var_key = initial_carry(var_key)

            self.mv.general_nd[initial_var_key] = NdarrayVariableMetadata(
                    definition_metadata=initial_ndm,
                    defining_element_id=element_id,
                    defining_subphase=Subphase.Prep)

            if initial_ndm.sampled:
                assert initial_var_key not in self.mv.sample_sites
                self.mv.sample_sites[initial_var_key] = str(initial_var_key)

            self.mv.this_carry_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=NdarrayDefinitionMetadata(
                        value_type=initial_ndm.value_type,
                        sampled=False,
                        subphase=Subphase.Step,
                        ),
                    defining_element_id=element_id,
                    defining_subphase=Subphase.Step)

            self.mv.next_carry_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=next_ndm,
                    defining_element_id=element_id,
                    defining_subphase=Subphase.Step)

            final_var_key = final_carry(var_key)
            if next_ndm.sampled:
                assert final_var_key not in self.mv.sample_sites
                self.mv.sample_sites[final_var_key] = str(final_var_key)

            self.mv.general_nd[final_var_key] = NdarrayVariableMetadata(
                    definition_metadata=next_ndm,
                    defining_element_id=element_id,
                    defining_subphase=Subphase.Proc)

    def add_element(self, element):
        self.model_elements[element.identifier] = element
        self._add_define_d(
                element.identifier,
                Subphase.Prep,
                getattr(element.model_element_prepare, _deco_attr_define, {}))
        assert not hasattr(element.model_element_annual_step, _deco_attr_define)
        self._add_define_d(
                element.identifier,
                Subphase.Proc,
                getattr(element.model_element_postprocess, _deco_attr_define, {}))

        self._add_define_carry_d(
                element.identifier,
                getattr(element.model_element_prepare, _deco_attr_carry, {}))
        assert not hasattr(element.model_element_annual_step, _deco_attr_carry)
        assert not hasattr(element.model_element_postprocess, _deco_attr_carry)

        self._add_define_annual_d(
                element.identifier,
                Subphase.Prep,
                getattr(element.model_element_prepare, _deco_attr_annual, {}))
        self._add_define_annual_d(
                element.identifier,
                Subphase.Step,
                getattr(element.model_element_annual_step, _deco_attr_annual, {}))
        self._add_define_annual_d(
                element.identifier,
                Subphase.Proc,
                getattr(element.model_element_postprocess, _deco_attr_annual, {}))
