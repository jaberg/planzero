# this resolves circular type references
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jrandom
import numpyro
from jax.lax import scan as jax_scan
from jax.typing import ArrayLike
from numpyro.contrib.control_flow import scan as numpyro_scan
from numpyro.distributions.distribution import Distribution
from pydantic import BaseModel, computed_field

from .base import (
    InitialCarry,
    ModelVariables,
    NdarrayDefinitionMetadata,
    NdarrayDim,
    NdarrayType,
    NdarrayVariableMetadata,
    Phase,
    Subphase,
    VarKey,
    final_carry,
    initial_carry,
    next_carry,
    observation,
    observation_valid,
    years_dim,
)


class WorkSpacePrep_Dist_Attr:

    wsd: WorkSpacePrep_Dist | WorkSpaceProc_Dist
    comp: Computation

    def __init__(self, wsd:WorkSpacePrep_Dist|WorkSpaceProc_Dist):
        self.wsd = wsd
        self.comp = self.wsd.ws.comp


class WorkSpacePrep_Dist_General(WorkSpacePrep_Dist_Attr):
    """object to represent `ws.dist.general`"""

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # TODO: check that the shape is correct
        # TODO: check that the item is supposed to be sampled
        self.comp.storage_dist[item] = value

        obs = self.comp.storage_nd.get(observation(item))
        obs_valid = self.comp.storage_nd.get(observation_valid(item))

        if self.comp.phase == Phase.Prior:
            if obs is None and obs_valid is None:
                # TODO: check that the shape is correct
                self.comp.storage_nd[item] = numpyro.sample(
                        self.comp.model.mv.sample_sites[item],
                        fn=value,
                        rng_key=self.comp._split_rng_key(),
                        )
            elif obs_valid is None:
                self.comp.storage_nd[item] = numpyro.sample(
                        self.comp.model.mv.sample_sites[item],
                        fn=value,
                        rng_key=self.comp._split_rng_key(),
                        obs=obs,
                        )
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()


class WorkSpacePrep_Dist_InitialCarry(WorkSpacePrep_Dist_Attr):
    """object to represent `ws.dist.initial_carry`"""

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # TODO: check that the shape is correct
        # TODO: check that the item is supposed to be sampled
        self.comp.storage_dist[initial_carry(item)] = value


        obs = self.comp.storage_nd.get(observation(initial_carry(item)))
        obs_valid = self.comp.storage_nd.get(observation_valid(initial_carry(item)))

        if self.comp.phase == Phase.Prior:
            if obs or obs_valid:
                raise NotImplementedError()
            else:
                # TODO: check that the shape is correct
                self.comp.storage_nd[initial_carry(item)] = numpyro.sample(
                        self.comp.model.mv.sample_sites[initial_carry(item)],
                        fn=value,
                        rng_key=self.comp._split_rng_key(),
                        )
        else:
            raise NotImplementedError()


class WorkSpacePrep_Val_Attr:

    wsv: WorkSpacePrep_Val | WorkSpaceProc_Val
    comp: Computation

    def __init__(self, wsv:WorkSpacePrep_Val | WorkSpaceProc_Val):
        self.wsv = wsv
        self.comp = self.wsv.ws.comp


class WorkSpacePrep_Val_General(WorkSpacePrep_Val_Attr):
    """object to represent `ws.val.general`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.comp.storage_nd[item]

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # TODO: check if the element, subphase defines item
        # TODO: check that the shape is correct
        # TODO: check that the item is not supposed to be sampled
        #
        # TODO: if item is an `observation(obs_var_key)`
        #    check that obs_var_key has nothing in storage_nd or storage_dist
        #    because observations have to be defined before sampling.
        #    ditto if item is an `observation_valid(obs_var_key)`
        self.comp.storage_nd[item] = value


class WorkSpacePrep_Val_InitialCarry(WorkSpacePrep_Val_Attr):
    """object to represent `ws.val.initial_carry`"""

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # TODO: check if the element, subphase defines item
        # TODO: check that the shape is correct
        # TODO: check that the item is not supposed to be sampled
        self.comp.storage_nd[initial_carry(item)] = value


class WorkSpaceStep_Dist_Attr:

    wsd: WorkSpaceStep_Dist
    comp: Computation
    scan_storage: ScanStorage

    def __init__(self, wsd:WorkSpaceStep_Dist):
        self.wsd = wsd
        self.comp = self.wsd.ws.comp
        self.scan_storage = wsd.ws.scan_storage


class WorkSpaceStep_Dist_ThisY(WorkSpaceStep_Dist_Attr):
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


class WorkSpaceStep_Dist_NextCarry(WorkSpaceStep_Dist_Attr):
    """object to represent `ws.dist.next_carry`"""

    def __getitem__(self, item:VarKey) -> Distribution:
        return self.scan_storage.next_carry_dist_d[item]

    def __setitem__(self, item:VarKey, value:Distribution) -> None:
        # TODO: check if the element, subphase defines item
        # TODO: check that the shape is correct
        # TODO: check that the item is supposed to be sampled
        #
        self.scan_storage.next_carry_dist_d[item] = value

        obs = self.comp.storage_nd.get(observation(item))
        obs_valid = self.comp.storage_nd.get(observation_valid(item))

        if self.comp.phase == Phase.Prior:
            if obs or obs_valid:
                raise NotImplementedError()
            else:
                # TODO: check that the shape is correct
                self.scan_storage.next_carry_d[item] = numpyro.sample(
                        self.comp.model.mv.sample_sites[next_carry(item)],
                        fn=value,
                        rng_key=self.scan_storage._split_rng_key())
        else:
            raise NotImplementedError()


class WorkSpaceStep_Val_Attr:

    wsv: WorkSpaceStep_Val
    comp: Computation
    scan_storage: ScanStorage

    def __init__(self, wsv:WorkSpaceStep_Val):
        self.wsv = wsv
        self.comp = self.wsv.ws.comp
        self.scan_storage = wsv.ws.scan_storage


class WorkSpaceStep_Val_General(WorkSpaceStep_Val_Attr):
    """object to represent `ws.val.general`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.comp.storage_nd[item]


class WorkSpaceStep_Val_ThisCarry(WorkSpaceStep_Val_Attr):
    """object to represent `ws.val.this_carry`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        return self.scan_storage.this_carry_d[item]


class WorkSpaceStep_Val_NextCarry(WorkSpaceStep_Val_Attr):
    """object to represent `ws.val.next_carry`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        # TODO: check if the element, subphase defines item
        # TODO: check that the shape is correct
        raise NotImplementedError()

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # TODO: check if the element, subphase defines item
        # TODO: check that the shape is correct
        # TODO: check that the item is not supposed to be sampled
        self.scan_storage.next_carry_d[item] = value


class WorkSpaceStep_Val_ThisX(WorkSpaceStep_Val_Attr):
    """object to represent `ws.val.this_X`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        raise NotImplementedError()


class WorkSpaceStep_Val_ThisY(WorkSpaceStep_Val_Attr):
    """object to represent `ws.val.this_Y`"""

    def __getitem__(self, item:VarKey) -> ArrayLike:
        raise NotImplementedError()

    def __setitem__(self, item:VarKey, value:ArrayLike) -> None:
        # TODO: check if the element, subphase defines item
        # TODO: check that the shape is correct
        # TODO: check that the item is not supposed to be sampled
        #
        self.scan_storage.this_Y_d[item] = value


class WorkSpacePrep_Attr:

    ws: WorkSpace_Prep

    def __init__(self, ws:WorkSpace_Prep):
        self.ws = ws


class WorkSpacePrep_Dist(WorkSpacePrep_Attr):
    """object to represent `ws.dist`"""

    @property
    def general(self) -> WorkSpacePrep_Dist_General:
        return WorkSpacePrep_Dist_General(self)

    @property
    def initial_carry(self) -> WorkSpacePrep_Dist_InitialCarry:
        return WorkSpacePrep_Dist_InitialCarry(self)


class WorkSpacePrep_Val(WorkSpacePrep_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpacePrep_Val_General:
        return WorkSpacePrep_Val_General(self)

    @property
    def initial_carry(self) -> WorkSpacePrep_Val_InitialCarry:
        return WorkSpacePrep_Val_InitialCarry(self)


class WorkSpaceStep_Attr:

    ws: WorkSpace_Step

    def __init__(self, ws:WorkSpace_Step):
        self.ws = ws


class WorkSpaceStep_Dist(WorkSpaceStep_Attr):
    """object to represent `ws.dist`"""

    @property
    def this_Y(self) -> WorkSpaceStep_Dist_ThisY:
        return WorkSpaceStep_Dist_ThisY(self)

    @property
    def next_carry(self) -> WorkSpaceStep_Dist_NextCarry:
        return WorkSpaceStep_Dist_NextCarry(self)


class WorkSpaceStep_Val(WorkSpaceStep_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpaceStep_Val_General:
        return WorkSpaceStep_Val_General(self)

    @property
    def this_carry(self) -> WorkSpaceStep_Val_ThisCarry:
        return WorkSpaceStep_Val_ThisCarry(self)

    @property
    def next_carry(self) -> WorkSpaceStep_Val_NextCarry:
        return WorkSpaceStep_Val_NextCarry(self)

    @property
    def this_X(self) -> WorkSpaceStep_Val_ThisX:
        return WorkSpaceStep_Val_ThisX(self)

    @property
    def this_Y(self) -> WorkSpaceStep_Val_ThisY:
        return WorkSpaceStep_Val_ThisY(self)


class WorkSpaceProc_Attr:

    ws: WorkSpace_Proc

    def __init__(self, ws:WorkSpace_Proc):
        self.ws = ws


class WorkSpaceProc_Dist(WorkSpaceProc_Attr):
    """object to represent `ws.dist`"""

    @property
    def general(self) -> WorkSpacePrep_Dist_General:
        return WorkSpacePrep_Dist_General(self)


class WorkSpaceProc_Val(WorkSpaceProc_Attr):
    """object to represent `ws.val`"""

    @property
    def general(self) -> WorkSpacePrep_Val_General:
        return WorkSpacePrep_Val_General(self)

    # TODO: initial_carry
    # TODO: final_carry


class WorkSpace:

    year_0: int
    n_years: int
    phase: Phase
    comp: Computation

    def __init__(
            self,
            year_0:int,
            n_years:int,
            phase:Phase,
            comp:Computation,
            ):
        self.year_0 = year_0
        self.n_years = n_years
        assert n_years >= 0
        self.phase = phase
        self.comp = comp


class WorkSpace_Prep(WorkSpace):

    @property
    def dist(self) -> WorkSpacePrep_Dist:
        return WorkSpacePrep_Dist(self)

    @property
    def val(self) -> WorkSpacePrep_Val:
        return WorkSpacePrep_Val(self)


class WorkSpace_Step(WorkSpace):

    scan_storage: ScanStorage

    def __init__(self, scan_storage:ScanStorage, **kwargs):
        super().__init__(**kwargs)
        self.scan_storage = scan_storage

    @property
    def dist(self) -> WorkSpaceStep_Dist:
        return WorkSpaceStep_Dist(self)

    @property
    def val(self) -> WorkSpaceStep_Val:
        return WorkSpaceStep_Val(self)


class WorkSpace_Proc(WorkSpace):

    @property
    def dist(self) -> WorkSpaceProc_Dist:
        return WorkSpaceProc_Dist(self)

    @property
    def val(self) -> WorkSpaceProc_Val:
        return WorkSpaceProc_Val(self)


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
    rng_key: ArrayLike
    phase: Phase
    n_saved_samples: int

    def __init__(
            self,
            model:Model,
            grouped_samples:dict[str, ArrayLike]|None,
            rng_key:ArrayLike):

        self.model = model
        self.storage_dist = {}
        self.storage_nd = {}
        self.rng_key = rng_key
        self.phase = (Phase.Prior
                      if grouped_samples is None
                      else Phase.Posterior)

        if grouped_samples is not None:
            raise NotImplementedError()

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

    def _X_d(self, n_years):
        # I'm not sure what heuristic / policy to use here.
        # First try: all var_keys in general_nd whose first shape dim
        # is the inference_years_dim.
        rval = {
                var_key: self.storage_nd[var_key]
                for var_key, nvm in self.model.mv.general_nd.items()
                if (
                    nvm.definition_metadata.value_type.shape
                    and (nvm.definition_metadata.value_type.shape[0] == years_dim)
                    and var_key in self.storage_nd)
                }

        rval['__years'] = jnp.arange(
                self.model.mv.year_0,
                self.model.mv.year_0 + n_years)
        return rval

    def _elem_items(self):
        yield from self.model.model_elements.items()

    def _run_prep(self, n_years, phase):
        for elem_id, elem in self._elem_items():
            try:
                ws = WorkSpace_Prep(
                        comp=self,
                        year_0=self.model.mv.year_0,
                        n_years=n_years,
                        phase=phase,
                        )
                elem.model_element_prepare(ws)
            except Exception as err:
                err.add_note(f'element_id={elem_id}')
                raise

    def _run_step(self, n_years, phase):

        def scan_step(this_carry_d, this_X_d):
            scan_storage = ScanStorage(this_carry_d, this_X_d)
            scan_storage.carry_rng()
            for elem_id, elem in self._elem_items():
                try:
                    ws = WorkSpace_Step(
                            comp=self,
                            year_0=self.model.mv.year_0,
                            n_years=n_years,
                            phase=phase,
                            scan_storage=scan_storage)
                    elem.model_element_annual_step(ws)
                except Exception as err:
                    err.add_note(f'element_id={elem_id}')
                    raise
            return scan_storage.next_carry_d, scan_storage.this_Y_d

        initial_carry_d = self._initial_carry_d()
        initial_carry_d['__rng_key'] = self._split_rng_key()
        X_d = self._X_d(n_years)

        if phase == Phase.Prior:
            final_carry_d, Y_d = numpyro_scan(scan_step, initial_carry_d, X_d)
        else:
            final_carry_d, Y_d = jax_scan(scan_step, initial_carry_d, X_d)

        # TODO: check for collisions
        self.storage_nd.update(Y_d)
        self.rng_key = final_carry_d['__rng_key']

    def _run_proc(self, n_years, phase):
        for elem_id, elem in self._elem_items():
            try:
                ws = WorkSpace_Proc(
                        comp=self,
                        year_0=self.model.mv.year_0,
                        n_years=n_years,
                        phase=phase,
                        )
                elem.model_element_postprocess(ws)
            except Exception as err:
                err.add_note(f'element_id={elem_id}')
                raise

    def run_phase(self, phase):
        n_years = (self.model.mv.n_prior_years
                   if phase == Phase.Prior
                   else self.model.mv.n_posterior_years)
        self._run_prep(n_years, phase)
        self._run_step(n_years, phase)
        self._run_proc(n_years, phase)

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

            if next_ndm.sampled:
                next_var_key = next_carry(var_key)
                assert next_var_key not in self.mv.sample_sites
                self.mv.sample_sites[next_var_key] = str(next_var_key)

            final_var_key = final_carry(var_key)
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
