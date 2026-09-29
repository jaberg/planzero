# this resolves circular type references
from __future__ import annotations

import enum
from collections.abc import Callable
from typing import Any, Literal

from numpyro.contrib.control_flow import scan as numpyro_scan
import jax.numpy as jnp
import jax.random as jrandom
from jax.typing import ArrayLike
import numpyro
from numpyro.distributions.distribution import Distribution
from pydantic import BaseModel, ConfigDict, computed_field


class ModellingPhase(str, enum.Enum):
    Inference = 'Inference'
    Analysis = 'Analysis'


class Subphase(str, enum.Enum):
    Unknown = 'Unknown'
    Prep = 'Prep'
    Internal_Between_Prep_and_Step = 'Internal Subphase between Prep and Step'
    Step = 'Step'
    Internal_Between_Step_and_Post = 'Internal Subphase between Step and Post'
    Post = 'Post'
    Internal_After_Post = 'Internal Subphase after Post'


class VariableRole(str, enum.Enum):
    """
    Variables in PlanZero scenario models have one or more of these roles,
    which determine when they may be written, read, stored, retrieved, etc.
    by various stages of computation.
    """
    General = 'General'
    AnnualCarry = 'AnnualCarry'
    AnnualX = 'AnnualX'
    AnnualY = 'AnnualY'
    PosteriorCache = 'PosteriorCache'


class VarKeyBase(BaseModel, frozen=True):

    # let AnnualEmissionRate VarKey declare
    # incompatibility with everything but General
    # ... what about Ys though?
    # TODO: Scoping and Storage -- can Xs and Ys be part of general?
    # Maybe the answer is a simple "yes", Xs and Ys can be accessed
    # via ws.general, and are restricted subsets of ws.general
    disallowed_roles: frozenset[VariableRole] = frozenset()


VarKey = (str | VarKeyBase)


class Posterior(VarKeyBase, frozen=True):
    var_key_type: Literal['Posterior'] = "Posterior"
    prior_var_key: VarKey


def posterior(var_key:VarKey) -> Posterior:
    return Posterior(prior_var_key=var_key)


class InitialCarry(VarKeyBase, frozen=True):
    var_key_type: Literal['InitialCarry'] = "InitialCarry"
    carry_key: VarKey


def initial_carry(var_key: VarKey) -> InitialCarry:
    return InitialCarry(carry_key=var_key)


class NextCarry(VarKeyBase, frozen=True):
    var_key_type: Literal['NextCarry'] = "NextCarry"
    carry_key: VarKey


def next_carry(var_key: VarKey) -> NextCarry:
    return NextCarry(carry_key=var_key)


class FinalCarry(VarKeyBase, frozen=True):
    var_key_type: Literal['FinalCarry'] = "FinalCarry"
    carry_key: VarKey


def final_carry(var_key: VarKey) -> FinalCarry:
    return FinalCarry(carry_key=var_key)


class ValType(BaseModel, frozen=True):
    pass


class NdarrayDim(BaseModel, frozen=True):
    name: str
    unique_id: str


ndarray_dim_counter = 0
def ndarray_dimension(name):
    global ndarray_dim_counter
    ndarray_dim_counter += 1
    return NdarrayDim(
            name=name,
            unique_id=f'dim_{ndarray_dim_counter}')


class NdarrayType(ValType, frozen=True):
    shape: list[int|NdarrayDim]
    dtype: str = 'float64'


class DefinitionMetadata(BaseModel):

    observation: VarKey|None = None

    sample:bool = True

    subphase:Subphase = Subphase.Unknown


class NdarrayDefinitionMetadata(DefinitionMetadata):

    value_type: NdarrayType


def _definition_metadata_timeslice(
        dm: NdarrayDefinitionMetadata,
        inference_years_dim:NdarrayDim,
        ) -> NdarrayDefinitionMetadata:
    if isinstance(dm.value_type, NdarrayType):
        valtype = dm.value_type
        if (len(valtype.shape) and valtype.shape[0] == inference_years_dim):
            return NdarrayDefinitionMetadata(
                    value_type=NdarrayType(
                        shape=valtype.shape[1:],
                        dtype=valtype.dtype),
                    observation=dm.observation,
                    sample=dm.sample,
                    subphase=dm.subphase)
    raise ValueError(dm)


class VarAction(BaseModel):
    pass


class VarActionRead(VarAction):
    action_type: Literal['READ']


class VarActionWrite(VarAction):
    action_type: Literal['WRITE']


VarActionUnion = (
        VarActionRead
        | VarActionWrite)


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

    ws: InferenceWorkSpace_Prep

    def __init__(self, ws:InferenceWorkSpace_Prep):
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

    ws: InferenceWorkSpace_Step

    def __init__(self, ws:InferenceWorkSpace_Step):
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

    ws: InferenceWorkSpace_Post

    def __init__(self, ws:InferenceWorkSpace_Post):
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


class WorkSpace:
    pass


class InferenceWorkSpace(WorkSpace):
    ic: InferenceComputation

    def __init__(self, ic:InferenceComputation):
        self.ic = ic


class InferenceWorkSpace_Prep(InferenceWorkSpace):

    @property
    def dist(self) -> WorkSpacePrepInference_Dist:
        return WorkSpacePrepInference_Dist(self)

    @property
    def val(self) -> WorkSpacePrepInference_Val:
        return WorkSpacePrepInference_Val(self)


class InferenceWorkSpace_Step(InferenceWorkSpace):

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


class InferenceWorkSpace_Post(InferenceWorkSpace):

    @property
    def dist(self) -> WorkSpacePostInference_Dist:
        return WorkSpacePostInference_Dist(self)

    @property
    def val(self) -> WorkSpacePostInference_Val:
        return WorkSpacePostInference_Val(self)




def _workspace_no_op(ws: WorkSpace) -> None:
    pass


class VarKeyRole(BaseModel, frozen=True):

    var_key: VarKey
    role: VariableRole


if 0:
    def General(var_key: VarKey) -> VarKeyRole:
        return VarKeyRole(var_key=var_key, role=VariableRole.General)


    def AnnualX(var_key: VarKey) -> VarKeyRole:
        return VarKeyRole(var_key=var_key, role=VariableRole.AnnualX)


    def AnnualCarry(var_key: VarKey) -> VarKeyRole:
        return VarKeyRole(var_key=var_key, role=VariableRole.AnnualCarry)


    def AnnualY(var_key: VarKey) -> VarKeyRole:
        return VarKeyRole(var_key=var_key, role=VariableRole.AnnualY)


class ModelElementPhase:

    element_id: str
    defining_phase: ModellingPhase
    inference_years_dim: NdarrayDim
    mcmc_dim: NdarrayDim
    mv: ModelVariables

    def __init__(
            self,
            element_id: str,
            defining_phase: ModellingPhase,
            inference_years_dim: NdarrayDim,
            mcmc_dim: NdarrayDim,
            mv: ModelVariables,
            ):
        self.element_id = element_id
        self.defining_phase = defining_phase
        self.inference_years_dim = inference_years_dim
        self.mcmc_dim = mcmc_dim
        self.mv = mv

    def _ndm_from_kwargs(
            self,
            sample:bool,
            shape:list[int|NdarrayDim]|None=None,
            dtype:str|None=None,
            observation:VarKey|None=None,
            definition_subphase:Subphase=Subphase.Unknown,
            ) -> NdarrayDefinitionMetadata:
        if shape is None and dtype is None:
            raise NotImplementedError()
        else:
            if dtype:
                assert shape is not None
            ndm = NdarrayDefinitionMetadata(
                    value_type=NdarrayType(
                        shape=shape or [],
                        dtype=dtype or 'float64'),
                    sample=sample,
                    observation=observation,
                    subphase=definition_subphase)
            return ndm

    def general(self, var_key, **kwargs):
        ndm = self._ndm_from_kwargs(**kwargs)
        self.mv.general_nd[var_key] = NdarrayVariableMetadata(
                definition_metadata=ndm,
                defining_element_id=self.element_id,
                defining_phase=self.defining_phase)
        if ndm.sample:
            assert var_key not in self.mv.sample_sites
            self.mv.sample_sites[var_key] = str(var_key)

    def annual_X(self, var_key, **kwargs):
        ndm = self._ndm_from_kwargs(**kwargs)

        self.mv.Xs_nd[var_key] = NdarrayVariableMetadata(
                definition_metadata=ndm,
                defining_element_id=self.element_id,
                defining_phase=ModellingPhase.Inference,
                defining_subphase=Subphase.Prep)
        if ndm.sample:
            assert var_key not in self.mv.sample_sites
            self.mv.sample_sites[var_key] = str(var_key)
        try:
            self.mv.this_X_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=_definition_metadata_timeslice(
                        ndm,
                        inference_years_dim=self.inference_years_dim),
                    defining_element_id='__internal__',
                    defining_phase=ModellingPhase.Inference,
                    defining_subphase=Subphase.Internal_Between_Prep_and_Step)
        except Exception as err:
            err.add_note(f'var_key={var_key}')
            err.add_note(f'element_id={self.element_id}')
            raise

        # create an alias in general for convenience
        # and also so that posterior just works on general
        assert var_key not in self.mv.general_nd
        self.mv.general_nd[var_key] = self.mv.Xs_nd[var_key]

    def annual_Y(self, var_key, **kwargs):
        ndm = self._ndm_from_kwargs(**kwargs)
        try:
            self.mv.this_Y_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=_definition_metadata_timeslice(
                        ndm,
                        inference_years_dim=self.inference_years_dim),
                    defining_element_id=self.element_id,
                    defining_phase=ModellingPhase.Inference,
                    defining_subphase=Subphase.Internal_Between_Prep_and_Step)
        except Exception as err:
            err.add_note(f'var_key={var_key}')
            err.add_note(f'element_id={self.element_id}')
            raise

        self.mv.Ys_nd[var_key] = NdarrayVariableMetadata(
                definition_metadata=ndm,
                defining_element_id='__internal__',
                defining_phase=ModellingPhase.Inference,
                defining_subphase=Subphase.Internal_Between_Step_and_Post)

        # create an alias in general for convenience
        # and also so that posterior just works on general
        assert var_key not in self.mv.general_nd
        self.mv.general_nd[var_key] = self.mv.Ys_nd[var_key]

    def carry(self, var_key,
              sample_initial:bool,
              sample_next:bool,
              **kwargs):

        initial_ndm = self._ndm_from_kwargs(sample=sample_initial, **kwargs)
        this_ndm = self._ndm_from_kwargs(sample=False, **kwargs)
        next_ndm = self._ndm_from_kwargs(sample=sample_next, **kwargs)
        final_ndm = self._ndm_from_kwargs(sample=sample_next, **kwargs)

        if 'observation' in kwargs:
            raise NotImplementedError()

        self.mv.initial_carry_nd[var_key] = NdarrayVariableMetadata(
                definition_metadata=initial_ndm,
                defining_element_id=self.element_id,
                defining_phase=self.defining_phase,
                defining_subphase=Subphase.Prep)
        if initial_ndm.sample:
            site_var_key = initial_carry(var_key)
            assert site_var_key not in self.mv.sample_sites
            self.mv.sample_sites[site_var_key] = str(site_var_key)

        self.mv.this_carry_nd[var_key] = NdarrayVariableMetadata(
                definition_metadata=this_ndm,
                defining_element_id='__internal__',
                defining_phase=self.defining_phase,
                defining_subphase=Subphase.Internal_Between_Prep_and_Step)

        self.mv.next_carry_nd[var_key] = NdarrayVariableMetadata(
                definition_metadata=next_ndm,
                defining_element_id=self.element_id,
                defining_phase=self.defining_phase,
                defining_subphase=Subphase.Step)

        if next_ndm.sample:
            site_var_key = next_carry(var_key)
            assert site_var_key not in self.mv.sample_sites
            self.mv.sample_sites[site_var_key] = str(site_var_key)

        self.mv.final_carry_nd[var_key] = NdarrayVariableMetadata(
                definition_metadata=final_ndm,
                defining_element_id='__internal__',
                defining_phase=self.defining_phase,
                defining_subphase=Subphase.Internal_Between_Step_and_Post)

        if next_ndm.sample:
            site_var_key = final_carry(var_key)
            assert site_var_key not in self.mv.sample_sites
            self.mv.sample_sites[site_var_key] = str(site_var_key)


class InferenceElement(ModelElementPhase):

    annual_scan_prep_fn:Callable[[InferenceWorkSpace_Prep], None] = _workspace_no_op
    annual_scan_step_fn:Callable[[InferenceWorkSpace_Step], None] = _workspace_no_op
    annual_scan_post_fn:Callable[[InferenceWorkSpace_Post], None] = _workspace_no_op

    def __init__(self, **kwargs):
        super().__init__(defining_phase=ModellingPhase.Inference, **kwargs)

    def annual_scan_prep(self, *, reads:list[VarKeyRole]|None=None):
        reads = reads or []
        def decorator(fn: Callable[[InferenceWorkSpace_Prep], None]):
            #self.reads_by_subphase[Subphase.Prep] = reads
            self.annual_scan_prep_fn = fn
            return fn
        return decorator

    def annual_scan_step(self, *, reads:list[VarKeyRole]|None=None):
        reads = reads or []
        def decorator(fn: Callable[[InferenceWorkSpace_Step], None]):
            #self.reads_by_subphase[Subphase.Step] = reads
            self.annual_scan_step_fn = fn
            return fn
        return decorator

    def annual_scan_post(self, *, reads:list[VarKeyRole]|None=None):
        reads = reads or []
        def decorator(fn: Callable[[InferenceWorkSpace_Post], None]):
            #self.reads_by_subphase[Subphase.Post] = reads
            self.annual_scan_post_fn = fn
            return fn
        return decorator

    def _add_posterior_variable(self, var_key, vm):
        dm = vm.definition_metadata
        print(var_key)
        assert var_key not in self.mv.posterior_nd
        if (vm.defining_phase == ModellingPhase.Inference
            and dm.sample
            and dm.observation is None):
            self.mv.posterior_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=NdarrayDefinitionMetadata(
                        value_type=NdarrayType(
                            shape=[self.mcmc_dim] + dm.value_type.shape,
                            dtype=dm.value_type.dtype),
                        observation=None,
                        sample=False),
                    defining_element_id='__internal__',
                    defining_phase=ModellingPhase.Inference,
                    defining_subphase=Subphase.Internal_After_Post)

    def _add_posterior_variables(self):
        for var_key, vm in self.mv.general_nd.items():
            self._add_posterior_variable(var_key, vm)

        for var_key, vm in self.mv.initial_carry_nd.items():
            self._add_posterior_variable(initial_carry(var_key), vm)

        for var_key, vm in self.mv.final_carry_nd.items():
            self._add_posterior_variable(final_carry(var_key), vm)


class ModelElementAnalysis(ModelElementPhase):

    defines: dict[VarKeyRole, DefinitionMetadata]

    annual_scan_prep:Callable[[WorkSpace_AnnualScanPrep_Analysis], None] = _workspace_no_op
    annual_scan_prep_reads: list[VarKeyRole] = []

    annual_scan_step:Callable[[WorkSpace_AnnualScanStep_Analysis], None] = _workspace_no_op
    annual_scan_step_reads: list[VarKeyRole] = []

    annual_scan_post:Callable[[WorkSpace_AnnualScanPost_Analysis], None] = _workspace_no_op
    annual_scan_post_reads: list[VarKeyRole] = []



class ModelElement(BaseModel):

    _identifier: str|None = None

    @computed_field
    def identifier(self) -> str:
        # must be unique within a model
        return self._identifier or self.__class__.__name__

    def inference(self, ie:InferenceElement) -> None:
        raise NotImplementedError()

    def analysis(self, inference_years_dim:NdarrayDim) -> ModelElementAnalysis:
        raise NotImplementedError()


class ModelVariables(BaseModel):

    general_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    initial_carry_nd: dict[VarKey, NdarrayVariableMetadata] = {}
    this_carry_nd: dict[VarKey, NdarrayVariableMetadata] = {}
    next_carry_nd: dict[VarKey, NdarrayVariableMetadata] = {}
    final_carry_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    this_X_nd: dict[VarKey, NdarrayVariableMetadata] = {}
    Xs_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    this_Y_nd: dict[VarKey, NdarrayVariableMetadata] = {}
    Ys_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    posterior_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    # sample_sites has to be a single one-to-one dictionary because
    # numpyro's mcmc works on the basis of the string values
    # to return posteriors for sample sites.
    sample_sites: dict[VarKey, str] = {}


class Model(BaseModel):
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
    model_config = ConfigDict(strict=True)

    model_elements: dict[str, ModelElement] = {}
    _inference_elements: dict[str, InferenceElement] = {}

    inference_years_dim: NdarrayDim
    mcmc_dim: NdarrayDim
    mv: ModelVariables

    def model_post_init(self, context: Any) -> None:
        self._inference_elements = {}  #necessary to avoid mutable shared dict?

    def add_element(self, element):
        self.model_elements[element.identifier] = element
        inference_element = InferenceElement(
                element_id=element.identifier,
                inference_years_dim=self.inference_years_dim,
                mcmc_dim=self.mcmc_dim,
                mv=self.mv,
                )
        element.inference(inference_element)
        inference_element._add_posterior_variables()
        self._inference_elements[element.identifier] = inference_element


def AnnualScanModel():
    return Model(
            inference_years_dim=ndarray_dimension('inference_years_dim'),
            mcmc_dim=ndarray_dimension('mcmc_dim'),
            mv=ModelVariables(),
            )


class DefinitionError(Exception):
    pass


class VariableMetadata(BaseModel):

    defining_element_id: str
    defining_phase: ModellingPhase
    defining_subphase: Subphase = Subphase.Unknown


class NdarrayVariableMetadata(VariableMetadata):

    definition_metadata: NdarrayDefinitionMetadata


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


class InferenceComputation:

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
                        == self.model.inference_years_dim)
                    and var_key in self.storage_nd)
                }
        return rval

    def run_once(self):
        inference_elements = self.model._inference_elements
        for element_id, inference_element in inference_elements.items():
            try:
                ws = InferenceWorkSpace_Prep(ic=self)
                inference_element.annual_scan_prep_fn(ws)
            except Exception as err:
                err.add_note(f'element_id={element_id}')
                raise


        def scan_step(this_carry_d, this_X_d):
            scan_storage = ScanStorage(this_carry_d, this_X_d)
            scan_storage.carry_rng()
            for element_id, inference_element in inference_elements.items():
                try:
                    ws = InferenceWorkSpace_Step(
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
        print('final carry')
        print(final_carry_d)
        print('Y_d')
        print(Y_d)

        # TODO: check for collisions
        self.storage_nd.update(Y_d)
        self.rng_key = final_carry_d['__rng_key']

        for element_id, inference_element in inference_elements.items():
            try:
                ws = InferenceWorkSpace_Post(ic=self)
                inference_element.annual_scan_post_fn(ws)
            except Exception as err:
                err.add_note(f'element_id={element_id}')
                raise

    @classmethod
    def run_mcmc(
            cls,
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
            obj = cls(model=model, seed=1)
            obj.run_once()

        rng_key = jrandom.key(seed=seed)
        mcmc = MCMC(NUTS(trace_fn),
                    num_warmup=num_warmup,
                    thinning=thinning,
                    num_samples=num_samples)
        mcmc.run(rng_key=rng_key)
        mcmc.print_summary()
        grouped_samples = mcmc.get_samples(group_by_chain=True)
        for key, val in grouped_samples.items():
            print(key)
            print(val.shape)
            print()
        return mcmc


class AnalysisComputation:

    model: Model

    def __init__(self, model:Model):
        self.model = model
