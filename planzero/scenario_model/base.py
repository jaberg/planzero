# this resolves circular type references
from __future__ import annotations

import enum
from collections.abc import Callable
from typing import Any, Literal

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
    Proc = 'Proc'
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


class GroupedPosterior(VarKeyBase, frozen=True):
    var_key_type: Literal['GroupedPosterior'] = "GroupedPosterior"
    prior_var_key: VarKey


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


_ndarray_dim_counter = 0

def ndarray_dim(name:str="") -> NdarrayDim:
    global _ndarray_dim_counter
    _ndarray_dim_counter += 1
    return NdarrayDim(
            name=name,
            unique_id=f'dim_id_{_ndarray_dim_counter}')

years_dim = ndarray_dim('years_dim')
mcmc_dim = ndarray_dim('mcmc_dim')


class NdarrayType(ValType, frozen=True):

    shape: list[int|NdarrayDim]
    dtype: str = 'float64'


class DefinitionMetadata(BaseModel):

    #observation: VarKey|None = None
    sampled:bool = True
    subphase:Subphase = Subphase.Unknown


class NdarrayDefinitionMetadata(DefinitionMetadata):

    value_type: NdarrayType


def _definition_metadata_timeslice(
        dm: NdarrayDefinitionMetadata,
        ) -> NdarrayDefinitionMetadata:
    if isinstance(dm.value_type, NdarrayType):
        valtype = dm.value_type
        if (len(valtype.shape) and valtype.shape[0] == years_dim):
            return NdarrayDefinitionMetadata(
                    value_type=NdarrayType(
                        shape=valtype.shape[1:],
                        dtype=valtype.dtype),
                    # observation=dm.observation,
                    sampled=dm.sampled,
                    subphase=dm.subphase)
    raise ValueError(dm)


class DefinitionError(Exception):
    pass


class VariableMetadata(BaseModel):

    defining_element_id: str
    defining_subphase: Subphase = Subphase.Unknown


class NdarrayVariableMetadata(VariableMetadata):

    definition_metadata: NdarrayDefinitionMetadata


class ModelVariables(BaseModel):

    general_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    #initial_carry_nd: dict[VarKey, NdarrayVariableMetadata] = {}
    this_carry_nd: dict[VarKey, NdarrayVariableMetadata] = {}
    next_carry_nd: dict[VarKey, NdarrayVariableMetadata] = {}
    #final_carry_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    this_X_nd: dict[VarKey, NdarrayVariableMetadata] = {}
    #Xs_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    this_Y_nd: dict[VarKey, NdarrayVariableMetadata] = {}
    #Ys_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    #posterior_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    # sample_sites has to be a single one-to-one dictionary because
    # numpyro's mcmc works on the basis of the string values
    # to return posteriors for sample sites.
    sample_sites: dict[VarKey, str] = {}

    year_0:int

    n_prior_years:int

    n_posterior_years:int


class VarKeyRole(BaseModel, frozen=True):

    var_key: VarKey
    role: VariableRole


class ModelElementPhase:
    def carry(self, var_key,
              sample_initial:bool,
              sample_next:bool,
              **kwargs):

        initial_ndm = self._ndm_from_kwargs(sample=sample_initial, **kwargs)
        this_ndm = self._ndm_from_kwargs(sample=False, **kwargs)
        next_ndm = self._ndm_from_kwargs(sample=sample_next, **kwargs)
        final_ndm = self._ndm_from_kwargs(sample=sample_next, **kwargs)

        if next_ndm.sample:
            site_var_key = final_carry(var_key)
            assert site_var_key not in self.mv.sample_sites
            self.mv.sample_sites[site_var_key] = str(site_var_key)


def _no_op(*args, **kwargs):
    pass


class ModelInterface(ModelElementPhase):

    annual_scan_prep_fn:Callable = _no_op
    annual_scan_step_fn:Callable = _no_op
    annual_scan_step2_fn:Callable = _no_op
    annual_scan_post_fn:Callable = _no_op

    def __init__(self, **kwargs):
        super().__init__(defining_phase=ModellingPhase.Inference, **kwargs)

    def annual_scan_prep(self, *, reads:list[VarKeyRole]|None=None):
        reads = reads or []
        def decorator(fn):
            #self.reads_by_subphase[Subphase.Prep] = reads
            self.annual_scan_prep_fn = fn
            return fn
        return decorator

    def annual_scan_step(self, *, reads:list[VarKeyRole]|None=None):
        reads = reads or []
        def decorator(fn):
            #self.reads_by_subphase[Subphase.Step] = reads
            self.annual_scan_step_fn = fn
            return fn
        return decorator

    def annual_scan_step2(self, *, reads:list[VarKeyRole]|None=None):
        reads = reads or []
        def decorator(fn):
            #self.reads_by_subphase[Subphase.Step] = reads
            self.annual_scan_step2_fn = fn
            return fn
        return decorator

    def annual_scan_post(self, *, reads:list[VarKeyRole]|None=None):
        reads = reads or []
        def decorator(fn):
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
                            shape=[g_mcmc_dim] + dm.value_type.shape,
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
