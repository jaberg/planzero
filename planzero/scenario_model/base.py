# this resolves circular type references
from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Phase(str, enum.Enum):
    Prior = 'Prior'
    Posterior = 'Posterior'


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


class NamedKey(VarKeyBase, frozen=True):
    name: str
    unique_id: str

_named_key_counter = 0

def new_named_key(name:str) -> NamedKey:
    global _named_key_counter
    _named_key_counter += 1
    return NamedKey(
            name=name,
            unique_id=f'named_key_id_{_named_key_counter}')


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


class Observation(VarKeyBase, frozen=True):
    var_key_type: Literal['Observation'] = "Observation"
    prior_var_key: VarKey

def observation(var_key):
    return Observation(prior_var_key=var_key)


class ObservationValid(VarKeyBase, frozen=True):
    var_key_type: Literal['Observation'] = "Observation"
    obs_var_key: VarKey


def observation_valid(obs_var_key):
    return ObservationValid(obs_var_key=obs_var_key)


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


class NdarrayType_w_Properties(ValType, frozen=True):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shape: list[int|NdarrayDim|property]
    dtype: str = 'float64'


class NdarrayDefinitionMetadata_w_Properties(DefinitionMetadata):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value_type: NdarrayType_w_Properties


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
