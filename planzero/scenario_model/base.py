# this resolves circular type references
from __future__ import annotations

import enum
from typing import Any, Literal

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
    Annual = 'Annual'
    PosteriorCache = 'PosteriorCache'


class VarKey(BaseModel, frozen=True):
    var_key_type: Literal['Posterior',
                          'GroupedPosterior',
                          'NamedKey',
                          'InitialCarry',
                          'ThisCarry',
                          'NextCarry',
                          'FinalCarry',
                          'Observation',
                          'ObservationValid',
                          'ObservationWeight',
                          'ElementGeneralKey',
                          'ElementCarryKey',
                          'ElementAnnualKey',
                          ]

    def __lt__(self, other):
        return str(self) < str(other)


class NamedKey(VarKey, frozen=True):
    var_key_type: Literal['NamedKey'] = "NamedKey"
    name: str
    unique_id: str

_named_key_counter = 0

def _new_named_key(name:str) -> NamedKey:
    global _named_key_counter
    _named_key_counter += 1
    return NamedKey(
            name=name,
            unique_id=f'named_key_id_{_named_key_counter}')

rng_var_key = _new_named_key('rng')

class Posterior(VarKey, frozen=True):
    var_key_type: Literal['Posterior'] = "Posterior"
    prior_var_key: VarKey


class GroupedPosterior(VarKey, frozen=True):
    var_key_type: Literal['GroupedPosterior'] = "GroupedPosterior"
    prior_var_key: VarKey


class InitialCarry(VarKey, frozen=True):
    var_key_type: Literal['InitialCarry'] = "InitialCarry"
    carry_key: ElementCarryKey


class ThisCarry(VarKey, frozen=True):
    var_key_type: Literal['ThisCarry'] = "ThisCarry"
    carry_key: ElementCarryKey


class NextCarry(VarKey, frozen=True):
    var_key_type: Literal['NextCarry'] = "NextCarry"
    carry_key: ElementCarryKey


class FinalCarry(VarKey, frozen=True):
    var_key_type: Literal['FinalCarry'] = "FinalCarry"
    carry_key: ElementCarryKey


class Observation(VarKey, frozen=True):
    var_key_type: Literal['Observation'] = "Observation"
    prior_var_key: VarKey


class ObservationValid(VarKey, frozen=True):
    var_key_type: Literal['ObservationValid'] = "ObservationValid"
    obs_var_key: Observation


class ObservationWeight(VarKey, frozen=True):
    var_key_type: Literal['ObservationWeight'] = "ObservationWeight"
    obs_var_key: Observation


class ElementGeneralKey(VarKey, frozen=True):
    var_key_type: Literal['ElementGeneralKey'] = "ElementGeneralKey"
    elem_id: str
    name: str


class ElementCarryKey(VarKey, frozen=True):
    var_key_type: Literal['ElementCarryKey'] = "ElementCarryKey"
    elem_id: str
    name: str


class ElementAnnualKey(VarKey, frozen=True):
    var_key_type: Literal['ElementAnnualKey'] = "ElementAnnualKey"
    elem_id: str
    name: str


CarryKey = ElementCarryKey
GeneralKey = ElementGeneralKey
AnnualKey = ElementAnnualKey

years_key = ElementAnnualKey(elem_id='ModelBuiltIns', name='years')
scan_step_ii_key = ElementAnnualKey(elem_id="ModelBuiltIns", name='scan_step_ii')


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
    subphase:Subphase = Subphase.Unknown
    # If True, this definition is only required to be assigned during
    # Phase.Posterior. The check is skipped during Phase.Prior.
    posterior_only:bool = False


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


class UnassignedDefinitionError(DefinitionError):
    """A method declared a key with @define* but did not assign it."""
    pass


class VariableMetadata(BaseModel):

    defining_element_id: str
    defining_subphase: Subphase = Subphase.Unknown


class NdarrayVariableMetadata(VariableMetadata):

    definition_metadata: NdarrayDefinitionMetadata


ElementID = str


class AccessMetadata(BaseModel):
    """A value read declared with access_val or access_annual."""

    elem_id: ElementID
    subphase: Subphase
    var_key: VarKey
    # ndarray returned when no element in the model defines var_key.
    default: Any = None


class ModelVariables(BaseModel):
    """
    Things known about a model from just the add_element calls.
    """

    general_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    carry_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    # Annual variables declared with the define_annual decorator.
    # The metadata's defining_subphase determines the role of the variable
    # during the annual scan:
    # - Subphase.Prep: defined prior to the scan, and fed into the scan
    #   as a per-step input (the input side of this_annual).
    # - Subphase.Step: defined by the scan itself, one value per step
    #   (the output side of this_annual).
    annual_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    year_0:int

    n_prior_years:int

    n_posterior_years:int

    # Value reads declared with access_val / access_annual, used to order
    # model elements so that definers run before readers.
    accesses: list[AccessMetadata] = []
