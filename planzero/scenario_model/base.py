# this resolves circular type references
from __future__ import annotations

import enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Phase(str, enum.Enum):
    Prior = 'Prior'
    Posterior = 'Posterior'


class Subphase(str, enum.Enum):
    Prep = 'Prep'  # preparation for scan
    Step = 'Step'  # step during scan
    Proc = 'Proc'  # post-processing after scan


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
    """Declarative metadata for a definition made by a @define* decorator.

    The declaration is captured at decoration time, before the ModelElement is
    instantiated, so it cannot contain instance-specific state such as the
    element identifier. `declaration_subphase` records which ModelElement
    method carried the decorator; it is not necessarily the subphase in which
    the resulting keys become available (see VariableMetadata).

    The decorator-time form is NdarrayDefinitionDeclaration because it may
    still contain property references. add_element resolves it into
    NdarrayDefinitionMetadata, which is embedded in the registration record.
    """

    model_config = ConfigDict(frozen=True)

    #observation: VarKey|None = None
    declaration_subphase:Subphase

    # If True, this definition is only required to be assigned during
    # Phase.Posterior. The check is skipped during Phase.Prior.
    posterior_only:bool = False


class NdarrayDefinitionMetadata(DefinitionMetadata):
    """A resolved ndarray definition, stored in NdarrayVariableMetadata.

    Property-based keys and shapes have already been resolved against the
    ModelElement instance by add_element.
    """

    value_type: NdarrayType


class NdarrayType_w_Properties(ValType, frozen=True):

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    shape: list[int|NdarrayDim|property]
    dtype: str = 'float64'


class NdarrayDefinitionDeclaration(DefinitionMetadata):
    """Unresolved ndarray metadata created by a @define* decorator.

    This is the declaration-time form. Its shape may contain property
    references that cannot be resolved until a ModelElement instance exists.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    value_type: NdarrayType_w_Properties


class DefinitionError(Exception):
    pass


class UnassignedDefinitionError(DefinitionError):
    """A method declared a key with @define* but did not assign it."""


class VariableMetadata(BaseModel):
    """Registration metadata created when a ModelElement is added to a model.

    This object exists only after ModelElement instantiation. It records which
    element registered the key and the subphase in which that registered key
    becomes available.

    `defining_subphase` is distinct from the embedded definition's
    `declaration_subphase`: one decorator can produce registrations that
    become available in different subphases. For example, a carry declared on
    model_element_prepare produces an InitialCarry available in Prep, a
    carry value available in Step, and a FinalCarry available in Proc.
    """

    model_config = ConfigDict(frozen=True)

    defining_element_id: str
    defining_subphase: Subphase


class NdarrayVariableMetadata(VariableMetadata):
    """A registered ndarray variable: registration identity plus its resolved
    definition contract.
    """

    definition_metadata: NdarrayDefinitionMetadata


ElementID = str
