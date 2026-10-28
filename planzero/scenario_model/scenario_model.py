from __future__ import annotations

import hashlib
import heapq
from typing import Any, Literal, TypeAlias, TypeVar

import jax.numpy as jnp
from jax.typing import ArrayLike
from pydantic import BaseModel, ConfigDict

from .base import (
    DefinitionError,
    ElementAnnualKey,
    ElementCarryKey,
    ElementGeneralKey,
    ElementID,
    FinalCarry,
    InitialCarry,
    NdarrayDefinitionDeclaration,
    NdarrayDefinitionMetadata,
    NdarrayDim,
    NdarrayType,
    NdarrayType_w_Properties,
    NdarrayVariableMetadata,
    NextCarry,
    Subphase,
    ThisCarry,
    VarKey,
    scan_step_ii_key,
    years_dim,
    years_key,
)

VersionID: TypeAlias = int | float | str | tuple["VersionID", ...]


def hash_version_id(version_id: VersionID, n_chars=32) -> str:
    """Returns a stable, cross-session SHA-256 hex string."""
    # 1. Encode the string to bytes
    encoded_data = str(version_id).encode('utf-8')
    assert n_chars % 2 == 0

    # 2. Generate and return the hexadecimal digest
    rval = hashlib.shake_128(encoded_data).hexdigest(n_chars // 2)
    assert len(rval) == n_chars
    return rval



class UnresolvedAccessMetadata(BaseModel, frozen=True):
    """A value read declared with access_val or access_annual."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    elem_id: ElementID
    subphase: Subphase
    var_key_or_prop: VarKey|property
    # ndarray returned when no element in the model defines var_key.
    default: Any = None
    annual:bool
    which_carry: Literal['neither', 'this', 'next'] = "neither"


class ResolvedAccessMetadata(BaseModel, frozen=True):
    """A value read declared with access_val or access_annual."""

    elem_id: ElementID
    subphase: Subphase
    var_key: VarKey
    # ndarray returned when no element in the model defines var_key.
    default: Any = None
    which_carry: Literal['neither', 'this', 'next']


class ModelVariables(BaseModel):
    """Registration state accumulated from add_element calls.

    Each key maps to NdarrayVariableMetadata: registration identity plus the
    resolved definition contract. Decorator declarations are not stored here
    until an element is added to the model.
    """

    general_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    carry_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    # Annual variables declared with the define_annual decorator.
    # The metadata's defining_subphase determines the role of the variable
    # during the annual scan:
    # - Subphase.Prep: defined prior to the scan, available as a per-step
    #   input to scan step functions
    # - Subphase.Step: defined by the scan itself, one value per step
    #   (the output side of this_annual). Annual variables defined
    #   in this way *also* have a per-step NdarrayVariableMetadata in
    #   self.this_annual_nd.
    # - Subphase.Proc: defined after to the scan
    #
    # Invariant: the first element of the shape is always years_dim
    annual_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    # per-step metadata for annual_nd var_keys defined or potentially accessed
    # in the annual scan.
    this_annual_nd: dict[VarKey, NdarrayVariableMetadata] = {}

    year_0:int

    n_prior_years:int

    n_posterior_years:int

    # one element per @access decorator
    # properties remain unresolved until Computation
    unresolved_accesses: list[UnresolvedAccessMetadata] = []


class ModelElement:
    """
    Inherit from this to define a model element.
    """

    _identifier: str|None = None

    @property
    def identifier(self) -> str:
        # must be unique within a model
        return self._identifier or self.__class__.__name__

    def general_key(self, name:str) -> ElementGeneralKey:
        if name == "":
            raise NotImplementedError("name should not be empty")
        return ElementGeneralKey(
                elem_id=self.identifier,
                name=name)

    def annual_key(self, name:str) -> ElementAnnualKey:
        if name == "":
            raise NotImplementedError("name should not be empty")
        return ElementAnnualKey(
                elem_id=self.identifier,
                name=name)

    def carry_key(self, name:str) -> ElementCarryKey:
        if name == "":
            raise NotImplementedError("name should not be empty")
        return ElementCarryKey(
                elem_id=self.identifier,
                name=name)

    @property
    def version_id(self) -> VersionID:
        raise NotImplementedError(self)

    def model_element_prepare(self, ws):
        pass

    def model_element_annual_step(self, ws):
        pass

    def model_element_postprocess(self, ws):
        pass


def _subphase_from_f(f):
    if f.__name__ == 'model_element_prepare':
        return Subphase.Prep
    elif f.__name__ == 'model_element_annual_step':
        return Subphase.Step
    elif f.__name__ == 'model_element_postprocess':
        return Subphase.Proc
    else:
        raise NotImplementedError(f)


_deco_attr_access_val = 'model_element_access_val'
_deco_attr_access_annual = 'model_element_access_annual'
_deco_attr_access_this_carry = 'model_element_access_this_carry'
_deco_attr_access_next_carry = 'model_element_access_next_carry'


def _access_deco(
        var_key:str|VarKey|property,
        *,
        default:ArrayLike|None,
        deco_attr:str,
        ):
    def deco(f):
        if not hasattr(f, deco_attr):
            setattr(f, deco_attr, {})
        access_d = getattr(f, deco_attr)
        # doesn't work for property, which won't have resolved yet
        assert var_key not in access_d
        subphase=_subphase_from_f(f)
        access_d.setdefault(subphase, {})[var_key] = {
                'default': None if default is None else jnp.asarray(default),
                }
        return f
    return deco


def access_val(
        var_key:str|VarKey|property,
        *,
        default:ArrayLike|None=None,
        ):
    """Declare that a model element reads a general value or distribution.

    The declaration is used to order model elements so that any element that
    defines a key runs before an element that reads it.

    If default is not None, it is returned when the key has no definer
    anywhere in the ScenarioModel. Defaults apply to value reads
    (ws.val.general) only, not to distribution reads. If some element does
    define the key, the default is ignored and a missing value still raises.
    All defaults declared for one key in a model must broadcast to a common
    shape and be numerically equal there.
    """
    return _access_deco(
            var_key,
            default=default,
            deco_attr=_deco_attr_access_val)


def access_annual(
        var_key:str|VarKey|property,
        *,
        annual_shape:list[int|NdarrayDim]|None=None,
        default:ArrayLike|None=None,
        ):
    """Declare that a model element reads an annual value or distribution.

    Analogous to access_val, but the string form of var_key resolves to an
    ElementAnnualKey, and the declaration may name an annual variable defined
    either before the scan or by the scan step itself.

    If default is not None, it is returned when the key has no definer
    anywhere in the ScenarioModel. Defaults apply to value reads
    (ws.val.annual and ws.val.this_annual) only, not to distribution reads.
    During a scan step the default is returned literally as the value for
    that step, so it should have the per-step shape. If some element does
    define the key, the default is ignored and a missing value still raises.
    All defaults declared for one key in a model must broadcast to a common
    shape and be numerically equal there.
    """
    return _access_deco(
            var_key,
            default=default,
            deco_attr=_deco_attr_access_annual)


def access_next_carry(
        var_key:str|VarKey|property,
        *,
        shape:list[int|NdarrayDim]|None=None,
        default:ArrayLike|None=None,
        ):
    """Declare that a model element reads a carry value or distribution.

    Analogous to access_val, but the string form of var_key resolves to an
    ElementAnnualKey, and the declaration may name an carry variable defined
    either before the scan or by the scan step itself.

    If default is not None, it is returned when the key has no definer
    anywhere in the ScenarioModel. Defaults apply to value reads
    (ws.val.this_carry and ws.val.next_carry) only, not to distribution reads.
    During a scan step the default is returned literally as the value for
    that step, so it should have the per-step shape. If some element does
    define the key, the default is ignored and a missing value still raises.
    All defaults declared for one key in a model must broadcast to a common
    shape and be numerically equal there.
    """
    return _access_deco(
            var_key,
            default=default,
            deco_attr=_deco_attr_access_next_carry)


def access_this_carry(
        var_key:str|VarKey|property,
        *,
        shape:list[int|NdarrayDim]|None=None,
        default:ArrayLike|None=None,
        ):
    """Declare that a model element reads a carry value or distribution.

    Analogous to access_val, but the string form of var_key resolves to an
    ElementAnnualKey, and the declaration may name an carry variable defined
    either before the scan or by the scan step itself.

    If default is not None, it is returned when the key has no definer
    anywhere in the ScenarioModel. Defaults apply to value reads
    (ws.val.this_carry and ws.val.next_carry) only, not to distribution reads.
    During a scan step the default is returned literally as the value for
    that step, so it should have the per-step shape. If some element does
    define the key, the default is ignored and a missing value still raises.
    All defaults declared for one key in a model must broadcast to a common
    shape and be numerically equal there.
    """
    return _access_deco(
            var_key,
            default=default,
            deco_attr=_deco_attr_access_this_carry)


_deco_attr_define = 'model_element_define'

def define(
        var_key:str|VarKey|property, *,
        prior_shape:list[int|NdarrayDim|property],
        dtype:str='float64',
        posterior_only:bool=False,
        ):
    def deco(f):
        if not hasattr(f, _deco_attr_define):
            setattr(f, _deco_attr_define, {})
        # doesn't work for property, which won't have resolved yet
        assert var_key not in f.model_element_define
        ndm = NdarrayDefinitionDeclaration(
                value_type=NdarrayType_w_Properties(
                    shape=prior_shape,
                    dtype=dtype),
                declaration_subphase=_subphase_from_f(f),
                posterior_only=posterior_only)
        f.model_element_define[var_key] = ndm
        return f
    return deco


_deco_attr_annual = 'model_element_define_annual'

def define_annual(
        var_key:str|VarKey|property, *,
        annual_shape:list[int|NdarrayDim|property],
        dtype:str='float64',
        posterior_only:bool=False,
        ):
    """Declare an annual (timeseries) variable, i.e. one value per scan step.

    The decorated method determines the role of the variable, as reflected
    by ws.this_annual during a scan step:

    - on model_element_prepare: the variable is defined prior to the scan.
      The whole timeseries is computed during prepare, and the scan feeds
      it in one step at a time (readable as ws.val.this_annual during
      a step).
    - on model_element_annual_step: the variable is defined by the scan
      step itself (readable/writable as ws.val.this_annual, and its
      distribution as ws.dist.this_annual, during a step).
    """
    def deco(f):
        if not hasattr(f, _deco_attr_annual):
            setattr(f, _deco_attr_annual, {})
        # doesn't work for property, which won't have resolved yet
        assert var_key not in getattr(f, _deco_attr_annual)
        ndm = NdarrayDefinitionDeclaration(
                value_type=NdarrayType_w_Properties(
                    shape=[years_dim] + list(annual_shape),
                    dtype=dtype),
                declaration_subphase=_subphase_from_f(f),
                posterior_only=posterior_only)
        getattr(f, _deco_attr_annual)[var_key] = ndm
        return f
    return deco

_deco_attr_carry = 'model_element_define_carry'

def define_carry(
        var_key:str|VarKey|property, *,
        shape:list[int|NdarrayDim|property],
        dtype:str='float64',
        posterior_only:bool=False,
        ):
    def deco(f):
        if not hasattr(f, _deco_attr_carry):
            setattr(f, _deco_attr_carry, {})
        # doesn't work for property, which won't have resolved yet
        assert var_key not in f.model_element_define_carry
        initial_ndm = NdarrayDefinitionDeclaration(
                value_type=NdarrayType_w_Properties(
                    shape=shape,
                    dtype=dtype),
                declaration_subphase=_subphase_from_f(f),
                posterior_only=posterior_only)
        next_ndm = NdarrayDefinitionDeclaration(
                value_type=NdarrayType_w_Properties(
                    shape=shape,
                    dtype=dtype),
                declaration_subphase=_subphase_from_f(f),
                posterior_only=posterior_only)
        getattr(f, _deco_attr_carry)[var_key] = (initial_ndm, next_ndm)
        return f
    return deco


def _resolve_properties_ndarray_definition(
        element,
        ndmp:NdarrayDefinitionDeclaration,
        ) -> NdarrayDefinitionMetadata:

    def shape_elem_fn(shape_elem:int|NdarrayDim|property) -> int|NdarrayDim:
        if isinstance(shape_elem, (int, NdarrayDim)):
            return shape_elem
        else:
            return shape_elem.__get__(element, type(element))

    def shape_fn(shape) -> list[int|NdarrayDim]:
        return [shape_elem_fn(shape_elem) for shape_elem in shape]

    def value_type_fn(value_type) -> NdarrayType:
        return NdarrayType(
                shape=shape_fn(value_type.shape),
                dtype=value_type.dtype)

    return NdarrayDefinitionMetadata(
            value_type=value_type_fn(ndmp.value_type),
            declaration_subphase=ndmp.declaration_subphase,
            posterior_only=ndmp.posterior_only)


def _check_default_consistency(
        var_key:VarKey,
        existing:jnp.ndarray,
        new:jnp.ndarray,
        ) -> None:
    """Defaults declared for one key must agree where the read happens.

    Multiple access_val / access_annual declarations may name the same key
    with a default. Those defaults must broadcast to a common shape and be
    numerically equal there, so the default read does not depend on which
    element (or subphase) performs the read.
    """
    try:
        shape = jnp.broadcast_shapes(jnp.shape(existing), jnp.shape(new))
    except ValueError:
        raise DefinitionError(
                f'defaults for {var_key} are not broadcast-compatible:'
                f' shapes {jnp.shape(existing)} and {jnp.shape(new)}')
    existing_b = jnp.broadcast_to(existing, shape)
    new_b = jnp.broadcast_to(new, shape)
    if not bool(jnp.array_equal(existing_b, new_b, equal_nan=True)):
        raise DefinitionError(
                f'defaults for {var_key} are broadcast-compatible but not'
                f' numerically equal: {existing} and {new}')


class ModelBuiltIns(ModelElement):

    @property
    def version_id(self) -> VersionID:
        return 1

    @define_annual(years_key, annual_shape=[])
    @define_annual(scan_step_ii_key, annual_shape=[])
    def model_element_prepare(self, ws):
        ws.val.annual[years_key] = jnp.arange(
                ws.model.mv.year_0,
                ws.model.mv.year_0 + ws.n_years)
        ws.val.annual[scan_step_ii_key] = jnp.arange(0, ws.n_years)


ModelElementParamType = TypeVar("ModelElementParamType", bound=ModelElement)


class ScenarioModel:
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
    mcmc_group_by_elemid: dict[str, str|None]
    mv: ModelVariables

    # MCMC parameters, for how much sampling is recommended for this model
    # These should be associated with mcmc groups, not whole models
    num_warmup:int
    thinning:int
    num_samples:int
    n_divergences_acceptable:int = 0

    @property
    def first_year(self) -> int:
        return self.mv.year_0

    @property
    def n_prior_years(self) -> int:
        return self.mv.n_prior_years

    @property
    def n_posterior_years(self) -> int:
        return self.mv.n_posterior_years

    def __init__(self, first_year:int, n_prior_years:int, n_posterior_years:int,
                 num_warmup:int,
                 num_samples:int,
                 thinning:int=1,
                 ):
        self.model_elements = {}
        self.mcmc_group_by_elemid = {}
        self._element_order_cache = {}
        self._default_by_var_key:dict[VarKey, jnp.ndarray] = {}
        self.num_warmup = num_warmup
        self.thinning = thinning
        self.num_samples = num_samples
        self.mv = ModelVariables(
                year_0=first_year,
                n_prior_years=n_prior_years,
                n_posterior_years=n_posterior_years,
                )
        self.add_element(ModelBuiltIns())

    def _check_new_var_key(
            self,
            var_key:VarKey,
            elem_id:ElementID,
            subphase:Subphase,
            ) -> None:
        """Keys are single-assignment: only one element may define one.

        Raises DefinitionError if var_key is already defined, in any subphase.
        """
        for other in (self.mv.general_nd, self.mv.annual_nd, self.mv.carry_nd):
            if var_key in other:
                nvm = other[var_key]
                raise DefinitionError(
                        f'{var_key} is already defined by element'
                        f' {nvm.defining_element_id} in'
                        f' subphase {nvm.defining_subphase.value};'
                        f' element {elem_id} cannot also define it in'
                        f' subphase {subphase.value}')

    def _add_define_d(
            self,
            element:ModelElement,
            defining_subphase:Subphase,
            define_d):

        def add_var_key_ndm(var_key:VarKey, ndm):
            self._check_new_var_key(
                    var_key, element.identifier, defining_subphase)
            self.mv.general_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=_resolve_properties_ndarray_definition(
                        element, ndm),
                    defining_element_id=element.identifier,
                    defining_subphase=defining_subphase)

        for var_key, ndm in define_d.items():
            if isinstance(var_key, VarKey):
                add_var_key_ndm(var_key, ndm)
            elif isinstance(var_key, str):
                add_var_key_ndm(
                        ElementGeneralKey(
                            elem_id=element.identifier,
                            name=var_key),
                        ndm=ndm)
            elif isinstance(var_key, property):
                var_key_prop = var_key.__get__(element, type(element))
                if isinstance(var_key_prop, VarKey):
                    add_var_key_ndm(var_key_prop, ndm)
                elif isinstance(var_key_prop, dict):
                    for int_key in var_key_prop.values():
                        add_var_key_ndm(int_key, ndm)
                else:
                    raise NotImplementedError(var_key_prop)
            else:
                raise NotImplementedError(var_key)


    def _add_define_annual_d(self, element:ModelElement, defining_subphase, annual_d):
        element_id = element.identifier
        def add_var_key_ndm(var_key:ElementAnnualKey, ndm):
            self._check_new_var_key(
                    var_key, element_id, defining_subphase)
            resolved_ndm = _resolve_properties_ndarray_definition(element, ndm)
            assert resolved_ndm.value_type.shape[0] == years_dim
            # full timeseries metadata, with a leading years_dim
            self.mv.annual_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=resolved_ndm,
                    defining_element_id=element_id,
                    defining_subphase=defining_subphase)

            if defining_subphase in (Subphase.Prep, Subphase.Step):
                # per-scan-step metadata, with the years_dim dropped,
                # for this_annual access during a scan step
                step_ndm = NdarrayVariableMetadata(
                        definition_metadata=NdarrayDefinitionMetadata(
                            value_type=NdarrayType(
                                shape=resolved_ndm.value_type.shape[1:],
                                dtype=resolved_ndm.value_type.dtype),
                            declaration_subphase=(
                                resolved_ndm.declaration_subphase),
                            posterior_only=resolved_ndm.posterior_only,
                            ),
                        defining_element_id=element_id,
                        defining_subphase=defining_subphase)
                self.mv.this_annual_nd[var_key] = step_ndm

        for var_key, ndm in annual_d.items():
            if isinstance(var_key, ElementAnnualKey):
                add_var_key_ndm(var_key, ndm)
            elif isinstance(var_key, str):
                add_var_key_ndm(
                        ElementAnnualKey(
                            elem_id=element_id,
                            name=var_key),
                        ndm)
            elif isinstance(var_key, property):
                var_key_prop = var_key.__get__(element, type(element))
                if isinstance(var_key_prop, ElementAnnualKey):
                    add_var_key_ndm(var_key_prop, ndm)
                elif isinstance(var_key_prop, dict):
                    for int_key in var_key_prop.values():
                        if isinstance(int_key, ElementAnnualKey):
                            add_var_key_ndm(int_key, ndm)
                        else:
                            raise NotImplementedError(int_key)
                else:
                    raise NotImplementedError(var_key_prop)
            else:
                raise NotImplementedError(var_key)

    def _add_define_carry_d(self, element:ModelElement, carry_d):
        element_id = element.identifier
        def add_var_key_ndm(var_key:ElementCarryKey, initial_ndm, next_ndm):
            resolved_initial = _resolve_properties_ndarray_definition(
                    element, initial_ndm)
            resolved_next = _resolve_properties_ndarray_definition(
                    element, next_ndm)

            self._check_new_var_key(var_key, element_id, Subphase.Step)
            self.mv.carry_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=resolved_initial,
                    defining_element_id=element_id,
                    defining_subphase=Subphase.Step)

            initial_key = InitialCarry(carry_key=var_key)
            self._check_new_var_key(initial_key, element_id, Subphase.Prep)
            self.mv.general_nd[initial_key] = NdarrayVariableMetadata(
                    definition_metadata=resolved_initial,
                    defining_element_id=element_id,
                    defining_subphase=Subphase.Prep)

            final_key = FinalCarry(carry_key=var_key)
            self._check_new_var_key(final_key, element_id, Subphase.Proc)
            self.mv.general_nd[final_key] = NdarrayVariableMetadata(
                    definition_metadata=resolved_next,
                    defining_element_id=element_id,
                    defining_subphase=Subphase.Proc)

        for var_key, (initial_ndm, next_ndm) in carry_d.items():
            if isinstance(var_key, ElementCarryKey):
                add_var_key_ndm(var_key, initial_ndm, next_ndm)

            elif isinstance(var_key, str):
                add_var_key_ndm(
                        ElementCarryKey(
                            elem_id=element_id,
                            name=var_key),
                        initial_ndm, next_ndm)

            elif isinstance(var_key, property):
                var_key_prop = var_key.__get__(element, type(element))
                if isinstance(var_key_prop, ElementCarryKey):
                    add_var_key_ndm(var_key_prop, initial_ndm, next_ndm)
                elif isinstance(var_key_prop, dict):
                    for int_key in var_key_prop.values():
                        if isinstance(int_key, ElementCarryKey):
                            add_var_key_ndm(int_key, initial_ndm, next_ndm)
                        else:
                            raise NotImplementedError(int_key)
                else:
                    raise NotImplementedError(var_key_prop)
            else:
                raise NotImplementedError(var_key)

    def _process_access_d(
            self,
            element:ModelElement,
            avd,
            annual:bool,
            which_carry:Literal['this', 'next', 'neither']):
        for subphase, var_key_d in avd.items():
            for str_vk_or_prop, metadata in var_key_d.items():
                default = metadata['default']
                if isinstance(str_vk_or_prop, str):
                    if annual:
                        vk_or_prop = ElementAnnualKey(
                                elem_id=element.identifier, name=str_vk_or_prop)
                    elif which_carry != 'neither':
                        vk_or_prop = ElementCarryKey(
                                elem_id=element.identifier, name=str_vk_or_prop)
                    else:
                        vk_or_prop = ElementGeneralKey(
                                elem_id=element.identifier, name=str_vk_or_prop)
                else:
                    vk_or_prop = str_vk_or_prop

                self.mv.unresolved_accesses.append(
                        UnresolvedAccessMetadata(
                            elem_id=element.identifier,
                            subphase=subphase,
                            var_key_or_prop=vk_or_prop,
                            default=default,
                            annual=annual,
                            which_carry=which_carry))

    def add_element(
            self,
            element:ModelElementParamType,
            mcmc_group:str|None="default_mcmc_group") -> ModelElementParamType:
        assert element.identifier
        assert element.identifier not in self.model_elements, element.identifier
        self.model_elements[element.identifier] = element
        self.mcmc_group_by_elemid[element.identifier] = mcmc_group
        self._element_order_cache.clear()

        # general variables, not annual, not carry
        self._add_define_d(
                element,
                Subphase.Prep,
                getattr(element.model_element_prepare, _deco_attr_define, {}))
        assert not hasattr(element.model_element_annual_step, _deco_attr_define)
        self._add_define_d(
                element,
                Subphase.Proc,
                getattr(element.model_element_postprocess, _deco_attr_define, {}))

        # carry variables
        self._add_define_carry_d(
                element,
                getattr(element.model_element_prepare, _deco_attr_carry, {}))
        assert not hasattr(element.model_element_annual_step, _deco_attr_carry)
        assert not hasattr(element.model_element_postprocess, _deco_attr_carry)

        # scan-length annual timeseries
        self._add_define_annual_d(
                element,
                Subphase.Prep,
                getattr(element.model_element_prepare, _deco_attr_annual, {}))
        self._add_define_annual_d(
                element,
                Subphase.Step,
                getattr(element.model_element_annual_step, _deco_attr_annual, {}))
        self._add_define_annual_d(
                element,
                Subphase.Proc,
                getattr(element.model_element_postprocess, _deco_attr_annual, {}))

        # access_val and access_annual
        for subphase_method in (
                element.model_element_prepare,
                element.model_element_annual_step,
                element.model_element_postprocess):
            try:
                self._process_access_d(
                        element,
                        getattr(subphase_method, _deco_attr_access_val, {}),
                        annual=False,
                        which_carry='neither',
                        )
                self._process_access_d(
                        element,
                        getattr(subphase_method, _deco_attr_access_annual, {}),
                        annual=True,
                        which_carry='neither')
                self._process_access_d(
                        element,
                        getattr(subphase_method, _deco_attr_access_this_carry, {}),
                        annual=False,
                        which_carry='this')
                self._process_access_d(
                        element,
                        getattr(subphase_method, _deco_attr_access_next_carry, {}),
                        annual=False,
                        which_carry='next')
            except Exception as err:
                err.add_note(f'element={element}')
                raise

        return element

    def resolve_accesses(self) -> list[ResolvedAccessMetadata]:
        """Return a list of accesses that resolves the properties in mv.access_declarations.
        Also fill out default_by by side-effect.
        """
        rval:list[ResolvedAccessMetadata] = []

        def add_access(var_key_:VarKey):
            if isinstance(var_key_, ElementAnnualKey) and not uam.annual:
                raise TypeError(
                        f'access_val cannot read the annual key'
                        f' {var_key_}; use access_annual instead')
            if (not isinstance(var_key_, ElementAnnualKey) and uam.annual):
                raise TypeError(
                        f'access_annual requires an AnnualKey,'
                        f' got {var_key_}')
            if uam.annual:
                accessed_vk = var_key_
            elif uam.which_carry != 'neither':
                assert not uam.annual
                if not isinstance(var_key_, ElementCarryKey):
                    raise TypeError(
                            'access_carry requires a CarryKey,'
                            f' got {var_key_}')
                if uam.which_carry == 'this':
                    accessed_vk = ThisCarry(carry_key=var_key_)
                elif uam.which_carry == 'next':
                    accessed_vk = NextCarry(carry_key=var_key_)
                else:
                    raise NotImplementedError(uam.which_carry)
            else:
                accessed_vk = var_key_

            if uam.default is not None:
                existing = self._default_by_var_key.get(accessed_vk)
                if existing is None:
                    self._default_by_var_key[accessed_vk] = uam.default
                else:
                    _check_default_consistency(
                            accessed_vk, existing, uam.default)
            rval.append(
                    ResolvedAccessMetadata(
                        elem_id=uam.elem_id,
                        subphase=uam.subphase,
                        var_key=accessed_vk,
                        which_carry=uam.which_carry,
                        default=uam.default))

        for uam in self.mv.unresolved_accesses:
            if isinstance(uam.var_key_or_prop, VarKey):
                add_access(uam.var_key_or_prop)
            elif isinstance(uam.var_key_or_prop, property):
                prop = uam.var_key_or_prop
                element = self.model_elements[uam.elem_id]
                vk_or_dict = prop.__get__(element, type(element))
                if isinstance(vk_or_dict, VarKey):
                    add_access(vk_or_dict)
                elif isinstance(vk_or_dict, dict):
                    for vk in vk_or_dict.values():
                        add_access(vk)
                else:
                    raise NotImplementedError(vk_or_dict)
            else:
                raise NotImplementedError(uam.var_key_or_prop)
        return rval

    def _definitions_for_subphase(self, subphase:Subphase) -> dict[VarKey, ElementID]:
        """var_key -> defining element, for keys defined in this subphase."""
        rval:dict[VarKey, ElementID] = {}
        for metadata in (self.mv.general_nd, self.mv.annual_nd, self.mv.carry_nd):
            for var_key, nvm in metadata.items():
                if nvm.defining_subphase == subphase:
                    rval.setdefault(var_key, nvm.defining_element_id)
        return rval

    def _dependency_key(self, access:ResolvedAccessMetadata) -> VarKey|None:
        """The key whose definer must run before an access of var_key.

        Returns None when the access does not imply an ordering edge.
        """
        if isinstance(access.var_key, ElementCarryKey):
            if access.subphase == Subphase.Prep:
                return InitialCarry(carry_key=access.var_key)
            elif access.subphase == Subphase.Step:
                if access.which_carry == 'this':
                    return ThisCarry(carry_key=access.var_key)
                elif access.which_carry == 'next':
                    return NextCarry(carry_key=access.var_key)
                else:
                    raise NotImplementedError(access.which_carry)
            elif access.subphase == Subphase.Proc:
                return FinalCarry(carry_key=access.var_key)
            else:
                # during a scan step, this_carry is the previous step's value
                return None
        return access.var_key

    def element_order(self, subphase:Subphase) -> list[ElementID]:
        """Elements in an order where each definer precedes its readers.

        Only declared accesses (access_val / access_annual) create edges, and
        only within the same subphase. Ties keep the order elements were
        added to the model. Raises DefinitionError if the accesses cycle.
        """
        if subphase in self._element_order_cache:
            return list(self._element_order_cache[subphase])

        elem_ids = list(self.model_elements)
        elem_index = {elem_id: ii for ii, elem_id in enumerate(elem_ids)}
        definition_d = self._definitions_for_subphase(subphase)
        successors:dict[ElementID, set[ElementID]] = {
                elem_id: set() for elem_id in elem_ids}
        indegree = {elem_id: 0 for elem_id in elem_ids}

        for access in self.resolve_accesses():
            if access.subphase != subphase:
                continue
            dep_key = self._dependency_key(access)
            if dep_key is None:
                continue
            definer = definition_d.get(dep_key)
            # definer will be None to use access' default value
            if definer is None or definer == access.elem_id:
                continue
            if access.elem_id not in successors[definer]:
                successors[definer].add(access.elem_id)
                indegree[access.elem_id] += 1

        ready = [
                (elem_index[elem_id], elem_id)
                for elem_id in elem_ids if indegree[elem_id] == 0]
        heapq.heapify(ready)
        ordered:list[ElementID] = []
        while ready:
            _, elem_id = heapq.heappop(ready)
            ordered.append(elem_id)
            for successor in sorted(successors[elem_id], key=elem_index.__getitem__):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    heapq.heappush(ready, (elem_index[successor], successor))

        if len(ordered) != len(elem_ids):
            in_cycle = [elem_id for elem_id in elem_ids if elem_id not in ordered]
            raise DefinitionError(
                    f'access declarations for subphase {subphase.value}'
                    f' contain a dependency cycle among elements {in_cycle}')

        self._element_order_cache[subphase] = ordered
        return list(ordered)

    def mcmc_submodels(self) -> dict[str, ScenarioModel]:
        rval = {}
        for elem_id, elem in self.model_elements.items():
            if isinstance(elem, ModelBuiltIns):
                continue
            mcmc_group = self.mcmc_group_by_elemid[elem_id]
            if mcmc_group is None:
                continue
            if mcmc_group not in rval:
                rval[mcmc_group] = ScenarioModel(
                        first_year=self.first_year,
                        n_prior_years=self.n_prior_years,
                        n_posterior_years=self.n_posterior_years,
                        num_warmup=self.num_warmup,
                        num_samples=self.num_samples)
            rval[mcmc_group].add_element(elem)
        return rval

    def version_hash(self) -> str:
        # TODO: next time it's convenient,
        # add (self.first_year, self.n_prior_years, self.num_warmup)
        this_subsystem_version = 1.2
        version_id = (
                this_subsystem_version,
                self.first_year,
                self.n_prior_years,
                self.num_warmup,
                self.thinning,
                tuple(
                    [(elem_id, elem.version_id)
                     for elem_id, elem in sorted(self.model_elements.items())]),
                    )
        rval = hash_version_id(version_id)
        return rval
