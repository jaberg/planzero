# this resolves circular type references
from __future__ import annotations

import enum
from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
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
    prior_var_key: VarKey


def posterior(var_key:VarKey) -> Posterior:
    return Posterior(prior_var_key=var_key)


class InitialCarry(VarKeyBase, frozen=True):
    carry_key: VarKey


def initial_carry(var_key: VarKey) -> InitialCarry:
    return InitialCarry(carry_key=var_key)


class FinalCarry(VarKeyBase, frozen=True):
    carry_key: VarKey


def final_carry(var_key: VarKey) -> FinalCarry:
    return FinalCarry(carry_key=var_key)


class ValType(BaseModel):
    pass


class NdarrayDim(BaseModel):
    name: str
    unique_id: str


ndarray_dim_counter = 0
def ndarray_dimension(name):
    global ndarray_dim_counter
    ndarray_dim_counter += 1
    return NdarrayDim(
            name=name,
            unique_id=f'dim_{ndarray_dim_counter}')


class NdarrayType(ValType):
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
        annual_scan_dim:NdarrayDim,
        ) -> NdarrayDefinitionMetadata:
    if isinstance(dm.value_type, NdarrayType):
        valtype = dm.value_type
        if (len(valtype.shape) and valtype.shape[0] == annual_scan_dim):
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



class WorkSpaceInterface:

    element_id:str

    def __init__(self, *, element_id):
        self.element_id = element_id


class WorkSpace:
    reads: set[VarKey]


class NdarrayAccessor_RW:

    workspace:WorkSpace

    def __getitem__(self, item) -> jnp.ndarray:
        if item in self.workspace.reads:
            return self._storage[item]
        else:
            raise KeyError(item)



class WorkSpace_AnnualScanPrep(WorkSpace):

    @property
    def general_nd(self) -> NdarrayAccessor_RW:
        raise NotImplementedError()

    @property
    def Xs(self) -> dict[VarKey, object]:
        raise NotImplementedError()

    @property
    def initial_carry_nd(self) -> NdarrayAccessor_RW:
        raise NotImplementedError()


class WorkSpace_AnnualScanPrep_Inference(WorkSpace_AnnualScanPrep):
    pass


class WorkSpace_AnnualScanPrep_Analysis(WorkSpace_AnnualScanPrep):

    pass


class WorkSpace_AnnualScanStep(WorkSpace):

    @property
    def general(self) -> NdarrayAccessor_RW:
        return NdarrayAccessor_RW(self)

    @property
    def this_carry(self) -> dict[VarKey, object]:
        raise NotImplementedError()

    @property
    def next_carry(self) -> dict[VarKey, object]:
        # read/write
        raise NotImplementedError()

    @property
    def this_X(self) -> dict[VarKey, object]:
        # read only
        raise NotImplementedError()

    @property
    def Xs(self) -> dict[VarKey, object]:
        raise NotImplementedError()


    @property
    def this_Y(self) -> dict[VarKey, object]:
        # read/write
        raise NotImplementedError()


class WorkSpace_AnnualScanStep_Inference(WorkSpace_AnnualScanStep):
    pass


class WorkSpace_AnnualScanStep_Analysis(WorkSpace_AnnualScanStep):
    pass


class WorkSpace_AnnualScanPost(WorkSpace):

    @property
    def general(self) -> dict[VarKey, object]:
        raise NotImplementedError()

    @property
    def initial_carry(self) -> dict[VarKey, object]:
        raise NotImplementedError()

    @property
    def final_carry(self) -> dict[VarKey, object]:
        raise NotImplementedError()

    @property
    def Xs(self) -> dict[VarKey, object]:
        raise NotImplementedError()

    @property
    def Ys(self) -> dict[VarKey, object]:
        raise NotImplementedError()


class WorkSpace_AnnualScanPost_Inference(WorkSpace_AnnualScanPost):
    pass


class WorkSpace_AnnualScanPost_Analysis(WorkSpace_AnnualScanPost):
    pass


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
    annual_scan_dim: NdarrayDim
    mcmc_dim: NdarrayDim
    mv: ModelVariables

    def __init__(
            self,
            element_id: str,
            defining_phase: ModellingPhase,
            annual_scan_dim: NdarrayDim,
            mcmc_dim: NdarrayDim,
            mv: ModelVariables,
            ):
        self.element_id = element_id
        self.defining_phase = defining_phase
        self.annual_scan_dim = annual_scan_dim
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

    def annual_X(self, var_key, **kwargs):
        ndm = self._ndm_from_kwargs(**kwargs)

        self.mv.Xs_nd[var_key] = NdarrayVariableMetadata(
                definition_metadata=ndm,
                defining_element_id=self.element_id,
                defining_phase=ModellingPhase.Inference,
                defining_subphase=Subphase.Prep)
        try:
            self.mv.this_X_nd[var_key] = NdarrayVariableMetadata(
                    definition_metadata=_definition_metadata_timeslice(
                        ndm,
                        annual_scan_dim=self.annual_scan_dim),
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
                        annual_scan_dim=self.annual_scan_dim),
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
              initial_sample:bool,
              next_sample:bool,
              **kwargs):

        initial_ndm = self._ndm_from_kwargs(sample=initial_sample, **kwargs)
        this_ndm = self._ndm_from_kwargs(sample=False, **kwargs)
        next_ndm = self._ndm_from_kwargs(sample=next_sample, **kwargs)
        final_ndm = self._ndm_from_kwargs(sample=next_sample, **kwargs)

        self.mv.initial_carry_nd[var_key] = NdarrayVariableMetadata(
                definition_metadata=initial_ndm,
                defining_element_id=self.element_id,
                defining_phase=self.defining_phase,
                defining_subphase=Subphase.Prep)

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

        self.mv.final_carry_nd[var_key] = NdarrayVariableMetadata(
                definition_metadata=final_ndm,
                defining_element_id='__internal__',
                defining_phase=self.defining_phase,
                defining_subphase=Subphase.Internal_Between_Step_and_Post)


class InferenceElement(ModelElementPhase):

    annual_scan_prep_fn:Callable[[WorkSpace_AnnualScanPrep_Inference], None] = _workspace_no_op
    annual_scan_step_fn:Callable[[WorkSpace_AnnualScanStep_Inference], None] = _workspace_no_op
    annual_scan_post_fn:Callable[[WorkSpace_AnnualScanPost_Inference], None] = _workspace_no_op

    def __init__(self, **kwargs):
        super().__init__(defining_phase=ModellingPhase.Inference, **kwargs)

    def annual_scan_prep(self, *, reads:list[VarKeyRole]|None=None):
        reads = reads or []
        def decorator(fn: Callable[[WorkSpace_AnnualScanPrep_Inference], None]):
            #self.reads_by_subphase[Subphase.Prep] = reads
            self.annual_scan_prep_fn = fn
            return fn
        return decorator

    def annual_scan_step(self, *, reads:list[VarKeyRole]|None=None):
        reads = reads or []
        def decorator(fn: Callable[[WorkSpace_AnnualScanStep_Inference], None]):
            #self.reads_by_subphase[Subphase.Step] = reads
            self.annual_scan_step_fn = fn
            return fn
        return decorator

    def annual_scan_post(self, *, reads:list[VarKeyRole]|None=None):
        reads = reads or []
        def decorator(fn: Callable[[WorkSpace_AnnualScanPost_Inference], None]):
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

    def analysis(self, annual_scan_dim:NdarrayDim) -> ModelElementAnalysis:
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

    elements: dict[str, ModelElement] = {}

    annual_scan_dim: NdarrayDim
    mcmc_dim: NdarrayDim
    mv: ModelVariables

    def add_element(self, element):
        self.elements[element.identifier] = element
        inference_element = InferenceElement(
                element_id=element.identifier,
                annual_scan_dim=self.annual_scan_dim,
                mcmc_dim=self.mcmc_dim,
                mv=self.mv,
                )
        element.inference(inference_element)
        inference_element._add_posterior_variables()


def AnnualScanModel():
    return Model(
            annual_scan_dim=ndarray_dimension('annual_scan_dim'),
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


class InferenceComputation:

    model: Model

    storage_nd: dict[VarKey, jnp.ndarray]
    storage_obj: dict[VarKey, object]

    def __init__(self, model:Model):
        self.model = model

        self.storage_nd = {}
        self.storage_obj = {}

    def run(self):
        raise NotImplementedError()



class AnalysisComputation:

    model: Model

    def __init__(self, model:Model):
        self.model = model
