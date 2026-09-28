# this resolves circular type references
from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, computed_field


class ModelElement(BaseModel):

    _identifier: str|None = None

    @computed_field
    def identifier(self) -> str:
        # must be unique within a model
        return self._identifier or self.__class__.__name__

    def variables(self, vi: VariablesInterface):
        raise NotImplementedError()


class WorkSpaces(str, enum.Enum):
    General = 'General'
    InitialCarry = 'InitialCarry'
    ThisCarry = 'ThisCarry'
    NextCarry = 'NextCarry'
    FinalCarry = 'FinalCarry'
    ThisY = 'ThisY'
    ThisX = 'ThisX'
    Xs = 'Xs' # subsets of general
    Ys = 'Ys' # subsets of general
    PosteriorCache = 'PosteriorCache'


class VarKeyBase(BaseModel, frozen=True):

    # let AnnualEmissionRate VarKey declare
    # incompatibility with everything but General
    # ... what about Ys though?
    # TODO: Scoping and Storage -- can Xs and Ys be part of general?
    # Maybe the answer is a simple "yes", Xs and Ys can be accessed
    # via ws.general, and are restricted subsets of ws.general
    disallowed_workspaces: list[WorkSpaces] = []


VarKey = (str | VarKeyBase)


class Posterior(VarKeyBase, frozen=True):
    prior_varkey: VarKey


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


class Ndarray(ValType):
    shape: list[int|NdarrayDim] = []
    dtype: str = 'float64'


class Variable(BaseModel):
    value_type: ValType
    observation: VarKey|None = None

    # unknown variables
    # that are also not observed
    # are, by default, sampled and saved by MCMC samplers
    known:bool = True

    # I'm not sure if this is required, but
    # if we need a dependency graph, at least one of these is probably required
    definition_phase:str = 'unknown'
    requires: list[VarKey] = []


def UnknownVariable(**kwargs):
    return Variable(known=False, **kwargs)


class VarAction(BaseModel):
    pass


class VarActionRead(VarAction):
    action_type: Literal['READ']


class VarActionWrite(VarAction):
    action_type: Literal['WRITE']


VarActionUnion = (
        VarActionRead
        | VarActionWrite)


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

    elements: dict[str, ModelElement] = {}

    variables: dict[VarKey, VarAction] = {}

    def add_element(self, element):
        self.elements[element.identifier] = element

    def build_variables(self):
        for elem_id, element in self.elements.items():
            try:
                element.variables(VariablesInterface(self, elem_id))
            except Exception as err:
                err.add_note(f'elem_id={elem_id}')
                raise


class VariablesInterface:

    model: Model
    element_id: str

    def __init__(self, model, element_id):
        self.model = model
        self.element_id = element_id

    @property
    def inference_general(self) -> dict[VarKey, VarMeta]:
        raise NotImplementedError()

    @property
    def inference_scan_carry(self) -> dict[VarKey, VarMeta]:
        raise NotImplementedError()

    @property
    def inference_scan_x(self) -> dict[VarKey, VarMeta]:
        raise NotImplementedError()

    @property
    def inference_scan_y(self) -> dict[VarKey, VarMeta]:
        raise NotImplementedError()

    @property
    def analysis_general(self) -> dict[VarKey, VarMeta]:
        raise NotImplementedError()


class InferenceComputation:

    model: Model

    def __init__(self, model:Model):
        self.model = model


class AnalysisComputation:

    model: Model

    def __init__(self, model:Model):
        self.model = model
