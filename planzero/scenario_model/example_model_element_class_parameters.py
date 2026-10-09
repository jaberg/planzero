"""
It is good practice for a ModelElement to be configured
by class parameters, and then consequently define VarKeys unique
to itself, which are parameter-dependent keys.

It is also natural for a model element to define variables
corresponding to known cases, elements, e.g. in an enum.

This example file illustrates both cases, and the corresponding
test file verifies that they work as expected.
"""

import enum

import numpyro.distributions as dist

from .computation import (
    ElementGeneralKey,
    Model,
    ModelElement,
    Workspace,
    define,
)


class Case(str, enum.Enum):
    A = 'A'
    B = 'B'


class PropertyModelElement(ModelElement):
    _prop_key: ElementGeneralKey|None = None
    _dict_keys: dict[Case, ElementGeneralKey]|None = None
    _n_samples: int

    @property
    def identifier(self) -> str:
        return f'PropertyModelElement(n_samples={self.n_samples})'

    def __init__(self, n_samples):
        self._n_samples = n_samples

    @property
    def prop_key(self) -> ElementGeneralKey:
        if self._prop_key is None:
            self._prop_key = self.general_key(f'prop_key_{self.n_samples}')
        return self._prop_key

    @property
    def dict_keys(self) -> dict[Case, ElementGeneralKey]:
        if self._dict_keys is None:
            self._dict_keys = {
                    case: self.general_key(
                        f'dict_key_{self.n_samples}_{case}')
                    for case in Case}
        return self._dict_keys

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @define(prop_key, prior_shape=[n_samples])
    @define(dict_keys, prior_shape=[n_samples])
    def model_element_prepare(self, ws:Workspace):
        ws.dist.general[self.prop_key] = dist.Normal()
        ws.val.general[self.prop_key] # draw sample
        for external_key, var_key in self.dict_keys.items():
            ws.dist.general[var_key] = dist.Normal(external_key)
            # ws.val.general[var_key] # don't draw sample


def property_model() -> Model:
    model = Model(
            first_year=0,
            n_prior_years=1,
            n_posterior_years=1,
            num_warmup=7,
            num_samples=9,
            )
    model.add_element(PropertyModelElement(n_samples=1))
    model.add_element(PropertyModelElement(n_samples=2))
    return model
