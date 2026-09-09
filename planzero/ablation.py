"""
BaseModels for Ablation Studies
"""

from typing import ClassVar

from pydantic import BaseModel, computed_field

from .singleton_registry import SingletonRegistry

registry = SingletonRegistry()


class AblationStudy(BaseModel):
    """A set of SiteInference objects
    that correspond to the ablation of each one of a set of Strategies,
    as well as a reference SiteInference object corresponding to no ablation
    of any of the strategies.
    """

    # True indicates that the study is to be included on the site
    include_in_registry: ClassVar[bool] = False

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        if getattr(cls, 'include_in_registry', False):
            registry.add_class(cls)

    @property
    def dynamic_elements(self) -> dict:
        barriers = self.barriers
        strategies = self.strategies
        rval = dict(barriers, **strategies)
        assert len(rval) == len(barriers) + len(strategies)
        return rval

    @property
    def strategies(self) -> dict:
        return {}

    @property
    def barriers(self) -> dict:
        return {}

    @property
    def ablation(self) -> dict[str|None, str]:
        # [None] looks up the reference site_inference id
        # [strategy_id] looks up the site_inference id for the ablation of that strategy
        raise NotImplementedError()
