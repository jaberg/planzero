from pydantic import BaseModel, computed_field

from .ureg import u
from .enums import StandardScenarios
from .base import DynamicElement

from . import csfs
from . import barriers
from . import strategies


def collect_dynelems(scenario):
    rval = []
    rval.extend([val for val in barriers.barriers.values()])
    rval.extend([val for val in csfs.csfs.values()])
    rval.extend([val for val in strategies.strategies.values()
                if scenario in val.scenarios])
    return rval


scenarios = {} # classname -> Singleton instance


class Scenario(BaseModel):

    t_start_year: int
    short_descr: str
    research: dict[str, str]
    dynelems: list[DynamicElement]

    def __init__(self, **kwargs):
        if 'short_descr' not in kwargs:
            kwargs = dict(kwargs, short_descr=self.__class__.__name__)
        super().__init__(**kwargs)

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        scenarios[cls.__name__.lower()] = cls() # lower() to make url-compatible

    @computed_field
    def name(self) -> str:
        return self.__class__.__name__


class Scaling(Scenario):

    def __init__(self):
        super().__init__(
            t_start_year=1990,
            short_descr=f"Assume only the deployment of existing products",
            research={},
            state=None,
            dynelems=collect_dynelems(StandardScenarios.Scaling))


class Extrapolating(Scenario):

    def __init__(self):
        super().__init__(
            t_start_year=1990,
            short_descr=f"Assume the continuation of statistical trends",
            research={},
            state=None,
            dynelems=collect_dynelems(StandardScenarios.Extrapolating))
