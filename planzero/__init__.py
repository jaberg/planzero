from .base import Project
from .base import BaseScenarioProject
from .base import ProjectEvaluation
from .base import SparseTimeSeries
from .base import ureg
from .base import AtmosphericChemistry

from . import base
from . import battery_tech

from .ipcc__transport__marine__domestic_navigation import (
    IPCC_Transport_Marine_DomesticNavigation_Model,
)

from .ipcc_transport_road_heavydutydiesel import (
    IPCC_Transport_RoadTransportation_HeavyDutyDieselVehicles,
)

from . import barriers
from . import strategies
from . import scenarios
from . import sim
from .my_functools import cache as _cache


@_cache
def get_peval():
    peval = base.ProjectEvaluation(
        projects={strat.identifier: strat
                  for strat in strategies.standard_strategies()},
        common_projects=BaseScenarioProject.base_scenario_projects(),
    )
    peval.run_until(2125 * ureg.years)
    return peval
