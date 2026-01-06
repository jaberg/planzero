from .base import Project
from .base import ProjectEvaluation
from .base import SparseTimeSeries
from .base import ureg

from . import base
def _common_projects():
    return [
        base.GeometricBovinePopulationForecast(),
        base.GeometricHumanPopulationForecast(),
        base.IPCC_Forest_Land_Model(),
        base.IPCC_Transport_Marine_DomesticNavigation_Model(),
        base.IPCC_Transport_RoadTransportation_LightDutyGasolineTrucks(),
        base.PacificLogBargeForecast(),
        base.AtmosphericChemistry(),
    ]

from . import strategies

def standard_project_evaluation():
    rval = base.ProjectEvaluation(
        projects={prj.idea.name: prj for prj in strategies.standard_strategies()},
        common_projects=_common_projects(),
    )
    return rval
