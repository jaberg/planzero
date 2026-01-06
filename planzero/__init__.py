from .base import Project
from .base import BaseScenarioProject
from .base import ProjectEvaluation
from .base import SparseTimeSeries
from .base import ureg

from . import base

from .ipcc__transport__marine__domestic_navigation import IPCC_Transport_Marine_DomesticNavigation_Model

from . import strategies

def standard_project_evaluation():
    rval = base.ProjectEvaluation(
        projects={prj.idea.name: prj for prj in strategies.standard_strategies()},
        common_projects=BaseScenarioProject.base_scenario_projects(),
    )
    return rval
