import functools

from . import enums
from .base import (
    Project,
    ProjectEvaluation,
    AtmosphericChemistry, # TODO: move to this file
    SparseTimeSeries,
    ureg as u,
    )
from .ghgvalues import GWP_100


class EmissionsImpulseResponse(Project):
    ghg:object
    impulse_co2e:object
    catpath:str

    def on_add_project(self, state):
        with state.defining(self) as ctx:
            ctx.impulse_response = SparseTimeSeries(
                times=[2000 * u.years, 2001 * u.years],
                values=[self.impulse_co2e / GWP_100[self.ghg], 0.0 * u.kg_CO2e / GWP_100[self.ghg]],
                default_value=0.0 * u.kg_CO2e / GWP_100[self.ghg])

        # any catpath will do
        state.register_emission(self.catpath, self.ghg, 'impulse_response')


@functools.cache
def emissions_impulse_response_project_evaluation(impulse_co2e, years,
                                                  catpath='Forest_Land'):
    peval = ProjectEvaluation(
        projects={ghg: EmissionsImpulseResponse(impulse_co2e=impulse_co2e,
                                                ghg=ghg,
                                                catpath=catpath)
                  for ghg in enums.GHG},
        common_projects=[AtmosphericChemistry()],
        present=2000 * u.years,
    )
    t_end = (2000 + years) * u.years
    peval.run_until(t_end)
    return peval
