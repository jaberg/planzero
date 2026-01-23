import functools

from .base import (
    Project,
    ProjectEvaluation,
    AtmosphericChemistry, # TODO: move to this file
    SparseTimeSeries,
    ureg as u,
    )

GHGs = ['CO2', 'CH4', 'N2O', 'HFC', 'PFC', 'SF6', 'NF3']
GWP_100 = dict(
    CO2=1.0,
    CH4=28.0,
    N2O=265.0,
    HFC=1_430,
    PFC=6_630,
    SF6=23_500,
    NF3=17_200,
    )

class EmissionsImpulseResponse(Project):
    ghg:str
    impulse_co2e:object
    catpath:str

    def on_add_project(self, state):

        with state.defining(self) as ctx:
            ctx.impulse_response = SparseTimeSeries(
                times=[state.t_now, state.t_now + 1 * u.years],
                values=[self.impulse_co2e / GWP_100[self.ghg], 0.0 * u.kg],
                default_value=0.0 * u.kg)

        # any catpath will do
        state.register_emission(self.catpath, self.ghg, 'impulse_response')


@functools.cache
def emissions_impulse_response_project_evaluation(impulse_co2e, years,
                                                  catpath='Forest_Land'):
    peval = ProjectEvaluation(
        projects={ghg: EmissionsImpulseResponse(impulse_co2e=impulse_co2e,
                                                ghg=ghg,
                                                catpath=catpath)
                  for ghg in GHGs},
        common_projects=[AtmosphericChemistry()],
        present=2000 * u.years,
    )
    t_end = (2000 + years) * u.years
    peval.run_until(t_end)
    return peval
