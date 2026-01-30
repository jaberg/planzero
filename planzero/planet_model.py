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
    CO2=1.0 * u.kg_CO2e / u.kg_CO2,
    CH4=28.0 * u.kg_CO2e / u.kg_CH4,
    N2O=265.0 * u.kg_CO2e / u.kg_N2O,
    HFC=1_430 * u.kg_CO2e / u.kg_HFC,
    PFC=6_630 * u.kg_CO2e / u.kg_PFC,
    SF6=23_500 * u.kg_CO2e / u.kg_SF6,
    NF3=17_200 * u.kg_CO2e / u.kg_NF3,
    )


def CO2e_from_emissions(co2, ch4, n2o, hfc=None, pfc=None, sf6=None, nf3=None, unit=u.kt_CO2e):
    if hfc or pfc or sf6 or nf3:
        raise NotImplementedError()
    from_co2 = (co2 * GWP_100['CO2']).to(unit)
    from_ch4 = (ch4 * GWP_100['CH4']).to(unit)
    from_n2o = (n2o * GWP_100['N2O']).to(unit)
    return from_co2 + from_ch4 + from_n2o


class EmissionsImpulseResponse(Project):
    ghg:str
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
                  for ghg in GHGs},
        common_projects=[AtmosphericChemistry()],
        present=2000 * u.years,
    )
    t_end = (2000 + years) * u.years
    peval.run_until(t_end)
    return peval
