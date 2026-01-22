import time

import numpy as np

from . import (
    Project,
    ProjectEvaluation,
    AtmosphericChemistry,
    SparseTimeSeries,
    ureg as u,
    )

dummy_catpath = 'Enteric_Fermentation'
impulse_mass = 1000 * u.kg

class EmissionsImpulseResponse(Project):
    ghg: str

    def on_add_project(self, state):

        with state.defining(self) as ctx:
            ctx.impulse_response = SparseTimeSeries(
                times=[state.t_now, state.t_now + 1 * u.years],
                values=[impulse_mass, 0.0 * u.kg],
                default_value=0.0 * u.kg)

        # any catpath will do
        state.register_emission(dummy_catpath, self.ghg, 'impulse_response')


GWP_100 = dict(
    CO2=1.0,
    CH4=28.0,
    NO2=265.0,
    HFC=1_430,
    PFC=6_630,
    SF6=23_500,
    NF3=17_200,
    )


def test_co2e(assert_value=0, years=100):
    t0 = time.time()
    GHGs = ['CO2', 'CH4', 'NO2', 'HFC', 'PFC', 'SF6', 'NF3']
    peval = ProjectEvaluation(
        projects={ghg: EmissionsImpulseResponse(ghg=ghg) for ghg in GHGs},
        common_projects=[AtmosphericChemistry()],
        present=2000 * u.years,
    )
    t1 = time.time()
    print(t1 - t0, 'peval construction')
    peval.run_until((2000 + years) * u.years)
    t2 = time.time()
    print(t2 - t1, 'run_until')
    return
    for ghg in GHGs:
        comp = peval.comparisons[ghg]
        assert comp.state_A.sts['impulse_response'].max() == impulse_mass
        co2e_key = f'Predicted_Annual_Emitted_CO2e_mass_{dummy_catpath}'
        assert np.allclose(
            comp.state_A.sts[co2e_key].max(_i_start=1),
            GWP_100[ghg] * impulse_mass)
        assert comp.state_B.sts[co2e_key].max(_i_start=1) == 0 * u.kg
        
        print(ghg,
            comp.state_A.sts[co2e_key].max(_i_start=1).to(u.kilotonne),
            comp.state_B.sts[co2e_key].max(_i_start=1).to(u.kilotonne),
             )
    print(t1 - t0)
    assert assert_value
