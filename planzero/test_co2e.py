import numpy as np

from .ureg import u
from .enums import GHG
from .ghgvalues import GWP_100
from .sim import simulation_result


def test_co2e(assert_value=0, years=100):
    sim_result = simulation_result('Planet_Model')
    impulse_mass = 1_000_000 * u.kg_CO2e
    forcings = []
    for ghg in GHG:
        state_A = sim_result.state
        state_B = sim_result.ablations[f'EmissionsImpulseResponse_{ghg.value}']
        #assert state_A.sts[f'impulse_{ghg.value}'].max() == impulse_mass / GWP_100[ghg] / u.year
        #print(state_A.sts[f'impulse_{ghg.value}'])
        co2e_key = f'Predicted_Annual_Emitted_CO2e_mass'
        #print(ghg, state_A.sts[co2e_key].values)
        assert np.allclose(
            state_A.sts[co2e_key].max(_i_start=1),
            7 * impulse_mass)
        t_end = state_A.t_now

        energy_A = state_A.sts['Cumulative_Heat_Energy'].query(t_end)
        energy_B = state_B.sts['Cumulative_Heat_Energy'].query(t_end)

        forcing_energy_A = state_A.sts['Cumulative_Heat_Energy_forcing'].query(t_end)
        forcing_energy_B = state_B.sts['Cumulative_Heat_Energy_forcing'].query(t_end)

        forcing_delta = (forcing_energy_A - forcing_energy_B).to(u.terajoule)

        print(
            ghg,
            state_A.sts[co2e_key].max(_i_start=1).to(u.kilotonne_CO2e),
            #comp.state_B.sts[co2e_key].max(_i_start=1).to(u.kilotonne_CO2e),
            'remaining', (energy_A - energy_B).to('terajoule'),
            'forcing', forcing_delta.to('terajoule'),
            )
        #assert 900 * u.terajoule < forcing_delta < 2200 * u.terajoule
        # see blog post on unfccc / greenhouse gases for discussion of the remaining discrepancy
        # * model does not account for overlap in absorption by N2O and CH4, whereas GWP does.
        # * model is using start-with-a-guess-y initial atmospheric concentrations for all gases, which will lead to
        #   discrepancies here, especially in the case of N2O
        forcings.append(forcing_delta)
    min_forcing = min(forcings)
    max_forcing = max(forcings)
    ratio = (max_forcing / min_forcing).to('dimensionless').magnitude
    assert 2.05 <= ratio <= 2.25
