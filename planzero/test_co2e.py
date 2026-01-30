import numpy as np

from .planet_model import (
    emissions_impulse_response_project_evaluation,
    GWP_100,
    GHGs,
    u,
    )


def test_co2e(assert_value=0, years=100):
    impulse_mass = 1_000_000 * u.kg
    peval = emissions_impulse_response_project_evaluation(impulse_co2e=impulse_mass, years=years)
    catpath = peval.projects['CO2'].catpath
    forcings = []
    for ghg in GHGs:
        comp = peval.comparisons[ghg]
        assert comp.state_A.sts['impulse_response'].max() == impulse_mass / GWP_100[ghg]
        co2e_key = f'Predicted_Annual_Emitted_CO2e_mass_{catpath}'
        assert np.allclose(
            comp.state_A.sts[co2e_key].max(_i_start=1),
            impulse_mass)
        assert comp.state_B.sts[co2e_key].max(_i_start=1) == 0 * u.kg
        t_end = comp.state_A.t_now

        energy_A = comp.state_A.sts['Cumulative_Heat_Energy'].query(t_end)
        energy_B = comp.state_B.sts['Cumulative_Heat_Energy'].query(t_end)

        forcing_energy_A = comp.state_A.sts['Cumulative_Heat_Energy_forcing'].query(t_end)
        forcing_energy_B = comp.state_B.sts['Cumulative_Heat_Energy_forcing'].query(t_end)

        forcing_delta = forcing_energy_A - forcing_energy_B

        print(
            ghg,
            comp.state_A.sts[co2e_key].max(_i_start=1).to(u.kilotonne),
            comp.state_B.sts[co2e_key].max(_i_start=1).to(u.kilotonne),
            'remaining', (energy_A - energy_B).to('terajoule'),
            'forcing', forcing_delta.to('terajoule'),
            )
        assert 900 * u.terajoule < forcing_delta < 2200 * u.terajoule
        # see blog post on unfccc / greenhouse gases for discussion of the remaining discrepancy
        # * model does not account for overlap in absorption by N2O and CH4, whereas GWP does.
        # * model is using start-with-a-guess-y initial atmospheric concentrations for all gases, which will lead to
        #   discrepancies here, especially in the case of N2O
        forcings.append(forcing_delta)
    min_forcing = min(forcings)
    max_forcing = max(forcings)
    ratio = (max_forcing / min_forcing).to('dimensionless').magnitude
    assert 2.15 <= ratio <= 2.25
