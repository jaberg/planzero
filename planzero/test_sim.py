from .base import Scenario, u
from .sim import simulation_result

def test_empty_sim():
    scenario = Scenario(name='foo', t_start=2000 * u.years)
    scenario.run_until(2010 * u.years)
    scenario.compute_annual_emissions()
    scenario.compute_annual_subsidies()
    # no crash, yay

def test_extrapolation_by_ipcc_sector():
    sim_result = simulation_result('Extrapolation').by_ipcc_sector
