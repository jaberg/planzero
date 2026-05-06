from .base import Scenario, u
from .sim import simulation_result, site_simulations

def test_empty_sim():
    scenario = Scenario(name='foo', t_start=2000 * u.years)
    scenario.run_until(2010 * u.years)
    scenario.compute_annual_emissions()
    scenario.compute_annual_subsidies()
    # no crash, yay

def test_extrapolation_by_ipcc_sector():
    sim_result = simulation_result('NIR2025').by_ipcc_sector

def test_planet_model():
    site_sim = site_simulations['Planet_Model']
    for d in site_sim.dynamic_elements():
        if 'strategy' in d.tags:
            print(d)
        else:
            print('no strat', d)
    sim = simulation_result('Planet_Model')
    baseline_state = sim.state
    ablated_state = sim.ablations.get('EmissionsImpuseResponse_CO2')
    if not ablated_state:
        assert 0
