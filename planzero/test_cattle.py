from pprint import pprint
from .cattle import *
from .base import Scenario
from .strategies.strategy2 import Scale_Bovaer


def test_cattle_CH4_emissions_sum():
    scenario = Scenario(name='foo', t_start=2000 * u.years)
    scenario.add_dynamic_elements([
        Cattle_Population_AR(),
        Cattle_Enteric_Emission_Rates_NIR2025_Bovaer(),
        Bovaer_Adoption_Limit(),
        Scale_Bovaer()
    ])
    scenario.run_until(2040 * u.years)
    emissions = scenario.compute_annual_emissions()
    n_provinces_and_territories_reporting = 10
    assert len(emissions.by_sector_ghg_pt_driver) == (
        n_provinces_and_territories_reporting * len(Livestock_nonsums))
    #print(emissions.by_sector_ghg_pt_driver.keys())
    val = emissions.sum().magnitude
    assert 1129142 < val < 1140150


def test_cattle_CO2_emissions_sum():
    scenario = Scenario(name='foo', t_start=2000 * u.years)
    scenario.add_dynamic_elements([
        Cattle_Population_AR(),
        Bovaer_Production_Emission_Factors(),
        Bovaer_Adoption_Limit(),
        Scale_Bovaer()
    ])
    scenario.run_until(2040 * u.years)
    emissions = scenario.compute_annual_emissions()
    n_provinces_and_territories_reporting = 10
    assert len(emissions.by_sector_ghg_pt_driver) == (
        n_provinces_and_territories_reporting * len(Livestock_nonsums))
    val = emissions.sum().magnitude
    assert 1500 < val < 1600


def test_cattle_emissions_sum():
    scenario = Scenario(name='foo', t_start=2000 * u.years)
    scenario.add_dynamic_elements([
        Cattle_Population_AR(),
        Bovaer_Production_Emission_Factors(),
        Cattle_Enteric_Emission_Rates_NIR2025_Bovaer(),
        Bovaer_Adoption_Limit(),
        Scale_Bovaer()
    ])
    scenario.run_until(2040 * u.years)
    emissions = scenario.compute_annual_emissions()
    n_provinces_and_territories_reporting = 10
    assert len(emissions.by_sector_ghg_pt_driver) == 2 * (
        n_provinces_and_territories_reporting * len(Livestock_nonsums))
    val = emissions.sum().magnitude
    assert 1130290 < val < 1143300.


def test_cattle_bavaer_subsidy_sum():
    elements = [
        Cattle_Population_AR(),
        Bovaer_Adoption_Limit(),
        Bovaer_Production_Emission_Factors(),
        Cattle_Enteric_Emission_Rates_NIR2025_Bovaer(),
        Bovaer_Purchase_Cost(),
        Bovaer_Farm_Subsidy(),
        Bovaer_Monitoring(),
    ]
    base_scenario = Scenario(name='foo', t_start=2000 * u.years)
    base_scenario.add_dynamic_elements(elements)
    base_scenario.run_until(2040 * u.years)
    emissions = base_scenario.compute_annual_emissions()
    subsidies = base_scenario.compute_annual_subsidies()

    with_strat = Scenario(name='foo', t_start=2000 * u.years)
    with_strat.add_dynamic_elements(elements + [Scale_Bovaer()])
    with_strat.run_until(2040 * u.years)
    emissions_strat = with_strat.compute_annual_emissions()
    subsidies_strat = with_strat.compute_annual_subsidies()

    print(emissions.sum() - emissions_strat.sum())
    print(subsidies_strat.sum() - subsidies.sum())
    co2e = emissions.sum() - emissions_strat.sum()
    cad = subsidies_strat.sum() - subsidies.sum()
    cost_per_tonne = (cad / co2e).to(u.CAD / u.tonne_CO2e)
    print(cost_per_tonne)
    assert 220 < cost_per_tonne.magnitude < 225
