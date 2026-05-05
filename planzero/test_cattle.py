from pprint import pprint
from .cattle import *
from .base import Scenario


def test_cattle_emissions_sum():
    scenario = Scenario(name='foo', t_start=2000 * u.years)
    scenario.add_dynamic_elements([
        Cattle_Population_AR(),
        Cattle_Enteric_Emission_Rates_NIR2025_Bovaer(),
    ])
    scenario.run_until(2040 * u.years)
    emissions = scenario.compute_annual_emissions()
    n_provinces_and_territories_reporting = 10
    assert len(emissions.by_sector_ghg_pt_driver) == (
        n_provinces_and_territories_reporting * len(Livestock_nonsums))
    print(emissions.by_sector_ghg_pt_driver.keys())
    assert 41433 < emissions.sum().magnitude < 41434
