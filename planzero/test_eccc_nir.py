import pytest

from .eccc_nir_annex6 import *
from .eccc_nir import *


def test_ngf_emissions_from_electricity_generation():
    plot_delta_natural_gas_for_electricity_generation()

def test_small_delta_natural_gas_for_electricity_2005():

    industry = Est_Natural_Gas_Used_by_Industry_Electricity_Generation_2005_to_2013()
    utility = Est_Natural_Gas_Used_by_Electricity_Utilities_2005_to_2013()
    CO2e = industry.CO2e() + utility.CO2e()
    estimate = CO2e.national_total()
    target = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['natural_gas']

    estimate_2005 = estimate.query(2005 * u.years)
    target_2005 = target.query(2005 * u.years)
    assert 14500 * u.kt_CO2e < target_2005 < 14505 * u.kt_CO2e
    assert 14100 * u.kt_CO2e < estimate_2005 < 14805 * u.kt_CO2e
