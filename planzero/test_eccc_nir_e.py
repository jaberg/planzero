from .ureg import u
from . import eccc_nir_annex13
from .planet_model import CO2e_from_emissions

from .eccc_nir_e_from_ng import (
    Est_Natural_Gas_Used_by_Industry_Electricity_Generation_2005_to_2013,
    Est_Natural_Gas_Used_by_Electricity_Utilities_2005_to_2013)

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

from . import eccc_nir_e_from_coal

def test_small_delta_coal_for_electricity_2005_2010_2015_2020():
    co2 = eccc_nir_e_from_coal.CO2()
    ch4 = eccc_nir_e_from_coal.CH4()
    n2o = eccc_nir_e_from_coal.N2O()
    co2e = CO2e_from_emissions(co2, ch4, n2o)
    estimate = co2e.national_total()

    a13_ng = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['coal']
    target = a13_ng.to(co2e.v_unit)

    estimate_2005 = estimate.query(2005 * u.years)
    target_2005 = target.query(2005 * u.years)
    assert 92_000 * u.kt_CO2e < target_2005 < 98_000 * u.kt_CO2e
    assert 92_000 * u.kt_CO2e < estimate_2005 < 98_000 * u.kt_CO2e

    estimate_2010 = estimate.query(2010 * u.years)
    target_2010 = target.query(2010 * u.years)
    assert 75_000 * u.kt_CO2e < target_2010 < 80_000 * u.kt_CO2e
    assert 75_000 * u.kt_CO2e < estimate_2010 < 80_000 * u.kt_CO2e

    estimate_2015 = estimate.query(2015 * u.years)
    target_2015 = target.query(2015 * u.years)
    assert 55_000 * u.kt_CO2e < target_2015 < 60_000 * u.kt_CO2e
    assert 55_000 * u.kt_CO2e < estimate_2015 < 60_000 * u.kt_CO2e

    estimate_2020 = estimate.query(2020 * u.years)
    target_2020 = target.query(2020 * u.years)
    assert 32_000 * u.kt_CO2e < target_2020 < 36_000 * u.kt_CO2e
    assert 32_000 * u.kt_CO2e < estimate_2020 < 36_000 * u.kt_CO2e
