import pytest
from .enums import PT
from .ureg import u
from . import est_nir
from . import eccc_nir_annex13

def test_small_delta_natural_gas_for_electricity_2005():
    est = est_nir.EstAnnex13ElectricityFromNaturalGas()
    estimate = est.co2e.sum(PT)

    target = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['natural_gas']

    target_2005 = target.query(2005 * u.years)
    estimate_2005 = estimate.query(2005 * u.years).to(target_2005.u)
    assert 14500 * u.kt_CO2e < target_2005 < 14505 * u.kt_CO2e
    assert 14100 * u.kt_CO2e < estimate_2005 < 14805 * u.kt_CO2e


def test_annex13_electricity_from_coal():
    est = est_nir.EstAnnex13ElectricityFromCoal()
    estimate = est.co2e.sum(PT)

    a13_ng = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['coal']
    target = a13_ng.to(est.co2e.as_one_v_unit())

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


def test_annex13_electricity_from_other():
    est = est_nir.EstAnnex13ElectricityFromOther()
    estimate = est.co2e.sum(PT)

    a13_ng = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['other_fuels']
    target = a13_ng.to(estimate.v_unit)

    estimate_2005 = estimate.query(2005 * u.years)
    target_2005 = target.query(2005 * u.years)
    assert 10_000 * u.kt_CO2e < target_2005 < 11_000 * u.kt_CO2e
    # TODO: Not sure why it's high.
    assert 10_000 * u.kt_CO2e < estimate_2005 < 12_000 * u.kt_CO2e

    estimate_2010 = estimate.query(2010 * u.years)
    target_2010 = target.query(2010 * u.years)
    assert 4_000 * u.kt_CO2e < target_2010 < 6_000 * u.kt_CO2e
    assert 4_000 * u.kt_CO2e < estimate_2010 < 6_000 * u.kt_CO2e

    estimate_2015 = estimate.query(2015 * u.years)
    target_2015 = target.query(2015 * u.years)
    assert 4_000 * u.kt_CO2e < target_2015 < 6_000 * u.kt_CO2e
    assert 4_000 * u.kt_CO2e < estimate_2015 < 6_000 * u.kt_CO2e
