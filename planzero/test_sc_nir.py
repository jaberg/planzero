#from . import sc_nir
from . import ipcc_canada
from . import ureg as u
import pytest

@pytest.mark.skip
def test_electric_power_annual_generation_1():
    import warnings
    warnings.filterwarnings("error")
    sc_nir.Electric_Power_Annual_Generation_by_Class_of_Producer.utility_gen_by_tech_geo()
    sc_nir.Electric_Power_Annual_Generation_by_Class_of_Producer.industry_gen_by_tech_geo()

@pytest.mark.skip
def test_CO2_2005():
    a = sc_nir.CO2_emissions_from_electricity_generation_2005()
    b = sc_nir.CO2_emissions_from_other_sources_2005_energy()
    c = sc_nir.CO2_emissions_from_other_sources_2005_natural_units()

    inv = ipcc_canada.inv

    target, = inv[(inv['Year'] == 2005) & inv['Region'].isin(['Canada', 'canada']) & (inv['Source'] == 'Energy') & (inv['Category'].isna())]['CO2']
    target = float(target) * u.kilotonne

    est = a + b

    print(target.to('megatonne').magnitude)
    print(est.to('megatonne').magnitude)
    print((a + c).to('megatonne').magnitude)
    print(((est - target) / target).to('dimensionless').magnitude)
    assert 0


