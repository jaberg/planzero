from .my_functools import maybecache
from enum import Enum

import numpy as np

from .ureg import u, Geo
from .ureg import ElectricityGenerationTech
from . import eccc_nir_annex6
from . import eccc_nir_annex13
from .sts import annual_report
from . import sc_nir
from . import sts
from .ptvalues import PTValues
from .planet_model import CO2e_from_emissions

import matplotlib.pyplot as plt


def _repeat_prev_for_omitted_years(years, values):
    ii = 0
    year = years[0]
    value = values[0]
    rval = []
    try:
        while True:
            rval.append(value)
            year += 1
            if year == years[ii + 1]:
                value = values[ii + 1]
                ii += 1
    except IndexError:
        return rval


# https://en.wikipedia.org/wiki/Natural_gas#Energy_content,_statistics,_and_pricing
# Wikipedia says 39
# also Gemini says there is a standard value used by Statistics Canada and National Energy Board
# of 38.32, but no citation
Natural_Gas_gross_heating_value_marketable = 38.32 * u.MJ / u.m3_NG_mk

# Gemini says 40 to 45, no citation given
# but could be corroborated by averaging the various typical gases? What about moisture?
Natural_Gas_gross_heating_value_nonmarketable = 42 * u.MJ / u.m3_NG_nmk

# This is a guess
Natural_Gas_Peaker_thermal_efficiency = .31

@maybecache
def sts_a6_1_1(include_CA=True):
    """Dictionary mapping geographies to emissions factors of marketable natural gas by year
    """

    df_marketable_ng = eccc_nir_annex6.df_a6_1_1
    in_times = df_marketable_ng.Year.values
    out_times = np.arange(1990, 2024) * u.years

    rval = {}
    for geo in Geo:
        if not include_CA and geo == Geo.CA:
            continue
        if geo == Geo.CA:
            rval[Geo.CA] = annual_report(
                times=out_times,
                values=_repeat_prev_for_omitted_years(
                    in_times,
                    df_marketable_ng.Canada.values * u.g_CO2 / u.m3_NG_mk))
        else:
            rval[geo] = annual_report(
                times=out_times,
                values=_repeat_prev_for_omitted_years(
                    in_times,
                    getattr(df_marketable_ng, geo.code()).values * u.g_CO2 / u.m3_NG_mk))
    return rval


@maybecache
def sts_a6_1_2():
    """Dictionary mapping geographies to emissions factors of non-marketable natural gas by year
    """

    df = eccc_nir_annex6.df_a6_1_2
    in_times = df.Year.values
    out_times = np.arange(1990, 2024) * u.years

    rval = {}
    for geo in Geo:
        if geo == Geo.CA: # not present
            continue
        rval[geo] = annual_report(
            times=out_times,
            values=_repeat_prev_for_omitted_years(
                in_times,
                getattr(df, geo.code()).values * u.g_CO2 / u.m3_NG_nmk))
    return rval


def nir_a6_1_3():
    """Two dictionaries:
    * one mapping natural gas user groups to CH4 emission factors
    * one mapping natural gas user groups to N2O emission factors
    """
    data = eccc_nir_annex6.data_a6_1_3
    sources = data['Emission Factor Source']
    denom_unit = [u.m3_NG_mk] * 8
    sources[2:4] = ['Producer Consumption',
                    'Producer Consumption - Newfoundland and Labrador']
    denom_unit[2:4] = [u.m3_NG_nmk, u.m3_NG_nmk]

    CH4_coefs = [vv * u.g_CH4 / du for vv, du in zip(data['CH4 (g/m3)'], denom_unit)]
    N2O_coefs = [vv * u.g_N2O / du for vv, du in zip(data['N2O (g/m3)'], denom_unit)]
    assert len(sources) == len(CH4_coefs) == len(N2O_coefs)
    rval_CH4 = dict(zip(sources, CH4_coefs))
    rval_N2O = dict(zip(sources, N2O_coefs))
    return rval_CH4, rval_N2O


class A6_1_1(PTValues):
    def __init__(self):
        super().__init__(val_d=sts_a6_1_1(include_CA=False))

    def scatter(self):
        super().scatter()
        plt.title('Marketable Natural Gas CO2 Emission Factors')
        plt.legend(loc='lower right')


class A6_1_2(PTValues):
    def __init__(self):
        super().__init__(val_d=sts_a6_1_2())

    def scatter(self):
        super().scatter()
        plt.title('Non-Marketable Natural Gas CO2 Emission Factors')
        plt.legend(loc='lower right')



class Est_Natural_Gas_Used_by_Electricity_Utilities_2005_to_2013(PTValues):
    """ Geographies to volumes of marketable natural gas
    """
    def __init__(self, volume_unit='m3_NG_mk'):
        utility_gen_by_tech_geo = sc_nir.Electric_Power_Annual_Generation_by_Class_of_Producer.utility_gen_by_tech_geo()
        utility_gen_by_geo = utility_gen_by_tech_geo[ElectricityGenerationTech.CombustionTurbine]

        rval = {}
        for pt in Geo.provinces_and_territories():
            if pt in (Geo.YT, Geo.NU):
                assert pt not in utility_gen_by_geo
                rval[pt] = None
                continue
            # how much power did they produce
            energy_out = utility_gen_by_geo[pt]

            if pt == Geo.ON:
                # there's a big jump in consumption in 2010 due to 3 plants coming online
                # Halton Hills Generating Station, capacity 632MW
                # York Energy Centre, capacity 400MW
                # Sarnia (St. Claire) Expansion, capacity ~540MW
                # TODO: model these plants specifically, and others in Ontario
                thermal_efficiency = .45
            else:
                thermal_efficiency = Natural_Gas_Peaker_thermal_efficiency
            volume = energy_out / (thermal_efficiency * Natural_Gas_gross_heating_value_marketable)
            rval[pt] = volume.to(volume_unit)
        super().__init__(val_d=rval)

    def CO2e(self):
        marketable_NG_CO2_coefs = A6_1_1()
        CH4_coef_by_ng_user, N2O_coef_by_ng_user = nir_a6_1_3()

        CO2 = self * marketable_NG_CO2_coefs
        CH4 = self * CH4_coef_by_ng_user['Electric Utilities']
        N2O = self * N2O_coef_by_ng_user['Electric Utilities']

        CO2e = CO2e_from_emissions(CO2, CH4, N2O)
        return CO2e


class Est_Natural_Gas_Used_by_Industry_Electricity_Generation_2005_to_2013(PTValues):
    """ Geographies to volumes of marketable natural gas
    """
    def __init__(self, volume_unit='m3_NG_nmk'):
        gen_by_tech_geo = sc_nir.Electric_Power_Annual_Generation_by_Class_of_Producer.industry_gen_by_tech_geo()
        gen_by_geo = gen_by_tech_geo[ElectricityGenerationTech.CombustionTurbine]

        val_d = {}
        for pt in Geo.provinces_and_territories():
            if pt in (Geo.BC, Geo.SK, Geo.MB, Geo.QC, Geo.NS, Geo.NB, Geo.PE, Geo.YT, Geo.NU):
                assert pt not in gen_by_geo
                val_d[pt] = None
                continue
            # how much power did they produce
            energy_out = gen_by_geo[pt]

            # TODO: this is an approximation based on the guess that at least oil sands
            # wouldn't use combined cycle plants because they actually need the steam more
            # than the electricity
            thermal_efficiency = Natural_Gas_Peaker_thermal_efficiency
            volume = energy_out / (thermal_efficiency * Natural_Gas_gross_heating_value_nonmarketable)
            val_d[pt] = volume.to(volume_unit)
        super().__init__(val_d=val_d)

    def CO2e(self):
        nonmarketable_NG_CO2_coefs = A6_1_2()
        CH4_coef_by_ng_user, N2O_coef_by_ng_user = nir_a6_1_3()

        # TODO: switch to a PTValues of coefs instead of raw scalars here
        # because there's a different Producer Consumption value for Newfoundland and Labrador
        CO2 = self * nonmarketable_NG_CO2_coefs
        CH4 = self * CH4_coef_by_ng_user['Producer Consumption']
        N2O = self * N2O_coef_by_ng_user['Producer Consumption']

        CO2e = CO2e_from_emissions(CO2, CH4, N2O)
        return CO2e


def plot_delta_natural_gas_for_electricity_generation():
    industry = Est_Natural_Gas_Used_by_Industry_Electricity_Generation_2005_to_2013()
    utility = Est_Natural_Gas_Used_by_Electricity_Utilities_2005_to_2013()
    CO2e = industry.CO2e() + utility.CO2e()

    plt.figure()
    CO2e.national_total().plot(label='Estimate')
    a13_ng = eccc_nir_annex13.national_electricity_CO2e_from_combustion()['natural_gas']
    a13_ng.to(CO2e.v_unit).plot(label='Annex13 (Target)')
    plt.title('Electricity from Natural Gas')
    plt.legend(loc='upper left')
