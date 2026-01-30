import functools
from enum import Enum

import numpy as np

from .ureg import u, Geo
from .ureg import ElectricityGenerationTech
from . import eccc_nir_annex6
from .sts import annual_summary
from . import sc_nir

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


def plot_sts_dict(d, t_unit, v_unit, **kwargs):
    for key, val in d.items():
        plt.scatter(
            (np.asarray(val.times) * val.t_unit).to(t_unit).magnitude,
            (np.asarray(val.values[1:]) * val.v_unit).to(v_unit).magnitude,
            label=key,
            **kwargs)
        plt.xlabel(t_unit)
        plt.ylabel(v_unit)


# https://en.wikipedia.org/wiki/Natural_gas#Energy_content,_statistics,_and_pricing
# Wikipedia says 39
# also Gemini says there is a standard value used by Statistics Canada and National Energy Board
# of 38.32, but no citation
Natural_Gas_gross_heating_value_marketable = 38.32 * u.MJ / u.m3

# Gemini says 40 to 45, no citation given
# but could be corroborated by averaging the various typical gases? What about moisture?
Natural_Gas_gross_heating_value_nonmarketable = 42 * u.MJ / u.m3

# This is a guess
Natural_Gas_Peaker_thermal_efficiency = .30

class NaturalGasFactorsCO2(object):

    @functools.cache
    @staticmethod
    def sts_a6_1_1():
        """Dictionary mapping geographies to emissions factors of marketable natural gas by year
        """

        df_marketable_ng = eccc_nir_annex6.df_a6_1_1
        in_times = df_marketable_ng.Year.values
        out_times = np.arange(1990, 2024) * u.years

        rval = {}
        for geo in Geo:
            if geo == Geo.CA:
                rval[Geo.CA] = annual_summary(
                    times=out_times,
                    values=_repeat_prev_for_omitted_years(
                        in_times,
                        df_marketable_ng.Canada.values * u.g / u.m3))
            else:
                rval[geo] = annual_summary(
                    times=out_times,
                    values=_repeat_prev_for_omitted_years(
                        in_times,
                        getattr(df_marketable_ng, geo.code()).values * u.g / u.m3))
        return rval

    @staticmethod
    def plot_marketable():
        plot_sts_dict(NaturalGasFactorsCO2.sts_a6_1_1(), t_unit=u.year, v_unit=u.g / u.m3)
        plt.title('Marketable Natural Gas CO2 Emission Factors')
        plt.legend(loc='lower right')

    @functools.cache
    @staticmethod
    def sts_a6_1_2():
        """Dictionary mapping geographies to emissions factors of non-marketable natural gas by year
        """

        df = eccc_nir_annex6.df_a6_1_2
        in_times = df.Year.values
        out_times = np.arange(1990, 2024) * u.years

        rval = {}
        for geo in Geo:
            if geo == Geo.CA:
                continue
            rval[geo] = annual_summary(
                times=out_times,
                values=_repeat_prev_for_omitted_years(
                    in_times,
                    getattr(df, geo.code()).values * u.g / u.m3))
        return rval

    @staticmethod
    def plot_nonmarketable():
        plot_sts_dict(NaturalGasFactorsCO2.sts_a6_1_2(), t_unit=u.year, v_unit=u.g / u.m3)
        plt.title('Non-Marketable Natural Gas CO2 Emission Factors')
        plt.legend(loc='lower right')

    @staticmethod
    def utilities_emissions_from_electricity_generation_2005_to_2013():
        utility_gen_by_tech_geo = sc_nir.Electric_Power_Annual_Generation_by_Class_of_Producer.utility_gen_by_tech_geo()
        utility_gen_by_geo = utility_gen_by_tech_geo[ElectricityGenerationTech.CombustionTurbine]

        marketable = NaturalGasFactorsCO2.sts_a6_1_1()

        rval = {}
        for pt in Geo.provinces_and_territories():
            if pt in (Geo.YT, Geo.NU):
                # no generation in these areas
                continue
            # how much power did they produce
            energy_out = utility_gen_by_geo[pt]

            # guessing here, can't find data.
            # gas turbines of 2005-2014 era were mostly simple cycle
            # so... ?
            thermal_efficiency = Natural_Gas_Peaker_thermal_efficiency
            mass_in = energy_out / (thermal_efficiency * Natural_Gas_gross_heating_value_marketable)

            emissions_out = mass_in * marketable[pt]
            rval[pt] = emissions_out
        return rval

    @staticmethod
    def plot_utilities_emissions_from_electricity_generation_2005_to_2013():
        plot_sts_dict(
            NaturalGasFactorsCO2.utilities_emissions_from_electricity_generation_2005_to_2013(),
            t_unit=u.year,
            v_unit=u.megatonne)
        plt.title('Electricity Utilities CO2 Emissions from Natural Gas')
        plt.legend(loc='upper left')

    @staticmethod
    def industries_emissions_from_electricity_generation_2005_to_2013():
        industry_gen_by_tech_geo = sc_nir.Electric_Power_Annual_Generation_by_Class_of_Producer.industry_gen_by_tech_geo()
        industry_gen_by_geo = industry_gen_by_tech_geo[ElectricityGenerationTech.CombustionTurbine]
        nonmarketable = NaturalGasFactorsCO2.sts_a6_1_2()
        rval = {}
        for pt in Geo.provinces_and_territories():
            if pt in (Geo.BC, Geo.SK, Geo.MB, Geo.QC, Geo.NS, Geo.NB, Geo.PE, Geo.YT, Geo.NU):
                # TODO: try/except to at least check if these areas have data
                # Then if they don't, it's okay if they're in the list
                # no generation in these areas
                continue
            # how much power did they produce
            energy_out = industry_gen_by_geo[pt]

            # guessing here, can't find data.
            # gas turbines of 2005-2014 era were mostly simple cycle
            # also in Alberta they wouldn't be combined cycle for oil sands
            # because they use the energy from the steam underground?
            # so... ?
            thermal_efficiency = Natural_Gas_Peaker_thermal_efficiency
            # This gross_heating_value is for *marketable* Natural gas
            mass_in = energy_out / (thermal_efficiency * Natural_Gas_gross_heating_value_nonmarketable)
            emissions_out = mass_in * nonmarketable[pt]
            rval[pt] = emissions_out
        return rval

    @staticmethod
    def plot_industries_emissions_from_electricity_generation_2005_to_2013():
        plot_sts_dict(
            NaturalGasFactorsCO2.industries_emissions_from_electricity_generation_2005_to_2013(),
            t_unit=u.year,
            v_unit=u.megatonne)
        plt.title('Industrial Electricity CO2 Emissions from Natural Gas')
        plt.legend(loc='upper left')
