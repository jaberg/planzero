import pandas as pd
import numpy as np

from . import enums
from .enums import GHG, CoalType, PT
from .ureg import u, tonne_by_coal_type, kg_by_ghg, litres_by_fuel_type
from .sc_utils import zip_table_to_dataframe
from . import objtensor
from . import sts
from . import eccc_nir_annex6


def _repeat_prev_for_omitted_years(years, values, replace_nan_with=None):
    ii = 0
    year = years[0]
    value = values[0]
    rval = []
    try:
        while True:
            if np.isnan(value) and replace_nan_with is not None:
                rval.append(replace_nan_with)
            else:
                rval.append(value)
            year += 1
            if year == years[ii + 1]:
                value = values[ii + 1]
                ii += 1
    except IndexError:
        return rval


def A6_1_1(): # marketable natural gas

    df = eccc_nir_annex6.df_a6_1_1
    in_times = df.Year.values
    out_times = np.arange(1990, 2024) * u.years
    notice_if_used = 1e10 * u.g_CO2 / u.m3_NG_mk # There is a "Canada" column that could work

    rval = {}
    for pt in PT:
        if pt == PT.XX: # not present
            rval[pt] = notice_if_used
        else:
            values = _repeat_prev_for_omitted_years(
                in_times,
                getattr(df, pt.two_letter_code()).values * u.g_CO2 / u.m3_NG_mk,
                replace_nan_with=notice_if_used)
            rval[pt] = sts.annual_report(times=out_times, values=values)
            assert np.isfinite(rval[pt].values[1:]).all()
            assert len(rval[pt].times) > 0, (pt, values)
    return objtensor.from_dict(rval)


def A6_1_2(): # non-marketable natural gas
    df = eccc_nir_annex6.df_a6_1_2
    in_times = df.Year.values
    out_times = np.arange(1990, 2024) * u.years
    notice_if_used = 1e10 * u.g_CO2 / u.m3_NG_nmk

    rval = {}
    for pt in PT:
        if pt == PT.XX:
            rval[pt] = notice_if_used
            continue
        values = _repeat_prev_for_omitted_years(
            in_times,
            getattr(df, pt.two_letter_code()).values * u.g_CO2 / u.m3_NG_nmk,
            replace_nan_with=notice_if_used)
        rval[pt] = sts.annual_report(times=out_times, values=values)
        assert len(rval[pt].times)
    return objtensor.from_dict(rval)


def A6_1_3_and_1_4():
    GHG = enums.GHG
    NGU = enums.NaturalGasUser
    PT = enums.PT
    rval = objtensor.empty([GHG.CH4, GHG.N2O], NGU, PT)

    # start with Table 1-3, CH4
    rval[GHG.CH4] = .037 * u.g_CH4 / u.m3_NG_mk
    rval[GHG.CH4, NGU.ElectricUtilities] = .490 * u.g_CH4 / u.m3_NG_mk
    rval[GHG.CH4, NGU.Producer] = 6.4 * u.g_CH4 / u.m3_NG_nmk
    rval[GHG.CH4, NGU.Producer, PT.NL] = .490 * u.g_CH4 / u.m3_NG_nmk
    rval[GHG.CH4, NGU.Pipelines] = 1.90 * u.g_CH4 / u.m3_NG_mk

    # start with Table 1-3, N2O
    rval[GHG.N2O] = .035 * u.g_N2O / u.m3_NG_mk
    rval[GHG.N2O, NGU.ElectricUtilities] = .0490 * u.g_N2O / u.m3_NG_mk
    rval[GHG.N2O, NGU.Producer] = 0.06 * u.g_N2O / u.m3_NG_nmk
    rval[GHG.N2O, NGU.Pipelines] = 0.05 * u.g_N2O / u.m3_NG_mk
    rval[GHG.N2O, NGU.Cement] = 0.034 * u.g_N2O / u.m3_NG_mk
    rval[GHG.N2O, NGU.Manufacturing] = 0.033 * u.g_N2O / u.m3_NG_mk

    # Now bring in Table 1-4
    # with year-by-year CH4 factors for western provinces
    df = eccc_nir_annex6.df_a6_1_4
    rval[GHG.CH4, NGU.Producer, PT.BC] = sts.annual_report(
        times=list(df.Year.values * u.years),
        values=list(df.BC.values * u.g_CH4 / u.m3_NG_nmk))
    rval[GHG.CH4, NGU.Producer, PT.AB] = sts.annual_report(
        times=list(df.Year.values * u.years),
        values=list(df.AB.values * u.g_CH4 / u.m3_NG_nmk))
    rval[GHG.CH4, NGU.Producer, PT.SK] = sts.annual_report(
        times=list(df.Year.values * u.years),
        values=list(df.SK.values * u.g_CH4 / u.m3_NG_nmk))

    return rval


def A6_1_6_LFO_HFO_Kerosene():
    GHG = enums.GHG
    FuelType = enums.FuelType
    RPP_User = enums.RPP_User
    Fuels = [FuelType.LightFuelOil, FuelType.HeavyFuelOil, FuelType.Kerosene]

    rval = objtensor.empty(GHG, Fuels, RPP_User)

    for ghg in [GHG.HFCs, GHG.PFCs, GHG.SF6, GHG.NF3]:
        for fuel_type in Fuels:
            rval[ghg, fuel_type] = (
                0 * kg_by_ghg[ghg] / litres_by_fuel_type[fuel_type])

    # CO2
    rval[GHG.CO2, FuelType.LightFuelOil] = 2753 * u.g_CO2 / u.l_LFO
    rval[GHG.CO2, FuelType.LightFuelOil, RPP_User.Producer] = 2670 * u.g_CO2 / u.l_LFO
    rval[GHG.CO2, FuelType.HeavyFuelOil] = 3156 * u.g_CO2 / u.l_HFO
    rval[GHG.CO2, FuelType.LightFuelOil, RPP_User.Producer] = 3190 * u.g_CO2 / u.l_LFO
    rval[GHG.CO2, FuelType.Kerosene] = 2560 * u.g_CO2 / u.l_kerosene

    # CH4
    rval[GHG.CH4, FuelType.LightFuelOil] = 0.026 * u.g_CH4 / u.l_LFO
    rval[GHG.CH4, FuelType.LightFuelOil, RPP_User.ElectricUtilities] = 0.18 * u.g_CH4 / u.l_LFO
    rval[GHG.CH4, FuelType.LightFuelOil, RPP_User.Industrial] = 0.006 * u.g_CH4 / u.l_LFO
    rval[GHG.CH4, FuelType.LightFuelOil, RPP_User.Producer] = 0.006 * u.g_CH4 / u.l_LFO

    rval[GHG.CH4, FuelType.HeavyFuelOil] = 0.057 * u.g_CH4 / u.l_HFO
    rval[GHG.CH4, FuelType.HeavyFuelOil, RPP_User.ElectricUtilities] = 0.034 * u.g_CH4 / u.l_HFO
    rval[GHG.CH4, FuelType.HeavyFuelOil, RPP_User.Industrial] = 0.12 * u.g_CH4 / u.l_HFO
    rval[GHG.CH4, FuelType.HeavyFuelOil, RPP_User.Producer] = 0.12 * u.g_CH4 / u.l_HFO

    rval[GHG.CH4, FuelType.Kerosene] = .026 * u.g_CH4 / u.l_kerosene
    rval[GHG.CH4, FuelType.Kerosene, [
        RPP_User.ElectricUtilities,
        RPP_User.Industrial,
        RPP_User.Producer,
    ]] = .006 * u.g_CH4 / u.l_kerosene

    # N2O
    rval[GHG.N2O, FuelType.LightFuelOil] = 0.031 * u.g_N2O / u.l_LFO
    rval[GHG.N2O, FuelType.LightFuelOil, RPP_User.Residential] = 0.006 * u.g_N2O / u.l_LFO
    rval[GHG.N2O, FuelType.HeavyFuelOil] = 0.064 * u.g_N2O / u.l_HFO
    rval[GHG.N2O, FuelType.Kerosene] = 0.031 * u.g_N2O / u.l_kerosene
    rval[GHG.N2O, FuelType.Kerosene, RPP_User.Residential] = 0.006 * u.g_N2O / u.l_kerosene
    
    return rval


def A6_1_6_Diesel_and_Gasoline():
    GHG = enums.GHG
    FuelType = enums.FuelType
    Fuels = FuelType.Diesel, FuelType.Gasoline
    rval = objtensor.empty(GHG, Fuels)

    for ghg in [GHG.HFCs, GHG.PFCs, GHG.SF6, GHG.NF3]:
        for fuel_type in Fuels:
            rval[ghg, fuel_type] = (
                0 * kg_by_ghg[ghg] / litres_by_fuel_type[fuel_type])

    rval[GHG.CO2, FuelType.Diesel] = 2681 * u.g_CO2 / u.l_diesel
    rval[GHG.CO2, FuelType.Gasoline] = 2307 * u.g_CO2 / u.l_gasoline

    rval[GHG.CH4, FuelType.Diesel] = .78 * u.g_CH4 / u.l_diesel
    rval[GHG.CH4, FuelType.Gasoline] = .1 * u.g_CH4 / u.l_gasoline

    rval[GHG.N2O, FuelType.Diesel] = .022 * u.g_N2O / u.l_diesel
    rval[GHG.N2O, FuelType.Gasoline] = .02 * u.g_N2O / u.l_gasoline

    return rval


def A6_1_7_and_1_8_and_1_9():
    GHG = enums.GHG

    FT = enums.FuelType
    Fuels = FT.PetCoke, FT.StillGas

    # TODO: RPP stands for "Refined Petroleum Products"
    #       which isn't the intended descriptor of users of PetCoke and StillGas
    RPP_User = enums.RPP_User
    Users = RPP_User.UpgradingFacilities, RPP_User.RefineriesAndOthers

    rval = objtensor.empty(GHG, Fuels, Users) # TABLE DIMENSIONS

    table7 = eccc_nir_annex6.data_a6_1_7
    table7_years = [int(year) for year in table7['Year']]
    years_since_1990 = [year * u.years for year in range(1990, max(table7_years) + 1)]

    # CO2 petcoke upgraders
    data_cell = table7['Petroleum Coke - Upgrading Facilities (g/L)']
    table_vals = [float(val) * u.g_CO2 / u.l_petcoke for val in data_cell]
    vals_since_1990 = _repeat_prev_for_omitted_years(table7_years, table_vals)
    rval[GHG.CO2, FT.PetCoke, RPP_User.UpgradingFacilities] = \
            sts.annual_report(times=years_since_1990, values=vals_since_1990)

    # CO2 petcoke refineries and others
    data_cell = table7['Petroleum Coke - Refineries and Others (g/L)']
    table_vals = [float(val) * u.g_CO2 / u.l_petcoke for val in data_cell]
    vals_since_1990 = _repeat_prev_for_omitted_years(table7_years, table_vals)
    rval[GHG.CO2, FT.PetCoke, RPP_User.RefineriesAndOthers] = \
            sts.annual_report(times=years_since_1990, values=vals_since_1990)

    # CO2 stillgas upgraders
    data_cell = table7['Still Gas - Upgrading Facilities (g/m3)']
    table_vals = [float(val) * u.g_CO2 / u.m3_stillgas for val in data_cell]
    vals_since_1990 = _repeat_prev_for_omitted_years(table7_years, table_vals)
    rval[GHG.CO2, FT.StillGas, RPP_User.UpgradingFacilities] = \
            sts.annual_report(times=years_since_1990, values=vals_since_1990)

    # CO2 stillgas refineries and others
    data_cell = table7['Still Gas - Refineries and Others (g/m3)']
    table_vals = [float(val) * u.g_CO2 / u.m3_stillgas for val in data_cell]
    vals_since_1990 = _repeat_prev_for_omitted_years(table7_years, table_vals)
    rval[GHG.CO2, FT.StillGas, RPP_User.RefineriesAndOthers] = \
            sts.annual_report(times=years_since_1990, values=vals_since_1990)

    # CH4 petcoke (from A6 Table 1-6)
    rval[GHG.CH4, FT.PetCoke] = 0.12 * u.g_CH4 / u.l_petcoke

    # CH4 stillgas (aka A6 Table 1-9)
    table9 = eccc_nir_annex6.data_a6_1_9
    table9_years = [int(year) * u.years for year in table9['Year']]
    table9_vals = [vv * u.g_CH4 / u.m3_stillgas for vv in table9['CH4 Emission Factor (g/m^3)']]
    rval[GHG.CH4, FT.StillGas, RPP_User.UpgradingFacilities] = \
            0.000039 * u.g_CH4 / u.m3_stillgas
    rval[GHG.CH4, FT.StillGas, RPP_User.RefineriesAndOthers] = \
            sts.annual_report(times=table9_years, values=table9_vals)

    # N2O petcoke (aka A6 Table 1-8)
    table8 = eccc_nir_annex6.data_a6_1_8
    table8_years = [int(year) for year in table8['Year']]
    table8_vals_up = [vv * u.g_N2O / u.m3_petcoke for vv in table8['Upgrading Facilities (g/m3)']]
    vals_since_1990_up = _repeat_prev_for_omitted_years(table8_years, table8_vals_up)
    rval[GHG.N2O, FT.PetCoke, RPP_User.UpgradingFacilities] = \
            sts.annual_report(times=years_since_1990, values=vals_since_1990_up)

    table8_vals_ref = [vv * u.g_N2O / u.m3_petcoke for vv in table8['Refineries and Others (g/m3)']]
    vals_since_1990_ref = _repeat_prev_for_omitted_years(table8_years, table8_vals_ref)
    rval[GHG.N2O, FT.PetCoke, RPP_User.RefineriesAndOthers] = \
            sts.annual_report(times=years_since_1990, values=vals_since_1990_ref)

    # N2O stillgas (from A6 Table 1-6)
    rval[GHG.N2O, FT.StillGas] = 0.00002 * u.g_N2O / u.l_stillgas

    for ghg in [GHG.HFCs, GHG.PFCs, GHG.SF6, GHG.NF3]:
        for fuel_type in Fuels:
            rval[ghg, fuel_type] = (
                0 * kg_by_ghg[ghg] / litres_by_fuel_type[fuel_type])

    assert None not in rval.buf, [elem is None for elem in rval.buf]
    return rval


def A6_1_10_and_12():
    rval = objtensor.empty(GHG, CoalType, PT)

    for ghg in [GHG.HFCs, GHG.PFCs, GHG.SF6, GHG.NF3]:
        for coal_type in CoalType:
            rval[ghg, coal_type] = (
                0 * kg_by_ghg[ghg] / tonne_by_coal_type[coal_type])

    # put a massive number here so that it shows up if it's used
    # but if it's multiplied by 0, it's fine (not e.g. NaN).
    for coal_type in CoalType:
        rval[GHG.CO2, coal_type] = (
            1e10 * kg_by_ghg[GHG.CO2] / tonne_by_coal_type[coal_type])

    # table 1-12
    for coal_type in CoalType:
        rval[GHG.CH4, coal_type] = (
            .02 / 1000 * kg_by_ghg[GHG.CH4] / tonne_by_coal_type[coal_type])

    for coal_type in CoalType:
        rval[GHG.N2O, coal_type] = (
            .03 / 1000 * kg_by_ghg[GHG.N2O] / tonne_by_coal_type[coal_type])

    # table 1-10
    co2_canbit = rval[GHG.CO2, CoalType.CanadianBituminous]
    co2_canbit[[PT.NL, PT.NS, PT.PE, PT.QC]] = sts.SparseTimeSeries(
        times=[2000 * u.years],
        values=[2218 * u.kg_CO2 / u.tonne_coal_bit],
        default_value=2344 * u.kg_CO2 / u.tonne_coal_bit,
        t_unit=u.years)
    co2_canbit[PT.NB] = sts.SparseTimeSeries(
        times=[2010 * u.years],
        values=[2212 * u.kg_CO2 / u.tonne_coal_bit],
        default_value=2333 * u.kg_CO2 / u.tonne_coal_bit,
        t_unit=u.years)
    co2_canbit[[PT.ON, PT.MB, PT.SK, PT.AB, PT.BC]] = 2212 * u.kg_CO2 / u.tonne_coal_bit

    co2_impbit = rval[GHG.CO2, CoalType.ImportedBituminous]
    co2_impbit[[PT.NB, PT.NS, PT.PE, PT.NL]] = 2571 * u.kg_CO2 / u.tonne_coal_bit
    co2_impbit[[PT.MB, PT.ON]] = 2651 * u.kg_CO2 / u.tonne_coal_bit
    co2_impbit[[PT.QC, PT.AB, PT.BC]] = 2662 * u.kg_CO2 / u.tonne_coal_bit

    rval[GHG.CO2, CoalType.Lignite] = 1463 * u.kg_CO2 / u.tonne_lignite

    subs = [CoalType.CanadianSubbituminous, CoalType.ImportedSubbituminous]
    rval[GHG.CO2, subs, [PT.QC, PT.ON, PT.MB]] = 1865 * u.kg_CO2 / u.tonne_coal_subbit
    rval[GHG.CO2, subs, [PT.NS, PT.PE]] = 1743 * u.kg_CO2 / u.tonne_coal_subbit
    rval[GHG.CO2, subs, [PT.SK, PT.AB, PT.BC]] = 1775 * u.kg_CO2 / u.tonne_coal_subbit
    rval[GHG.CO2, subs, PT.NB] = sts.SparseTimeSeries(
            times=[
                2010 * u.years,
                2011 * u.years,
                2012 * u.years,
                2013 * u.years,
                2014 * u.years,
                2015 * u.years],
            values=[
                2189 * u.kg_CO2 / u.tonne_coal_subbit,
                2189 * u.kg_CO2 / u.tonne_coal_subbit,
                2352 * u.kg_CO2 / u.tonne_coal_subbit,
                2189 * u.kg_CO2 / u.tonne_coal_subbit,
                2189 * u.kg_CO2 / u.tonne_coal_subbit,
                2352 * u.kg_CO2 / u.tonne_coal_subbit],
            default_value=1e10 * u.kg_CO2 / u.tonne_coal_subbit, # should only be used to multiply by zero
            t_unit=u.years)

    return rval
