import enum
import functools

import pandas as pd

from .sc_utils import zip_table_to_dataframe, Factors, times_values
from .ureg import u
from . import enums
from . import objtensor
from . import sts
from .enums import NAICS, PT


class FuelType(str, enum.Enum):
    TotalGeneration = 'Total generation'

    LightFuelOil = 'Light fuel oil'
    TotalHeavyFuelOil = 'Total heavy fuel oil'
    CanadianHeavyFuelOil = 'Canadian heavy fuel oil'
    ImportedHeavyFuelOil = 'Imported heavy fuel oil'
    Diesel = 'Diesel'
    TotalPPs = 'Total petroleum products'
    SpentPulpingLiquor = 'Spent pulping liquor'
    OtherLiquid = "Other liquid fuels"
    TotalLiquids = 'Total liquids'

    NaturalGasMkt = 'Natural gas'
    CokeOvenGas = 'Coke oven gas'
    Biogas = 'Biogas'
    Propane = 'Propane'
    OtherGaseous = "Other gaseous fuels"
    SteamFromWasteHeat = 'Steam from waste heat'
    TotalGas = 'Total gas'

    Uranium = 'Uranium'
    TotalBituminousCoal = 'Total bituminous coal'
    TotalSubbituminous = 'Total subbituminous coal'
    CanadianBituminous = 'Canadian bituminous coal'
    CanadianSubbituminous = 'Canadian subbituminous coal'
    Lignite = 'Lignite'
    ImportedBituminous = 'Imported bituminous coal'
    ImportedSubbituminous = 'Imported subbituminous coal'
    TotalCoal = 'Total coal'
    PetCoke = 'Petroleum coke'
    Wood = 'Wood'
    OtherSolid = "Other solid fuels"
    TotalSolids = 'Total solids'


class MetaFuelType(str, enum.Enum):
    Electricity_Generated = 'electricity generated'
    Cost_of_Fuel = 'cost of fuel'
    Fuel_Consumed = 'fuel consumed'


UoM_by_fuel_type = {
    'Biogas': ('Cubic metres', u.m3_biogas),
    'Canadian bituminous coal': ('Metric tonnes', u.tonne_coal_bit),
    'Canadian heavy fuel oil': ('Kilolitres', u.kilolitres_HFO),
    'Canadian subbituminous coal': ('Metric tonnes', u.tonne_coal_subbit),
    'Coke oven gas': ('Cubic metres', u.m3_ovengas),
    'Diesel': ('Kilolitres', u.kilolitres_diesel),
    'Imported bituminous coal': ('Metric tonnes', u.tonne_coal_bit),
    'Imported heavy fuel oil': ('Kilolitres', u.kilolitres_HFO),
    'Imported subbituminous coal': ('Metric tonnes', u.tonne_coal_subbit),
    'Light fuel oil': ('Kilolitres', u.kilolitres_LFO),
    'Lignite': ('Metric tonnes', u.tonne_lignite),
    'Natural gas': ('Cubic metres', u.m3_NG_mk),
    'Other gaseous fuels': ('Cubic metres', u.m3),
    'Other liquid fuels': ('Kilolitres', u.kilolitres),
    'Other solid fuels': ('Metric tonnes', u.tonne),
    'Petroleum coke': ('Metric tonnes', u.tonne_petcoke),
    'Propane': ('Kilolitres', u.kilolitres_propane),
    'Spent pulping liquor': ('Kilolitres', u.kilolitres_pulpingliquor),
    'Total bituminous coal': ('Metric tonnes', u.tonne_coal_bit),
    'Total coal': ('Metric tonnes', u.tonne_coal),
    'Total gas': ('Cubic metres', u.m3),
    'Total heavy fuel oil': ('Kilolitres', u.kilolitres_HFO),
    'Total liquids': ('Kilolitres', u.kilolitres),
    'Total petroleum products': ('Kilolitres', u.kilolitres),
    'Total solids': ('Metric tonnes', u.tonne),
    'Total subbituminous coal': ('Metric tonnes', u.tonne_coal_subbit),
    'Uranium': ('Kilograms', u.kg_uranium),
    'Wood': ('Metric tonnes',u.tonne_wood_mc50)}


def _get_df():
    df = zip_table_to_dataframe("25-10-0084-01")
    df['REF_DATE'] = df['REF_DATE'].apply(pd.to_numeric)
    return df


@functools.cache
def electric_power_generation_fuel_consumed_cost_of_fuel():
    df = _get_df()

    rval_pt = objtensor.empty(MetaFuelType, FuelType, NAICS, PT)
    rval_ca = objtensor.empty(MetaFuelType, FuelType, NAICS)

    for rval in rval_pt, rval_ca:
        rval[MetaFuelType.Electricity_Generated] = 0 * u.megawatt_hours
        rval[MetaFuelType.Cost_of_Fuel] = 0 * u.CAD
        for ft in FuelType:
            rval[MetaFuelType.Fuel_Consumed, ft] \
                    = 0 * UoM_by_fuel_type.get(ft, (None, u.dimensionless))[1]

    min_year_incl = 2020
    max_year_excl = 2025

    for ft_, df_ft in df.groupby('Fuel type'):
        ft_str, mft_str = ft_.split(',')
        ft = FuelType(ft_str)
        mft = MetaFuelType(mft_str.lstrip())
        for naics_str, df_naics in df_ft.groupby('North American Industry Classification System (NAICS)'):
            naics = NAICS(naics_str)
            for geo_str, df_geo in df_naics.groupby('GEO'):
                if mft == MetaFuelType.Cost_of_Fuel:
                    dollars, = set(df_geo.UOM.values)
                    assert dollars == 'Dollars', dollars
                    v_unit = u.CAD
                elif mft == MetaFuelType.Electricity_Generated:
                    mwh, = set(df_geo.UOM.values)
                    assert mwh == 'Megawatt hours', mwh
                    v_unit = u.megawatt_hours
                else:
                    expected_uom, v_unit = UoM_by_fuel_type[ft_str]
                    actual_uom, = set(df_geo.UOM.values)
                    assert actual_uom == expected_uom

                factor_str, = set(df_geo.SCALAR_FACTOR.values)
                factor = Factors[factor_str]
                times, values = times_values(df_geo, min_year_incl, max_year_excl)
                rep = sts.annual_report2(
                    years=times,
                    values=[vv * factor for vv in values],
                    v_unit=v_unit)

                if geo_str == 'Canada':
                    rval_ca[mft, ft, naics] += rep
                else:
                    pt = enums.PT(geo_str)
                    rval_pt[mft, ft, naics, pt] += rep
    return rval_pt, rval_ca
