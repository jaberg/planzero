import enum
import functools

import pandas as pd
import numpy as np

from .ureg import u
from . import enums
from . import objtensor
from . import sts
from .sc_utils import zip_table_to_dataframe
from .sc_np import Factors, set_ptxx

class Fuel_Type(str, enum.Enum):
    Total_Coal = 'Total coal, primary energy'
    Canadian_Bituminous = 'Canadian bituminous, primary energy'
    Sub_Bituminous = 'Sub bituminous, primary energy'
    Lignite = 'Lignite, primary energy'
    Anthracite = 'Anthracite, primary energy'
    Imported_Bituminous = 'Imported bituminous, primary energy'
    Crude = 'Crude oil, primary energy'
    Natural_Gas = 'Natural gas, primary energy'
    NGLs = "Gas plant natural gas liquids (NGL's), primary energy"
    Primary_Electricity = 'Primary electricity, hydro and nuclear, primary energy'
    Steam_Primary = 'Steam, primary energy'
    Coke = 'Coke, secondary energy'
    Coke_Oven_Gas = 'Coke oven gas, secondary energy'
    Total_RPPs = 'Total refined petroleum products, secondary energy'
    LPGs = "Refinery liquefied petroleum gases (LPG's), secondary energy"
    Still_Gas = 'Still gas, secondary energy'
    Gasoline = 'Motor gasoline, secondary energy'
    Kerosene = 'Kerosene and stove oil, secondary energy'
    Diesel = 'Diesel fuel oil, secondary energy'
    Light_Fuel_Oil = 'Light fuel oil, secondary energy'
    Heavy_Fuel_Oil = 'Heavy fuel oil, secondary energy'
    PetCoke = 'Petroleum coke, secondary energy'
    Aviation_Gasoline = 'Aviation gasoline, secondary energy'
    Aviation_Turbo_Fuel = 'Aviation turbo fuel, secondary energy'
    Non_Energy_Products = 'Non-energy products, secondary energy'
    Secondary_Electricity_Thermal = 'Secondary electricity, thermal, secondary energy'
    Renewable_Fuels = 'Renewable fuels, primary energy'


UoM_by_fuel_type = {
    Fuel_Type.Anthracite: ('Kilotonnes', u.kt_anthracite),
    Fuel_Type.Aviation_Gasoline: ('Megalitres', u.megalitres_aviation_gasoline),
    Fuel_Type.Aviation_Turbo_Fuel: ('Megalitres', u.megalitres_aviation_turbo_fuel),
    Fuel_Type.Canadian_Bituminous: ('Kilotonnes', u.kt_coal_bit),
    Fuel_Type.Coke_Oven_Gas: ('Gigalitres', u.gigalitres_ovengas),
    Fuel_Type.Coke: ('Kilotonnes', u.kt_petcoke),
    Fuel_Type.Crude: ('Thousands of cubic metres', u.kilo_m3_crude),
    Fuel_Type.Diesel: ('Megalitres', u.megalitres_diesel),
    Fuel_Type.NGLs: ('Megalitres', u.megalitres_NGLs),
    Fuel_Type.Heavy_Fuel_Oil: ('Megalitres', u.megalitres_HFO),
    Fuel_Type.Imported_Bituminous: ('Kilotonnes', u.kt_coal_bit),
    Fuel_Type.Kerosene: ('Megalitres', u.megalitres_kerosene),
    Fuel_Type.Light_Fuel_Oil: ('Megalitres', u.megalitres_LFO),
    Fuel_Type.Lignite: ('Kilotonnes', u.kt_lignite),
    Fuel_Type.Gasoline: ('Megalitres', u.megalitres_gasoline),
    Fuel_Type.Natural_Gas: ('Gigalitres', u.gigalitres_NG),
    Fuel_Type.Non_Energy_Products: ('Megalitres', u.megalitres),
    Fuel_Type.PetCoke: ('Megalitres', u.megalitres_petcoke),
    Fuel_Type.Primary_Electricity: ('Gigawatt hours', u.gigawatt * u.hours),
    Fuel_Type.LPGs: ('Megalitres', u.megalitres_LPGs),
    Fuel_Type.Renewable_Fuels: ('Megalitres', u.megalitres),
    Fuel_Type.Secondary_Electricity_Thermal: ('Gigawatt hours', u.gigawatt * u.hours),
    Fuel_Type.Steam_Primary: ('Kilotonnes', u.kt_steam),
    Fuel_Type.Still_Gas: ('Megalitres', u.megalitres_stillgas),
    Fuel_Type.Sub_Bituminous: ('Kilotonnes', u.kt_coal_subbit),
    Fuel_Type.Total_Coal: ('Kilotonnes', u.kt_coal),
    Fuel_Type.Total_RPPs: ('Megalitres', u.megalitres_RPPs),
}


class Supply_And_Demand_Characteristics(str, enum.Enum):
    Production = 'Production'
    Exports = 'Exports'
    Imports = 'Imports'
    Inter_Regional_Transfers = 'Inter-regional transfers'
    Stock_Variation = 'Stock variation'
    Other_Adjustments = 'Other adjustments'
    Availability = 'Availability'
    Stock_Change = 'Stock change, utilities and industry'
    Transformed_to_Electricity_by_Utilities = 'Transformed to electricity by utilities'
    Transformed_to_Electricity_by_Industry = 'Transformed to electricity by industry'
    Transformed_to_Coke_and_Manufactured_Gases = 'Transformed to coke and manufactured gases'
    Transformed_to_Steam = 'Transformed to steam generation'
    Net_Supply = 'Net supply'
    Producer_Consumption ='Producer consumption'
    Non_Energy_Use = 'Non-energy use'
    Energy_Use_Final_Demand = 'Energy use, final demand'
    Total_Industrial = 'Total industrial'
    Total_Mining_and_Oil_and_Gas = 'Total mining and oil and gas extraction'
    Total_Manufacturing = 'Total manufacturing'
    Pulp_and_Paper = 'Pulp and paper manufacturing'
    Iron_and_Steel = 'Iron and steel manufacturing'
    Aluminum_and_Non_Ferrous = 'Aluminum and non-ferrous metal manufacturing'
    Cement = 'Cement manufacturing'
    Chemicals = 'Chemicals manufacturing'
    Other_Manufacturing = 'All other manufacturing'
    Residential = 'Residential'
    Commercial_and_Institutional = 'Commercial and other institutional'
    Statistical_Difference = 'Statistical difference'
    Inter_Product_Transfers = 'Inter-product transfers'
    Transformed_to_RPPs = 'Transformed to refined petroleum products'
    RPP_Manufacturing = 'Refined petroleum products manufacturing'
    Forestry_and_Logging = 'Forestry and logging and support activities'
    Construction = 'Construction'
    Total_Transportation = 'Total transportation'
    Pipelines = 'Pipelines'
    Road_Transport = 'Road transport and urban transit'
    Retail_Sales = 'Retail pump sales'
    Agriculture = 'Agriculture, fishing, hunting and trapping'
    Public_Administration = 'Public administration'
    Railways = 'Railways'
    Total_Airlines = 'Total airlines'
    Canadian_Airlines = 'Canadian airlines'
    Foreign_Airlines = 'Foreign airlines'
    Total_Marine = 'Total marine'
    Canadian_Marine = 'Canadian marine'
    Foreign_Marine = 'Foreign marine'


@functools.cache
def supply_and_demand_of_primary_and_secondary_energy():
    df = zip_table_to_dataframe("25-10-0030-01")
    df['REF_DATE'] = df['REF_DATE'].apply(pd.to_numeric)
    
    FT = Fuel_Type
    SDC = Supply_And_Demand_Characteristics
    PT = enums.PT
    rval_pt = objtensor.empty(FT, SDC, PT)
    rval_ca = objtensor.empty(FT, SDC)

    years = list(range(1995, 2024))

    for ft, df_ft in df.groupby('Fuel type'):
        ft = FT(ft)
        UOM, ft_unit = UoM_by_fuel_type[ft]
        rval_pt[ft] = 0 * ft_unit
        rval_ca[ft] = 0 * ft_unit
        for sdc, df_ft_sdc in df_ft.groupby('Supply and demand characteristics'):
            sdc = SDC(sdc)
            for geo, df_geo in df_ft_sdc.groupby('GEO'):
                uom, = set(df_geo.UOM.values)
                assert uom == UOM
                factor_str, = set(df_geo.SCALAR_FACTOR.values)
                factor = Factors[factor_str]
                as_d = dict(zip(df_geo.REF_DATE.values, df_geo.VALUE.values))
                for year in years:
                    as_d.setdefault(year, 0)
                    if np.isnan(as_d[year]):
                        as_d[year] = 0 # the hope is to pick up missing or sensored data via PT.XX
                rep = sts.annual_report(
                    times=[year * u.years for year in sorted(as_d)],
                    values=[vv * factor * ft_unit for _, vv in sorted(as_d.items())])
                if geo == 'Canada':
                    rval_ca[ft, sdc] = rep
                elif geo == 'Atlantic provinces':
                    rval_pt[ft, sdc, PT.XX] += rep
                elif geo == 'Yukon, Northwest Territories and Nunavut':
                    rval_pt[ft, sdc, PT.XX] += rep
                else:
                    rval_pt[ft, sdc, geo] = rep
        set_ptxx(rval_pt[ft], rval_ca[ft])
    return rval_pt, rval_ca
