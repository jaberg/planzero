"""
British Columbia Provincial Inventory
"""

import enum
import functools

import pandas as pd

from .enums import GHG
from .ghgvalues import GWP_100
from .ureg import u
from . import objtensor
from . import sts


def read_spreadsheet():
    df = pd.read_excel(
        "data/bc_provincial_inventory_of_greenhouse_gas_emissions_1990-2023.xlsx",
        sheet_name="Gases",
        usecols='B:AK',
        header=2,
        nrows=520)
    df["Sector"] = df["Unnamed: 2"]
    df["Gas"] = df["Unit: MtCO2e"]
    return df


class Sector(str, enum.Enum):
    Not_Applicable = 'Not Applicable'
    ENERGY = 'ENERGY'
    STATIONARY_COMBUSTION = 'STATIONARY COMBUSTION'
    Public_Electricity_and_Heat_Production_ = 'Public Electricity and Heat Production '
    Petroleum_Refining_Industries = 'Petroleum Refining Industries'
    Oil_and_Gas_Extraction = 'Oil and Gas Extraction'
    Mining = 'Mining'
    Manufacturing_Industries = 'Manufacturing Industries'
    Construction = 'Construction'
    Commercial_and_Institutional = 'Commercial and Institutional'
    Residential = 'Residential'
    Agriculture_and_Forestry = 'Agriculture and Forestry'
    TRANSPORT2 = 'TRANSPORT2'
    Domestic_Aviation = 'Domestic Aviation'
    Road_Transport = 'Road Transport'
    LightDuty_Gasoline_Vehicles = 'Light-Duty Gasoline Vehicles'
    LightDuty_Gasoline_Trucks = 'Light-Duty Gasoline Trucks'
    HeavyDuty_Gasoline_Vehicles = 'Heavy-Duty Gasoline Vehicles'
    Motorcycles = 'Motorcycles'
    LightDuty_Diesel_Vehicles = 'Light-Duty Diesel Vehicles'
    LightDuty_Diesel_Trucks = 'Light-Duty Diesel Trucks'
    HeavyDuty_Diesel_Vehicles = 'Heavy-Duty Diesel Vehicles'
    Propane_and_Natural_Gas_Vehicles = 'Propane and Natural Gas Vehicles'
    Railways = 'Railways'
    Domestic_Marine = 'Domestic Marine '
    OffRoad_Transport = 'Off-Road Transport'
    OffRoad_Agriculture_and_Forestry = 'Off-Road Agriculture and Forestry'
    OffRoad_Commercial_and_Institutional = 'Off-Road Commercial and Institutional'
    OffRoad_Manufacturing_Mining_and_Construction = 'Off-Road Manufacturing, Mining, and Construction'
    OffRoad_Residential = 'Off-Road Residential'
    OffRoad_Other_Transport = 'Off-Road Other Transport'
    Pipeline_Transport = 'Pipeline Transport'
    FUGITIVE_SOURCES = 'FUGITIVE SOURCES'
    Coal_Mining_ = 'Coal Mining '
    Oil_and_Natural_Gas = 'Oil and Natural Gas'
    Oil = 'Oil'
    Natural_Gas = 'Natural Gas'
    Venting = 'Venting'
    Flaring = 'Flaring'
    CO2_Transport_and_Storage = 'CO2 Transport and Storage'
    IPPU_AGRICULTURE_AND_WASTE = 'IPPU, AGRICULTURE, AND WASTE'
    INDUSTRIAL_PROCESSES_AND_PRODUCT_USE = 'INDUSTRIAL PROCESSES AND PRODUCT USE (IPPU)'
    Mineral_Products = 'Mineral Products'
    Cement_Production = 'Cement Production'
    Lime_Production = 'Lime Production'
    Mineral_Products_Use = 'Mineral Products Use'
    Chemical_Industry3 = 'Chemical Industry3'
    Metal_Production = 'Metal Production'
    Iron_and_Steel_Production = 'Iron and Steel Production'
    Aluminum_Production = 'Aluminum Production'
    SF6_Used_in_Magnesium_Smelters_and_Casters = 'SF6 Used in Magnesium Smelters and Casters'
    Production_and_Consumption_of_Halocarbons_SF6_and_NF3 = 'Production and Consumption of Halocarbons, SF6, and NF3'
    NonEnergy_Products_from_Fuels_and_Solvent_Use = 'Non-Energy Products from Fuels and Solvent Use'
    Other_Product_Manufacture_and_Use = 'Other Product Manufacture and Use'
    AGRICULTURE = 'AGRICULTURE'
    Enteric_Fermentation = 'Enteric Fermentation'
    Manure_Management = 'Manure Management'
    Agricultural_Soils = 'Agricultural Soils'
    Direct_Sources = 'Direct Sources'
    Indirect_Sources = 'Indirect Sources'
    Field_Burning_of_Agricultural_Residues = 'Field Burning of Agricultural Residues'
    Liming_Urea_Application_and_Other_CarbonContaining_Fertilizers = 'Liming, Urea Application, and Other Carbon-Containing Fertilizers '
    WASTE = 'WASTE'
    Solid_Waste_Disposal = 'Solid Waste Disposal  '
    Biological_Treatment_of_Solid_Waste = 'Biological Treatment of Solid Waste'
    Wastewater_Treatment_and_Discharge = 'Wastewater Treatment and Discharge  '
    Incineration_and_Open_Burning_of_Waste = 'Incineration and Open Burning of Waste  '
    Industrial_Wood_Waste_Landfills = 'Industrial Wood Waste Landfills'
    LAND_USE_CHANGE = 'LAND-USE CHANGE'
    LAND_USE_CHANGE5 = 'LAND-USE CHANGE5'
    Deforestation = 'Deforestation'
    Afforestation = 'Afforestation'
    Grassland_Converted_to_Cropland = 'Grassland Converted to Cropland'
    Other_Land_Converted_to_Wetlands = 'Other Land Converted to Wetlands'

ghg_by_str = {
    'TOTAL1': None,
    'CARBON DIOXIDE (CO2)': GHG.CO2,
    'METHANE (CH4)': GHG.CH4,
    'NITROUS OXIDE (N2O)b': GHG.N2O,
    'HYDROFLUOROCARBONS (HFCs)c': GHG.HFCs,
    'PERFLUOROCARBONS (PFCs)c': GHG.PFCs,
    'SULPHUR HEXAFLUORIDE (SF6)d': GHG.SF6,
    'NITROGEN TRIFLUORIDE (NF3)e': GHG.NF3,
}


@functools.cache
def bc_provincial_inventory():

    rval = objtensor.empty(GHG, Sector)

    years = list(range(1990, 2024))

    factor_by_ghg = {
        ghg: 1.0 * u.Mt_CO2e / GWP_100[ghg]
        for ghg in GHG}

    df = read_spreadsheet()

    for row in df.iloc:
        # set gas first, b/c it's set when sector would be nan
        gas_str = row.Gas
        if gas_str == gas_str: # not nan
            gas = ghg_by_str[row.Gas]

        if row.Sector == row.Sector:
            sector = Sector(row.Sector)
        else:
            continue
        values = [
            0.0 if row[year] == '-' else (float(row[year]) * factor_by_ghg[gas].magnitude)
            for year in years]
        report = sts.annual_report2(
            years=years,
            values=values,
            v_unit=factor_by_ghg[gas].u)
        rval[gas, sector] = report

    return rval
