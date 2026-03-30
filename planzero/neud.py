"""National Energy Use Database

https://oee.nrcan.gc.ca/corporate/statistics/neud/dpa/menus/trends/comprehensive_tables/list.cfm
"""
import enum
import functools
import glob
import os

import pandas as pd

from .enums import PT
from .ureg import u
from . import objtensor
from . import sts

class EnergySource(str, enum.Enum):
    Electricity = 'Electricity'
    NaturalGas = 'Natural Gas'
    HeatingOil = 'Heating Oil'
    Other = 'Other'
    Wood = 'Wood'


class EndUse(str, enum.Enum):
    SpaceHeating = 'Space Heating'
    WaterHeating = 'Water Heating'
    Appliances = 'Appliances'
    Lighting = 'Lighting'
    SpaceCooling = 'Space Cooling'

@functools.cache
def df_residential_table2():
    file_pattern = os.path.join('data/neud', "res_*.xls")
    dataframes = []
    for xlspath in glob.glob(file_pattern):
        prefix = 'data/neud/res_'
        pt_letters = xlspath[len(prefix): len(prefix) + 2]
        if pt_letters == 'ca':
            continue
        elif pt_letters == 'tr':
            # The data is for all territories
            pt = PT.XX
        else:
            pt = getattr(PT, pt_letters.upper())
        df = pd.read_excel(xlspath,
                       sheet_name='Table 2',
                       usecols='B:Z',
                       header=10,
                       nrows=44)
        df['Geo'] = [pt.value] * len(df)
        dataframes.append(df)
    rval = pd.concat(dataframes, ignore_index=True)
    return rval


@functools.cache
def ghg_emissions_excl_electricity_by_end_use():
    rval = objtensor.empty(EndUse, PT)
    counter = 0
    for row in df_residential_table2().iloc:
        colB = str(row['Unnamed: 1'])
        if colB.startswith('Total GHG Emissions'):
            counter = 7

        years = list(range(2000, 2023 + 1))

        if 0 < counter < 6:
            pt = PT(row['Geo'])
            end_use = EndUse(colB)
            values = [float(row[year]) for year in years]
            rval[end_use, pt] = sts.annual_report2(
                years=years,
                values=values,
                v_unit=u.Mt_CO2e)

        if counter:
            counter -= 1

    # setting per-territory emissions to zero because
    # their emissions are aggregated in PT.XX
    rval[:, [PT.NU, PT.YT, PT.NT]] = 0 * u.Mt_CO2e

    assert None not in rval.buf
    return rval


@functools.cache
def df_transportation_table8():
    file_pattern = os.path.join('data/neud', "tran_*.xls")
    dataframes = []
    for xlspath in glob.glob(file_pattern):
        prefix = 'data/neud/tran_'
        pt_letters = xlspath.split('_')[1]
        print(prefix, pt_letters)
        if pt_letters == 'bct':
            # The data is for BC and all territories, so technically
            # the emissions should be assigned to XX, even if they
            # are almost all from BC.
            pt = PT.XX
        else:
            pt = getattr(PT, pt_letters.upper())
        df = pd.read_excel(xlspath,
                       sheet_name='Table 8',
                       usecols='B:Z',
                       header=10,
                       nrows=41)
        df['Geo'] = [pt.value] * len(df)
        dataframes.append(df)
    rval = pd.concat(dataframes, ignore_index=True)
    return rval


class TransportationMode(str, enum.Enum):
    Cars = "Cars"
    PassengerLightTrucks = "Passenger Light Trucks"
    FreightLightTrucks = "Freight Light Trucks"
    MediumTrucks = "Medium Trucks"
    HeavyTrucks = "Heavy Trucks"
    Motorcycles = "Motorcycles"
    SchoolBuses = "School Buses"
    UrbanTransit = "Urban Transit"
    InterCityBuses = "Inter-City Buses"
    PassengerAir = "Passenger Air"
    FreightAir = "Freight Air"
    PassengerRail = "Passenger Rail"
    FreightRail = "Freight Rail"
    Marine = "Marine"
    OffRoad = "Off-Road2"


@functools.cache
def ghg_emissions_by_transportation_mode():
    rval = objtensor.empty(TransportationMode, PT)
    counter = 0
    years = list(range(2000, 2023 + 1))
    for row in df_transportation_table8().iloc:
        colB = str(row['Unnamed: 1'])
        if colB == 'GHG Emissions by Transportation Mode (Mt of CO2e)':
            counter = 15
            continue
        if counter:
            pt = PT(row['Geo'])
            trans_mode = TransportationMode(colB)
            values = [float(row[year]) for year in years]
            rval[trans_mode, pt] = sts.annual_report2(
                years=years,
                values=values,
                v_unit=u.Mt_CO2e)

        if counter:
            counter -= 1

    # setting per-territory emissions to zero because
    # their emissions are aggregated in PT.XX
    # and BC too !?
    rval[:, [PT.BC, PT.NU, PT.YT, PT.NT]] = 0 * u.Mt_CO2e

    assert None not in rval.buf
    return rval


def demo():
    ghg_emissions_by_transportation_mode()


if __name__ == '__main__':
    demo()
