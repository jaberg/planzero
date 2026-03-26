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
def read_Table2_as_df():
    file_pattern = os.path.join('data/neud', "*.xls")
    dataframes = []
    for xlspath in glob.glob(file_pattern):
        prefix = 'data/neud/res_'
        pt_letters = xlspath[len(prefix): len(prefix) + 2]
        if pt_letters == 'ca':
            continue
        elif pt_letters == 'tr':
            pt = PT.XX # The data is for all territories
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
    for row in read_Table2_as_df().iloc:
        colB = str(row['Unnamed: 1'])
        #print(row['Geo'], counter, colB, row)
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


def demo():
    ghg_emissions_excl_electricity_by_end_use()

if __name__ == '__main__':
    demo()
