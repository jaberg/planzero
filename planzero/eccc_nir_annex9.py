import numpy as np
import pandas as pd
from .ureg import u
from .sts import annual_report
from .my_functools import maybecache
from . import enums

@maybecache
def national_emissions_IPCC(year):
    """Call with year as an integer,
    Get back a dataframe with the body of that year's sheet.
    """
    assert year == int(year)
    assert 1990 <= year <= 2023
    yod = year % 10
    decade = (year // 10) % 100

    file_path = '/mnt/data/EN_Annex9_GHG_IPCC_Canada.xlsx'

    df = pd.read_excel(
        file_path, sheet_name=f'{decade}{yod}', usecols='A:O',
        skiprows=5, nrows=92,
        names=['Category', # A
               'Sub-category', # B
               'Sub-sub-category', # C
               'skip-1', # D
               'skip-2', # E
               'CO2',
               'CH4',
               'CH4_CO2e',
               'N2O',
               'N2O_CO2e',
               'HFCs_CO2e',
               'PFCs_CO2e',
               'SF6_CO2e',
               'NF3_CO2e',
               'Total_CO2e'
              ])
    return df


_unit_by_column_name = dict(
    CO2=u.kt_CO2,
    CH4=u.kt_CH4,
    CH4_CO2e=u.kt_CO2e,
    N2O=u.kt_N2O,
    N2O_CO2e=u.kt_CO2e,
    HFCs_CO2e=u.kt_CO2e,
    PFCs_CO2e=u.kt_CO2e,
    SF6_CO2e=u.kt_CO2e,
    NF3_CO2e=u.kt_CO2e,
    Total_CO2e=u.kt_CO2e,
    )


def emissions_by_IPCC_sector(year, column_name):

    df = national_emissions_IPCC(year)
    rval = {}

    for row in df.iloc:
        try:
            ipcc_sector = enums.IPCC_Sector(str(row['Sub-category']).strip())
        except ValueError as exc:
            try:
                ipcc_sector = enums.IPCC_Sector(str(row['Sub-sub-category']).strip())
            except:
                if row['Category'] in (
                    'TOTALb',
                    'ENERGY',
                    'INDUSTRIAL PROCESSES AND PRODUCT USE',
                    'AGRICULTURE',
                    'WASTE',
                    'LAND USE, LAND-USE CHANGE AND FORESTRY'
                ):
                    continue
                if str(row['Sub-category']).strip() in (
                    'Stationary Combustion Sources',
                    'Manufacturing Industries',
                    'Transportc', 'Aviation', 'Road Transportation', 'Marine', 'Other Transportation',
                    'Fugitive Sources', 'Oil and Natural Gas',
                    'Mineral Products', 'Chemical Industry', 'Metal Production',
                    'Agricultural Soils', 'Landfills', 'Wastewater Treatment and Discharge'
                ):
                    continue
                raise ValueError(row) from exc
        rval[ipcc_sector] = row[column_name] * _unit_by_column_name[column_name]

    assert len(enums.IPCC_Sector) == 71
    assert len(rval) == len(enums.IPCC_Sector), len(rval)

    return rval
