import numpy as np
import pandas as pd
from .ureg import u
from .sts import annual_report
from .my_functools import maybecache

@maybecache
def national_electricity_CO2e_from_combustion():

    # Define the file path (use forward slashes or a raw string for Windows paths)
    file_path = '/mnt/data/EN_Annex13_Electricity_Intensity.xlsx'

    years = [1990, 2005, 2010, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    # Import the Excel file into a pandas DataFrame
    df_A13_1 = pd.read_excel(file_path, sheet_name='Table A13-1', usecols='B:N',
                             skiprows=4, nrows=4,
                             names=['Fuel type'] + years)
    df_A13_1.index = df_A13_1['Fuel type']
    del df_A13_1['Fuel type']
    df_A13_1_t = df_A13_1.transpose()

    rval = dict(
        natural_gas=annual_report(
            times=np.asarray(years) * u.years,
            values=df_A13_1_t['Natural Gas'].values * u.kt_CO2e),
        coal=annual_report(
            times=np.asarray(years) * u.years,
            values=df_A13_1_t['Coal'].values * u.kt_CO2e),
        )
    return rval
