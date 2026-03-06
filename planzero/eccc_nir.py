import enum

from .ureg import u
from . import objtensor
from . import sts


class Table3_11_Rows(str, enum.Enum):
    Total = 'Abandoned Oil And Gas Wells'
    AbandonedOilWells = 'Abandoned Oil Wells'
    AbandonedGasWells = 'Abandoned Gas Wells'


def table3_11():
    """GHG Emissions from Abandoned Oil and Gas Wells
    """
    rval = objtensor.empty(Table3_11_Rows)
    rval[Table3_11_Rows.Total] = sts.annual_report2(
        years=[1990, 2005, 2018, 2019, 2020, 2021, 2022, 2023],
        values=[170,  320,  590,  600,  610,  600,  560,  570],
        v_unit=u.kt_CO2e)
    rval[Table3_11_Rows.AbandonedOilWells] = sts.annual_report2(
        years=[1990, 2005, 2018, 2019, 2020, 2021, 2022, 2023],
        values=[90,  140,  290,  300,  310,  300,  280,  280],
        v_unit=u.kt_CO2e)
    rval[Table3_11_Rows.AbandonedGasWells] = sts.annual_report2(
        years=[1990, 2005, 2018, 2019, 2020, 2021, 2022, 2023],
        values=[80,  180,  290,  300,  300,  300,  280,  290],
        v_unit=u.kt_CO2e)
    return rval


class Table3_12_Rows(str, enum.Enum):
    Total = 'Post-Meter Fugitives'
    Appliances = 'Natural gas appliances in residential and commercial sectors'
    Vehicles = 'Natural gas fueled vehicles'
    Industrial = 'Power plants and industrial facilities consuming natural gas'

def table3_12():
    """GHG Emissions from Post-Meter Fugitives"""
    rval = objtensor.empty(Table3_12_Rows)
    rval[Table3_12_Rows.Total] = sts.annual_report2(
        years=[ 1990, 2005, 2018, 2019, 2020, 2021, 2022, 2023],
        values=[1200, 1600, 1900, 2000, 2000, 2000, 2100, 2100],
        v_unit=u.kt_CO2e)
    rval[Table3_12_Rows.Appliances] = sts.annual_report2(
        years=[ 1990, 2005, 2018, 2019, 2020, 2021, 2022, 2023],
        values=[1000, 1400, 1600, 1600, 1700, 1700, 1700, 1700],
        v_unit=u.kt_CO2e)
    rval[Table3_12_Rows.Vehicles] = sts.annual_report2(
        years=[1990, 2005, 2018, 2019, 2020, 2021, 2022, 2023],
        values=[.04,  .01,  .02,  .02,  .02,  .03,  .03,  .03],
        v_unit=u.kt_CO2e)
    rval[Table3_12_Rows.Industrial] = sts.annual_report2(
        years=[1990, 2005, 2018, 2019, 2020, 2021, 2022, 2023],
        values=[250,  260,  320,  330,  320,  350,  370,  400],
        v_unit=u.kt_CO2e)
    return rval
